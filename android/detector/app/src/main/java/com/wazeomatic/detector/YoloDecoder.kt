package com.wazeomatic.detector

import kotlin.math.max
import kotlin.math.min

// Port of the pre/post-processing in wazeomatic/detector.py. Pure Kotlin so it can be
// unit tested on the JVM.

data class Letterbox(
    val scale: Float,
    val padX: Float,
    val padY: Float,
    val inW: Int,
    val inH: Int,
    val srcW: Int,
    val srcH: Int,
    val newW: Int,
    val newH: Int,
) {
    companion object {
        /** Keep aspect ratio, center, pad the rest (pad colour 114 is applied by the caller). */
        fun compute(srcW: Int, srcH: Int, inW: Int, inH: Int): Letterbox {
            val scale = min(inH.toDouble() / srcH, inW.toDouble() / srcW)
            // python round() is round-half-even, Math.rint does the same
            val nw = Math.rint(srcW * scale).toInt()
            val nh = Math.rint(srcH * scale).toInt()
            val px = (inW - nw) / 2
            val py = (inH - nh) / 2
            return Letterbox(scale.toFloat(), px.toFloat(), py.toFloat(), inW, inH, srcW, srcH, nw, nh)
        }

        /**
         * Smallest input [width, height], in multiples of [stride], that holds a [srcW] x [srcH] image scaled so
         * its long side is [size] (detector.rect_shape): 640 x 384 for a 16:9 frame, 640 x 288 for the zoom crop.
         */
        fun rectShape(srcW: Int, srcH: Int, size: Int, stride: Int = 32): IntArray {
            val scale = size.toDouble() / maxOf(srcW, srcH)
            val nw = Math.rint(srcW * scale).toInt()
            val nh = Math.rint(srcH * scale).toInt()
            return intArrayOf((nw + stride - 1) / stride * stride, (nh + stride - 1) / stride * stride)
        }
    }
}

object YoloDecoder {
    const val PAD_VALUE = 114

    /** Greedy NMS. boxes is flat [n*4] xyxy. Returns kept indices, best score first. */
    fun nms(boxes: FloatArray, scores: FloatArray, iouThres: Float): List<Int> {
        val n = scores.size
        if (n == 0) return emptyList()
        val areas = FloatArray(n) { i ->
            max(boxes[i * 4 + 2] - boxes[i * 4], 0f) * max(boxes[i * 4 + 3] - boxes[i * 4 + 1], 0f)
        }
        var order = (0 until n).sortedByDescending { scores[it] }
        val keep = ArrayList<Int>()
        while (order.isNotEmpty()) {
            val i = order[0]
            keep.add(i)
            if (order.size == 1) break
            val next = ArrayList<Int>(order.size - 1)
            for (k in 1 until order.size) {
                val j = order[k]
                val xx1 = max(boxes[i * 4], boxes[j * 4])
                val yy1 = max(boxes[i * 4 + 1], boxes[j * 4 + 1])
                val xx2 = min(boxes[i * 4 + 2], boxes[j * 4 + 2])
                val yy2 = min(boxes[i * 4 + 3], boxes[j * 4 + 3])
                val inter = max(xx2 - xx1, 0f) * max(yy2 - yy1, 0f)
                val iou = inter / (areas[i] + areas[j] - inter + 1e-9f)
                if (iou <= iouThres) next.add(j)
            }
            order = next
        }
        return keep
    }

    /**
     * Decode a [1, d1, d2] YOLO output: either [1, 4+nc, N] (Ultralytics default) or
     * [1, N, 4+nc]. Boxes are xywh in letterboxed pixels. Returned bboxes are
     * normalized center-xywh in the source frame.
     */
    fun decode(
        raw: FloatArray,
        d1: Int,
        d2: Int,
        lb: Letterbox,
        labels: List<String>,
        confThres: Float,
        iouThres: Float,
        modelName: String = "hazard",
        classAgnostic: Boolean = false,
        maxDet: Int = MAX_DET,
        maxPerClass: Int = MAX_PER_CLASS,
    ): List<Detection> {
        require(raw.size >= d1 * d2) { "tensor too small: ${raw.size} < $d1 x $d2" }
        val channelsFirst = d1 < d2 && d1 == 4 + labels.size
        val n = if (channelsFirst) d2 else d1
        val ch = if (channelsFirst) d1 else d2
        val nc = ch - 4
        fun v(i: Int, c: Int): Float = if (channelsFirst) raw[c * d2 + i] else raw[i * d2 + c]

        val selBoxes = ArrayList<Float>()
        val selConf = ArrayList<Float>()
        val selCls = ArrayList<Int>()
        for (i in 0 until n) {
            var best = 0
            var bestScore = v(i, 4)
            for (c in 1 until nc) {
                val s = v(i, 4 + c)
                if (s > bestScore) {
                    bestScore = s
                    best = c
                }
            }
            if (bestScore < confThres) continue
            val cx = v(i, 0)
            val cy = v(i, 1)
            val w = v(i, 2)
            val h = v(i, 3)
            selBoxes.add(cx - w / 2); selBoxes.add(cy - h / 2)
            selBoxes.add(cx + w / 2); selBoxes.add(cy + h / 2)
            selConf.add(bestScore)
            selCls.add(best)
        }
        if (selConf.isEmpty()) return emptyList()
        if (selConf.size > MAX_CANDIDATES) {
            // best first, ties in index order (as numpy's stable argsort in detector.py)
            val top = selConf.indices.sortedByDescending { selConf[it] }.take(MAX_CANDIDATES)
            val b = top.flatMap { i -> (0 until 4).map { selBoxes[i * 4 + it] } }
            val c = top.map { selConf[it] }
            val k = top.map { selCls[it] }
            selBoxes.clear(); selBoxes.addAll(b)
            selConf.clear(); selConf.addAll(c)
            selCls.clear(); selCls.addAll(k)
        }
        val m = selConf.size
        val xyxy = selBoxes.toFloatArray()
        // per-class NMS via coordinate offset trick
        val shifted = FloatArray(m * 4) { k ->
            val off = if (classAgnostic) 0f else selCls[k / 4] * 4096f
            xyxy[k] + off
        }
        val keep = nms(shifted, selConf.toFloatArray(), iouThres)
        val out = ArrayList<Detection>()
        val sw = lb.srcW.toFloat()
        val sh = lb.srcH.toFloat()
        for (i in keep) {
            val bx1 = ((xyxy[i * 4] - lb.padX) / lb.scale).coerceIn(0f, sw)
            val by1 = ((xyxy[i * 4 + 1] - lb.padY) / lb.scale).coerceIn(0f, sh)
            val bx2 = ((xyxy[i * 4 + 2] - lb.padX) / lb.scale).coerceIn(0f, sw)
            val by2 = ((xyxy[i * 4 + 3] - lb.padY) / lb.scale).coerceIn(0f, sh)
            val w = bx2 - bx1
            val h = by2 - by1
            if (w <= 1f || h <= 1f) continue
            val cls = selCls[i]
            out.add(
                Detection(
                    label = labels.getOrElse(cls) { "class$cls" },
                    confidence = selConf[i],
                    bbox = BBox((bx1 + w / 2) / sw, (by1 + h / 2) / sh, w / sw, h / sh),
                    model = modelName,
                )
            )
        }
        return if (maxDet > 0 || maxPerClass > 0) cap(out, maxDet, maxPerClass) else out
    }

    // Per pass, best first, as detector.py. A normal frame has far fewer (on 1,600 road, work-zone and city frames
    // at most 22 of one class and 38 in one pass); they bound a frame gone wrong, whose hundreds of boxes would
    // otherwise stall NMS and the whole second stage.
    const val MAX_CANDIDATES = 1000   // boxes over the threshold that go into NMS
    const val MAX_DET = 40            // boxes out of one pass
    const val MAX_PER_CLASS = 12      // of one class in one pass

    /** The best [maxPerClass] of each label and [maxDet] in all (0 = no cap), best first. */
    fun cap(dets: List<Detection>, maxDet: Int = MAX_DET, maxPerClass: Int = MAX_PER_CLASS): List<Detection> {
        val out = ArrayList<Detection>()
        val seen = HashMap<String, Int>()
        for (d in dets.sortedByDescending { it.confidence }) {
            if (maxDet > 0 && out.size >= maxDet) break
            val n = seen[d.label] ?: 0
            if (maxPerClass > 0 && n >= maxPerClass) continue
            seen[d.label] = n + 1
            out.add(d)
        }
        return out
    }

    /** Ultralytics stores class names in metadata as "{0: 'person', 1: 'bicycle', ...}". */
    fun parseNamesMetadata(s: String): List<String> {
        val re = Regex("""(\d+)\s*:\s*(?:'([^']*)'|"([^"]*)")""")
        val map = sortedMapOf<Int, String>()
        for (mt in re.findAll(s)) {
            val name = mt.groups[2]?.value ?: mt.groups[3]?.value ?: continue
            map[mt.groupValues[1].toInt()] = name
        }
        if (map.isEmpty()) return emptyList()
        val n = map.lastKey() + 1
        return List(n) { map[it] ?: "class$it" }
    }
}
