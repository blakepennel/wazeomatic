package com.wazeomatic.detector

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.tan

/**
 * Road-surface classifier on the road ahead; class names come back prefixed "surface_". Twin of
 * classifiers.RoadSurface.
 *
 * A **top-view** model ([topview] = hfov degrees, mount height m, horizon fraction; trained on RoadSaW / RoadSC,
 * classes dry, damp, wet, snow, slush) gets the 3.6 m square of road 15 m ahead warped to a bird's-eye patch with
 * the flat-road camera model, as those datasets were made. A frame where a vehicle or a person covers that square,
 * or where the square reaches into the hood zone, is skipped and the next frame is tried. Other models get a fixed
 * patch just above the hood zone, moved up when the learnt hood zone is taller than the default.
 */
class RoadSurface(
    val clf: ImageClassifier,
    val everyS: Double = 0.5,
    val patch: DoubleArray = ROAD_PATCH,
    var topview: DoubleArray? = null,
    val distM: Double = 15.0,
    val sizeM: Double = 3.6,
    val maxOcclusion: Double = 0.10,
) : ScenePart {
    /** Learnt hood zone (fraction of the height); set through [setHoodFrac]. */
    @Volatile var hood = 0.0
        private set
    private var lastT = -1e9
    var last: Map<String, Float> = emptyMap()
        private set
    override var lastMs = 0.0
        private set
    /** Frames skipped because the top-view square was covered or in the hood zone. */
    var skipped = 0
        private set

    /** Whether the model wants the bird's-eye patch (RoadSaW / RoadSC classes). */
    val isTopView: Boolean get() = "damp" in clf.labels

    override fun setHoodFrac(frac: Double) {
        hood = frac
    }

    /** x1, y1, x2, y2 in pixels. */
    fun region(h: Int, w: Int): IntArray {
        val (x1, y1, x2, y2) = patch.toList()
        val bottom = min(y2, 1.0 - max(hood, 0.10) - 0.02)
        val top = max(0.45, bottom - (y2 - y1))
        return intArrayOf((x1 * w).toInt(), (top * h).toInt(), (x2 * w).toInt(), (bottom * h).toInt())
    }

    /** Whether a vehicle or person box covers more than [maxOcclusion] of the quad's bounding box. */
    fun occluded(quad: Array<DoubleArray>, w: Int, h: Int, dets: List<Detection>): Boolean {
        val x1 = quad.minOf { it[0] } / w
        val y1 = quad.minOf { it[1] } / h
        val x2 = quad.maxOf { it[0] } / w
        val y2 = quad.maxOf { it[1] } / h
        val area = max((x2 - x1) * (y2 - y1), 1e-9)
        for (d in dets) {
            if (d.label !in OCCLUDERS) continue
            val b = d.bbox
            val ix = max(0.0, min(x2, b.cx + b.w / 2.0) - max(x1, b.cx - b.w / 2.0))
            val iy = max(0.0, min(y2, b.cy + b.h / 2.0) - max(y1, b.cy - b.h / 2.0))
            if (ix * iy / area > maxOcclusion) return true
        }
        return false
    }

    override fun invoke(image: RgbImage, t: Double, dets: List<Detection>): Map<String, Float>? {
        if (t - lastT < everyS) {
            lastMs = 0.0
            return null
        }
        val s = clf.size
        val tv = topview
        val input = if (tv != null) {
            val quad = topviewQuad(image.width, image.height, tv[0], tv[1], tv[2], distM, sizeM)
            if (quad.maxOf { it[1] } > (1.0 - max(hood, 0.10)) * image.height || occluded(quad, image.width, image.height, dets)) {
                lastMs = 0.0
                skipped++
                return null
            }
            warpToSquare(image, quad, s)
        } else {
            val r = region(image.height, image.width)
            ImageOps.resize(image.crop(r[0], r[1], r[2], r[3]), s, s, area = true)
        }
        lastT = t
        val p = clf.predict(listOf(input))[0]
        lastMs = clf.lastMs
        last = clf.labels.withIndex().filter { it.index < p.size }.associate { "surface_${it.value}" to p[it.index] }
        return last
    }

    override fun close() = clf.close()

    companion object {
        /** x1, y1, x2, y2 as fractions of the frame: centre of the lane, just above the hood zone. */
        val ROAD_PATCH = doubleArrayOf(0.40, 0.60, 0.60, 0.86)

        /** Boxes that hide the road: a top-view square they overlap is not classified. */
        val OCCLUDERS = setOf("car", "truck", "bus", "motorcycle", "bicycle", "person")

        /**
         * Image corners in pixels (far left, far right, near right, near left) of a [sizeM] square of flat road
         * centred [distM] ahead: the pinhole model of the blocked-lane rule. Twin of classifiers.topview_quad.
         */
        fun topviewQuad(w: Int, h: Int, hfovDeg: Double, heightM: Double, horizonY: Double, distM: Double,
                        sizeM: Double): Array<DoubleArray> {
            val fx = 0.5 / tan(Math.toRadians(hfovDeg) / 2)
            val fy = fx * w / h
            val near = distM - sizeM / 2
            val far = distM + sizeM / 2
            val half = sizeM / 2
            fun px(x: Double, z: Double) = doubleArrayOf((0.5 + fx * x / z) * w, (horizonY + fy * heightM / z) * h)
            return arrayOf(px(-half, far), px(half, far), px(half, near), px(-half, near))
        }

        /**
         * 3x3 homography (row-major) taking (0,0), (1,0), (1,1), (0,1) scaled by [s] to the four [quad] corners:
         * the inverse map cv2.warpPerspective applies for cv2.getPerspectiveTransform(quad, square).
         */
        fun squareToQuad(quad: Array<DoubleArray>, s: Int): DoubleArray {
            val src = arrayOf(doubleArrayOf(0.0, 0.0), doubleArrayOf(s.toDouble(), 0.0),
                doubleArrayOf(s.toDouble(), s.toDouble()), doubleArrayOf(0.0, s.toDouble()))
            // 8 unknowns h0..h7 (h8 = 1): x' = (h0 u + h1 v + h2) / (h6 u + h7 v + 1), same for y' with h3..h5
            val a = Array(8) { DoubleArray(9) }
            for (i in 0 until 4) {
                val (u, v) = src[i].toList()
                val (x, y) = quad[i].toList()
                a[2 * i] = doubleArrayOf(u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x, x)
                a[2 * i + 1] = doubleArrayOf(0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y, y)
            }
            for (c in 0 until 8) {                       // Gaussian elimination with partial pivoting
                var p = c
                for (r in c + 1 until 8) if (abs(a[r][c]) > abs(a[p][c])) p = r
                val tmp = a[c]; a[c] = a[p]; a[p] = tmp
                for (r in 0 until 8) {
                    if (r == c) continue
                    val f = a[r][c] / a[c][c]
                    if (f != 0.0) for (k in c until 9) a[r][k] -= f * a[c][k]
                }
            }
            return DoubleArray(9) { if (it == 8) 1.0 else a[it][8] / a[it][it] }
        }

        /** The quad warped to an [s] x [s] square, bilinear, black outside the frame (cv2.warpPerspective). */
        fun warpToSquare(img: RgbImage, quad: Array<DoubleArray>, s: Int): RgbImage {
            val m = squareToQuad(quad, s)
            val out = IntArray(s * s)
            val w = img.width
            val h = img.height
            fun px(x: Int, y: Int): Int = if (x < 0 || y < 0 || x >= w || y >= h) 0 else img.argb[y * w + x]
            for (v in 0 until s) for (u in 0 until s) {
                val d = m[6] * u + m[7] * v + m[8]
                val x = (m[0] * u + m[1] * v + m[2]) / d
                val y = (m[3] * u + m[4] * v + m[5]) / d
                val x0 = floor(x).toInt()
                val y0 = floor(y).toInt()
                val fx = x - x0
                val fy = y - y0
                val p00 = px(x0, y0); val p10 = px(x0 + 1, y0); val p01 = px(x0, y0 + 1); val p11 = px(x0 + 1, y0 + 1)
                fun ch(sh: Int): Int {
                    val c = ((p00 shr sh) and 0xff) * (1 - fx) * (1 - fy) + ((p10 shr sh) and 0xff) * fx * (1 - fy) +
                        ((p01 shr sh) and 0xff) * (1 - fx) * fy + ((p11 shr sh) and 0xff) * fx * fy
                    return Math.rint(c).toInt().coerceIn(0, 255)
                }
                out[v * s + u] = RgbImage.pack(ch(16), ch(8), ch(0))
            }
            return RgbImage(s, s, out)
        }
    }
}
