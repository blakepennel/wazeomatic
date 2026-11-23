package com.wazeomatic.detector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.Log
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * One YOLO model (Ultralytics ONNX export). Tries the XNNPACK execution provider first and
 * falls back to the plain CPU provider if the session can't be created with it.
 *
 * Each pass ([SLOT_FULL], [SLOT_ZOOM]) has its own session and buffers, so the whole frame and the zoom crop
 * can run on two threads at once. A model exported with an open input shape (`export_onnx.py --dynamic`) gets
 * the shape of what it is fed when [rect] is on: a 16:9 frame runs at 640 x 384 and the 640 x 288 zoom crop at
 * 640 x 288 instead of both at 640 x 640, the same pixels without the grey padding (as `--rect` on the desktop).
 * A static export always runs at its square size.
 */
class OnnxDetector(
    private val env: OrtEnvironment,
    private val modelBytes: ByteArray,
    labelsFromFile: List<String>?,
    val name: String,
    var confThres: Float,
    var iouThres: Float,
    private val rect: Boolean = true,
    private val threads: Int = 0,
    /** Boxes per pass and of one class per pass, best first (0 = no cap); see [YoloDecoder.MAX_DET]. */
    val maxDet: Int = YoloDecoder.MAX_DET,
    val maxPerClass: Int = YoloDecoder.MAX_PER_CLASS,
    /** xnnpack | nnapi | nnapi_fp16 | cpu; one that fails to load falls back to XNNPACK, then CPU. */
    provider: String = "xnnpack",
) : Closeable {
    /** The provider the sessions run on (after any fallback). */
    val backend: String get() = provider.uppercase()
    val inputName: String
    /** Long side of the input (the square size of a static export, the training size of an open one). */
    val size: Int
    /** True when the export leaves the input height and width open. */
    val openShape: Boolean
    val labels: List<String>
    /** Model time of the last whole-frame pass (ms). */
    val lastMs: Double get() = slots[SLOT_FULL]?.lastMs ?: 0.0
    /** Model time of the last zoom pass (ms). */
    val lastZoomMs: Double get() = slots[SLOT_ZOOM]?.lastMs ?: 0.0
    /** Letterbox + float conversion, and output copy + decode, of the last whole-frame pass (ms). */
    val lastPreMs: Double get() = slots[SLOT_FULL]?.preMs ?: 0.0
    val lastDecodeMs: Double get() = slots[SLOT_FULL]?.decodeMs ?: 0.0
    /** "640x384" etc. of the whole-frame pass once it has run, for the status line. */
    val shapeNote: String get() = slots[SLOT_FULL]?.let { "${it.inW}x${it.inH}" } ?: if (openShape) "$size open" else "${size}x$size"

    private val slots = arrayOfNulls<Slot>(2)
    // the provider sessions are made with; steps down nnapi -> xnnpack -> cpu when one fails to load
    private var provider = provider.lowercase().takeIf { it in PROVIDERS || it.startsWith("nnapi") } ?: "xnnpack"

    init {
        // a first session for the model's description; a static model keeps it for the whole-frame pass
        val probe = newSession(null)
        val inp = probe.inputInfo.entries.first()
        inputName = inp.key
        val shape = (inp.value.info as TensorInfo).shape // [1, 3, H, W], -1 for open dims
        openShape = shape.size == 4 && (shape[2] <= 0 || shape[3] <= 0)
        val meta = try { probe.metadata.customMetadata } catch (e: Exception) { emptyMap<String, String>() }
        size = when {
            !openShape && shape.size == 4 -> maxOf(shape[2], shape[3]).toInt()
            else -> Regex("\\d+").find(meta["imgsz"] ?: "")?.value?.toInt() ?: 640
        }
        val outShape = (probe.outputInfo.values.first().info as TensorInfo).shape
        val metaNames = meta["names"]?.let { try { YoloDecoder.parseNamesMetadata(it) } catch (e: Exception) { null } }
        labels = when {
            !labelsFromFile.isNullOrEmpty() -> labelsFromFile
            !metaNames.isNullOrEmpty() -> metaNames
            else -> throw IllegalArgumentException("$name: no labels file and no names in model metadata")
        }
        if (outShape.size == 3) {
            // [1, 4 + classes, anchors]; anchors is open (-1) in an open-shape export
            val ch = if (outShape[2] <= 0) outShape[1] else minOf(outShape[1], outShape[2])
            if (ch > 0 && ch - 4 != labels.size.toLong()) {
                throw IllegalArgumentException("$name: model has ${ch - 4} classes but ${labels.size} labels given")
            }
        }
        if (openShape) probe.close() else slots[SLOT_FULL] = Slot(probe, shape[3].toInt(), shape[2].toInt())
        Log.i(TAG, "$name: input ${if (openShape) "open, trained at $size" else "${size}x$size"}, ${labels.size} classes, " +
            "output ${outShape.contentToString()}, backend $backend, ${threadCount()} threads")
    }

    private fun threadCount() = if (threads > 0) threads else autoThreads()

    /** [inW] x [inH] fixes an open input shape; null keeps the model's own. */
    // synchronized: the whole-frame and zoom slots are made on two threads, and both may step the provider down
    @Synchronized
    private fun newSession(fixed: IntArray?): OrtSession {
        while (true) {
            try {
                return env.createSession(modelBytes, options(provider, fixed))
            } catch (e: Exception) {
                val next = if (provider.startsWith("nnapi")) "xnnpack" else if (provider == "xnnpack") "cpu" else throw e
                Log.w(TAG, "$name: $provider session failed, using $next: ${e.message}")
                provider = next
            }
        }
    }

    private fun options(provider: String, fixed: IntArray?): OrtSession.SessionOptions {
        val o = OrtSession.SessionOptions()
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        if (fixed != null) {
            // Ultralytics names the open dims batch / height / width; fixing them lets XNNPACK plan a static graph
            o.setSymbolicDimensionValue("batch", 1)
            o.setSymbolicDimensionValue("height", fixed[1].toLong())
            o.setSymbolicDimensionValue("width", fixed[0].toLong())
        }
        if (provider.startsWith("nnapi")) {
            // NNAPI needs fixed shapes (the symbolic dims above); what its drivers can't run stays on XNNPACK.
            // CPU_DISABLED keeps NNAPI's slow reference CPU device out of it unless "cpu" is in the name
            // (nnapi, nnapi_fp16, nnapi_fp16_cpu, nnapi_nchw ...)
            val flags = java.util.EnumSet.noneOf(ai.onnxruntime.providers.NNAPIFlags::class.java)
            if ("cpu" !in provider) flags.add(ai.onnxruntime.providers.NNAPIFlags.CPU_DISABLED)
            if ("fp16" in provider) flags.add(ai.onnxruntime.providers.NNAPIFlags.USE_FP16)
            if ("nchw" in provider) flags.add(ai.onnxruntime.providers.NNAPIFlags.USE_NCHW)
            o.addNnapi(flags)
        }
        if (provider != "cpu") {
            o.addXnnpack(mapOf("intra_op_num_threads" to threadCount().toString()))
            // XNNPACK has its own thread pool, ORT docs recommend 1 intra-op thread with it
            o.setIntraOpNumThreads(1)
            o.addConfigEntry("session.intra_op.allow_spinning", "0")
        } else if (threads > 0) {
            o.setIntraOpNumThreads(threads)
        }
        return o
    }

    /** A session at one input shape with its own buffers; one thread at a time. */
    private class Slot(val session: OrtSession, val inW: Int, val inH: Int) {
        val canvasBmp: Bitmap = Bitmap.createBitmap(inW, inH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(canvasBmp)
        val pixels = IntArray(inW * inH)
        // the canvas bitmap's raw memory (R, G, B, A bytes); read as little-endian ints: A B G R from the top byte
        val rgba: ByteBuffer = ByteBuffer.allocateDirect(4 * inW * inH).order(ByteOrder.LITTLE_ENDIAN)
        val rgbaInts: java.nio.IntBuffer = rgba.asIntBuffer()
        var checked = false
        var copyOk = true
        val floats = FloatArray(3 * inW * inH)
        val input: FloatBuffer = ByteBuffer.allocateDirect(4 * 3 * inW * inH).order(ByteOrder.nativeOrder()).asFloatBuffer()
        var lastMs = 0.0
        var preMs = 0.0
        var decodeMs = 0.0
    }

    /** The slot for an [sw] x [sh] image, (re)made when the shape it needs changed (rotation, another crop). */
    private fun slotFor(slot: Int, sw: Int, sh: Int): Slot {
        val want = if (openShape && rect) Letterbox.rectShape(sw, sh, size) else intArrayOf(size, size)
        val cur = slots[slot]
        if (cur != null && cur.inW == want[0] && cur.inH == want[1]) return cur
        cur?.session?.close()
        val s = Slot(newSession(if (openShape) want else null), want[0], want[1])
        slots[slot] = s
        Log.i(TAG, "$name: ${if (slot == SLOT_ZOOM) "zoom" else "full"} pass at ${want[0]}x${want[1]}")
        return s
    }

    /**
     * Letterbox (114 gray, centered) and write RGB float [0,1] NCHW into the slot's input. With [region]
     * only that part of [src] is letterboxed (drawn straight from the source bitmap, no copy).
     */
    private fun preprocess(s: Slot, src: Bitmap, region: Rect?): Letterbox {
        val sw = region?.width() ?: src.width
        val sh = region?.height() ?: src.height
        val lb = Letterbox.compute(sw, sh, s.inW, s.inH)
        s.canvas.drawColor(Color.rgb(YoloDecoder.PAD_VALUE, YoloDecoder.PAD_VALUE, YoloDecoder.PAD_VALUE))
        s.canvas.drawBitmap(src, region, RectF(lb.padX, lb.padY, lb.padX + lb.newW, lb.padY + lb.newH), paint)
        // a plain copy of the bitmap's memory: getPixels converts each pixel (un-premultiplies, packs ARGB) and took
        // 5 ms of a 640x384 pass on a mid-range phone. The canvas is opaque, so premultiplied is the same.
        val plane = s.inW * s.inH
        val f = s.floats
        val px = s.pixels
        if (s.copyOk) {
            s.rgba.rewind()
            s.canvasBmp.copyPixelsToBuffer(s.rgba)
            s.rgbaInts.rewind()
            s.rgbaInts.get(px)
        }
        if (!s.checked) {
            // once per slot: the copy must give what getPixels gives (R, G, B, A in memory), else use getPixels
            s.checked = true
            val ref = IntArray(plane)
            s.canvasBmp.getPixels(ref, 0, s.inW, 0, 0, s.inW, s.inH)
            var bad = 0
            for (i in 0 until plane) {
                val p = px[i]
                val q = (0xff shl 24) or ((p and 0xff) shl 16) or (p and 0xff00) or ((p shr 16) and 0xff)
                if (q != ref[i]) bad++
            }
            if (bad > 0) {
                Log.w(TAG, "$name: raw pixel copy differs from getPixels in $bad of $plane pixels, using getPixels")
                s.copyOk = false
            }
        }
        if (!s.copyOk) {
            // getPixels gives packed ARGB: put it in the copy's A B G R order for the loop below
            s.canvasBmp.getPixels(px, 0, s.inW, 0, 0, s.inW, s.inH)
            for (i in 0 until plane) {
                val p = px[i]
                px[i] = (p and 0xff00ff00.toInt()) or ((p and 0xff) shl 16) or ((p shr 16) and 0xff)
            }
        }
        for (i in 0 until plane) {
            val p = px[i]
            f[i] = (p and 0xff) * INV255
            f[plane + i] = ((p shr 8) and 0xff) * INV255
            f[2 * plane + i] = ((p shr 16) and 0xff) * INV255
        }
        s.input.rewind()
        s.input.put(f)
        s.input.rewind()
        return lb
    }

    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    /**
     * Run the model on [frame], or only on [region] of it when given. Returned boxes are
     * normalized to whatever was fed to the model (the whole frame or the region). Different [slot]s may run
     * on different threads at the same time; one slot must not.
     */
    fun detect(frame: Bitmap, region: Rect? = null, slot: Int = if (region == null) SLOT_FULL else SLOT_ZOOM): List<Detection> {
        val s = slotFor(slot, region?.width() ?: frame.width, region?.height() ?: frame.height)
        val tp = System.nanoTime()
        val lb = preprocess(s, frame, region)
        val t0 = System.nanoTime()
        s.preMs = (t0 - tp) / 1e6
        val raw: FloatArray
        val d1: Int
        val d2: Int
        OnnxTensor.createTensor(env, s.input, longArrayOf(1, 3, s.inH.toLong(), s.inW.toLong())).use { tensor ->
            s.session.run(mapOf(inputName to tensor)).use { res ->
                val out = res[0] as OnnxTensor
                val shape = out.info.shape
                d1 = shape[shape.size - 2].toInt()
                d2 = shape[shape.size - 1].toInt()
                val fb = out.floatBuffer
                raw = FloatArray(fb.remaining())
                fb.get(raw)
            }
        }
        val t1 = System.nanoTime()
        s.lastMs = (t1 - t0) / 1e6
        return YoloDecoder.decode(raw, d1, d2, lb, labels, confThres, iouThres, name, maxDet = maxDet, maxPerClass = maxPerClass)
            .also { s.decodeMs = (System.nanoTime() - t1) / 1e6 }
    }

    override fun close() {
        for (s in slots) s?.session?.close()
    }

    companion object {
        private const val TAG = "Wazeomatic"
        const val SLOT_FULL = 0
        const val SLOT_ZOOM = 1
        private const val INV255 = 1f / 255f
        val PROVIDERS = setOf("xnnpack", "nnapi", "nnapi_fp16", "cpu")

        /** XNNPACK threads per session when [passes] sessions run at the same time: the cores split between them. */
        fun autoThreads(passes: Int = 1, cores: Int = Runtime.getRuntime().availableProcessors()): Int =
            (cores / passes.coerceAtLeast(1)).coerceIn(1, 4)
    }
}
