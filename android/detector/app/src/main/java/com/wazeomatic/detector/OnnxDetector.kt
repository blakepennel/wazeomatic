package com.wazeomatic.detector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * One YOLO model (Ultralytics ONNX export). Tries the XNNPACK execution provider first and
 * falls back to the plain CPU provider if the session can't be created with it.
 */
class OnnxDetector(
    private val env: OrtEnvironment,
    modelBytes: ByteArray,
    labelsFromFile: List<String>?,
    val name: String,
    var confThres: Float,
    var iouThres: Float,
) : Closeable {
    val session: OrtSession
    val backend: String
    val inputName: String
    val inW: Int
    val inH: Int
    val labels: List<String>
    var lastMs = 0.0
        private set

    private val canvasBmp: Bitmap
    private val canvas: Canvas
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val pixels: IntArray
    private val input: FloatBuffer

    init {
        var be = "XNNPACK"
        val s = try {
            env.createSession(modelBytes, options(xnnpack = true))
        } catch (e: Exception) {
            Log.w(TAG, "$name: XNNPACK session failed, using CPU provider", e)
            be = "CPU"
            env.createSession(modelBytes, options(xnnpack = false))
        }
        session = s
        backend = be
        val inp = session.inputInfo.entries.first()
        inputName = inp.key
        val shape = (inp.value.info as TensorInfo).shape // [1, 3, H, W], -1 for dynamic
        inH = if (shape.size == 4 && shape[2] > 0) shape[2].toInt() else 640
        inW = if (shape.size == 4 && shape[3] > 0) shape[3].toInt() else inH

        val outShape = (session.outputInfo.values.first().info as TensorInfo).shape
        val metaNames = try {
            session.metadata.customMetadata["names"]?.let { YoloDecoder.parseNamesMetadata(it) }
        } catch (e: Exception) {
            null
        }
        labels = when {
            !labelsFromFile.isNullOrEmpty() -> labelsFromFile
            !metaNames.isNullOrEmpty() -> metaNames
            else -> throw IllegalArgumentException("$name: no labels file and no names in model metadata")
        }
        if (outShape.size == 3) {
            val ch = minOf(outShape[1], outShape[2])
            if (ch > 0 && ch - 4 != labels.size.toLong()) {
                throw IllegalArgumentException("$name: model has ${ch - 4} classes but ${labels.size} labels given")
            }
        }

        canvasBmp = Bitmap.createBitmap(inW, inH, Bitmap.Config.ARGB_8888)
        canvas = Canvas(canvasBmp)
        pixels = IntArray(inW * inH)
        input = ByteBuffer.allocateDirect(4 * 3 * inW * inH).order(ByteOrder.nativeOrder()).asFloatBuffer()
        Log.i(TAG, "$name: loaded ${inW}x$inH, ${labels.size} classes, output ${outShape.contentToString()}, backend $backend")
    }

    private fun options(xnnpack: Boolean): OrtSession.SessionOptions {
        val o = OrtSession.SessionOptions()
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        if (xnnpack) {
            val threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
            o.addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
            // XNNPACK has its own thread pool, ORT docs recommend 1 intra-op thread with it
            o.setIntraOpNumThreads(1)
            o.addConfigEntry("session.intra_op.allow_spinning", "0")
        }
        return o
    }

    /** Letterbox (114 gray, centered) and write RGB float [0,1] NCHW into [input]. */
    private fun preprocess(src: Bitmap): Letterbox {
        val lb = Letterbox.compute(src.width, src.height, inW, inH)
        canvas.drawColor(Color.rgb(YoloDecoder.PAD_VALUE, YoloDecoder.PAD_VALUE, YoloDecoder.PAD_VALUE))
        canvas.drawBitmap(src, null, RectF(lb.padX, lb.padY, lb.padX + lb.newW, lb.padY + lb.newH), paint)
        canvasBmp.getPixels(pixels, 0, inW, 0, 0, inW, inH)
        val plane = inW * inH
        input.rewind()
        for (i in 0 until plane) {
            val p = pixels[i]
            input.put(i, ((p shr 16) and 0xff) / 255f)
            input.put(plane + i, ((p shr 8) and 0xff) / 255f)
            input.put(2 * plane + i, (p and 0xff) / 255f)
        }
        return lb
    }

    fun detect(frame: Bitmap): List<Detection> {
        val lb = preprocess(frame)
        val t0 = System.nanoTime()
        val raw: FloatArray
        val d1: Int
        val d2: Int
        OnnxTensor.createTensor(env, input, longArrayOf(1, 3, inH.toLong(), inW.toLong())).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { res ->
                val out = res[0] as OnnxTensor
                val shape = out.info.shape
                d1 = shape[shape.size - 2].toInt()
                d2 = shape[shape.size - 1].toInt()
                val fb = out.floatBuffer
                raw = FloatArray(fb.remaining())
                fb.get(raw)
            }
        }
        lastMs = (System.nanoTime() - t0) / 1e6
        return YoloDecoder.decode(raw, d1, d2, lb, labels, confThres, iouThres, name)
    }

    override fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "Wazeomatic"
    }
}
