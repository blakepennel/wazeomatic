package com.wazeomatic.detector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.Closeable
import java.nio.FloatBuffer
import kotlin.math.exp

/** An image classifier: square RGB crops of [size] in, one probability row per crop out. */
interface ImageClassifier : Closeable {
    val size: Int
    val labels: List<String>
    /** Time of the last predict() call. */
    val lastMs: Double
    fun predict(crops: List<RgbImage>): Array<FloatArray>
    override fun close() {}
}

/**
 * Ultralytics -cls model exported to ONNX (tools/train_classifier.py): input [N, 3, S, S] RGB in [0, 1],
 * output [N, C]. Twin of classifiers.OnnxClassifier: a dynamic batch runs all crops at once, a fixed batch one
 * crop per run, and logits get a softmax here.
 */
class OnnxClassifier(
    private val env: OrtEnvironment,
    modelBytes: ByteArray,
    override val labels: List<String>,
    val name: String,
    threads: Int = 2,
) : ImageClassifier {
    private val session: OrtSession
    private val inputName: String
    override val size: Int
    val dynamicBatch: Boolean
    override var lastMs = 0.0
        private set

    init {
        val o = OrtSession.SessionOptions()
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        o.setIntraOpNumThreads(threads)
        session = env.createSession(modelBytes, o)
        val inp = session.inputInfo.entries.first()
        inputName = inp.key
        val shape = (inp.value.info as TensorInfo).shape
        // a fixed input wins; a dynamic one (train_classifier exports dynamic=True) runs at the size the model
        // was trained at, which Ultralytics writes into the metadata as "imgsz": "[160, 160]"
        size = if (shape.size == 4 && shape[2] > 0) shape[2].toInt()
        else trainedSize(runCatching { session.metadata.customMetadata }.getOrNull().orEmpty()) ?: 128
        dynamicBatch = shape.isEmpty() || shape[0] <= 0
        val outShape = (session.outputInfo.values.first().info as TensorInfo).shape
        val classes = outShape.lastOrNull() ?: -1
        require(labels.isNotEmpty()) { "$name: empty labels file" }
        require(classes <= 0 || classes == labels.size.toLong()) { "$name: model has $classes classes but ${labels.size} labels" }
    }

    override fun predict(crops: List<RgbImage>): Array<FloatArray> {
        if (crops.isEmpty()) return emptyArray()
        val t0 = System.nanoTime()
        val plane = 3 * size * size
        val rows = ArrayList<FloatArray>(crops.size)
        val batches = if (dynamicBatch) listOf(crops) else crops.map { listOf(it) }
        for (batch in batches) {
            val x = FloatArray(batch.size * plane)
            batch.forEachIndexed { i, c ->
                require(c.width == size && c.height == size) { "$name: crop ${c.width}x${c.height}, model wants $size" }
                ImageOps.toChw(c, x, i * plane)
            }
            OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(batch.size.toLong(), 3, size.toLong(), size.toLong())).use { t ->
                session.run(mapOf(inputName to t)).use { res ->
                    val out = res[0] as OnnxTensor
                    val fb = out.floatBuffer
                    val all = FloatArray(fb.remaining())
                    fb.get(all)
                    val c = all.size / batch.size
                    for (i in batch.indices) rows.add(all.copyOfRange(i * c, (i + 1) * c))
                }
            }
        }
        lastMs = (System.nanoTime() - t0) / 1e6
        return softmaxIfNeeded(rows.toTypedArray())
    }

    override fun close() = session.close()

    companion object {
        /** Logits (a negative value, or a first row that does not sum to 1) get a row-wise softmax. */
        fun softmaxIfNeeded(out: Array<FloatArray>): Array<FloatArray> {
            if (out.isEmpty()) return out
            val min = out.minOf { r -> r.minOrNull() ?: 0f }
            if (min >= 0f && kotlin.math.abs(out[0].sum() - 1.0) <= 0.05) return out
            return Array(out.size) { i ->
                val r = out[i]
                val m = r.maxOrNull() ?: 0f
                val e = DoubleArray(r.size) { exp((r[it] - m).toDouble()) }
                val s = e.sum()
                FloatArray(r.size) { (e[it] / s).toFloat() }
            }
        }

        /** Square input size from Ultralytics export metadata ("imgsz": "[160, 160]" or "160"), else null. */
        fun trainedSize(meta: Map<String, String>): Int? =
            Regex("\\d+").find(meta["imgsz"].orEmpty())?.value?.toIntOrNull()?.takeIf { it > 0 }

        /** Labels file text (one class per line, output order) to a list. */
        fun parseLabels(text: String): List<String> = text.lines().map { it.trim() }.filter { it.isNotEmpty() }
    }
}
