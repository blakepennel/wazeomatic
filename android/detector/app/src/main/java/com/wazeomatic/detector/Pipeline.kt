package com.wazeomatic.detector

import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.Closeable
import java.util.Locale

/**
 * Frame -> all loaded models -> merged detections -> ReportManager -> Reporter.
 * process() must only be called from one thread at a time (the frame source's thread).
 */
class Pipeline(ctx: Context, private val settings: Settings, private val reporter: Reporter) : Closeable {
    val detectors: List<OnnxDetector>
    val modelStatus: String
    private val reportManager = ReportManager(settings.reportConfig(), source = "phone")
    private var frameIndex = 0
    private var lastFrameNs = 0L
    private var fps = 0.0

    init {
        val env = OrtEnvironment.getEnvironment()
        val list = ArrayList<OnnxDetector>()
        val notes = ArrayList<String>()
        for (name in listOf("hazard", "coco")) {
            val bytes = readAsset(ctx, "$name.onnx")
            if (bytes == null) {
                notes.add("$name: missing")
                continue
            }
            val labels = readAsset(ctx, "${name}_labels.txt")?.toString(Charsets.UTF_8)
                ?.lines()?.map { it.trim() }?.filter { it.isNotEmpty() }
            try {
                val d = OnnxDetector(env, bytes, labels, name, settings.effectiveDetConf, settings.iou)
                list.add(d)
                notes.add("$name: ${d.inW}x${d.inH} ${d.labels.size} cls ${d.backend}")
            } catch (e: Exception) {
                Log.e(TAG, "failed to load $name.onnx", e)
                notes.add("$name: load failed (${e.message})")
            }
        }
        detectors = list
        modelStatus = if (list.isEmpty()) "model missing" else notes.joinToString(" | ")
        Log.i(TAG, "models: $modelStatus")
    }

    val hasModel: Boolean get() = detectors.isNotEmpty()

    fun process(frame: Bitmap) {
        frameIndex++
        val lat = LinkedHashMap<String, Double>()
        val dets = ArrayList<Detection>()
        for (d in detectors) {
            try {
                dets.addAll(d.detect(frame))
                lat[d.name] = d.lastMs
            } catch (e: Exception) {
                Log.e(TAG, "inference failed in ${d.name}", e)
            }
        }
        val now = System.nanoTime()
        if (lastFrameNs != 0L) {
            val inst = 1e9 / (now - lastFrameNs)
            fps = if (fps == 0.0) inst else fps * 0.8 + inst * 0.2
        }
        lastFrameNs = now
        if (lat.isNotEmpty()) {
            Log.d(PERF_TAG, String.format(Locale.US, "frame=%d %s fps=%.1f dets=%d", frameIndex,
                lat.entries.joinToString(" ") { String.format(Locale.US, "%s=%.1fms", it.key, it.value) }, fps, dets.size))
        }
        if (hasModel) {
            for (ev in reportManager.update(dets, frameIndex)) reporter.report(ev)
        }
        PipelineBus.postFrame(FrameResult(frame, dets, lat, fps, frameIndex))
    }

    override fun close() {
        detectors.forEach { try { it.close() } catch (_: Exception) {} }
    }

    companion object {
        private const val TAG = "Wazeomatic"
        const val PERF_TAG = "WazeomaticPerf"

        fun readAsset(ctx: Context, name: String): ByteArray? = try {
            ctx.assets.open(name).use { it.readBytes() }
        } catch (e: Exception) {
            null
        }

        fun assetExists(ctx: Context, name: String): Boolean =
            try { ctx.assets.list("")?.contains(name) == true } catch (e: Exception) { false }
    }
}
