package com.wazeomatic.detector

/**
 * One part of the scene classifiers: (image, t, detections) -> class probabilities, or null when it did not run.
 * The detections are the frame's boxes; the top-view road surface skips a frame whose road patch they cover.
 */
interface ScenePart : java.io.Closeable {
    val lastMs: Double
    operator fun invoke(image: RgbImage, t: Double, dets: List<Detection> = emptyList()): Map<String, Float>?
    fun setHoodFrac(frac: Double) {}
    override fun close() {}
}

/**
 * Whole-frame weather classifier (clear, rain, snow, fog), run every [everyS] seconds of frame time; the frame
 * is squashed to S x S with INTER_AREA, as in training. Twin of classifiers.SceneClassifier.
 */
class SceneClassifier(val clf: ImageClassifier, val everyS: Double = 0.5) : ScenePart {
    private var lastT = -1e9
    var last: Map<String, Float> = emptyMap()
        private set
    override var lastMs = 0.0
        private set

    override fun invoke(image: RgbImage, t: Double, dets: List<Detection>): Map<String, Float>? {
        if (t - lastT < everyS) {
            lastMs = 0.0
            return null
        }
        lastT = t
        val s = clf.size
        val p = clf.predict(listOf(ImageOps.resize(image, s, s, area = true)))[0]
        lastMs = clf.lastMs
        last = clf.labels.withIndex().filter { it.index < p.size }.associate { it.value to p[it.index] }
        return last
    }

    override fun close() = clf.close()
}

/** Weather and road surface together, as one (image, t) -> probs call for the pipeline. */
class SceneClassifiers(parts: List<ScenePart?>) : java.io.Closeable {
    val parts: List<ScenePart> = parts.filterNotNull()
    var lastMs = 0.0
        private set

    fun setHoodFrac(frac: Double) = parts.forEach { it.setHoodFrac(frac) }

    operator fun invoke(image: RgbImage, t: Double, dets: List<Detection> = emptyList()): Map<String, Float>? {
        val merged = LinkedHashMap<String, Float>()
        var ms = 0.0
        for (p in parts) {
            val r = p(image, t, dets)
            ms += p.lastMs
            if (!r.isNullOrEmpty()) merged.putAll(r)
        }
        lastMs = ms
        return merged.ifEmpty { null }
    }

    override fun close() = parts.forEach { try { it.close() } catch (_: Exception) {} }
}
