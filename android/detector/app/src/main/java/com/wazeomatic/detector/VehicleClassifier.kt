package com.wazeomatic.detector

/**
 * Second stage for the stock detector's vehicles (twin of classifiers.VehicleClassifier): the largest few
 * car / truck / bus boxes are cropped, padded to a square and classified as civilian, police, ambulance or fire
 * engine. Emergency classes come back as detections with the vehicle's box (model "vehicle"), so the report
 * manager treats them like any other hazard class. [maxCrops] bounds the per-frame cost.
 */
class VehicleClassifier(
    val clf: ImageClassifier,
    val maxCrops: Int = 4,
    val minBoxPx: Int = 28,
    val minVehicleConf: Float = 0.35f,
    val pad: Double = 0.12,
    val minProb: Float = 0.05f,
) : java.io.Closeable {
    var lastMs = 0.0
        private set
    /** Probabilities of the last frame's crops, for logs. */
    var lastProbs: List<Pair<Detection, Map<String, Float>>> = emptyList()
        private set

    fun candidates(width: Int, height: Int, dets: List<Detection>): List<Detection> =
        dets.filter {
            it.label in VEHICLE_LABELS && it.confidence >= minVehicleConf &&
                minOf(it.bbox.w * width, it.bbox.h * height) >= minBoxPx
        }.sortedByDescending { it.bbox.w * it.bbox.h }.take(maxCrops)

    operator fun invoke(image: RgbImage, dets: List<Detection>): List<Detection> {
        val cand = candidates(image.width, image.height, dets)
        if (cand.isEmpty()) {
            lastProbs = emptyList()
            lastMs = 0.0
            return emptyList()
        }
        val crops = cand.map { ImageOps.squareCrop(image, it.bbox, pad, clf.size) }
        val probs = clf.predict(crops)
        lastMs = clf.lastMs
        val out = ArrayList<Detection>()
        val seen = ArrayList<Pair<Detection, Map<String, Float>>>()
        for ((d, p) in cand.zip(probs.toList())) {
            val pd = LinkedHashMap<String, Float>()
            for ((i, lab) in clf.labels.withIndex()) if (i < p.size) pd[lab] = p[i]
            seen.add(d to pd)
            for ((cls, label) in VEHICLE_CLASS_TO_LABEL) {
                val v = pd[cls] ?: continue
                if (v >= minProb) out.add(Detection(label, v, d.bbox, "vehicle"))
            }
        }
        lastProbs = seen
        return out
    }

    override fun close() = clf.close()

    companion object {
        val VEHICLE_LABELS = setOf("car", "truck", "bus")
        /** classifier class -> detection label handed to the report manager and the scene rules */
        val VEHICLE_CLASS_TO_LABEL = linkedMapOf("police" to "police_vehicle", "ambulance" to "ambulance", "fire" to "fire_truck")
    }
}
