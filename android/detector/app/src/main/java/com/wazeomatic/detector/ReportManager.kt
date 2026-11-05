package com.wazeomatic.detector

// Port of wazeomatic/report_manager.py: persistence window, per-type cooldown,
// global cooldown and the (optional) stopped-vehicle heuristic. Same defaults.

data class ReportConfig(
    val window: Int = 10,
    val minHits: Int = 3,
    val minConfidence: Float = 0.45f,
    val cooldownS: Double = 60.0,
    val globalCooldownS: Double = 5.0,
    val classToHazard: Map<String, String> = Schema.DEFAULT_CLASS_TO_HAZARD,
    val stoppedVehicleEnabled: Boolean = false,
    val shoulderXMin: Float = 0.62f,
    val shoulderYMin: Float = 0.40f,
    val vehicleLabels: Set<String> = setOf("car", "truck", "bus"),
    val vehicleMinArea: Float = 0.02f,
    val vehicleMinGrowth: Float = 1.35f,
)

class ReportManager(
    val cfg: ReportConfig = ReportConfig(),
    val source: String = "phone",
    private val clock: () -> Double = ::nowSeconds,
) {
    private class Hit(val frame: Int, val conf: Float, val bbox: BBox, val label: String)

    private class Track {
        val hits = ArrayDeque<Hit>()
        var lastEmittedTs = -1e9
    }

    private val tracks = LinkedHashMap<String, Track>()
    private var lastGlobalTs = -1e9
    private var frame = 0

    /** Feed one inference frame; returns zero or more events to report. */
    fun update(
        detections: Iterable<Detection>,
        frameIndex: Int? = null,
        lat: Double? = null,
        lon: Double? = null,
    ): List<HazardEvent> {
        frame = frameIndex ?: (frame + 1)
        val now = clock()
        val best = LinkedHashMap<String, Detection>()
        for (d in detections) {
            val hazard = hazardFor(d) ?: continue
            if (d.confidence < cfg.minConfidence) continue
            val prev = best[hazard]
            if (prev == null || d.confidence > prev.confidence) best[hazard] = d
        }
        // one hit per hazard per frame, then trim the window
        for ((hazard, d) in best) {
            tracks.getOrPut(hazard) { Track() }.hits.addLast(Hit(frame, d.confidence, d.bbox, d.label))
        }
        for (tr in tracks.values) {
            while (tr.hits.isNotEmpty() && tr.hits.first().frame <= frame - cfg.window) tr.hits.removeFirst()
        }

        val events = ArrayList<HazardEvent>()
        for ((hazard, tr) in tracks) {
            if (tr.hits.size < cfg.minHits) continue
            if (now - tr.lastEmittedTs < cfg.cooldownS) continue
            if (now - lastGlobalTs < cfg.globalCooldownS) continue
            if (hazard == "HAZARD_SHOULDER_CAR_STOPPED" && !stoppedVehicleConfirmed(tr)) continue
            val top = tr.hits.maxByOrNull { it.conf } ?: continue
            events.add(
                HazardEvent(
                    hazardType = hazard,
                    confidence = top.conf.toString().toDouble(),
                    source = source,
                    ts = now,
                    frameIndex = frame,
                    bbox = top.bbox,
                    label = top.label,
                    lat = lat,
                    lon = lon,
                )
            )
            tr.lastEmittedTs = now
            lastGlobalTs = now
            tr.hits.clear()
        }
        return events
    }

    fun hazardFor(d: Detection): String? {
        cfg.classToHazard[d.label]?.let { return it }
        if (cfg.stoppedVehicleEnabled && d.label in cfg.vehicleLabels && inShoulder(d)) {
            return "HAZARD_SHOULDER_CAR_STOPPED"
        }
        return null
    }

    private fun inShoulder(d: Detection): Boolean {
        val b = d.bbox
        return b.cx >= cfg.shoulderXMin && b.cy >= cfg.shoulderYMin && b.w * b.h >= cfg.vehicleMinArea
    }

    /** A parked car grows in the frame as we approach and stays on the right edge. */
    private fun stoppedVehicleConfirmed(tr: Track): Boolean {
        if (tr.hits.size < 2) return false
        val first = tr.hits.first().bbox
        val last = tr.hits.last().bbox
        val a0 = first.w * first.h
        val a1 = last.w * last.h
        return a1 >= a0 * cfg.vehicleMinGrowth && last.cx >= first.cx - 0.02f
    }
}
