package com.wazeomatic.detector

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

// Port of wazeomatic/report_manager.py: persistence window, per-type cooldown, global
// cooldown, the (optional) stopped-vehicle heuristic, per-label confidence floors and the
// optional spatial tracking (hits must chain through overlapping boxes), the longer cooldowns of
// scene conditions and the Waze check (a hazard Waze already shows is counted, not sent). Same defaults.

data class ReportConfig(
    val window: Int = 10,
    val minHits: Int = 3,
    val minConfidence: Float = 0.40f,
    /** Per-label overrides of [minConfidence]; potholes ship at 0.45 (see DEFAULT_LABEL_FLOORS). */
    val minConfidenceByLabel: Map<String, Float> = DEFAULT_LABEL_FLOORS,
    /**
     * > 0: the hits behind a report must come from boxes that overlap frame to frame (grown by
     * [trackGrow]) or whose centres lie within [trackGrow] box sizes; 0 = any box of the type counts.
     */
    val trackIou: Float = 0f,
    val trackGrow: Float = 1.5f,
    // ignore boxes whose bottom edge is in the lowest 10 % of the frame: the car's own bonnet reflection
    val hoodFrac: Float = 0.10f,
    val cooldownS: Double = 60.0,
    /** Longer per-type cooldowns for conditions that last minutes: one report per traffic jam or fog
     *  bank, not one a minute for as long as it lasts. Overrides [cooldownS] for these types. */
    val cooldownByHazard: Map<String, Double> = SCENE_COOLDOWNS,
    /** Between any two events. The macro files at most one report a minute in total, so a sooner
     *  event would be dropped there unseen; holding it back here keeps the loss visible ([ReportManager.heldBack]). */
    val globalCooldownS: Double = 60.0,
    val classToHazard: Map<String, String> = Schema.DEFAULT_CLASS_TO_HAZARD,
    val stoppedVehicleEnabled: Boolean = false,
    val shoulderXMin: Float = 0.62f,
    val shoulderYMin: Float = 0.40f,
    val vehicleLabels: Set<String> = setOf("car", "truck", "bus"),
    // red-and-white chevrons on the back of a works van look like a barricade; a box of these labels that lies
    // almost wholly inside a vehicle box is the vehicle's livery, not a barrier on the road
    val liveryLabels: Set<String> = setOf("barricade"),
    val liveryInside: Float = 0.8f,
    val vehicleMinArea: Float = 0.02f,
    val vehicleMinGrowth: Float = 1.35f,
)

/** Same value as DEFAULT_LABEL_FLOORS in wazeomatic/report_manager.py. */
val DEFAULT_LABEL_FLOORS: Map<String, Float> = mapOf("pothole" to 0.45f, "police_vehicle" to 0.85f)

/** Same values as SCENE_COOLDOWNS in wazeomatic/scene.py (seconds). */
val SCENE_COOLDOWNS: Map<String, Double> = mapOf(
    "TRAFFIC_HEAVY" to 900.0, "TRAFFIC_STANDSTILL" to 900.0,
    "WEATHER_FOG" to 1800.0, "WEATHER_FLOOD" to 900.0, "WEATHER_SLIPPERY" to 1800.0, "WEATHER_ICE" to 1800.0,
    "WEATHER_UNPLOWED" to 1800.0, "CRASH" to 300.0, "CRASH_OTHER_SIDE" to 300.0,
    "BLOCKED_LANE_LEFT" to 300.0, "BLOCKED_LANE_CENTER" to 300.0, "BLOCKED_LANE_RIGHT" to 300.0,
    "HAZARD_ON_ROAD_TRAFFIC_LIGHT" to 600.0,
)

/** A hazard that qualified while Waze already showed the same report nearby: counted, not sent. */
data class Confirmation(
    val hazardType: String,
    val confidence: Double,
    val label: String,
    val ts: Double,
    val frameIndex: Int,
    val waze: WazeAlert,
) {
    /** Same keys as the confirmation dicts of report_manager.py. */
    fun toJson(): String = "{" +
        "\"hazard_type\":" + WazeText.quote(hazardType) + "," +
        "\"confidence\":" + String.format(java.util.Locale.US, "%.3f", confidence) + "," +
        "\"label\":" + WazeText.quote(label) + "," +
        "\"ts\":" + String.format(java.util.Locale.US, "%.3f", ts) + "," +
        "\"frame_index\":" + frameIndex + "," +
        "\"waze\":" + waze.toJson() +
        "}"
}

class ReportManager(
    val cfg: ReportConfig = ReportConfig(),
    val source: String = "phone",
    /** Read-only view of the Waze screen: a hazard Waze already shows nearby is a confirmation, not a new report. */
    private val waze: WazeAlerts? = null,
    // last, so tests can pass the clock as a trailing lambda
    private val clock: () -> Double = ::nowSeconds,
) {
    private class Hit(val frame: Int, val conf: Float, val bbox: BBox, val label: String)

    private class Tracklet(var bbox: BBox) {
        val hits = ArrayDeque<Hit>()
    }

    private class Track {
        val hits = ArrayDeque<Hit>()
        var lastEmittedTs = -1e9
        val tracklets = ArrayList<Tracklet>()
    }

    private val tracks = LinkedHashMap<String, Track>()
    private var lastGlobalTs = -1e9
    private var frame = 0

    /** Hazards that qualified while another type's report held the global cooldown, counted once
     *  per hazard type per cooldown window. */
    var heldBack = 0
        private set
    private val held = HashSet<String>()
    private var hoodFracDynamic = 0f

    /** Events not sent because Waze already showed them ([confirmations] has the details). */
    @Volatile var alreadyReported = 0
        private set
    private val confirmationList = ArrayList<Confirmation>()
    val confirmations: List<Confirmation> get() = confirmationList

    /** Active hood zone: the configured minimum or a larger value learnt by [HoodEstimator]. */
    val hoodFrac: Float get() = maxOf(cfg.hoodFrac, hoodFracDynamic)

    fun setHoodFrac(frac: Float) {
        hoodFracDynamic = frac
    }

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
        val accepted = LinkedHashMap<String, ArrayList<Detection>>()
        val all = detections.toList()
        val vehicles = all.filter { it.label in cfg.vehicleLabels }.map { it.bbox }
        for (d in all) {
            val hazard = hazardFor(d) ?: continue
            if (d.confidence < minConf(d.label)) continue
            // scene labels are whole-frame boxes, the hood zone does not apply to them
            if (d.model != "scene" && inHoodZone(d)) continue
            if (d.label in cfg.liveryLabels && vehicles.any { inside(d.bbox, it) >= cfg.liveryInside }) continue
            accepted.getOrPut(hazard) { ArrayList() }.add(d)
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
        if (cfg.trackIou > 0f) updateTracklets(accepted)

        val events = ArrayList<HazardEvent>()
        for ((hazard, tr) in tracks) {
            val hits = qualifyingHits(tr) ?: continue
            if (now - tr.lastEmittedTs < (cfg.cooldownByHazard[hazard] ?: cfg.cooldownS)) continue
            if (hazard == "HAZARD_SHOULDER_CAR_STOPPED" && !stoppedVehicleConfirmed(tr)) continue
            val cover = waze?.covering(hazard)
            if (cover != null) {
                // Waze already shows it: count the agreement, mute the type for a cooldown, send nothing.
                // The global cooldown is left alone because no report goes out.
                val top = hits.maxByOrNull { it.conf } ?: continue
                alreadyReported++
                confirmationList.add(Confirmation(hazard, top.conf.toString().toDouble(), top.label, now, frame, cover))
                tr.lastEmittedTs = now
                tr.hits.clear()
                tr.tracklets.clear()
                continue
            }
            if (now - lastGlobalTs < cfg.globalCooldownS) {
                if (held.add(hazard)) heldBack++
                continue
            }
            val top = hits.maxByOrNull { it.conf } ?: continue
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
            held.clear()
            tr.hits.clear()
            tr.tracklets.clear()
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

    /** A per-label floor can only raise the bar above minConfidence, never lower it. */
    private fun minConf(label: String): Float =
        maxOf(cfg.minConfidence, cfg.minConfidenceByLabel[label] ?: 0f)

    /** Fraction of box [a] that lies inside box [b]. */
    private fun inside(a: BBox, b: BBox): Float {
        val ix = maxOf(0f, minOf(a.cx + a.w / 2, b.cx + b.w / 2) - maxOf(a.cx - a.w / 2, b.cx - b.w / 2))
        val iy = maxOf(0f, minOf(a.cy + a.h / 2, b.cy + b.h / 2) - maxOf(a.cy - a.h / 2, b.cy - b.h / 2))
        return ix * iy / maxOf(a.w * a.h, 1e-9f)
    }

    private fun inHoodZone(d: Detection): Boolean {
        val hood = hoodFrac
        if (hood <= 0f) return false
        // a police car or ambulance close by reaches the bottom of the frame; only a vehicle box that lies
        // entirely in the hood zone is the car itself
        if (d.model == "vehicle") return (d.bbox.cy - d.bbox.h / 2f) > (1f - hood)
        return (d.bbox.cy + d.bbox.h / 2f) > (1f - hood)
    }

    private fun inShoulder(d: Detection): Boolean {
        val b = d.bbox
        return b.cx >= cfg.shoulderXMin && b.cy >= cfg.shoulderYMin && b.w * b.h >= cfg.vehicleMinArea
    }

    // ------------------------------------------------------------------ spatial tracking

    /** Greedy association of this frame's accepted boxes to per-hazard tracklets. */
    private fun updateTracklets(accepted: Map<String, List<Detection>>) {
        for ((hazard, dets) in accepted) {
            val tr = tracks.getOrPut(hazard) { Track() }
            val used = HashSet<Int>()
            for (d in dets.sortedByDescending { it.confidence }) {
                var bestI = -1
                var bestIou = -1f
                for ((i, tl) in tr.tracklets.withIndex()) {
                    if (i in used || !associated(tl.bbox, d.bbox)) continue
                    val v = grownIou(tl.bbox, d.bbox, cfg.trackGrow)
                    if (v > bestIou) { bestI = i; bestIou = v }
                }
                val hit = Hit(frame, d.confidence, d.bbox, d.label)
                if (bestI < 0) {
                    tr.tracklets.add(Tracklet(d.bbox).also { it.hits.addLast(hit) })
                    used.add(tr.tracklets.size - 1)
                } else {
                    val tl = tr.tracklets[bestI]
                    tl.bbox = d.bbox
                    tl.hits.addLast(hit)
                    used.add(bestI)
                }
            }
        }
        for (tr in tracks.values) {
            for (tl in tr.tracklets) {
                while (tl.hits.isNotEmpty() && tl.hits.first().frame <= frame - cfg.window) tl.hits.removeFirst()
            }
            tr.tracklets.removeAll { it.hits.isEmpty() }
        }
    }

    /** The hits that justify a report now, or null. */
    private fun qualifyingHits(tr: Track): Collection<Hit>? {
        if (cfg.trackIou <= 0f) return if (tr.hits.size >= cfg.minHits) tr.hits else null
        val ready = tr.tracklets.filter { it.hits.size >= cfg.minHits }
        if (ready.isEmpty()) return null
        return ready.maxByOrNull { tl -> tl.hits.maxOf { it.conf } }!!.hits
    }

    private fun associated(a: BBox, b: BBox): Boolean {
        if (grownIou(a, b, cfg.trackGrow) >= cfg.trackIou) return true
        val dx = a.cx - b.cx
        val dy = a.cy - b.cy
        return sqrt(dx * dx + dy * dy) <= cfg.trackGrow * maxOf(a.w, a.h, b.w, b.h)
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

    companion object {
        /** IoU of two boxes after scaling both by [grow] around their centres. */
        fun grownIou(a: BBox, b: BBox, grow: Float): Float {
            val ax1 = a.cx - a.w * grow / 2; val ay1 = a.cy - a.h * grow / 2
            val ax2 = a.cx + a.w * grow / 2; val ay2 = a.cy + a.h * grow / 2
            val bx1 = b.cx - b.w * grow / 2; val by1 = b.cy - b.h * grow / 2
            val bx2 = b.cx + b.w * grow / 2; val by2 = b.cy + b.h * grow / 2
            val iw = max(0f, min(ax2, bx2) - max(ax1, bx1))
            val ih = max(0f, min(ay2, by2) - max(ay1, by1))
            val inter = iw * ih
            val union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            return if (union > 0f) inter / union else 0f
        }
    }
}
