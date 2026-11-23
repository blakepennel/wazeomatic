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
    /** Per-type overrides of [minHits]; Police needs two hits (see DEFAULT_MIN_HITS_BY_HAZARD). */
    val minHitsByHazard: Map<String, Int> = DEFAULT_MIN_HITS_BY_HAZARD,
    val minConfidence: Float = 0.40f,
    /** Per-label overrides of [minConfidence]; potholes ship at 0.55 (see DEFAULT_LABEL_FLOORS). */
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
    // a dog or cat beside a person standing on the same ground is a pet being walked on the pavement, not an
    // animal loose on the road
    val petLabels: Set<String> = setOf("dog", "cat"),
    /** Horizontal gap to the person's box, in person heights. */
    val petReach: Float = 0.75f,
    // an object lying on the road touches the road: a box of these labels whose bottom edge is above roadTop
    // (a vent on a building front) is not on the road, nor is one mostly inside a person or a vehicle
    val roadLabels: Set<String> = setOf("debris"),
    /** The horizon row (0.5 on a level mount) less 0.15 for tilt; Settings derives it from cameraGeometry. */
    val roadTop: Float = 0.35f,
    val onObjectInside: Float = 0.5f,
    val vehicleMinArea: Float = 0.02f,
    val vehicleMinGrowth: Float = 1.35f,
)

/** Same value as DEFAULT_LABEL_FLOORS in wazeomatic/report_manager.py. */
val DEFAULT_LABEL_FLOORS: Map<String, Float> = mapOf("pothole" to 0.55f, "police_vehicle" to 0.85f, "worker" to 0.6f, "debris" to 0.5f)

/** Same value as DEFAULT_MIN_HITS_BY_HAZARD in wazeomatic/report_manager.py: two classifier hits above the
 *  police floor in ten frames (measured with the classifiers at their trained 160 px input). */
val DEFAULT_MIN_HITS_BY_HAZARD: Map<String, Int> = mapOf("POLICE" to 2)

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
        val people = all.filter { it.label == "person" }.map { it.bbox }
        for (d in all) {
            val hazard = hazardFor(d) ?: continue
            if (d.confidence < minConf(d.label)) continue
            // scene labels are whole-frame boxes, the hood zone does not apply to them
            if (d.model != "scene" && inHoodZone(d)) continue
            if (d.label in cfg.liveryLabels && vehicles.any { inside(d.bbox, it) >= cfg.liveryInside }) continue
            if (d.label in cfg.petLabels && people.any { beside(d.bbox, it, cfg.petReach) }) continue
            if (d.label in cfg.roadLabels && (d.bbox.cy + d.bbox.h / 2f < cfg.roadTop ||
                    (people + vehicles).any { inside(d.bbox, it) >= cfg.onObjectInside })) continue
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
        // the type reported longest ago goes first when the global cooldown frees, so a work zone that keeps
        // qualifying does not hold back a police car every minute (sortedBy is stable: ties keep first-seen order)
        for ((hazard, tr) in tracks.entries.sortedBy { it.value.lastEmittedTs }) {
            val hits = qualifyingHits(tr, cfg.minHitsByHazard[hazard] ?: cfg.minHits) ?: continue
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
    /** [pet] stands next to [person]: a horizontal gap of at most [reach] person heights, and the bottoms of
     *  the two boxes (where they touch the ground) within half a person height. Twin of report_manager._beside. */
    private fun beside(pet: BBox, person: BBox, reach: Float): Boolean {
        val gap = maxOf(0f, kotlin.math.abs(pet.cx - person.cx) - (pet.w + person.w) / 2)
        val ground = kotlin.math.abs((pet.cy + pet.h / 2) - (person.cy + person.h / 2))
        return gap <= reach * person.h && ground <= 0.5f * person.h
    }

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

    /** The hits that justify a report now (at least [need] in the window), or null. */
    private fun qualifyingHits(tr: Track, need: Int): Collection<Hit>? {
        if (cfg.trackIou <= 0f) return if (tr.hits.size >= need) tr.hits else null
        val ready = tr.tracklets.filter { it.hits.size >= need }
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
