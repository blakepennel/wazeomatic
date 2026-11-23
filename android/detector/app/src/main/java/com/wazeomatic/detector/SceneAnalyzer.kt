package com.wazeomatic.detector

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

// Scene-level reports: conditions that are not one box in one frame. Twin of wazeomatic/scene.py; keep the
// thresholds, the order of the checks and the defaults in step with it. Everything runs on the frame timestamp
// [t] (seconds), so a replayed clip and the live camera behave the same.

object SceneLabels {
    val CONSTRUCTION = setOf("cone", "barricade", "delineator", "construction_sign", "barrel")
    val VEHICLE = setOf("car", "truck", "bus")
    val EMERGENCY = setOf("police_vehicle", "ambulance", "fire_truck")
    val FEATURES = listOf("traffic", "crash", "lanes", "signals", "weather")
}

/** Classifier class key -> scene label, probability threshold and hold time (scene.DEFAULT_WEATHER_RULES). */
data class WeatherRule(val label: String, val threshold: Double, val holdS: Double)

fun defaultWeatherRules(): LinkedHashMap<String, WeatherRule> = linkedMapOf(
    "fog" to WeatherRule("scene_fog", 0.70, 20.0),
    "surface_water" to WeatherRule("scene_flood", 0.85, 8.0),
    "surface_ice" to WeatherRule("scene_ice", 0.80, 10.0),
    "surface_snow" to WeatherRule("scene_unplowed", 0.75, 15.0),
    // the smaller of rain and (wet + water + slush): rain falling AND the road wet
    "rain&surface_wet|surface_water|surface_slush" to WeatherRule("scene_slippery", 0.65, 60.0),
)

private fun whole(label: String, conf: Double) =
    Detection(label, conf.coerceIn(0.0, 1.0).toFloat(), BBox(0.5f, 0.5f, 1f, 1f), "scene")

/** Whether a run of boxes (oldest first) stays put: the centre of the last third is within [maxShift] box widths
 *  of the first third's and the size changed by less than 40 %. */
internal fun standing(boxes: List<BBox>, maxShift: Double): Boolean {
    if (boxes.size < 3) return false
    val third = max(1, boxes.size / 3)
    val first = boxes.take(third)
    val last = boxes.takeLast(third)
    val width = boxes.map { it.w.toDouble() }.average()
    val shift = abs(last.map { it.cx.toDouble() }.average() - first.map { it.cx.toDouble() }.average())
    val a0 = first.map { it.w.toDouble() * it.h }.average()
    val a1 = last.map { it.w.toDouble() * it.h }.average()
    val ratio = a1 / maxOf(a0, 1e-9)
    return width > 0 && shift <= maxShift * width && ratio in 0.7..1.4
}

internal fun boxIou(a: BBox, b: BBox): Double {
    val ax1 = a.cx - a.w / 2.0; val ay1 = a.cy - a.h / 2.0; val ax2 = a.cx + a.w / 2.0; val ay2 = a.cy + a.h / 2.0
    val bx1 = b.cx - b.w / 2.0; val by1 = b.cy - b.h / 2.0; val bx2 = b.cx + b.w / 2.0; val by2 = b.cy + b.h / 2.0
    val iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    val ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    val inter = iw * ih
    val u = a.w.toDouble() * a.h + b.w.toDouble() * b.h - inter
    return if (u > 0) inter / u else 0.0
}

/** Shares of a box (its upper [topFrac]) that are bright, saturated red / amber / green / blue, and lit. */
data class LampFractions(val red: Double, val amber: Double, val green: Double, val blue: Double, val lit: Double)

object Lamps {
    private val NONE = LampFractions(0.0, 0.0, 0.0, 0.0, 0.0)

    fun fractions(img: RgbImage, bbox: BBox, topFrac: Double = 1.0): LampFractions {
        val h = img.height
        val w = img.width
        val cx = bbox.cx.toDouble(); val cy = bbox.cy.toDouble(); val bw = bbox.w.toDouble(); val bh = bbox.h.toDouble()
        val x1 = max(0.0, (cx - bw / 2) * w).toInt()
        val x2 = min(w.toDouble(), (cx + bw / 2) * w).toInt()
        val y1 = max(0.0, (cy - bh / 2) * h).toInt()
        val y2 = min(h.toDouble(), y1 + bh * topFrac * h).toInt()
        if (x2 <= x1 || y2 <= y1 || x1 >= w || y1 >= h) return NONE
        var roi = img.crop(x1, y1, x2, y2)
        if (roi.width * roi.height > 96 * 96) {
            val s = 96.0 / max(roi.width, roi.height)
            roi = ImageOps.resize(roi, max(1, (roi.width * s).toInt()), max(1, (roi.height * s).toInt()), area = true)
        }
        var red = 0; var amber = 0; var green = 0; var blue = 0; var lit = 0
        for (p in roi.argb) {
            val hsv = ImageOps.hsv(p)
            val hh = hsv shr 16
            val ss = (hsv shr 8) and 0xff
            val vv = hsv and 0xff
            if (vv >= 235) lit++
            if (vv < 200 || ss < 120) continue
            when {
                hh <= 8 || hh >= 170 -> red++
                hh <= 30 -> amber++
                hh in 45..95 -> green++
                hh in 100..135 -> blue++
            }
        }
        val n = (roi.width * roi.height).toDouble()
        return LampFractions(red / n, amber / n, green / n, blue / n, lit / n)
    }

    /** How many times a series crosses its midpoint by at least a quarter of its range each way. */
    fun alternations(xs: List<Double>, minAmp: Double): Int {
        if (xs.size < 4) return 0
        val lo = xs.min()
        val hi = xs.max()
        if (hi - lo < minAmp) return 0
        val mid = (hi + lo) / 2
        val band = (hi - lo) * 0.25
        var state: Boolean? = null
        var flips = 0
        for (x in xs) {
            val s = if (x > mid + band) true else if (x < mid - band) false else state
            if (state != null && s != null && s != state) flips++
            if (s != null) state = s
        }
        return flips
    }
}

/**
 * Red / blue flashing on a vehicle. Brake lights are red and steady, hazard flashers amber; a light bar flips
 * red or blue on and off several times a second. A box qualifies when, within [windowS], blue or red flips at
 * least [minFlips] times. Open so tests can stub update().
 */
open class EmergencyLights(
    val windowS: Double = 2.0,
    val minFlips: Int = 3,
    val minAmp: Double = 0.004,
    val minBoxPx: Int = 24,
) {
    private class Track(var bbox: BBox, var lastT: Double) {
        val samples = ArrayDeque<DoubleArray>()      // (t, red, blue), at most 40
    }

    private var tracks = ArrayList<Track>()

    open fun update(t: Double, img: RgbImage, dets: List<Detection>): List<Detection> {
        val h = img.height
        val w = img.width
        val boxes = dets.filter {
            (it.label in SceneLabels.VEHICLE || it.label in SceneLabels.EMERGENCY) &&
                min(it.bbox.w.toDouble() * w, it.bbox.h.toDouble() * h) >= minBoxPx
        }
        val out = ArrayList<Detection>()
        val used = HashSet<Int>()
        for (d in boxes.sortedByDescending { it.bbox.w * it.bbox.h }) {
            var best = -1
            var bestIou = 0.3
            for ((i, tr) in tracks.withIndex()) {
                if (i in used) continue
                val v = boxIou(tr.bbox, d.bbox)
                if (v > bestIou) { best = i; bestIou = v }
            }
            if (best < 0) {
                tracks.add(Track(d.bbox, t))
                best = tracks.size - 1
            }
            used.add(best)
            val tr = tracks[best]
            tr.bbox = d.bbox
            tr.lastT = t
            val f = Lamps.fractions(img, d.bbox, topFrac = 0.6)
            tr.samples.addLast(doubleArrayOf(t, f.red, f.blue))
            while (tr.samples.size > 40) tr.samples.removeFirst()
            val recent = tr.samples.filter { t - it[0] <= windowS }
            val reds = recent.map { it[1] }
            val blues = recent.map { it[2] }
            val flips = max(Lamps.alternations(blues, minAmp), Lamps.alternations(reds, minAmp * 1.5))
            val both = (blues.maxOrNull() ?: 0.0) > minAmp && (reds.maxOrNull() ?: 0.0) > minAmp
            if (flips >= minFlips || (both && flips >= minFlips - 1)) {
                val conf = min(1.0, 0.5 + 0.1 * flips + if (both) 0.15 else 0.0)
                out.add(Detection("emergency_lights", conf.toFloat(), d.bbox, "lights"))
            }
        }
        tracks = ArrayList(tracks.filter { t - it.lastT <= windowS })
        return out
    }
}

/**
 * Is the car moving? From how much the image's horizontal gradients change between frames (scene.EgoMotion).
 * The frame is box-averaged to 80 x 45 grey, the lowest 15 % (hood) and every vehicle / person box are masked,
 * and the score is the median absolute change of the central-difference gradient, scaled to a 10 fps frame gap
 * and smoothed. A fresh GPS speed replaces it.
 */
class EgoMotion(
    val stationaryThr: Double = 0.6,
    val slowThr: Double = 2.5,
    val stationaryMps: Double = 0.8,
    val slowMps: Double = 6.0,
    val hoodFrac: Double = 0.15,
) {
    private var prev: FloatArray? = null
    private var prevT = 0.0
    var flow = 0.0
        private set
    @Volatile var gpsMps: Double? = null
        private set
    @Volatile private var gpsT = -1e9

    fun setGpsSpeed(mps: Double?, t: Double) {
        gpsMps = mps
        gpsT = t
    }

    /** The GPS speed (m/s) when the last fix is under 5 s old, else null. */
    fun gpsSpeed(t: Double): Double? = gpsMps?.takeIf { t - gpsT < 5.0 }

    fun update(t: Double, img: RgbImage, dets: List<Detection>): Double {
        val g = ImageOps.greySmall(img, W, H)
        val gw = W - 2
        val gx = FloatArray(gw * H)
        for (y in 0 until H) for (x in 0 until gw) gx[y * gw + x] = g[y * W + x + 2] - g[y * W + x]
        val p = prev
        if (p != null && t > prevT) {
            val mask = BooleanArray(gw * H) { true }
            for (y in (H * (1 - hoodFrac)).toInt() until H) for (x in 0 until gw) mask[y * gw + x] = false
            for (d in dets) {
                if (d.label !in SceneLabels.VEHICLE && d.label !in SceneLabels.EMERGENCY && d.label != "person") continue
                val cx = d.bbox.cx.toDouble(); val cy = d.bbox.cy.toDouble()
                val bw = d.bbox.w.toDouble(); val bh = d.bbox.h.toDouble()
                val y1 = max(0.0, (cy - bh / 2) * H).toInt()
                val y2 = min(H.toDouble(), ceil((cy + bh / 2) * H)).toInt()
                val x1 = (max(0.0, (cx - bw / 2) * W) - 1).toInt()
                val x2 = min((W - 2).toDouble(), ceil((cx + bw / 2) * W)).toInt()
                for (y in max(0, y1) until min(H, y2)) for (x in max(0, x1) until min(gw, x2)) mask[y * gw + x] = false
            }
            val vals = ArrayList<Float>(gw * H)
            for (i in gx.indices) if (mask[i]) vals.add(abs(gx[i] - p[i]))
            if (vals.size >= 200) {
                val score = median(vals) * min(3.0, 0.1 / (t - prevT))
                flow = 0.6 * flow + 0.4 * score
            }
        }
        prev = gx
        prevT = t
        return flow
    }

    /** stationary | slow | moving */
    fun state(t: Double): String {
        val v = gpsSpeed(t)
        if (v != null) return if (v < stationaryMps) "stationary" else if (v < slowMps) "slow" else "moving"
        val f = flow
        return if (f < stationaryThr) "stationary" else if (f < slowThr) "slow" else "moving"
    }

    companion object {
        const val W = 80
        const val H = 45

        fun median(xs: List<Float>): Double {
            val s = xs.sorted()
            val n = s.size
            return if (n % 2 == 1) s[n / 2].toDouble() else (s[n / 2 - 1].toDouble() + s[n / 2]) / 2.0
        }
    }
}

/** Defaults are scene.SceneConfig's. */
data class SceneConfig(
    val enabled: Set<String> = SceneLabels.FEATURES.toSet(),
    val driveSide: String = "right",            // right-hand traffic (US); "left" mirrors the side rules
    // traffic
    val standstillS: Double = 75.0,
    val standstillStationaryFrac: Double = 0.9,
    val standstillMovingMax: Double = 0.01,
    val heavyWindowS: Double = 150.0,
    val heavySlowFrac: Double = 0.5,
    val heavyStationaryFrac: Double = 0.3,
    val heavyMinStops: Int = 2,
    val heavyMinVehicles: Double = 3.0,
    val aheadMinArea: Double = 0.015,
    val redVetoS: Double = 20.0,
    // crash
    val crashHoldS: Double = 3.0,
    val crashWindowS: Double = 10.0,
    val crashMaxShift: Double = 0.5,      // standing still: box centre moves less than this many box widths
    // the other side of the road: where the box meets the road lies this far to the left of the camera (right in
    // left-hand traffic); same as scene.py other_side_m
    val otherSideM: Double = 4.0,
    val policeOtherSide: Boolean = false,
    val policeStoppedOnly: Boolean = true,  // Police only for a police car that stands (PoliceMotion)
    val shoulderCars: Boolean = false,      // Hazard > Car on shoulder (ShoulderCars), off by default like scene.py
    val shoulderCameraSpeed: Boolean = true,
    // lanes
    val laneMinObjects: Int = 3,
    val laneWindowS: Double = 3.0,
    // flat-road pinhole camera, same as scene.py: a box meeting the road at row gy is
    // fy * camHeightM / (gy - horizonY) m ahead and (cx - 0.5) * camHeightM * aspect / (gy - horizonY) m to the side
    val camHeightM: Double = 1.3,
    val horizonY: Double = 0.50,
    val hfovDeg: Double = 70.0,
    val laneWidthM: Double = 3.6,
    val laneMaxDistM: Double = 40.0,
    val laneMinDistM: Double = 3.0,
    val laneMaxOffset: Double = 2.0,
    val laneReach: Double = 0.8,
    val laneSideU: Double = 0.6,
    val laneHoldS: Double = 1.5,
    // dark signals
    val darkSignalS: Double = 4.0,
    val signalMinPx: Int = 18,
    // weather
    val weatherRules: Map<String, WeatherRule> = defaultWeatherRules(),
    val weatherEma: Double = 0.15,
) {
    companion object {
        /** A --scene-features style comma list; unknown names are dropped. */
        fun parseFeatures(list: String): Set<String> =
            list.split(',').map { it.trim().lowercase() }.filter { it in SceneLabels.FEATURES }.toSet()
    }
}

class SceneAnalyzer(val cfg: SceneConfig = SceneConfig()) {
    val ego = EgoMotion()
    var lights: EmergencyLights = EmergencyLights()
    val motion: PoliceMotion? = if (cfg.policeStoppedOnly)
        PoliceMotion(MotionConfig(horizonY = cfg.horizonY, driveSide = cfg.driveSide)) else null
    val shoulder: ShoulderCars? = if (cfg.shoulderCars)
        ShoulderCars(ShoulderConfig(cameraSpeed = cfg.shoulderCameraSpeed), horizonY = cfg.horizonY,
            hfovDeg = cfg.hfovDeg, camHeightM = cfg.camHeightM, driveSide = cfg.driveSide) else null

    private class Sample(val t: Double, val ego: String, val vehicles: Int, val ahead: Boolean)
    private class Emergency(val t: Double, val label: String, val bbox: BBox, val lit: Boolean)
    private class Work(val t: Double, val u: Double, val groundY: Double, val label: String)

    private val hist = ArrayDeque<Sample>()
    private var lastRed = -1e9
    private val emergency = ArrayDeque<Emergency>()
    private val construction = ArrayDeque<Work>()
    private var laneSide: String? = null
    private var laneSince = 0.0
    private var darkSince: Double? = null
    private val weatherEma = LinkedHashMap<String, Double>()
    private val weatherSince = HashMap<String, Double>()

    /** ego, flow, vehicles of the last frame. */
    @Volatile var state: Map<String, Any> = emptyMap()
        private set
    /** Scene labels returned for the last frame (status line). */
    @Volatile var activeLabels: List<String> = emptyList()
        private set

    val weather: Map<String, Double> get() = LinkedHashMap(weatherEma)

    /**
     * After [update]: police boxes of cars not read as standing still are relabelled so that no Police report
     * counts them (Waze's Police report is for a police car that stands). The crash rule has seen them as they were.
     */
    fun gatePolice(t: Double, img: RgbImage, dets: List<Detection>, hoodFrac: Double = 0.0): List<Detection> {
        val m = motion ?: return dets.toList()
        return m.update(t, img, dets, (state["ego"] as? String) ?: "moving", hoodFrac)
    }

    /** After [update]: scene_car_on_shoulder boxes ([ShoulderCars]), none when it is off. */
    fun shoulderCars(t: Double, img: RgbImage, dets: List<Detection>, hoodFrac: Double = 0.0): List<Detection> {
        val sc = shoulder ?: return emptyList()
        val gps = ego.gpsSpeed(t)
        val out = sc.update(t, img, dets, (state["ego"] as? String) ?: "moving", gps, gps != null, hoodFrac)
        val st = LinkedHashMap(state)
        st["fast_road"] = sc.highway
        sc.speed?.let { st["speed"] = Math.round(it * 10) / 10.0 }
        state = st
        return out
    }

    /** The box meets the road at least otherSideM across the centre line from us (left in right-hand traffic). */
    fun otherSide(b: BBox, aspect: Double): Boolean {
        val g = groundPoint(b, aspect) ?: return false
        return (if (cfg.driveSide == "right") -g.second else g.second) >= cfg.otherSideM
    }

    /** Extra detections for one frame: emergency_lights boxes and scene_* labels. */
    fun update(t: Double, img: RgbImage, dets: List<Detection>, weather: Map<String, Float>? = null): List<Detection> {
        val out = ArrayList<Detection>()
        val h = img.height
        val w = img.width
        ego.update(t, img, dets)
        val egoState = ego.state(t)
        val lit = lights.update(t, img, dets)
        out.addAll(lit)
        val vehicles = dets.filter { it.label in SceneLabels.VEHICLE && it.confidence >= 0.35f }
        val ahead = vehicles.any { abs(it.bbox.cx - 0.5) < 0.18 && it.bbox.w.toDouble() * it.bbox.h >= cfg.aheadMinArea }
        val signals = dets.filter { it.label == "traffic light" && it.confidence >= 0.35f && it.bbox.h.toDouble() * h >= cfg.signalMinPx }
        val colours = signals.map { Lamps.fractions(img, it.bbox) }
        if (colours.any { it.red > 0.02 || it.amber > 0.03 }) lastRed = t
        hist.addLast(Sample(t, egoState, vehicles.size, ahead))
        while (hist.isNotEmpty() && t - hist.first().t > max(cfg.heavyWindowS, cfg.standstillS) + 5) hist.removeFirst()
        state = mapOf("ego" to egoState, "flow" to Math.round(ego.flow * 10000) / 10000.0, "vehicles" to vehicles.size)

        if ("traffic" in cfg.enabled) out.addAll(traffic(t))
        if ("crash" in cfg.enabled) out.addAll(crash(t, dets, lit, egoState, w.toDouble() / h))
        if ("lanes" in cfg.enabled) out.addAll(lanes(t, dets, w.toDouble() / h))
        if ("signals" in cfg.enabled) out.addAll(signals(t, signals, colours))
        if ("weather" in cfg.enabled && weather != null) updateWeather(t, weather)
        if ("weather" in cfg.enabled) out.addAll(weatherLabels(t))
        if (cfg.policeOtherSide) out.addAll(policeSide(dets, w.toDouble() / h))
        activeLabels = out.filter { it.model == "scene" }.map { it.label }.distinct()
        return out
    }

    private fun traffic(t: Double): List<Detection> {
        if (t - lastRed < cfg.redVetoS) return emptyList()
        // the camera-only ego estimate tells stopped from moving but not slow from moving, so both rules lean
        // on stationary time (scene.py has the numbers)
        val span = hist.filter { t - it.t <= cfg.standstillS }
        if (span.isNotEmpty() && t - span[0].t >= cfg.standstillS * 0.95) {
            val n = span.size
            if (span.count { it.ego == "stationary" } >= cfg.standstillStationaryFrac * n &&
                span.count { it.ego == "moving" } <= cfg.standstillMovingMax * n &&
                span.count { it.ahead } >= 0.7 * n) {
                return listOf(whole("scene_traffic_standstill", 0.9))
            }
        }
        val win = hist.filter { t - it.t <= cfg.heavyWindowS }
        if (win.isNotEmpty() && t - win[0].t >= cfg.heavyWindowS * 0.95) {
            val slow = win.count { it.ego != "moving" }.toDouble() / win.size
            val stationary = win.count { it.ego == "stationary" }.toDouble() / win.size
            var stops = 0
            var prev: String? = null
            for (s in win) {
                if (s.ego == "stationary" && prev != null && prev != "stationary") stops++
                prev = s.ego
            }
            val meanVeh = win.sumOf { it.vehicles }.toDouble() / win.size
            if (slow >= cfg.heavySlowFrac && stationary >= cfg.heavyStationaryFrac &&
                stops >= cfg.heavyMinStops && meanVeh >= cfg.heavyMinVehicles) {
                return listOf(whole("scene_traffic_heavy", min(1.0, 0.5 + 0.5 * slow)))
            }
        }
        return emptyList()
    }

    private fun crash(t: Double, dets: List<Detection>, lights: List<Detection>, ego: String,
                      aspect: Double = 16.0 / 9.0): List<Detection> {
        // the vehicle classifier scores every class on a box; a box is one responder, of its best class
        val best = HashMap<BBox, Detection>()
        for (d in dets) {
            if (d.label !in SceneLabels.EMERGENCY) continue
            val prev = best[d.bbox]
            if (prev == null || d.confidence > prev.confidence) best[d.bbox] = d
        }
        for (d in best.values) {
            if (d.confidence >= 0.5f) {
                // lights count only on a responder: a work truck's beacon or a flickering tail light is not one
                val lit = lights.any { boxIou(d.bbox, it.bbox) > 0.5 }
                emergency.addLast(Emergency(t, d.label, d.bbox, lit))
            }
        }
        while (emergency.isNotEmpty() && t - emergency.first().t > cfg.crashWindowS) emergency.removeFirst()
        // responders still in view: the window keeps them for 10 s, the scene ends when they leave the frame
        if (emergency.isEmpty() || t - emergency.last().t > 1.0) return emptyList()
        if (emergency.last().t - emergency.first().t < cfg.crashHoldS) return emptyList()
        val heavy = emergency.count { it.label == "ambulance" || it.label == "fire_truck" }
        val lit = emergency.count { it.lit }
        // a kind counts when seen at least twice: one truck called police once and fire engine once is not two
        val kinds = emergency.groupingBy { it.label }.eachCount().filterValues { it >= 2 }.keys
        // responders on scene, not on their way. While we drive, we approach them: their boxes grow. While we
        // stand, they must stand too: an ambulance driving past a car waiting at a light is not a crash.
        val areas = emergency.map { it.bbox.w.toDouble() * it.bbox.h }
        val third = max(1, areas.size / 3)
        val approaching = areas.takeLast(third).average() >= 1.3 * areas.take(third).average()
        val standing = standing(emergency.map { it.bbox }, cfg.crashMaxShift)
        val onScene = (approaching && ego != "stationary") || (standing && ego != "moving")
        // an ambulance or fire engine with lights, or two kinds of responder; police alone is a traffic stop
        val responders = (heavy >= 3 && lit >= 2) || (kinds.size >= 2 && lit >= 2)
        if (!(responders && onScene)) return emptyList()
        val far = emergency.all { otherSide(it.bbox, aspect) }
        val conf = min(1.0, 0.55 + 0.05 * lit + 0.05 * heavy)
        return listOf(whole(if (far) "scene_crash_other_side" else "scene_crash", conf))
    }

    private fun policeSide(dets: List<Detection>, aspect: Double = 16.0 / 9.0): List<Detection> {
        val out = ArrayList<Detection>()
        for (d in dets) {
            if (d.label == "police_vehicle" && otherSide(d.bbox, aspect)) out.add(Detection("scene_police_other_side", d.confidence, d.bbox, "scene"))
        }
        return out
    }

    /** (metres ahead, metres to the right) where the box meets the road, from the flat-road camera model; null
     *  above the horizon. */
    fun groundPoint(b: BBox, aspect: Double): Pair<Double, Double>? {
        val dv = b.cy.toDouble() + b.h / 2.0 - cfg.horizonY
        if (dv <= 1e-3) return null
        val fy = 0.5 / kotlin.math.tan(Math.toRadians(cfg.hfovDeg) / 2) * aspect
        return Pair(fy * cfg.camHeightM / dv, (b.cx - 0.5) * cfg.camHeightM * aspect / dv)
    }

    /** Lateral offset in half lane widths (negative: left); null outside the distance range. */
    fun laneOffset(b: BBox, aspect: Double = 16.0 / 9.0): Double? {
        val g = groundPoint(b, aspect) ?: return null
        if (g.first < cfg.laneMinDistM || g.first > cfg.laneMaxDistM) return null
        return g.second / (cfg.laneWidthM / 2)
    }

    private fun lanes(t: Double, dets: List<Detection>, aspect: Double): List<Detection> {
        for (d in dets) {
            if (d.label in SceneLabels.CONSTRUCTION && d.label != "construction_sign" && d.confidence >= 0.45f) {
                val u = laneOffset(d.bbox, aspect) ?: continue
                if (abs(u) <= cfg.laneMaxOffset) construction.addLast(Work(t, u, d.bbox.cy + d.bbox.h / 2.0, d.label))
            }
        }
        while (construction.isNotEmpty() && t - construction.first().t > cfg.laneWindowS) construction.removeFirst()
        val now = construction.filter { t - it.t <= 0.5 }
        val us = now.map { it.u }.sorted()
        var side: String? = null
        if (us.size >= cfg.laneMinObjects && us.count { abs(it) <= cfg.laneReach } >= 2) {
            val mid = us[us.size / 2]
            side = if (mid < -cfg.laneSideU) "left" else if (mid > cfg.laneSideU) "right" else "center"
        }
        if (side != laneSide) { laneSide = side; laneSince = t }
        if (side == null || t - laneSince < cfg.laneHoldS) return emptyList()
        return listOf(whole("scene_lane_$side", min(1.0, 0.5 + 0.08 * now.size)))
    }

    private fun signals(t: Double, signals: List<Detection>, colours: List<LampFractions>): List<Detection> {
        val litAny = colours.any { it.red > 0.01 || it.amber > 0.01 || it.green > 0.01 || it.lit > 0.03 }
        val close = signals.filter { it.bbox.h >= 0.04f }
        if (close.size >= 2 && !litAny) {
            val since = darkSince ?: t.also { darkSince = it }
            if (t - since >= cfg.darkSignalS) return listOf(whole("scene_dark_signal", 0.6))
        } else {
            darkSince = null
        }
        return emptyList()
    }

    private fun updateWeather(t: Double, probs: Map<String, Float>) {
        val a = cfg.weatherEma
        for ((k, v) in probs) weatherEma[k] = (1 - a) * (weatherEma[k] ?: v.toDouble()) + a * v
        for ((cls, rule) in cfg.weatherRules) {
            if (ruleValue(cls) >= rule.threshold) weatherSince.getOrPut(cls) { t } else weatherSince.remove(cls)
        }
    }

    // a term none of whose classes any loaded model has is left out: without the road-surface model, Slippery
    // road rests on rain alone. A rule with no term left never fires.
    private fun ruleValue(key: String): Double {
        val terms = key.split('&').filter { part -> part.split('|').any { it in weatherEma } }
            .map { part -> part.split('|').sumOf { weatherEma[it] ?: 0.0 } }
        return terms.minOrNull() ?: 0.0
    }

    private fun weatherLabels(t: Double): List<Detection> {
        val out = ArrayList<Detection>()
        for ((cls, rule) in cfg.weatherRules) {
            val since = weatherSince[cls] ?: continue
            if (t - since >= rule.holdS) out.add(whole(rule.label, ruleValue(cls)))
        }
        return out
    }
}
