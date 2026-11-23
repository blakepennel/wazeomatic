package com.wazeomatic.detector

import kotlin.math.min
import kotlin.math.tan

// Twin of wazeomatic/shoulder.py: Hazard > Car on shoulder, a vehicle standing beside the road on our side while
// we drive on a fast road. In town every kerb is lined with parked cars, so the rule runs only above 50 km/h:
// GPS speed when there is a fresh one, otherwise the camera's own speed from the road fit of StandingTracker.

/** Defaults are shoulder.ShoulderConfig's. */
data class ShoulderConfig(
    val requireHighway: Boolean = true,
    val highwayMps: Double = 13.9,
    val highwayHoldS: Double = 15.0,
    val highwayFrac: Double = 0.8,
    val cameraSpeed: Boolean = true,
    val cameraHighwayMps: Double = 12.0,
    val cameraSpeedEveryS: Double = 0.5,
    val minConf: Double = 0.35,
    val minBoxPx: Int = 28,
    val minDistM: Double = 8.0,
    val maxDistM: Double = 60.0,
    val minOffsetM: Double = 2.5,
    val maxOffsetM: Double = 8.0,
)

class ShoulderCars(
    val cfg: ShoulderConfig = ShoulderConfig(),
    val horizonY: Double = 0.5,
    val hfovDeg: Double = 70.0,
    val camHeightM: Double = 1.3,
    val driveSide: String = "right",
    val tracker: StandingTracker = StandingTracker(MotionConfig(horizonY = horizonY, driveSide = driveSide)),
) {
    private class Sample(val t: Double, val mps: Double, val gps: Boolean)
    private val speeds = ArrayDeque<Sample>()
    private var lastCamT = -1e9
    var highway = false
        private set
    /** Latest speed sample (m/s), for the status line. */
    var speed: Double? = null
        private set
    var lastCandidates: List<Detection> = emptyList()
        private set

    /** Focal length in frame heights. */
    private fun fy(aspect: Double) = 0.5 / tan(Math.toRadians(hfovDeg) / 2) * aspect

    /** (metres ahead, metres to the right) where the box meets a flat road; null at or above the horizon. */
    fun groundPoint(b: BBox, aspect: Double): DoubleArray? {
        val dv = b.cy + b.h / 2.0 - horizonY
        if (dv <= 1e-3) return null
        return doubleArrayOf(fy(aspect) * camHeightM / dv, (b.cx - 0.5) * camHeightM * aspect / dv)
    }

    fun isCandidate(d: Detection, w: Int, h: Int): Boolean {
        if (d.label !in VEHICLES || d.confidence < cfg.minConf || min(d.bbox.w * w, d.bbox.h * h) < cfg.minBoxPx) return false
        val g = groundPoint(d.bbox, w.toDouble() / h) ?: return false
        val side = if (driveSide == "right") g[1] else -g[1]
        return g[0] in cfg.minDistM..cfg.maxDistM && side in cfg.minOffsetM..cfg.maxOffsetM
    }

    /** Our speed from the tracker's last road fit (m/s): A times the focal length times the mount height. */
    fun cameraSpeed(aspect: Double): Double? = tracker.lastRoad?.let { it[0] * fy(aspect) * camHeightM }

    private fun updateSpeed(t: Double, gpsMps: Double?, gpsFresh: Boolean) {
        if (gpsFresh && gpsMps != null) {
            speeds.addLast(Sample(t, gpsMps, true))
            speed = gpsMps
        }
        while (speeds.isNotEmpty() && t - speeds.first().t > cfg.highwayHoldS) speeds.removeFirst()
        if (speeds.isEmpty() || t - speeds.first().t < cfg.highwayHoldS * 0.6) {
            highway = false
            return
        }
        val gps = speeds.filter { it.gps }
        val use = if (gps.isNotEmpty()) gps else speeds.toList()
        val thr = if (gps.isNotEmpty()) cfg.highwayMps else cfg.cameraHighwayMps
        highway = use.count { it.mps >= thr } >= cfg.highwayFrac * use.size
    }

    fun update(t: Double, img: RgbImage, dets: List<Detection>, egoState: String, gpsMps: Double? = null,
               gpsFresh: Boolean = false, hoodFrac: Double = 0.0): List<Detection> {
        val w = img.width
        val h = img.height
        val cand = dets.filter { isCandidate(it, w, h) }
        lastCandidates = cand
        val needSpeed = cfg.cameraSpeed && !gpsFresh && t - lastCamT >= cfg.cameraSpeedEveryS
        val before = tracker.lastRoad
        // off the fast road only the speed is measured: no car LK pass for every parked car in town
        val idle = cfg.requireHighway && !highway
        val targets = if (idle) emptyList() else cand
        val tracks = tracker.states(t, img, dets, targets, egoState, hoodFrac, forceRoad = needSpeed)
        if (needSpeed && tracker.lastRoad != null && tracker.lastRoad !== before) {
            lastCamT = t
            cameraSpeed(w.toDouble() / h)?.let {
                speeds.addLast(Sample(t, it, false))
                speed = it
            }
        }
        updateSpeed(t, gpsMps, gpsFresh)
        if (idle || (cfg.requireHighway && !highway)) return emptyList()
        return targets.zip(tracks).filter { it.second.state == "stopped" }
            .map { Detection(LABEL, it.first.confidence, it.first.bbox, "scene") }
    }

    companion object {
        const val LABEL = "scene_car_on_shoulder"
        val VEHICLES = setOf("car", "truck", "bus")
    }
}
