package com.wazeomatic.detector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.floor
import kotlin.math.tan

/** Mirrors tests/test_shoulder.py. */
class ShoulderCarsTest {
    private val w = 640
    private val h = 360
    private val aspect = w.toDouble() / h
    private val fy = 0.5 / tan(Math.toRadians(35.0)) * aspect

    private fun car(b: BBox, conf: Float = 0.8f, label: String = "car") = Detection(label, conf, b, "coco")

    /** A box [dist] m ahead whose ground point lies [lateral] m to the right (mount height 1.3 m). */
    private fun at(lateral: Double, dist: Double = 20.0, bw: Float = 0.08f, bh: Float = 0.08f): BBox {
        val dv = fy * 1.3 / dist
        return BBox((0.5 + lateral * dv / (1.3 * aspect)).toFloat(), (0.5 + dv - bh / 2).toFloat(), bw, bh)
    }

    @Test
    fun groundPointAndCandidates() {
        val sc = ShoulderCars(horizonY = 0.5, hfovDeg = 70.0, camHeightM = 1.3)
        val g = sc.groundPoint(at(4.0), aspect)!!
        assertEquals(20.0, g[0], 1e-3)
        assertEquals(4.0, g[1], 1e-3)
        assertTrue(sc.isCandidate(car(at(4.0)), w, h))                  // beside the road on the right
        assertFalse(sc.isCandidate(car(at(0.0)), w, h))                 // ahead in our lane
        assertFalse(sc.isCandidate(car(at(-4.0)), w, h))                // the other side
        assertFalse(sc.isCandidate(car(at(12.0)), w, h))                // too far out
        assertFalse(sc.isCandidate(car(at(4.0), conf = 0.2f), w, h))
        assertFalse(sc.isCandidate(car(at(4.0, bw = 0.02f, bh = 0.02f)), w, h))
        assertFalse(sc.isCandidate(car(at(4.0), label = "person"), w, h))
        val left = ShoulderCars(horizonY = 0.5, hfovDeg = 70.0, camHeightM = 1.3, driveSide = "left")
        assertTrue(left.isCandidate(car(at(-4.0)), w, h))
        assertFalse(left.isCandidate(car(at(4.0)), w, h))
    }

    @Test
    fun cameraSpeedFromTheRoadFit() {
        val sc = ShoulderCars(horizonY = 0.5, hfovDeg = 70.0, camHeightM = 1.3)
        assertNull(sc.cameraSpeed(aspect))
        sc.tracker.lastRoad = doubleArrayOf(20.0 / (fy * 1.3), -10.0, 0.0, 0.0)
        assertEquals(20.0, sc.cameraSpeed(aspect)!!, 1e-9)
    }

    private fun gate(speeds: List<Double?>, cfg: ShoulderConfig = ShoulderConfig()): ShoulderCars {
        val sc = ShoulderCars(cfg)
        val img = RgbImage(w, h, IntArray(w * h))
        speeds.forEachIndexed { k, v -> sc.update(k * 0.5, img, emptyList(), "moving", v, v != null) }
        return sc
    }

    @Test
    fun fastRoadNeedsSustainedGpsSpeed() {
        assertTrue(gate(List(32) { 20.0 }).highway)                      // 16 s at 72 km/h
        assertFalse(gate(List(10) { 20.0 }).highway)                     // 5 s is too short
        assertFalse(gate(List(32) { 10.0 }).highway)                     // 36 km/h: town
        assertFalse(gate(List(20) { 20.0 } + List(12) { 8.0 }).highway)  // slowing down for a junction
        assertTrue(gate(List(28) { 20.0 } + List(4) { 12.0 }).highway)   // one slow patch is fine
        assertFalse(gate(List(40) { null }, ShoulderConfig(cameraSpeed = false)).highway)
    }

    // ---------------------------------------------------------------- synthetic drive, as in PoliceMotionTest
    private fun texture(seed: Long, tw: Int, th: Int): FloatArray {
        val rng = Random(seed)
        val bw = tw / 4 + 2
        val bh = th / 4 + 2
        val blocks = FloatArray(bw * bh) { rng.nextInt(256).toFloat() }
        return FloatArray(tw * th) { i ->
            val x = (i % tw) / 4.0
            val y = (i / tw) / 4.0
            val x0 = floor(x).toInt(); val y0 = floor(y).toInt()
            val fx = x - x0; val fy = y - y0
            val a = blocks[y0 * bw + x0] * (1 - fx) + blocks[y0 * bw + x0 + 1] * fx
            val b = blocks[(y0 + 1) * bw + x0] * (1 - fx) + blocks[(y0 + 1) * bw + x0 + 1] * fx
            (a * (1 - fy) + b * fy).toFloat()
        }
    }

    private fun render(world: FloatArray, s: Double, tx: Double, ty: Double): FloatArray {
        val img = GreyImage(w, h, world)
        return FloatArray(w * h) { i -> img.sample(((i % w) - tx) / s, ((i / w) - ty) / s).toFloat() }
    }

    private fun toRgb(g: FloatArray): RgbImage = RgbImage(w, h, IntArray(w * h) {
        val v = Math.rint(g[it].toDouble()).toInt().coerceIn(0, 255)
        RgbImage.pack(v, v, v)
    })

    // the synthetic world zooms uniformly, a wall rather than a road, so the tracker's horizon check is off
    private fun shoulder(cfg: ShoulderConfig) = ShoulderCars(cfg, horizonY = 0.5, hfovDeg = 70.0, camHeightM = 1.5,
        tracker = StandingTracker(MotionConfig(horizonTol = 0.0)))

    private fun drive(sc: ShoulderCars, carStatic: Boolean, frames: Int = 8, gps: Double? = null): List<List<Detection>> {
        val world = texture(1, w, h)
        val patch = texture(2, 60, 40)
        val box0 = BBox(0.69f, 0.58f, 60f / w, 40f / h)       // 14 m ahead, 3.7 m to the right
        val x0 = (box0.cx * w - 30).toInt()
        val y0 = (box0.cy * h - 20).toInt()
        if (carStatic) for (y in 0 until 40) for (x in 0 until 60) world[(y0 + y) * w + x0 + x] = patch[y * 60 + x]
        val out = ArrayList<List<Detection>>()
        var s = 1.0; var tx = 0.0; var ty = 0.0
        val scale = 1.03
        for (k in 0 until frames) {
            if (k > 0) {
                s *= scale
                tx = scale * tx + (1 - scale) * w / 2.0
                ty = scale * ty + (1 - scale) * h / 2.0
            }
            val g = render(world, s, tx, ty)
            val box = if (carStatic) {
                BBox(((s * box0.cx * w + tx) / w).toFloat(), ((s * box0.cy * h + ty) / h).toFloat(),
                    (box0.w * s).toFloat(), (box0.h * s).toFloat())
            } else {
                for (y in 0 until 40) for (x in 0 until 60) g[(y0 + y) * w + x0 + x] = patch[y * 60 + x]
                box0
            }
            out.add(sc.update(0.2 * k, toRgb(g), listOf(car(box)), "moving", gps, gps != null))
        }
        return out
    }

    @Test
    fun standingCarBesideTheRoadIsMarked() {
        val outs = drive(shoulder(ShoulderConfig(requireHighway = false)), carStatic = true)
        assertTrue(outs[0].isEmpty() && outs[1].isEmpty())               // pending until three measurements
        assertTrue(outs.drop(4).all { it.size == 1 && it[0].label == ShoulderCars.LABEL })
    }

    @Test
    fun carDrivingBesideUsIsNotMarked() {
        assertTrue(drive(shoulder(ShoulderConfig(requireHighway = false)), carStatic = false).all { it.isEmpty() })
    }

    @Test
    fun nothingInTown() {
        // the same standing car at 30 km/h on GPS: every kerb in town looks like this
        assertTrue(drive(shoulder(ShoulderConfig()), carStatic = true, gps = 8.0).all { it.isEmpty() })
    }

    @Test
    fun otherSideFromTheGroundPoint() {
        val sa = SceneAnalyzer(SceneConfig(horizonY = 0.5, hfovDeg = 70.0, camHeightM = 1.3))
        assertTrue(sa.otherSide(at(-5.0, 25.0), aspect))                // oncoming lane, 25 m ahead
        assertFalse(sa.otherSide(at(-1.0, 25.0), aspect))               // ahead in our lane
        assertFalse(sa.otherSide(at(4.0, 25.0), aspect))                // our side
        assertFalse(sa.otherSide(BBox(0.1f, 0.3f, 0.05f, 0.05f), aspect))  // above the horizon
        val lht = SceneAnalyzer(SceneConfig(horizonY = 0.5, hfovDeg = 70.0, camHeightM = 1.3, driveSide = "left"))
        assertTrue(lht.otherSide(at(5.0, 25.0), aspect))
        assertFalse(lht.otherSide(at(-5.0, 25.0), aspect))
    }

    @Test
    fun sceneAnalyzerRunsItOnlyWhenAsked() {
        val img = RgbImage(w, h, IntArray(w * h))
        val off = SceneAnalyzer(SceneConfig())
        assertNull(off.shoulder)
        assertTrue(off.shoulderCars(0.0, img, emptyList()).isEmpty())
        val on = SceneAnalyzer(SceneConfig(shoulderCars = true))
        for (k in 0 until 32) {
            on.ego.setGpsSpeed(20.0, k * 0.5)
            on.update(k * 0.5, img, emptyList())
            on.shoulderCars(k * 0.5, img, emptyList())
        }
        assertEquals(true, on.state["fast_road"])
        assertEquals(20.0, on.state["speed"])
    }
}
