package com.wazeomatic.detector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.floor
import kotlin.math.ln

/** Mirrors tests/test_motion.py: a textured world we drive into, with a police car that is part of it or not. */
class PoliceMotionTest {
    private val w = 640
    private val h = 360
    private val foeX = w / 2.0
    private val foeY = h / 2.0
    private val moving = PoliceMotion.POLICE + "_moving"
    private val pending = PoliceMotion.POLICE + "_pending"

    /** Smooth random texture: 4 x 4 px random blocks, bilinearly upsampled. */
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

    /** Frame pixel (x, y) shows world point m^-1 (x, y); m = (s, tx, ty) maps world to frame as s * p + t. */
    private fun render(world: FloatArray, s: Double, tx: Double, ty: Double): FloatArray {
        val img = GreyImage(w, h, world)
        return FloatArray(w * h) { i -> img.sample(((i % w) - tx) / s, ((i / w) - ty) / s).toFloat() }
    }

    private fun toRgb(g: FloatArray): RgbImage = RgbImage(w, h, IntArray(w * h) {
        val v = Math.rint(g[it].toDouble()).toInt().coerceIn(0, 255)
        RgbImage.pack(v, v, v)
    })

    private fun drive(carStatic: Boolean, frames: Int = 6, scale: Double = 1.03, turnX: Double = 0.0,
                      ego: String = "moving", cfg: MotionConfig = MotionConfig(horizonTol = 0.0)): List<String> {
        val world = texture(1, w, h)
        val car = texture(2, 60, 40)
        val box0 = BBox(0.75f, 0.68f, 60f / w, 40f / h)
        val x0 = (box0.cx * w - 30).toInt()
        val y0 = (box0.cy * h - 20).toInt()
        if (carStatic) for (y in 0 until 40) for (x in 0 until 60) world[(y0 + y) * w + x0 + x] = car[y * 60 + x]
        // the synthetic world zooms uniformly, a wall rather than a road: no zero row, so the horizon check is off
        val pm = PoliceMotion(cfg)
        val out = ArrayList<String>()
        var s = 1.0; var tx = 0.0; var ty = 0.0
        for (k in 0 until frames) {
            if (k > 0) {      // the last frame scaled about the focus of expansion, then shifted (turning)
                s *= scale
                tx = scale * tx + (1 - scale) * foeX + turnX
                ty = scale * ty + (1 - scale) * foeY
            }
            val g = render(world, s, tx, ty)
            val box = if (carStatic) {
                BBox(((s * box0.cx * w + tx) / w).toFloat(), ((s * box0.cy * h + ty) / h).toFloat(),
                    (box0.w * s).toFloat(), (box0.h * s).toFloat())
            } else {
                for (y in 0 until 40) for (x in 0 until 60) g[(y0 + y) * w + x0 + x] = car[y * 60 + x]
                box0
            }
            val res = pm.update(0.2 * k, toRgb(g), listOf(Detection(PoliceMotion.POLICE, 0.9f, box, "vehicle")), ego)
            out.add(res[0].label)
        }
        return out
    }

    @Test
    fun gateKeepsEveryBoxWhenNoPoliceIsInView() {
        // Pipeline clears its detection list and refills it from the gate's result: an aliased result emptied every
        // frame without a police car (all hazards were lost on the phone from 2026-11-22 to 2026-11-23)
        val img = RgbImage(64, 36, IntArray(64 * 36))
        val dets = arrayListOf(Detection("cone", 0.8f, BBox(0.5f, 0.7f, 0.05f, 0.08f), "hazard"),
            Detection("car", 0.9f, BBox(0.3f, 0.6f, 0.2f, 0.2f), "coco"))
        for (sa in listOf(SceneAnalyzer(SceneConfig()), SceneAnalyzer(SceneConfig(policeStoppedOnly = false)))) {
            val gated = sa.gatePolice(0.0, img, dets)
            assertTrue(gated !== dets)
            dets.clear()
            dets.addAll(gated)
            assertEquals(listOf("cone", "car"), dets.map { it.label })
        }
    }

    @Test
    fun roadPairRateRecoversUniformExpansion() {
        val rng = Random(0)
        val a = List(300) { doubleArrayOf(50 + 540 * rng.nextDouble(), 200 + 140 * rng.nextDouble()) }
        val b = a.map { doubleArrayOf((it[0] - foeX) * 1.05 + foeX + 3.0, (it[1] - foeY) * 1.05 + foeY - 1.0) }
        val fit = Flow.roadPairRate(a, b, h, 0.2)
        assertNotNull(fit)
        assertEquals(0.0, fit!![0], 1e-6)
        assertEquals(ln(1.05) / 0.2, fit[1], 1e-6)
    }

    @Test
    fun lucasKanadeFollowsAShift() {
        val world = texture(5, w, h)
        val a = GreyImage(w, h, world)
        val b = GreyImage(w, h, render(world, 1.0, 7.5, -3.0))
        val pts = Flow.corners(a, 50, 0.01, 6.0, null).filter { it[0] in 60.0..580.0 && it[1] in 40.0..320.0 }
        assertTrue(pts.size >= 20)
        val (pa, pb) = Flow.track(Flow.pyramid(a, 4), Flow.pyramid(b, 4), pts, 21)
        assertTrue(pa.size >= pts.size * 0.8)
        for (i in pa.indices) {
            assertEquals(7.5, pb[i][0] - pa[i][0], 0.2)
            assertEquals(-3.0, pb[i][1] - pa[i][1], 0.2)
        }
    }

    @Test
    fun uniformZoomIsNotARoad() = assertEquals(pending, drive(true, cfg = MotionConfig()).last())

    @Test
    fun pendingUntilEnoughMeasurements() = assertEquals(List(3) { pending }, drive(true, frames = 3))

    @Test
    fun staticCarPassesWhileWeDrive() = assertEquals(PoliceMotion.POLICE, drive(true).last())

    @Test
    fun carKeepingPaceIsMoving() = assertEquals(moving, drive(false).last())

    @Test
    fun rotationIsTakenOff() {
        assertEquals(PoliceMotion.POLICE, drive(true, turnX = 6.0).last())
        assertEquals(moving, drive(false, turnX = 6.0).last())
    }

    @Test
    fun whileWeStandAStillBoxPassesAndACrossingOneDoesNot() {
        val img = toRgb(texture(3, w, h))
        for ((dx, want) in listOf(0.0 to PoliceMotion.POLICE, 0.03 to moving)) {
            val pm = PoliceMotion()
            var last = ""
            for (k in 0 until 12) {
                val box = BBox((0.3 + dx * k).toFloat(), 0.6f, 0.15f, 0.12f)
                last = pm.update(0.2 * k, img, listOf(Detection(PoliceMotion.POLICE, 0.9f, box, "vehicle")), "stationary")[0].label
            }
            assertEquals(want, last)
        }
    }

    @Test
    fun otherLabelsPassUntouched() {
        val dets = listOf(Detection("car", 0.8f, BBox(0.5f, 0.6f, 0.1f, 0.1f)),
            Detection(PoliceMotion.POLICE, 0.2f, BBox(0.2f, 0.6f, 0.1f, 0.1f)))
        val out = PoliceMotion().update(0.0, toRgb(texture(4, w, h)), dets, "moving")
        assertEquals(listOf("car", PoliceMotion.POLICE), out.map { it.label })
    }
}
