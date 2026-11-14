package com.wazeomatic.detector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random

/**
 * Mirrors tests/test_scene.py. Python builds BGR images; these are RGB, so a Python (255, 40, 0) bar is
 * RGB (0, 40, 255) here (blue) and (0, 0, 255) is RGB (255, 0, 0) (red).
 */
class SceneAnalyzerTest {
    private val car = BBox(0.5f, 0.5f, 0.3f, 0.3f)
    private val blue = intArrayOf(0, 40, 255)
    private val red = intArrayOf(255, 0, 0)

    private fun frameWithBar(rgb: IntArray? = null, h: Int = 360, w: Int = 640): RgbImage {
        val img = RgbImage.filled(w, h, 90, 90, 90)
        if (rgb != null) {           // a light bar across the top of the car box
            val y1 = ((0.5 - 0.15) * h).toInt()
            val x1 = ((0.5 - 0.15) * w).toInt()
            img.fill(x1, y1, x1 + (0.3 * w).toInt(), y1 + 12, rgb[0], rgb[1], rgb[2])
        }
        return img
    }

    private val textureBase: IntArray by lazy {
        // random 4 x 4 blocks over a 180 x 640 strip, as texture() in the Python test
        val h = 180
        val w2 = 640
        val rng = Random(0)
        val blocks = Array((h + 3) / 4) { IntArray((w2 + 3) / 4) { rng.nextInt(256) } }
        IntArray(h * w2) { i -> blocks[(i / w2) / 4][(i % w2) / 4] }
    }

    private fun texture(shift: Int, h: Int = 180, w: Int = 320): RgbImage {
        val base = textureBase
        val out = IntArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            val g = base[y * 640 + x + shift]
            out[y * w + x] = RgbImage.pack(g, g, g)
        }
        return RgbImage(w, h, out)
    }

    @Test
    fun everySceneLabelMapsToAHazard() {
        for ((label, key) in Schema.DEFAULT_CLASS_TO_HAZARD) assertTrue(label, key in Schema.HAZARD_TYPES)
        for (rule in defaultWeatherRules().values) assertTrue(rule.label, rule.label in Schema.DEFAULT_CLASS_TO_HAZARD)
        for (l in listOf("scene_traffic_standstill", "scene_traffic_heavy", "scene_crash", "scene_crash_other_side",
            "scene_police_other_side", "scene_lane_left", "scene_lane_center", "scene_lane_right", "scene_dark_signal",
            "police_vehicle")) {
            assertTrue(l, l in Schema.DEFAULT_CLASS_TO_HAZARD)
        }
    }

    @Test
    fun lampFractionsColours() {
        var f = Lamps.fractions(frameWithBar(blue), car, topFrac = 0.6)
        assertTrue(f.blue > 0.02)
        assertEquals(0.0, f.red, 0.0)
        f = Lamps.fractions(frameWithBar(red), car, topFrac = 0.6)
        assertTrue(f.red > 0.02)
        assertEquals(0.0, f.blue, 0.0)
    }

    @Test
    fun alternations() {
        assertEquals(5, Lamps.alternations(listOf(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), 0.5))
        assertEquals(0, Lamps.alternations(listOf(1.0, 1.0, 1.0, 1.0, 1.0), 0.1))
        assertEquals(1, Lamps.alternations(listOf(0.0, 0.0, 1.0, 1.0, 1.0, 1.0), 0.5))   // brake lights coming on once
    }

    @Test
    fun lightBarFlashingIsDetectedAndBrakeLightsAreNot() {
        val cars = listOf(Detection("car", 0.9f, car))
        val lights = EmergencyLights()
        val hits = ArrayList<Detection>()
        for (i in 0 until 20) hits += lights.update(i * 0.1, frameWithBar(if (i % 2 == 1) blue else red), cars)
        assertTrue(hits.isNotEmpty())
        assertEquals("emergency_lights", hits.last().label)
        val steady = EmergencyLights()
        val none = ArrayList<Detection>()
        for (i in 0 until 20) none += steady.update(i * 0.1, frameWithBar(red), cars)
        assertTrue(none.isEmpty())
    }

    @Test
    fun egoMotionStationaryVsMoving() {
        val still = EgoMotion()
        val moving = EgoMotion()
        for (i in 0 until 10) {
            still.update(i * 0.1, texture(0), emptyList())
            moving.update(i * 0.1, texture(i * 6), emptyList())
        }
        assertEquals("stationary", still.state(1.0))
        assertNotEquals("stationary", moving.state(1.0))
        val gps = EgoMotion()
        gps.setGpsSpeed(20.0, 0.0)
        assertEquals("moving", gps.state(1.0))
        assertEquals("stationary", gps.state(6.0))      // stale GPS: back to the image (no frames: flow 0)
    }

    @Test
    fun egoMasksVehicleBoxes() {
        // only the part under a car changes: masked, so the view reads as stationary
        val ego = EgoMotion()
        val box = listOf(Detection("car", 0.9f, BBox(0.5f, 0.4f, 0.5f, 0.6f)))
        for (i in 0 until 10) {
            val img = texture(0)
            val moved = texture(i * 6)
            for (y in (0.1 * 180).toInt() until (0.7 * 180).toInt()) for (x in (0.25 * 320).toInt() until (0.75 * 320).toInt()) {
                img.argb[y * 320 + x] = moved.argb[y * 320 + x]
            }
            ego.update(i * 0.1, img, box)
        }
        assertEquals("stationary", ego.state(1.0))
    }

    private fun runScene(
        cfg: SceneConfig, frames: Int, dets: (Double) -> List<Detection>,
        weather: ((Double) -> Map<String, Float>)? = null, dt: Double = 0.5, image: RgbImage? = null,
    ): List<Set<String>> {
        val sa = SceneAnalyzer(cfg)
        val img = image ?: texture(0)
        return (0 until frames).map { i ->
            val t = i * dt
            sa.update(t, img, dets(t), weather?.invoke(t)).map { it.label }.toSet()
        }
    }

    private fun queueAhead(@Suppress("UNUSED_PARAMETER") t: Double) = listOf(
        Detection("car", 0.9f, BBox(0.5f, 0.55f, 0.25f, 0.2f)),
        Detection("car", 0.8f, BBox(0.2f, 0.55f, 0.1f, 0.1f)),
        Detection("car", 0.8f, BBox(0.8f, 0.55f, 0.1f, 0.1f)),
    )

    @Test
    fun standstillAfterLongStop() {
        val labels = runScene(SceneConfig(enabled = setOf("traffic")), 170, ::queueAhead)
        assertFalse("scene_traffic_standstill" in labels[100])      // 50 s: could be a red light
        assertTrue("scene_traffic_standstill" in labels.last())    // 85 s stopped behind a car
    }

    @Test
    fun redLightVetoesStandstill() {
        val sa = SceneAnalyzer(SceneConfig(enabled = setOf("traffic")))
        val img = texture(0)
        val h = img.height
        val w = img.width
        img.fill((0.485 * w).toInt(), (0.14 * h).toInt(), (0.515 * w).toInt(), (0.18 * h).toInt(), 255, 0, 0)   // lit red lamp
        var last = emptySet<String>()
        for (i in 0 until 170) {
            val t = i * 0.5
            last = sa.update(t, img, queueAhead(t) + Detection("traffic light", 0.9f, BBox(0.5f, 0.2f, 0.04f, 0.14f)))
                .map { it.label }.toSet()
        }
        assertFalse("scene_traffic_standstill" in last)
    }

    @Test
    fun darkSignalsNeedTwoHeadsAndNoLitOne() {
        val twoDark = { _: Double ->
            listOf(Detection("traffic light", 0.9f, BBox(0.4f, 0.2f, 0.04f, 0.14f)),
                Detection("traffic light", 0.9f, BBox(0.6f, 0.2f, 0.04f, 0.14f)))
        }
        val grey = RgbImage.filled(320, 180, 60, 60, 60)            // unlit housings
        val cfg = SceneConfig(enabled = setOf("signals"))
        val labels = runScene(cfg, 12, twoDark, image = grey)
        assertTrue("scene_dark_signal" in labels.last())
        assertFalse("scene_dark_signal" in labels[3])
        assertTrue(runScene(cfg, 12, { t -> twoDark(t).take(1) }, image = grey).none { "scene_dark_signal" in it })
        val lit = grey.copy().fill((0.585 * 320).toInt(), (0.14 * 180).toInt(), (0.615 * 320).toInt(), (0.18 * 180).toInt(), 0, 200, 0)
        assertTrue(runScene(cfg, 12, twoDark, image = lit).none { "scene_dark_signal" in it })
    }

    @Test
    fun blockedLaneSide() {
        fun cones(vararg xs: Double) = { _: Double -> xs.map { Detection("cone", 0.8f, BBox(it.toFloat(), 0.7f, 0.03f, 0.06f)) } }
        val cfg = SceneConfig(enabled = setOf("lanes"))
        assertTrue("scene_lane_right" in runScene(cfg, 4, cones(0.55, 0.62, 0.7, 0.78)).last())
        assertTrue("scene_lane_left" in runScene(cfg, 4, cones(0.22, 0.3, 0.38, 0.46)).last())
        assertTrue(runScene(cfg, 4, cones(0.85, 0.9, 0.95)).all { s -> s.none { it.startsWith("scene_lane") } })
    }

    @Test
    fun blockedLaneIgnoresDistantCurbsideCones() {
        val far = { _: Double -> listOf(0.58, 0.6, 0.62, 0.64).map { Detection("cone", 0.8f, BBox(it.toFloat(), 0.52f, 0.01f, 0.03f)) } }
        val labels = runScene(SceneConfig(enabled = setOf("lanes")), 8, far)
        assertTrue(labels.all { s -> s.none { it.startsWith("scene_lane") } })
    }

    @Test
    fun groundPointInvertsTheCameraModel() {
        val sa = SceneAnalyzer(SceneConfig(camHeightM = 1.3, horizonY = 0.5, hfovDeg = 70.0))
        val aspect = 16.0 / 9.0
        val fy = 0.5 / kotlin.math.tan(Math.toRadians(35.0)) * aspect
        val gy = 0.5 + fy * 1.3 / 10.0
        val cx = 0.5 + 1.8 * (gy - 0.5) / (1.3 * aspect)
        val b = BBox(cx.toFloat(), (gy - 0.02).toFloat(), 0.02f, 0.04f)
        val g = sa.groundPoint(b, aspect)!!
        assertEquals(10.0, g.first, 1e-3)
        assertEquals(1.8, g.second, 1e-3)
        assertEquals(1.0, sa.laneOffset(b, aspect)!!, 1e-3)
        assertNull(sa.laneOffset(BBox(0.5f, 0.49f, 0.02f, 0.02f), aspect))
    }

    @Test
    fun blockedLaneNeedsOneSideHeld() {
        val cfg = SceneConfig(enabled = setOf("lanes"))
        val flip = { t: Double ->
            listOf(0.55, 0.62, 0.7).map { x ->
                Detection("cone", 0.8f, BBox((if (t.toInt() % 2 == 1) x else 1 - x).toFloat(), 0.7f, 0.03f, 0.06f))
            }
        }
        assertTrue(runScene(cfg, 24, flip, dt = 0.25).all { s -> s.none { it.startsWith("scene_lane") } })
        val one = { _: Double -> listOf(0.55, 0.62, 0.7).map { Detection("cone", 0.8f, BBox(it.toFloat(), 0.7f, 0.03f, 0.06f)) } }
        val labels = runScene(cfg, 4, one)
        assertTrue(labels[2].none { it.startsWith("scene_lane") })
        assertTrue("scene_lane_right" in labels[3])
    }

    @Test
    fun weatherNeedsHoldTime() {
        val cfg = SceneConfig(enabled = setOf("weather"))
        val labels = runScene(cfg, 60, { emptyList() }, { mapOf("clear" to 0.05f, "fog" to 0.95f) })
        assertFalse("scene_fog" in labels[20])
        assertTrue("scene_fog" in labels.last())
        val brief = { t: Double -> mapOf("fog" to if (t < 5) 0.95f else 0f, "clear" to if (t < 5) 0.05f else 1f) }
        assertTrue(runScene(cfg, 60, { emptyList() }, brief).none { "scene_fog" in it })
    }

    private class StubLights : EmergencyLights() {
        override fun update(t: Double, img: RgbImage, dets: List<Detection>): List<Detection> =
            dets.filter { it.label != "car" }.map { Detection("emergency_lights", 0.8f, it.bbox, "lights") }
    }

    @Test
    fun crashNeedsRespondersOnScene() {
        fun responders(t: Double): List<Detection> {
            val grow = (1 + t * 0.2).toFloat()
            return listOf(Detection("ambulance", 0.9f, BBox(0.6f, 0.5f, 0.1f * grow, 0.1f * grow), "vehicle"),
                Detection("police_vehicle", 0.9f, BBox(0.7f, 0.5f, 0.08f * grow, 0.08f * grow), "vehicle"))
        }
        val sa = SceneAnalyzer(SceneConfig(enabled = setOf("crash"))).apply { lights = StubLights() }
        var out = emptySet<String>()
        for (i in 0 until 12) out = sa.update(i * 0.5, texture(i * 6), responders(i * 0.5)).map { it.label }.toSet()
        assertTrue("scene_crash" in out)
        // police alone at the roadside is a traffic stop: Police, not Crash
        val sb = SceneAnalyzer(SceneConfig(enabled = setOf("crash"))).apply { lights = StubLights() }
        out = emptySet()
        for (i in 0 until 12) out = sb.update(i * 0.5, texture(i * 6), responders(i * 0.5).drop(1)).map { it.label }.toSet()
        assertFalse("scene_crash" in out)
    }

    @Test
    fun crashNotForRespondersPassingAWaitingCar() {
        class AllLit : EmergencyLights() {
            override fun update(t: Double, img: RgbImage, dets: List<Detection>): List<Detection> =
                dets.map { Detection("emergency_lights", 0.8f, it.bbox, "lights") }
        }
        fun run(boxes: (Double) -> List<Detection>): Boolean {
            val sa = SceneAnalyzer(SceneConfig(enabled = setOf("crash"))).apply { lights = AllLit() }
            var fired = false
            for (i in 0 until 16) {
                val t = i * 0.5
                fired = fired || sa.update(t, texture(0), boxes(t)).any { it.label == "scene_crash" }
            }
            return fired
        }
        // we wait at a light; an ambulance and a police car drive across the frame
        assertFalse(run { t -> listOf(
            Detection("ambulance", 0.9f, BBox((0.1 + 0.1 * t).toFloat(), 0.5f, 0.12f, 0.12f), "vehicle"),
            Detection("police_vehicle", 0.9f, BBox((0.05 + 0.1 * t).toFloat(), 0.5f, 0.1f, 0.1f), "vehicle")) })
        // the same two parked at the scene ahead of us
        assertTrue(run { t -> listOf(
            Detection("ambulance", 0.9f, BBox((0.6 + 0.002 * kotlin.math.sin(t)).toFloat(), 0.5f, 0.12f, 0.12f), "vehicle"),
            Detection("police_vehicle", 0.9f, BBox(0.75f, 0.5f, 0.1f, 0.1f), "vehicle")) })
    }

    @Test
    fun crashNeedsLightsOnTheRespondersAndEndsWithThem() {
        val amb = Detection("ambulance", 0.9f, BBox(0.6f, 0.5f, 0.12f, 0.12f), "vehicle")
        val pol = Detection("police_vehicle", 0.9f, BBox(0.75f, 0.5f, 0.1f, 0.1f), "vehicle")
        val car = Detection("car", 0.9f, BBox(0.3f, 0.5f, 0.1f, 0.1f), "coco")
        class LitOn(val label: String) : EmergencyLights() {
            override fun update(t: Double, img: RgbImage, dets: List<Detection>): List<Detection> =
                dets.filter { it.label == label }.map { Detection("emergency_lights", 0.8f, it.bbox, "lights") }
        }
        fun run(litLabel: String, present: (Double) -> Boolean = { true }): List<Boolean> {
            val sa = SceneAnalyzer(SceneConfig(enabled = setOf("crash"))).apply { lights = LitOn(litLabel) }
            return (0 until 24).map { i ->
                val t = i * 0.5
                sa.update(t, texture(0), if (present(t)) listOf(amb, pol, car) else listOf(car)).any { it.label == "scene_crash" }
            }
        }
        // parked responders, but the flashing light is on a work truck next to them
        assertFalse(run("car").any { it })
        // lights on the responders: a crash, which stops once they are out of view
        val fired = run("ambulance") { it < 7.0 }
        assertTrue(fired.take(14).any { it })
        assertFalse(fired.drop(16).any { it })
    }

    @Test
    fun slipperyNeedsRainAndWetRoad() {
        val both = { _: Double -> mapOf("rain" to 0.9f, "clear" to 0.1f, "surface_wet" to 0.7f, "surface_dry" to 0.3f) }
        val rainOnly = { _: Double -> mapOf("rain" to 0.9f, "clear" to 0.1f, "surface_dry" to 0.95f, "surface_wet" to 0.05f) }
        val wetOnly = { _: Double -> mapOf("clear" to 0.95f, "rain" to 0.05f, "surface_wet" to 0.9f) }
        val cfg = SceneConfig(enabled = setOf("weather"))
        assertTrue("scene_slippery" in runScene(cfg, 140, { emptyList() }, both).last())
        for (fn in listOf(rainOnly, wetOnly)) assertTrue(runScene(cfg, 140, { emptyList() }, fn).none { "scene_slippery" in it })
    }

    @Test
    fun policeOtherSideIsOptIn() {
        val far = { _: Double -> listOf(Detection("police_vehicle", 0.8f, BBox(0.1f, 0.5f, 0.1f, 0.1f), "vehicle")) }
        assertTrue(runScene(SceneConfig(enabled = emptySet()), 2, far).all { "scene_police_other_side" !in it })
        assertTrue("scene_police_other_side" in runScene(SceneConfig(enabled = emptySet(), policeOtherSide = true), 2, far).last())
        // left-hand traffic: the far side is the right
        assertTrue(runScene(SceneConfig(enabled = emptySet(), policeOtherSide = true, driveSide = "left"), 2, far)
            .all { "scene_police_other_side" !in it })
    }

    @Test
    fun featureListParsing() {
        assertEquals(setOf("traffic", "weather"), SceneConfig.parseFeatures(" Traffic, weather ,bogus"))
        assertTrue(SceneConfig.parseFeatures("none").isEmpty())
        assertEquals(SceneLabels.FEATURES.toSet(), SceneConfig().enabled)
    }
}
