package com.wazeomatic.detector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SettingsSpecTest {
    private val specs = SettingsSpec.ALL

    @Test
    fun everyLaunchKeyHasAControl() {
        val shown = specs.map { it.key }.toSet()
        val missing = Settings.KNOWN_KEYS - shown
        assertTrue("no control for $missing", missing.isEmpty())
    }

    @Test
    fun everyControlIsALaunchKey() {
        val unknown = specs.map { it.key }.filter { it !in Settings.KNOWN_KEYS && it !in SettingsSpec.OPTIONAL }
        assertTrue("not in Settings.KNOWN_KEYS: $unknown", unknown.isEmpty())
        assertEquals("keys appear once", specs.size, specs.map { it.key }.toSet().size)
    }

    @Test
    fun optionalKeysOnlyShowOnceKnown() {
        val without = SettingsSpec.visible(Settings.KNOWN_KEYS - "realtime").flatMap { it.items }.map { it.key }
        assertFalse("realtime" in without)
        val with = SettingsSpec.visible(Settings.KNOWN_KEYS + "realtime").flatMap { it.items }.map { it.key }
        assertTrue("realtime" in with)
    }

    @Test
    fun dependsOnNamesAToggle() {
        val toggles = specs.filterIsInstance<SettingSpec.Toggle>().map { it.key }.toSet()
        for (s in specs) s.dependsOn?.let { assertTrue("${s.key} depends on $it", it in toggles) }
    }

    @Test
    fun rangesAreSaneAndHoldTheDefaults() {
        val d = Settings()
        val defaults = mapOf(
            "frame_step" to d.frameStep.toFloat(), "max_fps" to d.maxFps, "conf" to d.confThreshold, "min_hits" to d.minHits.toFloat(),
            "window" to d.window.toFloat(), "cooldown" to d.cooldownS, "global_cooldown" to d.globalCooldownS, "det_conf" to d.detConf,
            "iou" to d.iou, "vehicle_crops" to d.vehicleCrops.toFloat(), "hood_frac" to d.hoodFrac, "track_iou" to d.trackIou,
            "track_grow" to d.trackGrow, "ort_threads" to d.ortThreads.toFloat(), "zoom_every" to d.zoomEvery.toFloat(),
            "max_det" to d.maxDet.toFloat(), "max_per_class" to d.maxPerClass.toFloat(),
        )
        val ranges = specs.filterIsInstance<SettingSpec.Range>()
        assertEquals(defaults.keys, ranges.map { it.key }.toSet())
        for (r in ranges) {
            assertTrue(r.key, r.from < r.to && r.step > 0f)
            val v = defaults.getValue(r.key)
            assertTrue("${r.key} default $v outside ${r.from}..${r.to}", v in r.from..r.to)
            assertEquals("${r.key} default $v is off the step grid", v, r.snap(v), 1e-4f)
        }
    }

    @Test
    fun rangeSnapFormatEncode() {
        val conf = specs.first { it.key == "conf" } as SettingSpec.Range
        assertEquals(0.40f, conf.snap(0.41f), 1e-6f)
        assertEquals(0.95f, conf.snap(5f), 1e-6f)
        assertEquals("0.40", conf.format(0.4f))
        assertEquals("0.40", conf.encode(0.4f))
        val threads = specs.first { it.key == "ort_threads" } as SettingSpec.Range
        assertEquals("auto", threads.format(0f))
        assertEquals("3", threads.encode(3f))
        val hood = specs.first { it.key == "hood_frac" } as SettingSpec.Range
        assertEquals("10 %", hood.format(0.10f))
        assertEquals("0.10", hood.encode(0.1f))
    }

    @Test
    fun choicesHoldTheDefaults() {
        val d = Settings()
        val defaults = mapOf("source" to d.source, "orientation" to d.orientation, "zoom_crop" to d.zoomCrop, "fusion" to d.fusion,
            "drive_side" to d.driveSide, "ort_backend" to d.ortBackend)
        for (c in specs.filterIsInstance<SettingSpec.Choice>()) {
            assertTrue("${c.key} has no option for its default", c.options.any { it.first == defaults.getValue(c.key) })
        }
        val backend = specs.first { it.key == "ort_backend" } as SettingSpec.Choice
        assertEquals("Custom: hazard=nnapi,coco=xnnpack", backend.labelOf("hazard=nnapi,coco=xnnpack"))
        val mount = specs.first { it.key == "orientation" } as SettingSpec.Choice
        assertEquals(MountOrientation.MODES.toSet(), mount.options.map { it.first }.toSet())
    }

    @Test
    fun sceneChipsRoundTrip() {
        val m = specs.first { it.key == "scene_features" } as SettingSpec.Multi
        assertEquals(SceneLabels.FEATURES.toSet(), m.options.map { it.first }.toSet())
        assertEquals(SceneLabels.FEATURES.toSet(), m.parse(Settings().sceneFeatures))
        assertEquals("none", m.encode(emptySet()))
        assertTrue(m.parse("none").isEmpty())
        assertEquals("traffic,weather", m.encode(setOf("weather", "traffic")))
    }

    @Test
    fun geometryEncodesWhatSettingsParses() {
        val g = specs.first { it.key == "camera_geometry" } as SettingSpec.Geometry
        val s = g.encode(listOf(70f, 1.3f, 0.5f))
        assertEquals("70,1.30,0.50", s)
        assertNotNull(Settings.parseGeometry(s))
        for (p in g.parts) assertNotNull(Settings.parseGeometry(g.encode(g.parts.map { if (it === p) it.from else it.to })))
    }

    @Test
    fun textValidation() {
        val port = specs.first { it.key == "tcp_port" } as SettingSpec.Text
        assertNull(port.validate("8765"))
        assertNotNull(port.validate("80"))
        assertNotNull(port.validate("abc"))
        val floors = specs.first { it.key == "report_class_conf" } as SettingSpec.Text
        assertNull(floors.validate(Settings().reportClassConf))
        assertNull(floors.validate(""))
        assertNotNull(floors.validate("pothole"))
    }
}
