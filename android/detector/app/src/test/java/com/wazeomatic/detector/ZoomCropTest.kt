package com.wazeomatic.detector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ZoomCropTest {
    private fun det(label: String, conf: Float, cx: Float, cy: Float, w: Float, h: Float, model: String = "hazard") =
        Detection(label, conf, BBox(cx, cy, w, h), model)

    @Test
    fun mapsCropCoordinatesBackToFrame() {
        val crop = ZoomCrop(0.25f, 0.35f, 0.75f, 0.75f)
        // center of the crop -> center of the crop region in the frame
        val c = crop.toFrame(det("cone", 0.8f, 0.5f, 0.5f, 0.2f, 0.5f))
        assertEquals(0.5f, c.bbox.cx, 1e-6f)
        assertEquals(0.55f, c.bbox.cy, 1e-6f)
        assertEquals(0.1f, c.bbox.w, 1e-6f)
        assertEquals(0.2f, c.bbox.h, 1e-6f)
        // top-left corner of the crop -> (x1, y1)
        val tl = crop.toFrame(det("cone", 0.8f, 0f, 0f, 0.1f, 0.1f))
        assertEquals(0.25f, tl.bbox.cx, 1e-6f)
        assertEquals(0.35f, tl.bbox.cy, 1e-6f)
        // label, confidence and model untouched
        assertEquals("cone", c.label)
        assertEquals(0.8f, c.confidence, 0f)
        assertEquals("hazard", c.model)
    }

    @Test
    fun pixelRectRoundsAndRejectsTinyCrops() {
        val r = ZoomCrop.DEFAULT.pixelRect(1280, 720)
        assertEquals(CropRect(320, 252, 960, 540), r)
        assertNull(ZoomCrop(0.4f, 0.4f, 0.42f, 0.6f).pixelRect(1280, 720)) // 26 px wide
        assertNull(ZoomCrop.DEFAULT.pixelRect(60, 60))                       // 30x24 px
        // the fractions actually covered after rounding
        val eff = ZoomCrop.fromRect(r!!, 1280, 720)
        assertEquals(0.25f, eff.x1, 1e-6f)
        assertEquals(0.35f, eff.y1, 1e-6f)
        assertEquals(0.75f, eff.x2, 1e-6f)
        assertEquals(0.75f, eff.y2, 1e-6f)
    }

    @Test
    fun mergeKeepsHigherConfidenceDuplicate() {
        val full = det("cone", 0.55f, 0.50f, 0.55f, 0.04f, 0.06f)
        val zoomed = det("cone", 0.82f, 0.505f, 0.552f, 0.04f, 0.06f)   // same cone, seen bigger
        val other = det("pothole", 0.60f, 0.50f, 0.55f, 0.04f, 0.06f)  // other class, same spot
        val far = det("cone", 0.40f, 0.20f, 0.80f, 0.05f, 0.05f)        // separate cone
        val out = ZoomCrop.merge(listOf(full, other, far, zoomed), 0.5f)
        assertEquals(listOf(0.82f, 0.60f, 0.40f), out.map { it.confidence })
        assertEquals(listOf("cone", "pothole", "cone"), out.map { it.label })
        assertEquals(zoomed, out[0])
    }

    @Test
    fun mergePassesThroughSingletons() {
        assertTrue(ZoomCrop.merge(emptyList()).isEmpty())
        val one = det("dog", 0.5f, 0.1f, 0.1f, 0.1f, 0.1f)
        assertEquals(listOf(one), ZoomCrop.merge(listOf(one)))
    }

    @Test
    fun parseSetting() {
        assertNull(ZoomCrop.parse(""))
        assertNull(ZoomCrop.parse(null))
        assertNull(ZoomCrop.parse("false"))
        assertEquals(ZoomCrop.DEFAULT, ZoomCrop.parse("true"))
        assertEquals(ZoomCrop.DEFAULT, ZoomCrop.parse("TRUE"))
        assertEquals(ZoomCrop(0.1f, 0.2f, 0.9f, 0.8f), ZoomCrop.parse("0.1,0.2,0.9,0.8"))
        assertEquals(ZoomCrop(0.1f, 0.2f, 0.9f, 0.8f), ZoomCrop.parse(" 0.1, 0.2, 0.9, 0.8 "))
        // malformed -> off
        assertNull(ZoomCrop.parse("0.1,0.2,0.9"))
        assertNull(ZoomCrop.parse("a,b,c,d"))
        assertNull(ZoomCrop.parse("0.9,0.2,0.1,0.8"))   // x2 <= x1
        assertNull(ZoomCrop.parse("0.1,0.2,1.5,0.8"))   // out of range
        assertNull(ZoomCrop.parse("yes"))
    }

    @Test
    fun settingsExposeParsedCrop() {
        assertNull(Settings().zoomCropOrNull)
        assertEquals(ZoomCrop.DEFAULT, Settings(zoomCrop = "true").zoomCropOrNull)
        assertNotNull(Settings(zoomCrop = "0.2,0.3,0.8,0.7").zoomCropOrNull)
        assertNull(Settings(zoomCrop = "garbage").zoomCropOrNull)
    }
}
