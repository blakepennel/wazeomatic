package com.wazeomatic.detector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ClipsTest {
    @Test
    fun safeNames() {
        assertEquals("drive_01.mp4", Clips.safeName("drive 01.mp4"))
        assertEquals("evil.mp4", Clips.safeName("../../evil.mp4"))
        assertEquals("clip.mp4", Clips.safeName(null))
        assertEquals("clip.mp4", Clips.safeName("   "))
        assertEquals("notes.txt.mp4", Clips.safeName("notes.txt"))
        assertEquals("dash_cam_.MOV", Clips.safeName("dash:cam?.MOV"))
        assertTrue(Clips.safeName("a".repeat(400) + ".mp4").length <= 120)
    }

    @Test
    fun uniqueNames() {
        val taken = setOf("a.mp4", "a-2.mp4")
        assertEquals("b.mp4", Clips.uniqueName("b.mp4") { it in taken })
        assertEquals("a-3.mp4", Clips.uniqueName("a.mp4") { it in taken })
    }

    @Test
    fun videoNames() {
        assertTrue(Clips.isVideoName("x.MP4"))
        assertTrue(Clips.isVideoName("x.mkv"))
        assertFalse(Clips.isVideoName("x.mp4.part"))
        assertFalse(Clips.isVideoName("hazard.onnx"))
    }

    @Test
    fun formats() {
        assertEquals("512 B", Clips.formatSize(512))
        assertEquals("1.5 MB", Clips.formatSize(1536L * 1024))
        assertEquals("1:05", Clips.formatDuration(65_000))
        assertEquals("1:01:01", Clips.formatDuration(3_661_000))
    }

    @Test
    fun reportTitles() {
        assertEquals("Police", ReportFormat.title(HazardEvent("POLICE", 0.9)))
        assertEquals("Hazard › Pothole", ReportFormat.title(HazardEvent("HAZARD_ON_ROAD_POTHOLE", 0.7)))
        assertEquals("90%  ·  police vehicle", ReportFormat.detail(HazardEvent("POLICE", 0.9, label = "police_vehicle"), withTime = false))
    }
}
