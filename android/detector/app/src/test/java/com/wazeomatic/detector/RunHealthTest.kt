package com.wazeomatic.detector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class RunHealthTest {
    @Test
    fun stallOnlyAfterFiveSeconds() {
        assertNull(RunHealth.stallNote(nowMs = 10_000, lastFrameMs = 9_800))
        assertNull(RunHealth.stallNote(nowMs = 10_000, lastFrameMs = 5_000))
        assertEquals("stalled 5s", RunHealth.stallNote(nowMs = 10_001, lastFrameMs = 5_000))
        assertEquals("stalled 42s", RunHealth.stallNote(nowMs = 52_500, lastFrameMs = 10_000))
    }

    @Test
    fun portraitFramesGetAWarning() {
        assertNull(RunHealth.frameWarning(1280, 720))
        assertNull(RunHealth.frameWarning(640, 640))
        assertEquals(true, RunHealth.frameWarning(720, 1280)?.startsWith("portrait frames 720x1280"))
    }

    @Test
    fun batteryNote() {
        assertEquals("battery: unrestricted", RunHealth.batteryNote(true))
        assertEquals("battery: restricted", RunHealth.batteryNote(false))
    }
}
