package com.wazeomatic.detector

import org.junit.Assert.assertEquals
import org.junit.Test

class FrameThrottleTest {
    private fun taken(fpsCap: Double, cameraFps: Double, seconds: Double, busyNs: Long = 0L): Int {
        val t = FrameThrottle((1e9 / fpsCap).toLong())
        val step = 1e9 / cameraFps
        var busyUntil = 0L
        var n = 0
        var k = 0
        while (k * step < seconds * 1e9) {
            val now = (k * step).toLong()
            k++
            if (now < busyUntil) continue   // the analyzer is still on the last frame (KEEP_ONLY_LATEST drops this one)
            if (t.accept(now)) {
                n++
                busyUntil = now + busyNs
            }
        }
        return n
    }

    @Test
    fun keepsTheCapOnAThirtyFpsCamera() {
        // the old test (>= 1/6 s since the last frame) took every sixth frame: 5 a second
        assertEquals(60, taken(6.0, 30.0, 10.0))
    }

    @Test
    fun neverRunsAboveTheCap() {
        assertEquals(100, taken(10.0, 30.0, 10.0))
        assertEquals(30, taken(3.0, 30.0, 10.0))
    }

    @Test
    fun takesEveryFrameItCanWhenProcessingIsSlower() {
        // 200 ms a frame: the next camera frame after each one is taken (one every 200 ms)
        assertEquals(50, taken(6.0, 30.0, 10.0, busyNs = 190_000_000L))
    }
}
