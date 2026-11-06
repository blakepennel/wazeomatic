package com.wazeomatic.detector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

class HoodEstimatorTest {
    private val w = 160
    private val h = 90

    /** Scrolling noise texture (road) with an optional static band at the bottom. */
    private fun frames(n: Int, dashFrac: Float, still: Boolean = false): Sequence<FloatArray> {
        val rnd = Random(0)
        val texture = FloatArray(h * 3 * w) { rnd.nextInt(0, 255).toFloat() }
        val dash = FloatArray(h * w) { rnd.nextInt(60, 120).toFloat() }
        val cut = (h * (1 - dashFrac)).toInt()
        return (0 until n).asSequence().map { i ->
            val off = if (still) 0 else (i * 7) % (h * 2)
            val img = texture.copyOfRange(off * w, (off + h) * w)
            if (dashFrac > 0f) for (y in cut until h) for (x in 0 until w) img[y * w + x] = dash[y * w + x]
            img
        }
    }

    @Test
    fun findsStaticDashboardBand() {
        val est = HoodEstimator()
        var frac = 0f
        for (f in frames(60, 0.25f)) frac = est.update(f)
        assertTrue("got $frac", frac in 0.20f..0.30f)
    }

    @Test
    fun clampsToMinimumAndMaximum() {
        val est = HoodEstimator()
        var frac = 0f
        for (f in frames(60, 0f)) frac = est.update(f)
        assertEquals(0.10f, frac, 1e-6f)
        val est2 = HoodEstimator(maxFrac = 0.30f)
        for (f in frames(60, 0.4f)) frac = est2.update(f)
        assertEquals(0.30f, frac, 1e-6f)
    }

    @Test
    fun waitsForMinFramesAndIgnoresStandstill() {
        val est = HoodEstimator()
        for (f in frames(20, 0.25f)) assertEquals(0.10f, est.update(f), 1e-6f)
        val est2 = HoodEstimator()
        var frac = 0f
        for (f in frames(60, 0.25f, still = true)) frac = est2.update(f)
        assertEquals(0.10f, frac, 1e-6f)
    }
}
