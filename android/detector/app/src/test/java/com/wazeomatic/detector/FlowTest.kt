package com.wazeomatic.detector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sin

class FlowTest {
    private fun scene(w: Int, h: Int, shiftX: Double, shiftY: Double, seed: Int): GreyImage {
        val r = java.util.Random(seed.toLong())
        val blobs = List(40) { doubleArrayOf(r.nextDouble() * w, r.nextDouble() * h, 3.0 + r.nextDouble() * 12, 40 + r.nextDouble() * 150) }
        val px = FloatArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            var v = 60.0 + 20 * sin((x + shiftX) * 0.07) * sin((y + shiftY) * 0.05)
            for (b in blobs) {
                val dx = x + shiftX - b[0]
                val dy = y + shiftY - b[1]
                v += b[3] * Math.exp(-(dx * dx + dy * dy) / (2 * b[2] * b[2]))
            }
            px[y * w + x] = v.coerceIn(0.0, 255.0).toFloat()
        }
        return GreyImage(w, h, px)
    }

    @Test
    fun fastLucasKanadeMatchesTheSampledOne() {
        val a = Flow.pyramid(scene(320, 180, 0.0, 0.0, 7), 4)
        val b = Flow.pyramid(scene(320, 180, 2.6, -1.3, 7), 4)
        val r = java.util.Random(3)
        var both = 0
        // points all over, edges included (those take the sampled path for the windows that stick out)
        val pts = List(300) { doubleArrayOf(r.nextDouble() * 319, r.nextDouble() * 179) } +
            listOf(doubleArrayOf(0.0, 0.0), doubleArrayOf(319.0, 179.0), doubleArrayOf(3.5, 90.25), doubleArrayOf(160.0, 1.0))
        for (win in listOf(15, 21)) for (p in pts) {
            val slow = Flow.lk(a, b, p[0], p[1], win, fast = false)
            val fast = Flow.lk(a, b, p[0], p[1], win, fast = true)
            assertEquals("lost on one path only at ${p.toList()}", slow == null, fast == null)
            if (slow != null && fast != null) {
                both++
                assertTrue("at ${p.toList()}: ${slow.toList()} vs ${fast.toList()}",
                    abs(slow[0] - fast[0]) < 1e-6 && abs(slow[1] - fast[1]) < 1e-6)
            }
        }
        assertTrue("too few tracked points ($both)", both > 300)
    }
}
