package com.wazeomatic.detector

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class MountOrientationTest {
    private val r0 = MountOrientation.ROTATION_0
    private val r90 = MountOrientation.ROTATION_90
    private val r180 = MountOrientation.ROTATION_180
    private val r270 = MountOrientation.ROTATION_270

    @Test
    fun snapsToNearestQuarterTurn() {
        assertEquals(r0, MountOrientation.snap(0))
        assertEquals(r0, MountOrientation.snap(44))
        assertEquals(r0, MountOrientation.snap(359))
        assertEquals(r270, MountOrientation.snap(90))
        assertEquals(r180, MountOrientation.snap(180))
        assertEquals(r90, MountOrientation.snap(270))
        assertEquals(r90, MountOrientation.snap(-90))
    }

    @Test
    fun rotationValuesMatchAndroidSurfaceConstants() {
        assertEquals(android.view.Surface.ROTATION_0, r0)
        assertEquals(android.view.Surface.ROTATION_90, r90)
        assertEquals(android.view.Surface.ROTATION_180, r180)
        assertEquals(android.view.Surface.ROTATION_270, r270)
    }

    @Test
    fun followsThePhoneThroughAFullTurn() {
        var r = MountOrientation.DEFAULT_ROTATION
        assertEquals(r90, r)
        r = MountOrientation.next(r, 0); assertEquals(r0, r)
        r = MountOrientation.next(r, 90); assertEquals(r270, r)
        r = MountOrientation.next(r, 180); assertEquals(r180, r)
        r = MountOrientation.next(r, 270); assertEquals(r90, r)
    }

    @Test
    fun flatOrUnknownKeepsTheLastRotation() {
        for (r in listOf(r0, r90, r180, r270)) assertEquals(r, MountOrientation.next(r, MountOrientation.UNKNOWN))
    }

    @Test
    fun doesNotFlapNearTheDiagonal() {
        // landscape (270 degrees): the boundary to portrait is at 315, the switch only above 330
        var r = r90
        for (deg in listOf(310, 318, 312, 320, 330, 316)) {
            r = MountOrientation.next(r, deg)
            assertEquals(r90, r)
        }
        r = MountOrientation.next(r, 331)
        assertEquals(r0, r)
        // and it does not fall straight back: portrait holds down to 300
        for (deg in listOf(318, 312, 300)) {
            r = MountOrientation.next(r, deg)
            assertEquals(r0, r)
        }
        assertEquals(r90, MountOrientation.next(r, 299))
    }

    @Test
    fun hysteresisWrapsAroundZero() {
        assertEquals(r0, MountOrientation.next(r0, 55))
        assertEquals(r0, MountOrientation.next(r0, 305))
        assertEquals(r270, MountOrientation.next(r0, 61))
    }

    @Test
    fun fixedSettingsPinTheRotation() {
        assertNull(MountOrientation.fixedRotation("auto"))
        assertNull(MountOrientation.fixedRotation(""))
        assertNull(MountOrientation.fixedRotation("sideways"))
        assertEquals(r90, MountOrientation.fixedRotation("landscape"))
        assertEquals(r270, MountOrientation.fixedRotation("Landscape_Reverse"))
        assertEquals(r0, MountOrientation.fixedRotation("portrait"))
    }

    @Test
    fun newSettingsDefaults() {
        assertEquals("auto", Settings().orientation)
        assertEquals(false, Settings().zoomCoco)
    }
}
