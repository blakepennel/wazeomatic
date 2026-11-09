package com.wazeomatic.detector

import kotlin.math.abs
import kotlin.math.min

/**
 * Which way up the camera frames have to be turned, from how the phone is physically held.
 * Pure Kotlin (no Android classes) so it can be unit tested; the ROTATION_* values are the
 * same numbers as android.view.Surface.ROTATION_*.
 *
 * Input is the angle an OrientationEventListener reports: 0 = upright portrait, 90 = turned
 * clockwise (top edge to the right), 270 = turned counter-clockwise, -1 = flat / unknown.
 */
object MountOrientation {
    const val ROTATION_0 = 0      // portrait
    const val ROTATION_90 = 1     // landscape, top edge of the phone to the left
    const val ROTATION_180 = 2    // portrait upside down
    const val ROTATION_270 = 3    // landscape, top edge of the phone to the right

    const val UNKNOWN = -1
    /** Degrees past the 45 degree boundary before the rotation switches. */
    const val HYSTERESIS = 15

    val MODES = listOf("auto", "landscape", "landscape_reverse", "portrait")

    /** Target rotation used in auto mode until the first sensor reading (landscape mount). */
    const val DEFAULT_ROTATION = ROTATION_90

    /** Pinned rotation for a fixed `orientation` setting, null for "auto" (and anything unknown). */
    fun fixedRotation(setting: String): Int? = when (setting.trim().lowercase()) {
        "landscape" -> ROTATION_90
        "landscape_reverse" -> ROTATION_270
        "portrait" -> ROTATION_0
        else -> null
    }

    /** Nearest rotation for a device angle in degrees (any integer, wrapped to 0..359). */
    fun snap(degrees: Int): Int = when (((wrap(degrees) + 45) / 90) % 4) {
        0 -> ROTATION_0
        1 -> ROTATION_270
        2 -> ROTATION_180
        else -> ROTATION_90
    }

    /**
     * Rotation to use after a new sensor reading. Keeps [current] while the phone is flat
     * (negative angle) and until the angle is more than 45 + [hysteresis] degrees away from the
     * current rotation, so it does not flap when the phone sits near a diagonal.
     */
    fun next(current: Int, degrees: Int, hysteresis: Int = HYSTERESIS): Int {
        if (degrees < 0) return current
        val d = wrap(degrees)
        val diff = abs(d - centre(current)).let { min(it, 360 - it) }
        return if (diff <= 45 + hysteresis) current else snap(d)
    }

    /** Device angle at the middle of a rotation. */
    private fun centre(rotation: Int): Int = when (rotation) {
        ROTATION_90 -> 270
        ROTATION_180 -> 180
        ROTATION_270 -> 90
        else -> 0
    }

    private fun wrap(degrees: Int): Int = ((degrees % 360) + 360) % 360

    fun name(rotation: Int): String = when (rotation) {
        ROTATION_90 -> "landscape"
        ROTATION_180 -> "portrait_reverse"
        ROTATION_270 -> "landscape_reverse"
        else -> "portrait"
    }
}
