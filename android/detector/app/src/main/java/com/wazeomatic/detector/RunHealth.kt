package com.wazeomatic.detector

/** Small pure helpers for the status line: frame heartbeat, frame shape, battery state. */
object RunHealth {
    /** No processed frame for longer than this while running counts as a stall. */
    const val STALL_AFTER_MS = 5000L

    /**
     * "stalled Ns" when the last processed frame (or the start of the source, before the first
     * frame) is more than [afterMs] old, else null. Both times come from one monotonic clock.
     */
    fun stallNote(nowMs: Long, lastFrameMs: Long, afterMs: Long = STALL_AFTER_MS): String? {
        val age = nowMs - lastFrameMs
        return if (age > afterMs) "stalled ${age / 1000}s" else null
    }

    /** Warning for upright frames; the models are trained on landscape dashcam pictures. */
    fun frameWarning(width: Int, height: Int): String? =
        if (height > width) "portrait frames ${width}x$height (model is trained on landscape)" else null

    fun batteryNote(ignoringOptimizations: Boolean): String =
        if (ignoringOptimizations) "battery: unrestricted" else "battery: restricted"
}
