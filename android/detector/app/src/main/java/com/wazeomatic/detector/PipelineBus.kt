package com.wazeomatic.detector

import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper

/** Latest processed frame plus the numbers shown on screen. */
class FrameResult(
    val bitmap: Bitmap,
    val detections: List<Detection>,
    val latencyMs: Map<String, Double>,
    val fps: Double,
    val frameIndex: Int,
)

/**
 * Small in-process bridge between the foreground service (producer) and the activity
 * (consumer). Everything is delivered on the main thread.
 */
object PipelineBus {
    interface Listener {
        fun onFrame(r: FrameResult) {}
        fun onEvent(ev: HazardEvent, dryRun: Boolean) {}
        fun onStatus(status: String) {}
    }

    private val main = Handler(Looper.getMainLooper())
    private val listeners = mutableListOf<Listener>()

    @Volatile var status: String = "stopped"
        private set
    @Volatile var running: Boolean = false

    val recentEvents = ArrayDeque<Pair<HazardEvent, Boolean>>()

    fun addListener(l: Listener) = main.post { listeners.add(l) }
    fun removeListener(l: Listener) = main.post { listeners.remove(l) }

    fun postStatus(s: String) {
        status = s
        main.post { listeners.toList().forEach { it.onStatus(s) } }
    }

    fun postFrame(r: FrameResult) {
        main.post { listeners.toList().forEach { it.onFrame(r) } }
    }

    fun postEvent(ev: HazardEvent, dryRun: Boolean) {
        main.post {
            recentEvents.addFirst(ev to dryRun)
            while (recentEvents.size > 200) recentEvents.removeLast()
            listeners.toList().forEach { it.onEvent(ev, dryRun) }
        }
    }
}
