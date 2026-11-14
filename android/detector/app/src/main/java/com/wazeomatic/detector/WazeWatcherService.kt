package com.wazeomatic.detector

import android.accessibilityservice.AccessibilityService
import android.content.ComponentName
import android.content.Context
import android.graphics.Rect
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo

/**
 * Reads what Waze (or mock Waze) shows on screen, so the detector does not send a report Waze already
 * has. Strictly read only: the service config (res/xml/waze_watcher.xml) asks for window content and
 * nothing else, no gestures and no key events, and this class never performs an action on a node.
 *
 * On a window change of a Waze package it walks the window tree (at most [MIN_READ_INTERVAL_MS] apart,
 * with one trailing read so the last state of a burst of changes is seen), collects the visible texts
 * with their bounds, parses them with [WazeText] and merges them into [WazeAlerts.shared], which the
 * pipeline's ReportManager consults before it sends an event. New alerts go to logcat as
 * `WAZE_ALERT {json}` under tag [TAG].
 *
 * The user switches it on under Settings > Accessibility; the "waze_reader" setting turns the check off
 * without touching the system setting.
 */
class WazeWatcherService : AccessibilityService() {
    private val handler = Handler(Looper.getMainLooper())
    private var lastReadAt = 0L
    private var readScheduled = false
    private var enabledCache = true
    private var enabledCheckedAt = 0L
    private val readRunnable = Runnable {
        readScheduled = false
        readNow()
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        connected = true
        Log.i(TAG, "Waze reader connected (read only)")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        val pkg = event?.packageName?.toString() ?: return
        if (pkg !in WazeText.WAZE_PACKAGES) return
        if (!readerEnabled()) return
        val now = SystemClock.uptimeMillis()
        val wait = lastReadAt + MIN_READ_INTERVAL_MS - now
        if (wait <= 0 && !readScheduled) {
            readNow()
        } else if (!readScheduled) {
            readScheduled = true
            handler.postDelayed(readRunnable, maxOf(wait, 0L))
        }
    }

    /** The setting is re-read once every few seconds, not on every event. */
    private fun readerEnabled(): Boolean {
        val now = SystemClock.uptimeMillis()
        if (now - enabledCheckedAt > 5000L) {
            enabledCheckedAt = now
            enabledCache = try { Settings.load(this).wazeReader } catch (e: Exception) { true }
        }
        return enabledCache
    }

    private fun readNow() {
        lastReadAt = SystemClock.uptimeMillis()
        val root = try { rootInActiveWindow } catch (e: Exception) { null } ?: return
        val pkg = root.packageName?.toString() ?: ""
        if (pkg !in WazeText.WAZE_PACKAGES) return
        val nodes = ArrayList<TextNode>()
        collect(root, nodes, 0)
        val screen = WazeText.parseNodes(pkg, nodes)
        reads++
        for (a in WazeAlerts.shared.update(screen)) Log.i(TAG, "WAZE_ALERT ${a.toJson()}")
    }

    /** Depth-first walk; only reads text, content description and bounds. */
    private fun collect(node: AccessibilityNodeInfo, out: MutableList<TextNode>, depth: Int) {
        if (depth > MAX_DEPTH || out.size >= MAX_NODES) return
        if (node.isVisibleToUser) {
            val text = node.text?.toString()?.trim().orEmpty().ifEmpty { node.contentDescription?.toString()?.trim().orEmpty() }
            if (text.isNotEmpty()) {
                val r = Rect()
                node.getBoundsInScreen(r)
                out.add(TextNode(text, r.left, r.top, r.right, r.bottom))
            }
        }
        for (i in 0 until node.childCount) {
            val child = try { node.getChild(i) } catch (e: Exception) { null } ?: continue
            collect(child, out, depth + 1)
        }
    }

    override fun onInterrupt() {}

    override fun onDestroy() {
        handler.removeCallbacks(readRunnable)
        connected = false
        Log.i(TAG, "Waze reader disconnected")
        super.onDestroy()
    }

    companion object {
        /** Logcat tag of the reader lines (WAZE_ALERT / CONFIRMED), next to WazeomaticEvent. */
        const val TAG = "WazeomaticWaze"
        private const val MIN_READ_INTERVAL_MS = 500L
        private const val MAX_DEPTH = 40
        private const val MAX_NODES = 1500

        @Volatile var connected = false
            private set
        @Volatile var reads = 0
            private set

        /** Is the service switched on in the system accessibility settings? */
        fun isEnabled(ctx: Context): Boolean {
            val list = android.provider.Settings.Secure.getString(
                ctx.contentResolver, android.provider.Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES
            ) ?: return false
            val me = ComponentName(ctx, WazeWatcherService::class.java)
            return list.split(':').any { ComponentName.unflattenFromString(it.trim()) == me }
        }
    }
}
