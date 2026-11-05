package com.wazeomatic.detector

import android.content.Context
import android.content.Intent
import android.util.Log
import java.io.File

/**
 * Sends one broadcast per HazardEvent for the MacroDroid macro, and records it in logcat
 * (tag WazeomaticEvent, one JSON line) and in files/events.jsonl.
 */
class Reporter(
    private val ctx: Context,
    private val dryRun: Boolean,
    private val targetPackage: String,
) {
    val eventsFile: File = File(ctx.getExternalFilesDir(null) ?: ctx.filesDir, "events.jsonl")

    @Synchronized
    fun report(ev: HazardEvent) {
        val intent = Intent(INTENT_ACTION).apply {
            for ((k, v) in ev.intentExtras()) putExtra(k, v)
            putExtra("dry_run", if (dryRun) "true" else "false")
            addFlags(Intent.FLAG_INCLUDE_STOPPED_PACKAGES)
            if (targetPackage.isNotBlank()) setPackage(targetPackage.trim())
        }
        try {
            ctx.sendBroadcast(intent)
        } catch (e: Exception) {
            Log.e(TAG, "broadcast failed", e)
        }
        val json = ev.toJson()
        Log.i(EVENT_TAG, json)
        try {
            eventsFile.appendText(json + "\n")
        } catch (e: Exception) {
            Log.e(TAG, "could not append to $eventsFile", e)
        }
        PipelineBus.postEvent(ev, dryRun)
    }

    companion object {
        const val EVENT_TAG = "WazeomaticEvent"
        private const val TAG = "Wazeomatic"
    }
}
