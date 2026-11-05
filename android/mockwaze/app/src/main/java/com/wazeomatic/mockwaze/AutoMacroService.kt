package com.wazeomatic.mockwaze

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log

/**
 * Foreground service that keeps a dynamically registered receiver for
 * com.wazeomatic.HAZARD_DETECTED alive while auto macro is on.
 *
 * Why not a manifest receiver: since Android 8 (API 26) implicit broadcasts (no package or
 * component set, which is how the detector and `am broadcast -a ...` send them) are not
 * delivered to receivers declared in the manifest. A receiver registered at runtime does get
 * them, and the foreground service keeps the process (and the receiver) alive even when the
 * detector app is on screen. MacroDroid works the same way.
 */
class AutoMacroService : Service() {

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                MacroEngine.ACTION_HAZARD -> MacroEngine.onHazardIntent(context, intent)
                Config.ACTION_SET_TIMING -> {
                    Config.applyExtras(context, intent)
                    if (!Config.bool(context, "auto_macro")) stopSelf()
                }
            }
        }
    }
    private var registered = false

    override fun onCreate() {
        super.onCreate()
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL, "Auto macro", NotificationManager.IMPORTANCE_LOW))
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE)
        val n = Notification.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_report)
            .setContentTitle("Mock Waze auto macro is on")
            .setContentText("Listening for ${MacroEngine.ACTION_HAZARD}")
            .setContentIntent(open)
            .setOngoing(true)
            .build()
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(NOTIF_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(NOTIF_ID, n)
        }
        val filter = IntentFilter().apply {
            addAction(MacroEngine.ACTION_HAZARD)
            addAction(Config.ACTION_SET_TIMING)
        }
        if (Build.VERSION.SDK_INT >= 33) {
            registerReceiver(receiver, filter, Context.RECEIVER_EXPORTED)
        } else {
            registerReceiver(receiver, filter)
        }
        registered = true
        Log.i(ReportLog.MACRO_TAG, "auto macro service started, listening for ${MacroEngine.ACTION_HAZARD}")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_STICKY

    override fun onDestroy() {
        if (registered) unregisterReceiver(receiver)
        registered = false
        Log.i(ReportLog.MACRO_TAG, "auto macro service stopped")
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    companion object {
        private const val CHANNEL = "auto_macro"
        private const val NOTIF_ID = 42

        /** Start or stop the service to match the auto_macro setting. */
        fun sync(ctx: Context) {
            val i = Intent(ctx, AutoMacroService::class.java)
            try {
                if (Config.bool(ctx, "auto_macro")) ctx.startForegroundService(i) else ctx.stopService(i)
            } catch (e: Exception) {
                // e.g. ForegroundServiceStartNotAllowedException when started from the background
                Log.w(ReportLog.MACRO_TAG, "could not start auto macro service: ${e.message}")
            }
        }
    }
}

/** Manifest receiver for SET_TIMING sent with an explicit package (-p com.wazeomatic.mockwaze). */
class ControlReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Config.ACTION_SET_TIMING) {
            Config.applyExtras(context, intent)
        }
    }
}
