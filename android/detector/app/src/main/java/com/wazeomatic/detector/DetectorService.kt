package com.wazeomatic.detector

import android.Manifest
import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import java.io.File

/**
 * Foreground service that owns the whole pipeline so detection keeps running while Waze is
 * in front. Started and stopped from MainActivity; reads its config from Settings.
 *
 * Holds a partial wake lock while the pipeline runs, and the status line / notification say
 * whether battery optimisation is on for the app and whether frames have stopped arriving.
 */
class DetectorService : LifecycleService() {
    @Volatile private var pipeline: Pipeline? = null
    private var camera: CameraSource? = null
    private var video: VideoSource? = null
    private var tcp: TcpEventServer? = null
    private var reporter: Reporter? = null
    private var worker: HandlerThread? = null
    private var wakeLock: PowerManager.WakeLock? = null

    // status line = main text | frame warning | stalled Ns | battery state (guarded by statusLock)
    private val statusLock = Any()
    private var statusMain = ""
    private var frameWarning: String? = null
    private var stallNote: String? = null
    private var batteryNote: String? = null
    private var notifBase = ""
    private var lastNotifText = ""
    private var stopReason: String? = null

    /** Frames are expected (camera or video is running) since this elapsedRealtime; 0 = not watching. */
    @Volatile private var watchSinceMs = 0L
    private val mainHandler by lazy { Handler(mainLooper) }
    private val heartbeat = object : Runnable {
        override fun run() {
            val since = watchSinceMs
            val note = if (since == 0L) null else
                RunHealth.stallNote(SystemClock.elapsedRealtime(), maxOf(since, PipelineBus.lastFrameAtMs))
            val changed = synchronized(statusLock) {
                val was = stallNote
                stallNote = note
                if (was == null && note != null) Log.w(TAG, "no frame processed for ${RunHealth.STALL_AFTER_MS / 1000} s")
                if (was != null && note == null) Log.i(TAG, "frames are arriving again (was $was)")
                was != note
            }
            if (changed) publishStatus()
            mainHandler.postDelayed(this, HEARTBEAT_MS)
        }
    }

    override fun onBind(intent: Intent): IBinder? {
        super.onBind(intent)
        return null
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        val s = Settings.load(this)
        val camOk = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        val useCamera = s.source == "camera" && camOk
        stopReason = null
        val unrestricted = try {
            getSystemService(PowerManager::class.java).isIgnoringBatteryOptimizations(packageName)
        } catch (e: Exception) {
            false
        }
        synchronized(statusLock) {
            batteryNote = RunHealth.batteryNote(unrestricted)
            notifBase = "source: ${s.source}" + if (s.dryRun) " (dry run)" else ""
            lastNotifText = notificationText()
        }
        if (unrestricted) Log.i(TAG, "battery: unrestricted")
        else Log.w(TAG, "battery: restricted (battery optimisation is on for this app, the system may stop it in the background)")
        try {
            goForeground(useCamera)
        } catch (e: Exception) {
            Log.e(TAG, "startForeground failed", e)
            stopReason = "could not start foreground service: ${e.message}"
            PipelineBus.postStatus(stopReason!!)
            stopSelf()
            return START_NOT_STICKY
        }
        stopPipeline()
        startPipeline(s, camOk)
        return START_NOT_STICKY
    }

    private fun buildNotification(text: String): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        return Notification.Builder(this, CHANNEL)
            .setContentTitle("Wazeomatic is watching the road")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setContentIntent(open)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
    }

    private fun goForeground(useCamera: Boolean) {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL, "Detector", NotificationManager.IMPORTANCE_LOW))
        val n = buildNotification(synchronized(statusLock) { lastNotifText })
        val type = if (useCamera) ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA else ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        ServiceCompat.startForeground(this, NOTIF_ID, n, type)
    }

    /** Caller holds statusLock. */
    private fun notificationText(): String =
        listOfNotNull(notifBase, frameWarning?.let { "portrait frames" }, stallNote, batteryNote).joinToString(" | ")

    private fun setStatus(main: String) {
        synchronized(statusLock) { statusMain = main }
        publishStatus()
    }

    /** Posts the composed status line and refreshes the notification when its text changed. */
    private fun publishStatus() {
        val notif: String?
        synchronized(statusLock) {
            PipelineBus.postStatus(listOfNotNull(statusMain, frameWarning, stallNote, batteryNote).joinToString(" | "))
            val t = notificationText()
            notif = if (t != lastNotifText) t else null
            if (notif != null) lastNotifText = t
        }
        if (notif != null && PipelineBus.running) {
            try {
                getSystemService(NotificationManager::class.java).notify(NOTIF_ID, buildNotification(notif))
            } catch (e: Exception) {
                Log.w(TAG, "notification update failed: ${e.message}")
            }
        }
    }

    /** Camera or video is about to deliver frames: start counting towards "stalled". */
    private fun watchFrames(on: Boolean) {
        watchSinceMs = if (on) SystemClock.elapsedRealtime() else 0L
    }

    @SuppressLint("WakelockTimeout")
    private fun acquireWakeLock() {
        try {
            val wl = wakeLock ?: getSystemService(PowerManager::class.java)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKE_TAG).apply { setReferenceCounted(false) }
            wakeLock = wl
            if (!wl.isHeld) wl.acquire()
        } catch (e: Exception) {
            Log.e(TAG, "could not take the wake lock", e)
        }
    }

    private fun releaseWakeLock() {
        try {
            wakeLock?.let { if (it.isHeld) it.release() }
        } catch (e: Exception) {
            Log.w(TAG, "wake lock release failed: ${e.message}")
        }
    }

    private fun startPipeline(s: Settings, camOk: Boolean) {
        PipelineBus.running = true
        acquireWakeLock()
        mainHandler.postDelayed(heartbeat, HEARTBEAT_MS)
        val rep = Reporter(this, s.dryRun, s.targetPackage)
        reporter = rep
        if (s.tcpEnabled) {
            tcp = TcpEventServer(s.tcpPort) { ev -> rep.report(ev) }.also { it.start() }
        }
        if (s.source == "tcp") {
            setStatus("tcp only, listening on ${s.tcpPort}")
            return
        }
        setStatus("loading models...")
        // model loading can take a second, keep it off the main thread
        val ht = HandlerThread("pipeline-init").also { it.start() }
        worker = ht
        Handler(ht.looper).post {
            val p = try {
                Pipeline(this, s, rep)
            } catch (e: Throwable) {
                Log.e(TAG, "pipeline init failed", e)
                setStatus("init failed: ${e.message}")
                return@post
            }
            pipeline = p
            val base = p.modelStatus + if (s.tcpEnabled) " | tcp:${s.tcpPort}" else ""
            setStatus(base)
            when (s.source) {
                "video" -> {
                    val dir = getExternalFilesDir(null) ?: filesDir
                    val f = if (s.video.startsWith("/")) File(s.video) else File(dir, s.video)
                    watchFrames(true)
                    video = VideoSource(f, s.frameStep, s.loop, { bmp -> p.process(bmp) }) { st ->
                        // only a playing video is expected to deliver frames ("video <name>")
                        if (st != "video ${f.name}") watchFrames(false)
                        setStatus("$base | $st")
                    }.also { it.start() }
                }
                else -> {
                    if (!camOk) {
                        setStatus("$base | camera permission missing")
                        return@post
                    }
                    Handler(mainLooper).post {
                        if (pipeline !== p) return@post
                        watchFrames(true)
                        val onWarning = { w: String? ->
                            synchronized(statusLock) { frameWarning = w }
                            publishStatus()
                        }
                        camera = CameraSource(this, this, s.maxFps, s.orientation, onWarning) { bmp -> p.process(bmp) }.also {
                            it.start { err ->
                                watchFrames(false)
                                setStatus("$base | $err")
                            }
                        }
                    }
                }
            }
        }
    }

    /** Safe to call more than once; every stop path (restart, timeout, destroy) goes through here. */
    private fun stopPipeline() {
        mainHandler.removeCallbacks(heartbeat)
        watchFrames(false)
        worker?.quitSafely()
        worker?.join(5000)
        worker = null
        camera?.stop(); camera = null
        video?.stop(); video = null
        tcp?.stop(); tcp = null
        pipeline?.close(); pipeline = null
        reporter = null
        synchronized(statusLock) { frameWarning = null; stallNote = null }
        releaseWakeLock()
    }

    /**
     * Android 15: a dataSync foreground service (video / tcp source, the camera type has no
     * limit) may run about 6 hours in 24, then the system calls this and expects the service to
     * stop within a few seconds. Stop cleanly and say why; the user starts it again by hand.
     */
    @RequiresApi(35)
    override fun onTimeout(startId: Int, fgsType: Int) {
        Log.w(TAG, "foreground service time limit reached (type $fgsType), stopping")
        stopReason = "stopped: Android's time limit for a background video/tcp run was reached (about 6 h a day), press Start to run again"
        PipelineBus.running = false
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        stopPipeline()
        PipelineBus.postStatus(stopReason!!)
        stopSelf()
    }

    override fun onDestroy() {
        stopPipeline()
        PipelineBus.running = false
        PipelineBus.postStatus(stopReason ?: "stopped")
        super.onDestroy()
    }

    companion object {
        private const val TAG = "Wazeomatic"
        private const val CHANNEL = "detector"
        private const val NOTIF_ID = 1
        private const val WAKE_TAG = "wazeomatic:detector"
        private const val HEARTBEAT_MS = 1000L

        fun start(ctx: Context) {
            ContextCompat.startForegroundService(ctx, Intent(ctx, DetectorService::class.java))
        }

        fun stop(ctx: Context) {
            ctx.stopService(Intent(ctx, DetectorService::class.java))
        }
    }
}
