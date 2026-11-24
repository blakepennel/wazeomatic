package com.wazeomatic.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import android.util.Size
import android.view.OrientationEventListener
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Back camera at 1280x720 through CameraX ImageAnalysis. KEEP_ONLY_LATEST plus a time gate
 * means frames are dropped, never queued, while inference is busy.
 *
 * Frames are turned upright for how the phone is physically mounted, not for the display:
 * [orientation] "auto" follows the accelerometer (works with auto-rotate off and while another
 * app is in front), the fixed values pin it (see [MountOrientation]).
 */
class CameraSource(
    private val ctx: Context,
    private val owner: LifecycleOwner,
    maxFps: Float,
    private val orientation: String,
    /** Called when the delivered frame shape changes: a warning for portrait frames, else null. */
    private val onFrameWarning: (String?) -> Unit,
    /** The frame and its capture time (imageInfo.timestamp, nanoseconds; elapsedRealtimeNanos on most phones). */
    private val onFrame: (Bitmap, Long) -> Unit,
) {
    private val throttle = FrameThrottle((1e9 / maxFps.coerceIn(0.5f, 30f)).toLong())
    private val executor: ExecutorService = Executors.newSingleThreadExecutor { Thread(it, "camera-analysis") }
    private var provider: ProcessCameraProvider? = null
    private var orientationListener: OrientationEventListener? = null
    private var targetRotation = MountOrientation.fixedRotation(orientation) ?: MountOrientation.DEFAULT_ROTATION
    // last delivered frame shape, analysis thread only
    private var lastW = 0
    private var lastH = 0
    private var lastRot = -1

    fun start(onError: (String) -> Unit) {
        val future = ProcessCameraProvider.getInstance(ctx)
        future.addListener({
            try {
                val p = future.get()
                provider = p
                // 16:9 like the test clips: CameraX picks 4:3 unless asked (1280x960 on a OnePlus N10), which costs
                // a quarter more inference for sky and dashboard; the horizontal field of view is the same
                val selector = ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                    .setResolutionStrategy(
                        ResolutionStrategy(Size(1280, 720), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                    ).build()
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(selector)
                    .setTargetRotation(targetRotation)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                analysis.setAnalyzer(executor) { proxy ->
                    if (!throttle.accept(System.nanoTime())) {
                        proxy.close()
                        return@setAnalyzer
                    }
                    val rot = proxy.imageInfo.rotationDegrees
                    val captured = proxy.imageInfo.timestamp
                    val bmp = try {
                        val raw = proxy.toBitmap()
                        if (rot == 0) raw else {
                            val m = Matrix().apply { postRotate(rot.toFloat()) }
                            Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
                        }
                    } finally {
                        proxy.close()
                    }
                    noteFrameShape(bmp.width, bmp.height, rot)
                    try {
                        onFrame(bmp, captured)
                    } catch (e: Exception) {
                        Log.e(TAG, "frame processing failed", e)
                    }
                }
                p.unbindAll()
                p.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
                Log.i(TAG, "camera started, orientation=$orientation target=${MountOrientation.name(targetRotation)}")
                if (MountOrientation.fixedRotation(orientation) == null) followDevice(analysis)
            } catch (e: Exception) {
                Log.e(TAG, "camera start failed", e)
                onError("camera error: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(ctx))
    }

    /** Auto mode: keep the target rotation in step with how the phone is held (main thread). */
    private fun followDevice(analysis: ImageAnalysis) {
        val l = object : OrientationEventListener(ctx) {
            override fun onOrientationChanged(degrees: Int) {
                val next = MountOrientation.next(targetRotation, degrees)
                if (next == targetRotation) return
                targetRotation = next
                analysis.targetRotation = next
                Log.i(TAG, "mount orientation ${MountOrientation.name(next)} (device at $degrees deg)")
            }
        }
        if (l.canDetectOrientation()) {
            l.enable()
            orientationListener = l
        } else {
            Log.w(TAG, "no orientation sensor, assuming a ${MountOrientation.name(targetRotation)} mount")
        }
    }

    private fun noteFrameShape(w: Int, h: Int, rot: Int) {
        if (w == lastW && h == lastH && rot == lastRot) return
        val shapeChanged = w != lastW || h != lastH
        lastW = w; lastH = h; lastRot = rot
        val warning = RunHealth.frameWarning(w, h)
        Log.i(TAG, "camera frames ${w}x$h rotation=$rot" + (warning?.let { " WARNING: $it" } ?: ""))
        if (shapeChanged) {
            try { onFrameWarning(warning) } catch (e: Exception) { Log.e(TAG, "frame warning callback failed", e) }
        }
    }

    /** Call on the main thread. */
    fun stop() {
        orientationListener?.disable()
        orientationListener = null
        try { provider?.unbindAll() } catch (_: Exception) {}
        executor.shutdown()
        try { executor.awaitTermination(2, java.util.concurrent.TimeUnit.SECONDS) } catch (_: Exception) {}
    }

    companion object {
        private const val TAG = "Wazeomatic"
    }
}

/**
 * The max_fps cap on a camera that delivers on a fixed grid (30 fps here). A plain "at least 1/6 s since the last
 * frame" test drops the frame that lands a hair early and waits a whole extra camera frame: 200 ms between
 * frames, 5 fps instead of 6. This one keeps a schedule instead: a frame is taken when it is no more than [slackNs]
 * before its slot, and the next slot follows the last one, so the rate averages out at the cap.
 */
class FrameThrottle(private val intervalNs: Long, private val slackNs: Long = minOf(20_000_000L, intervalNs / 4)) {
    private var nextNs = Long.MIN_VALUE

    fun accept(nowNs: Long): Boolean {
        if (nextNs != Long.MIN_VALUE && nowNs < nextNs - slackNs) return false
        // behind by more than a slot (slow frames, a pause): start the schedule again from now
        nextNs = if (nextNs == Long.MIN_VALUE || nowNs - nextNs > intervalNs) nowNs + intervalNs else nextNs + intervalNs
        return true
    }
}
