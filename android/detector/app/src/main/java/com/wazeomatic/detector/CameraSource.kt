package com.wazeomatic.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import android.util.Size
import android.view.OrientationEventListener
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
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
    private val onFrame: (Bitmap) -> Unit,
) {
    private val minIntervalNs = (1e9 / maxFps.coerceIn(0.5f, 30f)).toLong()
    private var lastNs = 0L
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
                val selector = ResolutionSelector.Builder()
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
                    val now = System.nanoTime()
                    if (now - lastNs < minIntervalNs) {
                        proxy.close()
                        return@setAnalyzer
                    }
                    lastNs = now
                    val rot = proxy.imageInfo.rotationDegrees
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
                        onFrame(bmp)
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
