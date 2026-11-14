package com.wazeomatic.detector

import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.SystemClock
import android.util.Log
import java.io.Closeable
import java.util.Locale
import java.util.concurrent.Callable
import java.util.concurrent.Executors

/**
 * Frame -> detectors -> vehicle classifier -> scene classifiers -> scene analyzer -> ReportManager -> Reporter,
 * the same order as Pipeline.process in wazeomatic/pipeline.py. The second stage and the scene classifiers are
 * optional assets; without them that part is off and [modelStatus] says so.
 * process() must only be called from one thread at a time (the frame source's thread).
 */
class Pipeline(ctx: Context, private val settings: Settings, private val reporter: Reporter) : Closeable {
    val detectors: List<OnnxDetector>
    /** Hazard models that are fused into one result (hazard.onnx [+ hazard2.onnx]). */
    val hazardMembers: List<OnnxDetector>
    val modelStatus: String
    private val fusionWeights: FloatArray?
    private val classFloors: Map<String, Float> = settings.classFloors
    private val pool = Executors.newFixedThreadPool(3) { r -> Thread(r, "detect").apply { isDaemon = true } }
    /** Learns the dashboard / hood band at the bottom of the picture (null = fixed zone). */
    val hood: HoodEstimator? = if (settings.autoHood) HoodEstimator(minFrac = settings.hoodFrac) else null
    private val hoodBmp: Bitmap? = hood?.let { Bitmap.createBitmap(it.width, it.height, Bitmap.Config.ARGB_8888) }
    private val hoodCanvas = hoodBmp?.let { android.graphics.Canvas(it) }
    private val hoodPaint = android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG)
    private val hoodPixels: IntArray? = hood?.let { IntArray(it.width * it.height) }
    private val hoodGrey: FloatArray? = hood?.let { FloatArray(it.width * it.height) }
    /** Current hood fraction (for the overlay). */
    val hoodFrac: Float get() = reportManager.hoodFrac
    // guarded by itself: frames and the debug injection can arrive on different threads
    @Volatile private var reportManager = newReportManager()
    /** Events not sent because Waze already showed the same report nearby. */
    val alreadyReported: Int get() = reportManager.alreadyReported
    /** Second inference pass on a zoomed region of the road ahead; null = single pass. */
    val zoomCrop: ZoomCrop? = settings.zoomCropOrNull
    private val cropRect = Rect()
    private var frameIndex = 0
    private var lastFrameNs = 0L
    private var fps = 0.0
    /** Police / ambulance / fire on the COCO vehicles; null when off or vehicle_cls.onnx is not bundled. */
    val vehicle: VehicleClassifier?
    /** Weather (whole frame) and road surface (patch ahead), each every 0.5 s of frame time; null when neither is there. */
    val sceneClassifiers: SceneClassifiers?
    /** Scene rules (traffic, crash, lanes, signals, weather) and ego motion; null when scene_features is "none". */
    val scene: SceneAnalyzer? = settings.sceneConfigOrNull()?.let { SceneAnalyzer(it) }
    // the frame as packed pixels for the second stage and the scene layer (reused between frames)
    private var framePixels = IntArray(0)

    init {
        val env = OrtEnvironment.getEnvironment()
        val list = ArrayList<OnnxDetector>()
        val notes = ArrayList<String>()
        val names = ArrayList<String>()
        names.add("hazard")
        if (settings.ensemble && assetExists(ctx, "hazard2.onnx")) names.add("hazard2")
        names.add("coco")
        for (name in names) {
            val bytes = readAsset(ctx, "$name.onnx")
            if (bytes == null) {
                notes.add("$name: missing")
                continue
            }
            val labels = readAsset(ctx, "${name}_labels.txt")?.toString(Charsets.UTF_8)
                ?.lines()?.map { it.trim() }?.filter { it.isNotEmpty() }
            try {
                val d = OnnxDetector(env, bytes, labels, name, settings.effectiveDetConf, settings.iou)
                list.add(d)
                notes.add("$name: ${d.inW}x${d.inH} ${d.labels.size} cls ${d.backend}")
            } catch (e: Exception) {
                Log.e(TAG, "failed to load $name.onnx", e)
                notes.add("$name: load failed (${e.message})")
            }
        }
        detectors = list
        hazardMembers = list.filter { it.name.startsWith("hazard") }
        fusionWeights = BoxFusion.parseWeights(settings.ensembleWeights, hazardMembers.size)
        if (hazardMembers.size > 1) notes.add("fusion ${settings.fusion}" + (fusionWeights?.let { " w=${it.toList()}" } ?: ""))
        if (classFloors.isNotEmpty()) notes.add("floors $classFloors")
        if (settings.trackIou > 0f) notes.add("track ${settings.trackIou}")
        if (hood != null) notes.add("auto hood")
        if (zoomCrop != null) {
            val zoomed = list.filter { ZoomCrop.appliesTo(it.name, settings.zoomCoco) }.joinToString("+") { it.name }
            notes.add("zoom crop $zoomCrop on ${zoomed.ifEmpty { "no model" }}" + if (settings.zoomCoco) "" else " (coco not zoomed)")
        }
        // second stage: needs the stock detector's vehicles (as build_second_stage: no coco, no vehicle classifier)
        val hasCoco = list.any { it.name == "coco" }
        vehicle = if (!settings.vehicleClassifier) null else if (!hasCoco) {
            notes.add("vehicle: off (no coco)")
            null
        } else loadClassifier(ctx, env, "vehicle_cls", "vehicle", notes)?.let { VehicleClassifier(it, maxCrops = settings.vehicleCrops) }
        sceneClassifiers = if (!settings.weather) null else {
            val w = loadClassifier(ctx, env, "weather_cls", "weather", notes, threads = 1)?.let { SceneClassifier(it) }
            val r = loadClassifier(ctx, env, "surface_cls", "surface", notes, threads = 1)?.let { RoadSurface(it) }
            if (w == null && r == null) null else SceneClassifiers(listOf(w, r))
        }
        notes.add(scene?.let { s -> "scene " + (s.cfg.enabled.joinToString(",").ifEmpty { "lights only" }) +
            (if (s.cfg.policeOtherSide) " +police other side" else "") + (if (s.cfg.driveSide == "left") " (left-hand traffic)" else "") }
            ?: "scene off")
        modelStatus = if (list.isEmpty()) "model missing" else notes.joinToString(" | ")
        Log.i(TAG, "models: $modelStatus")
    }

    val hasModel: Boolean get() = detectors.isNotEmpty()

    /** [ts] is the frame time in seconds (video: position in the clip, camera: elapsedRealtime); the scene rules run on it. */
    fun process(frame: Bitmap, ts: Double = SystemClock.elapsedRealtimeNanos() / 1e9) {
        frameIndex++
        val lat = LinkedHashMap<String, Double>()
        val dets = ArrayList<Detection>()
        // crop region in pixels; null when off or when the frame is too small for it
        val crop = zoomCrop?.pixelRect(frame.width, frame.height)
        val effCrop = crop?.let { ZoomCrop.fromRect(it, frame.width, frame.height) }
        if (crop != null) cropRect.set(crop.x1, crop.y1, crop.x2, crop.y2)
        val t0 = System.nanoTime()
        // every model (and its zoom pass) runs on its own thread; ORT releases the Java monitor.
        // The zoom pass is for the hazard model(s); COCO only with zoom_coco (as in build.py).
        val tasks = detectors.map { d ->
            val zoomThis = effCrop != null && ZoomCrop.appliesTo(d.name, settings.zoomCoco)
            Callable {
                try {
                    val full = d.detect(frame)
                    val msFull = d.lastMs
                    if (effCrop == null || !zoomThis) {
                        Triple(d, full, msFull to null)
                    } else {
                        val zoomed = d.detect(frame, cropRect)
                        Triple(d, ZoomCrop.merge(full + effCrop.toFrame(zoomed), d.iouThres), msFull to d.lastMs)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "inference failed in ${d.name}", e)
                    Triple(d, emptyList(), 0.0 to null)
                }
            }
        }
        val perModel = LinkedHashMap<OnnxDetector, List<Detection>>()
        for (f in pool.invokeAll(tasks)) {
            val (d, out, ms) = f.get()
            perModel[d] = out
            lat[d.name] = ms.first
            ms.second?.let { lat["${d.name}_zoom"] = it }
        }
        lat["wall"] = (System.nanoTime() - t0) / 1e6
        val hazardLists = hazardMembers.map { perModel[it] ?: emptyList() }
        val hazardDets = when {
            hazardLists.isEmpty() -> emptyList()
            hazardLists.size == 1 -> hazardLists[0]
            settings.fusion == "nms" -> BoxFusion.fuseNms(hazardLists, hazardMembers[0].iouThres)
            else -> BoxFusion.fuseWbf(hazardLists, fusionWeights, modelName = "hazard")
        }
        dets.addAll(BoxFusion.applyClassFloors(hazardDets, classFloors))
        for (d in detectors) if (d !in hazardMembers) dets.addAll(perModel[d] ?: emptyList())
        secondStage(frame, ts, dets, lat)
        val now = System.nanoTime()
        if (lastFrameNs != 0L) {
            val inst = 1e9 / (now - lastFrameNs)
            fps = if (fps == 0.0) inst else fps * 0.8 + inst * 0.2
        }
        lastFrameNs = now
        if (lat.isNotEmpty()) {
            Log.d(PERF_TAG, String.format(Locale.US, "frame=%d %s fps=%.1f dets=%d", frameIndex,
                lat.entries.joinToString(" ") { String.format(Locale.US, "%s=%.1fms", it.key, it.value) }, fps, dets.size))
        }
        updateHood(frame)
        sceneClassifiers?.setHoodFrac(reportManager.hoodFrac.toDouble())
        if (hasModel) feed(dets)
        PipelineBus.postFrame(FrameResult(frame, dets, lat, fps, frameIndex))
    }

    /**
     * Vehicle classifier, scene classifiers and scene analyzer on one frame; their detections are appended to
     * [dets]. Bounded per frame: at most [VehicleClassifier.maxCrops] crops, the scene classifiers at 2 Hz.
     */
    private fun secondStage(frame: Bitmap, ts: Double, dets: MutableList<Detection>, lat: MutableMap<String, Double>) {
        if (vehicle == null && sceneClassifiers == null && scene == null) return
        val n = frame.width * frame.height
        if (framePixels.size != n) framePixels = IntArray(n)
        frame.getPixels(framePixels, 0, frame.width, 0, 0, frame.width, frame.height)
        val img = RgbImage(frame.width, frame.height, framePixels)
        try {
            vehicle?.let { v ->
                dets.addAll(v(img, dets))
                if (v.lastMs > 0) lat["vehicle"] = v.lastMs
            }
            val weather = sceneClassifiers?.let { c -> c(img, ts).also { if (c.lastMs > 0) lat["weather"] = c.lastMs } }
            scene?.let { s ->
                val t0 = System.nanoTime()
                dets.addAll(s.update(ts, img, dets, weather))
                lat["scene"] = (System.nanoTime() - t0) / 1e6
            }
        } catch (e: Exception) {
            Log.e(TAG, "second stage failed", e)
        }
    }

    /** GPS speed for the ego-motion estimate (m/s, null = unknown), on the same clock as [process]'s ts. */
    fun setGpsSpeed(mps: Double?, ts: Double = SystemClock.elapsedRealtimeNanos() / 1e9) {
        scene?.ego?.setGpsSpeed(mps, ts)
    }

    /** "ego slow | scene_fog" for the status line; null without the scene layer or before the first frame. */
    fun sceneStatus(): String? {
        val s = scene ?: return null
        val ego = s.state["ego"] ?: return null
        return (listOf("ego $ego") + s.activeLabels).joinToString(" ")
    }

    private fun newReportManager() = ReportManager(
        settings.reportConfig(), source = "phone", waze = if (settings.wazeReader) WazeAlerts.shared else null
    )

    /** One frame's detections into the ReportManager; events go to the reporter, confirmations to logcat. */
    private fun feed(dets: List<Detection>, index: Int = frameIndex) {
        val rm = reportManager
        val events: List<HazardEvent>
        val confirmed: List<Confirmation>
        synchronized(rm) {
            val before = rm.confirmations.size
            events = rm.update(dets, index)
            confirmed = rm.confirmations.drop(before)
        }
        for (c in confirmed) Log.i(WazeWatcherService.TAG, "CONFIRMED ${c.toJson()}")
        for (ev in events) reporter.report(ev)
    }

    /**
     * Debug builds only (DetectorService checks BuildConfig.DEBUG): feeds [count] synthetic frames that
     * each hold one [label] detection, so the end-to-end test can produce an event without a matching clip.
     * [reset] starts from a fresh ReportManager (as a service restart would), [clearWaze] forgets what the
     * Waze reader has seen so far.
     */
    fun inject(label: String, confidence: Float, count: Int, reset: Boolean, clearWaze: Boolean) {
        if (clearWaze) WazeAlerts.shared.clear()
        if (reset) reportManager = newReportManager()
        val det = Detection(label, confidence, BBox(0.6f, 0.5f, 0.1f, 0.1f), "inject")
        Log.i(TAG, "inject: $count frame(s) of $label @ $confidence reset=$reset clear_waze=$clearWaze")
        // injected frames take the next frame numbers, so the persistence window counts them like real ones
        // (meant for a run without a frame source; with one running the two simply interleave)
        repeat(count) { feed(listOf(det), ++frameIndex) }
    }

    /** Reduce the frame to the estimator's grey image and feed it (a few hundred microseconds). */
    private fun updateHood(frame: Bitmap) {
        val est = hood ?: return
        val bmp = hoodBmp ?: return
        val canvas = hoodCanvas ?: return
        val px = hoodPixels ?: return
        val grey = hoodGrey ?: return
        canvas.drawBitmap(frame, null, Rect(0, 0, bmp.width, bmp.height), hoodPaint)
        bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        for (i in px.indices) {
            val c = px[i]
            grey[i] = 0.299f * ((c shr 16) and 0xFF) + 0.587f * ((c shr 8) and 0xFF) + 0.114f * (c and 0xFF)
        }
        reportManager.setHoodFrac(est.update(grey))
    }

    override fun close() {
        pool.shutdownNow()
        detectors.forEach { try { it.close() } catch (_: Exception) {} }
        try { vehicle?.close() } catch (_: Exception) {}
        try { sceneClassifiers?.close() } catch (_: Exception) {}
    }

    companion object {
        private const val TAG = "Wazeomatic"
        const val PERF_TAG = "WazeomaticPerf"

        fun readAsset(ctx: Context, name: String): ByteArray? = try {
            ctx.assets.open(name).use { it.readBytes() }
        } catch (e: Exception) {
            null
        }

        /** <asset>.onnx + <asset>_labels.txt as a classifier, or null (and a status note) when absent or broken. */
        private fun loadClassifier(
            ctx: Context, env: OrtEnvironment, asset: String, label: String, notes: MutableList<String>, threads: Int = 2,
        ): ImageClassifier? {
            val bytes = readAsset(ctx, "$asset.onnx")
            val labels = readAsset(ctx, "${asset}_labels.txt")?.toString(Charsets.UTF_8)?.let { OnnxClassifier.parseLabels(it) }
            if (bytes == null || labels.isNullOrEmpty()) {
                notes.add("$label: missing")
                return null
            }
            return try {
                OnnxClassifier(env, bytes, labels, asset, threads).also { notes.add("$label: ${it.size}px ${labels.size} cls") }
            } catch (e: Exception) {
                Log.e(TAG, "failed to load $asset.onnx", e)
                notes.add("$label: load failed (${e.message})")
                null
            }
        }

        fun assetExists(ctx: Context, name: String): Boolean =
            try { ctx.assets.list("")?.contains(name) == true } catch (e: Exception) { false }
    }
}
