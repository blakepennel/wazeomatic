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
    // one thread per pass: every model's whole frame, each zoom pass, the second hazard model
    private val pool = Executors.newFixedThreadPool(5) { r -> Thread(r, "detect").apply { isDaemon = true } }
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
    private var lastReturnNs = 0L
    private val post = Executors.newSingleThreadExecutor { r -> Thread(r, "post").apply { isDaemon = true } }
    @Volatile private var pending: java.util.concurrent.Future<*>? = null
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
        // passes that run at once (every model plus each zoom pass) share the cores: on 8 cores three passes at
        // two XNNPACK threads each beat four each (OnePlus N10: 180 against 250 ms a frame, docs/model-research.md)
        val zoomOn = settings.zoomCropOrNull != null
        val passCount = names.size + names.count { zoomOn && ZoomCrop.appliesTo(it, settings.zoomCoco) }
        val threads = if (settings.ortThreads > 0) settings.ortThreads else OnnxDetector.autoThreads(passCount)
        for (name in names) {
            val bytes = readAsset(ctx, "$name.onnx")
            if (bytes == null) {
                notes.add("$name: missing")
                continue
            }
            val labels = readAsset(ctx, "${name}_labels.txt")?.toString(Charsets.UTF_8)
                ?.lines()?.map { it.trim() }?.filter { it.isNotEmpty() }
            try {
                val d = OnnxDetector(env, bytes, labels, name, settings.effectiveDetConf, settings.iou,
                    rect = settings.rectInput, threads = threads)
                list.add(d)
                notes.add("$name: ${d.shapeNote}${if (d.openShape && settings.rectInput) " rect" else ""} ${d.labels.size} cls ${d.backend}")
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
        } else loadClassifier(ctx, env, "vehicle_cls", "vehicle", notes)?.let { first ->
            // a second vehicle model, when bundled, is averaged with the first (models/README.md)
            val second = if (assetExists(ctx, "vehicle_cls2.onnx")) loadClassifier(ctx, env, "vehicle_cls2", "vehicle 2", notes) else null
            VehicleClassifier(first, maxCrops = settings.vehicleCrops, more = listOfNotNull(second))
        }
        sceneClassifiers = if (!settings.weather) null else {
            val w = loadClassifier(ctx, env, "weather_cls", "weather", notes, threads = 1)?.let { SceneClassifier(it) }
            val r = loadClassifier(ctx, env, "surface_cls", "surface", notes, threads = 1)?.let { RoadSurface(it) }
            // a RoadSaW / RoadSC model wants the bird's-eye patch, warped with the mount's camera geometry
            if (r != null && r.isTopView) {
                r.topview = settings.geometry
                notes.add("surface top view")
            }
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
        val tIn = System.nanoTime()
        val lat = LinkedHashMap<String, Double>()
        val dets = ArrayList<Detection>()
        // crop region in pixels; null when off or when the frame is too small for it
        val crop = zoomCrop?.pixelRect(frame.width, frame.height)
        val effCrop = crop?.let { ZoomCrop.fromRect(it, frame.width, frame.height) }
        if (crop != null) cropRect.set(crop.x1, crop.y1, crop.x2, crop.y2)
        val t0 = System.nanoTime()
        // every model and every zoom pass runs on its own thread (each pass has its own session); ORT releases
        // the Java monitor. The zoom pass is for the hazard model(s); COCO only with zoom_coco (as in build.py).
        class Pass(val d: OnnxDetector, val zoom: Boolean)
        val passes = ArrayList<Pass>()
        for (d in detectors) {
            passes.add(Pass(d, false))
            if (effCrop != null && ZoomCrop.appliesTo(d.name, settings.zoomCoco)) passes.add(Pass(d, true))
        }
        val tasks = passes.map { p ->
            Callable {
                try {
                    if (p.zoom) p.d.detect(frame, cropRect, OnnxDetector.SLOT_ZOOM) else p.d.detect(frame)
                } catch (e: Exception) {
                    Log.e(TAG, "inference failed in ${p.d.name}${if (p.zoom) " (zoom)" else ""}", e)
                    emptyList()
                }
            }
        }
        val full = HashMap<OnnxDetector, List<Detection>>()
        val zoomed = HashMap<OnnxDetector, List<Detection>>()
        for ((p, f) in passes.zip(pool.invokeAll(tasks))) (if (p.zoom) zoomed else full)[p.d] = f.get()
        val perModel = LinkedHashMap<OnnxDetector, List<Detection>>()
        for (d in detectors) {
            val z = zoomed[d]
            lat[d.name] = d.lastMs
            perModel[d] = if (z == null || effCrop == null) full[d] ?: emptyList() else {
                lat["${d.name}_zoom"] = d.lastZoomMs
                ZoomCrop.merge((full[d] ?: emptyList()) + effCrop.toFrame(z), d.iouThres)
            }
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
        val idx = frameIndex
        val tA = System.nanoTime()
        lat["infer"] = (tA - tIn) / 1e6
        // Second stage on its own thread: while it works on this frame the caller decodes the next one and its
        // inference starts. At most one frame waits here, so frames stay in order and memory stays bounded.
        val tWait = System.nanoTime()
        try { pending?.get() } catch (e: Exception) { Log.e(TAG, "second stage of the previous frame failed", e) }
        val waited = (System.nanoTime() - tWait) / 1e6
        val src = if (lastReturnNs != 0L) (tIn - lastReturnNs) / 1e6 else null
        pending = post.submit { finish(frame, ts, dets, lat, idx, src, waited) }
        lastReturnNs = System.nanoTime()
    }

    /** Second stage, hood, Report Manager and the screen for one frame; runs on [post], one frame at a time. */
    private fun finish(frame: Bitmap, ts: Double, dets: ArrayList<Detection>, lat: LinkedHashMap<String, Double>,
                       idx: Int, src: Double?, waited: Double) {
        val tB = System.nanoTime()
        secondStage(frame, ts, dets, lat)
        val now = System.nanoTime()
        if (lastFrameNs != 0L) {
            val inst = 1e9 / (now - lastFrameNs)
            fps = if (fps == 0.0) inst else fps * 0.8 + inst * 0.2
        }
        lastFrameNs = now
        updateHood(frame)
        sceneClassifiers?.setHoodFrac(reportManager.hoodFrac.toDouble())
        if (hasModel) feed(dets, idx)
        PipelineBus.postFrame(FrameResult(frame, dets, lat, fps, idx))
        if (lat.isNotEmpty()) {
            // infer: models and fusion (caller thread); post: this function; src: between two process() calls
            // (decoding and converting the next frame); wait: the caller blocked on the previous frame's post
            val extra = String.format(Locale.US, "post=%.1fms wait=%.1fms", (System.nanoTime() - tB) / 1e6, waited) +
                (src?.let { String.format(Locale.US, " src=%.1fms", it) } ?: "")
            Log.d(PERF_TAG, String.format(Locale.US, "frame=%d %s %s fps=%.1f dets=%d", idx,
                lat.entries.joinToString(" ") { String.format(Locale.US, "%s=%.1fms", it.key, it.value) }, extra, fps, dets.size))
        }
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
                // emergency-class scores worth a look, so a missed or false Police report can be traced on the phone
                for ((d, p) in v.lastProbs) {
                    val top = p.filterKeys { it != "civilian" }.maxByOrNull { it.value } ?: continue
                    if (top.value >= 0.3f) Log.d(VEHICLE_TAG, String.format(Locale.US, "frame=%d %s=%.2f box=%.3f,%.3f,%.3f,%.3f",
                        frameIndex, top.key, top.value, d.bbox.cx, d.bbox.cy, d.bbox.w, d.bbox.h))
                }
            }
            val weather = sceneClassifiers?.let { c -> c(img, ts, dets).also { if (c.lastMs > 0) lat["weather"] = c.lastMs } }
            scene?.let { s ->
                val t0 = System.nanoTime()
                dets.addAll(s.update(ts, img, dets, weather))
                val t1 = System.nanoTime()
                lat["scene_rules"] = (t1 - t0) / 1e6
                // a copy in: the result replaces dets, so it must never be dets itself (clear() would empty both)
                val gated = s.gatePolice(ts, img, dets.toList(), reportManager.hoodFrac.toDouble())
                dets.clear()
                dets.addAll(gated)
                lat["police"] = (System.nanoTime() - t1) / 1e6
                dets.addAll(s.shoulderCars(ts, img, dets, reportManager.hoodFrac.toDouble()))
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
        // the last frame's second stage uses the classifiers closed below
        try { pending?.get(2, java.util.concurrent.TimeUnit.SECONDS) } catch (_: Exception) {}
        post.shutdownNow()
        pool.shutdownNow()
        detectors.forEach { try { it.close() } catch (_: Exception) {} }
        try { vehicle?.close() } catch (_: Exception) {}
        try { sceneClassifiers?.close() } catch (_: Exception) {}
    }

    companion object {
        private const val TAG = "Wazeomatic"
        const val PERF_TAG = "WazeomaticPerf"
        const val VEHICLE_TAG = "WazeomaticVehicle"

        /**
         * The bundled asset, or in a debug build <external files>/models/<name> when that exists, so a model can be
         * tried on the phone without a rebuild: adb push hazard.onnx /sdcard/Android/data/<package>/files/models/
         */
        fun readAsset(ctx: Context, name: String): ByteArray? {
            if (BuildConfig.DEBUG) {
                val f = ctx.getExternalFilesDir(null)?.let { java.io.File(it, "models/$name") }
                if (f != null && f.isFile) {
                    Log.w(TAG, "debug: $name from ${f.path} instead of the bundled asset")
                    return f.readBytes()
                }
            }
            return try {
                ctx.assets.open(name).use { it.readBytes() }
            } catch (e: Exception) {
                null
            }
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
