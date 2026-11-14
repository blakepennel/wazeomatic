package com.wazeomatic.detector

import android.content.Context
import android.os.Bundle

/**
 * User settings, stored in SharedPreferences. Launch extras (am start --es/--ez/--ef/--ei)
 * override them and are saved, so the UI always shows what is actually running.
 * Pass --ez reset_settings true to go back to the defaults first.
 */
data class Settings(
    val source: String = "camera",          // camera | video | tcp
    val video: String = "test.mp4",         // file name in the app external files dir
    val frameStep: Int = 5,                 // video: run inference on every Nth decoded frame
    val loop: Boolean = true,               // video: restart at the end
    val maxFps: Float = 6f,                 // camera: inference rate cap
    val confThreshold: Float = 0.40f,       // ReportManager min_confidence
    val detConf: Float = 0.35f,             // detector box threshold, same as --conf on the desktop
    val iou: Float = 0.5f,
    val cooldownS: Float = 60f,
    val globalCooldownS: Float = 60f,
    val minHits: Int = 3,
    val window: Int = 10,
    val stoppedVehicle: Boolean = false,
    val dryRun: Boolean = true,
    val targetPackage: String = "",
    val tcpEnabled: Boolean = true,
    val tcpPort: Int = 8765,
    val classMap: String = "",              // extra "label=HAZARD_KEY,label2=HAZARD_KEY" pairs, mostly for testing
    val zoomCrop: String = "true",          // "" = off, "true" = default crop, "x1,y1,x2,y2" = second pass on that region
    val ensemble: Boolean = true,           // also run hazard2.onnx (if bundled) and fuse the two hazard models
    val fusion: String = "wbf",             // wbf | nms
    val ensembleWeights: String = "",       // "2,1" = weight per hazard model (default equal)
    val classConf: String = "",             // per-label floors after fusion, "pothole=0.5,cone=0.4"
    val reportClassConf: String = "pothole=0.45,police_vehicle=0.85,worker=0.6", // ReportManager hit floors (same default as Python)
    val trackIou: Float = 0f,               // > 0 = spatial tracking in the ReportManager
    val trackGrow: Float = 1.5f,
    val hoodFrac: Float = 0.10f,            // minimum hood / dashboard zone (fraction of the frame height)
    val autoHood: Boolean = true,           // grow the zone from the static bottom of the picture (HoodEstimator)
    val zoomCoco: Boolean = false,          // also run the zoom pass on the COCO model (same default as Python)
    val orientation: String = "auto",       // camera: auto | landscape | landscape_reverse | portrait (how the phone is mounted)
    val wazeReader: Boolean = true,         // skip events Waze already shows (needs WazeWatcherService switched on)
    val vehicleClassifier: Boolean = true,  // police / ambulance / fire on COCO vehicles (needs vehicle_cls.onnx)
    val vehicleCrops: Int = 4,              // largest vehicles classified per frame
    val weather: Boolean = true,            // weather + road-surface classifiers (weather_cls / surface_cls.onnx)
    val sceneFeatures: String = SceneLabels.FEATURES.joinToString(","), // scene rules, like --scene-features; "none" = no scene layer
    val policeOtherSide: Boolean = false,   // report a police car far on the opposite side as Police > Other side
    val driveSide: String = "right",        // right | left (which side the far carriageway is on)
    val useGpsSpeed: Boolean = false,       // feed the GPS speed into the ego-motion estimate (asks for location)
    val cameraGeometry: String = "70,1.3,0.5", // HFOV deg, mount height m, horizon row (fraction), like --camera-geometry
) {
    /** Camera geometry (hfov degrees, height metres, horizon fraction); the default when the string is malformed. */
    val geometry: DoubleArray get() = parseGeometry(cameraGeometry) ?: parseGeometry(Settings().cameraGeometry)!!

    /** Scene analyzer config, or null when the scene layer is off. */
    fun sceneConfigOrNull(): SceneConfig? {
        val raw = sceneFeatures.trim().lowercase()
        if (raw == "none" || raw == "off") return null
        val g = geometry
        return SceneConfig(
            enabled = SceneConfig.parseFeatures(raw),
            driveSide = if (driveSide == "left") "left" else "right",
            policeOtherSide = policeOtherSide,
            hfovDeg = g[0],
            camHeightM = g[1],
            horizonY = g[2],
        )
    }

    val classFloors: Map<String, Float> get() = BoxFusion.parseFloors(classConf)
    /** Two-pass zoom region, null when disabled or malformed (see [ZoomCrop.parse]). */
    val zoomCropOrNull: ZoomCrop? get() = ZoomCrop.parse(zoomCrop)

    /** Detector threshold never above the report threshold, otherwise lowering conf does nothing. */
    val effectiveDetConf: Float get() = minOf(detConf, confThreshold)

    fun reportConfig() = ReportConfig(
        window = window,
        minHits = minHits,
        minConfidence = confThreshold,
        minConfidenceByLabel = BoxFusion.parseFloors(reportClassConf),
        trackIou = trackIou,
        trackGrow = trackGrow,
        hoodFrac = hoodFrac,
        cooldownS = cooldownS.toDouble(),
        globalCooldownS = globalCooldownS.toDouble(),
        stoppedVehicleEnabled = stoppedVehicle,
        classToHazard = Schema.DEFAULT_CLASS_TO_HAZARD + parseClassMap(classMap),
    )

    fun save(ctx: Context) {
        prefs(ctx).edit()
            .putString("source", source)
            .putString("video", video)
            .putInt("frame_step", frameStep)
            .putBoolean("loop", loop)
            .putFloat("max_fps", maxFps)
            .putFloat("conf", confThreshold)
            .putFloat("det_conf", detConf)
            .putFloat("iou", iou)
            .putFloat("cooldown", cooldownS)
            .putFloat("global_cooldown", globalCooldownS)
            .putInt("min_hits", minHits)
            .putInt("window", window)
            .putBoolean("stopped_vehicle", stoppedVehicle)
            .putBoolean("dry_run", dryRun)
            .putString("target_package", targetPackage)
            .putBoolean("tcp", tcpEnabled)
            .putInt("tcp_port", tcpPort)
            .putString("class_map", classMap)
            .putString("zoom_crop", zoomCrop)
            .putBoolean("ensemble", ensemble)
            .putString("fusion", fusion)
            .putString("ensemble_weights", ensembleWeights)
            .putString("class_conf", classConf)
            .putString("report_class_conf", reportClassConf)
            .putFloat("track_iou", trackIou)
            .putFloat("track_grow", trackGrow)
            .putFloat("hood_frac", hoodFrac)
            .putBoolean("auto_hood", autoHood)
            .putBoolean("zoom_coco", zoomCoco)
            .putString("orientation", orientation)
            .putBoolean("waze_reader", wazeReader)
            .putBoolean("vehicle_classifier", vehicleClassifier)
            .putInt("vehicle_crops", vehicleCrops)
            .putBoolean("weather", weather)
            .putString("scene_features", sceneFeatures)
            .putBoolean("police_other_side", policeOtherSide)
            .putString("drive_side", driveSide)
            .putBoolean("use_gps_speed", useGpsSpeed)
            .putString("camera_geometry", cameraGeometry)
            .apply()
    }

    /** Applies launch extras. Values are read as strings so --es conf 0.2 works as well as --ef. */
    fun withExtras(b: Bundle?): Settings {
        if (b == null) return this
        fun str(k: String): String? = if (b.containsKey(k)) {
            @Suppress("DEPRECATION")
            b.get(k)?.toString()
        } else null
        fun f(k: String, d: Float) = str(k)?.toFloatOrNull() ?: d
        fun i(k: String, d: Int) = str(k)?.toFloatOrNull()?.toInt() ?: d
        fun z(k: String, d: Boolean) = str(k)?.let { it.equals("true", true) || it == "1" } ?: d
        return copy(
            source = str("source")?.lowercase() ?: source,
            video = str("video") ?: video,
            frameStep = i("frame_step", frameStep).coerceAtLeast(1),
            loop = z("loop", loop),
            maxFps = f("max_fps", maxFps),
            confThreshold = f("conf", confThreshold),
            detConf = f("det_conf", detConf),
            iou = f("iou", iou),
            cooldownS = f("cooldown", cooldownS),
            globalCooldownS = f("global_cooldown", globalCooldownS),
            minHits = i("min_hits", minHits),
            window = i("window", window),
            stoppedVehicle = z("stopped_vehicle", stoppedVehicle),
            dryRun = z("dry_run", dryRun),
            targetPackage = str("target_package") ?: targetPackage,
            tcpEnabled = z("tcp", tcpEnabled),
            tcpPort = i("tcp_port", tcpPort),
            classMap = str("class_map") ?: classMap,
            zoomCrop = str("zoom_crop") ?: zoomCrop,
            ensemble = z("ensemble", ensemble),
            fusion = str("fusion")?.lowercase() ?: fusion,
            ensembleWeights = str("ensemble_weights") ?: ensembleWeights,
            classConf = str("class_conf") ?: classConf,
            reportClassConf = str("report_class_conf") ?: reportClassConf,
            trackIou = f("track_iou", trackIou),
            trackGrow = f("track_grow", trackGrow),
            hoodFrac = f("hood_frac", hoodFrac),
            autoHood = z("auto_hood", autoHood),
            zoomCoco = z("zoom_coco", zoomCoco),
            orientation = str("orientation")?.trim()?.lowercase() ?: orientation,
            wazeReader = z("waze_reader", wazeReader),
            vehicleClassifier = z("vehicle_classifier", vehicleClassifier),
            vehicleCrops = i("vehicle_crops", vehicleCrops).coerceIn(0, 16),
            weather = z("weather", weather),
            sceneFeatures = str("scene_features") ?: sceneFeatures,
            policeOtherSide = z("police_other_side", policeOtherSide),
            driveSide = str("drive_side")?.trim()?.lowercase() ?: driveSide,
            useGpsSpeed = z("use_gps_speed", useGpsSpeed),
            cameraGeometry = str("camera_geometry")?.trim() ?: cameraGeometry,
        )
    }

    companion object {
        val KNOWN_KEYS = setOf(
            "source", "video", "frame_step", "loop", "max_fps", "conf", "det_conf", "iou", "cooldown",
            "global_cooldown", "min_hits", "window", "stopped_vehicle", "dry_run", "target_package",
            "tcp", "tcp_port", "class_map", "zoom_crop", "ensemble", "fusion", "ensemble_weights",
            "class_conf", "report_class_conf", "track_iou", "track_grow", "hood_frac", "auto_hood",
            "zoom_coco", "orientation", "waze_reader", "vehicle_classifier", "vehicle_crops", "weather",
            "scene_features", "police_other_side", "drive_side", "use_gps_speed", "camera_geometry",
        )

        /** "70,1.3,0.5" -> [hfov, height, horizon], or null unless all three are numbers in a sane range. */
        fun parseGeometry(s: String): DoubleArray? {
            val v = s.split(',').map { it.trim().toDoubleOrNull() ?: return null }
            if (v.size != 3) return null
            val (hfov, height, horizon) = v
            if (hfov !in 20.0..170.0 || height !in 0.3..5.0 || horizon !in 0.0..1.0) return null
            return v.toDoubleArray()
        }

        /** "car=HAZARD_ON_ROAD_OBJECT,cat=HAZARD_SHOULDER_ANIMALS" -> map; unknown hazard keys are skipped. */
        fun parseClassMap(s: String): Map<String, String> = s.split(',', ';')
            .mapNotNull { part ->
                val kv = part.split('=', ':').map { it.trim() }
                if (kv.size == 2 && kv[0].isNotEmpty() && kv[1] in Schema.HAZARD_TYPES) kv[0] to kv[1] else null
            }.toMap()

        private fun prefs(ctx: Context) = ctx.getSharedPreferences("settings", Context.MODE_PRIVATE)

        fun reset(ctx: Context) = prefs(ctx).edit().clear().apply()

        fun load(ctx: Context): Settings {
            val p = prefs(ctx)
            val d = Settings()
            return Settings(
                source = p.getString("source", d.source) ?: d.source,
                video = p.getString("video", d.video) ?: d.video,
                frameStep = p.getInt("frame_step", d.frameStep),
                loop = p.getBoolean("loop", d.loop),
                maxFps = p.getFloat("max_fps", d.maxFps),
                confThreshold = p.getFloat("conf", d.confThreshold),
                detConf = p.getFloat("det_conf", d.detConf),
                iou = p.getFloat("iou", d.iou),
                cooldownS = p.getFloat("cooldown", d.cooldownS),
                globalCooldownS = p.getFloat("global_cooldown", d.globalCooldownS),
                minHits = p.getInt("min_hits", d.minHits),
                window = p.getInt("window", d.window),
                stoppedVehicle = p.getBoolean("stopped_vehicle", d.stoppedVehicle),
                dryRun = p.getBoolean("dry_run", d.dryRun),
                targetPackage = p.getString("target_package", d.targetPackage) ?: d.targetPackage,
                tcpEnabled = p.getBoolean("tcp", d.tcpEnabled),
                tcpPort = p.getInt("tcp_port", d.tcpPort),
                classMap = p.getString("class_map", d.classMap) ?: d.classMap,
                zoomCrop = p.getString("zoom_crop", d.zoomCrop) ?: d.zoomCrop,
                ensemble = p.getBoolean("ensemble", d.ensemble),
                fusion = p.getString("fusion", d.fusion) ?: d.fusion,
                ensembleWeights = p.getString("ensemble_weights", d.ensembleWeights) ?: d.ensembleWeights,
                classConf = p.getString("class_conf", d.classConf) ?: d.classConf,
                reportClassConf = p.getString("report_class_conf", d.reportClassConf) ?: d.reportClassConf,
                trackIou = p.getFloat("track_iou", d.trackIou),
                trackGrow = p.getFloat("track_grow", d.trackGrow),
                hoodFrac = p.getFloat("hood_frac", d.hoodFrac),
                autoHood = p.getBoolean("auto_hood", d.autoHood),
                zoomCoco = p.getBoolean("zoom_coco", d.zoomCoco),
                orientation = p.getString("orientation", d.orientation) ?: d.orientation,
                wazeReader = p.getBoolean("waze_reader", d.wazeReader),
                vehicleClassifier = p.getBoolean("vehicle_classifier", d.vehicleClassifier),
                vehicleCrops = p.getInt("vehicle_crops", d.vehicleCrops),
                weather = p.getBoolean("weather", d.weather),
                sceneFeatures = p.getString("scene_features", d.sceneFeatures) ?: d.sceneFeatures,
                policeOtherSide = p.getBoolean("police_other_side", d.policeOtherSide),
                driveSide = p.getString("drive_side", d.driveSide) ?: d.driveSide,
                useGpsSpeed = p.getBoolean("use_gps_speed", d.useGpsSpeed),
                cameraGeometry = p.getString("camera_geometry", d.cameraGeometry) ?: d.cameraGeometry,
            )
        }
    }
}
