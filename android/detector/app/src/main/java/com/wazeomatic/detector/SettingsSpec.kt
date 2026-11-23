package com.wazeomatic.detector

import java.util.Locale
import kotlin.math.roundToInt

/**
 * What the settings screen shows for each launch-extra key of [Settings]. The screen reads the saved value by key
 * and writes a change back through [Settings.withExtras], the same path `am start --es key value` takes, so the
 * two can never disagree. A key [Settings] does not know (yet) is simply not shown (see [SettingsSpec.visible]).
 */
sealed class SettingSpec(val key: String, val title: String, val help: String, val dependsOn: String?) {
    class Toggle(key: String, title: String, help: String, dependsOn: String? = null) :
        SettingSpec(key, title, help, dependsOn)

    /** A slider from [from] to [to]; [zeroLabel] names the value 0 ("auto", "off"), [percent] shows 0.1 as 10 %. */
    class Range(
        key: String, title: String, help: String,
        val from: Float, val to: Float, val step: Float,
        val unit: String = "", val decimals: Int = 0, val zeroLabel: String? = null, val percent: Boolean = false,
        dependsOn: String? = null,
    ) : SettingSpec(key, title, help, dependsOn) {
        /** [v] clamped into the range and onto the step grid (the slider refuses anything else). */
        fun snap(v: Float): Float {
            val n = ((v.coerceIn(from, to) - from) / step).roundToInt()
            return (from + n * step).coerceIn(from, to)
        }

        fun format(v: Float): String {
            if (zeroLabel != null && v == 0f) return zeroLabel
            if (percent) return "${(v * 100).roundToInt()} %"
            val num = if (decimals == 0) v.roundToInt().toString() else String.format(Locale.US, "%.${decimals}f", v)
            return num + unit
        }

        /** The string written for [v]: whole numbers without ".0" so integer keys parse either way. */
        fun encode(v: Float): String = if (decimals == 0 && !percent) v.roundToInt().toString()
        else String.format(Locale.US, "%.${maxOf(decimals, 2)}f", v)
    }

    /** A dropdown of value to label; a saved value that is not one of them (set over adb) is kept as "Custom". */
    class Choice(key: String, title: String, help: String, val options: List<Pair<String, String>>, dependsOn: String? = null) :
        SettingSpec(key, title, help, dependsOn) {
        fun labelOf(v: String) = options.firstOrNull { it.first == v }?.second ?: "Custom: $v"
    }

    /** Free text; [validate] returns an error message or null. */
    class Text(
        key: String, title: String, help: String, val hint: String = "", val numeric: Boolean = false,
        val validate: (String) -> String? = { null }, dependsOn: String? = null,
    ) : SettingSpec(key, title, help, dependsOn)

    /** A comma list picked with chips; nothing picked is written as [noneValue]. */
    class Multi(key: String, title: String, help: String, val options: List<Pair<String, String>>, val noneValue: String) :
        SettingSpec(key, title, help, null) {
        fun parse(v: String): Set<String> = v.split(',').map { it.trim().lowercase() }.filter { o -> options.any { it.first == o } }.toSet()
        fun encode(picked: Set<String>): String =
            options.map { it.first }.filter { it in picked }.joinToString(",").ifEmpty { noneValue }
    }

    /** camera_geometry: three sliders written back as "hfov,height,horizon". */
    class Geometry(key: String, title: String, help: String) : SettingSpec(key, title, help, null) {
        val parts = listOf(
            Range("$key.hfov", "Horizontal field of view", "", 40f, 120f, 1f, unit = "°"),
            Range("$key.height", "Mount height", "", 0.5f, 3f, 0.05f, unit = " m", decimals = 2),
            Range("$key.horizon", "Horizon row", "", 0.3f, 0.7f, 0.01f, percent = true),
        )

        fun encode(v: List<Float>): String = String.format(Locale.US, "%.0f,%.2f,%.2f", v[0], v[1], v[2])
    }

    /** The video clip, picked from a sheet (no typing). */
    class Clip(key: String, title: String, help: String) : SettingSpec(key, title, help, null)
}

class SettingGroup(val title: String, val items: List<SettingSpec>)

object SettingsSpec {
    /** Keys that may not exist in [Settings] yet; their controls appear once it knows them. */
    val OPTIONAL = setOf("realtime")

    private fun port(s: String): String? {
        val p = s.trim().toIntOrNull() ?: return "A whole number"
        return if (p in 1024..65535) null else "Between 1024 and 65535"
    }

    private fun pairs(s: String): String? {
        if (s.isBlank()) return null
        val bad = s.split(',').map { it.trim() }.filter { it.isNotEmpty() }.firstOrNull { p ->
            val kv = p.split('=')
            kv.size != 2 || kv[0].isBlank() || kv[1].trim().toFloatOrNull() == null
        }
        return bad?.let { "\"$it\" is not label=number" }
    }

    private fun weights(s: String): String? {
        if (s.isBlank()) return null
        return if (s.split(',').all { it.trim().toFloatOrNull()?.let { w -> w >= 0f } == true }) null else "Numbers like 2,1"
    }

    val GROUPS: List<SettingGroup> = listOf(
        SettingGroup("Source", listOf(
            SettingSpec.Choice("source", "Source", "Where the frames come from. TCP runs no detector: it only passes on events from the Pi or the desktop.",
                listOf("camera" to "Camera", "video" to "Video clip", "tcp" to "TCP events only")),
            SettingSpec.Clip("video", "Video clip", "A clip in the app's folder. Pick one, or add one from the phone."),
            SettingSpec.Toggle("loop", "Loop the clip", "Start the clip again when it ends."),
            SettingSpec.Toggle("realtime", "Play clips at real speed",
                "Like a camera: a slow phone skips frames instead of falling behind. Off runs every Nth frame as fast as detection allows, for repeatable tests."),
            SettingSpec.Range("frame_step", "Clip: detect on every Nth frame", "Used when clips do not play at real speed. 1 = every frame.",
                1f, 30f, 1f),
            SettingSpec.Range("max_fps", "Detection rate cap", "Most frames a second that are analysed. Lower runs cooler and saves battery.",
                1f, 15f, 0.5f, unit = " fps", decimals = 1),
            SettingSpec.Choice("orientation", "Phone mount", "How the phone sits in the car. Landscape works best: the models are trained on wide pictures.",
                listOf("auto" to "Auto (from the motion sensor)", "landscape" to "Landscape", "landscape_reverse" to "Landscape, turned over", "portrait" to "Portrait")),
            SettingSpec.Toggle("use_gps_speed", "Use GPS speed",
                "Helps tell stopped from moving traffic. Asks for location; the speed stays on the phone."),
        )),
        SettingGroup("Reporting", listOf(
            SettingSpec.Toggle("dry_run", "Dry run",
                "The macro walks through the report without submitting it. Leave on unless you mean to file real Waze reports."),
            SettingSpec.Range("conf", "Report confidence", "A detection must score at least this to count towards a report.",
                0.05f, 0.95f, 0.05f, decimals = 2),
            SettingSpec.Range("min_hits", "Hits needed", "Frames that must show the hazard within the window before it is reported (Police needs 2).",
                1f, 10f, 1f),
            SettingSpec.Range("window", "Window", "How many recent frames the hits are counted over.", 2f, 30f, 1f, unit = " frames"),
            SettingSpec.Range("cooldown", "Cooldown per hazard", "Least time between two reports of the same kind.",
                0f, 600f, 5f, unit = " s"),
            SettingSpec.Range("global_cooldown", "Cooldown between any reports",
                "Least time between any two reports. 60 s matches the macro, which files one a minute at most.", 0f, 600f, 5f, unit = " s"),
            SettingSpec.Text("report_class_conf", "Per-class report scores", "label=score pairs: a detection of that label needs this score to count.",
                hint = "pothole=0.55,police_vehicle=0.85", validate = ::pairs),
            SettingSpec.Toggle("waze_reader", "Skip what Waze already shows",
                "No report for an alert Waze already has nearby. Needs \"Wazeomatic Waze reader\" on in Accessibility."),
            SettingSpec.Text("target_package", "Receiving app", "Package that gets the report broadcast (the macro app). Empty = any app that listens.",
                hint = "empty = any receiver"),
            SettingSpec.Toggle("tcp", "TCP listener", "Accept events from the Pi or the desktop simulator."),
            SettingSpec.Text("tcp_port", "TCP port", "Port the listener uses.", hint = "8765", numeric = true, validate = ::port, dependsOn = "tcp"),
        )),
        SettingGroup("Detection", listOf(
            SettingSpec.Range("det_conf", "Detector threshold", "Boxes scoring below this are dropped. Never above the report confidence.",
                0.05f, 0.9f, 0.05f, decimals = 2),
            SettingSpec.Range("iou", "Box overlap merge", "Boxes overlapping more than this (IoU) count as one.", 0.1f, 0.9f, 0.05f, decimals = 2),
            SettingSpec.Choice("zoom_crop", "Zoom pass", "A second, zoomed look at the road ahead finds small and far hazards. Costs time on a slow phone.",
                listOf("true" to "Road ahead (default)", "" to "Off")),
            SettingSpec.Toggle("zoom_coco", "Zoom the vehicle model too", "Also run the zoom pass on the stock COCO model. Slower."),
            SettingSpec.Toggle("ensemble", "Fuse two hazard models", "Also runs hazard2.onnx when the build bundles it."),
            SettingSpec.Choice("fusion", "Fusion method", "How the two models' boxes are combined.",
                listOf("wbf" to "Weighted box fusion", "nms" to "Non-max suppression"), dependsOn = "ensemble"),
            SettingSpec.Text("ensemble_weights", "Model weights", "Weight per hazard model. Empty = equal.", hint = "2,1",
                validate = ::weights, dependsOn = "ensemble"),
            SettingSpec.Text("class_conf", "Per-class box floors", "label=score pairs: boxes of that label below the score are dropped.",
                hint = "pothole=0.5,cone=0.4", validate = ::pairs),
            SettingSpec.Toggle("vehicle_classifier", "Police, ambulance and fire", "Classifies the vehicles the COCO model finds (needs vehicle_cls.onnx)."),
            SettingSpec.Range("vehicle_crops", "Vehicles classified per frame", "The largest first.", 0f, 16f, 1f, dependsOn = "vehicle_classifier"),
            SettingSpec.Toggle("weather", "Weather and road surface", "Fog, rain, snow and wet-road classifiers (weather_cls / surface_cls.onnx)."),
            SettingSpec.Toggle("auto_hood", "Find the hood automatically", "Grows the ignored bottom zone from the part of the picture that never moves."),
            SettingSpec.Range("hood_frac", "Hood zone, at least", "Bottom share of the picture that is always ignored (hood, dashboard).",
                0f, 0.4f, 0.01f, percent = true),
            SettingSpec.Range("track_iou", "Track hazards by position",
                "Above 0, each hazard is followed by box overlap, so two potholes count separately.", 0f, 0.9f, 0.05f, decimals = 2, zeroLabel = "off"),
            SettingSpec.Range("track_grow", "Track search growth", "How far a tracked box may move or grow between frames (with tracking on).",
                1f, 3f, 0.1f, unit = "×", decimals = 1),
        )),
        SettingGroup("Scene", listOf(
            SettingSpec.Multi("scene_features", "Scene rules",
                "Reports for traffic, crashes, blocked lanes, broken lights and weather. Clear them all to switch the scene layer off.",
                listOf("traffic" to "Traffic", "crash" to "Crash", "lanes" to "Blocked lanes", "signals" to "Broken lights", "weather" to "Weather"),
                noneValue = "none"),
            SettingSpec.Toggle("police_stopped_only", "Police only when standing", "Report a police car only when it is standing still, not driving in traffic."),
            SettingSpec.Toggle("police_other_side", "Police on the other side", "Report a police car far on the opposite carriageway as Police > Other side."),
            SettingSpec.Toggle("shoulder_cars", "Car on shoulder", "A vehicle standing beside the road while you drive above 50 km/h."),
            SettingSpec.Toggle("shoulder_camera_speed", "Camera speed for the shoulder rule",
                "Without GPS, the camera's own speed estimate may open the 50 km/h condition.", dependsOn = "shoulder_cars"),
            SettingSpec.Choice("drive_side", "Traffic drives on the", "Which side the far carriageway is on.",
                listOf("right" to "Right (US, Europe)", "left" to "Left (UK, Japan)")),
            SettingSpec.Geometry("camera_geometry", "Camera geometry", "Field of view, mount height and where the horizon sits; used for distances on the road."),
        )),
        SettingGroup("Advanced", listOf(
            SettingSpec.Choice("ort_backend", "Model engine",
                "What runs the models. NNAPI is experimental and was slower on the test phone.",
                listOf("xnnpack" to "XNNPACK (default)", "cpu" to "Plain CPU", "nnapi" to "NNAPI", "nnapi_fp16" to "NNAPI fp16")),
            SettingSpec.Range("ort_threads", "Threads per model", "0 shares the cores between the passes that run at once.", 0f, 8f, 1f, zeroLabel = "auto"),
            SettingSpec.Range("zoom_every", "Zoom pass every Nth frame", "Auto: every frame while it fits the rate cap, else every other frame.",
                0f, 30f, 1f, zeroLabel = "auto"),
            SettingSpec.Range("max_det", "Boxes per pass", "Most boxes one model pass keeps, best first.", 0f, 200f, 5f, zeroLabel = "no cap"),
            SettingSpec.Range("max_per_class", "Boxes of one class per pass", "Most boxes of a single class one pass keeps.", 0f, 100f, 1f, zeroLabel = "no cap"),
            SettingSpec.Toggle("rect_input", "Run at the picture's shape", "Open-shape models run at 640x384 for a wide frame instead of a padded square: faster, same pixels."),
            SettingSpec.Text("class_map", "Extra class mapping", "For testing: send a model label as a report type.", hint = "car=HAZARD_ON_ROAD_OBJECT"),
        )),
    )

    val ALL: List<SettingSpec> get() = GROUPS.flatMap { it.items }

    /** The groups with only the keys [known] holds (a key another branch adds shows up once it is merged). */
    fun visible(known: Set<String> = Settings.KNOWN_KEYS): List<SettingGroup> =
        GROUPS.map { g -> SettingGroup(g.title, g.items.filter { it.key in known }) }.filter { it.items.isNotEmpty() }
}
