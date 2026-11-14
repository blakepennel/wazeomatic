package com.wazeomatic.detector

import org.json.JSONObject
import java.math.BigDecimal
import java.util.Locale

// Kotlin mirror of wazeomatic/schema.py. hazard_type strings are the contract with
// the MacroDroid macro, keep them identical to the Python side.

const val SCHEMA_VERSION = 1
const val INTENT_ACTION = "com.wazeomatic.HAZARD_DETECTED"

data class HazardType(
    val key: String,
    val wazeCategory: String,
    val wazeSubcategory: String,
    val description: String,
)

object Schema {
    val HAZARD_TYPES: Map<String, HazardType> = listOf(
        HazardType("HAZARD_ON_ROAD_POTHOLE", "Hazard", "Pothole", "Pothole / road damage"),
        HazardType("HAZARD_ON_ROAD_CONSTRUCTION", "Hazard", "Construction", "Cones, work-zone signs, delineators"),
        HazardType("HAZARD_ON_ROAD_OBJECT", "Hazard", "Object", "Debris or object on the road"),
        HazardType("HAZARD_ON_ROAD_TRAFFIC_LIGHT", "Hazard", "Broken traffic light",
            "Signal head in view with no lamp lit for several seconds"),
        HazardType("HAZARD_SHOULDER_CAR_STOPPED", "Hazard", "Car on shoulder", "Vehicle stopped on the shoulder"),
        HazardType("HAZARD_SHOULDER_ANIMALS", "Hazard", "", "Animal on or near the road (plain Hazard; Waze removed the Animals subtype)"),
        HazardType("POLICE", "Police", "", "Police vehicle visible"),
        HazardType("POLICE_OTHER_SIDE", "Police", "Other side", "Police vehicle on the opposite carriageway"),
        HazardType("CRASH", "Crash", "", "Crash / collision"),
        HazardType("CRASH_OTHER_SIDE", "Crash", "Other side", "Crash scene on the opposite carriageway"),
        HazardType("TRAFFIC_HEAVY", "Traffic", "Heavy", "Slow, dense traffic for a sustained stretch"),
        HazardType("TRAFFIC_STANDSTILL", "Traffic", "Standstill", "Traffic not moving, not at a red light"),
        HazardType("WEATHER_FOG", "Bad weather", "Fog", "Fog limiting visibility"),
        HazardType("WEATHER_FLOOD", "Bad weather", "Flood", "Standing water across the road"),
        HazardType("WEATHER_SLIPPERY", "Bad weather", "Slippery road", "Wet road in heavy rain"),
        HazardType("WEATHER_ICE", "Bad weather", "Icy road", "Ice on the road surface"),
        HazardType("WEATHER_UNPLOWED", "Bad weather", "Unplowed road", "Snow covering the lanes"),
        HazardType("BLOCKED_LANE_LEFT", "Blocked lane", "Left lane", "Left lane blocked by a work zone or vehicle"),
        HazardType("BLOCKED_LANE_CENTER", "Blocked lane", "Center lane", "Center lane blocked"),
        HazardType("BLOCKED_LANE_RIGHT", "Blocked lane", "Right lane", "Right lane blocked"),
    ).associateBy { it.key }

    val DEFAULT_CLASS_TO_HAZARD: Map<String, String> = linkedMapOf(
        // custom hazard model
        "pothole" to "HAZARD_ON_ROAD_POTHOLE",
        "cone" to "HAZARD_ON_ROAD_CONSTRUCTION",
        "construction_sign" to "HAZARD_ON_ROAD_CONSTRUCTION",
        "delineator" to "HAZARD_ON_ROAD_CONSTRUCTION",
        "barricade" to "HAZARD_ON_ROAD_CONSTRUCTION",
        "worker" to "HAZARD_ON_ROAD_CONSTRUCTION",
        "police_vehicle" to "POLICE",          // the vehicle classifier's label for a police car
        "debris" to "HAZARD_ON_ROAD_OBJECT",
        "deer" to "HAZARD_SHOULDER_ANIMALS",
        // COCO classes used from the stock detector
        "dog" to "HAZARD_SHOULDER_ANIMALS",
        "horse" to "HAZARD_SHOULDER_ANIMALS",
        "sheep" to "HAZARD_SHOULDER_ANIMALS",
        "cow" to "HAZARD_SHOULDER_ANIMALS",
        "bear" to "HAZARD_SHOULDER_ANIMALS",
        // the stock COCO model calls a deer a giraffe now and then; on a North American road these are
        // deer, elk or moose
        "giraffe" to "HAZARD_SHOULDER_ANIMALS",
        "zebra" to "HAZARD_SHOULDER_ANIMALS",
        "elephant" to "HAZARD_SHOULDER_ANIMALS",
        // scene-level labels (wazeomatic/scene.py): one whole-frame detection per frame while a condition holds
        "scene_traffic_heavy" to "TRAFFIC_HEAVY",
        "scene_traffic_standstill" to "TRAFFIC_STANDSTILL",
        "scene_fog" to "WEATHER_FOG",
        "scene_flood" to "WEATHER_FLOOD",
        "scene_slippery" to "WEATHER_SLIPPERY",
        "scene_ice" to "WEATHER_ICE",
        "scene_unplowed" to "WEATHER_UNPLOWED",
        "scene_crash" to "CRASH",
        "scene_crash_other_side" to "CRASH_OTHER_SIDE",
        "scene_police_other_side" to "POLICE_OTHER_SIDE",
        "scene_lane_left" to "BLOCKED_LANE_LEFT",
        "scene_lane_center" to "BLOCKED_LANE_CENTER",
        "scene_lane_right" to "BLOCKED_LANE_RIGHT",
        "scene_dark_signal" to "HAZARD_ON_ROAD_TRAFFIC_LIGHT",
        // car/truck/bus only become a hazard through the stopped-vehicle heuristic
    )
}

/** Normalized center xywh in the source frame. */
data class BBox(val cx: Float, val cy: Float, val w: Float, val h: Float)

data class Detection(
    val label: String,
    val confidence: Float,
    val bbox: BBox,
    val model: String = "hazard",
)

fun nowSeconds(): Double = System.currentTimeMillis() / 1000.0

data class HazardEvent(
    val hazardType: String,
    val confidence: Double,
    val source: String = "phone",
    val ts: Double = nowSeconds(),
    val frameIndex: Int? = null,
    val bbox: BBox? = null,
    val label: String? = null,
    val lat: Double? = null,
    val lon: Double? = null,
    val schema: Int = SCHEMA_VERSION,
) {
    val waze: HazardType get() = Schema.HAZARD_TYPES.getValue(hazardType)

    /** Same keys and order as HazardEvent.to_json() in schema.py. */
    fun toJson(): String {
        val sb = StringBuilder("{")
        fun key(k: String) {
            if (sb.length > 1) sb.append(',')
            sb.append(quote(k)).append(':')
        }
        key("hazard_type"); sb.append(quote(hazardType))
        key("confidence"); sb.append(num(confidence))
        key("source"); sb.append(quote(source))
        key("ts"); sb.append(num(ts))
        key("frame_index"); sb.append(frameIndex?.toString() ?: "null")
        key("bbox")
        if (bbox == null) {
            sb.append("null")
        } else {
            sb.append('[')
            sb.append(listOf(bbox.cx, bbox.cy, bbox.w, bbox.h).joinToString(",") { num(it.toString().toDouble()) })
            sb.append(']')
        }
        key("label"); sb.append(label?.let { quote(it) } ?: "null")
        key("lat"); sb.append(lat?.let { num(it) } ?: "null")
        key("lon"); sb.append(lon?.let { num(it) } ?: "null")
        key("schema"); sb.append(schema)
        key("waze_category"); sb.append(quote(waze.wazeCategory))
        key("waze_subcategory"); sb.append(quote(waze.wazeSubcategory))
        return sb.append('}').toString()
    }

    /** String extras for the broadcast (MacroDroid reads strings). */
    fun intentExtras(): Map<String, String> = linkedMapOf(
        "hazard_type" to hazardType,
        "confidence" to String.format(Locale.US, "%.3f", confidence),
        "ts" to String.format(Locale.US, "%.3f", ts),
        "source" to source,
        "waze_category" to waze.wazeCategory,
        "waze_subcategory" to waze.wazeSubcategory,
    )

    companion object {
        /** Parses one JSON line as produced by schema.py (extra keys are ignored). */
        fun fromJson(line: String, defaultSource: String = "pi"): HazardEvent {
            val o = JSONObject(line)
            val type = o.getString("hazard_type")
            require(type in Schema.HAZARD_TYPES) { "unknown hazard_type $type" }
            fun has(k: String) = o.has(k) && !o.isNull(k)
            val bbox = if (has("bbox")) {
                val a = o.getJSONArray("bbox")
                BBox(a.getDouble(0).toFloat(), a.getDouble(1).toFloat(), a.getDouble(2).toFloat(), a.getDouble(3).toFloat())
            } else null
            return HazardEvent(
                hazardType = type,
                confidence = if (has("confidence")) o.getDouble("confidence") else 1.0,
                source = if (has("source")) o.getString("source") else defaultSource,
                ts = if (has("ts")) o.getDouble("ts") else nowSeconds(),
                frameIndex = if (has("frame_index")) o.getInt("frame_index") else null,
                bbox = bbox,
                label = if (has("label")) o.getString("label") else null,
                lat = if (has("lat")) o.getDouble("lat") else null,
                lon = if (has("lon")) o.getDouble("lon") else null,
                schema = if (has("schema")) o.getInt("schema") else SCHEMA_VERSION,
            )
        }

        private fun num(d: Double): String =
            if (d.isNaN() || d.isInfinite()) "null" else BigDecimal(d.toString()).toPlainString()

        private fun quote(s: String): String {
            val sb = StringBuilder("\"")
            for (c in s) {
                when {
                    c == '"' -> sb.append("\\\"")
                    c == '\\' -> sb.append("\\\\")
                    c == '\n' -> sb.append("\\n")
                    c == '\r' -> sb.append("\\r")
                    c == '\t' -> sb.append("\\t")
                    c < ' ' -> sb.append(String.format(Locale.US, "\\u%04x", c.code))
                    else -> sb.append(c)
                }
            }
            return sb.append('"').toString()
        }
    }
}
