package com.wazeomatic.mockwaze

/** One round tile: label under an icon. */
data class Option(val label: String, val icon: Int)

/**
 * One top-level tile of the "What do you see?" sheet. options[0] is the category itself and is
 * preselected when its sheet opens (that is what Waze does: tap Hazard, then Report sends a plain
 * Hazard unless you pick a more specific tile).
 */
data class Category(val label: String, val icon: Int, val sheetTitle: String, val options: List<Option>) {
    val hasSubmenu: Boolean get() = options.size > 1
    val subLabels: List<String> get() = options.drop(1).map { it.label }
}

/**
 * A full path through the report menu. item is the specific tile ("" = the category itself).
 * Same strings as waze_subcategory in schema.py.
 */
data class MenuPath(val category: String, val item: String = "") {
    val subcategory: String get() = item
}

object WazeMenu {
    private fun o(label: String, icon: Int) = Option(label, icon)

    // Tile order of the Android report sheet after the February 2024 redesign ("What do you
    // see?"), from the Waze help pages and press screenshots: Traffic, Police, Crash / Hazard,
    // Bad weather, Blocked lane / Map issue, Closure, Gas prices / and a fourth row. The fourth
    // row moves around between versions and regions; re-check it on the real phone before
    // relying on coordinates. Each category opens a second sheet with the category itself
    // preselected plus its specific tiles. The hazard tiles are the ones documented on
    // support.google.com/waze/answer/13739290; the other sub-tiles follow press screenshots
    // (traffic, police, crash, blocked lane) or are approximations (bad weather).
    val categories: List<Category> = listOf(
        Category("Traffic", R.drawable.ic_cat_traffic, "Report traffic", listOf(
            o("Traffic", R.drawable.ic_cat_traffic),
            o("Heavy", R.drawable.ic_sub_traffic_heavy),
            o("Standstill", R.drawable.ic_sub_traffic_standstill))),
        Category("Police", R.drawable.ic_cat_police, "Report police", listOf(
            o("Police", R.drawable.ic_cat_police),
            o("Hidden", R.drawable.ic_sub_police_hidden),
            o("Other side", R.drawable.ic_sub_police_other_side))),
        Category("Crash", R.drawable.ic_cat_crash, "Report a crash", listOf(
            o("Crash", R.drawable.ic_cat_crash),
            o("Pile-up", R.drawable.ic_sub_crash_pileup),
            o("Other side", R.drawable.ic_sub_crash_other_side))),
        Category("Hazard", R.drawable.ic_cat_hazard, "Report a hazard", listOf(
            o("Hazard", R.drawable.ic_sub_hazard),
            o("Construction", R.drawable.ic_sub_construction),
            o("Car on shoulder", R.drawable.ic_sub_car_shoulder),
            o("Broken traffic light", R.drawable.ic_sub_traffic_light),
            o("Pothole", R.drawable.ic_sub_pothole),
            o("Object", R.drawable.ic_sub_object))),
        Category("Bad weather", R.drawable.ic_cat_weather, "Report bad weather", listOf(
            o("Bad weather", R.drawable.ic_cat_weather),
            o("Fog", R.drawable.ic_sub_weather_fog),
            o("Hail", R.drawable.ic_sub_weather_hail),
            o("Flood", R.drawable.ic_sub_weather_flood),
            o("Ice", R.drawable.ic_sub_weather_ice),
            o("Snow", R.drawable.ic_sub_weather_snow))),
        Category("Blocked lane", R.drawable.ic_cat_blocked_lane, "Report blocked lane", listOf(
            o("Blocked lane", R.drawable.ic_cat_blocked_lane),
            o("Left lane", R.drawable.ic_sub_lane_left),
            o("Right lane", R.drawable.ic_sub_lane_right),
            o("Center lane", R.drawable.ic_sub_lane_center))),
        Category("Map issue", R.drawable.ic_cat_map_issue, "Report a map issue", listOf(
            o("Map issue", R.drawable.ic_cat_map_issue))),
        Category("Closure", R.drawable.ic_cat_closure, "Report a closure", listOf(
            o("Closure", R.drawable.ic_cat_closure))),
        Category("Gas prices", R.drawable.ic_cat_gas, "Report gas prices", listOf(
            o("Gas prices", R.drawable.ic_cat_gas))),
        Category("Roadside help", R.drawable.ic_cat_roadside, "Roadside help", listOf(
            o("Roadside help", R.drawable.ic_cat_roadside))),
        Category("Map chat", R.drawable.ic_cat_map_chat, "Map chat", listOf(
            o("Map chat", R.drawable.ic_cat_map_chat))),
        Category("Place", R.drawable.ic_cat_place, "Add a place", listOf(
            o("Place", R.drawable.ic_cat_place))),
    )

    /** The specific hazard tiles (everything under Hazard except Hazard itself). */
    val hazardSubtypes: List<String> get() = category("Hazard")!!.subLabels

    /** hazard_type (contract key from schema.py) -> menu path. */
    val hazardTypes: Map<String, MenuPath> = mapOf(
        "HAZARD_ON_ROAD_POTHOLE" to MenuPath("Hazard", "Pothole"),
        "HAZARD_ON_ROAD_CONSTRUCTION" to MenuPath("Hazard", "Construction"),
        "HAZARD_ON_ROAD_OBJECT" to MenuPath("Hazard", "Object"),
        "HAZARD_SHOULDER_CAR_STOPPED" to MenuPath("Hazard", "Car on shoulder"),
        "HAZARD_SHOULDER_ANIMALS" to MenuPath("Hazard", ""),
        "POLICE" to MenuPath("Police"),
        "CRASH" to MenuPath("Crash"),
    )

    fun category(label: String): Category? = categories.firstOrNull { it.label == label }

    /**
     * Resolve the menu path for an incoming event. hazard_type wins; if it is unknown we try
     * waze_category / waze_subcategory so new hazard keys still work if those extras are valid.
     */
    fun pathFor(hazardType: String?, wazeCategory: String?, wazeSub: String?): MenuPath? {
        hazardTypes[hazardType ?: ""]?.let { return it }
        val cat = category(wazeCategory ?: "") ?: return null
        val sub = (wazeSub ?: "").trim()
        return when {
            sub.isEmpty() || sub == cat.label -> MenuPath(cat.label)
            sub in cat.subLabels -> MenuPath(cat.label, sub)
            else -> null
        }
    }
}
