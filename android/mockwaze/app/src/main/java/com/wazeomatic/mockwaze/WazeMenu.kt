package com.wazeomatic.mockwaze

/** One tile in the top-level report sheet. */
data class Category(val label: String, val color: Int, val glyph: String) {
    val hasSubmenu: Boolean get() = label == "Hazard"
}

/**
 * A full path through the report menu. item is the Hazard subtype ("" = plain Hazard, or a
 * one-step category like Police / Crash). Same strings as waze_subcategory in schema.py.
 */
data class MenuPath(val category: String, val item: String = "") {
    val subcategory: String get() = item
}

object WazeMenu {
    // Tile order of the Android report sheet after the 2024 redesign. Checked against Waze
    // help pages and press screenshots (9to5google / piunikaweb, Feb 2024): the first row
    // Traffic, Police, Crash is certain, and Hazard / Closure / Blocked lane / Bad weather
    // follow. The order of the last tiles (Gas prices, Map chat, Map issue, Place, Roadside
    // help) moves around between app versions and regions, so the macro should tap by text
    // where possible and the coordinates must be re-checked on the real phone.
    // Real Waze also has a second step for Police (Visible / Hidden / Other side) and Crash
    // (Minor / Major / Other side); schema.py reports those with no subcategory, so the mock
    // keeps them as a one-step confirm.
    val categories = listOf(
        Category("Traffic", 0xFFE5484D.toInt(), "T"),
        Category("Police", 0xFF3B82F6.toInt(), "P"),
        Category("Crash", 0xFFD9466F.toInt(), "C"),
        Category("Hazard", 0xFFF59E0B.toInt(), "!"),
        Category("Closure", 0xFFB91C1C.toInt(), "X"),
        Category("Blocked lane", 0xFFEA580C.toInt(), "L"),
        Category("Bad weather", 0xFF0EA5E9.toInt(), "W"),
        Category("Gas prices", 0xFF16A34A.toInt(), "G"),
        Category("Map chat", 0xFF2563EB.toInt(), "M"),
        Category("Map issue", 0xFF6B7280.toInt(), "?"),
        Category("Place", 0xFF7C3AED.toInt(), "+"),
        Category("Roadside help", 0xFF0284C7.toInt(), "H"),
    )

    // Hazard subtypes, a flat list in the order of the Waze help page "Report hazards"
    // (support.google.com/waze/answer/13739290). The older grouped menu (On road / Shoulder /
    // Weather, with Animals and Roadkill) is gone. Hazard is preselected when the sheet opens,
    // so Send without picking a subtype reports a plain "Hazard" (schema.py uses that for animals).
    val hazardSubtypes = listOf("Construction", "Car on shoulder", "Broken traffic light", "Pothole", "Object")

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
        if (!cat.hasSubmenu) return MenuPath(cat.label)
        val sub = (wazeSub ?: "").trim()
        return when {
            sub.isEmpty() -> MenuPath(cat.label)
            sub in hazardSubtypes -> MenuPath(cat.label, sub)
            else -> null
        }
    }
}
