package com.wazeomatic.detector

import java.util.Locale

// Port of wazeomatic/waze_reader.py: what the Waze screen says, as text. Pure Kotlin (no Android
// imports) so it runs in the JVM unit tests; WazeWatcherService feeds it the accessibility tree.
// Patterns, their order and the grouping rules are the same as the Python file; change both together.

/** One on-screen text with its bounds in screen pixels (left, top, right, bottom). */
data class TextNode(val text: String, val left: Int, val top: Int, val right: Int, val bottom: Int)

/** One alert Waze shows (a card of another driver's report, or a "Still there?" prompt). */
class WazeAlert(
    /** Schema key, null when the wording is not one we know (kept so the log shows what to add). */
    val hazardType: String?,
    /** The texts of the card, joined with " · ". */
    val text: String,
    distanceM: Double?,
    /** "alert" or "still_there". */
    val kind: String = "alert",
    ts: Double = nowSeconds(),
) {
    @Volatile var distanceM: Double? = distanceM
        internal set
    @Volatile var ts: Double = ts
        internal set

    val category: String get() = hazardType?.let { Schema.HAZARD_TYPES[it]?.wazeCategory } ?: ""

    /** Same keys as WazeAlert.to_dict() in waze_reader.py. */
    fun toJson(): String {
        val d = distanceM
        return "{" +
            "\"hazard_type\":" + (hazardType?.let { WazeText.quote(it) } ?: "null") + "," +
            "\"category\":" + WazeText.quote(category) + "," +
            "\"text\":" + WazeText.quote(text) + "," +
            "\"distance_m\":" + (if (d == null) "null" else String.format(Locale.US, "%.1f", d)) + "," +
            "\"kind\":" + WazeText.quote(kind) + "," +
            "\"ts\":" + String.format(Locale.US, "%.3f", ts) +
            "}"
    }

    override fun toString(): String = toJson()
}

/** One look at the screen. */
data class WazeScreen(
    val packageName: String = "",
    val alerts: List<WazeAlert> = emptyList(),
    val reportSheetOpen: Boolean = false,
    val texts: List<String> = emptyList(),
)

object WazeText {
    val WAZE_PACKAGES = listOf("com.waze", "com.wazeomatic.mockwaze")

    /** (pattern, hazard key). First match wins: specific phrases come before the bare category words. */
    val ALERT_PATTERNS: List<Pair<String, String>> = listOf(
        "police.*other side|other side.*police" to "POLICE_OTHER_SIDE",
        "\\bpolice\\b|speed trap|\\bcops?\\b" to "POLICE",
        "(crash|accident).*other side|other side.*(crash|accident)" to "CRASH_OTHER_SIDE",
        "pile-?up|\\bcrash\\b|\\baccident\\b|collision" to "CRASH",
        "standstill|stopped traffic|traffic (is )?stopped" to "TRAFFIC_STANDSTILL",
        "heavy traffic|traffic jam|\\bjam\\b|slow traffic" to "TRAFFIC_HEAVY",
        "pothole" to "HAZARD_ON_ROAD_POTHOLE",
        "construction|road ?work|work zone" to "HAZARD_ON_ROAD_CONSTRUCTION",
        "(car|vehicle) (stopped )?on (the )?shoulder|stopped (car|vehicle)|car on shoulder" to "HAZARD_SHOULDER_CAR_STOPPED",
        "(broken|faulty|dark) traffic light|traffic light (out|not working|broken)" to "HAZARD_ON_ROAD_TRAFFIC_LIGHT",
        "object on (the )?road|\\bdebris\\b|\\bobject\\b|lost load" to "HAZARD_ON_ROAD_OBJECT",
        "\\banimals?\\b|\\bdeer\\b|roadkill" to "HAZARD_SHOULDER_ANIMALS",
        "\\bfog\\b|foggy" to "WEATHER_FOG",
        "\\bflood" to "WEATHER_FLOOD",
        "\\bic(e|y)\\b" to "WEATHER_ICE",
        "unplowed|snow on (the )?road|\\bsnow\\b" to "WEATHER_UNPLOWED",
        "slippery" to "WEATHER_SLIPPERY",
        "left lane (closed|blocked)|blocked.*left lane" to "BLOCKED_LANE_LEFT",
        "right lane (closed|blocked)|blocked.*right lane" to "BLOCKED_LANE_RIGHT",
        "(center|centre|middle) lane (closed|blocked)|blocked.*(center|centre|middle) lane" to "BLOCKED_LANE_CENTER",
    )
    private val COMPILED = ALERT_PATTERNS.map { (p, k) -> Regex(p, RegexOption.IGNORE_CASE) to k }

    /** Texts that mark a card as somebody else's report (an alert), not a tile of our own report sheet. */
    private val ALERT_CONTEXT = Regex(
        "reported|ahead|still there|\\bago\\b|\\bthanks\\b|not there|on your route|" +
            "\\b\\d+(\\.\\d+)?\\s?(mi|ft|km|m|yd)\\b",
        RegexOption.IGNORE_CASE,
    )
    private val STILL_THERE = Regex("still there\\??|is it still there|not there", RegexOption.IGNORE_CASE)
    // whole-text titles of our own report sheet; anchored so the "Report button" on the map is not one
    private val REPORT_SHEET = Regex(
        "^what do you see\\??$|^report (a |an )?(traffic|police|crash|hazard|bad weather|" +
            "blocked lane|map issue|closure|gas prices)$",
        RegexOption.IGNORE_CASE,
    )
    private val REPORTED = Regex("reported", RegexOption.IGNORE_CASE)

    private val DIST = Regex("(\\d+(?:[.,]\\d+)?)\\s?(mi|miles?|ft|feet|yd|km|m|meters?|metres?)\\b", RegexOption.IGNORE_CASE)
    private val TO_M = mapOf(
        "mi" to 1609.344, "mile" to 1609.344, "miles" to 1609.344, "ft" to 0.3048, "feet" to 0.3048, "yd" to 0.9144,
        "km" to 1000.0, "m" to 1.0, "meter" to 1.0, "meters" to 1.0, "metre" to 1.0, "metres" to 1.0,
    )

    /** "0.3 mi" -> 482.8, "500 ft" -> 152.4, "1,2 km" -> 1200. Null if the text carries no distance. */
    fun parseDistance(text: String): Double? {
        val m = DIST.find(text) ?: return null
        val v = m.groupValues[1].replace(',', '.').toDoubleOrNull() ?: return null
        return v * (TO_M[m.groupValues[2].lowercase(Locale.US)] ?: return null)
    }

    fun classifyText(text: String): String? = COMPILED.firstOrNull { it.first.containsMatchIn(text) }?.second

    fun isReportSheet(text: String): Boolean = REPORT_SHEET.containsMatchIn(text)

    /**
     * Group on-screen texts into cards by vertical proximity: an alert card is a title, a distance and a
     * "reported ... ago" line stacked within a few dozen pixels. Nodes taller than [maxHeightFrac] of the
     * screen (the map, a full-screen container with a description) are dropped first: they would swallow
     * every other text into one card. Screen height = the lowest bottom edge among the nodes.
     */
    fun groupCards(nodes: List<TextNode>, gapPx: Int = 40, maxHeightFrac: Double = 0.33): List<List<String>> {
        val cards = ArrayList<ArrayList<String>>()
        var lastBottom: Int? = null
        val screenH = nodes.maxOfOrNull { it.bottom } ?: 0
        val kept = nodes.filter { screenH <= 0 || (it.bottom - it.top) <= maxHeightFrac * screenH }
        for (n in kept.sortedWith(compareBy<TextNode>({ it.top }, { it.left }))) {
            val lb = lastBottom
            if (lb == null || n.top - lb > gapPx) cards.add(ArrayList())
            cards.last().add(n.text)
            lastBottom = if (lb == null) n.bottom else maxOf(lb, n.bottom)
        }
        return cards
    }

    /**
     * Turn groups of texts into alerts. A group is an alert when it names a report type and carries alert
     * context (a distance, "reported", "ahead", "Still there?"), so tiles of the report sheet are not alerts.
     */
    fun parseCards(cards: Iterable<List<String>>, now: Double = nowSeconds()): List<WazeAlert> {
        val out = ArrayList<WazeAlert>()
        for (card in cards) {
            val joined = card.map { it.trim() }.filter { it.isNotEmpty() }.joinToString(" · ")
            if (joined.isEmpty() || !ALERT_CONTEXT.containsMatchIn(joined)) continue
            // the title line names the type; try every line before the joined text
            val key = card.firstNotNullOfOrNull { classifyText(it) } ?: classifyText(joined)
            val kind = if (STILL_THERE.containsMatchIn(joined)) "still_there" else "alert"
            // a distance alone is the speedometer or a turn instruction
            if (key == null && kind == "alert" && !REPORTED.containsMatchIn(joined)) continue
            out.add(WazeAlert(key, joined, parseDistance(joined), kind, now))
        }
        return out
    }

    /**
     * One screen from the nodes of a Waze package (the accessibility tree, or a uiautomator dump).
     * While our own report sheet is open its tiles are not alerts, so nothing else is read from that screen.
     */
    fun parseNodes(packageName: String, nodes: List<TextNode>, now: Double = nowSeconds()): WazeScreen {
        val texts = nodes.map { it.text }
        val sheet = texts.any { isReportSheet(it) }
        val alerts = if (sheet) emptyList() else parseCards(groupCards(nodes), now)
        return WazeScreen(packageName, alerts, sheet, texts)
    }

    /**
     * Does an alert of [alertKey] already stand for an event of [eventKey]? Same key, always. Traffic and
     * Bad weather are one condition per stretch of road whatever the sub-tile, so any alert of the category
     * covers; Police and Crash alerts cover either side (Waze shows one icon).
     */
    fun sameReport(alertKey: String, eventKey: String): Boolean {
        if (alertKey == eventKey) return true
        val a = Schema.HAZARD_TYPES[alertKey] ?: return false
        val e = Schema.HAZARD_TYPES[eventKey] ?: return false
        if (a.wazeCategory != e.wazeCategory) return false
        return a.wazeCategory in setOf("Traffic", "Bad weather", "Police", "Crash")
    }

    internal fun quote(s: String): String {
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

/**
 * What Waze has shown recently, with expiry. Thread safe: the accessibility service writes, the
 * pipeline thread reads. Same rules as WazeAlerts in waze_reader.py.
 */
class WazeAlerts(
    val ttlS: Double = 120.0,
    val maxDistanceM: Double = 1200.0,
    private val clock: () -> Double = ::nowSeconds,
) {
    private val lock = Any()
    private val alerts = ArrayList<WazeAlert>()

    @Volatile var reportSheetOpen = false
        private set
    /** Last time a Waze screen was read at all (0 = never). */
    @Volatile var lastSeen = 0.0
        private set

    /** Merge one screen read. Returns the alerts that are new (not seen within the TTL). */
    fun update(screen: WazeScreen): List<WazeAlert> {
        val now = clock()
        val new = ArrayList<WazeAlert>()
        synchronized(lock) {
            if (screen.packageName.isNotEmpty()) lastSeen = now
            reportSheetOpen = screen.reportSheetOpen
            expire(now)
            for (a in screen.alerts) {
                a.ts = now
                val same = alerts.firstOrNull { it.hazardType == a.hazardType && it.text == a.text }
                if (same != null) {
                    same.ts = now
                    same.distanceM = a.distanceM
                } else {
                    alerts.add(a)
                    new.add(a)
                }
            }
        }
        return new
    }

    /** Caller holds lock. */
    private fun expire(now: Double) {
        alerts.removeAll { now - it.ts > ttlS }
    }

    /** The alert that already reports [hazardType] nearby, if any (most recently seen first). */
    fun covering(hazardType: String): WazeAlert? {
        val now = clock()
        synchronized(lock) {
            expire(now)
            for (a in alerts.sortedByDescending { it.ts }) {
                val key = a.hazardType ?: continue
                if (!WazeText.sameReport(key, hazardType)) continue
                val d = a.distanceM
                if (d != null && d > maxDistanceM) continue
                return a
            }
        }
        return null
    }

    /** Recent alerts, nearest first; alerts without a distance last. */
    fun ahead(): List<WazeAlert> {
        val now = clock()
        synchronized(lock) {
            expire(now)
            return alerts.sortedWith(compareBy<WazeAlert>({ it.distanceM == null }, { it.distanceM ?: 0.0 }))
        }
    }

    fun clear() {
        synchronized(lock) { alerts.clear() }
    }

    fun summary(limit: Int = 3): String {
        val items = ahead().take(limit)
        if (items.isEmpty()) return ""
        return "Waze ahead: " + items.joinToString(", ") { a ->
            val h = a.hazardType?.let { Schema.HAZARD_TYPES[it] }
            val name = h?.wazeCategory ?: "?"
            val sub = h?.wazeSubcategory ?: ""
            val d = a.distanceM?.let { String.format(Locale.US, " %,.0f m", it) } ?: ""
            name + (if (sub.isNotEmpty()) " $sub" else "") + d
        }
    }

    companion object {
        /** The process-wide instance: WazeWatcherService writes it, the detector's ReportManager reads it. */
        val shared = WazeAlerts()
    }
}
