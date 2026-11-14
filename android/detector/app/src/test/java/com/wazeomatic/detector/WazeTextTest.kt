package com.wazeomatic.detector

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Mirrors tests/test_waze_reader.py (the XML cases go through parseNodes with the same nodes). */
class WazeTextTest {
    private var now = 0.0

    @Test
    fun distances() {
        assertEquals(482.8, WazeText.parseDistance("Police · 0.3 mi")!!, 0.5)
        assertEquals(152.4, WazeText.parseDistance("500 ft ahead")!!, 0.1)
        assertEquals(1200.0, WazeText.parseDistance("1,2 km")!!, 1e-9)
        assertEquals(200.0, WazeText.parseDistance("200 m")!!, 1e-9)
        assertNull(WazeText.parseDistance("Reported 5 min ago"))
    }

    @Test
    fun classifyKnownWording() {
        assertEquals("POLICE", WazeText.classifyText("Police reported ahead"))
        assertEquals("POLICE_OTHER_SIDE", WazeText.classifyText("Police on the other side"))
        assertEquals("CRASH", WazeText.classifyText("Crash reported"))
        assertEquals("HAZARD_SHOULDER_CAR_STOPPED", WazeText.classifyText("Car stopped on shoulder"))
        assertEquals("HAZARD_ON_ROAD_POTHOLE", WazeText.classifyText("Pothole"))
        assertEquals("TRAFFIC_HEAVY", WazeText.classifyText("Heavy traffic"))
        assertEquals("WEATHER_FOG", WazeText.classifyText("Fog"))
        assertEquals("BLOCKED_LANE_RIGHT", WazeText.classifyText("Right lane closed"))
        assertNull(WazeText.classifyText("Where to?"))
        for ((_, key) in WazeText.ALERT_PATTERNS) assertTrue(key, key in Schema.HAZARD_TYPES)
    }

    @Test
    fun cardsNeedAlertContext() {
        // a report-sheet tile named Police is not an alert; an alert card is
        val alerts = WazeText.parseCards(listOf(listOf("Police"), listOf("Police", "0.4 mi", "Reported 3 min ago")), now = 10.0)
        assertEquals(1, alerts.size)
        assertEquals("POLICE", alerts[0].hazardType)
        assertEquals(643.7, alerts[0].distanceM!!, 1.0)
    }

    @Test
    fun stillTherePromptAndUnknownWordingKept() {
        val alerts = WazeText.parseCards(
            listOf(listOf("Pothole", "Still there?", "Yes", "No"), listOf("Sinkhole reported", "200 ft")), now = 0.0
        )
        assertEquals("still_there", alerts[0].kind)
        assertEquals("HAZARD_ON_ROAD_POTHOLE", alerts[0].hazardType)
        assertNull(alerts[1].hazardType)              // kept, so the log shows wording to add
    }

    @Test
    fun speedometerIsNotAnAlert() {
        assertTrue(WazeText.parseCards(listOf(listOf("35", "km/h"), listOf("0.5 mi", "Turn right")), now = 0.0).isEmpty())
    }

    @Test
    fun groupingByVerticalGap() {
        val nodes = listOf(
            TextNode("Police", 0, 100, 300, 140), TextNode("0.4 mi", 0, 150, 200, 180), TextNode("Where to?", 0, 900, 400, 950)
        )
        assertEquals(listOf(listOf("Police", "0.4 mi"), listOf("Where to?")), WazeText.groupCards(nodes))
    }

    private val mockNodes = listOf(
        TextNode("Police", 100, 300, 600, 360),
        TextNode("0.3 mi", 100, 370, 300, 410),
        TextNode("Reported 2 min ago", 100, 420, 700, 460),
        TextNode("Where to?", 100, 2600, 900, 2700),
    )

    @Test
    fun screenOfTheMock() {
        val screen = WazeText.parseNodes("com.wazeomatic.mockwaze", mockNodes, now = 5.0)
        assertEquals(listOf("POLICE"), screen.alerts.map { it.hazardType })
        assertEquals(482.8, screen.alerts[0].distanceM!!, 0.5)
        assertFalse(screen.reportSheetOpen)
    }

    @Test
    fun reportSheetSuppressesAlerts() {
        val nodes = mockNodes.map { if (it.text == "Where to?") it.copy(text = "What do you see?") else it }
        val screen = WazeText.parseNodes("com.wazeomatic.mockwaze", nodes, now = 0.0)
        assertTrue(screen.reportSheetOpen)
        assertTrue(screen.alerts.isEmpty())
    }

    @Test
    fun reportButtonIsNotAnOpenSheet() {
        val nodes = mockNodes.map { if (it.text == "Where to?") it.copy(text = "Report button") else it }
        val screen = WazeText.parseNodes("com.wazeomatic.mockwaze", nodes, now = 0.0)
        assertFalse(screen.reportSheetOpen)
        assertEquals(listOf("POLICE"), screen.alerts.map { it.hazardType })
        for (title in listOf("What do you see?", "Report a hazard", "Report police", "Report bad weather")) {
            val sheet = mockNodes.map { if (it.text == "Where to?") it.copy(text = title) else it }
            assertTrue(title, WazeText.parseNodes("com.wazeomatic.mockwaze", sheet, now = 0.0).reportSheetOpen)
        }
    }

    @Test
    fun fullScreenNodeDoesNotMergeCards() {
        // the map container with a description, as the first node of the dump
        val nodes = listOf(TextNode("Map", 0, 0, 1280, 2856)) +
            mockNodes.map { if (it.text == "Where to?") it.copy(text = "35 MPH") else it }
        val alerts = WazeText.parseNodes("com.wazeomatic.mockwaze", nodes, now = 0.0).alerts
        assertEquals(1, alerts.size)
        assertFalse(alerts[0].text.contains("MPH"))
        assertFalse(alerts[0].text.contains("Map"))
    }

    @Test
    fun alertsExpireAndCoverByCategory() {
        val w = WazeAlerts(ttlS = 60.0, maxDistanceM = 1000.0, clock = { now })
        val new = w.update(WazeScreen("com.waze", listOf(WazeAlert("TRAFFIC_HEAVY", "Heavy traffic · 0.5 mi", 800.0))))
        assertEquals(1, new.size)
        assertTrue(w.update(WazeScreen("com.waze", listOf(WazeAlert("TRAFFIC_HEAVY", "Heavy traffic · 0.5 mi", 700.0)))).isEmpty())
        assertNotNull(w.covering("TRAFFIC_STANDSTILL"))   // any traffic alert covers a traffic report
        assertNull(w.covering("HAZARD_ON_ROAD_POTHOLE"))
        now = 61.0
        assertNull(w.covering("TRAFFIC_HEAVY"))
    }

    @Test
    fun farAlertDoesNotCover() {
        val w = WazeAlerts(maxDistanceM = 500.0, clock = { now })
        w.update(WazeScreen("com.waze", listOf(WazeAlert("POLICE", "Police · 2 mi", 3218.0))))
        assertNull(w.covering("POLICE"))
        assertTrue(w.summary().contains("Police"))
        assertEquals("Waze ahead: Police 3,218 m", w.summary())
    }

    @Test
    fun hazardSubtypesDoNotCoverEachOther() {
        val w = WazeAlerts(clock = { now })
        w.update(WazeScreen("com.waze", listOf(WazeAlert("HAZARD_ON_ROAD_POTHOLE", "Pothole · 300 ft", 91.0))))
        assertNotNull(w.covering("HAZARD_ON_ROAD_POTHOLE"))
        assertNull(w.covering("HAZARD_ON_ROAD_CONSTRUCTION"))
    }

    @Test
    fun policeCoversEitherSide() {
        val w = WazeAlerts(clock = { now })
        w.update(WazeScreen("com.waze", listOf(WazeAlert("POLICE", "Police · 0.2 mi", 320.0))))
        assertNotNull(w.covering("POLICE_OTHER_SIDE"))
        assertNull(w.covering("CRASH"))
    }

    @Test
    fun aheadIsNearestFirstAndAlertJsonHasThePythonKeys() {
        val w = WazeAlerts(clock = { now })
        w.update(WazeScreen("com.waze", listOf(
            WazeAlert(null, "Sinkhole reported", null),
            WazeAlert("WEATHER_FOG", "Fog · 0.5 mi", 804.7),
            WazeAlert("POLICE", "Police · 0.2 mi", 321.9),
        )))
        assertEquals(listOf("POLICE", "WEATHER_FOG", null), w.ahead().map { it.hazardType })
        assertEquals("Waze ahead: Police 322 m, Bad weather Fog 805 m, ?", w.summary())
        val o = JSONObject(w.ahead()[0].toJson())
        assertEquals(listOf("category", "distance_m", "hazard_type", "kind", "text", "ts"), o.keys().asSequence().toList().sorted())
        assertEquals("Police", o.getString("category"))
        assertEquals(321.9, o.getDouble("distance_m"), 1e-9)
        assertTrue(JSONObject(w.ahead()[2].toJson()).isNull("hazard_type"))
    }
}
