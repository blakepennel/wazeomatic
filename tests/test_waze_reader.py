from wazeomatic.report_manager import ReportConfig, ReportManager
from wazeomatic.schema import Detection, HAZARD_TYPES
from wazeomatic.waze_reader import (WazeAlert, WazeAlerts, WazeScreen, classify_text, group_cards, parse_cards,
                                    parse_distance, parse_uiautomator_xml)


def test_distances():
    assert abs(parse_distance("Police · 0.3 mi") - 482.8) < 0.5
    assert abs(parse_distance("500 ft ahead") - 152.4) < 0.1
    assert parse_distance("1,2 km") == 1200.0
    assert parse_distance("200 m") == 200.0
    assert parse_distance("Reported 5 min ago") is None


def test_classify_known_wording():
    assert classify_text("Police reported ahead") == "POLICE"
    assert classify_text("Police on the other side") == "POLICE_OTHER_SIDE"
    assert classify_text("Crash reported") == "CRASH"
    assert classify_text("Car stopped on shoulder") == "HAZARD_SHOULDER_CAR_STOPPED"
    assert classify_text("Pothole") == "HAZARD_ON_ROAD_POTHOLE"
    assert classify_text("Heavy traffic") == "TRAFFIC_HEAVY"
    assert classify_text("Fog") == "WEATHER_FOG"
    assert classify_text("Right lane closed") == "BLOCKED_LANE_RIGHT"
    assert classify_text("Where to?") is None
    for _rx, key in __import__("wazeomatic.waze_reader", fromlist=["ALERT_PATTERNS"]).ALERT_PATTERNS:
        assert key in HAZARD_TYPES


def test_wording_seen_in_the_real_app():
    # phrases quoted in press reports of the real app (2024-2026), not only the mock: the alert banner and the
    # prompt that asks whether a report is still there
    assert classify_text("Car on shoulder reported ahead") == "HAZARD_SHOULDER_CAR_STOPPED"
    assert classify_text("Police reported ahead") == "POLICE"
    (a,) = parse_cards([["Police reported ahead", "0.3 mi", "Are they still there?", "Not there"]])
    assert (a.hazard_type, a.kind) == ("POLICE", "still_there")


def test_cards_need_alert_context():
    # a report-sheet tile named Police is not an alert; an alert card is
    cards = [["Police"], ["Police", "0.4 mi", "Reported 3 min ago"]]
    alerts = parse_cards(cards, now=10.0)
    assert len(alerts) == 1 and alerts[0].hazard_type == "POLICE"
    assert abs(alerts[0].distance_m - 643.7) < 1


def test_still_there_prompt_and_unknown_wording_kept():
    alerts = parse_cards([["Pothole", "Still there?", "Yes", "No"], ["Sinkhole reported", "200 ft"]], now=0)
    assert alerts[0].kind == "still_there" and alerts[0].hazard_type == "HAZARD_ON_ROAD_POTHOLE"
    assert alerts[1].hazard_type is None             # kept, so the log shows wording to add


def test_speedometer_is_not_an_alert():
    assert parse_cards([["35", "km/h"], ["0.5 mi", "Turn right"]], now=0) == []


def test_grouping_by_vertical_gap():
    nodes = [("Police", (0, 100, 300, 140)), ("0.4 mi", (0, 150, 200, 180)), ("Where to?", (0, 900, 400, 950))]
    assert group_cards(nodes) == [["Police", "0.4 mi"], ["Where to?"]]


XML = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?><hierarchy rotation="0">
<node index="0" text="" package="com.wazeomatic.mockwaze" bounds="[0,0][1280,2856]">
<node index="1" text="Police" package="com.wazeomatic.mockwaze" content-desc="" bounds="[100,300][600,360]"/>
<node index="2" text="0.3 mi" package="com.wazeomatic.mockwaze" content-desc="" bounds="[100,370][300,410]"/>
<node index="3" text="Reported 2 min ago" package="com.wazeomatic.mockwaze" content-desc="" bounds="[100,420][700,460]"/>
<node index="4" text="Where to?" package="com.wazeomatic.mockwaze" content-desc="" bounds="[100,2600][900,2700]"/>
</node>
<node index="0" text="Police" package="com.android.systemui" bounds="[0,0][10,10]"/>
</hierarchy>"""


def test_uiautomator_dump():
    screen = parse_uiautomator_xml("UI hierchary dumped to: /dev/tty" + XML, now=5.0)
    assert screen.package == "com.wazeomatic.mockwaze"
    assert [a.hazard_type for a in screen.alerts] == ["POLICE"]
    assert not screen.report_sheet_open


def test_report_sheet_suppresses_alerts():
    xml = XML.replace('text="Where to?"', 'text="What do you see?"')
    screen = parse_uiautomator_xml(xml, now=0)
    assert screen.report_sheet_open and screen.alerts == []


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_alerts_expire_and_cover_by_category():
    clk = Clock()
    w = WazeAlerts(ttl_s=60, max_distance_m=1000, clock=clk)
    new = w.update(WazeScreen(package="com.waze", alerts=[WazeAlert("TRAFFIC_HEAVY", "Heavy traffic · 0.5 mi", 800)]))
    assert len(new) == 1
    assert w.update(WazeScreen(package="com.waze", alerts=[WazeAlert("TRAFFIC_HEAVY", "Heavy traffic · 0.5 mi", 700)])) == []
    assert w.covering("TRAFFIC_STANDSTILL") is not None      # any traffic alert covers a traffic report
    assert w.covering("HAZARD_ON_ROAD_POTHOLE") is None
    clk.t = 61
    assert w.covering("TRAFFIC_HEAVY") is None


def test_far_alert_does_not_cover():
    w = WazeAlerts(max_distance_m=500, clock=Clock())
    w.update(WazeScreen(package="com.waze", alerts=[WazeAlert("POLICE", "Police · 2 mi", 3218)]))
    assert w.covering("POLICE") is None
    assert "Police" in w.summary()


def test_hazard_subtypes_do_not_cover_each_other():
    w = WazeAlerts(clock=Clock())
    w.update(WazeScreen(package="com.waze", alerts=[WazeAlert("HAZARD_ON_ROAD_POTHOLE", "Pothole · 300 ft", 91)]))
    assert w.covering("HAZARD_ON_ROAD_POTHOLE") is not None
    assert w.covering("HAZARD_ON_ROAD_CONSTRUCTION") is None


def test_manager_turns_known_report_into_confirmation():
    clk = Clock()
    w = WazeAlerts(clock=clk)
    w.update(WazeScreen(package="com.waze", alerts=[WazeAlert("POLICE", "Police · 0.2 mi", 320)]))
    m = ReportManager(ReportConfig(), clock=clk, waze=w)
    det = Detection("police_vehicle", 0.9, (0.6, 0.5, 0.1, 0.1), model="vehicle")
    events = []
    for i in range(5):
        clk.t = i * 0.1
        events += m.update([det], frame_index=i)
    assert events == []
    assert m.already_reported == 1 and m.confirmations[0]["waze"]["hazard_type"] == "POLICE"
    # no report went out, so a different hazard may still be sent straight away
    cone = Detection("cone", 0.9, (0.5, 0.5, 0.05, 0.1))
    for i in range(5, 9):
        clk.t = i * 0.1
        events += m.update([cone], frame_index=i)
    assert [e.hazard_type for e in events] == ["HAZARD_ON_ROAD_CONSTRUCTION"]


def test_manager_without_reader_reports():
    clk = Clock()
    m = ReportManager(ReportConfig(), clock=clk)
    det = Detection("police_vehicle", 0.9, (0.6, 0.5, 0.1, 0.1), model="vehicle")
    events = []
    for i in range(4):
        clk.t = i * 0.1
        events += m.update([det], frame_index=i)
    assert [e.hazard_type for e in events] == ["POLICE"]


def test_report_button_is_not_an_open_sheet():
    xml = XML.replace('text="Where to?"', 'text="Report button"')
    screen = parse_uiautomator_xml(xml, now=0)
    assert not screen.report_sheet_open and [a.hazard_type for a in screen.alerts] == ["POLICE"]
    for title in ("What do you see?", "Report a hazard", "Report police", "Report bad weather"):
        assert parse_uiautomator_xml(XML.replace('text="Where to?"', f'text="{title}"'), now=0).report_sheet_open


def test_full_screen_node_does_not_merge_cards():
    xml = XML.replace('<node index="0" text="" package="com.wazeomatic.mockwaze" bounds="[0,0][1280,2856]">',
                      '<node index="0" text="Map" package="com.wazeomatic.mockwaze" bounds="[0,0][1280,2856]">')
    xml = xml.replace('text="Where to?"', 'text="35 MPH"')
    alerts = parse_uiautomator_xml(xml, now=0).alerts
    assert len(alerts) == 1 and "MPH" not in alerts[0].text and "Map" not in alerts[0].text
