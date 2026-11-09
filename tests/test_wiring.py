"""Wiring tests: the defaults a user actually gets from the command-line tools, and how a run
that dies is told apart from one that finished."""
import sys
from pathlib import Path

import numpy as np
import pytest

from wazeomatic import run_video
from wazeomatic.build import DetectorSpec, build_detectors, build_hazard_detector
from wazeomatic.fusion import parse_class_thresholds
from wazeomatic.pipeline import Pipeline
from wazeomatic.report_manager import DEFAULT_LABEL_FLOORS, ReportConfig, ReportManager, default_label_floor_spec
from wazeomatic.schema import Detection
from wazeomatic.sinks import AdbBroadcastSink, build_sinks
from wazeomatic.sources import Frame, Prefetch, SourceError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def test_cli_default_keeps_the_shipped_label_floors():
    args = run_video.build_parser().parse_args(["clip.mp4"])
    assert parse_class_thresholds(args.report_class_conf) == DEFAULT_LABEL_FLOORS
    assert parse_class_thresholds(default_label_floor_spec()) == DEFAULT_LABEL_FLOORS
    # an empty string is the way to ask for no floors at all
    args = run_video.build_parser().parse_args(["clip.mp4", "--report-class-conf", ""])
    assert parse_class_thresholds(args.report_class_conf) == {}


def test_cli_default_report_config_rejects_a_weak_pothole():
    args = run_video.build_parser().parse_args(["clip.mp4"])
    cfg = ReportConfig(min_confidence=args.report_conf, global_cooldown_s=0,
                       min_confidence_by_label=parse_class_thresholds(args.report_class_conf))
    rm = ReportManager(cfg, clock=lambda: 1000.0)
    weak = Detection("pothole", 0.42, (0.5, 0.5, 0.1, 0.1))   # above the 0.40 floor, below the pothole floor
    assert [e for i in range(4) for e in rm.update([weak], frame_index=i)] == []
    cone = Detection("cone", 0.42, (0.5, 0.5, 0.1, 0.1))
    assert len([e for i in range(4, 8) for e in rm.update([cone], frame_index=i)]) == 1


def test_eval_events_default_keeps_the_shipped_label_floors():
    import eval_events
    args = eval_events.build_parser().parse_args([])
    assert parse_class_thresholds(args.report_class_conf) == DEFAULT_LABEL_FLOORS


def test_label_floor_only_raises_the_bar():
    cfg = ReportConfig(min_confidence=0.6, min_confidence_by_label={"pothole": 0.45}, global_cooldown_s=0)
    rm = ReportManager(cfg, clock=lambda: 1000.0)
    d = Detection("pothole", 0.5, (0.5, 0.5, 0.1, 0.1))        # above the label floor, below min_confidence
    assert [e for i in range(4) for e in rm.update([d], frame_index=i)] == []


def test_adb_sink_is_dry_run_unless_asked():
    from wazeomatic.schema import HazardEvent
    ev = HazardEvent("HAZARD_ON_ROAD_POTHOLE", 0.9, source="sim")
    cmd = AdbBroadcastSink().command(ev)
    assert cmd[cmd.index("dry_run") + 1] == "true"
    (sink,) = build_sinks(["adb"])
    assert sink.dry_run is True
    args = run_video.build_parser().parse_args(["clip.mp4", "--sink", "adb"])
    assert args.adb_live is False
    live = AdbBroadcastSink(dry_run=False).command(ev)
    assert live[live.index("dry_run") + 1] == "false"


def test_missing_hazard_model_is_fatal(tmp_path):
    spec = DetectorSpec(hazard_models=[tmp_path / "nope.onnx"], coco_model=None)
    with pytest.raises(SystemExit) as e:
        build_hazard_detector(spec)
    assert "hazard model not found" in str(e.value)
    with pytest.raises(SystemExit):
        build_detectors(spec)
    # asking for no hazard model at all is still allowed
    assert build_hazard_detector(DetectorSpec(hazard_models=[], coco_model=None)) is None


class _Dying:
    """Three frames, then the decoder blows up."""
    effective_fps = 10.0

    def __iter__(self):
        for i in range(3):
            yield Frame(np.zeros((36, 64, 3), np.uint8), i, i / 10)
        raise RuntimeError("decoder exploded")


class _Finishing(_Dying):
    def __iter__(self):
        for i in range(3):
            yield Frame(np.zeros((36, 64, 3), np.uint8), i, i / 10)


def test_prefetch_passes_a_source_failure_on():
    seen = []
    with pytest.raises(SourceError):
        for fr in Prefetch(_Dying(), depth=2):
            seen.append(fr.index)
    assert seen == [0, 1, 2]
    assert [fr.index for fr in Prefetch(_Finishing(), depth=2)] == [0, 1, 2]


def test_pipeline_stats_tell_a_dead_source_from_a_finished_one():
    def run(src):
        pipe = Pipeline([], ReportManager(ReportConfig()), [], parallel=False)
        return pipe.run(src)

    ok = run(Prefetch(_Finishing(), depth=2))
    assert (ok.ended, ok.error, ok.frames) == ("completed", "", 3)
    bad = run(Prefetch(_Dying(), depth=2))
    assert bad.ended == "error" and "decoder exploded" in bad.error and bad.frames == 3
    assert bad.summary()["ended"] == "error"


def test_unknown_hazard_key_in_classes_yaml_is_dropped(tmp_path):
    from wazeomatic.schema import HAZARD_TYPES, load_class_to_hazard
    y = tmp_path / "classes.yaml"
    y.write_text("hazard_map:\n  pothole: HAZARD_ON_ROAD_POTHOLES\n  cone: HAZARD_ON_ROAD_CONSTRUCTION\n", encoding="utf-8")
    warnings = []
    m = load_class_to_hazard(y, warn=warnings.append)
    assert "pothole" not in m and m["cone"] == "HAZARD_ON_ROAD_CONSTRUCTION"
    assert len(warnings) == 1 and "HAZARD_ON_ROAD_POTHOLES" in warnings[0]
    assert all(v in HAZARD_TYPES for v in m.values())
    # the shipped file is clean
    shipped = load_class_to_hazard(ROOT / "configs" / "classes.yaml", warn=warnings.append)
    assert len(warnings) == 1 and shipped["worker"] == "HAZARD_ON_ROAD_CONSTRUCTION"


def test_datasets_root_resolution(monkeypatch, tmp_path):
    from wazeomatic import paths
    monkeypatch.setenv("WAZEOMATIC_DATASETS", str(tmp_path))
    assert paths.datasets_root() == tmp_path
    assert paths.merged_dataset() == tmp_path / "wazeomatic_hazards_v3"


def test_adb_sink_never_passes_an_empty_argument():
    from wazeomatic.schema import HAZARD_TYPES, HazardEvent
    for key in HAZARD_TYPES:                       # Police, Crash and plain Hazard have no subcategory
        cmd = AdbBroadcastSink().command(HazardEvent(key, 0.9, source="sim"))
        assert "" not in cmd, key
        assert cmd[cmd.index("hazard_type") + 1] == key
        assert cmd[-3:] == ["--es", "dry_run", "true"]
