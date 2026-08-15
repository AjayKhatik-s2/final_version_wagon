"""The existing door / load / damage / OCR / report layers still get what they expect.

This suite is the "everything else is unchanged" half of the swap.  It runs the
REAL Stage-4 fusion and the REAL reporting adapter over a roster produced by the
new counting engine, and pins the constants the dashboard/API contract depends on.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from _engine_harness import (
    CAMERA_LEFT_UP, CAMERA_LEFT_UP_TOP, CAMERA_RIGHT_UP, CAMERA_RIGHT_UP_TOP,
    as_v4_state, drifting_gap_times, run_counting_engine,
)
from core import constants as C


def _roster(n=10):
    times = drifting_gap_times(n, start=26.0)
    return as_v4_state(run_counting_engine(
        times,
        {CAMERA_LEFT_UP:      [t - 0.9 for t in times],
         CAMERA_RIGHT_UP_TOP: [t + 0.2 for t in times],
         CAMERA_LEFT_UP_TOP:  [t + 2.7 for t in times]})[0])


def _write_feature_states(root, state):
    """Per-feature per-wagon JSON in exactly the shape the processors emit."""
    payloads = {
        "door": lambda gw: {
            "global_id": gw, "feature": "door", "status": C.STATUS_OK,
            "left_door": C.DOOR_CLOSED, "left_door_confidence": 0.91,
            "right_door": C.DOOR_OPEN, "right_door_confidence": 0.84,
            "tracks": [], "supporting_cameras": [C.CAMERA_LEFT_UP,
                                                 C.CAMERA_RIGHT_UP],
            "frame_count": 12},
        "ocr": lambda gw: {
            "global_id": gw, "feature": "ocr", "status": C.STATUS_OK,
            "wagon_identifier": "12345678901",
            "wagon_identifier_confidence": 0.77,
            "candidates": [], "supporting_cameras": [C.CAMERA_RIGHT_UP],
            "frame_count": 12},
        "load": lambda gw: {
            "global_id": gw, "feature": "load", "status": C.STATUS_OK,
            "load_status": C.LOAD_LOADED, "load_confidence": 0.88,
            "per_camera": {}, "supporting_cameras": [C.CAMERA_RIGHT_UP_TOP],
            "frame_count": 12},
        "damage": lambda gw: {
            "global_id": gw, "feature": "damage", "status": C.STATUS_OK,
            "top_damage": C.DAMAGE_OK, "top_damage_details": [],
            "per_camera": {
                C.CAMERA_RIGHT_UP_TOP: {"damage_status": C.DAMAGE_OK,
                                        "frames_used": 12, "track_count": 0,
                                        "tracks": []},
                C.CAMERA_LEFT_UP_TOP: {"damage_status": C.DAMAGE_OK,
                                       "frames_used": 12, "track_count": 0,
                                       "tracks": []}},
            "supporting_cameras": [C.CAMERA_RIGHT_UP_TOP], "frame_count": 12},
    }
    for feature, build in payloads.items():
        d = os.path.join(root, feature)
        os.makedirs(d, exist_ok=True)
        for w in state.wagons:
            with open(os.path.join(d, f"{w.global_id}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(build(w.global_id), f)


class TestRosterExposesEverythingInspectionReads(unittest.TestCase):
    """Attributes the four feature processors touch on each global wagon."""

    def test_wagon_fields_present(self):
        state = _roster()
        for w in state.wagons:
            for attr in ("global_id", "wagon_index", "classification",
                         "classification_confidence", "start_time", "end_time",
                         "start_frame_master", "end_frame_master",
                         "supporting_cameras", "duration"):
                self.assertTrue(hasattr(w, attr), f"roster lost {attr}")

    def test_state_fields_present(self):
        state = _roster()
        for attr in ("total_wagons", "wagons", "master_camera", "master_fps",
                     "master_total_frames", "per_camera_local_counts",
                     "per_camera_gap_counts", "per_camera_status",
                     "corrections_applied", "fallback_used", "fallback_reason",
                     "notes", "engine_count", "brake_van_count",
                     "regular_wagon_count"):
            self.assertTrue(hasattr(state, attr), f"state lost {attr}")

    def test_classification_vocabulary_unchanged(self):
        state = _roster()
        allowed = {C.CLASS_WAGON, C.CLASS_ENGINE, C.CLASS_BRAKE_VAN,
                   C.CLASS_UNKNOWN}
        for w in state.wagons:
            self.assertIn(w.classification, allowed)


class TestStage4FusionContract(unittest.TestCase):
    def test_one_unified_state_per_global_wagon(self):
        from fusion import wagon_state_builder

        state = _roster()
        with tempfile.TemporaryDirectory() as tmp:
            _write_feature_states(tmp, state)
            unified = wagon_state_builder.build(
                state=state, wagon_states_root=tmp, verbose=False)

        self.assertEqual(list(unified), [w.global_id for w in state.wagons])
        for w in state.wagons:
            u = unified[w.global_id]
            self.assertEqual(u.global_id, w.global_id)
            self.assertEqual(u.wagon_index, w.wagon_index)
            self.assertEqual(u.classification, w.classification)

    def test_feature_values_survive_fusion_unchanged(self):
        from fusion import wagon_state_builder

        state = _roster(6)
        with tempfile.TemporaryDirectory() as tmp:
            _write_feature_states(tmp, state)
            unified = wagon_state_builder.build(
                state=state, wagon_states_root=tmp, verbose=False)
        u = unified["GW_1"]
        self.assertEqual(u.left_door, C.DOOR_CLOSED)
        self.assertEqual(u.right_door, C.DOOR_OPEN)
        self.assertEqual(u.load_status, C.LOAD_LOADED)
        self.assertEqual(u.top_damage, C.DAMAGE_OK)
        self.assertEqual(u.wagon_identifier, "12345678901")
        self.assertIn("RIGHT_DOOR_OPEN", u.anomalies)


class TestReportingAdapterContract(unittest.TestCase):
    def test_legacy_view_model_covers_the_whole_roster_in_order(self):
        from fusion import wagon_state_builder
        from reporting import _adapter

        state = _roster()
        with tempfile.TemporaryDirectory() as tmp:
            _write_feature_states(tmp, state)
            unified = wagon_state_builder.build(
                state=state, wagon_states_root=tmp, verbose=False)
            vm = _adapter.build_legacy_view_model(
                state=state, unified=unified, wagon_states_root=tmp,
                evidence_root=None, missing_cameras=[])

        self.assertEqual(len(vm.merged_wagons), len(state.wagons))
        self.assertEqual([r["global_id"] for r in vm.merged_wagons],
                         [w.global_id for w in state.wagons])
        self.assertEqual([r["wagon_sr_no"] for r in vm.merged_wagons],
                         list(range(1, len(state.wagons) + 1)))

    def test_summary_kpi_keys_unchanged(self):
        from fusion import wagon_state_builder
        from reporting import _adapter

        state = _roster(5)
        with tempfile.TemporaryDirectory() as tmp:
            _write_feature_states(tmp, state)
            unified = wagon_state_builder.build(
                state=state, wagon_states_root=tmp, verbose=False)
            vm = _adapter.build_legacy_view_model(
                state=state, unified=unified, wagon_states_root=tmp,
                evidence_root=None, missing_cameras=[])

        for key in ("total_wagons", "engine_count", "brake_van_count",
                    "wagon_count", "left_open", "right_open", "top_damages",
                    "left_top_damages", "left_partial", "right_partial",
                    "loaded_count", "empty_count", "ocr_captured",
                    "rake_type", "status", "loco_numbers"):
            self.assertIn(key, vm.summary_kpis)
        self.assertEqual(vm.summary_kpis["total_wagons"], state.total_wagons)


class TestDashboardJsonContractPinned(unittest.TestCase):
    def test_report_schema_string_unchanged(self):
        from reporting import combined_train_report
        self.assertEqual(combined_train_report._REPORT_SCHEMA,
                         "wagon_eye.combined_report.v4")

    def test_feature_model_filenames_unchanged(self):
        """The door / load / damage / OCR weights are untouched by this change."""
        self.assertEqual(C.MODEL_DOOR_STATE, "door_state.pt")
        self.assertEqual(C.MODEL_LOADED, "loaded.pt")
        self.assertEqual(C.MODEL_DAMAGE, "damage.pt")
        self.assertEqual(C.MODEL_WAGON_ID_COUNTING, "wagon_id_counting.pt")

    def test_feature_registry_and_status_vocabulary_unchanged(self):
        from core.feature_config import FEATURE_KEYS
        self.assertEqual(FEATURE_KEYS, ("door", "ocr", "load", "damage"))
        self.assertEqual(C.STATUS_OK, "OK")
        self.assertEqual(C.STATUS_FAILED, "FAILED")
        self.assertEqual(C.STATUS_NO_FRAMES, "NO_FRAMES")
        self.assertEqual(C.STATUS_DISABLED, "DISABLED_BY_USER")
        self.assertEqual(C.DISABLED_DISPLAY, "DISABLED BY USER")
        self.assertEqual(C.NO_DATA, "NO_DATA")

    def test_camera_names_and_roles_unchanged(self):
        self.assertEqual(C.ALL_CAMERAS,
                         ("RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP", "LEFT_UP_TOP"))
        self.assertEqual(C.MASTER_CAMERA, "RIGHT_UP")
        self.assertEqual(C.SIDE_CAMERAS, ("RIGHT_UP", "LEFT_UP"))
        self.assertEqual(C.TOP_CAMERAS, ("RIGHT_UP_TOP", "LEFT_UP_TOP"))
        self.assertEqual(C.CAMERA_FOLDER, {
            "RIGHT_UP": "right_up", "LEFT_UP": "left_up",
            "RIGHT_UP_TOP": "right_up_top", "LEFT_UP_TOP": "left_up_top"})

    def test_batch_outcome_vocabulary_unchanged(self):
        self.assertEqual(C.BATCH_COMPLETED, "completed")
        self.assertEqual(C.BATCH_COMPLETED_PARTIAL, "completed_partial")
        self.assertEqual(C.BATCH_REPORT_FAILED, "report_failed")
        self.assertEqual(C.BATCH_FAILED_NO_GLOBAL, "failed_no_global_state")
        self.assertEqual(C.BATCH_FAILED, "failed")


class TestDisabledFeatureBehaviourUnchanged(unittest.TestCase):
    def test_disabled_feature_still_marks_its_fields(self):
        from fusion import wagon_state_builder

        state = _roster(4)
        with tempfile.TemporaryDirectory() as tmp:
            _write_feature_states(tmp, state)
            # Re-write OCR as the orchestrator's disabled sentinel.
            d = os.path.join(tmp, "ocr")
            for w in state.wagons:
                with open(os.path.join(d, f"{w.global_id}.json"), "w",
                          encoding="utf-8") as f:
                    json.dump({"global_id": w.global_id, "feature": "ocr",
                               "status": C.STATUS_DISABLED, "frame_count": 0,
                               "disabled_by_user": True}, f)
            unified = wagon_state_builder.build(
                state=state, wagon_states_root=tmp, verbose=False)

        for u in unified.values():
            self.assertEqual(u.wagon_identifier, C.DISABLED_DISPLAY)
            self.assertNotIn("OCR_MISSING", u.anomalies)


class TestCameraVisibilityUsesGlobalRoster(unittest.TestCase):
    def test_camera_without_frames_reports_status_not_a_new_number(self):
        """A camera that cannot see a wagon yields NO_FRAMES for it -- never a
        different local wagon number."""
        from features._common import list_wagon_frames

        state = _roster(5)
        with tempfile.TemporaryDirectory() as cache_root:
            # Empty cache: no camera has frames for any wagon.
            for w in state.wagons:
                for cam in C.ALL_CAMERAS:
                    self.assertEqual(
                        list_wagon_frames(cache_root, w.global_id, cam), [])
        # The roster is unaffected by a camera's blindness.
        self.assertEqual([w.global_id for w in state.wagons],
                         [f"GW_{i}" for i in range(1, len(state.wagons) + 1)])


if __name__ == "__main__":
    unittest.main()
