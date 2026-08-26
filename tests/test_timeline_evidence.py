"""Observations become wagon assignments by timestamp, and by nothing else.

Not by segment index, array position, or a camera's own wagon numbering --
those are camera-local and cannot survive a clock offset. Not by which
directory a frame was read from either, which is how it used to work: the
materializer bucketed frames into `wagon_cache/<GW_n>/` and the assignment was
implicit in the path, so a wrong bucket was indistinguishable from a right one
afterwards.

The two genuine decisions -- an observation exactly on a gap, and one that
straddles a gap -- are configurable policies, and every assignment records
which one applied and why.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from core.global_state_loader import GlobalWagon
from core.master_timeline import BoundaryPolicy, CameraClock
from core.timeline_evidence import (
    ARTIFACT_NAME, KIND_DAMAGE, KIND_DOOR, KIND_GAP, KIND_LOAD, KIND_OCR,
    REASON_BOUNDARY, REASON_CONTAINED, REASON_OUTSIDE, REASON_SPAN_CENTER,
    REASON_SPAN_OVERLAP, SPAN_CENTER, SPAN_OVERLAP, AssignmentPolicy,
    Observation, TimelineEvidence, assign_observation, assign_observations,
    observations_from_feature_json, observations_from_gaps, write_artifact,
)

RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP
RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP
FPS = 15.0

#: Gaps at 100.0 and 200.0 -> three wagons.
ROSTER = [
    GlobalWagon(global_id="GW_1", wagon_index=1, start_frame_master=0,
                end_frame_master=1499, start_time=0.0, end_time=100.0,
                classification=C.CLASS_WAGON, classification_confidence=0.9),
    GlobalWagon(global_id="GW_2", wagon_index=2, start_frame_master=1500,
                end_frame_master=2999, start_time=100.0, end_time=200.0,
                classification=C.CLASS_WAGON, classification_confidence=0.9),
    GlobalWagon(global_id="GW_3", wagon_index=3, start_frame_master=3000,
                end_frame_master=4499, start_time=200.0, end_time=300.0,
                classification=C.CLASS_WAGON, classification_confidence=0.9),
]


def _obs(t, t_end=None, cam=RU, kind=KIND_DAMAGE, conf=0.8):
    return Observation(camera_id=cam, kind=kind, t_start=t, t_end=t_end,
                       confidence=conf)


class TestTheStatedExample(unittest.TestCase):
    """A gap at 100.0s: 98.5s belongs before it, 101.2s after it."""

    def test_before_the_gap_goes_to_the_previous_wagon(self):
        a = assign_observation(_obs(98.5), ROSTER)
        self.assertEqual(a.global_id, "GW_1")
        self.assertEqual(a.reason, REASON_CONTAINED)

    def test_after_the_gap_goes_to_the_next_wagon(self):
        a = assign_observation(_obs(101.2), ROSTER)
        self.assertEqual(a.global_id, "GW_2")

    def test_the_second_gap_behaves_the_same(self):
        self.assertEqual(assign_observation(_obs(199.0), ROSTER).global_id,
                         "GW_2")
        self.assertEqual(assign_observation(_obs(201.0), ROSTER).global_id,
                         "GW_3")


class TestBoundaryPolicy(unittest.TestCase):
    def test_exactly_on_a_gap_goes_next_by_default(self):
        a = assign_observation(_obs(100.0), ROSTER)
        self.assertEqual(a.global_id, "GW_2")
        self.assertEqual(a.reason, REASON_BOUNDARY)
        self.assertIn("next", a.detail)

    def test_previous_policy_is_honoured_and_recorded(self):
        p = AssignmentPolicy(boundary=BoundaryPolicy(on_boundary="previous"))
        a = assign_observation(_obs(100.0), ROSTER, policy=p)
        self.assertEqual(a.global_id, "GW_1")
        self.assertIn("previous", a.detail)

    def test_float_noise_either_side_lands_the_same_way(self):
        for t in (100.0 - 1e-9, 100.0, 100.0 + 1e-9):
            with self.subTest(t=t):
                self.assertEqual(assign_observation(_obs(t), ROSTER).global_id,
                                 "GW_2")

    def test_the_final_boundary_belongs_to_the_last_wagon(self):
        a = assign_observation(_obs(300.0), ROSTER)
        self.assertEqual(a.global_id, "GW_3")
        self.assertEqual(a.reason, REASON_BOUNDARY)


class TestSpanningObservations(unittest.TestCase):
    def test_a_span_inside_one_wagon_is_contained(self):
        a = assign_observation(_obs(10.0, 20.0), ROSTER)
        self.assertEqual(a.global_id, "GW_1")
        self.assertEqual(a.reason, REASON_CONTAINED)

    def test_center_policy_picks_the_wagon_holding_the_midpoint(self):
        a = assign_observation(_obs(95.0, 130.0), ROSTER)   # centre 112.5
        self.assertEqual(a.global_id, "GW_2")
        self.assertEqual(a.reason, REASON_SPAN_CENTER)
        self.assertIn("centre", a.detail)

    def test_overlap_policy_picks_the_longest_overlap(self):
        p = AssignmentPolicy(span=SPAN_OVERLAP)
        a = assign_observation(_obs(95.0, 130.0), ROSTER, policy=p)
        self.assertEqual(a.global_id, "GW_2")           # 30s vs 5s
        self.assertEqual(a.reason, REASON_SPAN_OVERLAP)

    def test_the_two_policies_can_genuinely_disagree(self):
        """A span whose centre and longest overlap differ."""
        obs = _obs(60.0, 105.0)                          # centre 82.5 -> GW_1
        centre = assign_observation(obs, ROSTER)
        overlap = assign_observation(
            obs, ROSTER, policy=AssignmentPolicy(span=SPAN_OVERLAP))
        self.assertEqual(centre.global_id, "GW_1")       # 40s vs 5s
        self.assertEqual(overlap.global_id, "GW_1")
        wide = _obs(95.0, 260.0)                         # centre 177.5 -> GW_2
        self.assertEqual(assign_observation(wide, ROSTER).global_id, "GW_2")
        self.assertEqual(assign_observation(
            wide, ROSTER, policy=AssignmentPolicy(span=SPAN_OVERLAP)
        ).global_id, "GW_2")

    def test_a_span_across_three_wagons_records_how_many(self):
        a = assign_observation(_obs(50.0, 250.0), ROSTER)
        self.assertIn("spans 3 wagons", a.detail)

    def test_the_decision_is_always_recorded(self):
        for obs in (_obs(50.0), _obs(100.0), _obs(95.0, 130.0), _obs(500.0)):
            a = assign_observation(obs, ROSTER)
            self.assertTrue(a.reason, "every assignment must carry a reason")
            self.assertIn(a.reason, (REASON_CONTAINED, REASON_BOUNDARY,
                                     REASON_SPAN_CENTER, REASON_SPAN_OVERLAP,
                                     REASON_OUTSIDE))


class TestOutsideTheWagonRegion(unittest.TestCase):
    def test_before_the_first_wagon_is_unassigned_not_forced(self):
        a = assign_observation(_obs(-50.0), ROSTER)
        self.assertIsNone(a.global_id)
        self.assertEqual(a.reason, REASON_OUTSIDE)

    def test_after_the_last_wagon_is_unassigned(self):
        self.assertIsNone(assign_observation(_obs(500.0), ROSTER).global_id)

    def test_an_engine_observation_is_not_forced_into_gw_1(self):
        """The engine precedes the wagon region; it must not become GW_1."""
        a = assign_observation(_obs(-10.0, -2.0), ROSTER)
        self.assertIsNone(a.global_id)
        self.assertIn("outside", a.detail)

    def test_an_empty_roster_assigns_nothing(self):
        self.assertIsNone(assign_observation(_obs(50.0), []).global_id)


class TestMultiCameraTimeProjection(unittest.TestCase):
    def test_different_fps_produces_the_same_master_time(self):
        clocks = {RU: CameraClock(RU, fps=15.0, total_frames=9000),
                  LU: CameraClock(LU, fps=30.0, total_frames=18000)}
        payload = {"top_damage_details": [
            {"camera_id": RU, "best_frame_idx": 1500, "best_confidence": 0.8,
             "class_name": "dent"},
            {"camera_id": LU, "best_frame_idx": 3000, "best_confidence": 0.8,
             "class_name": "dent"}]}
        obs = observations_from_feature_json(payload, feature=KIND_DAMAGE,
                                             clocks=clocks)
        self.assertEqual(len(obs), 2)
        self.assertAlmostEqual(obs[0].t_start, 100.0)
        self.assertAlmostEqual(obs[1].t_start, 100.0)
        for o in obs:
            self.assertEqual(assign_observation(o, ROSTER).global_id, "GW_2")

    def test_a_camera_offset_is_applied_per_camera(self):
        clocks = {LU: CameraClock(LU, fps=15.0, total_frames=9000,
                                  offset=5.0)}
        payload = {"top_damage_details": [
            {"camera_id": LU, "best_frame_idx": 1425, "best_confidence": 0.8}]}
        o = observations_from_feature_json(payload, feature=KIND_DAMAGE,
                                           clocks=clocks)[0]
        self.assertAlmostEqual(o.t_start, 100.0)       # 95s local + 5s
        self.assertEqual(assign_observation(o, ROSTER).global_id, "GW_2")

    def test_equal_frame_numbers_are_never_assumed_equal_in_time(self):
        clocks = {RU: CameraClock(RU, fps=15.0, total_frames=9000),
                  LU: CameraClock(LU, fps=25.0, total_frames=15000)}
        payload = {"top_damage_details": [
            {"camera_id": RU, "best_frame_idx": 1500, "best_confidence": 0.5},
            {"camera_id": LU, "best_frame_idx": 1500, "best_confidence": 0.5}]}
        obs = observations_from_feature_json(payload, feature=KIND_DAMAGE,
                                             clocks=clocks)
        self.assertNotAlmostEqual(obs[0].t_start, obs[1].t_start)

    def test_a_record_without_a_clock_is_dropped_not_guessed(self):
        payload = {"top_damage_details": [
            {"camera_id": "NO_SUCH_CAM", "best_frame_idx": 10}]}
        self.assertEqual(observations_from_feature_json(
            payload, feature=KIND_DAMAGE, clocks={}), [])


class TestSupportCameraGaps(unittest.TestCase):
    def test_support_gaps_are_observations_not_authority(self):
        class G:
            center_time, confidence, center_frame, track_id = 100.0, 0.9, 1500, 3
        obs = observations_from_gaps([G()], LUT, detected=True)
        self.assertEqual(obs[0].kind, KIND_GAP)
        self.assertEqual(obs[0].camera_id, LUT)
        self.assertTrue(obs[0].detected)

    def test_a_missing_support_gap_does_not_prevent_assignment(self):
        """A camera that missed the gap still assigns by projected time."""
        ev = TimelineEvidence(mode="test")
        ev.add(_obs(101.2, cam=LUT, kind=KIND_DAMAGE))
        ev.fuse(ROSTER)
        self.assertEqual(ev.assignments[0].global_id, "GW_2")

    def test_projected_provenance_is_preserved(self):
        class G:
            center_time, confidence, center_frame, track_id = 100.0, 0.0, 1500, 0
        obs = observations_from_gaps([G()], LUT, detected=False)
        self.assertFalse(obs[0].detected)


class TestEvidenceContainerAndAudit(unittest.TestCase):
    def _evidence(self):
        ev = TimelineEvidence(mode="sequential",
                              canonical_gaps=[100.0, 200.0])
        ev.extend([_obs(50.0, kind=KIND_DOOR, cam=RU),
                   _obs(101.2, kind=KIND_DAMAGE, cam=RUT),
                   _obs(150.0, kind=KIND_LOAD, cam=RUT),
                   _obs(250.0, kind=KIND_OCR, cam=RU),
                   _obs(-10.0, kind=KIND_DAMAGE, cam=LUT)])
        ev.fuse(ROSTER)
        return ev

    def test_observations_are_grouped_by_wagon(self):
        by = self._evidence().by_wagon()
        self.assertEqual([a.observation.kind for a in by["GW_1"]], [KIND_DOOR])
        self.assertEqual(len(by["GW_2"]), 2)
        self.assertEqual(len(by["UNASSIGNED"]), 1)

    def test_evidence_collected_before_the_roster_still_fuses(self):
        """Phase 1 collects; Phase 2 assigns. Nothing is discarded meanwhile."""
        ev = TimelineEvidence(mode="sequential")
        ev.add(_obs(101.2))
        self.assertEqual(ev.assignments, [], "nothing assigned before fusion")
        ev.fuse(ROSTER)
        self.assertEqual(ev.assignments[0].global_id, "GW_2")

    def test_the_artifact_explains_every_assignment(self):
        ev = self._evidence()
        with tempfile.TemporaryDirectory() as root:
            path = write_artifact(ev, root)
            self.assertEqual(os.path.basename(path), ARTIFACT_NAME)
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        self.assertEqual(doc["mode"], "sequential")
        self.assertEqual(doc["canonical_gaps"], [100.0, 200.0])
        self.assertEqual(doc["assignment_policy"]["on_boundary"], "next")
        self.assertEqual(doc["assignment_policy"]["span"], SPAN_CENTER)
        self.assertEqual(doc["unassigned"], 1)
        self.assertEqual(len(doc["roster"]), 3)
        for rec in doc["observations"]:
            self.assertIn("assigned_gw", rec)
            self.assertIn("assignment_reason", rec)
            self.assertIn("camera_id", rec)
            self.assertIn("center_time", rec)

    def test_the_summary_names_the_policy(self):
        text = " ".join(self._evidence().summary_lines())
        self.assertIn("on_boundary", text)
        self.assertIn("UNASSIGNED", text)


class TestBatchSequentialParity(unittest.TestCase):
    """One implementation, so the two modes cannot diverge.

    They differ in WHEN evidence is collected, never in HOW it is assigned.
    Given identical evidence and an identical roster the output must be
    identical, field for field.
    """

    def _run(self, mode):
        ev = TimelineEvidence(mode=mode, canonical_gaps=[100.0, 200.0])
        ev.extend([
            _obs(98.5, kind=KIND_DOOR, cam=RU),
            _obs(100.0, kind=KIND_DAMAGE, cam=RUT),
            _obs(101.2, kind=KIND_DAMAGE, cam=LUT),
            _obs(95.0, 130.0, kind=KIND_LOAD, cam=RUT),
            _obs(250.0, kind=KIND_OCR, cam=RU),
        ])
        ev.fuse(ROSTER)
        return ev

    def test_the_rosters_are_identical(self):
        self.assertEqual(self._run("batch").roster,
                         self._run("sequential").roster)

    def test_every_assignment_is_identical(self):
        b = [(a.observation.kind, a.observation.t_start, a.global_id, a.reason)
             for a in self._run("batch").assignments]
        s = [(a.observation.kind, a.observation.t_start, a.global_id, a.reason)
             for a in self._run("sequential").assignments]
        self.assertEqual(b, s)

    def test_the_artifacts_match_apart_from_the_mode_label(self):
        b, s = self._run("batch").to_dict(), self._run("sequential").to_dict()
        self.assertNotEqual(b.pop("mode"), s.pop("mode"))
        self.assertEqual(b, s)

    def test_assignment_is_order_independent(self):
        """Collection order differs between the modes; the result must not."""
        obs = [_obs(250.0, kind=KIND_OCR), _obs(98.5, kind=KIND_DOOR),
               _obs(101.2, kind=KIND_DAMAGE)]
        a = assign_observations(obs, ROSTER)
        b = assign_observations(list(reversed(obs)), ROSTER)
        self.assertEqual([(x.observation.t_start, x.global_id) for x in a],
                         [(x.observation.t_start, x.global_id) for x in b])

    def test_there_is_one_assignment_implementation(self):
        """Neither orchestrator may carry its own copy."""
        import inspect
        from orchestrator import global_assembler
        seq = inspect.getsource(global_assembler)
        batch = open(os.path.join(V4_ROOT, "wagon_count",
                                  "run_global_count.py"),
                     encoding="utf-8").read()
        for src in (seq, batch):
            self.assertNotIn("def assign_observation", src)
            self.assertNotIn("REASON_SPAN_CENTER", src)


if __name__ == "__main__":
    unittest.main()
