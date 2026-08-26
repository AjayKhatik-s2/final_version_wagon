"""RIGHT_UP owns the roster; the other three observe it.

The master-fixed architecture already enforces this structurally -- fusion's
`build_global_gap_sequence` is documented as "the only function in the codebase
that may mint a global_gap_id, and it consults no support camera". These tests
hold it to that from the outside, and, just as importantly, keep the two
outcomes apart:

    a support camera CHANGED the roster       -> violation, must be caught
    a support camera DISAGREED with it        -> expected, must NOT be caught

The second happens constantly. A support camera sees the train from a different
angle: it misses gaps, invents gaps, splits one wagon across two local
segments, merges two into one, and sometimes calls a wagon a brake van. An
audit that flagged those would cry wolf on every healthy run and bury a real
break in the noise.

Each violation check is also tested against a deliberately corrupted state, so
a check that can never fire cannot pass for free.
"""

from __future__ import annotations

import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from core.camera_evidence import LocalSegment, local_segment_id
from core.canonical_audit import (
    EXPECTED_DISAGREEMENTS, audit, check_invariant,
)
from core.global_state_loader import GlobalTrainState, GlobalWagon

RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP
RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP
FPS = 15.0


def _wagon(i, start, end, cls=C.CLASS_WAGON):
    return GlobalWagon(global_id=f"GW_{i}", wagon_index=i,
                       start_frame_master=int(start * FPS),
                       end_frame_master=int(end * FPS) - 1,
                       start_time=start, end_time=end,
                       classification=cls, classification_confidence=0.9)


def _roster(n=4):
    return [_wagon(i, (i - 1) * 4.0, i * 4.0) for i in range(1, n + 1)]


def _state(wagons, *, gaps=None, checks=None, master_wagon_count=None,
           extras=None, align=None, offsets=None):
    st = GlobalTrainState(total_wagons=len(wagons), wagons=tuple(wagons),
                          master_camera=RU, master_fps=FPS,
                          master_total_frames=3000)
    st.global_gaps = gaps if gaps is not None else [
        {"global_gap_id": i, "master_camera": RU,
         "master_observation": {"camera_id": RU, "center_time": w.start_time}}
        for i, w in enumerate(wagons[1:], start=1)]
    st.camera_offsets = offsets or {
        RU: {"status": "REFERENCE", "delta": 0.0},
        LU: {"status": "RESOLVED", "delta": -1.5},
        RUT: {"status": "RESOLVED", "delta": 2.0},
        LUT: {"status": "RESOLVED", "delta": 0.5},
    }
    st.invariant_checks = checks if checks is not None else {
        "right_up_final_gap_count": len(st.global_gaps),
        "global_gap_count": len(st.global_gaps)}
    st.master_wagon_count = (len(wagons) if master_wagon_count is None
                             else master_wagon_count)
    st.extra_support_observations = extras or {}
    st.support_alignment_summary = align or {}
    return st


def _segments(cam, spans):
    return [LocalSegment(local_id=local_segment_id(cam, i), index=i,
                         start_frame=int(s * FPS), end_frame=int(e * FPS) - 1,
                         start_time=s, end_time=e,
                         label=C.CLASS_WAGON, confidence=0.8)
            for i, (s, e) in enumerate(spans, start=1)]


class TestInvariantHoldsOnHealthyRuns(unittest.TestCase):
    def test_clean_state_has_no_violation(self):
        self.assertEqual(check_invariant(_state(_roster())), [])

    def test_audit_reports_the_canonical_roster(self):
        r = audit(_state(_roster(5)))
        self.assertTrue(r.invariant_holds)
        self.assertEqual(r.canonical_wagon_count, 5)
        self.assertEqual(r.master_camera, RU)
        self.assertEqual([w.canonical_wagon_id for w in r.wagons],
                         [f"GW_{i}" for i in range(1, 6)])
        self.assertEqual(r.timeline_start, 0.0)
        self.assertEqual(r.timeline_end, 20.0)

    def test_every_canonical_gap_comes_from_the_master(self):
        st = _state(_roster())
        for g in st.global_gaps:
            self.assertEqual(g["master_camera"], RU)
        self.assertEqual(check_invariant(st), [])


class TestSupportCamerasCannotChangeTheRoster(unittest.TestCase):
    """Disagreement in, identical roster out."""

    def test_a_missed_gap_does_not_merge_wagons(self):
        st = _state(_roster(4), align={
            LUT: {"missing_global_gap_ids": [2], "extra_observations": [],
                  "matches": {}}})
        r = audit(st)
        self.assertEqual(r.canonical_wagon_count, 4, "wagons were merged")
        self.assertTrue(r.invariant_holds)
        self.assertEqual(r.per_camera_stats[LUT]["missed_gaps"], 1)
        self.assertIn("GAP_MISSED", r.disagreements)

    def test_an_extra_gap_does_not_split_wagons(self):
        st = _state(_roster(4), extras={
            RUT: [{"global_time": 2.0, "camera_id": RUT}]},
            align={RUT: {"missing_global_gap_ids": [],
                         "extra_observations": [{"global_time": 2.0}],
                         "matches": {}}})
        r = audit(st)
        self.assertEqual(r.canonical_wagon_count, 4, "wagons were split")
        self.assertTrue(r.invariant_holds)
        self.assertEqual(r.per_camera_stats[RUT]["extra_gaps"], 1)
        self.assertIn("GAP_EXTRA", r.disagreements)

    def test_a_camera_with_no_gaps_at_all_changes_nothing(self):
        st = _state(_roster(4), align={LU: {"missing_global_gap_ids": [1, 2, 3],
                                            "extra_observations": [],
                                            "matches": {}}})
        r = audit(st)
        self.assertEqual(r.canonical_wagon_count, 4)
        self.assertTrue(r.invariant_holds)

    def test_more_local_segments_than_canonical_wagons(self):
        st = _state(_roster(2))
        r = audit(st, local_segments={
            LU: _segments(LU, [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0),
                               (4.0, 8.0)])})
        self.assertEqual(r.canonical_wagon_count, 2)
        self.assertTrue(r.invariant_holds)

    def test_fewer_local_segments_than_canonical_wagons(self):
        st = _state(_roster(4))
        r = audit(st, local_segments={LUT: _segments(LUT, [(0.0, 16.0)])})
        self.assertEqual(r.canonical_wagon_count, 4)
        self.assertTrue(r.invariant_holds)


class TestClassificationDisagreement(unittest.TestCase):
    def test_a_camera_calling_a_wagon_something_else_keeps_the_identity(self):
        st = _state(_roster(2))
        segs = _segments(LUT, [(0.0, 4.0)])
        segs[0].label = C.CLASS_BRAKE_VAN
        r = audit(st, local_segments={LUT: segs})
        gw1 = r.wagons[0]
        self.assertEqual(gw1.canonical_wagon_id, "GW_1")
        self.assertEqual(gw1.canonical_class, C.CLASS_WAGON,
                         "canonical class must not follow a support camera")
        self.assertEqual(gw1.observations[LUT].classification,
                         C.CLASS_BRAKE_VAN)
        self.assertEqual(gw1.classification_disagreements()[LUT],
                         C.CLASS_BRAKE_VAN)
        self.assertTrue(r.invariant_holds,
                        "a classification disagreement is not a violation")

    def test_the_disagreement_is_reported_not_resolved(self):
        st = _state(_roster(1))
        segs = _segments(LUT, [(0.0, 4.0)])
        segs[0].label = C.CLASS_BRAKE_VAN
        r = audit(st, local_segments={LUT: segs})
        self.assertIn("CLASS_DISAGREEMENT", r.disagreements)
        self.assertEqual(r.per_camera_stats[LUT]["class_disagreements"], 1)


class TestMappingStatusesRemainAuditable(unittest.TestCase):
    def test_statuses_are_visible_per_observation(self):
        st = _state(_roster(3))
        r = audit(st, local_segments={LU: _segments(LU, [(0.0, 4.0)])})
        statuses = {w.canonical_wagon_id: w.observations[LU].mapping_status
                    for w in r.wagons}
        self.assertIn(statuses["GW_1"], ("EXACT", "MANY_TO_ONE",
                                         "ONE_TO_MANY"))
        self.assertEqual(statuses["GW_3"], "UNMATCHED")

    def test_unresolved_offset_is_surfaced_not_hidden(self):
        st = _state(_roster(2), offsets={
            RU: {"status": "REFERENCE", "delta": 0.0},
            LUT: {"status": "UNRESOLVED", "delta": 0.0}})
        r = audit(st, local_segments={LUT: _segments(LUT, [(0.0, 4.0)])})
        self.assertEqual(r.wagons[0].observations[LUT].mapping_status,
                         "UNRESOLVED_OFFSET")
        self.assertTrue(r.invariant_holds)

    def test_every_expected_disagreement_is_a_named_non_violation(self):
        for kind in EXPECTED_DISAGREEMENTS:
            with self.subTest(kind=kind):
                self.assertNotIn("VIOLATION", kind)

    def test_missing_camera_evidence_stays_missing(self):
        """Absent evidence must never be filled in from somewhere else."""
        r = audit(_state(_roster(2)))
        for w in r.wagons:
            obs = w.observations[LUT]
            self.assertEqual(obs.local_segment_ids, [])
            self.assertIsNone(obs.classification)
            self.assertEqual(obs.provenance, "projected from master timeline")


class TestViolationsAreActuallyDetected(unittest.TestCase):
    """Negative controls: a check that cannot fire proves nothing."""

    def test_a_gap_attributed_to_a_support_camera_is_caught(self):
        st = _state(_roster(3))
        st.global_gaps[0]["master_camera"] = LUT
        kinds = [v.kind for v in check_invariant(st)]
        self.assertIn("GAP_FROM_SUPPORT_CAMERA", kinds)

    def test_gap_count_divergence_is_caught(self):
        st = _state(_roster(3), checks={"right_up_final_gap_count": 2,
                                        "global_gap_count": 5})
        self.assertIn("GAP_COUNT_DIVERGED",
                      [v.kind for v in check_invariant(st)])

    def test_an_extra_observation_becoming_canonical_is_caught(self):
        wagons = _roster(3)
        st = _state(wagons, extras={RUT: [{"global_time": wagons[1].start_time}]})
        self.assertIn("EXTRA_BECAME_CANONICAL",
                      [v.kind for v in check_invariant(st)])

    def test_renumbering_is_caught(self):
        wagons = _roster(3)
        wagons[1] = GlobalWagon(
            global_id="GW_99", wagon_index=2, start_frame_master=60,
            end_frame_master=119, start_time=4.0, end_time=8.0,
            classification=C.CLASS_WAGON, classification_confidence=0.9)
        self.assertIn("ROSTER_RENUMBERED",
                      [v.kind for v in check_invariant(_state(wagons))])

    def test_reordering_is_caught(self):
        wagons = _roster(3)
        wagons[0], wagons[2] = wagons[2], wagons[0]
        kinds = [v.kind for v in check_invariant(_state(wagons))]
        self.assertTrue({"ROSTER_REORDERED", "ROSTER_RENUMBERED"} & set(kinds))

    def test_count_diverging_from_the_master_window_is_caught(self):
        st = _state(_roster(3), master_wagon_count=7)
        self.assertIn("COUNT_DIVERGED_FROM_MASTER_WINDOW",
                      [v.kind for v in check_invariant(st)])

    def test_audit_surfaces_violations_in_its_summary(self):
        st = _state(_roster(3), checks={"right_up_final_gap_count": 2,
                                        "global_gap_count": 5})
        r = audit(st)
        self.assertFalse(r.invariant_holds)
        self.assertTrue(any("VIOLATION" in ln for ln in r.summary_lines()))


class TestNonWagonRegions(unittest.TestCase):
    def test_engine_and_brakevan_never_receive_a_gw_id(self):
        """They are excluded upstream; the audit must not reintroduce them."""
        st = _state(_roster(3))
        st.wagon_window = {
            "leading_non_wagon_objects": [
                {"classification": C.CLASS_ENGINE, "start_time": -8.0,
                 "end_time": 0.0, "segment_index": 0}],
            "trailing_non_wagon_objects": [
                {"classification": C.CLASS_BRAKE_VAN, "start_time": 12.0,
                 "end_time": 16.0, "segment_index": 4}]}
        r = audit(st)
        self.assertEqual(r.canonical_wagon_count, 3)
        ids = [w.canonical_wagon_id for w in r.wagons]
        self.assertEqual(ids, ["GW_1", "GW_2", "GW_3"])
        for w in r.wagons:
            self.assertEqual(w.canonical_class, C.CLASS_WAGON)


class TestPartialCoverage(unittest.TestCase):
    def test_a_short_camera_does_not_remove_a_canonical_wagon(self):
        st = _state(_roster(5))
        r = audit(st, per_camera_meta={
            RU: {"fps": FPS, "total_frames": 3000},
            LUT: {"fps": FPS, "total_frames": 30}})       # 2 seconds only
        self.assertEqual(r.canonical_wagon_count, 5)
        for w in r.wagons:
            self.assertIn(LUT, w.observations)
        later = r.wagons[-1].observations[LUT]
        self.assertFalse(later.local_start_frame is not None
                         and later.coverage_status == "AVAILABLE")
        self.assertEqual(later.coverage_status, "AFTER_END")

    def test_coverage_reason_is_recorded(self):
        st = _state(_roster(5))
        r = audit(st, per_camera_meta={LUT: {"fps": FPS, "total_frames": 30}})
        self.assertIn("follows the end",
                      r.wagons[-1].observations[LUT].coverage_reason)


class TestBatchSequentialParity(unittest.TestCase):
    """Identical persisted evidence must yield an identical canonical roster.

    Both modes route through `assemble_global_train_state_master_fixed`, so the
    roster is a function of the persisted state alone. This pins that: the audit
    is deterministic and mode-independent.
    """

    def _roster_of(self, r):
        return [(w.canonical_wagon_id, w.master_start_time, w.master_end_time,
                 w.canonical_class) for w in r.wagons]

    def test_same_state_yields_the_same_roster(self):
        st = _state(_roster(6))
        self.assertEqual(self._roster_of(audit(st)),
                         self._roster_of(audit(st)))

    def test_local_segment_evidence_does_not_alter_the_roster(self):
        """Sequential supplies local segments; batch may not. Roster is equal."""
        st = _state(_roster(6))
        without = self._roster_of(audit(st))
        with_segs = self._roster_of(audit(st, local_segments={
            LU: _segments(LU, [(0.0, 4.0), (4.0, 9.0)]),
            LUT: _segments(LUT, [(0.0, 24.0)])}))
        self.assertEqual(without, with_segs,
                         "camera evidence changed the canonical roster")

    def test_both_modes_share_one_fusion_entry_point(self):
        import inspect
        from orchestrator import global_assembler
        from reconstruction import runner
        self.assertIn("assemble_global_train_state_master_fixed",
                      inspect.getsource(global_assembler.assemble))
        # batch reaches the same function through run_global_count.py
        self.assertIn("run_global_count", inspect.getsource(runner))


if __name__ == "__main__":
    unittest.main()
