"""The finalized global wagon roster contract.

Deterministic `GW_1..GW_N`, no duplicates, no gaps, and ONE roster shared by
all four cameras -- never a per-camera local count.

Every assertion is a relationship the engine must satisfy.  No expected wagon
count is written down anywhere in this file, so the suite cannot be satisfied
by fabricating a number.
"""

from __future__ import annotations

import unittest

from _engine_harness import (
    ALL_CAMERAS, CAMERA_LEFT_UP, CAMERA_LEFT_UP_TOP, CAMERA_RIGHT_UP,
    CAMERA_RIGHT_UP_TOP, as_v4_state, drifting_gap_times, run_counting_engine,
)
from core.global_state_loader import roster_fingerprint, verify_roster_integrity


class TestRosterIdentity(unittest.TestCase):
    def setUp(self):
        self.times = drifting_gap_times(14, start=25.0)
        self.engine_state, self.tracks = run_counting_engine(
            self.times,
            {CAMERA_LEFT_UP:      [t - 0.9 for t in self.times],
             CAMERA_RIGHT_UP_TOP: [t + 0.2 for t in self.times],
             CAMERA_LEFT_UP_TOP:  [t + 2.7 for t in self.times]},
        )
        self.state = as_v4_state(self.engine_state)

    def test_ids_are_contiguous_gw_1_to_n(self):
        n = len(self.state.wagons)
        self.assertGreater(n, 0, "engine produced an empty roster")
        self.assertEqual([w.global_id for w in self.state.wagons],
                         [f"GW_{i}" for i in range(1, n + 1)])

    def test_no_duplicate_ids(self):
        ids = [w.global_id for w in self.state.wagons]
        self.assertEqual(len(ids), len(set(ids)))

    def test_no_missing_ids(self):
        ids = {w.global_id for w in self.state.wagons}
        for i in range(1, len(self.state.wagons) + 1):
            self.assertIn(f"GW_{i}", ids)

    def test_wagon_index_matches_position(self):
        for pos, w in enumerate(self.state.wagons, start=1):
            self.assertEqual(w.wagon_index, pos)

    def test_total_wagons_matches_roster_length(self):
        self.assertEqual(self.state.total_wagons, len(self.state.wagons))

    def test_structural_integrity_check_is_clean(self):
        self.assertEqual(verify_roster_integrity(self.state), [])

    def test_boundaries_are_ordered_and_non_overlapping(self):
        prev_end = None
        for w in self.state.wagons:
            self.assertLessEqual(w.start_time, w.end_time)
            self.assertLessEqual(w.start_frame_master, w.end_frame_master)
            if prev_end is not None:
                self.assertGreaterEqual(w.start_time, prev_end - 1e-6,
                                        f"{w.global_id} overlaps its predecessor")
            prev_end = w.end_time


class TestDeterminism(unittest.TestCase):
    def test_same_input_yields_identical_roster(self):
        times = drifting_gap_times(11, start=18.0)
        supports = {CAMERA_LEFT_UP: [t - 1.4 for t in times],
                    CAMERA_RIGHT_UP_TOP: [t + 0.3 for t in times],
                    CAMERA_LEFT_UP_TOP: [t + 1.1 for t in times]}
        a = as_v4_state(run_counting_engine(times, supports)[0])
        b = as_v4_state(run_counting_engine(times, supports)[0])
        self.assertEqual(roster_fingerprint(a), roster_fingerprint(b))
        self.assertEqual(a.total_wagons, b.total_wagons)

    def test_support_camera_order_does_not_change_the_roster(self):
        times = drifting_gap_times(9, start=22.0)
        s1 = {CAMERA_LEFT_UP: [t - 0.5 for t in times],
              CAMERA_RIGHT_UP_TOP: [t + 0.4 for t in times],
              CAMERA_LEFT_UP_TOP: [t + 1.9 for t in times]}
        # Same evidence, different dict insertion order.
        s2 = {CAMERA_LEFT_UP_TOP: s1[CAMERA_LEFT_UP_TOP],
              CAMERA_RIGHT_UP_TOP: s1[CAMERA_RIGHT_UP_TOP],
              CAMERA_LEFT_UP: s1[CAMERA_LEFT_UP]}
        self.assertEqual(roster_fingerprint(as_v4_state(run_counting_engine(times, s1)[0])),
                         roster_fingerprint(as_v4_state(run_counting_engine(times, s2)[0])))


class TestMasterIsTheSoleCountingAuthority(unittest.TestCase):
    """No per-camera independent count may become the final authority."""

    def setUp(self):
        self.times = drifting_gap_times(12, start=30.0)

    def _count(self, supports):
        return as_v4_state(run_counting_engine(self.times, supports)[0]).total_wagons

    def test_count_is_independent_of_support_camera_evidence(self):
        baseline = self._count(
            {CAMERA_LEFT_UP: [t - 0.9 for t in self.times],
             CAMERA_RIGHT_UP_TOP: [t + 0.2 for t in self.times],
             CAMERA_LEFT_UP_TOP: [t + 2.7 for t in self.times]})

        # A support camera that saw nothing at all.
        self.assertEqual(self._count({}), baseline)
        # A support camera hallucinating extra gaps between the real ones.
        noisy = [t + 2.0 for t in self.times] + [t + 2.4 for t in self.times]
        self.assertEqual(
            self._count({CAMERA_LEFT_UP: [t - 0.9 for t in self.times],
                         CAMERA_RIGHT_UP_TOP: sorted(noisy),
                         CAMERA_LEFT_UP_TOP: [t + 2.7 for t in self.times]}),
            baseline,
            "a support camera changed the wagon count -- support cameras must "
            "contribute evidence only")
        # A support camera that missed half the train.
        self.assertEqual(
            self._count({CAMERA_LEFT_UP: [t - 0.9 for t in self.times[::2]]}),
            baseline)

    def test_global_gap_sequence_equals_validated_master_gaps(self):
        state, tracks = run_counting_engine(
            self.times, {CAMERA_LEFT_UP: [t - 0.9 for t in self.times]})
        checks = state.invariant_checks
        self.assertTrue(checks["invariant_holds"], checks.get("violations"))
        self.assertEqual(checks["right_up_final_gap_count"],
                         checks["global_gap_count"])
        self.assertEqual(checks["global_gap_count"],
                         len(tracks[CAMERA_RIGHT_UP].gaps))


class TestOneRosterForAllFourCameras(unittest.TestCase):
    def test_every_camera_uses_the_same_global_ids(self):
        times = drifting_gap_times(10, start=28.0)
        state, tracks = run_counting_engine(
            times,
            {CAMERA_LEFT_UP:      [t - 0.9 for t in times],
             CAMERA_RIGHT_UP_TOP: [t + 0.2 for t in times],
             CAMERA_LEFT_UP_TOP:  [t + 2.7 for t in times]})
        v4 = as_v4_state(state)
        roster_ids = [w.global_id for w in v4.wagons]

        # The production projection each camera uses to find its own frames.
        from materializer.wagon_cache_builder import _wagon_local_range
        offsets = v4.camera_time_offsets()
        for cam in ALL_CAMERAS:
            with self.subTest(camera=cam):
                t = tracks[cam]
                projected = [
                    w.global_id for w in v4.wagons
                    if _wagon_local_range(w, t.fps, t.total_frames,
                                          offsets.get(cam, 0.0))[1] >= 0
                ]
                # A camera may legitimately not see every wagon, but it must
                # never invent an id of its own.
                self.assertTrue(set(projected).issubset(set(roster_ids)))
                self.assertEqual(projected, [i for i in roster_ids
                                             if i in set(projected)],
                                 "camera reordered the global roster")

    def test_supporting_cameras_are_a_subset_of_the_four_known_cameras(self):
        times = drifting_gap_times(8, start=26.0)
        v4 = as_v4_state(run_counting_engine(
            times, {CAMERA_LEFT_UP: [t - 0.9 for t in times]})[0])
        for w in v4.wagons:
            self.assertTrue(set(w.supporting_cameras).issubset(set(ALL_CAMERAS)))


class TestWagonOnlyRoster(unittest.TestCase):
    def test_roster_holds_wagons_only_and_structure_is_still_reported(self):
        """Engines / brake vans get no GW id but stay visible to the reports."""
        times = drifting_gap_times(10, start=30.0)
        v4 = as_v4_state(run_counting_engine(times)[0])
        for w in v4.wagons:
            self.assertNotIn(w.classification, ("ENGINE", "BRAKE_VAN"))
        # wagon_window is the structure metadata the reporting KPIs read back.
        self.assertIn("master_wagon_count", v4.wagon_window)
        self.assertEqual(v4.master_wagon_count, v4.total_wagons)


if __name__ == "__main__":
    unittest.main()
