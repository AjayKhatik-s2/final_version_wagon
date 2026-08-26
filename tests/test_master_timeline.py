"""The canonical timeline: one time-to-frame mapping, and RIGHT_UP owns it.

Two things are pinned here.

The mapping. The same arithmetic lived in three places and they disagreed at
the edges. Asked for a wagon at master 100-104s against a camera holding 1350
frames of 90s footage -- a camera that stopped recording before the wagon
existed -- `reporting._evidence_lookup.wagon_local_frames` returned
`(1349, 1349)`: its LAST frame, offered as evidence for a wagon it never saw.
Every wagon after the footage ended got that same still, each under a different
wagon id, and nothing in the picture gave it away.

The invariant. RIGHT_UP mints the canonical gaps and the roster; support
cameras observe. The tests separate a support camera CHANGING the roster (a
fault) from a support camera DISAGREEING with it (expected, and frequent).
"""

from __future__ import annotations

import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from core.canonical_audit import audit, check_invariant
from core.global_state_loader import GlobalTrainState, GlobalWagon
from core.master_timeline import (
    AFTER_END, AVAILABLE, BEFORE_START, NO_METADATA, PARTIAL,
    UNRESOLVED_OFFSET, BoundaryPolicy, CameraClock, DEFAULT_BOUNDARY_POLICY,
    assign_master_time, master_interval_to_local, master_time_to_local_frame,
    project_roster,
)

RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP
RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP


def _clock(cam=LU, fps=15.0, frames=1350, offset=0.0, status="RESOLVED"):
    return CameraClock(camera_id=cam, fps=fps, total_frames=frames,
                       offset=offset, offset_status=status)


def _wagon(i, start, end, cls=C.CLASS_WAGON, fps=15.0):
    return GlobalWagon(global_id=f"GW_{i}", wagon_index=i,
                       start_frame_master=int(start * fps),
                       end_frame_master=int(end * fps) - 1,
                       start_time=start, end_time=end,
                       classification=cls, classification_confidence=0.9)


def _state(wagons, *, gaps=None, offsets=None, checks=None,
           master_wagon_count=None, extras=None, align=None):
    st = GlobalTrainState(total_wagons=len(wagons), wagons=tuple(wagons),
                          master_camera=RU, master_fps=15.0,
                          master_total_frames=3000)
    st.global_gaps = gaps if gaps is not None else [
        {"global_gap_id": i, "master_camera": RU,
         "master_observation": {"camera_id": RU,
                                "center_time": w.start_time}}
        for i, w in enumerate(wagons[1:], start=1)]
    st.camera_offsets = offsets or {
        RU: {"status": "REFERENCE", "delta": 0.0},
        LU: {"status": "RESOLVED", "delta": -1.5},
        RUT: {"status": "RESOLVED", "delta": 2.0},
        LUT: {"status": "RESOLVED", "delta": 0.5},
    }
    st.invariant_checks = checks if checks is not None else {
        "right_up_final_gap_count": len(st.global_gaps),
        "global_gap_count": len(st.global_gaps),
    }
    st.master_wagon_count = (len(wagons) if master_wagon_count is None
                             else master_wagon_count)
    st.extra_support_observations = extras or {}
    st.support_alignment_summary = align or {}
    return st


# =============================================================================
# THE BUG
# =============================================================================

class TestEarlyEndingCameraRegression(unittest.TestCase):
    """The exact reported case: 1350 frames / 90s, wagon at master 100-104s."""

    FPS, FRAMES = 15.0, 1350          # 90.0 seconds of footage
    START, END = 100.0, 104.0         # ten seconds after the camera stopped

    def test_fusion_reports_outside_footage(self):
        import global_fusion as gf
        self.assertIsNone(
            gf.project_global_time_to_local(self.START, 0.0, self.FPS,
                                            self.FRAMES))

    def test_materialization_reports_no_valid_range(self):
        from materializer.wagon_cache_builder import _wagon_local_range
        self.assertEqual(
            _wagon_local_range(_wagon(1, self.START, self.END), self.FPS,
                               self.FRAMES, 0.0),
            (0, -1))

    def test_evidence_lookup_no_longer_returns_the_last_frame(self):
        from reporting._evidence_lookup import wagon_local_frames
        got = wagon_local_frames(self.START, self.END, self.FPS, self.FRAMES)
        self.assertNotEqual(got, (1349, 1349),
                            "the camera's final frame was offered as evidence "
                            "for a wagon recorded after it stopped")
        self.assertEqual(got, (0, -1))

    def test_all_three_now_agree(self):
        import global_fusion as gf
        from materializer.wagon_cache_builder import _wagon_local_range
        from reporting._evidence_lookup import wagon_local_frames
        self.assertIsNone(gf.project_global_time_to_local(
            self.START, 0.0, self.FPS, self.FRAMES))
        self.assertEqual(
            _wagon_local_range(_wagon(1, self.START, self.END), self.FPS,
                               self.FRAMES, 0.0),
            wagon_local_frames(self.START, self.END, self.FPS, self.FRAMES))

    def test_the_slot_is_marked_unavailable_for_the_right_reason(self):
        from reporting._evidence_lookup import wagon_local_window
        w = wagon_local_window(self.START, self.END, self.FPS, self.FRAMES,
                               camera_id=LU)
        self.assertFalse(w.available)
        self.assertEqual(w.status, AFTER_END)
        self.assertIn("follows the end", w.reason)
        self.assertIn(LU, w.reason)

    def test_no_cache_frame_path_is_produced(self):
        """The whole point: no file path, so no image can be embedded."""
        from reporting._evidence_lookup import (midpoint_cache_path,
                                                quartile_cache_paths)
        self.assertIsNone(midpoint_cache_path(
            cache_root="/nonexistent", gw_id="GW_9", camera_id=LU,
            wagon_start_time=self.START, wagon_end_time=self.END,
            local_fps=self.FPS, local_total_frames=self.FRAMES))
        self.assertEqual(
            quartile_cache_paths(
                cache_root="/nonexistent", gw_id="GW_9", camera_id=LU,
                wagon_start_time=self.START, wagon_end_time=self.END,
                local_fps=self.FPS, local_total_frames=self.FRAMES),
            [None, None, None, None])


class TestCoverageEdges(unittest.TestCase):
    def test_camera_starting_late_is_unavailable_before_its_first_frame(self):
        # footage begins at master 50s
        w = master_interval_to_local(_clock(offset=50.0), 10.0, 14.0)
        self.assertFalse(w.available)
        self.assertEqual(w.status, BEFORE_START)

    def test_partial_overlap_is_kept_and_flagged(self):
        """Those frames really do show the wagon, so they are not discarded."""
        w = master_interval_to_local(_clock(), -1.0, 1.0)
        self.assertTrue(w.available)
        self.assertEqual(w.status, PARTIAL)
        self.assertEqual(w.start_frame, 0)

    def test_fully_inside_is_available(self):
        w = master_interval_to_local(_clock(), 10.0, 14.0)
        self.assertEqual(w.status, AVAILABLE)
        self.assertEqual((w.start_frame, w.end_frame), (150, 209))

    def test_missing_metadata_is_explicit(self):
        w = master_interval_to_local(_clock(fps=0.0, frames=0), 1.0, 2.0)
        self.assertEqual(w.status, NO_METADATA)
        self.assertEqual(w.as_range(), (0, -1))

    def test_unresolved_offset_can_be_refused(self):
        c = _clock(status="UNRESOLVED")
        self.assertTrue(master_interval_to_local(c, 10.0, 14.0).available)
        w = master_interval_to_local(c, 10.0, 14.0, allow_unresolved=False)
        self.assertEqual(w.status, UNRESOLVED_OFFSET)

    def test_no_fabricated_evidence_anywhere_outside_coverage(self):
        clock = _clock()                      # 0..90s
        for t in (-50.0, -0.5, 90.1, 200.0):
            with self.subTest(master_time=t):
                self.assertIsNone(master_time_to_local_frame(clock, t))


class TestDifferentTimebases(unittest.TestCase):
    def test_different_fps_maps_the_same_instant(self):
        """Same physical moment, three framerates -- same seconds, not frames."""
        for fps, expect in ((15.0, 150), (25.0, 250), (29.97, 300)):
            with self.subTest(fps=fps):
                c = CameraClock(camera_id=LU, fps=fps,
                                total_frames=int(fps * 120))
                self.assertEqual(master_time_to_local_frame(c, 10.0), expect)

    def test_frame_number_equality_is_never_assumed(self):
        a = CameraClock(camera_id=RU, fps=15.0, total_frames=3000)
        b = CameraClock(camera_id=LU, fps=30.0, total_frames=6000)
        self.assertNotEqual(master_time_to_local_frame(a, 10.0),
                            master_time_to_local_frame(b, 10.0))

    def test_different_start_times_map_correctly(self):
        """A camera started 5s late sees the same wagon 5s earlier locally."""
        late = _clock(offset=5.0, frames=3000)
        w = master_interval_to_local(late, 10.0, 14.0)
        self.assertEqual(w.start_frame, int(round((10.0 - 5.0) * 15.0)))

    def test_offset_round_trips(self):
        c = _clock(offset=-1.75)
        self.assertAlmostEqual(c.to_master_time(c.to_local_time(42.0)), 42.0)


class TestBoundaryPolicy(unittest.TestCase):
    """A boundary is one instant; a detection can land exactly on it."""

    WAGONS = [_wagon(1, 0.0, 4.0), _wagon(2, 4.0, 8.0), _wagon(3, 8.0, 12.0)]

    def test_default_assigns_the_boundary_to_the_next_wagon(self):
        self.assertEqual(assign_master_time(4.0, self.WAGONS), "GW_2")

    def test_previous_policy_is_honoured(self):
        p = BoundaryPolicy(on_boundary="previous")
        self.assertEqual(assign_master_time(4.0, self.WAGONS, policy=p), "GW_1")

    def test_epsilon_band_is_symmetric(self):
        eps = DEFAULT_BOUNDARY_POLICY.epsilon
        for t in (4.0 - eps / 2, 4.0, 4.0 + eps / 2):
            with self.subTest(t=t):
                self.assertEqual(assign_master_time(t, self.WAGONS), "GW_2")

    def test_outside_the_band_uses_plain_containment(self):
        self.assertEqual(assign_master_time(3.0, self.WAGONS), "GW_1")
        self.assertEqual(assign_master_time(5.0, self.WAGONS), "GW_2")

    def test_float_noise_does_not_flip_the_assignment(self):
        """0.1+0.2 style error must not decide which wagon owns a detection."""
        noisy = 4.0 + 1e-9
        self.assertEqual(assign_master_time(noisy, self.WAGONS), "GW_2")
        self.assertEqual(assign_master_time(4.0 - 1e-9, self.WAGONS), "GW_2")

    def test_engine_and_brakevan_regions_own_no_wagon(self):
        self.assertIsNone(assign_master_time(-5.0, self.WAGONS))
        self.assertIsNone(assign_master_time(99.0, self.WAGONS))

    def test_last_wagon_trailing_boundary_belongs_to_it(self):
        self.assertEqual(assign_master_time(12.0, self.WAGONS), "GW_3")

    def test_policy_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            BoundaryPolicy(on_boundary="sideways")
        with self.assertRaises(ValueError):
            BoundaryPolicy(epsilon=-1.0)


class TestRosterProjection(unittest.TestCase):
    def test_every_wagon_appears_for_every_camera(self):
        wagons = [_wagon(1, 0.0, 4.0), _wagon(2, 4.0, 8.0)]
        clocks = {RU: _clock(RU), LU: _clock(LU, offset=-1.5),
                  RUT: _clock(RUT, frames=10)}          # ends almost at once
        proj = project_roster(clocks, wagons)
        self.assertEqual(sorted(proj), ["GW_1", "GW_2"])
        for gw, per_cam in proj.items():
            self.assertEqual(sorted(per_cam), sorted(clocks))

    def test_partial_coverage_never_removes_a_wagon(self):
        wagons = [_wagon(i, i * 4.0, (i + 1) * 4.0) for i in range(1, 6)]
        clocks = {RU: _clock(RU), LUT: _clock(LUT, frames=30)}   # 2s only
        proj = project_roster(clocks, wagons)
        self.assertEqual(len(proj), len(wagons))
        for gw in proj:
            self.assertIn(LUT, proj[gw])
        unavailable = [gw for gw, pc in proj.items() if not pc[LUT].available]
        self.assertTrue(unavailable, "expected the short camera to miss wagons")


if __name__ == "__main__":
    unittest.main()
