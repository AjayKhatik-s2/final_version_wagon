"""Adaptive local train-motion reference and train-stop handling.

Scope note. The full development video contains ZERO speed-based rejections, so
these tests exercise the mechanism on synthetic motion profiles the local train
does not contain (a genuine stop, hard acceleration). The local train IS a
decelerating train -- validated speed falls 560 -> 312 px/s across one pass --
and the local reference tracks that better than the global median (worst
deviation 1.24x vs 1.61x) while changing zero decisions.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gap_validation as gval
from global_train_state import CAMERA_LEFT_UP_TOP, CAMERA_RIGHT_UP, GapEvent

FPS = 15.0
WIDTH = 848


def track(tid, start_frame, span, x_start, x_end, *, conf=0.90,
          camera_id=CAMERA_RIGHT_UP, fps=FPS):
    """A confirmed track moving from x_start to x_end over `span` frames."""
    n = span + 1
    frames = list(range(start_frame, start_frame + n))
    xs = ([x_start] * n if n == 1 else
          [x_start + (x_end - x_start) * i / (n - 1) for i in range(n)])
    return GapEvent(track_id=tid, camera_id=camera_id, start_frame=frames[0],
                    end_frame=frames[-1], confidence=conf, hit_count=n,
                    center_x_trajectory=xs, fps=fps,
                    temporal_consistency_score=1.0, hit_frames=frames,
                    bbox_history=[[x - 20, 100, x + 20, 300] for x in xs])


def validate(gaps, cam=CAMERA_RIGHT_UP, cfg=None, width=WIDTH, fps=FPS):
    return gval.validate_gap_events(gaps, cam, cfg or gval.GapValidationConfig(),
                                    verbose=False, frame_width=width, fps=fps)


def ids(res):
    return sorted(g.track_id for g in res.accepted)


def reasons(res):
    return {r.features.track_id: r.reason for r in res.rejected}


# ===========================================================================
# constant speed / acceleration / deceleration
# ===========================================================================

class TestSpeedProfiles(unittest.TestCase):
    def _sequence(self, spans):
        """Build consecutive gaps whose crossing time is given by `spans`."""
        gaps, frame = [], 100
        for i, span in enumerate(spans, start=1):
            gaps.append(track(i, frame, span, 700.0, 250.0))
            frame += span + 60
        return gaps

    def test_constant_speed_all_accepted(self):
        res = validate(self._sequence([12] * 12))
        self.assertEqual(len(res.accepted), 12)
        self.assertEqual(res.rejected, [])

    def test_accelerating_train_all_accepted(self):
        """Crossing gets faster: span shrinks 20 -> 8 frames."""
        res = validate(self._sequence([20, 19, 17, 16, 14, 13, 12, 11, 10, 9, 8, 8]))
        self.assertEqual(len(res.accepted), 12, reasons(res))
        self.assertEqual(res.train_motion_state, "ACCELERATING")

    def test_decelerating_train_all_accepted(self):
        """The real local pattern: crossing gets slower."""
        res = validate(self._sequence([8, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 20]))
        self.assertEqual(len(res.accepted), 12, reasons(res))
        self.assertEqual(res.train_motion_state, "DECELERATING")

    def test_severe_deceleration_survives_via_local_reference(self):
        """A 5x slowdown across the pass would breach a GLOBAL 4x tolerance at
        the ends, but each gap is close to its NEIGHBOURS."""
        spans = [6, 6, 7, 8, 9, 11, 13, 15, 18, 21, 25, 28, 30, 30]
        res = validate(self._sequence(spans))
        self.assertEqual(len(res.accepted), len(spans), reasons(res))
        kinds = {f.motion_reference_kind for f in res.features
                 if f.motion_reference_kind}
        self.assertIn("local", kinds, "the local reference must be in use")

    def test_global_reference_is_still_reported(self):
        res = validate(self._sequence([12] * 8))
        self.assertIsNotNone(res.train_reference_speed)

    def test_within_track_speed_variation_is_tolerated(self):
        """Real gaps are not constant-speed inside one track."""
        xs = [700, 690, 672, 648, 620, 588, 552, 516, 482, 452, 428, 410, 400]
        g = GapEvent(track_id=1, camera_id=CAMERA_RIGHT_UP, start_frame=100,
                     end_frame=112, confidence=0.9, hit_count=13,
                     center_x_trajectory=[float(x) for x in xs], fps=FPS,
                     hit_frames=list(range(100, 113)))
        self.assertEqual(len(validate([g]).accepted), 1)


# ===========================================================================
# train stop and resume
# ===========================================================================

class TestTrainStop(unittest.TestCase):
    def _moving(self, n=8, start=100):
        gaps, frame = [], start
        for i in range(1, n + 1):
            gaps.append(track(i, frame, 12, 700.0, 250.0))
            frame += 72
        return gaps, frame

    def test_corroborated_stop_preserves_stalled_gaps(self):
        """Several confirmed tracks stalling TOGETHER means the train stopped."""
        gaps, frame = self._moving()
        # three gaps stall over the same frames, having previously moved
        for i, off in enumerate((0, 2, 4), start=100):
            gaps.append(track(i, frame + off, 40, 500.0 - i * 0.0, 500.0))
        res = validate(gaps)
        self.assertTrue(res.train_stopped_detected,
                        "simultaneous stalls must read as a train stop")
        self.assertEqual(res.train_motion_state, "STOPPED")
        paused = [f for f in res.features if f.motion_paused]
        self.assertGreaterEqual(len(paused), 1,
                                "stalled tracks must be marked MOTION_PAUSED")

    def test_isolated_stalled_track_is_still_rejected(self):
        """The measured false-positive signature must stay rejected."""
        gaps, frame = self._moving()
        gaps.append(track(99, frame, 40, 500.0, 500.0, conf=0.93))
        res = validate(gaps)
        self.assertFalse(res.train_stopped_detected)
        self.assertNotIn(99, ids(res), "an isolated static track is an artefact")

    def test_stop_does_not_create_a_new_gap(self):
        """A stop must not change how many gap events exist."""
        gaps, frame = self._moving()
        before = len(validate(gaps).accepted)
        # Two DISTINCT physical gaps, both stalled over the same frames. They
        # must sit at different image positions -- two stalled tracks at the same
        # place would (correctly) be collapsed as duplicates.
        gaps.append(track(100, frame, 40, 620.0, 620.0))
        gaps.append(track(101, frame + 2, 40, 300.0, 300.0))
        after = validate(gaps)
        self.assertTrue(after.train_stopped_detected)
        self.assertEqual(len(after.accepted), before + 2,
                         "a stop preserves the stalled gaps; it invents none")

    def test_resume_after_stop_creates_no_duplicate(self):
        """Moving -> stalled -> moving again must not double-count."""
        gaps, frame = self._moving(6)
        gaps.append(track(50, frame, 40, 500.0, 500.0))
        gaps.append(track(51, frame + 45, 40, 500.0, 500.0))
        frame += 120
        for i in range(60, 64):
            gaps.append(track(i, frame, 12, 700.0, 250.0))
            frame += 72
        res = validate(gaps)
        # every distinct track is one event; none is duplicated
        self.assertEqual(len(ids(res)), len(set(ids(res))))
        self.assertNotIn(gval.REJECTED_DUPLICATE, set(reasons(res).values()))

    def test_stop_corroboration_can_be_disabled(self):
        cfg = gval.GapValidationConfig(stop_corroboration_min_tracks=0)
        gaps, frame = self._moving()
        for i in (100, 101, 102):
            gaps.append(track(i, frame + i - 100, 40, 500.0, 500.0))
        res = validate(gaps, cfg=cfg)
        self.assertFalse(res.train_stopped_detected)

    def test_stalled_tracks_must_overlap_in_time_to_corroborate(self):
        """Stalls spread across the whole video are artefacts, not one stop."""
        gaps, frame = self._moving()
        gaps.append(track(100, 120, 30, 400.0, 400.0))
        gaps.append(track(101, frame + 400, 30, 600.0, 600.0))
        res = validate(gaps)
        self.assertFalse(res.train_stopped_detected,
                         "non-overlapping stalls are not a train stop")


# ===========================================================================
# local vs global reference, and fallback
# ===========================================================================

class TestMotionReference(unittest.TestCase):
    def test_local_reference_used_when_enough_neighbours(self):
        gaps, frame = [], 100
        for i in range(1, 13):
            gaps.append(track(i, frame, 12, 700.0, 250.0))
            frame += 72
        res = validate(gaps)
        local = [f for f in res.features if f.motion_reference_kind == "local"]
        self.assertGreater(len(local), 0)
        for f in local:
            self.assertIsNotNone(f.motion_reference_speed)

    def test_falls_back_to_global_with_insufficient_history(self):
        cfg = gval.GapValidationConfig(motion_reference_window=0)
        gaps, frame = [], 100
        for i in range(1, 9):
            gaps.append(track(i, frame, 12, 700.0, 250.0))
            frame += 72
        res = validate(gaps, cfg=cfg)
        self.assertEqual(len(res.accepted), 8)
        kinds = {f.motion_reference_kind for f in res.features
                 if f.motion_reference_kind}
        self.assertNotIn("local", kinds)

    def test_too_few_tracks_skips_the_motion_check_entirely(self):
        res = validate([track(1, 100, 12, 700.0, 250.0),
                        track(2, 200, 12, 700.0, 250.0)])
        self.assertEqual(len(res.accepted), 2)
        self.assertEqual(res.train_motion_state, "UNKNOWN")

    def test_a_single_candidate_cannot_define_train_motion(self):
        """A track is never its own reference: the median excludes itself.

        Speeds must VARY for this to be observable -- with identical synthetic
        speeds the neighbour median coincides with the candidate's own speed.
        """
        spans = [10, 11, 12, 13, 14, 15, 16, 17]
        gaps, frame = [], 100
        for i, span in enumerate(spans, start=1):
            gaps.append(track(i, frame, span, 700.0, 250.0))
            frame += span + 60
        res = validate(gaps)
        checked = 0
        for f in res.features:
            if f.motion_reference_kind == "local" and f.motion_reference_speed:
                self.assertNotAlmostEqual(f.motion_reference_speed,
                                          f.velocity_px_per_sec, places=3)
                checked += 1
        self.assertGreater(checked, 0, "no local reference was exercised")


# ===========================================================================
# the protections must all survive
# ===========================================================================

class TestProtectionsPreserved(unittest.TestCase):
    def _moving(self, n=8):
        gaps, frame = [], 100
        for i in range(1, n + 1):
            gaps.append(track(i, frame, 12, 700.0, 250.0))
            frame += 72
        return gaps, frame

    def test_static_artefact_still_rejected(self):
        g = track(1, 376, 29, 436.0, 436.0, conf=0.93,
                  camera_id=CAMERA_LEFT_UP_TOP)
        self.assertEqual(validate([g], CAMERA_LEFT_UP_TOP).accepted, [])

    def test_raw_one_frame_detection_still_rejected(self):
        g = GapEvent(track_id=1, camera_id=CAMERA_RIGHT_UP, start_frame=100,
                     end_frame=100, confidence=0.99, hit_count=1,
                     center_x_trajectory=[500.0], fps=FPS, hit_frames=[100])
        self.assertEqual(validate([g]).accepted, [])

    def test_wrong_direction_still_rejected(self):
        gaps, frame = self._moving()
        gaps.append(track(99, frame, 12, 250.0, 700.0))     # against the flow
        res = validate(gaps)
        self.assertEqual(reasons(res).get(99), gval.REJECTED_WRONG_DIRECTION)

    def test_duplicate_still_rejected(self):
        a = track(1, 186, 12, 518.0, 259.0)
        b = track(2, 188, 5, 501.0, 390.0)
        res = validate([a, b])
        self.assertEqual(len(res.accepted), 1)
        self.assertEqual(reasons(res)[2], gval.REJECTED_DUPLICATE)

    def test_stop_tolerance_does_not_admit_a_static_artefact_alone(self):
        """Even with a real stop elsewhere, an unrelated isolated static track
        must not ride along."""
        gaps, frame = self._moving()
        for i in (100, 101):
            gaps.append(track(i, frame + i - 100, 40, 500.0, 500.0))
        gaps.append(track(200, 130, 30, 300.0, 300.0))   # elsewhere, alone
        res = validate(gaps)
        self.assertTrue(res.train_stopped_detected)
        # the far-away isolated stall does not overlap the stop window
        far = next((f for f in res.features if f.track_id == 200), None)
        self.assertIsNotNone(far)

    def test_config_stays_camera_independent(self):
        for key in gval.GapValidationConfig().describe():
            self.assertFalse(key.endswith("_px") or key.endswith("_frames"),
                             f"{key} reintroduces absolute units")

    def test_new_settings_scale_with_geometry(self):
        c = gval.GapValidationConfig()
        for w, f in ((640, 10.0), (848, 15.0), (1920, 30.0), (3840, 60.0)):
            r = c.resolve(w, f)
            self.assertGreater(r.min_motion_px, 0)
            self.assertGreater(r.min_separation_frames, 0)

    def test_independent_runs_do_not_leak_motion_state(self):
        fast, frame = [], 100
        for i in range(1, 9):
            fast.append(track(i, frame, 8, 700.0, 250.0))
            frame += 72
        slow, frame = [], 100
        for i in range(1, 9):
            slow.append(track(i, frame, 24, 700.0, 250.0))
            frame += 96
        a = validate(fast)
        b = validate(slow)
        c = validate(fast)
        self.assertEqual(len(a.accepted), len(c.accepted))
        self.assertAlmostEqual(a.train_reference_speed, c.train_reference_speed,
                               places=6)
        self.assertNotAlmostEqual(a.train_reference_speed,
                                  b.train_reference_speed, places=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
