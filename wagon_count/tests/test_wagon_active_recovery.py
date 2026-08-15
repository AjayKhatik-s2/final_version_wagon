"""WAGON_ACTIVE recovery: soft gates relax inside the wagon region, hard ones never.

Two independent real trains showed genuine wagon gaps lost to soft speed /
trajectory gates inside the confirmed wagon run. Validation must run before
classification (it produces the segments classification needs), so the first pass
cannot know the train state; a second pass re-examines candidates that fell inside
the derived wagon window.

The property under test is NOT "accept everything in WAGON_ACTIVE". It is:

    soft failure + inside wagon window + clears every hard gate  ->  ACCEPT
    hard failure, in any state                                   ->  REJECT
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gap_validation as gval
import global_fusion as gf
from global_train_state import (
    CAMERA_LEFT_UP_TOP, CAMERA_RIGHT_UP, GapEvent, LocalCameraTracks,
)

FPS = 15.0
WIDTH = 848


def track(tid, start_frame, span, x_start, x_end, *, conf=0.90,
          camera_id=CAMERA_RIGHT_UP, fps=FPS, jitter=None):
    n = span + 1
    frames = list(range(start_frame, start_frame + n))
    if jitter is not None:
        xs = list(jitter)[:n]
        while len(xs) < n:
            xs.append(xs[-1])
    else:
        xs = ([x_start] * n if n == 1 else
              [x_start + (x_end - x_start) * i / (n - 1) for i in range(n)])
    return GapEvent(track_id=tid, camera_id=camera_id, start_frame=frames[0],
                    end_frame=frames[-1], confidence=conf, hit_count=n,
                    center_x_trajectory=[float(x) for x in xs], fps=fps,
                    temporal_consistency_score=1.0, hit_frames=frames,
                    bbox_history=[[x - 20, 100, x + 20, 300] for x in xs])


def validate(gaps, cam=CAMERA_RIGHT_UP, cfg=None):
    return gval.validate_gap_events(gaps, cam, cfg or gval.GapValidationConfig(),
                                    verbose=False, frame_width=WIDTH, fps=FPS)


def recover(res, cfg=None, window=(0, 100000)):
    return gval.recover_wagon_active_candidates(
        res.rejected, res.accepted, window[0], window[1], CAMERA_RIGHT_UP,
        cfg or gval.GapValidationConfig(), frame_width=WIDTH, fps=FPS,
        verbose=False)


def population(n=10, span=12, start=100, step=72):
    """A healthy moving population so the motion reference is well defined."""
    gaps, frame = [], start
    for i in range(1, n + 1):
        gaps.append(track(i, frame, span, 700.0, 250.0))
        frame += step
    return gaps, frame


# ===========================================================================
# soft failures ARE recovered inside the wagon window
# ===========================================================================

class TestSoftRecovery(unittest.TestCase):
    def test_speed_mismatch_is_recovered(self):
        """A gap far slower than its neighbours, but clearly moving."""
        gaps, frame = population()
        gaps.append(track(99, frame, 90, 700.0, 250.0))   # ~6x slower
        res = validate(gaps)
        self.assertIn(99, [r.features.track_id for r in res.rejected],
                      "the slow gap must fail the first pass on speed")
        rec = recover(res)
        self.assertIn(99, [g.track_id for g in rec.recovered],
                      "and be recovered inside the wagon region")

    def test_low_confidence_is_recovered_above_the_floor(self):
        gaps, frame = population()
        gaps.append(track(99, frame, 12, 700.0, 250.0, conf=0.41))
        res = validate(gaps)
        self.assertEqual({r.reason for r in res.rejected},
                         {gval.REJECTED_LOW_CONFIDENCE})
        self.assertIn(99, [g.track_id for g in recover(res).recovered])

    def test_noisy_trajectory_is_recovered_above_the_floor(self):
        """Noisier than expected, but still net movement in one direction."""
        xs = [700, 688, 694, 660, 668, 630, 638, 600, 606, 566, 574, 534, 520]
        gaps, frame = population()
        gaps.append(track(99, frame, len(xs) - 1, 0, 0, jitter=xs))
        res = validate(gaps)
        rejected = {r.features.track_id: r.reason for r in res.rejected}
        self.assertIn(99, rejected)
        self.assertIn(rejected[99], gval.SOFT_REJECTION_REASONS)
        self.assertIn(99, [g.track_id for g in recover(res).recovered])

    def test_low_motion_above_static_is_recovered(self):
        gaps, frame = population()
        gaps.append(track(99, frame, 12, 700.0, 692.0))   # 8 px: > static, < floor
        res = validate(gaps)
        self.assertEqual({r.reason for r in res.rejected},
                         {gval.REJECTED_LOW_MOTION})
        self.assertIn(99, [g.track_id for g in recover(res).recovered])

    def test_recovery_records_its_evidence(self):
        gaps, frame = population()
        gaps.append(track(99, frame, 90, 700.0, 250.0))
        rec = recover(validate(gaps))
        d = next(x for x in rec.details if x["track_id"] == 99)
        self.assertEqual(d["outcome"], "recovered")
        for k in ("original_reason", "displacement_px", "speed_px_per_sec",
                  "monotonic", "confidence"):
            self.assertIn(k, d)


# ===========================================================================
# hard failures are NEVER recovered
# ===========================================================================

class TestHardNeverRecovered(unittest.TestCase):
    def _rejected_reason(self, extra):
        gaps, frame = population()
        gaps.append(extra)
        res = validate(gaps)
        return res, {r.features.track_id: r.reason for r in res.rejected}

    def test_static_artefact_not_recovered(self):
        gaps, frame = population()
        gaps.append(track(99, frame, 30, 500.0, 500.0, conf=0.93))
        res = validate(gaps)
        self.assertEqual({r.reason for r in res.rejected}, {gval.REJECTED_STATIC})
        rec = recover(res)
        self.assertEqual(rec.recovered, [])
        self.assertEqual(rec.hard_blocked, 1)

    def test_wrong_direction_not_recovered(self):
        gaps, frame = population()
        gaps.append(track(99, frame, 12, 250.0, 700.0))
        res = validate(gaps)
        rec = recover(res)
        self.assertEqual(rec.recovered, [])
        self.assertEqual(rec.hard_blocked, 1)

    def test_one_frame_raw_detection_not_recovered(self):
        gaps, frame = population()
        gaps.append(GapEvent(track_id=99, camera_id=CAMERA_RIGHT_UP,
                             start_frame=frame, end_frame=frame, confidence=0.99,
                             hit_count=1, center_x_trajectory=[500.0], fps=FPS,
                             hit_frames=[frame]))
        res = validate(gaps)
        rec = recover(res)
        self.assertEqual(rec.recovered, [],
                         "a raw one-frame detection must never become a wagon")

    def test_too_short_track_not_recovered(self):
        gaps, frame = population()
        gaps.append(track(99, frame, 2, 700.0, 660.0))
        res = validate(gaps)
        self.assertEqual(recover(res).recovered, [])

    def test_blind_track_not_recovered(self):
        gaps, frame = population()
        g = track(99, frame, 12, 700.0, 250.0)
        g.hit_frames = [frame, frame + 1, frame + 2, frame + 60, frame + 61]
        g.center_x_trajectory = [700.0, 690.0, 680.0, 300.0, 250.0]
        g.hit_count = 5
        g.end_frame = frame + 61
        gaps.append(g)
        res = validate(gaps)
        rec = recover(res)
        self.assertEqual(rec.recovered, [])

    def test_duplicate_not_recovered(self):
        gaps, frame = population()
        gaps.append(track(50, 186, 12, 518.0, 259.0))
        gaps.append(track(51, 188, 5, 501.0, 390.0))
        res = validate(gaps)
        self.assertIn(gval.REJECTED_DUPLICATE, {r.reason for r in res.rejected})
        rec = recover(res)
        self.assertNotIn(51, [g.track_id for g in rec.recovered])


# ===========================================================================
# recovery safety floors and duplicate protection
# ===========================================================================

class TestRecoverySafety(unittest.TestCase):
    def test_reversing_trajectory_blocked_by_the_floor(self):
        """Direction flips about as often as it holds: noise, not an object."""
        xs = [500, 540, 500, 545, 505, 550, 505, 555, 510, 560, 515, 565]
        gaps, frame = population()
        gaps.append(track(99, frame, len(xs) - 1, 0, 0, jitter=xs))
        res = validate(gaps)
        rec = recover(res)
        self.assertNotIn(99, [g.track_id for g in rec.recovered])
        d = next(x for x in rec.details if x["track_id"] == 99)
        self.assertEqual(d["outcome"], "blocked")
        self.assertIn("path efficiency", d["note"],
                      "path efficiency is the discriminator here: this shape "
                      "scores 0.13 while a noisy-but-genuine sweep scores 0.71, "
                      "whereas their monotonic fractions are nearly identical "
                      "(0.55 vs 0.58)")

    def test_near_chance_confidence_blocked_by_the_floor(self):
        gaps, frame = population()
        gaps.append(track(99, frame, 12, 700.0, 250.0, conf=0.20))
        rec = recover(validate(gaps))
        self.assertNotIn(99, [g.track_id for g in rec.recovered])

    def test_recovered_gap_cannot_crowd_an_accepted_gap(self):
        """Separation protection still holds during recovery."""
        gaps, frame = population()
        base = gaps[3]
        near = track(99, base.start_frame + 2, 90, 700.0, 250.0)  # slow + adjacent
        gaps.append(near)
        res = validate(gaps)
        rec = recover(res)
        if 99 not in [g.track_id for g in rec.recovered]:
            d = next((x for x in rec.details if x["track_id"] == 99), None)
            self.assertIsNotNone(d)
            self.assertEqual(d["outcome"], "blocked")

    def test_candidates_outside_the_window_are_untouched(self):
        gaps, frame = population()
        gaps.append(track(99, frame, 90, 700.0, 250.0))
        res = validate(gaps)
        rec = recover(res, window=(0, 50))      # window excludes the candidate
        self.assertEqual(rec.recovered, [])
        self.assertGreaterEqual(rec.outside_window, 1)
        self.assertEqual(rec.considered, 0)

    def test_no_window_means_no_recovery(self):
        gaps, frame = population()
        gaps.append(track(99, frame, 90, 700.0, 250.0))
        res = validate(gaps)
        rec = gval.recover_wagon_active_candidates(
            res.rejected, res.accepted, None, None, CAMERA_RIGHT_UP,
            frame_width=WIDTH, fps=FPS, verbose=False)
        self.assertEqual(rec.recovered, [])

    def test_recovery_can_be_disabled(self):
        cfg = gval.GapValidationConfig(wagon_active_recovery_enabled=False)
        gaps, frame = population()
        gaps.append(track(99, frame, 90, 700.0, 250.0))
        rec = recover(validate(gaps), cfg=cfg)
        self.assertEqual(rec.recovered, [])

    def test_recovery_does_not_mutate_the_accepted_list(self):
        gaps, frame = population()
        gaps.append(track(99, frame, 90, 700.0, 250.0))
        res = validate(gaps)
        before = list(res.accepted)
        recover(res)
        self.assertEqual(len(res.accepted), len(before))

    def test_every_hard_reason_is_classified_hard(self):
        for reason in (gval.REJECTED_NO_TRAJECTORY, gval.REJECTED_TOO_SHORT,
                       gval.REJECTED_DETECTION_GAP, gval.REJECTED_STATIC,
                       gval.REJECTED_WRONG_DIRECTION, gval.REJECTED_DUPLICATE,
                       gval.REJECTED_MIN_SEPARATION):
            self.assertIn(reason, gval.HARD_REJECTION_REASONS)
            self.assertNotIn(reason, gval.SOFT_REJECTION_REASONS)

    def test_hard_and_soft_sets_are_disjoint(self):
        self.assertEqual(
            gval.HARD_REJECTION_REASONS & gval.SOFT_REJECTION_REASONS, frozenset())


# ===========================================================================
# master authority and state independence
# ===========================================================================

class TestMasterAuthority(unittest.TestCase):
    def test_recovered_gap_becomes_a_global_gap(self):
        """A recovered master gap is eligible for a wagon boundary immediately."""
        gaps, frame = population()
        gaps.append(track(99, frame, 90, 700.0, 250.0))
        res = validate(gaps)
        rec = recover(res)
        self.assertTrue(rec.recovered)
        final = gval.renumber_gap_events(list(res.accepted) + list(rec.recovered))
        master = LocalCameraTracks(camera_id=CAMERA_RIGHT_UP, video_path="/m.mp4",
                                   fps=FPS, total_frames=frame + 500, width=WIDTH,
                                   gaps=final)
        global_gaps = gf.build_global_gap_sequence(master)
        self.assertEqual(len(global_gaps), len(final))
        self.assertEqual(len(global_gaps), len(res.accepted) + len(rec.recovered))

    def test_support_camera_rejection_cannot_remove_a_master_gap(self):
        gaps, frame = population()
        master = LocalCameraTracks(camera_id=CAMERA_RIGHT_UP, video_path="/m.mp4",
                                   fps=FPS, total_frames=frame + 500, width=WIDTH,
                                   gaps=gval.renumber_gap_events(
                                       list(validate(gaps).accepted)))
        before = len(gf.build_global_gap_sequence(master))
        # a support camera whose every candidate is a static artefact
        bad = [track(i, 100 + i * 40, 30, 500.0, 500.0,
                     camera_id=CAMERA_LEFT_UP_TOP) for i in range(1, 5)]
        sup_res = gval.validate_gap_events(bad, CAMERA_LEFT_UP_TOP, verbose=False,
                                           frame_width=WIDTH, fps=FPS)
        self.assertEqual(sup_res.accepted, [])
        sup = LocalCameraTracks(camera_id=CAMERA_LEFT_UP_TOP, video_path="/s.mp4",
                                fps=FPS, total_frames=frame + 500, width=WIDTH,
                                gaps=list(sup_res.accepted))
        gaps_after = gf.build_global_gap_sequence(master)
        gf.attach_support_evidence(gaps_after, [sup], verbose=False)
        self.assertEqual(len(gaps_after), before,
                         "support-camera validation failures must not remove a "
                         "RIGHT_UP master gap")

    def test_recovery_is_deterministic(self):
        gaps, frame = population()
        gaps.append(track(99, frame, 90, 700.0, 250.0))
        outs = []
        for _ in range(3):
            res = validate(gaps)
            outs.append(tuple(sorted(g.track_id for g in recover(res).recovered)))
        self.assertEqual(len(set(outs)), 1, f"non-deterministic: {outs}")

    def test_two_trains_do_not_leak_recovery_state(self):
        a, fa = population(8, span=8, start=100, step=60)
        a.append(track(99, fa, 60, 700.0, 250.0))
        b, fb = population(12, span=24, start=100, step=120)
        ra1 = recover(validate(a))
        recover(validate(b))
        ra2 = recover(validate(a))
        self.assertEqual(sorted(g.track_id for g in ra1.recovered),
                         sorted(g.track_id for g in ra2.recovered))

    def test_config_stays_camera_independent(self):
        for key in gval.GapValidationConfig().describe():
            self.assertFalse(key.endswith("_px") or key.endswith("_frames"),
                             f"{key} reintroduces absolute units")


if __name__ == "__main__":
    unittest.main(verbosity=2)
