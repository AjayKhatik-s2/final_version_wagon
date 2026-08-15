"""Unit tests for gap validation: a raw YOLO gap is a CANDIDATE, not a boundary.

Covers requirements 10-20 of the gap-validation brief:
static rejection, motion acceptance, missed detections, duplicates, low
confidence, implausible speed, inconsistent trajectory, wrong direction, and
perspective variation.

The synthetic tracks are built to match the MEASURED behaviour of the real
tracks (848x480 @ 15 fps): real gaps move 110-615 px at 74-555 px/s with
monotonic_fraction >= 0.94, while the three confirmed false positives on
LEFT_UP_TOP moved <= 0.2 px with confidence 0.93 and coverage 1.00.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gap_validation as gv
from global_train_state import CAMERA_LEFT_UP_TOP, CAMERA_RIGHT_UP, GapEvent

FPS = 15.0


def track(track_id, start_frame, n_hits, x_start, x_end, *,
          camera_id=CAMERA_RIGHT_UP, confidence=0.90, step=1,
          missing_after=None, missing_len=0, jitter=None, fps=FPS):
    """Build a GapEvent with an explicit trajectory.

    step         : frames between consecutive hits
    missing_after: insert a blind run after this many hits
    jitter       : optional explicit list of x positions (overrides the ramp)
    """
    frames, xs = [], []
    f = start_frame
    for i in range(n_hits):
        frames.append(f)
        f += step
        if missing_after is not None and i == missing_after - 1:
            f += missing_len
    if jitter is not None:
        xs = list(jitter)[:n_hits]
        while len(xs) < n_hits:
            xs.append(xs[-1] if xs else x_start)
    elif n_hits == 1:
        xs = [x_start]
    else:
        xs = [x_start + (x_end - x_start) * i / (n_hits - 1) for i in range(n_hits)]
    return GapEvent(
        track_id=track_id, camera_id=camera_id,
        start_frame=frames[0], end_frame=frames[-1],
        confidence=confidence, hit_count=n_hits,
        center_x_trajectory=xs, fps=fps, temporal_consistency_score=1.0,
        hit_frames=frames,
        bbox_history=[[x - 20, 100, x + 20, 300] for x in xs],
    )


def validate(gaps, cam=CAMERA_RIGHT_UP, cfg=None):
    return gv.validate_gap_events(gaps, cam, cfg or gv.GapValidationConfig(),
                                  verbose=False)


def reasons(res):
    return {r.features.track_id: r.reason for r in res.rejected}


# ===========================================================================
# static vs moving
# ===========================================================================

class TestStaticRejection(unittest.TestCase):
    def test_static_false_positive_is_rejected(self):
        """The measured real case: pinned centre, conf 0.93, 30 hits, coverage 1."""
        g = track(1, 376, 30, 436.0, 436.0, camera_id=CAMERA_LEFT_UP_TOP,
                  confidence=0.93)
        res = validate([g], CAMERA_LEFT_UP_TOP)
        self.assertEqual(res.accepted, [])
        self.assertEqual(reasons(res)[1], gv.REJECTED_STATIC)

    def test_high_confidence_does_not_rescue_a_static_track(self):
        g = track(1, 100, 30, 500.0, 500.0, confidence=0.99)
        res = validate([g])
        self.assertEqual(reasons(res)[1], gv.REJECTED_STATIC)

    def test_long_persistence_does_not_rescue_a_static_track(self):
        g = track(1, 100, 60, 500.0, 500.5)
        res = validate([g])
        self.assertEqual(reasons(res)[1], gv.REJECTED_STATIC)

    def test_moving_gap_is_accepted(self):
        """Matches a measured RIGHT_UP gap: 13 hits, 594 -> 156 px in 0.87 s."""
        g = track(1, 134, 13, 594.0, 156.0, confidence=0.81)
        res = validate([g])
        self.assertEqual([x.track_id for x in res.accepted], [1])
        self.assertEqual(res.rejected, [])

    def test_low_but_nonzero_motion_is_low_motion_not_static(self):
        g = track(1, 100, 12, 500.0, 508.0)      # 8 px: > static, < min_motion
        res = validate([g])
        self.assertEqual(reasons(res)[1], gv.REJECTED_LOW_MOTION)


# ===========================================================================
# temporal persistence and continuity
# ===========================================================================

class TestPersistence(unittest.TestCase):
    def test_single_frame_detection_is_rejected(self):
        g = track(1, 100, 1, 500.0, 500.0)
        res = validate([g])
        self.assertEqual(res.accepted, [])
        self.assertIn(reasons(res)[1],
                      (gv.REJECTED_NO_TRAJECTORY, gv.REJECTED_TOO_SHORT))

    def test_two_frame_detection_is_too_short(self):
        g = track(1, 100, 2, 600.0, 560.0)
        res = validate([g])
        self.assertEqual(reasons(res)[1], gv.REJECTED_TOO_SHORT)

    def test_one_missed_detection_keeps_a_single_track(self):
        """HIT HIT HIT MISS HIT HIT -> still one valid gap."""
        g = track(1, 100, 12, 620.0, 210.0, missing_after=6, missing_len=1)
        res = validate([g])
        self.assertEqual([x.track_id for x in res.accepted], [1])

    def test_long_blind_run_is_rejected(self):
        g = track(1, 100, 8, 620.0, 210.0, missing_after=4, missing_len=25)
        res = validate([g])
        self.assertEqual(reasons(res)[1], gv.REJECTED_DETECTION_GAP)

    def test_sparse_coverage_is_rejected(self):
        # 4 hits spread over ~60 frames -> coverage well under 0.20
        g = track(1, 100, 4, 620.0, 210.0, step=19)
        res = validate([g])
        self.assertEqual(reasons(res)[1], gv.REJECTED_DETECTION_GAP)


# ===========================================================================
# confidence, speed, trajectory, direction
# ===========================================================================

class TestMultiSignal(unittest.TestCase):
    def test_low_confidence_is_rejected(self):
        g = track(1, 100, 13, 594.0, 156.0, confidence=0.41)
        res = validate([g])
        self.assertEqual(reasons(res)[1], gv.REJECTED_LOW_CONFIDENCE)

    def test_implausibly_fast_is_rejected(self):
        # 600 px/s on an 848 px frame, expressed camera-independently
        cfg = gv.GapValidationConfig(max_motion_frac_per_sec=600.0 / 848)
        g = track(1, 100, 5, 0.0, 848.0, step=1)      # 848 px in 0.33 s
        res = validate([g], cfg=cfg)
        self.assertEqual(reasons(res)[1], gv.REJECTED_IMPLAUSIBLE_SPEED)

    def test_inconsistent_trajectory_is_rejected(self):
        # zig-zag: large net displacement but direction flips every step
        xs = [500, 540, 500, 545, 505, 550, 505, 555, 510, 560, 515, 565]
        g = track(1, 100, len(xs), 0, 0, jitter=xs)
        res = validate([g])
        self.assertEqual(reasons(res)[1], gv.REJECTED_INCONSISTENT_TRAJECTORY)

    def test_wrong_direction_is_rejected_against_camera_consensus(self):
        """Direction is derived per camera, never assumed."""
        gaps = [track(i, 100 + i * 60, 12, 620.0, 210.0) for i in range(1, 7)]
        gaps.append(track(7, 520, 12, 210.0, 620.0))     # against the flow
        res = validate(gaps)
        self.assertEqual(reasons(res).get(7), gv.REJECTED_WRONG_DIRECTION)
        self.assertEqual(len(res.accepted), 6)

    def test_opposite_direction_is_fine_when_the_whole_camera_agrees(self):
        """LEFT_UP_TOP gaps really do travel +x; that must not be penalised."""
        gaps = [track(i, 100 + i * 80, 14, 250.0, 570.0,
                      camera_id=CAMERA_LEFT_UP_TOP) for i in range(1, 7)]
        res = validate(gaps, CAMERA_LEFT_UP_TOP)
        self.assertEqual(len(res.accepted), 6)
        self.assertEqual(res.rejected, [])

    def test_speed_far_from_the_camera_median_is_rejected(self):
        gaps = [track(i, 100 + i * 60, 12, 620.0, 210.0) for i in range(1, 7)]
        # ~10x slower than the others, same direction
        gaps.append(track(7, 600, 40, 620.0, 580.0, step=2))
        res = validate(gaps)
        self.assertIn(reasons(res).get(7),
                      (gv.REJECTED_TRAIN_MOTION_MISMATCH, gv.REJECTED_LOW_MOTION,
                       gv.REJECTED_IMPLAUSIBLE_SPEED))
        self.assertEqual(len(res.accepted), 6)

    def test_perspective_variation_is_tolerated(self):
        """Valid gaps may have quite different apparent speeds."""
        gaps = [
            track(1, 100, 12, 620.0, 210.0),      # ~500 px/s
            track(2, 200, 18, 619.0, 277.0),      # ~285 px/s
            track(3, 300, 11, 615.0, 288.0),      # ~450 px/s
            track(4, 400, 9, 636.0, 22.0),        # ~1150 px/s
            track(5, 500, 16, 617.0, 205.0),      # ~386 px/s
            track(6, 600, 13, 594.0, 156.0),      # ~500 px/s
        ]
        res = validate(gaps)
        self.assertEqual(len(res.accepted), 6, reasons(res))


# ===========================================================================
# duplicates
# ===========================================================================

class TestDuplicates(unittest.TestCase):
    def test_repeated_detection_of_one_gap_is_one_event(self):
        """Five consecutive frames of one gap is ONE track and ONE GapEvent."""
        g = track(1, 100, 5, 600.0, 480.0)
        res = validate([g])
        self.assertEqual(len(res.accepted), 1)

    def test_two_overlapping_tracks_of_one_gap_collapse(self):
        """The measured RIGHT_UP_TOP case: trk3 (188-193) inside trk2 (186-198)."""
        a = track(2, 186, 10, 518.0, 259.0)
        b = track(3, 188, 6, 501.0, 390.0)
        res = validate([a, b])
        self.assertEqual(len(res.accepted), 1)
        self.assertEqual(res.accepted[0].track_id, 2, "keep the better-evidenced track")
        self.assertEqual(reasons(res)[3], gv.REJECTED_DUPLICATE)

    def test_two_separate_moving_gaps_stay_two_events(self):
        a = track(1, 100, 13, 620.0, 200.0)
        b = track(2, 160, 13, 620.0, 200.0)       # disjoint in time
        res = validate([a, b])
        self.assertEqual(len(res.accepted), 2)
        self.assertEqual(res.rejected, [])

    def test_far_apart_in_x_is_not_a_duplicate(self):
        """Two gaps visible simultaneously at different image positions."""
        a = track(1, 100, 12, 700.0, 400.0)
        b = track(2, 100, 12, 300.0, 20.0)
        res = validate([a, b])
        self.assertEqual(len(res.accepted), 2)


# ===========================================================================
# diagnostics and configuration
# ===========================================================================

class TestDiagnostics(unittest.TestCase):
    def test_every_rejection_records_reason_and_features(self):
        gaps = [track(1, 100, 30, 500.0, 500.0),          # static
                track(2, 200, 13, 594.0, 156.0, confidence=0.30),  # low conf
                track(3, 300, 2, 600.0, 560.0),           # too short
                track(4, 400, 13, 594.0, 156.0)]          # good
        res = validate(gaps)
        self.assertEqual(len(res.accepted), 1)
        self.assertEqual(len(res.rejected), 3)
        for r in res.rejected:
            self.assertIn(r.reason, gv.ALL_REJECTION_REASONS)
            self.assertTrue(r.detail)
            self.assertIsNotNone(r.features.velocity_px_per_sec)
        counts = res.rejection_counts
        self.assertEqual(sum(counts.values()), 3)

    def test_raw_tracked_valid_chain_is_reported(self):
        gaps = [track(1, 100, 30, 500.0, 500.0), track(2, 200, 13, 594.0, 156.0)]
        res = gv.validate_gap_events(gaps, CAMERA_RIGHT_UP, raw_detection_count=267,
                                     verbose=False)
        d = res.to_dict()
        self.assertEqual(d["raw_detections"], 267)
        self.assertEqual(d["tracked_candidates"], 2)
        self.assertEqual(d["valid_gap_events"], 1)
        self.assertEqual(d["rejected_total"], 1)

    def test_disabled_validation_passes_everything(self):
        cfg = gv.GapValidationConfig(enabled=False)
        gaps = [track(1, 100, 30, 500.0, 500.0)]          # a static FP
        res = validate(gaps, cfg=cfg)
        self.assertEqual(len(res.accepted), 1, "disabled means no filtering")

    def test_renumbering_restores_a_contiguous_rank(self):
        gaps = [track(1, 100, 13, 594.0, 156.0),
                track(5, 300, 13, 594.0, 156.0),
                track(9, 500, 13, 594.0, 156.0)]
        res = validate(gaps)
        out = gv.renumber_gap_events(res.accepted)
        self.assertEqual([g.track_id for g in out], [1, 2, 3])

    def test_config_is_fully_serializable(self):
        d = gv.GapValidationConfig().describe()
        for key in ("min_motion_frac", "static_max_motion_frac",
                    "train_motion_tolerance", "min_mean_confidence",
                    "max_detection_gap_seconds", "min_monotonic_fraction",
                    "min_separation_seconds"):
            self.assertIn(key, d)
        # ...and every distance/duration threshold must be camera-independent
        for key in d:
            self.assertFalse(key.endswith("_px") or key.endswith("_frames"),
                             f"{key} is in absolute units; thresholds must be "
                             f"frame-width fractions or seconds so they "
                             f"generalize across trains and camera geometry")

    def test_motion_features_match_the_measured_values(self):
        """Feature extraction must reproduce a real measured track."""
        g = track(1, 134, 13, 594.0, 156.0, confidence=0.81)
        f = gv.compute_motion_features(g)
        self.assertAlmostEqual(f.abs_displacement_px, 438.0, delta=1.0)
        self.assertAlmostEqual(f.duration_s, 13 / FPS, places=3)
        self.assertEqual(f.direction, -1)
        self.assertAlmostEqual(f.monotonic_fraction, 1.0, places=3)
        self.assertEqual(f.max_detection_gap, 0)
        self.assertAlmostEqual(f.coverage, 1.0, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
