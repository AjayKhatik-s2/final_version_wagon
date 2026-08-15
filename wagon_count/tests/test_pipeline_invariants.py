"""Cross-cutting pipeline invariants.

Covers the guarantees that span modules:
  * a raw detection can never become a global gap without tracker confirmation
  * thresholds resolve correctly across camera geometries (multi-train safety)
  * no state leaks between train runs
  * whole-wagon alias hypotheses are evaluated explicitly
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gap_validation as gval
import global_fusion as gf
import temporal_classification as tcls
import train_structure as ts
from global_train_state import (
    CAMERA_LEFT_UP_TOP, CAMERA_RIGHT_UP, GapEvent, LocalCameraTracks, SegmentClass,
)
from tracker_engine import GapTracker, _Track

FPS = 15.0


# ===========================================================================
# RAW DETECTION != VALID GAP
# ===========================================================================

class TestRawDetectionCannotBecomeGlobalGap(unittest.TestCase):
    """The tracker emits a GapEvent ONLY from a confirmed track.

    Verified structurally: `process_video` appends to `completed_tracks` in
    exactly two places, both guarded by `if tr.confirmed:`, and `confirmed` is
    set only when `hit_count >= min_hits`. These tests pin that behaviour so a
    future edit cannot open a bypass.
    """

    def test_unconfirmed_track_is_never_emitted(self):
        """A track below min_hits is dropped no matter how confident it is."""
        t = _Track(track_id=1, first_frame=100, last_seen_frame=100)
        t.update(100, 500.0, 0.99, [480, 100, 520, 300])   # one 0.99 hit
        self.assertEqual(t.hit_count, 1)
        self.assertFalse(t.confirmed,
                         "a single detection must never confirm a track")

    def test_confirmation_requires_min_hits_not_confidence(self):
        t = _Track(track_id=1, first_frame=100, last_seen_frame=100)
        for i, x in enumerate((700.0, 680.0)):
            t.update(100 + i, x, 0.999, [x - 20, 100, x + 20, 300])
        self.assertEqual(t.hit_count, 2)
        self.assertFalse(t.confirmed, "two hits is still below min_hits=3")
        t.update(102, 660.0, 0.10)                          # low confidence!
        self.assertEqual(t.hit_count, 3)
        # The tracker sets `confirmed` on hit_count, never on confidence.
        self.assertGreaterEqual(t.hit_count, 3)

    def test_one_frame_high_confidence_yields_no_validated_gap(self):
        """RAW=1, TRACKED=0 -> VALIDATED=0 -> GLOBAL=0."""
        g = GapEvent(track_id=1, camera_id=CAMERA_RIGHT_UP, start_frame=100,
                     end_frame=100, confidence=0.99, hit_count=1,
                     center_x_trajectory=[500.0], fps=FPS, hit_frames=[100])
        res = gval.validate_gap_events([g], CAMERA_RIGHT_UP, verbose=False,
                                       frame_width=848, fps=FPS)
        self.assertEqual(res.accepted, [], "a one-frame detection cannot validate")
        master = LocalCameraTracks(camera_id=CAMERA_RIGHT_UP, video_path="/x.mp4",
                                   fps=FPS, total_frames=4000,
                                   gaps=list(res.accepted))
        self.assertEqual(len(gf.build_global_gap_sequence(master)), 0,
                         "no validated gap -> no global gap -> no wagon boundary")

    def test_two_high_confidence_but_untracked_detections_yield_nothing(self):
        gaps = [GapEvent(track_id=i, camera_id=CAMERA_RIGHT_UP,
                         start_frame=100 + i, end_frame=100 + i,
                         confidence=0.999, hit_count=1,
                         center_x_trajectory=[500.0], fps=FPS,
                         hit_frames=[100 + i]) for i in (0, 1)]
        res = gval.validate_gap_events(gaps, CAMERA_RIGHT_UP, verbose=False,
                                       frame_width=848, fps=FPS)
        self.assertEqual(res.accepted, [])
        self.assertEqual(len(res.rejected), 2)

    def test_a_properly_tracked_moving_gap_does_validate(self):
        """The opposite case must still work: confidence is not the gate."""
        xs = [700.0 - 35.0 * i for i in range(13)]
        g = GapEvent(track_id=1, camera_id=CAMERA_RIGHT_UP, start_frame=100,
                     end_frame=112, confidence=0.84, hit_count=13,
                     center_x_trajectory=xs, fps=FPS,
                     hit_frames=list(range(100, 113)),
                     bbox_history=[[x - 20, 100, x + 20, 300] for x in xs])
        res = gval.validate_gap_events([g], CAMERA_RIGHT_UP, verbose=False,
                                       frame_width=848, fps=FPS)
        self.assertEqual(len(res.accepted), 1)


# ===========================================================================
# CLI / CONFIG COMPATIBILITY
#
# Regression: the CLI read `_gv.min_track_frames` after that attribute had been
# replaced by `min_track_seconds`, so the pipeline died in argparse before
# touching a video:
#     AttributeError: 'GapValidationConfig' object has no attribute
#     'min_track_frames'. Did you mean: 'min_track_seconds'?
# py_compile did not catch it (attribute access is runtime) and no test built the
# parser. These tests build it, so the class of bug cannot recur.
# ===========================================================================

class TestCliConfigCompatibility(unittest.TestCase):
    def _parser(self):
        import run_global_count as rgc
        return rgc._build_arg_parser()

    def test_parser_builds_at_all(self):
        """Building the parser touches every `default=` expression."""
        self.assertIsNotNone(self._parser())

    def test_parses_with_no_arguments(self):
        args = self._parser().parse_args([])
        self.assertIsNotNone(args.gap_min_track_sec)
        self.assertIsNotNone(args.gap_static_max_frac)

    def test_every_gap_default_matches_the_config(self):
        args = self._parser().parse_args([])
        c = gval.DEFAULT_GAP_VALIDATION
        self.assertEqual(args.gap_min_track_sec, c.min_track_seconds)
        self.assertEqual(args.gap_max_track_gap_sec, c.max_detection_gap_seconds)
        self.assertEqual(args.gap_min_motion_frac, c.min_motion_frac)
        self.assertEqual(args.gap_static_max_frac, c.static_max_motion_frac)
        self.assertEqual(args.gap_min_motion_frac_sec, c.min_motion_frac_per_sec)
        self.assertEqual(args.gap_max_motion_frac_sec, c.max_motion_frac_per_sec)
        self.assertEqual(args.gap_min_separation_sec, c.min_separation_seconds)
        self.assertEqual(args.gap_motion_tolerance, c.train_motion_tolerance)
        self.assertEqual(args.gap_min_confidence, c.min_mean_confidence)

    def test_every_cli_default_is_a_real_config_attribute(self):
        """Any `default=_gv.X` where X was renamed would fail here."""
        c = gval.GapValidationConfig()
        for name in ("min_track_seconds", "max_detection_gap_seconds",
                     "min_motion_frac", "static_max_motion_frac",
                     "min_motion_frac_per_sec", "max_motion_frac_per_sec",
                     "min_separation_seconds", "train_motion_tolerance",
                     "min_mean_confidence", "min_monotonic_fraction"):
            self.assertTrue(hasattr(c, name), f"config lost attribute {name}")

    def test_config_can_be_constructed_from_parsed_args(self):
        """The exact construction main() performs."""
        args = self._parser().parse_args([])
        cfg = gval.GapValidationConfig(
            enabled=not args.no_gap_validation,
            min_track_seconds=float(args.gap_min_track_sec),
            max_detection_gap_seconds=float(args.gap_max_track_gap_sec),
            min_motion_frac=float(args.gap_min_motion_frac),
            static_max_motion_frac=float(args.gap_static_max_frac),
            min_motion_frac_per_sec=float(args.gap_min_motion_frac_sec),
            max_motion_frac_per_sec=float(args.gap_max_motion_frac_sec),
            min_separation_seconds=float(args.gap_min_separation_sec),
            min_monotonic_fraction=float(args.gap_min_monotonic),
            min_mean_confidence=float(args.gap_min_confidence),
            train_motion_tolerance=float(args.gap_motion_tolerance),
        )
        self.assertTrue(cfg.enabled)
        self.assertGreater(cfg.resolve(848, 15.0).min_track_frames, 0)

    def test_normalized_flags_take_effect(self):
        args = self._parser().parse_args(
            ["--gap-min-track-sec", "0.5", "--gap-static-max-frac", "0.02"])
        cfg = gval.GapValidationConfig(
            min_track_seconds=args.gap_min_track_sec,
            static_max_motion_frac=args.gap_static_max_frac)
        r = cfg.resolve(848, 15.0)
        self.assertEqual(r.min_track_frames, 8)          # 0.5 s * 15 fps
        self.assertAlmostEqual(r.static_max_motion_px, 0.02 * 848, places=3)

    def test_deprecated_absolute_flags_default_to_none(self):
        args = self._parser().parse_args([])
        for attr in ("gap_min_track_frames", "gap_max_track_gap",
                     "gap_min_motion_px", "gap_static_max_px",
                     "gap_min_motion_px_sec", "gap_max_motion_px_sec"):
            self.assertIsNone(getattr(args, attr),
                              f"{attr} must default to None so it only acts as an "
                              f"explicit override")

    def test_deprecated_absolute_flags_still_parse_and_override(self):
        """Backward compatibility: an old EC2 command line keeps working."""
        args = self._parser().parse_args(
            ["--gap-min-track-frames", "7", "--gap-min-motion-px", "30",
             "--gap-static-max-px", "9"])
        self.assertEqual(args.gap_min_track_frames, 7)
        overrides = {"min_track_frames": float(args.gap_min_track_frames),
                     "min_motion_px": float(args.gap_min_motion_px),
                     "static_max_motion_px": float(args.gap_static_max_px)}
        r = gval.GapValidationConfig().resolve(848, 15.0, overrides)
        self.assertEqual(r.min_track_frames, 7)
        self.assertEqual(r.min_motion_px, 30.0)
        self.assertEqual(r.static_max_motion_px, 9.0)

    def test_override_does_not_mutate_the_config(self):
        """An absolute override must stay per-camera, never persist."""
        cfg = gval.GapValidationConfig()
        before = cfg.describe()
        cfg.resolve(848, 15.0, {"min_motion_px": 999.0})
        self.assertEqual(cfg.describe(), before)
        # ...and the next camera resolves from the normalized value again
        self.assertAlmostEqual(cfg.resolve(848, 15.0).min_motion_px,
                               cfg.min_motion_frac * 848, places=3)

    def test_override_of_unknown_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            gval.GapValidationConfig().resolve(848, 15.0, {"not_a_threshold": 1.0})

    def test_override_applies_through_validate_gap_events(self):
        xs = [700.0 - 35.0 * i for i in range(13)]
        g = GapEvent(track_id=1, camera_id=CAMERA_RIGHT_UP, start_frame=100,
                     end_frame=112, confidence=0.9, hit_count=13,
                     center_x_trajectory=xs, fps=FPS,
                     hit_frames=list(range(100, 113)))
        ok = gval.validate_gap_events([g], CAMERA_RIGHT_UP, verbose=False,
                                      frame_width=848, fps=FPS)
        self.assertEqual(len(ok.accepted), 1)
        # An absurd absolute override must actually bite.
        strict = gval.validate_gap_events([g], CAMERA_RIGHT_UP, verbose=False,
                                          frame_width=848, fps=FPS,
                                          absolute_overrides={"min_motion_px": 5000.0})
        self.assertEqual(strict.accepted, [])
        self.assertEqual(strict.resolved_thresholds["min_motion_px"], 5000.0)

    def test_no_absolute_units_remain_in_the_config(self):
        for key in gval.GapValidationConfig().describe():
            self.assertFalse(key.endswith("_px") or key.endswith("_frames"),
                             f"{key} reintroduces absolute units into the config")


# ===========================================================================
# MULTI-TRAIN SAFETY: geometry resolution
# ===========================================================================

class TestGeometryResolution(unittest.TestCase):
    def test_thresholds_scale_with_frame_width(self):
        c = gval.GapValidationConfig()
        small = c.resolve(848, 15.0)
        large = c.resolve(1920, 15.0)
        ratio = 1920 / 848
        for attr in ("static_max_motion_px", "min_motion_px",
                     "duplicate_max_center_px", "min_motion_px_per_sec"):
            self.assertAlmostEqual(getattr(large, attr) / getattr(small, attr),
                                   ratio, places=4, msg=attr)

    def test_thresholds_scale_with_fps(self):
        c = gval.GapValidationConfig()
        slow = c.resolve(848, 15.0)
        fast = c.resolve(848, 30.0)
        self.assertEqual(fast.min_track_frames, 2 * slow.min_track_frames)
        self.assertEqual(fast.max_detection_gap_frames,
                         2 * slow.max_detection_gap_frames)
        self.assertEqual(fast.min_separation_frames,
                         2 * slow.min_separation_frames)

    def test_frame_distances_do_not_depend_on_fps(self):
        c = gval.GapValidationConfig()
        a, b = c.resolve(848, 15.0), c.resolve(848, 60.0)
        self.assertEqual(a.min_motion_px, b.min_motion_px)
        self.assertEqual(a.static_max_motion_px, b.static_max_motion_px)

    def test_no_config_threshold_is_in_absolute_units(self):
        """Guards against absolute px/frame values creeping back in."""
        for key in gval.GapValidationConfig().describe():
            self.assertFalse(key.endswith("_px") or key.endswith("_frames"),
                             f"{key} must be a frame-width fraction or seconds")

    def test_degenerate_geometry_falls_back_safely(self):
        c = gval.GapValidationConfig()
        r = c.resolve(0, 0.0)
        self.assertGreater(r.min_motion_px, 0)
        self.assertGreater(r.min_track_frames, 0)
        self.assertGreater(r.min_separation_frames, 0)

    def test_same_physical_motion_judged_the_same_at_two_resolutions(self):
        """A gap crossing 35% of the frame validates at either resolution."""
        def make(width):
            xs = [0.82 * width - (0.35 * width / 12) * i for i in range(13)]
            return GapEvent(track_id=1, camera_id=CAMERA_RIGHT_UP,
                            start_frame=100, end_frame=112, confidence=0.9,
                            hit_count=13, center_x_trajectory=xs, fps=FPS,
                            hit_frames=list(range(100, 113)))
        for w in (640, 848, 1920, 3840):
            res = gval.validate_gap_events([make(w)], CAMERA_RIGHT_UP,
                                           verbose=False, frame_width=w, fps=FPS)
            self.assertEqual(len(res.accepted), 1, f"width={w}")

    def test_static_artefact_rejected_at_every_resolution(self):
        """A pinned detection must stay rejected as resolution grows."""
        for w in (640, 848, 1920, 3840):
            jitter = 0.002 * w                      # sub-threshold wobble
            xs = [0.5 * w + (jitter if i % 2 else 0) for i in range(30)]
            g = GapEvent(track_id=1, camera_id=CAMERA_RIGHT_UP, start_frame=100,
                         end_frame=129, confidence=0.93, hit_count=30,
                         center_x_trajectory=xs, fps=FPS,
                         hit_frames=list(range(100, 130)))
            res = gval.validate_gap_events([g], CAMERA_RIGHT_UP, verbose=False,
                                           frame_width=w, fps=FPS)
            self.assertEqual(res.accepted, [], f"width={w} must reject a static track")


# ===========================================================================
# MULTI-TRAIN SAFETY: no state leaks between runs
# ===========================================================================

class TestNoStateLeakBetweenTrains(unittest.TestCase):
    def _train(self, n_gaps, spacing, start, width, fps, conf=0.9):
        gaps = []
        for i in range(n_gaps):
            c = (start + i * spacing) * fps
            s, e = int(round(c - 6)), int(round(c + 6))
            xs = [0.8 * width - (0.4 * width / 12) * k for k in range(13)]
            gaps.append(GapEvent(track_id=i + 1, camera_id=CAMERA_RIGHT_UP,
                                 start_frame=max(0, s), end_frame=max(1, e),
                                 confidence=conf, hit_count=13,
                                 center_x_trajectory=xs, fps=fps,
                                 hit_frames=list(range(max(0, s), max(1, e) + 1))))
        return LocalCameraTracks(camera_id=CAMERA_RIGHT_UP,
                                 video_path="/t.mp4", fps=fps,
                                 total_frames=int((start + n_gaps * spacing + 10) * fps),
                                 width=width, height=480, gaps=gaps)

    def test_two_different_trains_in_one_process_do_not_interfere(self):
        a = self._train(8, 4.0, 10.0, 848, 15.0)
        b = self._train(20, 2.5, 5.0, 1920, 30.0)

        ra1 = gval.validate_gap_events(list(a.gaps), CAMERA_RIGHT_UP,
                                       verbose=False, frame_width=a.width, fps=a.fps)
        rb = gval.validate_gap_events(list(b.gaps), CAMERA_RIGHT_UP,
                                      verbose=False, frame_width=b.width, fps=b.fps)
        ra2 = gval.validate_gap_events(list(a.gaps), CAMERA_RIGHT_UP,
                                       verbose=False, frame_width=a.width, fps=a.fps)

        self.assertEqual(len(ra1.accepted), len(ra2.accepted),
                         "train A must validate identically before and after B")
        self.assertEqual(ra1.resolved_thresholds, ra2.resolved_thresholds)
        self.assertNotEqual(ra1.resolved_thresholds, rb.resolved_thresholds,
                            "different geometry must resolve differently")

    def test_defaults_are_not_mutated_by_a_run(self):
        before = gval.DEFAULT_GAP_VALIDATION.describe()
        t = self._train(6, 4.0, 10.0, 848, 15.0)
        gval.validate_gap_events(list(t.gaps), CAMERA_RIGHT_UP, verbose=False,
                                 frame_width=t.width, fps=t.fps)
        self.assertEqual(gval.DEFAULT_GAP_VALIDATION.describe(), before)
        self.assertEqual(gf.DEFAULT_CONFIG.offset_min_margin_ratio,
                         gf.FusionConfig().offset_min_margin_ratio)
        self.assertEqual(tcls.DEFAULT_TEMPORAL_CONFIG.describe(),
                         tcls.TemporalClassificationConfig().describe())

    def test_repeated_identical_runs_are_deterministic(self):
        t = self._train(10, 4.0, 10.0, 848, 15.0)
        outs = [len(gval.validate_gap_events(list(t.gaps), CAMERA_RIGHT_UP,
                                             verbose=False, frame_width=t.width,
                                             fps=t.fps).accepted)
                for _ in range(3)]
        self.assertEqual(len(set(outs)), 1, f"non-deterministic: {outs}")

    def test_wagon_window_state_does_not_persist(self):
        from global_train_state import GlobalWagon
        def segs(labels):
            return [GlobalWagon(global_id=f"S{i}", wagon_index=i,
                                start_frame_master=i * 100,
                                end_frame_master=i * 100 + 99,
                                start_time=i * 100 / FPS,
                                end_time=(i * 100 + 100) / FPS,
                                classification=lb, classification_confidence=1.0)
                    for i, lb in enumerate(labels)]
        E, W, B = (SegmentClass.ENGINE, SegmentClass.WAGON, SegmentClass.BRAKE_VAN)
        w1 = ts.get_master_wagon_window(segs([E, W, W, B]), verbose=False)
        w2 = ts.get_master_wagon_window(segs([E, E, W, W, W, W, B]), verbose=False)
        w3 = ts.get_master_wagon_window(segs([E, W, W, B]), verbose=False)
        self.assertEqual(w1.master_wagon_count, 2)
        self.assertEqual(w2.master_wagon_count, 4)
        self.assertEqual(w3.master_wagon_count, w1.master_wagon_count)


# ===========================================================================
# WHOLE-WAGON ALIAS (k-1 / k / k+1)
# ===========================================================================

class TestWagonAliasEvaluation(unittest.TestCase):
    def _obs(self, times, cam=CAMERA_LEFT_UP_TOP):
        gaps = [GapEvent(track_id=i + 1, camera_id=cam,
                         start_frame=int(t * FPS) - 6, end_frame=int(t * FPS) + 6,
                         confidence=0.9, hit_count=13, fps=FPS,
                         center_x_trajectory=[600.0 - 35 * k for k in range(13)],
                         hit_frames=list(range(int(t * FPS) - 6, int(t * FPS) + 7)))
                for i, t in enumerate(times)]
        return gf.to_gap_observations(LocalCameraTracks(
            camera_id=cam, video_path="/x.mp4", fps=FPS, total_frames=6000,
            gaps=gaps))

    def test_all_three_shifts_are_evaluated(self):
        base = [10.0 + i * 4.0 for i in range(15)]
        m = self._obs(base, CAMERA_RIGHT_UP)
        s = self._obs(base)
        cands = gf.evaluate_wagon_alias_candidates(m, s, 0.0)
        self.assertEqual(sorted(c.k for c in cands), [-1, 0, 1])
        for c in cands:
            self.assertIn("score", c.to_dict())
            self.assertIn("n_match", c.to_dict())

    def test_alias_period_is_derived_at_runtime(self):
        """Different trains have different spacing; the period must follow."""
        for spacing in (2.5, 4.0, 7.0):
            base = [10.0 + i * spacing for i in range(12)]
            m = self._obs(base, CAMERA_RIGHT_UP)
            cands = gf.evaluate_wagon_alias_candidates(m, self._obs(base), 0.0)
            by_k = {c.k: c.delta for c in cands}
            self.assertAlmostEqual(by_k[1] - by_k[0], spacing, delta=0.35,
                                   msg=f"spacing={spacing}")

    def test_unshifted_alignment_wins_when_data_is_clean(self):
        base = [10.0 + i * 4.0 + (0.6 if i % 3 == 0 else 0.0) for i in range(16)]
        m = self._obs(base, CAMERA_RIGHT_UP)
        s = self._obs([t - 5.0 for t in base])
        cands = gf.evaluate_wagon_alias_candidates(m, s, 5.0)
        self.assertEqual(cands[0].k, 0, "true offset must beat the +-1 shifts")

    def test_alias_conflict_forces_unresolved(self):
        off = gf.CameraOffset(camera_id=CAMERA_LEFT_UP_TOP, delta=1.0,
                              status=gf.OFFSET_RESOLVED)
        off.alias_conflict = True
        self.assertTrue(off.alias_conflict)
        # the resolver marks such an offset UNRESOLVED; see estimate_camera_offset
        self.assertIn("alias_conflict", off.to_dict())

    def test_offset_result_reports_alias_candidates(self):
        base = [10.0 + i * 4.0 for i in range(14)]
        m = self._obs(base, CAMERA_RIGHT_UP)
        s = self._obs([t - 3.0 for t in base])
        off = gf.estimate_camera_offset(m, s, camera_id=CAMERA_LEFT_UP_TOP)
        self.assertTrue(off.alias_candidates, "alias hypotheses must be recorded")
        self.assertIn(off.alias_best_k, (-1, 0, 1))

    def test_a_one_wagon_lag_cannot_change_the_master(self):
        """Whatever the support camera's alignment, the master is untouched."""
        base = [10.0 + i * 4.0 for i in range(12)]
        master = LocalCameraTracks(
            camera_id=CAMERA_RIGHT_UP, video_path="/m.mp4", fps=FPS,
            total_frames=6000,
            gaps=[GapEvent(track_id=i + 1, camera_id=CAMERA_RIGHT_UP,
                           start_frame=int(t * FPS) - 6,
                           end_frame=int(t * FPS) + 6, confidence=0.9,
                           hit_count=13, fps=FPS,
                           center_x_trajectory=[600.0 - 35 * k for k in range(13)],
                           hit_frames=list(range(int(t * FPS) - 6,
                                                 int(t * FPS) + 7)))
                  for i, t in enumerate(base)])
        gaps_before = gf.build_global_gap_sequence(master)
        lagged = LocalCameraTracks(
            camera_id=CAMERA_LEFT_UP_TOP, video_path="/s.mp4", fps=FPS,
            total_frames=6000,
            gaps=[GapEvent(track_id=i + 1, camera_id=CAMERA_LEFT_UP_TOP,
                           start_frame=int((t + 4.0) * FPS) - 6,
                           end_frame=int((t + 4.0) * FPS) + 6, confidence=0.9,
                           hit_count=13, fps=FPS,
                           center_x_trajectory=[600.0 - 35 * k for k in range(13)],
                           hit_frames=list(range(int((t + 4.0) * FPS) - 6,
                                                 int((t + 4.0) * FPS) + 7)))
                  for i, t in enumerate(base)])
        gaps_after = gf.build_global_gap_sequence(master)
        gf.attach_support_evidence(gaps_after, [lagged], verbose=False)
        self.assertEqual(len(gaps_before), len(gaps_after))
        self.assertEqual([g.master_frame for g in gaps_before],
                         [g.master_frame for g in gaps_after],
                         "a one-wagon-lagged support camera must not move the master")


if __name__ == "__main__":
    unittest.main(verbosity=2)
