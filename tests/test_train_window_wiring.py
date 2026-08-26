"""The train window is wired in, and it only ever removes phantom boundaries.

The stage sits between gap validation and fusion, in both modes:

    gaps + classification -> TRAIN_START/TRAIN_END -> restrict master gaps
                          -> build_global_gap_sequence -> GW_1..GW_N

Its effect on the roster is strictly subtractive. It can drop a validated
RIGHT_UP gap lying outside the classified train -- empty track ahead of the
rake, or the ENGINE's leading face -- and it can never add, move, split or
renumber one. `build_global_gap_sequence` remains the sole minter of the global
sequence and still consults no support camera.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from core import train_window as TW
from core.train_window import LabelledSpan, TrainWindowPolicy

RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP
RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP
ENGINE, WAGON, BRAKE, UNKNOWN = (C.CLASS_ENGINE, C.CLASS_WAGON,
                                 C.CLASS_BRAKE_VAN, C.CLASS_UNKNOWN)
FPS = 15.0


def _gap(t, fps=FPS, track_id=1):
    """A real GapEvent at master time `t`."""
    from global_train_state import GapEvent
    f = int(t * fps)
    return GapEvent(track_id=track_id, camera_id=RU, start_frame=f - 3,
                    end_frame=f + 3, confidence=0.9, hit_count=7,
                    center_x_trajectory=[100.0, 700.0], fps=fps,
                    temporal_consistency_score=1.0,
                    hit_frames=list(range(f - 3, f + 4)),
                    bbox_history=[[0.0, 0.0, 1.0, 1.0]] * 7)


def _train(cam=RU, shift=0.0):
    """ENGINE 10-16, WAGON 16-28, WAGON 28-36, BRAKE_VAN 36-40."""
    return [LabelledSpan(cam, 10.0 + shift, 16.0 + shift, ENGINE),
            LabelledSpan(cam, 16.0 + shift, 28.0 + shift, WAGON),
            LabelledSpan(cam, 28.0 + shift, 36.0 + shift, WAGON),
            LabelledSpan(cam, 36.0 + shift, 40.0 + shift, BRAKE)]


class TestFalseBoundariesCannotDefineTheTrain(unittest.TestCase):
    def test_a_false_initial_gap_cannot_start_the_train(self):
        gaps = [_gap(3.0), _gap(16.0), _gap(28.0), _gap(36.0)]
        w = TW.detect_train_window(
            master_spans=_train(),
            master_gap_times=[g.center_time for g in gaps])
        self.assertAlmostEqual(w.start_time, 10.0)
        f = TW.filter_gaps_to_window(gaps, w, fps=FPS)
        self.assertEqual([round(g.center_time, 1) for g in f.kept],
                         [16.0, 28.0, 36.0])
        self.assertEqual(len(f.dropped_leading), 1)

    def test_engine_classification_keeps_the_start_before_the_first_wagon(self):
        w = TW.detect_train_window(master_spans=_train())
        self.assertAlmostEqual(w.start_time, 10.0)
        self.assertLess(w.start_time, 16.0, "start must precede the first WAGON")
        self.assertEqual(w.per_camera[RU].first_label, ENGINE)

    def test_a_false_trailing_gap_cannot_end_the_train(self):
        gaps = [_gap(16.0), _gap(28.0), _gap(36.0), _gap(52.0)]
        w = TW.detect_train_window(
            master_spans=_train(),
            master_gap_times=[g.center_time for g in gaps])
        self.assertAlmostEqual(w.end_time, 40.0)
        f = TW.filter_gaps_to_window(gaps, w, fps=FPS)
        self.assertEqual(len(f.dropped_trailing), 1)
        self.assertNotIn(52.0, [round(g.center_time, 1) for g in f.kept])

    def test_the_brake_van_is_inside_the_physical_train(self):
        w = TW.detect_train_window(master_spans=_train())
        self.assertAlmostEqual(w.end_time, 40.0)
        self.assertTrue(w.contains(38.0))
        self.assertEqual(w.per_camera[RU].last_label, BRAKE)

    def test_an_isolated_classification_error_does_not_move_the_edges(self):
        spans = ([LabelledSpan(RU, 0.5, 1.0, WAGON)] + _train()
                 + [LabelledSpan(RU, 90.0, 90.5, WAGON)])
        w = TW.detect_train_window(master_spans=spans)
        self.assertAlmostEqual(w.start_time, 10.0)
        self.assertAlmostEqual(w.end_time, 40.0)

    def test_side_and_top_classification_jointly_support_the_window(self):
        w = TW.detect_train_window(
            master_spans=_train(RU),
            support_spans={LU: _train(LU, 0.3), RUT: _train(RUT, -0.2),
                           LUT: _train(LUT, 0.5)})
        self.assertEqual(sorted(w.start_corroborating), sorted([LU, RUT, LUT]))
        self.assertEqual(sorted(w.end_corroborating), sorted([LU, RUT, LUT]))


class TestTheFilterIsSubtractiveOnly(unittest.TestCase):
    def test_no_gap_is_ever_invented(self):
        gaps = [_gap(16.0), _gap(28.0)]
        w = TW.detect_train_window(master_spans=_train())
        f = TW.filter_gaps_to_window(gaps, w, fps=FPS)
        self.assertLessEqual(len(f.kept), len(gaps))
        for g in f.kept:
            self.assertIn(g, gaps)

    def test_gap_times_are_not_moved(self):
        gaps = [_gap(16.0), _gap(28.0), _gap(36.0)]
        w = TW.detect_train_window(master_spans=_train())
        f = TW.filter_gaps_to_window(gaps, w, fps=FPS)
        self.assertEqual([g.center_time for g in f.kept],
                         [g.center_time for g in gaps])

    def test_no_window_means_no_filtering(self):
        """Refusing to filter beats filtering on an unconfirmed boundary."""
        gaps = [_gap(3.0), _gap(16.0), _gap(52.0)]
        w = TW.detect_train_window(
            master_spans=[LabelledSpan(RU, 0.0, 60.0, UNKNOWN)])
        self.assertFalse(w.found)
        f = TW.filter_gaps_to_window(gaps, w, fps=FPS)
        self.assertFalse(f.applied)
        self.assertEqual(len(f.kept), 3)

    def test_a_gap_whose_time_is_unknown_is_kept(self):
        class Opaque:
            pass
        w = TW.detect_train_window(master_spans=_train())
        f = TW.filter_gaps_to_window([Opaque()], w, fps=0.0)
        self.assertEqual(len(f.kept), 1, "an unplaceable gap must not be lost")

    def test_gaps_inside_the_train_all_survive(self):
        gaps = [_gap(16.0), _gap(28.0), _gap(36.0)]
        w = TW.detect_train_window(master_spans=_train())
        f = TW.filter_gaps_to_window(gaps, w, fps=FPS)
        self.assertEqual(len(f.kept), 3)
        self.assertEqual(f.dropped, 0)


class TestMasterFixedInvariantIsUntouched(unittest.TestCase):
    def test_right_up_is_still_the_only_minter(self):
        import global_fusion as gf
        src = inspect.getsource(gf.build_global_gap_sequence)
        self.assertIn("master_tracks", src)
        self.assertNotIn("support", src.split('"""')[-1],
                         "the minter must not consult a support camera")

    def test_the_stage_does_not_touch_fusion(self):
        import ast
        src = inspect.getsource(TW)
        called = {ast.unparse(n.func) for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.Call)}
        for banned in ("build_global_gap_sequence",
                       "assemble_global_train_state", "build_global_wagons"):
            self.assertFalse([c for c in called if banned in c])

    def test_support_camera_gaps_are_never_inputs_to_the_filter(self):
        """Only the master's gaps are restricted; support gaps are untouched."""
        sig = inspect.signature(TW.filter_gaps_to_window)
        self.assertIn("gaps", sig.parameters)
        self.assertNotIn("support_gaps", sig.parameters)

    def test_counted_numbering_still_excludes_engine_and_brake_van(self):
        """The train window includes them; the WAGON window still does not."""
        from train_structure import get_master_wagon_window
        from global_train_state import GlobalWagon as EngineWagon

        def seg(i, cls, a, b):
            return EngineWagon(global_id=f"GW_{i}", wagon_index=i,
                               start_frame_master=a, end_frame_master=b,
                               start_time=a / FPS, end_time=(b + 1) / FPS,
                               classification=cls,
                               classification_confidence=0.9)
        segments = [seg(1, ENGINE, 150, 239), seg(2, WAGON, 240, 419),
                    seg(3, WAGON, 420, 539), seg(4, BRAKE, 540, 599)]
        win = get_master_wagon_window(segments, verbose=False)
        self.assertEqual(len(win.wagon_units), 2, "only the WAGONs are counted")
        self.assertEqual([w.global_id for w in win.wagon_units],
                         ["GW_1", "GW_2"])
        # ...while the physical train window spans engine through brake van.
        tw = TW.detect_train_window(master_spans=_train())
        self.assertTrue(tw.contains(12.0) and tw.contains(38.0))


class TestArtifact(unittest.TestCase):
    def test_the_artifact_carries_everything_the_next_stage_needs(self):
        w = TW.detect_train_window(master_spans=_train(RU),
                                   support_spans={LUT: _train(LUT, 0.2)},
                                   master_gap_times=[3.0, 16.0, 52.0])
        with tempfile.TemporaryDirectory() as root:
            path = TW.write_artifact(w, root)
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        self.assertEqual(doc["schema"], TW.ARTIFACT_SCHEMA)
        self.assertEqual(doc["train_start_global_time"], 10.0)
        self.assertEqual(doc["train_end_global_time"], 40.0)
        self.assertEqual(doc["duration_seconds"], 30.0)
        self.assertIn(RU, doc["per_camera"])
        self.assertIn(LUT, doc["per_camera"])
        self.assertEqual(doc["start_source"], TW.SOURCE_MASTER)
        kinds = {r["position"] for r in doc["rejected_boundaries"]}
        self.assertEqual(kinds, {"leading", "trailing"})
        for r in doc["rejected_boundaries"]:
            self.assertTrue(r["reason"], "a rejection must say why")

    def test_a_window_that_was_not_found_reads_back_as_none(self):
        w = TW.detect_train_window(master_spans=[])
        with tempfile.TemporaryDirectory() as root:
            TW.write_artifact(w, root)
            self.assertIsNone(TW.read_artifact(root))

    def test_absent_artifact_reads_back_as_none(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(TW.read_artifact(root))


class TestBothModesAreWired(unittest.TestCase):
    """One implementation, two call sites, same semantics."""

    def test_sequential_computes_the_window_before_fusion(self):
        from orchestrator import global_assembler
        src = inspect.getsource(global_assembler.assemble)
        self.assertIn("detect_train_window", src)
        self.assertLess(src.index("detect_train_window"),
                        src.index("assemble_global_train_state_master_fixed"),
                        "the window must constrain the gaps fusion receives")
        self.assertIn("filter_gaps_to_window", src)

    def test_batch_computes_the_window_before_fusion(self):
        path = os.path.join(V4_ROOT, "wagon_count", "run_global_count.py")
        src = open(path, encoding="utf-8").read()
        self.assertIn("detect_train_window", src)
        self.assertLess(src.index("detect_train_window"),
                        src.index("assemble_global_train_state_master_fixed"),
                        "STEP 2d must precede STEP 3")
        self.assertIn("filter_gaps_to_window", src)

    def test_neither_mode_reimplements_the_detector(self):
        from orchestrator import global_assembler
        seq = inspect.getsource(global_assembler)
        batch = open(os.path.join(V4_ROOT, "wagon_count",
                                  "run_global_count.py"), encoding="utf-8").read()
        for src in (seq, batch):
            self.assertNotIn("TRAIN_CLASSES =", src)
            self.assertNotIn("_longest_train_run", src)

    def test_both_modes_produce_the_same_window_from_the_same_evidence(self):
        """The detector is a pure function of the evidence, so modes agree."""
        args = dict(master_spans=_train(RU),
                    support_spans={LU: _train(LU, 0.3)},
                    master_gap_times=[3.0, 16.0, 28.0, 36.0, 52.0])
        a = TW.detect_train_window(**args)
        b = TW.detect_train_window(**args)
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_both_modes_yield_the_same_restricted_gap_sequence(self):
        gaps = [_gap(3.0), _gap(16.0), _gap(28.0), _gap(36.0), _gap(52.0)]
        w = TW.detect_train_window(
            master_spans=_train(),
            master_gap_times=[g.center_time for g in gaps])
        seq = TW.filter_gaps_to_window(gaps, w, fps=FPS)
        batch = TW.filter_gaps_to_window(gaps, w, fps=FPS)
        self.assertEqual([g.center_time for g in seq.kept],
                         [g.center_time for g in batch.kept])
        self.assertEqual(len(seq.kept), 3, "3 gaps -> 4 canonical wagons")

    def test_batch_can_be_switched_back_to_the_previous_behaviour(self):
        path = os.path.join(V4_ROOT, "wagon_count", "run_global_count.py")
        src = open(path, encoding="utf-8").read()
        self.assertIn("--no-train-window", src)
        self.assertIn("if not args.no_train_window:", src)

    def test_sequential_can_be_switched_back_too(self):
        from orchestrator.global_assembler import assemble
        p = inspect.signature(assemble).parameters
        self.assertIn("use_train_window", p)
        self.assertIs(p["use_train_window"].default, True)

    def test_a_stage_failure_leaves_the_gaps_untouched(self):
        """A detector crash must never take the count with it."""
        path = os.path.join(V4_ROOT, "wagon_count", "run_global_count.py")
        src = open(path, encoding="utf-8").read()
        i = src.index("STEP 2d")
        block = src[i:src.index("STEP 3 -- cross-camera fusion", i)]
        self.assertIn("except Exception", block)
        self.assertIn("master gaps left", block)


if __name__ == "__main__":
    unittest.main()
