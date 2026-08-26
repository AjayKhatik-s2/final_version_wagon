"""TRAIN_START / TRAIN_END come from classification, never from a gap alone.

The failure this prevents: the first "gap" a detector reports at the head of a
run is often not an inter-wagon gap at all. It is empty track before the train
arrives, or the leading face of the ENGINE crossing the frame. Anchoring
TRAIN_START on it starts the train early, on nothing. The same at the tail
truncates a real brake van.

So every test here checks the same thing from a different angle: a gap with no
classified train behind it must not move the boundary, and a classified train
region must not be cut by the absence of a gap.
"""

from __future__ import annotations

import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from core.camera_evidence import LocalSegment, local_segment_id
from core.train_window import (
    SOURCE_MASTER, SOURCE_SUPPORT, TRAIN_CLASSES, LabelledSpan,
    TrainWindowPolicy, camera_evidence, detect_train_window,
    spans_from_local_segments, spans_from_master_classifications,
)

RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP
RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP
ENGINE, WAGON, BRAKE, UNKNOWN = (C.CLASS_ENGINE, C.CLASS_WAGON,
                                 C.CLASS_BRAKE_VAN, C.CLASS_UNKNOWN)


def _spans(cam, items):
    """items: [(start, end, label)]"""
    return [LabelledSpan(camera_id=cam, start_time=s, end_time=e, label=l,
                         confidence=0.9, segment_id=f"{cam}_{i}")
            for i, (s, e, l) in enumerate(items, start=1)]


def _typical(cam=RU, shift=0.0):
    """ENGINE, three WAGONs, BRAKE_VAN -- a whole train, 10s..40s."""
    return _spans(cam, [(10.0 + shift, 16.0 + shift, ENGINE),
                        (16.0 + shift, 24.0 + shift, WAGON),
                        (24.0 + shift, 32.0 + shift, WAGON),
                        (32.0 + shift, 36.0 + shift, WAGON),
                        (36.0 + shift, 40.0 + shift, BRAKE)])


class TestClassificationSetsTheBoundary(unittest.TestCase):
    def test_window_spans_engine_through_brake_van(self):
        w = detect_train_window(master_spans=_typical())
        self.assertTrue(w.found)
        self.assertAlmostEqual(w.start_time, 10.0)
        self.assertAlmostEqual(w.end_time, 40.0)
        self.assertAlmostEqual(w.duration, 30.0)
        self.assertEqual(w.start_source, SOURCE_MASTER)

    def test_the_engine_is_inside_the_train_window(self):
        """Unlike the WAGON window, which excludes it."""
        w = detect_train_window(master_spans=_typical())
        self.assertTrue(w.contains(12.0), "engine must be inside the train")
        self.assertTrue(w.contains(38.0), "brake van must be inside the train")

    def test_unknown_alone_is_not_a_train(self):
        w = detect_train_window(master_spans=_spans(RU, [(0.0, 50.0, UNKNOWN)]))
        self.assertFalse(w.found)
        self.assertIn("no camera classified", w.reason)

    def test_no_classification_at_all_yields_no_window(self):
        w = detect_train_window(master_spans=[],
                                master_gap_times=[5.0, 9.0, 21.0])
        self.assertFalse(w.found)
        self.assertIn("gap evidence alone cannot establish", " ".join(w.notes))


class TestGapsNeverSetTheBoundary(unittest.TestCase):
    """The reported failure mode, from both ends."""

    def test_a_leading_gap_on_empty_track_is_rejected(self):
        w = detect_train_window(master_spans=_typical(),
                                master_gap_times=[2.0, 6.0, 18.0, 28.0])
        self.assertAlmostEqual(w.start_time, 10.0,
                               msg="an early gap moved TRAIN_START")
        rejected = [r for r in w.rejected_boundaries if r.position == "leading"]
        self.assertEqual([round(r.time, 1) for r in rejected], [2.0, 6.0])
        self.assertIn("ENGINE", rejected[0].reason)

    def test_the_engine_face_does_not_become_train_start(self):
        """A gap at the ENGINE's leading edge is inside, not the boundary."""
        w = detect_train_window(master_spans=_typical(),
                                master_gap_times=[9.5, 16.0, 24.0])
        self.assertAlmostEqual(w.start_time, 10.0)
        self.assertEqual([round(r.time, 1) for r in w.rejected_boundaries],
                         [9.5])

    def test_a_trailing_gap_after_the_train_is_rejected(self):
        w = detect_train_window(master_spans=_typical(),
                                master_gap_times=[24.0, 44.0, 60.0])
        self.assertAlmostEqual(w.end_time, 40.0,
                               msg="a late gap extended TRAIN_END")
        trailing = [r for r in w.rejected_boundaries if r.position == "trailing"]
        self.assertEqual([round(r.time, 1) for r in trailing], [44.0, 60.0])

    def test_the_train_is_not_cut_early_when_no_gap_follows(self):
        """Absence of a trailing gap must not truncate the brake van."""
        w = detect_train_window(master_spans=_typical(),
                                master_gap_times=[16.0, 24.0])
        self.assertAlmostEqual(w.end_time, 40.0)

    def test_gaps_inside_the_train_are_not_rejected(self):
        w = detect_train_window(master_spans=_typical(),
                                master_gap_times=[16.0, 24.0, 32.0, 36.0])
        self.assertEqual(w.rejected_boundaries, [])


class TestTemporalContinuity(unittest.TestCase):
    def test_a_short_unreadable_stretch_does_not_split_the_train(self):
        spans = _spans(RU, [(10.0, 16.0, ENGINE), (16.0, 20.0, WAGON),
                            (20.0, 24.0, UNKNOWN),      # one bad read
                            (24.0, 30.0, WAGON), (30.0, 34.0, BRAKE)])
        w = detect_train_window(master_spans=spans)
        self.assertAlmostEqual(w.start_time, 10.0)
        self.assertAlmostEqual(w.end_time, 34.0)

    def test_an_isolated_far_away_misclassification_is_discarded(self):
        """A lone WAGON label out on empty track must not extend the train."""
        spans = _spans(RU, [(0.5, 1.0, WAGON)]) + _typical()
        w = detect_train_window(master_spans=spans)
        self.assertAlmostEqual(w.start_time, 10.0,
                               msg="a distant blip became TRAIN_START")

    def test_the_bridge_distance_is_configurable(self):
        spans = _spans(RU, [(10.0, 14.0, WAGON), (30.0, 34.0, WAGON)])
        tight = detect_train_window(
            master_spans=spans, policy=TrainWindowPolicy(max_discontinuity=2.0))
        self.assertAlmostEqual(tight.duration, 4.0, msg="runs should stay split")
        loose = detect_train_window(
            master_spans=spans, policy=TrainWindowPolicy(max_discontinuity=30.0))
        self.assertAlmostEqual(loose.duration, 24.0)


class TestMultiCameraCorroboration(unittest.TestCase):
    def test_support_cameras_corroborate_the_master(self):
        w = detect_train_window(
            master_spans=_typical(RU),
            support_spans={LU: _typical(LU, shift=0.4),
                           RUT: _typical(RUT, shift=-0.3)})
        self.assertEqual(w.start_source, SOURCE_MASTER)
        self.assertEqual(w.start_camera, RU)
        self.assertEqual(sorted(w.start_corroborating), sorted([LU, RUT]))

    def test_a_disagreeing_camera_does_not_move_the_boundary(self):
        w = detect_train_window(
            master_spans=_typical(RU),
            support_spans={LUT: _spans(LUT, [(0.0, 60.0, WAGON)])})
        self.assertAlmostEqual(w.start_time, 10.0)
        self.assertAlmostEqual(w.end_time, 40.0)
        self.assertNotIn(LUT, w.start_corroborating)

    def test_support_stands_in_only_when_the_master_classified_nothing(self):
        w = detect_train_window(
            master_spans=_spans(RU, [(0.0, 50.0, UNKNOWN)]),
            support_spans={LU: _typical(LU), RUT: _typical(RUT)})
        self.assertTrue(w.found)
        self.assertEqual(w.start_source, SOURCE_SUPPORT)
        self.assertIn("no classified train region", " ".join(w.notes))

    def test_camera_offsets_are_applied_before_comparing(self):
        """LEFT_UP runs 3s behind; projected it agrees with the master."""
        local = _typical(LU, shift=-3.0)
        w = detect_train_window(
            master_spans=_typical(RU),
            support_spans={LU: [LabelledSpan(LU, s.start_time + 3.0,
                                             s.end_time + 3.0, s.label)
                                for s in local]})
        self.assertIn(LU, w.start_corroborating)

    def test_a_camera_with_no_train_evidence_is_reported_not_dropped(self):
        w = detect_train_window(master_spans=_typical(RU),
                                support_spans={LUT: []})
        self.assertIn(LUT, w.per_camera)
        self.assertFalse(w.per_camera[LUT].usable)
        self.assertIn("no classified segments", w.per_camera[LUT].reason)

    def test_corroboration_shortfall_is_noted_not_fatal(self):
        w = detect_train_window(
            master_spans=_typical(RU),
            policy=TrainWindowPolicy(min_corroborating_cameras=3))
        self.assertTrue(w.found, "a shortfall must not discard the window")
        self.assertTrue(any("corroborated by" in n for n in w.notes))


class TestEvidenceNormalisation(unittest.TestCase):
    def test_master_classifications_convert_frames_to_master_seconds(self):
        class MC:
            def __init__(s, i, a, b, l):
                s.segment_index, s.start_frame, s.end_frame = i, a, b
                s.label, s.confidence = l, 0.9
        spans = spans_from_master_classifications(
            [MC(0, 0, 149, ENGINE), MC(1, 150, 299, WAGON)], fps=15.0)
        self.assertAlmostEqual(spans[0].start_time, 0.0)
        self.assertAlmostEqual(spans[0].end_time, 10.0)
        self.assertAlmostEqual(spans[1].start_time, 10.0)

    def test_local_segments_are_projected_by_the_offset(self):
        segs = [LocalSegment(local_id=local_segment_id(LU, 1), index=1,
                             start_frame=0, end_frame=59, start_time=5.0,
                             end_time=9.0, label=WAGON, confidence=0.8)]
        spans = spans_from_local_segments(segs, LU, offset=-1.5)
        self.assertAlmostEqual(spans[0].start_time, 3.5)
        self.assertAlmostEqual(spans[0].end_time, 7.5)

    def test_only_the_named_classes_count_as_train(self):
        self.assertEqual(set(TRAIN_CLASSES), {ENGINE, WAGON, BRAKE})
        self.assertNotIn(UNKNOWN, TRAIN_CLASSES)

    def test_camera_evidence_reports_what_it_saw(self):
        ev = camera_evidence(_typical(RU), RU)
        self.assertTrue(ev.usable)
        self.assertEqual(ev.train_segments, 5)
        self.assertEqual(ev.first_label, ENGINE)
        self.assertEqual(ev.last_label, BRAKE)


class TestOutputContract(unittest.TestCase):
    def test_the_window_serialises_with_its_provenance(self):
        w = detect_train_window(master_spans=_typical(),
                                master_gap_times=[2.0])
        d = w.to_dict()
        self.assertEqual(d["train_start"], 10.0)
        self.assertEqual(d["train_end"], 40.0)
        self.assertEqual(d["duration"], 30.0)
        self.assertEqual(d["start_source"], SOURCE_MASTER)
        self.assertEqual(len(d["rejected_boundaries"]), 1)
        self.assertIn(RU, d["per_camera"])

    def test_summary_names_the_rejected_candidates(self):
        w = detect_train_window(master_spans=_typical(),
                                master_gap_times=[2.0, 55.0])
        text = " ".join(w.summary_lines())
        self.assertIn("rejected leading", text)
        self.assertIn("rejected trailing", text)

    def test_nothing_upstream_is_modified(self):
        """This stage reports an interval; it changes no roster and no gap."""
        import ast
        import inspect
        from core import train_window
        src = inspect.getsource(train_window)
        tree = ast.parse(src)
        # Inspect CALLS, not prose: the docstring names get_master_wagon_window
        # to explain how the train window differs from the wagon window.
        called = {ast.unparse(n.func) for n in ast.walk(tree)
                  if isinstance(n, ast.Call)}
        for banned in ("build_global_wagons", "get_master_wagon_window",
                       "renumber_gap_events", "GapTracker",
                       "assemble_global_train_state"):
            with self.subTest(token=banned):
                self.assertFalse([c for c in called if banned in c],
                                 f"{banned} is invoked here")
        names = {a.name for n in ast.walk(ast.parse(src))
                 if isinstance(n, (ast.Import, ast.ImportFrom))
                 for a in n.names}
        self.assertFalse([x for x in names if "fusion" in x.lower()])


if __name__ == "__main__":
    unittest.main()


class TestCompleteTrainTimeline(unittest.TestCase):
    """The timeline is the whole train; the roster is only its wagons.

    Order: complete physical train first, canonical gaps second, counted
    GW_n third. Deriving the train from the wagons instead would start the
    coordinate system behind the locomotive.
    """

    def _timeline(self, spans=None):
        from core.train_window import build_train_timeline
        spans = spans if spans is not None else _typical()
        w = detect_train_window(master_spans=spans)
        return build_train_timeline(w, spans)

    def test_engine_and_brake_van_are_on_the_timeline(self):
        tl = self._timeline()
        kinds = [r.kind for r in tl.regions]
        self.assertEqual(kinds, [ENGINE, WAGON, WAGON, WAGON, BRAKE])
        self.assertAlmostEqual(tl.start_time, 10.0)
        self.assertAlmostEqual(tl.end_time, 40.0)

    def test_gw_numbering_starts_at_the_first_wagon(self):
        tl = self._timeline()
        self.assertEqual([r.global_id for r in tl.regions],
                         [None, "GW_1", "GW_2", "GW_3", None])
        self.assertEqual(tl.counted_wagon_count, 3)

    def test_engine_and_brake_van_receive_no_global_id(self):
        tl = self._timeline()
        for r in tl.non_counted_regions:
            self.assertIn(r.kind, (ENGINE, BRAKE, UNKNOWN))
            self.assertIsNone(r.global_id)
            self.assertFalse(r.counted)

    def test_the_timeline_is_longer_than_the_counted_region(self):
        """The train starts before GW_1 and ends after the last wagon."""
        tl = self._timeline()
        first, last = tl.counted_regions[0], tl.counted_regions[-1]
        self.assertLess(tl.start_time, first.start_time,
                        "the ENGINE must precede GW_1")
        self.assertGreater(tl.end_time, last.end_time,
                           "the BRAKE_VAN must follow the last wagon")

    def test_a_time_inside_the_engine_maps_to_no_wagon(self):
        tl = self._timeline()
        self.assertEqual(tl.region_at(12.0).kind, ENGINE)
        self.assertIsNone(tl.global_id_at(12.0))
        self.assertEqual(tl.global_id_at(20.0), "GW_1")
        self.assertIsNone(tl.global_id_at(38.0), "brake van is not a wagon")

    def test_numbering_matches_the_counted_wagon_window(self):
        """Same classification in, same roster out as the counting engine."""
        from global_train_state import GlobalWagon as EngineWagon
        from train_structure import get_master_wagon_window
        fps = 15.0

        def seg(i, cls, a, b):
            return EngineWagon(global_id=f"GW_{i}", wagon_index=i,
                               start_frame_master=a, end_frame_master=b,
                               start_time=a / fps, end_time=(b + 1) / fps,
                               classification=cls,
                               classification_confidence=0.9)
        segments = [seg(1, ENGINE, 150, 239), seg(2, WAGON, 240, 359),
                    seg(3, WAGON, 360, 479), seg(4, WAGON, 480, 539),
                    seg(5, BRAKE, 540, 599)]
        win = get_master_wagon_window(segments, verbose=False)
        tl = self._timeline()
        self.assertEqual(len(win.wagon_units), tl.counted_wagon_count)
        self.assertEqual([w.global_id for w in win.wagon_units],
                         [r.global_id for r in tl.counted_regions])

    def test_an_unknown_region_stays_on_the_timeline_uncounted(self):
        spans = _spans(RU, [(10.0, 14.0, ENGINE), (14.0, 20.0, WAGON),
                            (20.0, 24.0, UNKNOWN), (24.0, 30.0, WAGON)])
        tl = self._timeline(spans)
        kinds = [r.kind for r in tl.regions]
        self.assertIn(UNKNOWN, kinds)
        self.assertEqual([r.global_id for r in tl.regions],
                         [None, "GW_1", None, "GW_2"])

    def test_regions_are_clipped_to_the_physical_train(self):
        """Nothing outside TRAIN_START..TRAIN_END enters the coordinate system."""
        spans = _spans(RU, [(0.0, 2.0, UNKNOWN)]) + _typical() + \
            _spans(RU, [(50.0, 55.0, UNKNOWN)])
        tl = self._timeline(spans)
        for r in tl.regions:
            self.assertGreaterEqual(r.start_time, tl.start_time - 1e-9)
            self.assertLessEqual(r.end_time, tl.end_time + 1e-9)

    def test_regions_are_ordered_and_contiguous_in_index(self):
        tl = self._timeline()
        self.assertEqual([r.index for r in tl.regions],
                         list(range(len(tl.regions))))
        times = [r.start_time for r in tl.regions]
        self.assertEqual(times, sorted(times))

    def test_no_timeline_when_no_train_was_found(self):
        from core.train_window import build_train_timeline
        spans = _spans(RU, [(0.0, 50.0, UNKNOWN)])
        w = detect_train_window(master_spans=spans)
        tl = build_train_timeline(w, spans)
        self.assertFalse(tl.found)
        self.assertEqual(tl.regions, [])

    def test_the_timeline_serialises_with_both_views(self):
        d = self._timeline().to_dict()
        self.assertEqual(d["counted_wagon_count"], 3)
        self.assertEqual(d["region_count"], 5)
        self.assertEqual(d["train_start_global_time"], 10.0)
        kinds = [r["kind"] for r in d["regions"]]
        self.assertIn(ENGINE, kinds)
        self.assertIn(BRAKE, kinds)
