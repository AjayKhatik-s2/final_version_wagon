"""A non-wagon misread as WAGON must die in its own camera, before assembly.

The risk in sequential mode is specific. Cameras are processed independently and
persisted as they arrive, so a false WAGON on any camera is written to disk
before the other three exist. If that survived into assembly it could anchor the
alignment on an engine-to-wagon transition -- which is not a wagon boundary --
and shift the Global Wagon Timeline.

These tests VERIFY THE EXISTING CHAIN rather than adding a second one. The
validation is already there, per camera, at camera-processing time:

    camera_pipeline.run_{master,support}_camera
        _track_stitch_validate    fragment reassembly + gval.validate_gap_events
        tcls.apply_temporal_classification   layer 1 vote + layer 2 hysteresis
        ts.build_local_wagon_region          engine/brake-van region, this camera
              |
    camera_runner persists     gap_validation.json (with rejections)
                               wagon_region.json
                               classification.json
              |
    global_assembler._load_wagon_region -> support_regions
        gf.assemble_global_train_state_master_fixed(wagon_regions=...)
            global_fusion.filter_observations_to_wagon_region
              |
    Global Wagon Timeline

Both modes call those same four functions, so the rules and decision semantics
are shared; only the serialization differs (sequential round-trips through
`wagon_region.json`, batch keeps the objects in memory). Asserted below.

Two failure directions matter equally, and the module under test was written for
both: a 0.33 s noise burst must not create a BRAKE_VAN, and the real
single-segment brake van behind the loco -- 3.87 s at 0.998 on this project's own
data -- must not be deleted. A rule that only suppressed noise would inflate the
count by erasing real vehicles.

No model, no video, no S3: these drive the real validation functions with
constructed labels.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "wagon_count"))

from core import constants as C                                   # noqa: E402
import train_structure as ts                                      # noqa: E402
import temporal_classification as tcls                            # noqa: E402
import global_fusion as gf                                        # noqa: E402
from global_train_state import SegmentClass                       # noqa: E402

FPS = 15.0
W, E, B = SegmentClass.WAGON, SegmentClass.ENGINE, SegmentClass.BRAKE_VAN


def _region(camera_id, labels, *, seg_frames=60):
    """This camera's wagon region, from ONLY this camera's labels."""
    segments = [(i * seg_frames, (i + 1) * seg_frames - 1)
                for i in range(len(labels))]
    return ts.build_local_wagon_region(
        camera_id=camera_id, segments=segments, labels=list(labels), fps=FPS,
        classifier_model="test.pt", verbose=False)


def _Cls(idx, sf, ef, label, conf):
    """A real `_MasterClassification`, as the classifier emits."""
    from global_train_state import _MasterClassification
    return _MasterClassification(segment_index=idx, start_frame=sf,
                                 end_frame=ef, label=label, confidence=conf)


class _Obs:
    """Minimal GapObservation stand-in: only `local_time` is consulted."""

    def __init__(self, t):
        self.local_time = t
        self.camera_id = "X"
        self.local_track_id = 1


# ---------------------------------------------------------------------------
# 1. Per camera, independently, with no other camera present
# ---------------------------------------------------------------------------

class TestEachCameraValidatesAlone(unittest.TestCase):
    """The decision must not need the other three feeds to exist."""

    def test_all_four_cameras_build_a_region_from_their_own_labels(self):
        for cam in C.ALL_CAMERAS:
            with self.subTest(camera=cam):
                reg = _region(cam, [E, W, W, W, W, B])
                self.assertEqual(reg.camera_id, cam)
                self.assertTrue(reg.found, f"{cam} found no wagon region")

    def test_left_up_top_validates_with_its_own_classifier(self):
        """LEFT_UP_TOP now loads ltop.pt; validation is unchanged by that."""
        self.assertEqual(
            ts.classification_model_for(C.CAMERA_LEFT_UP_TOP), "ltop.pt")
        reg = _region(C.CAMERA_LEFT_UP_TOP, [E, W, W, W])
        self.assertTrue(reg.found)
        self.assertEqual(reg.classifier_model, "test.pt")

    def test_the_region_excludes_the_leading_engine(self):
        reg = _region(C.CAMERA_RIGHT_UP_TOP, [E, W, W, W], seg_frames=60)
        # segment 0 is frames 0..59 -> 0.0 .. ~3.93 s
        self.assertFalse(reg.contains_time(1.0),
                         "an engine-time observation is inside the region")
        self.assertTrue(reg.contains_time(6.0),
                        "a wagon-time observation is outside the region")

    def test_the_region_excludes_a_trailing_brake_van(self):
        reg = _region(C.CAMERA_LEFT_UP, [W, W, W, B], seg_frames=60)
        self.assertTrue(reg.contains_time(1.0))
        self.assertFalse(reg.contains_time(13.0))

    def test_a_camera_with_no_classifier_restricts_nothing(self):
        """Unknown must not silently exclude real evidence."""
        reg = ts.build_local_wagon_region(
            camera_id=C.CAMERA_LEFT_UP_TOP, segments=[], labels=[], fps=FPS,
            classifier_model="", verbose=False)
        self.assertFalse(reg.found)
        self.assertTrue(reg.contains_time(1.0),
                        "an unknown region must not exclude anything")
        self.assertTrue(reg.reason, "an unknown region must say why")


# ---------------------------------------------------------------------------
# 2. The false positive the spec names
# ---------------------------------------------------------------------------

class TestFalseWagonIsRejectedBeforeAssembly(unittest.TestCase):

    def test_a_single_frame_burst_does_not_become_a_brake_van(self):
        """One 0.6-confidence sample must not flip a segment's class.

        This is the noise direction: a spurious non-wagon label at the end of
        the train moves FIRST/LAST_VALID_WAGON and changes the count for no
        physical reason.
        """
        cls = [_Cls(0, 0, 59, W, 0.99), _Cls(1, 60, 119, W, 0.99),
               _Cls(2, 120, 124, B, 0.60),          # 0.33 s burst
               _Cls(3, 125, 184, W, 0.99), _Cls(4, 185, 244, W, 0.99)]
        smoothed, _res = tcls.apply_temporal_classification(
            cls, FPS, camera_id=C.CAMERA_RIGHT_UP, verbose=False)
        self.assertEqual([c.label for c in smoothed],
                         [W, W, W, W, W],
                         "a 0.33 s burst survived into the stable labels")

    def test_an_engine_kept_as_wagon_is_still_excluded_by_the_region(self):
        """Defence in depth: even if a label slipped through, the engine's TIME
        is outside the wagon region, so fusion drops its observations."""
        reg = _region(C.CAMERA_RIGHT_UP_TOP, [E, E, W, W, W], seg_frames=60)
        inside, outside = gf.filter_observations_to_wagon_region(
            [_Obs(1.0), _Obs(5.0), _Obs(9.0)], reg)
        self.assertEqual(len(outside), 2, "engine observations were not excluded")
        self.assertEqual(len(inside), 1)

    def test_excluded_observations_are_reported_not_deleted(self):
        """A rejection must stay auditable."""
        reg = _region(C.CAMERA_LEFT_UP_TOP, [E, W, W], seg_frames=60)
        inside, outside = gf.filter_observations_to_wagon_region(
            [_Obs(1.0), _Obs(6.0)], reg)
        self.assertTrue(outside, "the rejected observation vanished")
        self.assertEqual(len(inside) + len(outside), 2)


# ---------------------------------------------------------------------------
# 3. The reverse direction, which matters just as much
# ---------------------------------------------------------------------------

class TestGenuineWagonsAreNotFalselyRejected(unittest.TestCase):

    def test_a_genuine_single_segment_brake_van_survives(self):
        """The real one on this project's data is ONE segment, 3.87 s at 0.998.

        A "require 3 consecutive segments" rule would delete it, move
        FIRST_VALID_WAGON and INFLATE the count -- suppressing a real vehicle.
        Persistence is measured in seconds for exactly this reason.
        """
        cls = [_Cls(0, 0, 57, E, 0.99),
               _Cls(1, 58, 116, B, 0.998),          # 3.87 s, one segment
               _Cls(2, 117, 176, W, 0.99), _Cls(3, 177, 236, W, 0.99)]
        smoothed, _res = tcls.apply_temporal_classification(
            cls, FPS, camera_id=C.CAMERA_RIGHT_UP, verbose=False)
        self.assertEqual(smoothed[1].label, B,
                         "the genuine single-segment brake van was deleted")

    def test_a_wagon_with_transient_misses_stays_a_wagon(self):
        cls = [_Cls(0, 0, 59, W, 0.97), _Cls(1, 60, 64, E, 0.55),
               _Cls(2, 65, 124, W, 0.97), _Cls(3, 125, 129, B, 0.51),
               _Cls(4, 130, 189, W, 0.97)]
        smoothed, _res = tcls.apply_temporal_classification(
            cls, FPS, camera_id=C.CAMERA_LEFT_UP, verbose=False)
        self.assertEqual([c.label for c in smoothed], [W] * 5,
                         "transient misses removed a genuine wagon")

    def test_an_all_wagon_camera_keeps_every_segment(self):
        cls = [_Cls(i, i * 60, i * 60 + 59, W, 0.95) for i in range(8)]
        smoothed, _res = tcls.apply_temporal_classification(
            cls, FPS, camera_id=C.CAMERA_RIGHT_UP_TOP, verbose=False)
        self.assertEqual([c.label for c in smoothed], [W] * 8)


# ---------------------------------------------------------------------------
# 4. Cameras arriving one at a time: the decision must survive on disk
# ---------------------------------------------------------------------------

class TestTheDecisionSurvivesSerialization(unittest.TestCase):
    """A camera finishes and persists before the others arrive, so the verdict
    and its reason must round-trip to final assembly through the bundle."""

    def test_the_region_round_trips_through_the_bundle(self):
        from core.camera_evidence import CameraEvidenceBundle
        from orchestrator.global_assembler import _load_wagon_region

        reg = _region(C.CAMERA_RIGHT_UP_TOP, [E, W, W, W], seg_frames=60)
        with tempfile.TemporaryDirectory() as tmp:
            b = CameraEvidenceBundle(root=tmp, camera_id=C.CAMERA_RIGHT_UP_TOP)
            b.write_json("wagon_region.json", reg.to_dict())
            back = _load_wagon_region(b)

        self.assertIsNotNone(back, "the persisted region did not load")
        self.assertEqual(back.camera_id, reg.camera_id)
        self.assertEqual(back.found, reg.found)
        self.assertEqual(back.start_time, reg.start_time)
        self.assertEqual(back.end_time, reg.end_time)
        # The decision must still EXCLUDE the engine after a disk round-trip.
        self.assertFalse(back.contains_time(1.0))
        self.assertTrue(back.contains_time(6.0))

    def test_the_reason_is_preserved_for_diagnosis(self):
        from core.camera_evidence import CameraEvidenceBundle
        from orchestrator.global_assembler import _load_wagon_region

        reg = ts.build_local_wagon_region(
            camera_id=C.CAMERA_LEFT_UP, segments=[], labels=[], fps=FPS,
            verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            b = CameraEvidenceBundle(root=tmp, camera_id=C.CAMERA_LEFT_UP)
            b.write_json("wagon_region.json", reg.to_dict())
            back = _load_wagon_region(b)
        self.assertTrue(getattr(back, "reason", ""),
                        "the rejection reason was lost in serialization")

    def test_a_camera_with_no_persisted_region_excludes_nothing(self):
        """A feed that has not arrived must not restrict the others."""
        from core.camera_evidence import CameraEvidenceBundle
        from orchestrator.global_assembler import _load_wagon_region

        with tempfile.TemporaryDirectory() as tmp:
            b = CameraEvidenceBundle(root=tmp, camera_id=C.CAMERA_LEFT_UP_TOP)
            self.assertIsNone(_load_wagon_region(b))
        inside, outside = gf.filter_observations_to_wagon_region(
            [_Obs(1.0), _Obs(9.0)], None)
        self.assertEqual(len(inside), 2)
        self.assertEqual(outside, [])


# ---------------------------------------------------------------------------
# 5. Batch and sequential share the rules
# ---------------------------------------------------------------------------

class TestBothModesShareTheValidation(unittest.TestCase):

    def _src(self, *parts):
        return open(os.path.join(ROOT, *parts), encoding="utf-8").read()

    def test_both_modes_call_the_same_four_functions(self):
        seq = self._src("orchestrator", "camera_pipeline.py")
        batch = self._src("wagon_count", "run_global_count.py")
        for fn in ("validate_gap_events", "apply_temporal_classification",
                   "build_local_wagon_region"):
            self.assertIn(fn, seq, f"sequential does not call {fn}")
            self.assertIn(fn, batch, f"batch does not call {fn}")

    @staticmethod
    def _calls(*parts):
        """Function names this module actually CALLS.

        AST, not substring search: `global_assembler`'s docstring names several
        of these while calling none of them, and a text match on prose reads a
        comment as a call.
        """
        import ast
        tree = ast.parse(open(os.path.join(ROOT, *parts), encoding="utf-8").read())
        out = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name):
                    out.add(f.id)
                elif isinstance(f, ast.Attribute):
                    out.add(f.attr)
        return out

    def test_sequential_creates_no_second_counting_system(self):
        """Assembly CONSUMES the persisted region; it never re-derives one, and
        runs no gap validation, classification or smoothing of its own."""
        calls = self._calls("orchestrator", "global_assembler.py")
        self.assertIn("_load_wagon_region", calls)
        self.assertIn("wagon_regions=",
                      self._src("orchestrator", "global_assembler.py"))
        for fn in ("build_local_wagon_region", "validate_gap_events",
                   "apply_temporal_classification", "classify_segments"):
            self.assertNotIn(fn, calls,
                             f"assembly re-runs {fn} instead of reading the "
                             f"camera's persisted decision")

    def test_fusion_is_what_applies_the_region_in_both_modes(self):
        self.assertTrue(hasattr(gf, "filter_observations_to_wagon_region"))

    def test_the_region_is_built_from_one_cameras_data_only(self):
        """Its signature is the arrival-order guarantee: no cross-camera input."""
        import inspect
        params = set(inspect.signature(ts.build_local_wagon_region).parameters)
        self.assertEqual(
            params & {"camera_id", "segments", "labels", "fps"},
            {"camera_id", "segments", "labels", "fps"})
        for forbidden in ("other_cameras", "all_tracks", "state", "master"):
            self.assertNotIn(forbidden, params)
