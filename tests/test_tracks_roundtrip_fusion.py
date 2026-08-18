"""Persisted tracks must fuse EXACTLY as the in-memory originals do.

This is the pre-flight check for the one thing the synthetic bundle tests
could not cover: `assemble_global_train_state_master_fixed()` consuming
`LocalCameraTracks` that came back off disk rather than straight from the
tracker.

A serialization bug here would not crash -- it would silently change the
roster on the expensive four-camera run. So the test does not merely assert
"fusion returned something": it runs the REAL fusion twice, once on the
in-memory tracks and once on the reconstructed ones, and requires the two
global rosters to be identical.

No YOLO, no video: the tracker output is synthesized by the shared harness
using the same motion band the engine's own validation tests use.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import unittest

from _engine_harness import (  # noqa: F401  (also bootstraps sys.path)
    V4_ROOT, camera_tracks, drifting_gap_times,
    whole_video_wagon_classification,
)

from core import constants as C
from core.camera_evidence import CameraEvidenceBundle
from core.camera_tracks_io import (
    GAP_FIELDS, TRACK_FIELDS, read_tracks, write_tracks,
)

MASTER = C.MASTER_CAMERA
CAMS = C.ALL_CAMERAS


def _four_cameras():
    """Master plus three supports, each on its own clock.

    The supports are offset from the master so the fusion path has real
    alignment work to do -- identical clocks would let a reconstruction bug
    that loses timing pass unnoticed.
    """
    master_times = drifting_gap_times(8, start=30.0)
    offsets = {"LEFT_UP": 1.7, "RIGHT_UP_TOP": -2.3, "LEFT_UP_TOP": 0.9}
    out = {MASTER: camera_tracks(MASTER, master_times)}
    for cam, off in offsets.items():
        out[cam] = camera_tracks(cam, [t + off for t in master_times])
    out[MASTER].classifications = whole_video_wagon_classification(out[MASTER])
    # Absolute frame indices keyed by int -- JSON turns keys into strings, so
    # this is exactly where a round-trip loses fidelity if _plain/reconstruct
    # disagree.
    for cam, tr in out.items():
        tr.raw_frame_detections = {
            g.start_frame: [{"bbox": [10.0, 20.0, 30.0, 40.0],
                             "conf": 0.87, "cls": 0}]
            for g in tr.gaps
        }
    return out


def _persist(root, tracks_by_cam):
    """Write each camera's tracking_full.json where camera_runner writes it."""
    paths = {}
    for cam, tr in tracks_by_cam.items():
        b = CameraEvidenceBundle(root, cam)
        os.makedirs(b.dir, exist_ok=True)
        paths[cam] = write_tracks(
            os.path.join(b.dir, "tracking_full.json"), tr)
        if tr.classifications:
            b.write_json("classification.json",
                         [c.to_dict() for c in tr.classifications])
    return paths


def _fuse(master, supports, classifications):
    """Call fusion exactly as global_assembler.assemble() does."""
    import global_fusion as gf
    return gf.assemble_global_train_state_master_fixed(
        master_tracks=master,
        support_tracks=supports,
        initial_classifications=classifications,
        config=gf.FusionConfig(),
        verbose=False,
        wagon_only=True,
    )


class TestRoundTripIsLossless(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.orig = _four_cameras()
        _persist(self.tmp.name, self.orig)
        self.back = {c: read_tracks(os.path.join(self.tmp.name, c,
                                                 "tracking_full.json"))
                     for c in self.orig}

    def tearDown(self):
        self.tmp.cleanup()

    def test_field_lists_still_match_the_dataclasses(self):
        """If upstream adds a field, this fails instead of losing it silently."""
        from global_train_state import GapEvent, LocalCameraTracks
        gap = {f.name for f in dataclasses.fields(GapEvent)}
        trk = {f.name for f in dataclasses.fields(LocalCameraTracks)}
        self.assertEqual(gap - set(GAP_FIELDS), set())
        self.assertEqual(
            trk - set(TRACK_FIELDS)
            - {"gaps", "classifications", "raw_frame_detections"}, set())

    def test_scalar_metadata_survives(self):
        for cam in self.orig:
            with self.subTest(camera=cam):
                o, b = self.orig[cam], self.back[cam]
                for f in TRACK_FIELDS:
                    self.assertEqual(getattr(o, f), getattr(b, f), f"{cam}.{f}")

    def test_every_gap_field_survives(self):
        for cam in self.orig:
            o, b = self.orig[cam], self.back[cam]
            self.assertEqual(len(o.gaps), len(b.gaps), f"{cam} gap count")
            for i, (go, gb) in enumerate(zip(o.gaps, b.gaps)):
                for f in GAP_FIELDS:
                    with self.subTest(camera=cam, gap=i, field=f):
                        self.assertEqual(getattr(go, f), getattr(gb, f))

    def test_rich_sequences_are_not_dropped(self):
        """The exact fields GapEvent.to_dict() would have thrown away."""
        for cam in self.orig:
            for i, (go, gb) in enumerate(zip(self.orig[cam].gaps,
                                             self.back[cam].gaps)):
                with self.subTest(camera=cam, gap=i):
                    self.assertTrue(gb.hit_frames)
                    self.assertTrue(gb.center_x_trajectory)
                    self.assertTrue(gb.bbox_history)
                    self.assertEqual(len(gb.hit_frames), len(go.hit_frames))
                    self.assertEqual(len(gb.bbox_history),
                                     len(go.bbox_history))
                    self.assertEqual(gb.center_x_trajectory,
                                     go.center_x_trajectory)

    def test_reporting_view_would_have_lost_data(self):
        """Negative control: proves the rich fields are worth persisting."""
        d = self.orig[MASTER].gaps[0].to_dict()
        lost = [f for f in ("center_x_trajectory", "hit_frames",
                            "bbox_history") if f not in d]
        self.assertTrue(lost, "if to_dict() is now lossless, simplify this "
                              "module rather than keeping a custom writer")

    def test_raw_detection_keys_come_back_as_ints(self):
        for cam in self.orig:
            with self.subTest(camera=cam):
                self.assertEqual(set(self.back[cam].raw_frame_detections),
                                 set(self.orig[cam].raw_frame_detections))
                for k in self.back[cam].raw_frame_detections:
                    self.assertIsInstance(k, int)

    def test_master_classifications_reload(self):
        from orchestrator.global_assembler import _load_master_classifications
        b = CameraEvidenceBundle(self.tmp.name, MASTER)
        cls = _load_master_classifications(b)
        self.assertEqual(len(cls), len(self.orig[MASTER].classifications))
        self.assertEqual(cls[0].label,
                         self.orig[MASTER].classifications[0].label)


class TestFusionAcceptsReconstructedTracks(unittest.TestCase):
    """The real question: does fusion produce the SAME roster off disk?"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.orig = _four_cameras()
        _persist(self.tmp.name, self.orig)
        self.back = {c: read_tracks(os.path.join(self.tmp.name, c,
                                                 "tracking_full.json"))
                     for c in self.orig}
        self.cls = self.orig[MASTER].classifications

    def tearDown(self):
        self.tmp.cleanup()

    def _both(self):
        a = _fuse(self.orig[MASTER],
                  [self.orig[c] for c in CAMS if c != MASTER], self.cls)
        b = _fuse(self.back[MASTER],
                  [self.back[c] for c in CAMS if c != MASTER], self.cls)
        return a, b

    def test_fusion_runs_on_reconstructed_tracks(self):
        _a, b = self._both()
        self.assertGreater(b.total_wagons, 0,
                           "reconstructed tracks produced an empty roster")

    def test_roster_is_identical(self):
        a, b = self._both()
        self.assertEqual(a.total_wagons, b.total_wagons)
        self.assertEqual([w.global_id for w in a.wagons],
                         [w.global_id for w in b.wagons])

    def test_wagon_boundaries_are_identical(self):
        a, b = self._both()
        for wa, wb in zip(a.wagons, b.wagons):
            with self.subTest(wagon=wa.global_id):
                self.assertEqual(wa.start_frame_master, wb.start_frame_master)
                self.assertEqual(wa.end_frame_master, wb.end_frame_master)
                self.assertAlmostEqual(wa.start_time, wb.start_time, places=6)
                self.assertAlmostEqual(wa.end_time, wb.end_time, places=6)

    def test_camera_offsets_are_identical(self):
        """Support alignment depends on the timing fields most at risk."""
        a, b = self._both()
        self.assertEqual(json.dumps(a.to_dict().get("camera_offsets"),
                                    sort_keys=True, default=str),
                         json.dumps(b.to_dict().get("camera_offsets"),
                                    sort_keys=True, default=str))

    def test_full_state_json_is_identical(self):
        """Strongest form: the whole downstream contract, byte for byte."""
        a, b = self._both()
        self.assertEqual(a.to_json(), b.to_json())

    def test_ids_are_global_only_after_fusion(self):
        _a, b = self._both()
        for w in b.wagons:
            self.assertTrue(w.global_id.startswith("GW_"), w.global_id)


class TestAssemblerInputPreparation(unittest.TestCase):
    """Drive the assembler's own load path, not a re-implementation of it."""

    def test_assembler_reads_the_bundles_it_was_given(self):
        from orchestrator import global_assembler as ga
        with tempfile.TemporaryDirectory() as root:
            orig = _four_cameras()
            _persist(root, orig)
            loaded = {}
            for cam in CAMS:
                p = os.path.join(root, cam, "tracking_full.json")
                t = ga.read_tracks(p)
                self.assertIsNotNone(t, f"{cam} tracking_full.json unreadable")
                loaded[cam] = t
            state = _fuse(loaded[MASTER],
                          [loaded[c] for c in CAMS if c != MASTER],
                          ga._load_master_classifications(
                              CameraEvidenceBundle(root, MASTER)))
            self.assertGreater(state.total_wagons, 0)

    def test_missing_master_tracking_is_reported_not_crashed(self):
        from orchestrator.global_assembler import read_tracks
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(read_tracks(os.path.join(root, "nope.json")))


if __name__ == "__main__":
    unittest.main()
