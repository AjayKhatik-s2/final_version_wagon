"""End-to-end integration across the counting -> inspection boundary.

Uses the SAME four-camera input representation the production pipeline uses
(`TrainBatch` of four `CameraVideo`s keyed by the four camera ids), drives the
real counting engine, writes the real Stage-1 artifacts to disk, loads them
through the real Stage-1 adapter, and pushes the result through the real
Stage-4 fusion and Stage-5 reporting adapter.

What is deliberately NOT exercised: YOLO inference and video decode.  Those
need model weights and a full production run, which must not happen locally.
The counting *logic* under test is entirely stdlib and runs for real.

The asserted wagon count is whatever the correct-count engine computes; it is
never written down as a literal.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from _engine_harness import (
    ALL_CAMERAS, CAMERA_LEFT_UP, CAMERA_LEFT_UP_TOP, CAMERA_RIGHT_UP,
    CAMERA_RIGHT_UP_TOP, drifting_gap_times, run_counting_engine,
    write_stage1_outputs,
)
from core import constants as C
from core.batch import TrainBatch, build_local_batch
from core.global_state_loader import (
    load_global_train_state, load_per_camera_fps, roster_fingerprint,
    verify_roster_integrity,
)


class TestFourCameraIntegration(unittest.TestCase):
    """four camera inputs -> counting -> fusion -> one roster -> inspection."""

    @classmethod
    def setUpClass(cls):
        cls.times = drifting_gap_times(13, start=27.0)
        # Distinct, realistic clock offsets per camera -- the reason the
        # counting engine estimates them at all.
        cls.deltas = {CAMERA_LEFT_UP: 0.9, CAMERA_RIGHT_UP_TOP: -0.2,
                      CAMERA_LEFT_UP_TOP: -2.7}
        cls.engine_state, cls.tracks = run_counting_engine(
            cls.times,
            {cam: [t - d for t in cls.times] for cam, d in cls.deltas.items()})

        cls.tmp = tempfile.TemporaryDirectory()
        cls.paths = write_stage1_outputs(cls.engine_state, cls.tracks,
                                         os.path.join(cls.tmp.name, "global_state"))
        # The real Stage-1 -> downstream boundary.
        cls.state = load_global_train_state(cls.paths["state"])
        cls.per_camera_fps = load_per_camera_fps(cls.paths["tracking"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # -- the four-camera input representation -------------------------------

    def test_batch_uses_the_existing_four_camera_representation(self):
        with tempfile.TemporaryDirectory() as d:
            paths = {}
            for cam in ALL_CAMERAS:
                p = os.path.join(d, f"{cam.lower()}.mp4")
                with open(p, "wb") as f:
                    f.write(b"\x00")
                paths[cam] = p
            batch = build_local_batch(paths, batch_key="20260101_000000")
        self.assertIsInstance(batch, TrainBatch)
        self.assertEqual(batch.present_cameras(), list(C.ALL_CAMERAS))
        self.assertEqual(batch.missing_cameras(), [])
        self.assertTrue(batch.is_complete())

    def test_all_four_cameras_contributed_tracking_metadata(self):
        self.assertEqual(sorted(self.per_camera_fps), sorted(ALL_CAMERAS))
        for cam in ALL_CAMERAS:
            self.assertGreater(self.per_camera_fps[cam], 0.0)

    # -- the count comes from the correct-count engine -----------------------

    def test_count_is_produced_by_the_new_engine(self):
        self.assertEqual(self.state.fusion_mode, "master-fixed")
        self.assertEqual(self.state.total_wagons,
                         self.engine_state.total_wagons)
        self.assertEqual(self.state.total_wagons, self.state.master_wagon_count)
        self.assertGreater(self.state.total_wagons, 0)

    def test_new_engine_invariants_are_carried_across_the_boundary(self):
        checks = self.state.invariant_checks
        self.assertTrue(checks, "invariant block lost crossing the boundary")
        self.assertTrue(checks["invariant_holds"], checks.get("violations"))
        self.assertEqual(checks["right_up_final_gap_count"],
                         checks["global_gap_count"])
        self.assertEqual(self.state.global_gap_count,
                         len(self.tracks[CAMERA_RIGHT_UP].gaps))

    def test_count_equals_master_wagon_units_not_a_per_camera_count(self):
        """RIGHT_UP is the authority; support local counts must not drive it."""
        per_cam = self.state.per_camera_local_counts
        self.assertIn(CAMERA_RIGHT_UP, per_cam)
        self.assertEqual(self.state.per_camera_status[CAMERA_RIGHT_UP],
                         "master/authoritative")
        for cam in (CAMERA_LEFT_UP, CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP):
            self.assertTrue(self.state.per_camera_status[cam].startswith("support/"))

    def test_camera_synchronization_is_exposed_and_plausible(self):
        offsets = self.state.camera_time_offsets()
        self.assertEqual(offsets[CAMERA_RIGHT_UP], 0.0, "master is the reference")
        for cam, injected in self.deltas.items():
            with self.subTest(camera=cam):
                self.assertIn(cam, offsets, f"{cam} offset unresolved")
                # Recovered to within the master's frame quantization.
                self.assertAlmostEqual(offsets[cam], injected,
                                       delta=2.0 / self.state.master_fps)

    # -- the roster handed to inspection ------------------------------------

    def test_roster_is_well_formed(self):
        self.assertEqual(verify_roster_integrity(self.state), [])
        self.assertEqual([w.global_id for w in self.state.wagons],
                         [f"GW_{i}" for i in range(1, self.state.total_wagons + 1)])

    def test_json_keeps_the_original_dashboard_keys(self):
        with open(self.paths["state"], "r", encoding="utf-8") as f:
            doc = json.load(f)
        # Same schema id and same key set the previous engine emitted, so the
        # dashboard cannot tell a different pipeline produced this.
        self.assertEqual(doc["schema"], "wagon_eye.global_train_state.v1")
        for key in ("master_camera", "master_fps", "master_total_frames",
                    "total_wagons", "regular_wagon_count", "engine_count",
                    "brake_van_count", "wagons", "per_camera_local_counts",
                    "per_camera_gap_counts", "per_camera_status",
                    "corrections_applied", "fallback_used", "notes"):
            self.assertIn(key, doc, f"dashboard key {key} disappeared")
        for key in ("global_id", "wagon_index", "start_frame_master",
                    "end_frame_master", "start_time", "end_time",
                    "classification", "classification_confidence",
                    "supporting_cameras", "split_from_global_id",
                    "leading_gap", "trailing_gap"):
            self.assertIn(key, doc["wagons"][0])

    # -- inspection consumes it unchanged -----------------------------------

    def test_full_downstream_chain_consumes_the_roster(self):
        from fusion import wagon_state_builder
        from reporting import _adapter

        fp = roster_fingerprint(self.state)
        with tempfile.TemporaryDirectory() as tmp:
            for feature in ("door", "ocr", "load", "damage"):
                d = os.path.join(tmp, feature)
                os.makedirs(d, exist_ok=True)
                for w in self.state.wagons:
                    with open(os.path.join(d, f"{w.global_id}.json"), "w",
                              encoding="utf-8") as f:
                        json.dump({"global_id": w.global_id,
                                   "feature": feature,
                                   "status": C.STATUS_OK,
                                   "supporting_cameras": [C.CAMERA_RIGHT_UP]}, f)
            unified = wagon_state_builder.build(
                state=self.state, wagon_states_root=tmp, verbose=False)
            vm = _adapter.build_legacy_view_model(
                state=self.state, unified=unified, wagon_states_root=tmp,
                evidence_root=None, missing_cameras=[])

        self.assertEqual(len(unified), self.state.total_wagons)
        self.assertEqual(len(vm.merged_wagons), self.state.total_wagons)
        self.assertEqual(vm.summary_kpis["total_wagons"], self.state.total_wagons)
        # Inspection did not touch the roster.
        self.assertEqual(roster_fingerprint(self.state), fp)

    def test_materializer_projects_the_same_roster_onto_every_camera(self):
        from materializer.wagon_cache_builder import _wagon_local_range

        offsets = self.state.camera_time_offsets()
        for cam in ALL_CAMERAS:
            with self.subTest(camera=cam):
                t = self.tracks[cam]
                seen = []
                for w in self.state.wagons:
                    sf, ef = _wagon_local_range(w, t.fps, t.total_frames,
                                                offsets.get(cam, 0.0))
                    if ef >= sf:
                        seen.append(w.global_id)
                        self.assertGreaterEqual(sf, 0)
                        self.assertLess(ef, t.total_frames)
                # Every camera sees the train, and only ever with global ids.
                self.assertGreater(len(seen), 0)
                self.assertTrue(set(seen).issubset(
                    {w.global_id for w in self.state.wagons}))

    def test_offset_free_state_reproduces_shared_t0_projection(self):
        """A state with no resolved offsets must project exactly as before."""
        from materializer.wagon_cache_builder import _wagon_local_range

        w = self.state.wagons[0]
        fps, total = 15.0, 10_000
        self.assertEqual(_wagon_local_range(w, fps, total, 0.0),
                         (int(round(w.start_time * fps)),
                          int(round(w.end_time * fps)) - 1))


class TestCameraOffsetsReachMaterialization(unittest.TestCase):
    """A resolved offset must actually move the frames a camera caches.

    If offsets were merely recorded and never applied, a camera whose clock
    differs from the master would cache a neighbouring wagon's frames and
    silently corrupt door / load / damage results.
    """

    def setUp(self):
        self.times = drifting_gap_times(11, start=30.0)
        delta = -2.7                     # LEFT_UP_TOP-sized skew
        self.delta = delta
        self.engine_state, self.tracks = run_counting_engine(
            self.times, {CAMERA_LEFT_UP_TOP: [t - delta for t in self.times]})
        with tempfile.TemporaryDirectory() as d:
            paths = write_stage1_outputs(self.engine_state, self.tracks, d)
            self.state = load_global_train_state(paths["state"])

    def test_offset_survives_the_stage1_json_round_trip(self):
        resolved = self.state.camera_time_offsets()
        self.assertIn(CAMERA_LEFT_UP_TOP, resolved)
        self.assertAlmostEqual(resolved[CAMERA_LEFT_UP_TOP], self.delta,
                               delta=2.0 / self.state.master_fps)

    def test_projection_shifts_by_exactly_the_offset(self):
        from materializer.wagon_cache_builder import _wagon_local_range

        t = self.tracks[CAMERA_LEFT_UP_TOP]
        delta = self.state.camera_time_offsets()[CAMERA_LEFT_UP_TOP]
        w = self.state.wagons[len(self.state.wagons) // 2]

        naive = _wagon_local_range(w, t.fps, t.total_frames, 0.0)
        shifted = _wagon_local_range(w, t.fps, t.total_frames, delta)

        expected_shift = int(round(-delta * t.fps))
        self.assertNotEqual(naive, shifted,
                            "a resolved offset had no effect on materialization")
        self.assertAlmostEqual(shifted[0] - naive[0], expected_shift, delta=1)
        self.assertAlmostEqual(shifted[1] - naive[1], expected_shift, delta=1)

    def test_renderer_applies_the_same_projection_as_the_materializer(self):
        """Overlay boxes and cached frames must agree, or the video lies."""
        from materializer.wagon_cache_builder import _wagon_local_range
        from rendering.feature_overlay_renderer import _map_wagon_to_local_frames

        t = self.tracks[CAMERA_LEFT_UP_TOP]
        delta = self.state.camera_time_offsets()[CAMERA_LEFT_UP_TOP]
        for w in self.state.wagons:
            self.assertEqual(
                _wagon_local_range(w, t.fps, t.total_frames, delta),
                _map_wagon_to_local_frames(w, t.fps, t.total_frames, delta),
                f"{w.global_id}: renderer and materializer disagree")

    def test_orchestrator_feeds_offsets_to_both_consumers(self):
        with open(os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "orchestrator",
                "master_runner.py"), "r", encoding="utf-8") as f:
            src = f.read()
        self.assertEqual(src.count("camera_offsets=recon.camera_offsets"), 2,
                         "camera offsets must reach BOTH the materializer "
                         "(Stage 2) and the overlay renderer (Stage 4b)")

    def test_unresolved_camera_falls_back_to_shared_t0(self):
        """An UNRESOLVED camera is never given a guessed shift."""
        doc = self.engine_state.to_dict()
        for meta in doc["camera_offsets"].values():
            meta["status"] = "UNRESOLVED"
        from core.global_state_loader import parse_global_train_state
        degraded = parse_global_train_state(doc)
        self.assertEqual(degraded.camera_time_offsets(), {})
        self.assertEqual(degraded.camera_time_offset(CAMERA_LEFT_UP_TOP), 0.0)
        # ...and the roster itself is completely unaffected by that.
        self.assertEqual(degraded.total_wagons, self.state.total_wagons)


if __name__ == "__main__":
    unittest.main()
