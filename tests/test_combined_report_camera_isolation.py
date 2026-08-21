"""A snapshot belongs to one camera, and the combined report must honour that.

The bug this pins: `_panel_snapshot()` used to resolve a damaged wagon's top
panel as "own camera first, then the best across BOTH top cameras". When only
one top camera had a damage track -- the usual case, since the two see
different sides of the roof -- both panels resolved to the SAME file, and the
PDF showed one picture twice under two different camera headings. The two top
views look alike, so it read as correct.

A second, quieter instance: `evidence/<gw>/load/best_frame.jpg` is a single
file written by whichever top camera won, with the winner recorded as
`source_camera`. Resolving it by wagon id alone handed a LEFT_UP_TOP frame to
the RIGHT_UP_TOP panel whenever the master had no load evidence.

Neither is a ranking problem, so neither is tested by ranking. Each camera gets
a distinct grey level and the assertions are on decoded pixels: the identity of
the image that actually came back.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

import cv2
import numpy as np

from core import constants as C
from core.global_state_loader import GlobalTrainState, GlobalWagon
from core.unified_wagon_state import UnifiedWagonState
from reporting import _evidence_lookup as ev
from test_camera_evidence_isolation import embedded_image_means

RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP
OTHER = {RUT: LUT, LUT: RUT}

#: Every source of an image gets its own value, so a decoded mean names its
#: origin unambiguously -- including which camera's cache a fallback used.
TINT = {
    "damage_RIGHT_UP_TOP": 30,
    "damage_LEFT_UP_TOP": 70,
    "load": 110,
    "cache_RIGHT_UP_TOP": 150,
    "cache_LEFT_UP_TOP": 190,
    "door_RIGHT_UP": 220,
    "door_LEFT_UP": 245,
}
GW = "GW_1"


def _tile(v):
    return np.full((40, 40, 3), v, dtype=np.uint8)


def _write(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, _tile(value))


def _fixture(root, *, damage_cams=(RUT, LUT), load_source=None,
             cache_cams=(), same_track_idx=False):
    """Build an evidence tree. Defaults make both top cameras report damage.

    `same_track_idx=True` gives BOTH cameras track_idx 1 with identical
    confidence and frame index, so nothing but `camera_id` separates the two
    records -- the condition under which a camera-blind key silently collapses.
    """
    evd = os.path.join(root, "evidence")
    states = os.path.join(root, "wagon_states")
    cache = os.path.join(root, "wagon_cache")

    tracks = []
    for i, cam in enumerate(damage_cams, start=1):
        idx = 1 if same_track_idx else i
        # Camera-scoped slot, exactly as features/damage writes it. With
        # same_track_idx the two cameras share an INDEX and still cannot share
        # a file -- which is the point of putting the camera in the identity.
        from core.evidence_identity import damage_track_slot
        _write(os.path.join(evd, GW, "damage",
                            f"{damage_track_slot(idx, cam)}.jpg"),
               TINT[f"damage_{cam}"])
        tracks.append({"track_idx": idx, "camera_id": cam, "track_id": 1,
                       "class_name": "inner_wall_damage",
                       "confidence": 0.77, "best_confidence": 0.77,
                       "best_frame_idx": 123, "bbox": [1, 1, 20, 20]})
    if tracks:
        d = os.path.join(evd, GW, "damage")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"global_id": GW, "feature": "damage",
                       "top_damage": C.DAMAGE_PRESENT, "tracks": tracks}, f)

    if load_source is not None:
        _write(os.path.join(evd, GW, "load", "best_frame.jpg"), TINT["load"])
        d = os.path.join(evd, GW, "load")
        with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"global_id": GW, "feature": "load",
                       "fused_status": C.LOAD_LOADED,
                       "source_camera": load_source}, f)

    for cam in cache_cams:
        folder = C.CAMERA_FOLDER[cam]
        for n in (10, 11, 12):
            _write(os.path.join(cache, GW, folder, f"frame_{n:06d}.jpg"),
                   TINT[f"cache_{cam}"])

    dd = os.path.join(states, "damage")
    os.makedirs(dd, exist_ok=True)
    with open(os.path.join(dd, f"{GW}.json"), "w", encoding="utf-8") as f:
        json.dump({"global_id": GW, "feature": "damage", "status": "OK",
                   "top_damage": (C.DAMAGE_PRESENT if damage_cams
                                  else C.NO_DATA),
                   "top_damage_details": tracks,
                   "supporting_cameras": list(damage_cams)}, f)
    return evd, states, cache, tracks


def _state():
    return GlobalTrainState(
        total_wagons=1,
        wagons=(GlobalWagon(global_id=GW, wagon_index=1,
                            start_frame_master=0, end_frame_master=59,
                            start_time=0.0, end_time=4.0,
                            classification=C.CLASS_WAGON,
                            classification_confidence=1.0),),
        master_camera=C.MASTER_CAMERA, master_fps=15.0,
        master_total_frames=600)


def _unified(states, tracks):
    u = UnifiedWagonState(global_id=GW, wagon_index=1)
    u.classification = C.CLASS_WAGON
    u.top_damage = C.DAMAGE_PRESENT if tracks else C.NO_DATA
    u.top_damage_details = list(tracks)
    u.load_status = C.LOAD_LOADED
    return u


def _mean_of(path):
    img = cv2.imread(path)
    return None if img is None else float(img.mean())


def _origin_of(path, tol=5.0):
    """Which fixture source produced the image at `path`."""
    m = _mean_of(path)
    if m is None:
        return None
    for name, v in TINT.items():
        if abs(m - v) < tol:
            return name
    return f"<unknown mean {m}>"


class TestPanelResolverNeverCrossesCameras(unittest.TestCase):
    """`_panel_snapshot` is the exact point where identity used to be lost."""

    def _panel(self, evd, cache, cam, u):
        from reporting.combined_train_report import _panel_snapshot
        return _panel_snapshot(u, cam, evd, cache, GW)

    def test_each_top_camera_gets_its_own_damage_frame(self):
        with tempfile.TemporaryDirectory() as root:
            evd, states, cache, tracks = _fixture(root)
            u = _unified(states, tracks)
            for cam in (RUT, LUT):
                with self.subTest(camera=cam):
                    got = self._panel(evd, cache, cam, u)
                    self.assertEqual(_origin_of(got), f"damage_{cam}")

    def test_identical_track_ids_do_not_collapse_the_two_cameras(self):
        """Same track_idx, confidence and frame on both -- only camera_id differs."""
        with tempfile.TemporaryDirectory() as root:
            evd, states, cache, tracks = _fixture(root, same_track_idx=True)
            u = _unified(states, tracks)
            origins = {cam: _origin_of(self._panel(evd, cache, cam, u))
                       for cam in (RUT, LUT)}
            for cam, origin in origins.items():
                self.assertEqual(origin, f"damage_{cam}",
                                 f"{cam} received {origin} -- identical track "
                                 f"indices must not collapse the two cameras")

    def test_a_camera_with_no_damage_does_not_borrow_the_other_s(self):
        """The reported bug, at its source."""
        for present in (RUT, LUT):
            absent = OTHER[present]
            with self.subTest(has_damage=present):
                with tempfile.TemporaryDirectory() as root:
                    evd, states, cache, tracks = _fixture(
                        root, damage_cams=(present,), cache_cams=(absent,))
                    u = _unified(states, tracks)
                    got = self._panel(evd, cache, absent, u)
                    self.assertNotEqual(_origin_of(got), f"damage_{present}",
                                        f"{absent} panel showed {present} damage")
                    # It falls back only WITHIN its own identity.
                    self.assertEqual(_origin_of(got), f"cache_{absent}")

    def test_missing_evidence_yields_nothing_rather_than_a_substitute(self):
        """No own damage, no own load, no own cache -> explicit nothing."""
        with tempfile.TemporaryDirectory() as root:
            evd, states, cache, tracks = _fixture(
                root, damage_cams=(RUT,), cache_cams=(RUT,))
            u = _unified(states, tracks)
            got = self._panel(evd, cache, LUT, u)
            self.assertIsNone(got, f"LEFT_UP_TOP fabricated {_origin_of(got)}")

    def test_load_frame_is_resolved_through_its_source_camera(self):
        """load/best_frame.jpg belongs to whichever camera actually produced it."""
        with tempfile.TemporaryDirectory() as root:
            evd, states, cache, tracks = _fixture(
                root, damage_cams=(), load_source=LUT)
            u = _unified(states, tracks)
            self.assertEqual(_origin_of(self._panel(evd, cache, LUT, u)),
                             "load")
            self.assertIsNone(self._panel(evd, cache, RUT, u),
                              "RIGHT_UP_TOP panel showed LEFT_UP_TOP's load frame")

    def test_the_camera_blind_resolver_is_gone(self):
        import reporting.combined_train_report as m
        self.assertFalse(hasattr(m, "_best_damage_snapshot_any"),
                         "the cross-camera fallback must not come back")

    def test_lookup_helper_requires_a_camera_for_load(self):
        import inspect
        p = inspect.signature(ev.load_snapshot).parameters
        self.assertIn("camera_id", p)
        self.assertIs(p["camera_id"].default, inspect.Parameter.empty)


class TestRenderedCombinedReport(unittest.TestCase):
    """End to end through the real builder, asserting on embedded pixels."""

    def _build(self, root, **kw):
        from fusion import wagon_state_builder
        from reporting import combined_train_report
        evd, states, cache, tracks = _fixture(root, **kw)
        state = _state()
        unified = wagon_state_builder.build(
            state=state, wagon_states_root=states,
            write_per_wagon_json=False, verbose=False)
        out = combined_train_report.build(
            state=state, unified=unified,
            output_dir=os.path.join(root, "reports"),
            batch_key="iso", source_video_urls={}, processed_video_urls={},
            evidence_root=evd, wagon_states_root=states, cache_root=cache,
            missing_cameras=[], camera_pdf_urls={}, logo_path=None,
            verbose=False)
        return out.get("pdf_path")

    def _origins(self, pdf, tol=5.0):
        found = set()
        for m in embedded_image_means(pdf):
            for name, v in TINT.items():
                if abs(m - v) < tol:
                    found.add(name)
        return found

    def test_both_cameras_damage_appears(self):
        with tempfile.TemporaryDirectory() as root:
            pdf = self._build(root, damage_cams=(RUT, LUT))
            self.assertTrue(pdf and os.path.isfile(pdf))
            origins = self._origins(pdf)
            self.assertIn("damage_RIGHT_UP_TOP", origins)
            self.assertIn("damage_LEFT_UP_TOP", origins)

    def test_one_sided_damage_never_appears_twice_as_two_cameras(self):
        """With only RIGHT_UP_TOP damaged, LEFT_UP_TOP must use its own frame."""
        with tempfile.TemporaryDirectory() as root:
            pdf = self._build(root, damage_cams=(RUT,), cache_cams=(RUT, LUT))
            origins = self._origins(pdf)
            self.assertIn("damage_RIGHT_UP_TOP", origins)
            self.assertNotIn("damage_LEFT_UP_TOP", origins)
            self.assertIn("cache_LEFT_UP_TOP", origins,
                          "LEFT_UP_TOP should fall back to its OWN cache frame")

    def test_absent_camera_contributes_no_image_at_all(self):
        with tempfile.TemporaryDirectory() as root:
            pdf = self._build(root, damage_cams=(LUT,), cache_cams=(LUT,))
            origins = self._origins(pdf)
            self.assertIn("damage_LEFT_UP_TOP", origins)
            self.assertNotIn("cache_RIGHT_UP_TOP", origins)
            self.assertNotIn("damage_RIGHT_UP_TOP", origins)

    def test_render_is_non_vacuous(self):
        """Guard the guard: a PDF with no decodable image proves nothing."""
        with tempfile.TemporaryDirectory() as root:
            pdf = self._build(root, damage_cams=(RUT, LUT))
            self.assertTrue(embedded_image_means(pdf),
                            "no image decoded from the combined report")


class TestGlobalDamageFusion(unittest.TestCase):
    """Damage is a wagon property assembled from camera observations.

    One camera is enough. Silence from the other camera is not evidence of
    absence -- the two differ in angle, timing, occlusion and detection
    quality.
    """

    def _fuse(self, root, damage_cams):
        from fusion import wagon_state_builder
        _evd, states, _cache, _tracks = _fixture(root, damage_cams=damage_cams)
        return wagon_state_builder.build(
            state=_state(), wagon_states_root=states,
            write_per_wagon_json=False, verbose=False)[GW]

    def test_one_camera_is_sufficient(self):
        for cam in (RUT, LUT):
            with self.subTest(camera=cam), tempfile.TemporaryDirectory() as root:
                u = self._fuse(root, (cam,))
                self.assertEqual(u.top_damage, C.DAMAGE_PRESENT,
                                 f"{cam} alone must mark the wagon damaged")
                self.assertEqual(u.damage_cameras, [cam])

    def test_silence_from_one_camera_does_not_erase_the_other(self):
        with tempfile.TemporaryDirectory() as root:
            u = self._fuse(root, (RUT,))
            self.assertEqual(u.top_damage, C.DAMAGE_PRESENT)
            self.assertNotIn(LUT, u.damage_observations_by_camera())

    def test_both_cameras_resolve_to_one_wagon_with_both_observations(self):
        with tempfile.TemporaryDirectory() as root:
            u = self._fuse(root, (RUT, LUT))
            obs = u.damage_observations_by_camera()
            self.assertEqual(sorted(obs), sorted([RUT, LUT]))
            self.assertEqual(u.global_id, GW, "must stay ONE global wagon")
            for cam in (RUT, LUT):
                self.assertEqual(len(obs[cam]), 1)
                self.assertEqual(obs[cam][0]["camera_id"], cam)

    def test_no_observation_means_absent_not_empty(self):
        with tempfile.TemporaryDirectory() as root:
            u = self._fuse(root, ())
            self.assertEqual(u.damage_observations_by_camera(), {})
            self.assertEqual(u.damage_cameras, [])

    def test_provenance_survives_into_the_state(self):
        with tempfile.TemporaryDirectory() as root:
            u = self._fuse(root, (RUT, LUT))
            for obs in u.top_damage_details:
                self.assertIn("camera_id", obs)
                self.assertIn(obs["camera_id"], (RUT, LUT))

    def test_grouping_invents_no_camera(self):
        u = UnifiedWagonState(global_id=GW, wagon_index=1)
        u.top_damage_details = [{"best_confidence": 0.9}]     # no camera_id
        self.assertEqual(u.damage_observations_by_camera(), {})


if __name__ == "__main__":
    unittest.main()
