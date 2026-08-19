"""Camera isolation is an invariant: a snapshot belongs to exactly one camera.

RIGHT_UP_TOP and LEFT_UP_TOP photograph the same wagons from similar angles, so
a swapped snapshot is invisible to the eye -- nothing about the image itself
says which camera took it. That makes the invariant untestable by inspection
and worth pinning hard.

The trap these tests are built to catch: a wagon's `evidence/<gw>/damage/`
directory holds BOTH top cameras' tracks, and `track_1..track_N` is ONE
sequence numbered across the two cameras together (features/damage/processor.py
enumerates `all_evidence`, which is extended once per camera). The filename
carries no camera identity -- only `camera_id` inside `metadata.json` does. Any
resolver that forgets that filter silently mixes the cameras.

So these tests never compare filenames, paths or marker counts. They give each
camera a distinct grey level, render through the real renderer, decode the
JPEGs actually embedded in the PDF, and assert on pixels: each camera's report
must contain its own tint and must NOT contain the other's.
"""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import unittest
import zlib

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

import cv2
import numpy as np

from core import constants as C
from core.camera_evidence import (
    LIFECYCLE, CameraEvidenceBundle, LocalSegment, local_segment_id,
)
from core.global_state_loader import GlobalTrainState, GlobalWagon
from orchestrator import camera_report_adapter as adapter

#: One flat grey per camera -- the mean pixel value IS the camera's fingerprint.
TINT = {"RIGHT_UP_TOP": 60, "LEFT_UP_TOP": 200,
        "RIGHT_UP": 100, "LEFT_UP": 150}
OTHER_TOP = {"RIGHT_UP_TOP": "LEFT_UP_TOP", "LEFT_UP_TOP": "RIGHT_UP_TOP"}
TOPS = ("RIGHT_UP_TOP", "LEFT_UP_TOP")


def _tile(cam):
    return np.full((40, 40, 3), TINT[cam], dtype=np.uint8)


def embedded_image_means(pdf_path):
    """Mean pixel value of every raster actually embedded in the PDF.

    Decodes the image streams rather than counting /DCTDecode markers: a marker
    count would pass on a PDF that embedded the wrong picture, and ReportLab
    ASCII85-encodes the stream so a raw byte scan finds nothing at all.
    """
    with open(pdf_path, "rb") as f:
        raw = f.read()
    means = []
    for m in re.finditer(br"stream\r?\n(.*?)endstream", raw, re.S):
        body = m.group(1).strip()
        for decode in (lambda d: base64.a85decode(d, adobe=True),
                       lambda d: zlib.decompress(d),
                       lambda d: d):
            try:
                data = decode(body)
            except Exception:
                continue
            if data[:2] == b"\xff\xd8":                    # JPEG SOI
                arr = cv2.imdecode(np.frombuffer(data, np.uint8),
                                   cv2.IMREAD_COLOR)
                if arr is not None:
                    means.append(float(arr.mean()))
                break
    return means


def _has_tint(means, cam, tol=5.0):
    return any(abs(m - TINT[cam]) < tol for m in means)


def _top_bundle(root, cam, *, n=2, track_idx=1, conf=0.77, frame_idx=123,
                with_damage=True):
    """A sealed top-camera bundle whose damage evidence is tinted for `cam`.

    Defaults are deliberately IDENTICAL across cameras -- same track index,
    same confidence, same best frame -- so nothing but `camera_id` can
    distinguish the two records.
    """
    b = CameraEvidenceBundle(root, cam)
    os.makedirs(b.dir, exist_ok=True)
    for st in LIFECYCLE[1:]:
        b.advance(st)
    segs = [LocalSegment(local_id=local_segment_id(cam, i), index=i,
                         start_frame=i * 60, end_frame=i * 60 + 59,
                         start_time=float((i - 1) * 4), end_time=float(i * 4),
                         label="WAGON", confidence=0.9)
            for i in range(1, n + 1)]
    b.write_segments(segs)
    img = _tile(cam)
    for s in segs:
        cd = os.path.join(b.dir, "camera_cache", s.local_id,
                          C.CAMERA_FOLDER[cam])
        os.makedirs(cd, exist_ok=True)
        cv2.imwrite(os.path.join(cd, "frame_000001.jpg"), img)
        if not with_damage:
            continue
        ed = os.path.join(b.dir, "evidence", s.local_id, "damage")
        os.makedirs(ed, exist_ok=True)
        cv2.imwrite(os.path.join(ed, f"track_{track_idx}.jpg"), img)
        with open(os.path.join(ed, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"global_id": s.local_id, "feature": "damage",
                       "top_damage": "DAMAGE",
                       "tracks": [{"track_idx": track_idx, "camera_id": cam,
                                   "track_id": 1,
                                   "class_name": "inner_wall_damage",
                                   "confidence": conf,
                                   "best_confidence": conf,
                                   "best_frame_idx": frame_idx,
                                   "bbox": [1, 1, 20, 20]}]}, f)
        fd = os.path.join(b.dir, "features", "damage")
        os.makedirs(fd, exist_ok=True)
        with open(os.path.join(fd, f"{s.local_id}.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"global_id": s.local_id, "feature": "damage",
                       "status": "OK", "top_damage": "DAMAGE",
                       "top_damage_details": ["inner_wall_damage"],
                       "supporting_cameras": [cam]}, f)
    return b, segs


def _render_local(bundle, out_dir):
    out = os.path.join(out_dir, f"{bundle.camera_id}_report.pdf")
    return adapter.build_local_camera_pdf(
        bundle, output_pdf=out,
        batch_key=f"{bundle.camera_id} (camera-local)",
        fps=15.0, total_frames=3555, verbose=False)


class TestFixtureIsHonest(unittest.TestCase):
    """If the harness cannot tell the tints apart, every test below is empty."""

    def test_tints_are_distinguishable(self):
        self.assertNotEqual(TINT["RIGHT_UP_TOP"], TINT["LEFT_UP_TOP"])
        self.assertGreater(abs(TINT["RIGHT_UP_TOP"] - TINT["LEFT_UP_TOP"]), 20)

    def test_decoder_recovers_a_known_tint(self):
        with tempfile.TemporaryDirectory() as root:
            b, _ = _top_bundle(root, "RIGHT_UP_TOP")
            p = _render_local(b, root)
            means = embedded_image_means(p)
            self.assertTrue(means, "no image decoded -- assertions are vacuous")
            self.assertTrue(_has_tint(means, "RIGHT_UP_TOP"))

    def test_no_images_when_evidence_absent(self):
        """Negative control: the positives must come from the evidence."""
        with tempfile.TemporaryDirectory() as root:
            b, _ = _top_bundle(root, "RIGHT_UP_TOP", with_damage=False)
            # strip the cache too, so nothing at all can be embedded
            import shutil
            shutil.rmtree(os.path.join(b.dir, "camera_cache"),
                          ignore_errors=True)
            p = _render_local(b, root)
            self.assertEqual(embedded_image_means(p), [])


class TestCameraLocalIsolation(unittest.TestCase):
    """Each top camera's own PDF, built from its own bundle."""

    def _render_both(self, root, order):
        pdfs = {}
        for cam in order:
            b, _ = _top_bundle(root, cam)
            pdfs[cam] = _render_local(b, root)
        return pdfs

    def test_each_camera_shows_only_its_own_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            pdfs = self._render_both(root, TOPS)
            for cam in TOPS:
                means = embedded_image_means(pdfs[cam])
                with self.subTest(camera=cam):
                    self.assertTrue(_has_tint(means, cam),
                                    f"{cam} PDF missing its own snapshot")
                    self.assertFalse(
                        _has_tint(means, OTHER_TOP[cam]),
                        f"{cam} PDF contains {OTHER_TOP[cam]} evidence")

    def test_processing_order_does_not_move_evidence(self):
        """Reversing arrival order must not change who owns which snapshot."""
        results = {}
        for order in (TOPS, tuple(reversed(TOPS))):
            with tempfile.TemporaryDirectory() as root:
                pdfs = self._render_both(root, order)
                results[order] = {c: sorted(embedded_image_means(pdfs[c]))
                                  for c in TOPS}
        a, b = results[TOPS], results[tuple(reversed(TOPS))]
        for cam in TOPS:
            with self.subTest(camera=cam):
                self.assertEqual(a[cam], b[cam],
                                 f"{cam} evidence changed with arrival order")

    def test_identical_track_ids_still_resolve_per_camera(self):
        """Both cameras use track_1 -- only camera_id separates them."""
        with tempfile.TemporaryDirectory() as root:
            pdfs = {}
            for cam in TOPS:
                b, _ = _top_bundle(root, cam, track_idx=1)
                pdfs[cam] = _render_local(b, root)
            for cam in TOPS:
                means = embedded_image_means(pdfs[cam])
                with self.subTest(camera=cam):
                    self.assertTrue(_has_tint(means, cam))
                    self.assertFalse(_has_tint(means, OTHER_TOP[cam]))

    def test_identical_scores_and_frame_indices_still_resolve_per_camera(self):
        with tempfile.TemporaryDirectory() as root:
            pdfs = {}
            for cam in TOPS:
                b, _ = _top_bundle(root, cam, track_idx=1, conf=0.5,
                                   frame_idx=777)
                pdfs[cam] = _render_local(b, root)
            for cam in TOPS:
                means = embedded_image_means(pdfs[cam])
                with self.subTest(camera=cam):
                    self.assertTrue(_has_tint(means, cam))
                    self.assertFalse(_has_tint(means, OTHER_TOP[cam]))

    def test_side_cameras_are_isolated_too(self):
        with tempfile.TemporaryDirectory() as root:
            pdfs = {}
            for cam, slot in (("RIGHT_UP", "right_best"),
                              ("LEFT_UP", "left_best")):
                b = CameraEvidenceBundle(root, cam)
                os.makedirs(b.dir, exist_ok=True)
                for st in LIFECYCLE[1:]:
                    b.advance(st)
                segs = [LocalSegment(local_id=local_segment_id(cam, 1),
                                     index=1, start_frame=0, end_frame=59,
                                     start_time=0.0, end_time=4.0,
                                     label="WAGON", confidence=0.9)]
                b.write_segments(segs)
                ed = os.path.join(b.dir, "evidence", segs[0].local_id, "door")
                os.makedirs(ed, exist_ok=True)
                cv2.imwrite(os.path.join(ed, f"{slot}.jpg"), _tile(cam))
                fd = os.path.join(b.dir, "features", "door")
                os.makedirs(fd, exist_ok=True)
                with open(os.path.join(fd, f"{segs[0].local_id}.json"), "w",
                          encoding="utf-8") as f:
                    json.dump({"global_id": segs[0].local_id,
                               "feature": "door", "status": "OK",
                               "left_door": "CLOSED", "right_door": "OPEN",
                               "supporting_cameras": [cam]}, f)
                pdfs[cam] = _render_local(b, root)
            r = embedded_image_means(pdfs["RIGHT_UP"])
            l = embedded_image_means(pdfs["LEFT_UP"])
            self.assertFalse(_has_tint(r, "LEFT_UP"),
                             "RIGHT_UP PDF contains LEFT_UP evidence")
            self.assertFalse(_has_tint(l, "RIGHT_UP"),
                             "LEFT_UP PDF contains RIGHT_UP evidence")


class TestSharedEvidenceRootIsolation(unittest.TestCase):
    """The harder layout: ONE evidence dir holding both cameras' tracks.

    This is what global assembly produces -- `track_1..track_N` numbered across
    both top cameras together, distinguished only by `camera_id` in the
    metadata. It is where a camera-blind resolver does its damage.
    """

    def _fixture(self, root):
        ev = os.path.join(root, "evidence")
        states = os.path.join(root, "wagon_states")
        cache = os.path.join(root, "wagon_cache")
        gw = "GW_1"
        ed = os.path.join(ev, gw, "damage")
        os.makedirs(ed)
        tracks = []
        for i, cam in enumerate(TOPS, start=1):
            cv2.imwrite(os.path.join(ed, f"track_{i}.jpg"), _tile(cam))
            tracks.append({"track_idx": i, "camera_id": cam, "track_id": 1,
                           "class_name": "inner_wall_damage",
                           "confidence": 0.77, "best_confidence": 0.77,
                           "best_frame_idx": 123, "bbox": [1, 1, 20, 20]})
            cd = os.path.join(cache, gw, C.CAMERA_FOLDER[cam])
            os.makedirs(cd, exist_ok=True)
            cv2.imwrite(os.path.join(cd, "frame_000001.jpg"), _tile(cam))
        with open(os.path.join(ed, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"global_id": gw, "feature": "damage",
                       "top_damage": "DAMAGE", "tracks": tracks}, f)
        dd = os.path.join(states, "damage")
        os.makedirs(dd)
        with open(os.path.join(dd, f"{gw}.json"), "w", encoding="utf-8") as f:
            json.dump({"global_id": gw, "feature": "damage", "status": "OK",
                       "top_damage": "DAMAGE",
                       "top_damage_details": ["inner_wall_damage"],
                       "supporting_cameras": list(TOPS)}, f)
        state = GlobalTrainState(
            total_wagons=1,
            wagons=(GlobalWagon(global_id=gw, wagon_index=1,
                                start_frame_master=0, end_frame_master=59,
                                start_time=0.0, end_time=4.0,
                                classification="WAGON",
                                classification_confidence=1.0),),
            master_camera=C.MASTER_CAMERA)
        return ev, states, cache, state, gw

    def test_shared_root_reports_stay_isolated(self):
        from fusion import wagon_state_builder
        from reporting import camera_reports
        with tempfile.TemporaryDirectory() as root:
            ev, states, cache, state, _gw = self._fixture(root)
            unified = wagon_state_builder.build(
                state=state, wagon_states_root=states,
                write_per_wagon_json=False, verbose=False)
            for cam in TOPS:
                out = os.path.join(root, f"{cam}.pdf")
                camera_reports.build_camera_report(
                    camera_id=cam, state=state, unified=unified,
                    evidence_root=ev, wagon_states_root=states,
                    cache_root=cache, per_camera_tracking_path=None,
                    output_pdf=out, batch_key="iso", logo_path=None,
                    verbose=False)
                means = embedded_image_means(out)
                with self.subTest(camera=cam):
                    self.assertTrue(_has_tint(means, cam),
                                    f"{cam} lost its own snapshot")
                    self.assertFalse(
                        _has_tint(means, OTHER_TOP[cam]),
                        f"{cam} report shows {OTHER_TOP[cam]} evidence")

    def test_lookup_helper_filters_by_camera(self):
        from reporting import _evidence_lookup as ev_mod
        with tempfile.TemporaryDirectory() as root:
            ev, _states, _cache, _state, gw = self._fixture(root)
            for cam in TOPS:
                got = ev_mod.damage_track_snapshots(ev, gw, cam)
                with self.subTest(camera=cam):
                    self.assertEqual(len(got), 1)
                    self.assertEqual(got[0][1]["camera_id"], cam)
                    arr = cv2.imread(got[0][0])
                    self.assertAlmostEqual(float(arr.mean()), TINT[cam],
                                           delta=5.0)

    def test_lookup_helper_requires_a_camera(self):
        """Camera identity is part of the lookup, not an optional refinement."""
        import inspect
        from reporting import _evidence_lookup as ev_mod
        p = inspect.signature(ev_mod.damage_track_snapshots).parameters
        self.assertIn("camera_id", p)
        self.assertIs(p["camera_id"].default, inspect.Parameter.empty,
                      "camera_id must be required, not defaulted")


class TestEveryResolverIsCameraScoped(unittest.TestCase):
    """Structural guard: no damage-track resolver may skip the camera filter."""

    RESOLVERS = (
        ("reporting/camera_reports.py", "_camera_damage_tracks"),
        ("reporting/combined_train_report.py", "_top_damage_snapshot"),
        ("reporting/_evidence_lookup.py", "damage_track_snapshots"),
    )

    def test_all_damage_resolvers_filter_on_camera_id(self):
        import ast
        for rel, fn_name in self.RESOLVERS:
            with self.subTest(function=fn_name):
                src = open(os.path.join(V4_ROOT, rel), encoding="utf-8").read()
                fn = next((n for n in ast.walk(ast.parse(src))
                           if isinstance(n, ast.FunctionDef)
                           and n.name == fn_name), None)
                self.assertIsNotNone(fn, f"{fn_name} not found in {rel}")
                body = ast.unparse(fn)
                self.assertIn("camera_id", body,
                              f"{fn_name} resolves damage tracks without "
                              f"consulting camera_id")

    def test_track_filenames_carry_no_camera_identity(self):
        """Documents WHY the metadata filter is mandatory.

        `track_N.jpg` is numbered across both top cameras in one sequence, so
        the filename can never be used to infer ownership. If this ever stops
        being true the resolvers can be simplified -- until then, do not.
        """
        import ast
        src = open(os.path.join(V4_ROOT, "features", "damage", "processor.py"),
                   encoding="utf-8").read()
        self.assertIn('f"track_{i}.jpg"', src)
        # the index comes from enumerating the combined list, not a per-camera one
        self.assertIn("for i, tr in enumerate(all_evidence, start=1)", src)
        self.assertIn("all_evidence.extend(tracks)", src)


if __name__ == "__main__":
    unittest.main()
