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
from core.evidence_identity import damage_track_slot
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
        cv2.imwrite(os.path.join(
            ed, f"{damage_track_slot(track_idx, cam)}.jpg"), img)
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
            cv2.imwrite(os.path.join(
                ed, f"{damage_track_slot(i, cam)}.jpg"), _tile(cam))
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

    def test_track_filenames_now_carry_camera_identity(self):
        """The filename itself names the camera, so a collision is impossible.

        `track_idx` alone is unique only within ONE processor invocation -- it
        comes from enumerating the list that was extended once per camera. Two
        invocations writing the same evidence directory would both start at
        track_1 and overwrite each other. The slot therefore carries the camera:
        `track_1__RIGHT_UP_TOP`.

        The metadata camera filter is still mandatory and still applied; this
        makes the on-disk identity unambiguous as well, rather than relying on
        one invocation owning the directory.
        """
        src = open(os.path.join(V4_ROOT, "features", "damage", "processor.py"),
                   encoding="utf-8").read()
        self.assertIn("damage_track_slot(i, tr[\"camera_id\"])", src)
        self.assertNotIn('f"track_{i}.jpg"', src,
                         "the camera-less filename must not come back")
        # the index still comes from the combined list -- unchanged behaviour
        self.assertIn("for i, tr in enumerate(all_evidence, start=1)", src)
        self.assertIn("all_evidence.extend(tracks)", src)

    def test_two_cameras_sharing_an_index_get_different_files(self):
        """The collision that camera-scoped naming rules out."""
        a = damage_track_slot(1, "RIGHT_UP_TOP")
        b = damage_track_slot(1, "LEFT_UP_TOP")
        self.assertNotEqual(a, b)
        from core.evidence_identity import parse_damage_track_slot
        self.assertEqual(parse_damage_track_slot(a), (1, "RIGHT_UP_TOP"))
        self.assertEqual(parse_damage_track_slot("track_1"), (1, None))


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# The COMBINED report's multi-angle grid
# ---------------------------------------------------------------------------

#: A tint used by NOTHING that any panel legitimately owns, so its appearance
#: is proof of a borrow rather than a coincidence.
POISON = 20


def _combined_fixture(root, *, load_source_camera=None, damage_cams=(),
                      right_top_cache=True, load_tint=POISON):
    """One anomalous wagon, four cameras, fully controlled evidence.

    The wagon is anomalous because its RIGHT door is OPEN -- deliberately NOT
    because of damage, so the multi-angle page renders while the top cameras
    have no damage evidence at all.  That is the exact state in which the
    combined report used to duplicate a top-camera view.
    """
    ev = os.path.join(root, "evidence")
    states = os.path.join(root, "wagon_states")
    cache = os.path.join(root, "wagon_cache")
    gw = "GW_1"

    # Per-camera wagon_cache frames: each camera's own fingerprint.
    for cam in ("RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP", "LEFT_UP_TOP"):
        if cam == "RIGHT_UP_TOP" and not right_top_cache:
            continue
        cd = os.path.join(cache, gw, C.CAMERA_FOLDER[cam])
        os.makedirs(cd, exist_ok=True)
        for fi in range(0, 60):
            cv2.imwrite(os.path.join(cd, f"frame_{fi:06d}.jpg"), _tile(cam))

    # Door evidence: slot names carry the camera, so these are safe by design.
    dd = os.path.join(ev, gw, "door")
    os.makedirs(dd)
    cv2.imwrite(os.path.join(dd, "right_best.jpg"), _tile("RIGHT_UP"))
    cv2.imwrite(os.path.join(dd, "left_best.jpg"), _tile("LEFT_UP"))
    with open(os.path.join(dd, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"global_id": gw, "feature": "door", "sides": {}}, f)

    # Load evidence: ONE file per wagon whose owner lives only in metadata.
    if load_source_camera:
        ld = os.path.join(ev, gw, "load")
        os.makedirs(ld)
        cv2.imwrite(os.path.join(ld, "best_frame.jpg"),
                    np.full((40, 40, 3), load_tint, dtype=np.uint8))
        with open(os.path.join(ld, "metadata.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"global_id": gw, "feature": "load",
                       "fused_status": "LOADED", "fused_confidence": 0.9,
                       "source_camera": load_source_camera,
                       "best_frame_idx": 10, "per_camera": {}}, f)

    # Optional damage, per camera, identical in every field but camera_id.
    if damage_cams:
        ed = os.path.join(ev, gw, "damage")
        os.makedirs(ed)
        tracks = []
        for i, cam in enumerate(damage_cams, start=1):
            cv2.imwrite(os.path.join(ed, f"track_{i}.jpg"), _tile(cam))
            tracks.append({"track_idx": i, "camera_id": cam, "track_id": 1,
                           "class_name": "inner_wall_damage",
                           "confidence": 0.77, "best_confidence": 0.77,
                           "best_frame_idx": 123, "bbox": [1, 1, 20, 20]})
        with open(os.path.join(ed, "metadata.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"global_id": gw, "feature": "damage",
                       "top_damage": "DAMAGE", "tracks": tracks}, f)

    # wagon_states: RIGHT door OPEN is the anomaly that renders the page.
    sd = os.path.join(states, "door")
    os.makedirs(sd)
    with open(os.path.join(sd, f"{gw}.json"), "w", encoding="utf-8") as f:
        json.dump({"global_id": gw, "feature": "door", "status": "OK",
                   "right_door": "OPEN", "left_door": "CLOSED",
                   "right_door_confidence": 0.95,
                   "left_door_confidence": 0.9,
                   "supporting_cameras": ["RIGHT_UP", "LEFT_UP"]}, f)
    sl = os.path.join(states, "load")
    os.makedirs(sl)
    with open(os.path.join(sl, f"{gw}.json"), "w", encoding="utf-8") as f:
        json.dump({"global_id": gw, "feature": "load", "status": "OK",
                   "load_status": "LOADED", "load_confidence": 0.9,
                   "supporting_cameras": list(TOPS)}, f)
    if damage_cams:
        sdm = os.path.join(states, "damage")
        os.makedirs(sdm)
        with open(os.path.join(sdm, f"{gw}.json"), "w", encoding="utf-8") as f:
            json.dump({"global_id": gw, "feature": "damage", "status": "OK",
                       "top_damage": "DAMAGE",
                       "top_damage_details": ["inner_wall_damage"],
                       "supporting_cameras": list(damage_cams)}, f)

    state = GlobalTrainState(
        total_wagons=1,
        wagons=(GlobalWagon(global_id=gw, wagon_index=1,
                            start_frame_master=0, end_frame_master=59,
                            start_time=0.0, end_time=4.0,
                            classification="WAGON",
                            classification_confidence=1.0),),
        master_camera=C.MASTER_CAMERA, master_fps=15.0,
        master_total_frames=60)
    return ev, states, cache, state, gw


def _unified_for(state, states_root):
    from fusion import wagon_state_builder
    return wagon_state_builder.build(
        state=state, wagon_states_root=states_root,
        write_per_wagon_json=False, verbose=False)


class TestPanelSnapshotIsCameraScoped(unittest.TestCase):
    """`_panel_snapshot` decides which image a panel shows. Pixels, not paths.

    Every assertion decodes the file the resolver returned and compares its
    mean value to the requesting camera's fingerprint, so a resolver that
    returned a plausibly-named file belonging to the other camera still fails.
    """

    def _resolve(self, cam, **kw):
        """Build a throwaway fixture and return the MEAN PIXEL of what the
        resolver chose. A fresh directory each time -- `_combined_fixture`
        creates its trees with bare `os.makedirs`, so reusing one collides."""
        from reporting import combined_train_report as CTR
        with tempfile.TemporaryDirectory() as root:
            ev, states, cache, state, gw = _combined_fixture(root, **kw)
            unified = _unified_for(state, states)
            path = CTR._panel_snapshot(unified[gw], cam, ev, cache, gw)
            if path is None:
                return None
            img = cv2.imread(path)
            return None if img is None else float(img.mean())

    def _assert_own(self, mean, cam):
        self.assertIsNotNone(mean, f"{cam} got no snapshot at all")
        self.assertAlmostEqual(
            mean, TINT[cam], delta=5.0,
            msg=f"{cam} panel resolved to a mean of {mean}, which is not its "
                f"own tint {TINT[cam]}")

    # ---- the reported bug: NO damage anywhere -------------------------

    def test_no_damage_right_top_does_not_borrow_the_load_frame(self):
        """The exact defect. `load/best_frame.jpg` is owned by LEFT_UP_TOP."""
        self._assert_own(self._resolve("RIGHT_UP_TOP",
                                       load_source_camera="LEFT_UP_TOP"),
                         "RIGHT_UP_TOP")

    def test_no_damage_left_top_keeps_its_own_view(self):
        self._assert_own(self._resolve("LEFT_UP_TOP",
                                       load_source_camera="RIGHT_UP_TOP"),
                         "LEFT_UP_TOP")

    def test_no_damage_the_two_top_panels_differ(self):
        """Both panels showing the same view is the symptom that was reported."""
        with tempfile.TemporaryDirectory() as root:
            from reporting import combined_train_report as CTR
            ev, states, cache, state, gw = _combined_fixture(
                root, load_source_camera="LEFT_UP_TOP")
            unified = _unified_for(state, states)
            means = []
            for cam in TOPS:
                p = CTR._panel_snapshot(unified[gw], cam, ev, cache, gw)
                self.assertIsNotNone(p, f"{cam} got no snapshot")
                means.append(round(float(cv2.imread(p).mean())))
            self.assertNotEqual(means[0], means[1],
                                "both top panels resolved to the same image")

    def test_load_owner_may_use_its_own_load_frame(self):
        """The fix must not throw away legitimately-owned evidence."""
        with tempfile.TemporaryDirectory() as root:
            from reporting import combined_train_report as CTR
            ev, states, cache, state, gw = _combined_fixture(
                root, load_source_camera="RIGHT_UP_TOP")
            unified = _unified_for(state, states)
            p = CTR._panel_snapshot(unified[gw], "RIGHT_UP_TOP", ev, cache, gw)
            self.assertAlmostEqual(float(cv2.imread(p).mean()), POISON,
                                   delta=5.0,
                                   msg="the owning camera lost its load frame")

    def test_unattributed_load_evidence_is_never_claimed(self):
        """No `source_camera` in the metadata means ownership is unproven."""
        with tempfile.TemporaryDirectory() as root:
            from reporting import combined_train_report as CTR
            ev, states, cache, state, gw = _combined_fixture(
                root, load_source_camera="RIGHT_UP_TOP")
            meta = os.path.join(ev, gw, "load", "metadata.json")
            doc = json.load(open(meta, encoding="utf-8"))
            doc.pop("source_camera")
            json.dump(doc, open(meta, "w", encoding="utf-8"))
            unified = _unified_for(state, states)
            for cam in TOPS:
                p = CTR._panel_snapshot(unified[gw], cam, ev, cache, gw)
                with self.subTest(camera=cam):
                    self.assertIsNotNone(p, f"{cam} got no snapshot")
                    self._assert_own(float(cv2.imread(p).mean()), cam)

    # ---- damage present: no borrowing either --------------------------

    def test_damage_on_one_top_camera_only_is_not_borrowed(self):
        self._assert_own(self._resolve("RIGHT_UP_TOP",
                                       damage_cams=("LEFT_UP_TOP",)),
                         "RIGHT_UP_TOP")

    def test_damage_on_own_camera_is_used(self):
        self._assert_own(self._resolve("RIGHT_UP_TOP",
                                       damage_cams=("RIGHT_UP_TOP",)),
                         "RIGHT_UP_TOP")

    def test_damage_on_both_cameras_stays_per_camera(self):
        for cam in TOPS:
            with self.subTest(camera=cam):
                self._assert_own(self._resolve(cam, damage_cams=TOPS), cam)

    # ---- the enumerated collision cases ------------------------------

    def test_identical_track_ids_frames_and_scores(self):
        """Identical in every field a resolver might sort or match on.

        Both records carry tracker `track_id` 1, `best_confidence` 0.77 and
        `best_frame_idx` 123 -- the fixture's defaults -- so `camera_id` is the
        only thing that can separate them.

        Note what is NOT forced identical: `track_idx`, the number in the
        FILENAME. The damage writer enumerates `all_evidence` once across both
        cameras, so two tracks can never share an index; forcing that would
        fabricate a state the writer cannot produce. `track_id` colliding is
        the collision that really happens, because each camera's tracker
        numbers its own tracks from 1.
        """
        with tempfile.TemporaryDirectory() as root:
            from reporting import combined_train_report as CTR
            ev, states, cache, state, gw = _combined_fixture(
                root, damage_cams=TOPS)
            doc = json.load(open(os.path.join(ev, gw, "damage",
                                              "metadata.json"),
                                 encoding="utf-8"))
            ids = [(t["track_id"], t["best_confidence"], t["best_frame_idx"])
                   for t in doc["tracks"]]
            self.assertEqual(ids[0], ids[1],
                             "fixture must make the two records identical")
            self.assertNotEqual(doc["tracks"][0]["track_idx"],
                                doc["tracks"][1]["track_idx"])
            unified = _unified_for(state, states)
            for cam in TOPS:
                p = CTR._panel_snapshot(unified[gw], cam, ev, cache, gw)
                with self.subTest(camera=cam):
                    self._assert_own(float(cv2.imread(p).mean()), cam)

    def test_processing_order_reversal_changes_nothing(self):
        """Which camera was written first must not decide who owns what."""
        self._assert_own(self._resolve("RIGHT_UP_TOP", damage_cams=TOPS),
                         "RIGHT_UP_TOP")
        self._assert_own(
            self._resolve("RIGHT_UP_TOP", damage_cams=tuple(reversed(TOPS))),
            "RIGHT_UP_TOP")

    def test_missing_evidence_uses_only_that_cameras_fallback(self):
        """RIGHT_UP_TOP has no evidence AND no cache -> nothing, not a borrow."""
        self.assertIsNone(
            self._resolve("RIGHT_UP_TOP", load_source_camera="LEFT_UP_TOP",
                          right_top_cache=False),
            "a camera with no evidence of its own must render a placeholder, "
            "never another camera's image")

    def test_side_cameras_never_cross(self):
        for cam in ("RIGHT_UP", "LEFT_UP"):
            with self.subTest(camera=cam):
                self._assert_own(self._resolve(cam), cam)


class TestCombinedPdfPixels(unittest.TestCase):
    """End to end through the real renderer, asserting on embedded pixels."""

    def _build(self, root, **kw):
        from reporting import combined_train_report as CTR
        ev, states, cache, state, gw = _combined_fixture(root, **kw)
        unified = _unified_for(state, states)
        out = CTR.build(
            state=state, unified=unified, output_dir=root,
            batch_key="iso", evidence_root=ev, wagon_states_root=states,
            cache_root=cache, verbose=False)
        self.assertIsNotNone(out.get("pdf_path"), "no PDF was produced")
        return embedded_image_means(out["pdf_path"])

    def test_no_damage_pdf_contains_right_up_tops_own_frame(self):
        """Decisive: under the old fallback RIGHT_UP_TOP's tint was ABSENT.

        With `load/best_frame.jpg` owned by LEFT_UP_TOP, the camera-blind
        lookup made the RIGHT_UP_TOP panel render the load frame, so
        RIGHT_UP_TOP's own tint never reached the page. Its presence is proof
        the panel fell through to its own camera instead of borrowing.
        """
        with tempfile.TemporaryDirectory() as root:
            means = self._build(root, load_source_camera="LEFT_UP_TOP")
            self.assertTrue(
                _has_tint(means, "RIGHT_UP_TOP"),
                f"RIGHT_UP_TOP's own view is missing from the combined "
                f"report; embedded means were {sorted(set(round(m) for m in means))}")

    def test_no_damage_pdf_shows_the_load_owner_its_own_load_frame(self):
        """RIGHT_UP_TOP owns the load frame, so it shows THAT (tint POISON),
        while LEFT_UP_TOP -- which does not own it -- shows its own cache view.
        Two distinct images, each belonging to the panel that renders it."""
        with tempfile.TemporaryDirectory() as root:
            means = self._build(root, load_source_camera="RIGHT_UP_TOP")
            self.assertTrue(any(abs(m - POISON) < 5.0 for m in means),
                            "the load owner lost its own load frame")
            self.assertTrue(_has_tint(means, "LEFT_UP_TOP"),
                            "LEFT_UP_TOP view missing from combined report")

    def test_damage_on_one_camera_does_not_poison_the_other_panel(self):
        with tempfile.TemporaryDirectory() as root:
            means = self._build(root, damage_cams=("LEFT_UP_TOP",))
            self.assertTrue(_has_tint(means, "RIGHT_UP_TOP"),
                            "RIGHT_UP_TOP borrowed instead of showing itself")

    def test_a_borrowed_only_image_never_reaches_the_page(self):
        """POISON is owned by nobody: LEFT_UP_TOP prefers its damage track.

        So the load frame is legitimately used by NO panel, and its tint
        appearing at all can only mean RIGHT_UP_TOP borrowed it.
        """
        with tempfile.TemporaryDirectory() as root:
            means = self._build(root, load_source_camera="LEFT_UP_TOP",
                                damage_cams=("LEFT_UP_TOP",))
            self.assertFalse(
                any(abs(m - POISON) < 5.0 for m in means),
                "an image owned by no panel was embedded -- a borrow occurred")


class TestNoCameraBlindResolverRemains(unittest.TestCase):
    """Structural: the removed fallbacks must not come back."""

    def test_best_damage_snapshot_any_is_gone(self):
        from reporting import combined_train_report as CTR
        self.assertFalse(
            hasattr(CTR, "_best_damage_snapshot_any"),
            "the cross-camera damage fallback was reintroduced")

    def test_panel_snapshot_never_calls_an_unscoped_evidence_lookup(self):
        """Every lookup inside `_panel_snapshot` must be camera-qualified."""
        import ast
        import inspect
        from reporting import combined_train_report as CTR
        src = inspect.getsource(CTR._panel_snapshot)
        tree = ast.parse(src.lstrip())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name != "evidence_snapshot":
                continue
            # The only permitted unscoped use is a door slot, whose NAME
            # carries the camera (`right_best` / `left_best`).
            slots = [a.value for a in node.args
                     if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            self.assertTrue(
                any(s in ("right_best", "left_best") for s in slots),
                f"unscoped evidence_snapshot({slots}) in _panel_snapshot")

    def test_camera_reports_load_lookup_is_scoped(self):
        import inspect
        from reporting import camera_reports
        src = inspect.getsource(camera_reports)
        self.assertNotIn('ev.evidence_snapshot(evidence_root, gw.global_id, "load"',
                         src, "the load snapshot lookup is camera-blind again")
        self.assertIn("evidence_snapshot_for_camera", src)

    def test_the_scoped_helper_demands_a_camera(self):
        from reporting import _evidence_lookup as ev_mod
        with self.assertRaises(ValueError):
            ev_mod.evidence_snapshot_for_camera(None, "GW_1", "load",
                                                "best_frame", "")
