"""The processed video must SHOW gaps, the active region, and canonical GW ids.

A renderer that silently draws nothing passes every structural test, so these
decode the written frames and look at the pixels. Both v4 pipelines call the
same `feature_overlay_renderer.render_all_cameras` -- `master_runner:427`
(batch) and `global_assembler:609` (sequential) -- so one renderer covers both
modes, and a test asserts that shared call.

Nothing here runs a detector. Every HUD element is replayed from artifacts
already on disk: gap boxes from the Stage-1 tracking JSON, wagon ids and
boundaries from the canonical roster, the region from the master's own
`wagon_window`, the load verdict from the fused state.

One limitation is asserted rather than hidden: LOAD has no per-frame boxes,
because the load processor persists only `load/best_frame.jpg` and a fused
per-wagon status. Per-frame load boxes would require a second inference pass,
which this module must never do.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "wagon_count"))

import numpy as np                                                # noqa: E402

from core import constants as C                                   # noqa: E402
from core.global_state_loader import parse_global_train_state      # noqa: E402
from rendering import feature_overlay_renderer as R                # noqa: E402

FPS = 10.0
NF = 60          # frames
WH = (320, 240)


def _write_video(path):
    import cv2
    w, _c = R._open_browser_playable_writer(path, FPS, WH[0], WH[1])
    for i in range(NF):
        f = np.full((WH[1], WH[0], 3), 40, np.uint8)
        f[:, :, 0] = (i * 3) % 255            # something that varies
        w.write(f)
    w.release()
    return path


def _state():
    """3 canonical wagons over frames 20..49, region = that span."""
    wagons = []
    for i in range(1, 4):
        sf, ef = 20 + (i - 1) * 10, 20 + i * 10 - 1
        wagons.append({
            "global_id": f"GW_{i}", "wagon_index": i,
            "start_frame_master": sf, "end_frame_master": ef,
            "start_time": sf / FPS, "end_time": (ef + 1) / FPS,
            "classification": C.CLASS_WAGON, "classification_confidence": 0.9,
            "supporting_cameras": list(C.ALL_CAMERAS)})
    return parse_global_train_state({
        "total_wagons": 3, "master_camera": C.CAMERA_RIGHT_UP,
        "master_fps": FPS, "master_total_frames": NF,
        "wagon_window": {"found": True, "wagon_start_time": 2.0,
                         "wagon_end_time": 5.0,
                         "first_wagon_segment_index": 1,
                         "last_wagon_segment_index": 3,
                         "wagon_start_frame": 20, "wagon_end_frame": 49},
        "wagons": wagons,
    })


def _tracking(camera_id):
    """A gap track with the full-fidelity arrays the renderer needs."""
    hits = list(range(25, 36))
    return {camera_id: {
        "fps": FPS, "total_frames": NF, "width": WH[0], "height": WH[1],
        "gaps": [{"track_id": 7, "start_frame": 25, "end_frame": 35,
                  "hit_frames": hits,
                  "bbox_history": [[100.0, 80.0, 160.0, 200.0] for _ in hits]}],
    }}


def _render(camera_id, tmp, *, tracking=None, state=None, unified=None):
    vid = _write_video(os.path.join(tmp, "raw.mp4"))
    out = os.path.join(tmp, f"{camera_id}_processed.mp4")
    tr = tracking if tracking is not None else _tracking(camera_id)
    R._render_one_camera(
        camera_id=camera_id, video_path=vid, output_path=out,
        state=state or _state(), unified=unified or {},
        evidence_root=os.path.join(tmp, "evidence"),
        camera_meta=tr.get(camera_id, {}), verbose=False,
        camera_tracking=tr)
    return out


def _render_with_regions(camera_id, tmp, *, state=None, regions=None,
                         tracking=None):
    """Render one camera, supplying per-camera `LocalWagonRegion` objects."""
    vid = _write_video(os.path.join(tmp, "raw.mp4"))
    out = os.path.join(tmp, f"{camera_id}_processed.mp4")
    tr = tracking if tracking is not None else _tracking(camera_id)
    R._render_one_camera(
        camera_id=camera_id, video_path=vid, output_path=out,
        state=state or _state(), unified={},
        evidence_root=os.path.join(tmp, "evidence"),
        camera_meta=tr.get(camera_id, {}), verbose=False,
        camera_tracking=tr, camera_regions=regions)
    return out


def _frames(path, idxs):
    import cv2
    cap = cv2.VideoCapture(path)
    got, i = {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in idxs:
            got[i] = f
        i += 1
    cap.release()
    return got


def _has_colour(frame, bgr, tol=40):
    """Is this exact-ish colour present? Proof a coloured element was drawn."""
    d = np.abs(frame.astype(int) - np.array(bgr, dtype=int)).sum(axis=2)
    return bool((d <= tol).any())


class TestTheHudIsActuallyDrawn(unittest.TestCase):

    def test_a_gap_box_appears_only_while_the_gap_is_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _render(C.CAMERA_RIGHT_UP, tmp)
            fr = _frames(out, {5, 30, 55})
            self.assertTrue(_has_colour(fr[30], R._GAP_COLOR),
                            "no cyan gap box while the gap was live")
            self.assertFalse(_has_colour(fr[5], R._GAP_COLOR),
                             "a gap box was drawn before the gap started")
            self.assertFalse(_has_colour(fr[55], R._GAP_COLOR),
                             "a gap box was drawn after the gap ended")

    def test_the_boundary_flash_appears_at_a_wagon_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _render(C.CAMERA_RIGHT_UP, tmp)
            fr = _frames(out, {30, 45})
            self.assertTrue(_has_colour(fr[30], R._BOUNDARY_COLOR),
                            "no magenta flash at the GW_2 boundary (frame 30)")
            self.assertFalse(_has_colour(fr[45], R._BOUNDARY_COLOR),
                             "a boundary flash appeared mid-wagon")

    def test_the_active_region_banner_appears_at_both_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _render(C.CAMERA_RIGHT_UP, tmp)
            fr = _frames(out, {20, 35, 49})
            self.assertTrue(_has_colour(fr[20], R._REGION_COLOR),
                            "no START banner at the region start")
            self.assertTrue(_has_colour(fr[49], R._REGION_COLOR),
                            "no END banner at the region end")

    def test_something_is_drawn_on_every_camera_including_the_tops(self):
        for cam in C.ALL_CAMERAS:
            with tempfile.TemporaryDirectory() as tmp:
                out = _render(cam, tmp)
                fr = _frames(out, {30})
                self.assertTrue(_has_colour(fr[30], R._GAP_COLOR),
                                f"{cam} drew no gap box")

    def test_the_output_is_a_real_playable_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _render(C.CAMERA_LEFT_UP, tmp)
            self.assertTrue(os.path.isfile(out))
            self.assertGreater(os.path.getsize(out), 1000)
            fr = _frames(out, set(range(NF)))
            self.assertEqual(len(fr), NF, "frames were dropped")


class TestTheHudReadsTheCanonicalSources(unittest.TestCase):

    def test_a_gap_without_a_trajectory_is_marked_unresolved_not_dropped(self):
        """`GapEvent.to_dict()` drops hit_frames/bbox_history. Such a gap must
        NOT get an invented box -- but it must not vanish either: silently
        omitting it makes the video claim the reconstruction found no boundary
        there, which is a different statement from "the boundary is known, its
        image position is not"."""
        reporting_view = {C.CAMERA_RIGHT_UP: {
            "fps": FPS, "total_frames": NF,
            "gaps": [{"track_id": 7, "start_frame": 25, "end_frame": 35}]}}
        kept = R._gap_tracks_for(reporting_view, C.CAMERA_RIGHT_UP)
        self.assertEqual(len(kept), 1, "the gap was dropped instead of flagged")
        self.assertFalse(kept[0]["_resolved"])
        with tempfile.TemporaryDirectory() as tmp:
            out = _render(C.CAMERA_RIGHT_UP, tmp, tracking=reporting_view)
            fr = _frames(out, {30})
            self.assertFalse(_has_colour(fr[30], R._GAP_COLOR),
                             "a gap box was invented from a reporting view")
            self.assertTrue(_has_colour(fr[30], R._UNRESOLVED_COLOR),
                            "no UNRESOLVED marker was shown for the gap")
        audit = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]
        self.assertEqual(audit["gap_tracks_unresolved"], 1)
        self.assertEqual(audit["gap_markers_drawn"], 0)

    def test_the_region_comes_from_the_master_wagon_window(self):
        st = _state()
        st.wagon_window = dict(st.wagon_window)
        st.wagon_window["found"] = False       # master says: no region
        with tempfile.TemporaryDirectory() as tmp:
            out = _render(C.CAMERA_RIGHT_UP, tmp, state=st)
            self.assertTrue(os.path.isfile(out))

    def test_the_load_verdict_comes_from_the_fused_state(self):
        from core.unified_wagon_state import UnifiedWagonState
        u = {"GW_2": UnifiedWagonState(global_id="GW_2", wagon_index=2,
                                       classification=C.CLASS_WAGON)}
        u["GW_2"].load_status = C.LOAD_LOADED
        with tempfile.TemporaryDirectory() as tmp:
            out = _render(C.CAMERA_RIGHT_UP_TOP, tmp, unified=u)
            self.assertTrue(os.path.isfile(out))

    def test_load_has_no_per_frame_boxes_and_the_module_says_so(self):
        """Stated, not hidden: drawing them would need a second inference pass."""
        src = open(os.path.join(ROOT, "rendering",
                                "feature_overlay_renderer.py"),
                   encoding="utf-8").read()
        self.assertIn("LOAD has no per-frame boxes", src)
        self.assertIn("never persisted per-frame detections", src)

    def test_the_renderer_still_runs_no_model(self):
        """AST, not substring search: the module's own docstring says it never
        invokes YOLO, so a text match finds that disclaimer and fails for the
        wrong reason. What matters is what it IMPORTS and CALLS."""
        import ast
        tree = ast.parse(open(os.path.join(ROOT, "rendering",
                                           "feature_overlay_renderer.py"),
                              encoding="utf-8").read())
        imported, called = set(), set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported.update(a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                imported.add((n.module or "").split(".")[0])
                imported.update(a.name for a in n.names)
            elif isinstance(n, ast.Call):
                f = n.func
                called.add(f.id if isinstance(f, ast.Name)
                           else getattr(f, "attr", ""))
        for forbidden in ("ultralytics", "YOLO"):
            self.assertNotIn(forbidden, imported,
                             f"the renderer imports {forbidden}")
        for forbidden in ("predict", "classify_segments",
                          "load_segment_classifier", "process_video"):
            self.assertNotIn(forbidden, called,
                             f"the renderer calls {forbidden}")

    def test_the_gap_interpolator_is_called_not_copied(self):
        src = open(os.path.join(ROOT, "rendering",
                                "feature_overlay_renderer.py"),
                   encoding="utf-8").read()
        self.assertIn("from video_segmenter import _interp_gap_bbox", src)


class TestBothPipelinesShareThisRenderer(unittest.TestCase):

    def _src(self, *p):
        return open(os.path.join(ROOT, *p), encoding="utf-8").read()

    def test_sequential_and_batch_call_the_same_renderer(self):
        for parts in (("orchestrator", "global_assembler.py"),
                      ("orchestrator", "master_runner.py")):
            src = self._src(*parts)
            self.assertIn("feature_overlay_renderer.render_all_cameras", src,
                          f"{parts} does not use the shared renderer")

    def test_both_pass_the_tracking_json_so_gaps_can_be_drawn(self):
        for parts in (("orchestrator", "global_assembler.py"),
                      ("orchestrator", "master_runner.py")):
            self.assertIn("per_camera_tracking_path", self._src(*parts))


def _state_with_engine():
    """ENGINE (0-19) -> 3 WAGONs (20-49) -> BRAKE_VAN (50-59)."""
    wagons = []
    for i in range(1, 4):
        sf, ef = 20 + (i - 1) * 10, 20 + i * 10 - 1
        wagons.append({
            "global_id": f"GW_{i}", "wagon_index": i,
            "start_frame_master": sf, "end_frame_master": ef,
            "start_time": sf / FPS, "end_time": (ef + 1) / FPS,
            "classification": C.CLASS_WAGON, "classification_confidence": 0.9,
            "supporting_cameras": list(C.ALL_CAMERAS)})
    return parse_global_train_state({
        "total_wagons": 3, "master_camera": C.CAMERA_RIGHT_UP,
        "master_fps": FPS, "master_total_frames": NF,
        "wagon_window": {
            "found": True,
            "wagon_start_time": 2.0, "wagon_end_time": 5.0,
            "wagon_start_frame": 20, "wagon_end_frame": 49,
            "first_wagon_segment_index": 1, "last_wagon_segment_index": 3,
            "leading_non_wagon_objects": [{
                "classification": C.CLASS_ENGINE, "position": "leading",
                "start_frame": 0, "end_frame": 19,
                "start_time": 0.0, "end_time": 2.0}],
            "trailing_non_wagon_objects": [{
                "classification": C.CLASS_BRAKE_VAN, "position": "trailing",
                "start_frame": 50, "end_frame": 59,
                "start_time": 5.0, "end_time": 6.0}],
        },
        "wagons": wagons,
    })


class TestNonWagonFramesNeverShowAGwId(unittest.TestCase):
    """(5) An ENGINE or BRAKE_VAN frame must be identified as such and must
    carry no global wagon id -- that is the region gate, made visible."""

    def test_the_engine_and_brakevan_spans_are_found(self):
        st = _state_with_engine()
        spans = R._non_wagon_spans(st, FPS, NF, 0.0)
        self.assertEqual(
            sorted((a, b, c, d) for a, b, c, d in spans),
            [(0, 19, C.CLASS_ENGINE, "leading"),
             (50, 59, C.CLASS_BRAKE_VAN, "trailing")])

    def test_the_audit_records_them_with_no_global_wagon_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP, tmp, state=_state_with_engine())
        audit = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]
        self.assertEqual(len(audit["non_wagon_objects"]), 2)
        for o in audit["non_wagon_objects"]:
            self.assertIsNone(o["global_wagon_id"],
                              "a non-wagon object was given a GW id")
        self.assertEqual(
            sorted(o["classification"] for o in audit["non_wagon_objects"]),
            [C.CLASS_BRAKE_VAN, C.CLASS_ENGINE])

    def test_a_non_wagon_frame_is_visually_marked(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _render(C.CAMERA_RIGHT_UP, tmp, state=_state_with_engine())
            fr = _frames(out, {5, 55, 35})
            self.assertTrue(_has_colour(fr[5], R._NONWAGON_COLOR),
                            "the leading ENGINE frame carried no label")
            self.assertTrue(_has_colour(fr[55], R._NONWAGON_COLOR),
                            "the trailing BRAKE_VAN frame carried no label")

    def test_only_canonical_wagons_are_listed_as_shown(self):
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP, tmp, state=_state_with_engine())
        audit = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]
        self.assertEqual(audit["global_wagon_ids_shown"],
                         ["GW_1", "GW_2", "GW_3"])
        self.assertEqual(audit["canonical_wagons_total"], 3)


class TestCameraEvidenceNeverCrosses(unittest.TestCase):
    """(7)(8) One camera's evidence must never appear on another's video, and
    the two top cameras must not produce the same frames."""

    def test_each_camera_renders_only_its_own_gap_tracks(self):
        """RIGHT_UP_TOP has a gap; LEFT_UP_TOP has none. LEFT_UP_TOP must draw
        no gap box -- it must not borrow its sibling's."""
        tracking = dict(_tracking(C.CAMERA_RIGHT_UP_TOP))
        tracking[C.CAMERA_LEFT_UP_TOP] = {
            "fps": FPS, "total_frames": NF, "gaps": []}
        with tempfile.TemporaryDirectory() as tmp:
            rut = _render(C.CAMERA_RIGHT_UP_TOP, tmp, tracking=tracking)
            a_rut = dict(R.RENDER_AUDITS[C.CAMERA_RIGHT_UP_TOP])
        with tempfile.TemporaryDirectory() as tmp:
            lut = _render(C.CAMERA_LEFT_UP_TOP, tmp, tracking=tracking)
            a_lut = dict(R.RENDER_AUDITS[C.CAMERA_LEFT_UP_TOP])
        self.assertGreater(a_rut["gap_markers_drawn"], 0)
        self.assertEqual(a_lut["gap_markers_drawn"], 0,
                         "LEFT_UP_TOP drew RIGHT_UP_TOP's gap")

    def test_the_audit_names_the_camera_it_belongs_to(self):
        for cam in C.ALL_CAMERAS:
            with tempfile.TemporaryDirectory() as tmp:
                out = _render(cam, tmp)
            a = R.RENDER_AUDITS[cam]
            self.assertEqual(a["camera_id"], cam)
            self.assertIn(cam, a["output"])

    def test_the_two_top_cameras_produce_distinct_frames(self):
        """They photograph the same roof from opposite sides, so identical
        output is the failure that looks like success."""
        with tempfile.TemporaryDirectory() as tmp:
            t = dict(_tracking(C.CAMERA_RIGHT_UP_TOP))
            t[C.CAMERA_LEFT_UP_TOP] = {"fps": FPS, "total_frames": NF,
                                       "gaps": []}
            rut = _render(C.CAMERA_RIGHT_UP_TOP, tmp, tracking=t)
            os.rename(rut, os.path.join(tmp, "a.mp4"))
            lut = _render(C.CAMERA_LEFT_UP_TOP, tmp, tracking=t)
            fa = _frames(os.path.join(tmp, "a.mp4"), {30})[30]
            fb = _frames(lut, {30})[30]
            self.assertFalse(np.array_equal(fa, fb),
                             "the two top cameras rendered identical frames")


class TestTheRenderAudit(unittest.TestCase):
    """The audit must let every annotation be traced to its source."""

    def test_an_audit_file_is_written_beside_each_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = _render(C.CAMERA_RIGHT_UP, tmp)
            expected = os.path.splitext(out)[0] + "_render_audit.json"
            self.assertTrue(os.path.isfile(expected))
            with open(expected, encoding="utf-8") as f:
                doc = json.load(f)
        for key in ("camera_id", "total_frames", "canonical_wagons_total",
                    "global_wagon_ids_shown", "gap_tracks_total",
                    "gap_markers_drawn", "boundary_frames",
                    "active_region_found", "active_region_start_frame",
                    "active_region_end_frame", "non_wagon_objects",
                    "door_detections_drawn", "damage_detections_drawn",
                    "load_status_frames", "load_per_frame_boxes",
                    "frames_written", "codec"):
            self.assertIn(key, doc, f"the audit omits {key}")

    def test_the_active_region_in_the_audit_matches_the_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP, tmp, state=_state_with_engine())
        a = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]
        self.assertTrue(a["active_region_found"])
        self.assertEqual(a["active_region_start_frame"], 20)
        self.assertEqual(a["active_region_end_frame"], 49)

    def test_unavailable_evidence_is_stated_not_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP_TOP, tmp)
        a = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP_TOP]
        self.assertIn("unavailable", a["load_per_frame_boxes"])
        self.assertIn("second inference pass", a["load_per_frame_boxes"])

    def test_the_counts_are_real_not_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP, tmp)
        a = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]
        self.assertEqual(a["frames_written"], NF)
        self.assertGreater(a["gap_markers_drawn"], 0)
        self.assertEqual(a["gap_tracks_total"], 1)
        self.assertEqual(a["gap_tracks_resolved"], 1)


class TestVideoAndPdfShareTheSameIdentities(unittest.TestCase):
    """(10) The report and the videos must name the same canonical wagons."""

    def test_the_gw_ids_on_the_video_are_the_report_rows(self):
        from reporting import combined_train_report as CTR
        st = _state_with_engine()
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP, tmp, state=st)
        shown = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]["global_wagon_ids_shown"]
        rows, _synth = CTR.canonical_wagons(st, {})
        self.assertEqual(shown, [u.global_id for u in rows])

    def test_no_non_wagon_appears_as_a_report_row_or_a_video_gw(self):
        from reporting import combined_train_report as CTR
        st = _state_with_engine()
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP, tmp, state=st)
        a = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]
        rows, _ = CTR.canonical_wagons(st, {})
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(a["global_wagon_ids_shown"]), 3)
        self.assertEqual(len(a["non_wagon_objects"]), 2)


class TestGapsAreTheRealDetectedGaps(unittest.TestCase):
    """The marker must be the gap's OWN tracked geometry -- not a vertical line
    at an assumed boundary, and not a fixed x for every gap."""

    def _moving_gap(self, camera_id, x0=60.0, drift=6.0):
        """A gap whose recorded trajectory MOVES across the frame."""
        hits = list(range(25, 36))
        hist = [[x0 + drift * i, 80.0, x0 + drift * i + 40.0, 200.0]
                for i, _ in enumerate(hits)]
        return {camera_id: {"fps": FPS, "total_frames": NF,
                            "width": WH[0], "height": WH[1],
                            "gaps": [{"track_id": 3, "start_frame": 25,
                                      "end_frame": 35, "hit_frames": hits,
                                      "bbox_history": hist}]}}

    def test_the_marker_follows_the_recorded_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP, tmp,
                    tracking=self._moving_gap(C.CAMERA_RIGHT_UP))
        prov = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]["gap_marker_provenance_sample"]
        xs = [p["center_x"] for p in sorted(prov, key=lambda p: p["frame"])]
        self.assertGreater(len(xs), 3)
        self.assertGreater(xs[-1], xs[0],
                           "the gap marker did not move with the trajectory")
        self.assertEqual(len(set(xs)), len(xs),
                         "the marker sat at one fixed x for every frame")

    def test_the_geometry_source_is_recorded_per_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP, tmp,
                    tracking=self._moving_gap(C.CAMERA_RIGHT_UP))
        prov = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]["gap_marker_provenance_sample"]
        for p in prov:
            self.assertIn(p["geometry"], ("recorded_hit", "interpolated"))
            self.assertEqual(p["camera_id"], C.CAMERA_RIGHT_UP)
            self.assertEqual(p["local_gap_track_id"], 3)
            self.assertTrue(p["resolved"])

    def test_two_cameras_do_not_share_a_pixel_coordinate(self):
        """Their image planes differ, so copying RIGHT_UP's x onto another
        camera would be wrong even when the global boundary is the same."""
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP, tmp,
                    tracking=self._moving_gap(C.CAMERA_RIGHT_UP, x0=40.0))
            a = [p["center_x"] for p in
                 R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]["gap_marker_provenance_sample"]]
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_LEFT_UP, tmp,
                    tracking=self._moving_gap(C.CAMERA_LEFT_UP, x0=180.0))
            b = [p["center_x"] for p in
                 R.RENDER_AUDITS[C.CAMERA_LEFT_UP]["gap_marker_provenance_sample"]]
        self.assertTrue(a and b)
        self.assertNotEqual(a, b,
                            "two cameras rendered identical gap coordinates")

    def test_the_gap_is_labelled_with_its_adjacent_canonical_wagons(self):
        st = _state_with_engine()          # GW_1..GW_3 at 20-49
        tracking = {C.CAMERA_RIGHT_UP: {
            "fps": FPS, "total_frames": NF,
            "gaps": [{"track_id": 9, "start_frame": 29, "end_frame": 31,
                      "hit_frames": [29, 30, 31],
                      "bbox_history": [[100.0, 80.0, 140.0, 200.0]] * 3}]}}
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP, tmp, state=st, tracking=tracking)
        prov = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]["gap_marker_provenance_sample"]
        labels = {p["label"] for p in prov}
        self.assertTrue(any("GW_1" in l and "GW_2" in l for l in labels),
                        f"the gap was not labelled with its neighbours: {labels}")
        self.assertTrue(any("GAP_9" in l for l in labels))

    def test_no_marker_where_the_gap_is_not_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP, tmp,
                    tracking=self._moving_gap(C.CAMERA_RIGHT_UP))
        prov = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]["gap_marker_provenance_sample"]
        frames = {p["frame"] for p in prov}
        self.assertFalse(frames & {0, 5, 20, 50, 59},
                         "a gap marker was drawn outside the gap's own range")

    def test_the_audit_separates_resolved_from_unresolved(self):
        mixed = {C.CAMERA_RIGHT_UP: {
            "fps": FPS, "total_frames": NF,
            "gaps": [
                {"track_id": 1, "start_frame": 25, "end_frame": 27,
                 "hit_frames": [25, 26, 27],
                 "bbox_history": [[10.0, 10.0, 50.0, 90.0]] * 3},
                {"track_id": 2, "start_frame": 40, "end_frame": 42},
            ]}}
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP, tmp, tracking=mixed)
        a = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]
        self.assertEqual(a["gap_tracks_total"], 2)
        self.assertEqual(a["gap_tracks_resolved"], 1)
        self.assertEqual(a["gap_tracks_unresolved"], 1)
        self.assertEqual([u["local_gap_track_id"] for u in a["unresolved_gaps"]],
                         [2])
        self.assertEqual(a["distinct_gap_tracks_drawn"], ["1"])

    def test_the_active_region_block_has_both_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP, tmp, state=_state_with_engine())
        ar = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]["active_region"]
        self.assertEqual(ar["start"], 20)
        self.assertEqual(ar["end"], 49)
        self.assertEqual(ar["start_reason"], "first_canonical_wagon")
        self.assertEqual(ar["end_reason"], "last_canonical_wagon")

    def test_region_markers_and_gap_markers_are_different_things(self):
        """The active-region edge must not be drawn as, or replace, a gap."""
        st = _state_with_engine()
        tracking = {C.CAMERA_RIGHT_UP: {
            "fps": FPS, "total_frames": NF,
            "gaps": [{"track_id": 4, "start_frame": 29, "end_frame": 31,
                      "hit_frames": [29, 30, 31],
                      "bbox_history": [[100.0, 80.0, 140.0, 200.0]] * 3}]}}
        with tempfile.TemporaryDirectory() as tmp:
            out = _render(C.CAMERA_RIGHT_UP, tmp, state=st, tracking=tracking)
            fr = _frames(out, {20, 30})
        # frame 20 = region START, no gap live -> region colour, no gap colour
        self.assertTrue(_has_colour(fr[20], R._REGION_COLOR))
        self.assertFalse(_has_colour(fr[20], R._GAP_COLOR),
                         "a gap box was drawn at the region boundary")
        # frame 30 = a real gap, not a region edge
        self.assertTrue(_has_colour(fr[30], R._GAP_COLOR))
        prov = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]["gap_marker_provenance_sample"]
        self.assertTrue(all(p["frame"] in (29, 30, 31) for p in prov))


class TestLeftUpTopKeepsTheEngineOutsideTheRegion(unittest.TestCase):
    """The reported failure: LEFT_UP_TOP showed WAGON_REGION_ACTIVE while an
    ENGINE filled the frame.

    Root cause: the region was the MASTER's `wagon_window` projected by
    `time_offset`, and `camera_time_offsets()` returns 0.0 for a camera whose
    offset the counter could not resolve -- deliberately, since guessing a shift
    is worse than assuming none. On a camera whose footage is genuinely
    displaced, the master's region then projects as though the clocks agreed and
    lands over the leading engine.

    Fix: prefer that camera's OWN `LocalWagonRegion` -- built by
    `build_local_wagon_region` from its own temporally-smoothed labels, in its
    own clock. Display only: it selects which frames carry a marker and can
    neither add, remove nor renumber a canonical wagon.
    """

    class _Region:
        """A `LocalWagonRegion` as `wagon_region.json` round-trips it."""

        def __init__(self, found=True, start_frame=None, end_frame=None,
                     reason=""):
            self.found = found
            self.start_frame = start_frame
            self.end_frame = end_frame
            self.reason = reason

    def test_the_engine_stays_outside_when_the_camera_supplies_its_region(self):
        """LEFT_UP_TOP's own wagons run 40..59; frames 0..39 are its engine."""
        st = _state()                       # master region is 20..49
        regions = {C.CAMERA_LEFT_UP_TOP: self._Region(
            start_frame=40, end_frame=59,
            reason="first WAGON at segment 2 on this camera")}
        with tempfile.TemporaryDirectory() as tmp:
            _render_with_regions(C.CAMERA_LEFT_UP_TOP, tmp, state=st,
                                 regions=regions)
        a = R.RENDER_AUDITS[C.CAMERA_LEFT_UP_TOP]["active_region"]
        self.assertEqual(a["start"], 40, "the region still used the master's")
        self.assertEqual(a["end"], 59)
        self.assertEqual(a["source"], f"{C.CAMERA_LEFT_UP_TOP}_local_wagon_region")
        self.assertIn("first WAGON", a["reason"])

    def test_the_start_banner_is_not_drawn_over_the_engine(self):
        """Checked on the BOTTOM STRIP only, where the START/END banner is
        drawn. The region colour also appears in the panel's state label on
        every frame, so a whole-frame colour test cannot tell the two apart --
        it would pass for the wrong reason."""
        st = _state()
        regions = {C.CAMERA_LEFT_UP_TOP: self._Region(start_frame=40,
                                                      end_frame=59)}
        with tempfile.TemporaryDirectory() as tmp:
            out = _render_with_regions(C.CAMERA_LEFT_UP_TOP, tmp, state=st,
                                       regions=regions)
            fr = _frames(out, {10, 40})
        strip = lambda f: f[-45:, :, :]
        self.assertFalse(_has_colour(strip(fr[10]), R._REGION_COLOR),
                         "a START banner was drawn over the engine at frame 10")
        self.assertTrue(_has_colour(strip(fr[40]), R._REGION_COLOR),
                        "no START banner at the camera's own region start")

    def test_a_noisy_wagon_prediction_does_not_open_the_region(self):
        """`found=False` means this camera's own smoothing accepted no wagon
        run, so a single misclassified frame cannot open the region -- the
        fallback is used and the camera's opinion is recorded as the reason."""
        st = _state()
        regions = {C.CAMERA_LEFT_UP_TOP: self._Region(
            found=False, reason="no WAGON segment identified on this camera")}
        with tempfile.TemporaryDirectory() as tmp:
            _render_with_regions(C.CAMERA_LEFT_UP_TOP, tmp, state=st,
                                 regions=regions)
        a = R.RENDER_AUDITS[C.CAMERA_LEFT_UP_TOP]["active_region"]
        self.assertEqual(a["source"], "master_window_projected")
        self.assertIn("no local wagon region", a["reason"])

    def test_the_other_three_cameras_are_unchanged(self):
        """Only LEFT_UP_TOP supplies a region here; the rest must still use the
        projected master window, exactly as before."""
        st = _state()
        regions = {C.CAMERA_LEFT_UP_TOP: self._Region(start_frame=40,
                                                      end_frame=59)}
        for cam in (C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP,
                    C.CAMERA_RIGHT_UP_TOP):
            with tempfile.TemporaryDirectory() as tmp:
                _render_with_regions(cam, tmp, state=st, regions=regions)
            a = R.RENDER_AUDITS[cam]["active_region"]
            self.assertEqual(a["source"], "master_window_projected", cam)
            self.assertEqual(a["start"], 20, cam)
            self.assertEqual(a["end"], 49, cam)

    def test_the_canonical_roster_is_untouched(self):
        """A display fix must not move a single canonical wagon."""
        st = _state()
        before = [(w.global_id, w.wagon_index, w.start_frame_master,
                   w.end_frame_master) for w in st.wagons]
        win_before = dict(st.wagon_window)
        regions = {C.CAMERA_LEFT_UP_TOP: self._Region(start_frame=40,
                                                      end_frame=59)}
        with tempfile.TemporaryDirectory() as tmp:
            _render_with_regions(C.CAMERA_LEFT_UP_TOP, tmp, state=st,
                                 regions=regions)
        after = [(w.global_id, w.wagon_index, w.start_frame_master,
                  w.end_frame_master) for w in st.wagons]
        self.assertEqual(before, after, "the canonical roster changed")
        self.assertEqual(win_before, dict(st.wagon_window),
                         "the canonical wagon_window changed")
        self.assertEqual(len(st.wagons), 3)

    def test_no_camera_region_at_all_behaves_as_before(self):
        st = _state()
        with tempfile.TemporaryDirectory() as tmp:
            _render_with_regions(C.CAMERA_LEFT_UP_TOP, tmp, state=st,
                                 regions=None)
        a = R.RENDER_AUDITS[C.CAMERA_LEFT_UP_TOP]["active_region"]
        self.assertEqual(a["source"], "master_window_projected")

    def test_the_assembler_passes_each_cameras_own_region(self):
        src = open(os.path.join(ROOT, "orchestrator", "global_assembler.py"),
                   encoding="utf-8").read()
        self.assertIn("camera_regions=support_regions", src)


class TestRegionEdgesAreNotPhysicalGaps(unittest.TestCase):
    """An active-region edge marks where the wagon SEQUENCE starts or ends. A
    physical gap is the boundary between two adjacent wagons. Conflating them
    would make the video claim a gap that the detector never found."""

    def test_the_audit_marks_the_region_as_not_a_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP, tmp, state=_state_with_engine())
        a = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]["active_region"]
        self.assertFalse(a["is_physical_wagon_gap"])

    def test_a_drawn_gap_marks_itself_as_a_physical_gap(self):
        tracking = {C.CAMERA_RIGHT_UP: {
            "fps": FPS, "total_frames": NF,
            "gaps": [{"track_id": 9, "start_frame": 29, "end_frame": 31,
                      "hit_frames": [29, 30, 31],
                      "bbox_history": [[100.0, 80.0, 140.0, 200.0]] * 3}]}}
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP, tmp, state=_state_with_engine(),
                    tracking=tracking)
        prov = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]["gap_marker_provenance_sample"]
        self.assertTrue(prov)
        for p in prov:
            self.assertTrue(p["is_physical_wagon_gap"])
            self.assertFalse(p["is_active_region_boundary"])

    def test_a_region_boundary_frame_produces_no_gap_marker(self):
        """Frame 20 is the region START and has no gap live on it."""
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP, tmp, state=_state_with_engine(),
                    tracking={C.CAMERA_RIGHT_UP: {"fps": FPS,
                                                  "total_frames": NF,
                                                  "gaps": []}})
        a = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]
        self.assertEqual(a["gap_markers_drawn"], 0,
                         "a region boundary was counted as a gap")
        self.assertEqual(a["active_region"]["start"], 20)

    def test_the_provenance_names_both_adjacent_wagons(self):
        tracking = {C.CAMERA_RIGHT_UP: {
            "fps": FPS, "total_frames": NF,
            "gaps": [{"track_id": 25, "start_frame": 29, "end_frame": 31,
                      "hit_frames": [29, 30, 31],
                      "bbox_history": [[100.0, 80.0, 140.0, 200.0]] * 3}]}}
        with tempfile.TemporaryDirectory() as tmp:
            _render(C.CAMERA_RIGHT_UP, tmp, state=_state_with_engine(),
                    tracking=tracking)
        prov = R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]["gap_marker_provenance_sample"]
        p = prov[0]
        self.assertEqual(p["left_global_wagon"], "GW_1")
        self.assertEqual(p["right_global_wagon"], "GW_2")
        self.assertIn("GAP_25", p["label"])
        self.assertIn(p["geometry_source"], ("recorded_hit", "interpolated"))
