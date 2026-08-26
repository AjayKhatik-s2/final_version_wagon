"""The active region is forward-derived; the gap overlay is real gap tracking.

Two separate things that were being confused, and this file keeps them apart.

ACTIVE_REGION_START / END say where the wagon SEQUENCE begins and ends. They come
from the forward RIGHT_UP master timeline -- the first segment labelled WAGON to
the last -- and from nothing else. An earlier experiment derived that boundary
backwards from the trailing edge; it is withdrawn, and the first class here
proves none of it survives.

GAP_n is the physical space between two adjacent wagons. It comes from the gap
tracks the reconstruction already produced: real frame ranges, real
trajectories, real per-camera image coordinates. A region edge is not a gap and
a gap is not a region edge.
"""

from __future__ import annotations

import ast
import os
import tempfile
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
import rendering.feature_overlay_renderer as R
from test_processed_video_hud import (
    _write_video, _state, _state_with_engine, _frames, _has_colour, FPS, NF, WH,
)

RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP
RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP


# ---------------------------------------------------------------------------
# Fixtures: a real gap track, and the canonical sequence it belongs to
# ---------------------------------------------------------------------------

def moving_gap(track_id=25, sf=25, ef=35, x0=100.0, dx=-6.0):
    """A gap track as Stage 1 records it: hits plus a per-hit bbox history.

    `dx` per frame, so the box genuinely moves -- a marker pinned to one
    x-coordinate would still pass a test built on a stationary fixture.
    """
    hits = list(range(sf, ef + 1))
    return {
        "track_id": track_id, "start_frame": sf, "end_frame": ef,
        "hit_frames": hits,
        "bbox_history": [[x0 + dx * i, 80.0, x0 + dx * i + 40.0, 200.0]
                         for i, _ in enumerate(hits)],
    }


def tracking(camera_id, gaps=None, **kw):
    return {camera_id: {"fps": FPS, "total_frames": NF,
                        "width": WH[0], "height": WH[1],
                        "gaps": [moving_gap()] if gaps is None else gaps,
                        **kw}}


def canonical_gaps(mapping, *, master=RU):
    """`state.global_gaps` as the counting engine emits it.

    `mapping` is `{global_gap_id: {camera: local_track_id}}`. A camera absent
    from an entry never observed that boundary.
    """
    out = []
    for gid, per_cam in sorted(mapping.items()):
        support = {cam: {"camera_id": cam, "local_track_id": tid,
                         "fps": FPS, "span_frames": 10}
                   for cam, tid in per_cam.items() if cam != master}
        out.append({
            "global_gap_id": gid, "master_camera": master,
            "master_track_id": per_cam.get(master),
            "master_time": gid * 1.0,
            "support_observations": support,
        })
    return out


def render_one(camera_id, tmp, *, gaps=None, gg=None, state=None, tr=None):
    """Render one camera. Its own helper rather than the HUD suite's, because
    that one has no `global_gaps` parameter and is shared by other tests."""
    vid = _write_video(os.path.join(tmp, "raw.mp4"))
    tracks = tr if tr is not None else tracking(camera_id, gaps)
    R._render_one_camera(
        camera_id=camera_id, video_path=vid,
        output_path=os.path.join(tmp, f"{camera_id}_processed.mp4"),
        state=state or _state_with_engine(), unified={},
        evidence_root=os.path.join(tmp, "evidence"),
        camera_meta=tracks.get(camera_id, {}), verbose=False,
        camera_tracking=tracks, global_gaps=gg)
    return R.RENDER_AUDITS[camera_id]


def render(camera_id, *, gaps=None, gg=None, state=None):
    with tempfile.TemporaryDirectory() as tmp:
        return render_one(camera_id, tmp, gaps=gaps, gg=gg, state=state)


# ===========================================================================
# 1. The reverse-anchor experiment is gone
# ===========================================================================

class TestReverseAnchorIsRemoved(unittest.TestCase):
    """Withdrawn in full. Not disabled behind a flag, not left dormant --
    removed, so there is no second way for the boundary to be derived."""

    #: Every name the experiment introduced. `tests/_engine_harness.py` is
    #: excluded from the scan below: its allow-list comment explains the revert
    #: and naturally names what was withdrawn, which is documentation, not code.
    WITHDRAWN = (
        "reverse_anchor", "ReverseAnchor", "ReverseAnchorConfig",
        "REVERSE-ANCHOR", "REVERSE_REASON", "reverse_extended",
        "RejectedGapSpan", "rejected_gap_spans", "master_rejected_gap_spans",
        "rejected_gap_spans_from_json", "rejected_gap_spans_from_validation",
        "_merge_evidence", "DEFAULT_REVERSE_CONFIG",
    )

    def _sources(self):
        for root in ("wagon_count", "orchestrator", "core", "rendering",
                     "fusion", "reporting", "materializer", "reconstruction",
                     "delivery", "features", "tests"):
            d = os.path.join(V4_ROOT, root)
            for base, _dirs, files in os.walk(d):
                if "__pycache__" in base:
                    continue
                for f in files:
                    if (f.endswith(".py")
                            and f != os.path.basename(__file__)
                            and f != "_engine_harness.py"):
                        yield os.path.join(base, f)

    def test_no_source_file_mentions_the_experiment(self):
        offenders = []
        for path in self._sources():
            src = open(path, encoding="utf-8").read()
            for name in self.WITHDRAWN:
                if name in src:
                    offenders.append(f"{os.path.relpath(path, V4_ROOT)}:{name}")
        self.assertEqual(offenders, [], f"reverse-anchor remnants: {offenders}")

    def test_the_reverse_test_file_is_gone(self):
        self.assertFalse(os.path.exists(
            os.path.join(V4_ROOT, "tests/test_reverse_anchor_active_region.py")))

    def test_the_window_selector_takes_no_rejected_gap_evidence(self):
        """The withdrawn mechanism fed it the master's DISCARDED gap candidates
        so it could walk backwards. That is gone.

        It does take `first_wagon_index` / `last_wagon_index` from
        `core.region_consensus` -- multi-camera boundaries already normalized
        onto the global timeline. Those select a different one of the master's
        OWN segments; they carry no evidence of their own and cannot widen the
        window to anything the master did not already produce.
        """
        import inspect
        import train_structure as ts
        params = set(inspect.signature(ts.get_master_wagon_window).parameters)
        self.assertEqual(params, {"segments", "verbose",
                                  "first_wagon_index", "last_wagon_index"})
        for banned in ("rejected_gap_spans", "reverse_cfg"):
            self.assertNotIn(banned, params)

    def test_the_wagon_window_carries_no_reverse_fields(self):
        import train_structure as ts
        fields = set(ts.WagonWindow.__dataclass_fields__)
        for banned in ("reverse_anchor", "reverse_extended_objects"):
            self.assertNotIn(banned, fields)

    def test_the_fusion_entry_point_takes_no_evidence_parameter(self):
        import inspect
        import global_fusion as gf
        params = set(inspect.signature(
            gf.assemble_global_train_state_master_fixed).parameters)
        self.assertNotIn("master_rejected_gap_spans", params)


# ===========================================================================
# 2. The forward RIGHT_UP master timeline is the canonical source
# ===========================================================================

class TestForwardMasterTimelineIsCanonical(unittest.TestCase):

    def setUp(self):
        import train_structure as ts
        from global_train_state import GlobalWagon, SegmentClass
        self.ts, self.GW, self.SC = ts, GlobalWagon, SegmentClass

    def _segments(self, spec, unit=60):
        out, f = [], 0
        for i, (label, mult) in enumerate(spec, start=1):
            span = int(round(unit * mult))
            out.append(self.GW(
                global_id=f"SEG_{i}", wagon_index=i,
                start_frame_master=f, end_frame_master=f + span - 1,
                start_time=f / FPS, end_time=(f + span) / FPS,
                classification=label, classification_confidence=0.95))
            f += span
        return out

    def test_the_region_is_first_wagon_label_to_last(self):
        E, W, B = self.SC.ENGINE, self.SC.WAGON, self.SC.BRAKE_VAN
        segs = self._segments([(E, 2.5)] + [(W, 1.0)] * 5 + [(B, 1.2), (E, 2.5)])
        win = self.ts.get_master_wagon_window(segs, verbose=False)
        self.assertEqual(win.first_wagon_segment_index, 1)
        self.assertEqual(win.last_wagon_segment_index, 5)
        self.assertEqual(win.master_wagon_count, 5)

    def test_a_long_leading_engine_never_becomes_a_wagon(self):
        """The failure mode the withdrawn experiment could produce. A locomotive
        is several wagons long, so length alone must not admit it."""
        E, W, B = self.SC.ENGINE, self.SC.WAGON, self.SC.BRAKE_VAN
        for engine_len in (1.0, 2.5, 4.0, 8.0):
            segs = self._segments([(E, engine_len)] + [(W, 1.0)] * 4 + [(B, 1.2)])
            win = self.ts.get_master_wagon_window(segs, verbose=False)
            self.assertEqual(win.master_wagon_count, 4, f"engine={engine_len}")
            self.assertEqual([o.classification
                              for o in win.leading_non_wagon_objects], [E])

    def test_leading_and_trailing_non_wagons_stay_outside(self):
        E, W, B = self.SC.ENGINE, self.SC.WAGON, self.SC.BRAKE_VAN
        segs = self._segments([(E, 2.5)] + [(W, 1.0)] * 3 + [(B, 1.2), (E, 2.0)])
        win = self.ts.get_master_wagon_window(segs, verbose=False)
        self.assertEqual(len(win.leading_non_wagon_objects), 1)
        self.assertEqual(len(win.trailing_non_wagon_objects), 2)
        for o in (win.leading_non_wagon_objects
                  + win.trailing_non_wagon_objects):
            self.assertIn(o.position, ("leading", "trailing"))

    def test_an_interior_non_wagon_label_is_still_counted(self):
        """Classification decides where the region starts and ends, never
        whether an individual wagon exists."""
        E, W, B = self.SC.ENGINE, self.SC.WAGON, self.SC.BRAKE_VAN
        segs = self._segments([(E, 2.5), (W, 1.0), (B, 1.0), (W, 1.0), (B, 1.2)])
        win = self.ts.get_master_wagon_window(segs, verbose=False)
        self.assertEqual(win.master_wagon_count, 3)
        self.assertEqual(len(win.interior_non_wagon_objects), 1)

    def test_the_window_frames_come_from_the_segments_themselves(self):
        E, W, B = self.SC.ENGINE, self.SC.WAGON, self.SC.BRAKE_VAN
        segs = self._segments([(E, 2.5)] + [(W, 1.0)] * 3 + [(B, 1.2)])
        win = self.ts.get_master_wagon_window(segs, verbose=False)
        self.assertEqual(win.wagon_start_frame,
                         win.wagon_units[0].start_frame_master)
        self.assertEqual(win.wagon_end_frame,
                         win.wagon_units[-1].end_frame_master)
        self.assertAlmostEqual(win.wagon_end_time,
                               (win.wagon_end_frame + 1) / FPS)

    def test_the_video_region_marker_follows_the_canonical_window(self):
        a = render(RU)["active_region"]
        self.assertEqual(a["source"], "master_window_projected")
        self.assertEqual(a["start"], 20)
        self.assertEqual(a["end"], 49)


# ===========================================================================
# 3. Gap rendering cannot touch the canonical roster
# ===========================================================================

class TestGapRenderingLeavesTheRosterAlone(unittest.TestCase):

    def test_the_roster_is_identical_before_and_after_rendering(self):
        st = _state_with_engine()
        before = [(w.global_id, w.wagon_index, w.start_frame_master,
                   w.end_frame_master, w.classification) for w in st.wagons]
        win_before = dict(st.wagon_window)
        with tempfile.TemporaryDirectory() as tmp:
            render_one(RU, tmp, state=st, gg=canonical_gaps({25: {RU: 25}}))
        after = [(w.global_id, w.wagon_index, w.start_frame_master,
                  w.end_frame_master, w.classification) for w in st.wagons]
        self.assertEqual(before, after)
        self.assertEqual(win_before, dict(st.wagon_window))

    def test_the_wagon_count_is_unchanged_by_many_gap_tracks(self):
        st = _state_with_engine()
        n = len(st.wagons)
        gaps = [moving_gap(track_id=i, sf=20 + i, ef=24 + i, x0=50.0 * i)
                for i in range(1, 6)]
        with tempfile.TemporaryDirectory() as tmp:
            render_one(RU, tmp, state=st, gaps=gaps)
        self.assertEqual(len(st.wagons), n)

    def test_rendering_with_no_gaps_also_changes_nothing(self):
        st = _state_with_engine()
        ids = [w.global_id for w in st.wagons]
        with tempfile.TemporaryDirectory() as tmp:
            render_one(RU, tmp, state=st, gaps=[])
        self.assertEqual([w.global_id for w in st.wagons], ids)


# ===========================================================================
# 4/5. Real trajectories, and markers that move
# ===========================================================================

class TestGapMarkersFollowTheRealTrajectory(unittest.TestCase):

    def test_the_marker_moves_across_frames(self):
        prov = render(RU)["gap_marker_provenance_sample"]
        xs = [p["center_x"] for p in prov]
        self.assertGreater(len(set(xs)), 1, "the marker never moved")
        self.assertEqual(xs, sorted(xs, reverse=True),
                         "the marker did not follow the track's direction")

    def test_every_marker_matches_the_recorded_bbox(self):
        g = moving_gap()
        prov = render(RU, gaps=[g])["gap_marker_provenance_sample"]
        by_frame = dict(zip(g["hit_frames"], g["bbox_history"]))
        for p in prov:
            if p["geometry_source"] != "recorded_hit":
                continue
            want = by_frame[p["frame"]]
            self.assertEqual(p["bbox"], [round(float(v), 2) for v in want])

    def test_a_frame_between_hits_is_interpolated_not_invented(self):
        """One recorded hit at each end, nothing in between: the middle frames
        must come from the counting engine's own interpolation."""
        g = {"track_id": 25, "start_frame": 25, "end_frame": 35,
             "hit_frames": [25, 35],
             "bbox_history": [[200.0, 80.0, 240.0, 200.0],
                              [100.0, 80.0, 140.0, 200.0]]}
        prov = render(RU, gaps=[g])["gap_marker_provenance_sample"]
        interp = [p for p in prov if p["geometry_source"] == "interpolated"]
        self.assertTrue(interp)
        for p in interp:
            self.assertLess(p["center_x"], 220.0)
            self.assertGreater(p["center_x"], 100.0)

    def test_the_provenance_never_claims_an_assumed_position(self):
        for p in render(RU)["gap_marker_provenance_sample"]:
            self.assertIn(p["geometry_source"], ("recorded_hit", "interpolated"))


# ===========================================================================
# 6. Only inside the gap's valid frame range
# ===========================================================================

class TestGapDrawnOnlyInItsFrameRange(unittest.TestCase):

    def test_no_marker_outside_the_tracks_frame_range(self):
        g = moving_gap(sf=25, ef=30)
        prov = render(RU, gaps=[g])["gap_marker_provenance_sample"]
        self.assertTrue(prov)
        for p in prov:
            self.assertGreaterEqual(p["frame"], 25)
            self.assertLessEqual(p["frame"], 30)

    def test_the_boundary_frames_themselves_are_drawn(self):
        g = moving_gap(sf=25, ef=30)
        frames = {p["frame"] for p in
                  render(RU, gaps=[g])["gap_marker_provenance_sample"]}
        self.assertIn(25, frames)
        self.assertIn(30, frames)

    def test_a_gap_entirely_outside_the_video_draws_nothing(self):
        g = moving_gap(sf=NF + 10, ef=NF + 20)
        self.assertEqual(render(RU, gaps=[g])["gap_markers_drawn"], 0)

    def test_no_gap_tracks_means_no_markers(self):
        self.assertEqual(render(RU, gaps=[])["gap_markers_drawn"], 0)


# ===========================================================================
# 7. Camera-specific coordinates stay isolated
# ===========================================================================

class TestCameraGeometryIsIsolated(unittest.TestCase):
    """The four cameras see the same coupling from four positions. Copying
    RIGHT_UP's pixels into the others would render a plausible box in the wrong
    place -- the kind of error nobody spots by eye."""

    #: The same physical boundary, as each camera actually observed it.
    PER_CAMERA = {RU: (100.0, -6.0), LU: (40.0, +5.0),
                  RUT: (250.0, -8.0), LUT: (20.0, +7.0)}

    def test_each_camera_uses_its_own_trajectory(self):
        seen = {}
        for cam, (x0, dx) in self.PER_CAMERA.items():
            prov = render(cam, gaps=[moving_gap(x0=x0, dx=dx)]
                          )["gap_marker_provenance_sample"]
            self.assertTrue(prov, cam)
            seen[cam] = [p["center_x"] for p in prov]
        for cam, xs in seen.items():
            self.assertEqual(len(set(xs)), len(xs) if len(xs) < 3 else len(set(xs)))
        # No two cameras produced the same pixel series.
        series = [tuple(v) for v in seen.values()]
        self.assertEqual(len(set(series)), len(series), seen)

    def test_a_cameras_audit_only_ever_names_itself(self):
        for cam in self.PER_CAMERA:
            prov = render(cam)["gap_marker_provenance_sample"]
            for p in prov:
                self.assertEqual(p["camera_id"], cam)

    def test_one_cameras_tracking_is_not_read_by_another(self):
        """LEFT_UP is given a gap; RIGHT_UP is given none. RIGHT_UP must draw
        nothing rather than borrowing it."""
        tr = {LU: {"fps": FPS, "total_frames": NF, "gaps": [moving_gap()]},
              RU: {"fps": FPS, "total_frames": NF, "gaps": []}}
        with tempfile.TemporaryDirectory() as tmp:
            render_one(RU, tmp, tr=tr)
        self.assertEqual(R.RENDER_AUDITS[RU]["gap_markers_drawn"], 0)


# ===========================================================================
# 8. A region edge is not a physical gap
# ===========================================================================

class TestRegionEdgeIsNotAGap(unittest.TestCase):

    def test_the_region_audit_says_it_is_not_a_gap(self):
        self.assertFalse(render(RU)["active_region"]["is_physical_wagon_gap"])

    def test_a_region_edge_alone_draws_no_gap_marker(self):
        a = render(RU, gaps=[])
        self.assertEqual(a["gap_markers_drawn"], 0)
        self.assertEqual(a["active_region"]["start"], 20)
        self.assertEqual(a["active_region"]["end"], 49)

    def test_a_real_gap_marks_itself_as_one(self):
        for p in render(RU)["gap_marker_provenance_sample"]:
            self.assertTrue(p["is_physical_wagon_gap"])
            self.assertFalse(p["is_active_region_boundary"])

    def test_a_gap_sitting_on_the_region_start_is_still_a_gap_not_an_edge(self):
        g = moving_gap(sf=19, ef=21)
        prov = render(RU, gaps=[g])["gap_marker_provenance_sample"]
        self.assertTrue(prov)
        for p in prov:
            self.assertTrue(p["is_physical_wagon_gap"])
            self.assertFalse(p["is_active_region_boundary"])


# ===========================================================================
# 9. Unresolved geometry is reported, never fabricated
# ===========================================================================

class TestUnresolvedGeometryIsReported(unittest.TestCase):

    def test_a_track_without_a_trajectory_is_unresolved_not_dropped(self):
        """Omitting it would say "no boundary here", which is a different claim
        from "the boundary is known, its image position is not"."""
        g = {"track_id": 25, "start_frame": 25, "end_frame": 30}
        a = render(RU, gaps=[g])
        self.assertEqual(a["gap_markers_drawn"], 0)
        self.assertEqual(a["gap_tracks_unresolved"], 1)
        self.assertTrue(a["gap_tracks_unresolved_detail"])

    def test_an_unresolved_track_gets_no_fabricated_bbox(self):
        g = {"track_id": 25, "start_frame": 25, "end_frame": 30}
        a = render(RU, gaps=[g])
        self.assertEqual(a["gap_marker_provenance_sample"], [])

    def test_a_canonical_gap_this_camera_never_saw_is_reported(self):
        """Reported, not drawn: it has no local frames on this camera, and the
        only way to place it would be to guess a moment."""
        gg = canonical_gaps({25: {RU: 25, LU: 7}, 26: {RU: 26}})
        a = render(LU, gaps=[moving_gap(track_id=7)], gg=gg)
        self.assertEqual(a["canonical_gaps_not_observed"], [26])
        self.assertEqual(a["canonical_gap_mapping"]["canonical_gaps"], 2)
        self.assertEqual(a["canonical_gap_mapping"]["mapped_to_local_track"], 1)

    def test_a_resolved_and_an_unresolved_track_coexist(self):
        gaps = [moving_gap(track_id=25),
                {"track_id": 26, "start_frame": 40, "end_frame": 45}]
        a = render(RU, gaps=gaps)
        self.assertEqual(a["gap_tracks_resolved"], 1)
        self.assertEqual(a["gap_tracks_unresolved"], 1)
        self.assertGreater(a["gap_markers_drawn"], 0)


# ===========================================================================
# 10. Canonical identity, and the same behaviour in both modes
# ===========================================================================

class TestCanonicalGapIdentity(unittest.TestCase):
    """`GAP_25` has to mean the same coupling on every camera. A local track id
    is a per-camera counter and does not."""

    def test_the_label_uses_the_canonical_id_not_the_local_track_id(self):
        gg = canonical_gaps({25: {RU: 3, LU: 91}})
        prov = render(LU, gaps=[moving_gap(track_id=91)], gg=gg
                      )["gap_marker_provenance_sample"]
        self.assertTrue(prov)
        for p in prov:
            self.assertIn("GAP_25", p["label"])
            self.assertNotIn("GAP_91", p["label"])
            self.assertEqual(p["local_gap_track_id"], 91)

    def test_two_cameras_label_one_boundary_identically(self):
        gg = canonical_gaps({25: {RU: 3, LU: 91}})
        a = render(RU, gaps=[moving_gap(track_id=3)], gg=gg)
        b = render(LU, gaps=[moving_gap(track_id=91, x0=40.0, dx=5.0)], gg=gg)
        la = a["gap_marker_provenance_sample"][0]["label"]
        lb = b["gap_marker_provenance_sample"][0]["label"]
        self.assertEqual(la, lb)
        self.assertIn("GAP_25", la)

    def test_without_a_canonical_mapping_it_falls_back_to_the_local_id(self):
        prov = render(RU, gaps=[moving_gap(track_id=25)], gg=None
                      )["gap_marker_provenance_sample"]
        self.assertIn("GAP_25", prov[0]["label"])

    def test_the_label_names_both_adjacent_wagons(self):
        gg = canonical_gaps({25: {RU: 25}})
        g = moving_gap(track_id=25, sf=29, ef=31)
        p = render(RU, gaps=[g], gg=gg)["gap_marker_provenance_sample"][0]
        self.assertEqual(p["left_global_wagon"], "GW_1")
        self.assertEqual(p["right_global_wagon"], "GW_2")
        self.assertEqual(p["label"], "GW_1 | GAP_25 | GW_2")


class TestBothModesRenderTheSameWay(unittest.TestCase):

    @staticmethod
    def _kwargs(path):
        """Keyword names passed to `render_all_cameras` in one pipeline."""
        src = open(os.path.join(V4_ROOT, path), encoding="utf-8").read()
        for n in ast.walk(ast.parse(src)):
            if (isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "render_all_cameras"):
                return {kw.arg for kw in n.keywords}
        return set()

    def test_both_modes_pass_the_canonical_gaps(self):
        for path in ("orchestrator/global_assembler.py",
                     "orchestrator/master_runner.py"):
            self.assertIn("global_gaps", self._kwargs(path), path)

    #: Everything the gap overlay and the active region are derived from.
    GAP_AND_TIMELINE_ARGS = {"global_gaps", "state", "per_camera_tracking_path",
                             "camera_offsets", "video_paths"}

    def test_both_modes_agree_on_the_gap_and_timeline_arguments(self):
        """Only these decide the gap markers and the region, so only these have
        to match. Asserting the FULL argument sets equal would be a stricter
        claim than the behaviour needs, and it would fail for reasons unrelated
        to gaps or the timeline."""
        seq = self._kwargs("orchestrator/global_assembler.py")
        batch = self._kwargs("orchestrator/master_runner.py")
        self.assertTrue(seq and batch)
        for arg in self.GAP_AND_TIMELINE_ARGS:
            self.assertIn(arg, seq, f"sequential is missing {arg}")
            self.assertIn(arg, batch, f"batch is missing {arg}")

    def test_the_only_differences_are_the_two_known_ones(self):
        """Pinned deliberately, so a THIRD divergence has to be looked at.

        `camera_regions` is sequential-only and correct: batch has no persisted
        per-camera region to restore.

        `enabled_features` is batch-only and is NOT correct -- sequential lets
        it default to None, so the renderer treats every feature as enabled and
        would draw overlays for a feature the operator switched off. That is a
        real defect, but it belongs to feature enablement rather than to gaps or
        the timeline, so it is recorded here instead of being changed under this
        task.
        """
        seq = self._kwargs("orchestrator/global_assembler.py")
        batch = self._kwargs("orchestrator/master_runner.py")
        self.assertEqual(seq - batch, {"camera_regions"})
        self.assertEqual(batch - seq, {"enabled_features"})

    def test_the_renderer_runs_no_inference(self):
        """AST, not text: the module docstring DESCRIBES the legacy YOLO
        annotations it reproduces, so a substring search matches prose and the
        assertion would be about a comment rather than about code."""
        src = open(os.path.join(V4_ROOT,
                                "rendering/feature_overlay_renderer.py"),
                   encoding="utf-8").read()
        tree = ast.parse(src)
        called = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                f = n.func
                called.add(f.attr if isinstance(f, ast.Attribute)
                           else getattr(f, "id", ""))
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported.update(a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                imported.add((n.module or "").split(".")[0])
                imported.update(a.name for a in n.names)
        for banned in ("YOLO", "load_yolo", "GapTracker", "predict",
                       "process_video", "run_detection"):
            self.assertNotIn(banned, called, f"{banned} is CALLED")
            self.assertNotIn(banned, imported, f"{banned} is IMPORTED")
        for banned in ("ultralytics", "torch"):
            self.assertNotIn(banned, imported)

    def test_stage3_sampling_is_untouched(self):
        self.assertEqual((C.STAGE3_DOOR_MODE, C.STAGE3_DOOR_STRIDE),
                         ("sampled", 3))
        self.assertEqual((C.STAGE3_DAMAGE_MODE, C.STAGE3_DAMAGE_STRIDE),
                         ("sampled", 3))
        self.assertEqual((C.STAGE3_LOAD_MODE, C.STAGE3_LOAD_STRIDE),
                         ("sampled", 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ===========================================================================
# 11. Adjacency on REAL spans, not just coincident ones
# ===========================================================================

class TestAdjacencyMatchesRealGapSpans(unittest.TestCase):
    """A regression from real footage.

    A real gap is visible for a dozen-odd frames as it crosses the view, while
    the roster's boundary between two wagons is a single frame inside that span.
    The lookup used to require the gap's FIRST or LAST frame to equal a boundary
    frame; on real data neither ever did, so every label fell back to a bare
    `GAP_n` with no wagons named. Synthetic fixtures happened to line the two up,
    which is exactly why only the rendered video showed it.
    """

    #: RIGHT_UP, real numbers: GW_1 ends 255, GW_2 starts 256, GAP_2 spans
    #: 249..262 -- straddling the boundary, coincident with neither end.
    BOUNDARY = {255: ("GW_1", "GW_2"), 256: ("GW_1", "GW_2")}

    def test_a_straddling_gap_names_both_wagons(self):
        gap = {"track_id": 2, "start_frame": 249, "end_frame": 262}
        self.assertEqual(R._gap_neighbour_pair(self.BOUNDARY, gap),
                         ("GW_1", "GW_2"))

    def test_the_label_is_the_full_triple(self):
        gap = {"track_id": 2, "global_gap_id": 2,
               "start_frame": 249, "end_frame": 262}
        self.assertEqual(R._gap_neighbours(self.BOUNDARY, gap),
                         "GW_1 | GAP_2 | GW_2")

    def test_a_gap_nowhere_near_a_boundary_names_nobody(self):
        gap = {"track_id": 9, "start_frame": 600, "end_frame": 612}
        self.assertEqual(R._gap_neighbour_pair(self.BOUNDARY, gap), ("", ""))
        self.assertEqual(R._gap_neighbours(self.BOUNDARY, gap), "GAP_9")

    def test_the_nearest_boundary_to_the_centre_wins(self):
        """An over-long track brushing two boundaries belongs to the one it is
        centred on, not to whichever was scanned first."""
        bmap = {100: ("GW_1", "GW_2"), 400: ("GW_2", "GW_3")}
        # Centres chosen well clear of the midpoint: a span centred at exactly
        # 250 is equidistant from both and the answer would be arbitrary, so
        # asserting one would be testing dict iteration order.
        gap = {"track_id": 5, "start_frame": 200, "end_frame": 410}   # c=305
        self.assertEqual(R._gap_neighbour_pair(bmap, gap), ("GW_2", "GW_3"))
        gap = {"track_id": 5, "start_frame": 90, "end_frame": 300}    # c=195
        self.assertEqual(R._gap_neighbour_pair(bmap, gap), ("GW_1", "GW_2"))

    def test_the_exactly_coincident_case_still_works(self):
        gap = {"track_id": 2, "start_frame": 255, "end_frame": 255}
        self.assertEqual(R._gap_neighbour_pair(self.BOUNDARY, gap),
                         ("GW_1", "GW_2"))

    def test_a_gap_with_no_span_falls_back_to_the_old_exact_match(self):
        gap = {"track_id": 2, "start_frame": 255, "end_frame": None}
        self.assertEqual(R._gap_neighbour_pair(self.BOUNDARY, gap),
                         ("GW_1", "GW_2"))

    def test_the_rendered_label_names_both_wagons_on_a_real_span(self):
        """End to end: a gap straddling the GW_1/GW_2 boundary, rendered."""
        g = moving_gap(track_id=2, sf=27, ef=33)
        prov = render(RU, gaps=[g], gg=canonical_gaps({2: {RU: 2}})
                      )["gap_marker_provenance_sample"]
        self.assertTrue(prov)
        for p in prov:
            self.assertEqual(p["label"], "GW_1 | GAP_2 | GW_2")


class TestGapLabelStaysReadable(unittest.TestCase):
    """The label is drawn below the box when the box starts near the top of the
    frame. A gap crossing a wagon boundary is exactly when the magenta
    GW_BOUNDARY banner is also drawn, so the label that matters most was the one
    being overprinted."""

    def test_a_high_box_puts_its_label_underneath(self):
        self.assertGreater(R._LABEL_TOP_MARGIN, 0)
        g = moving_gap(sf=28, ef=32)
        g["bbox_history"] = [[100.0, 4.0, 140.0, 120.0]
                             for _ in g["hit_frames"]]
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, f"{RU}_processed.mp4")
            render_one(RU, tmp, gaps=[g])
        # Drawn at all, and the audit still records the box we asked for.
        prov = R.RENDER_AUDITS[RU]["gap_marker_provenance_sample"]
        self.assertTrue(prov)
        self.assertEqual(prov[0]["bbox"][1], 4.0)

    def test_a_low_box_keeps_its_label_above(self):
        g = moving_gap(sf=28, ef=32)
        g["bbox_history"] = [[100.0, 150.0, 140.0, 230.0]
                             for _ in g["hit_frames"]]
        a = render(RU, gaps=[g])
        self.assertGreater(a["gap_markers_drawn"], 0)
