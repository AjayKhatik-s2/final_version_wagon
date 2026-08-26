"""Only the SIDE cameras may define the canonical global wagon timeline.

The concern this pins: the top-camera vehicle classifier calls an ENGINE or a
BRAKE_VAN a WAGON. If a top camera could define structure, that single
misclassification would extend the wagon-active region over the locomotive and
add a `GW_n` that is not a wagon -- and because `total_wagons` counts WAGON
units, it would change the count of the train.

So the rule is: RIGHT_UP is the master timeline, LEFT_UP corroborates it, and the
two top cameras contribute evidence, features, frames and overlays while having
NO write access to global identity. These tests assert that as a property of the
code, not as a convention -- each one describes a misclassification a top camera
realistically makes and shows the canonical roster does not move.

Where the authority actually lives, established by reading the pipeline:

    `initial_classifications` <- the MASTER camera alone, via
        `_classify_master_pre_fusion(master, side_classification.pt)` in batch
        and `_load_master_classifications(bundles[master_camera])` in sequential.
    `build_global_wagons(master.gaps, ..., initial_classifications)`
        -> segments from the master's own validated gaps
    `get_master_wagon_window(segments)`
        -> first WAGON..last WAGON, renumbered GW_1..GW_N

No top-camera classification appears anywhere on that path. Support cameras
reach `attach_support_evidence`, which "cannot add, remove or reorder a
GlobalGap", and `core.vehicle_type`, which lists them as
NON_AUTHORITATIVE_CAMERAS.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import tempfile
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from core import active_region as AR
from core import vehicle_type as VT
from core.global_state_loader import (GlobalTrainState, GlobalWagon,
                                      roster_fingerprint)

import train_structure as ts
import global_fusion as gf
from global_train_state import SegmentClass, GlobalWagon as EngineWagon

RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP
RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP

FPS = 15.0
UNIT = 60                      # one wagon = 60 frames = 4 s
W, E, B = SegmentClass.WAGON, SegmentClass.ENGINE, SegmentClass.BRAKE_VAN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def segments(spec):
    """Master segments as `build_global_wagons` produces them: gap-delimited,
    inclusive frames, `end_time == (end_frame + 1) / fps`."""
    out, f = [], 0
    for i, (label, mult) in enumerate(spec, start=1):
        span = int(round(UNIT * mult))
        out.append(EngineWagon(
            global_id=f"SEG_{i}", wagon_index=i,
            start_frame_master=f, end_frame_master=f + span - 1,
            start_time=f / FPS, end_time=(f + span) / FPS,
            classification=label, classification_confidence=0.95))
        f += span
    return out


#: ENGINE, five WAGONs, BRAKE_VAN, ENGINE -- the ordinary train.
SPEC = [(E, 2.5)] + [(W, 1.0)] * 5 + [(B, 1.2), (E, 2.5)]


def canonical_window(spec=None):
    return ts.get_master_wagon_window(segments(spec or SPEC), verbose=False)


class Cls:
    """One classification record, in a camera's own local clock."""

    def __init__(self, idx, sf, ef, label, conf=0.9):
        self.segment_index, self.start_frame, self.end_frame = idx, sf, ef
        self.label, self.confidence = label, conf


def top_says_wagon_everywhere(n_frames=int(9.2 * UNIT)):
    """The failure mode: a top camera calling WAGON across the whole train,
    locomotive and brake van included."""
    return [Cls(0, 0, n_frames - 1, W, 0.93)]


def state_from(win, *, master=RU):
    """A `core` GlobalTrainState carrying this canonical window."""
    wagons = tuple(
        GlobalWagon(
            global_id=u.global_id, wagon_index=u.wagon_index,
            start_frame_master=u.start_frame_master,
            end_frame_master=u.end_frame_master,
            start_time=u.start_time, end_time=u.end_time,
            classification=u.classification,
            classification_confidence=u.classification_confidence)
        for u in win.wagon_units)
    return GlobalTrainState(
        total_wagons=len(wagons), wagons=wagons, master_camera=master,
        master_fps=FPS, master_total_frames=int(9.2 * UNIT),
        wagon_window=win.summary(),
        camera_offsets={c: {"status": "RESOLVED", "delta": 0.0}
                        for c in C.ALL_CAMERAS})


def resolve(state, tops=None, fps=None):
    return AR.resolve(state, top_classifications=tops or {},
                      camera_fps=fps or {c: FPS for c in C.ALL_CAMERAS},
                      camera_offsets={c: 0.0 for c in C.ALL_CAMERAS},
                      verbose=False)


# ===========================================================================
# 1. A top camera cannot create a GW id
# ===========================================================================

class TestTopCameraCannotCreateAWagon(unittest.TestCase):

    def test_the_window_selector_sees_only_master_segments(self):
        """Its entire input is the master's classified segments. There is no
        parameter through which a top camera's opinion could arrive."""
        params = list(inspect.signature(ts.get_master_wagon_window).parameters)
        self.assertEqual(params, ["segments", "verbose"])

    def test_the_roster_is_the_same_however_loudly_a_top_camera_disagrees(self):
        win = canonical_window()
        st = state_from(win)
        before = roster_fingerprint(st)
        resolve(st, {RUT: top_says_wagon_everywhere(),
                     LUT: top_says_wagon_everywhere()})
        self.assertEqual(roster_fingerprint(st), before)
        self.assertEqual(st.total_wagons, 5)

    def test_no_gw_id_is_minted_outside_the_canonical_roster(self):
        win = canonical_window()
        st = state_from(win)
        res = resolve(st, {RUT: top_says_wagon_everywhere()})
        self.assertEqual(res.eligible_global_ids,
                         [f"GW_{i}" for i in range(1, 6)])

    def test_the_audit_states_the_authority_explicitly(self):
        res = resolve(state_from(canonical_window()))
        a = res.to_dict()["structure_authority"]
        self.assertEqual(a["timeline_master"], RU)
        self.assertEqual(a["side_support"], [LU])
        self.assertEqual(sorted(a["non_authoritative_cameras"]),
                         sorted(C.TOP_CAMERAS))
        self.assertTrue(a["top_camera_classification_is_read_only"])


# ===========================================================================
# 2/3. ENGINE->WAGON and BRAKE_VAN->WAGON misclassification
# ===========================================================================

class TestTopMisclassificationDoesNotChangeTheCount(unittest.TestCase):

    #: The locomotive occupies the first 2.5 wagons of the train.
    ENGINE_RUN = [Cls(0, 0, int(2.5 * UNIT) - 1, W, 0.94)]
    #: The brake van and trailing engine occupy the last 3.7.
    TAIL_RUN = [Cls(0, int(5.5 * UNIT), int(9.2 * UNIT) - 1, W, 0.94)]

    def test_engine_called_wagon_does_not_add_a_wagon(self):
        st = state_from(canonical_window())
        for cam in C.TOP_CAMERAS:
            res = resolve(st, {cam: self.ENGINE_RUN})
            self.assertEqual(st.total_wagons, 5, cam)
            self.assertEqual(len(res.eligible_global_ids), 5, cam)

    def test_brake_van_called_wagon_does_not_add_a_wagon(self):
        st = state_from(canonical_window())
        for cam in C.TOP_CAMERAS:
            res = resolve(st, {cam: self.TAIL_RUN})
            self.assertEqual(st.total_wagons, 5, cam)
            self.assertEqual(len(res.eligible_global_ids), 5, cam)

    def test_the_engine_misclassification_is_recorded_as_ignored(self):
        res = resolve(state_from(canonical_window()), {RUT: self.ENGINE_RUN})
        ignored = [p for p in res.ignored_predictions if p["camera"] == RUT]
        self.assertTrue(ignored, "the misclassification was not recorded")
        self.assertEqual(ignored[0]["reason"], AR.REASON_NOT_OPEN)
        self.assertEqual(res.to_dict()["structure_authority"]
                         ["would_have_moved_start"], 1)

    def test_the_tail_misclassification_is_recorded_as_ignored(self):
        res = resolve(state_from(canonical_window()), {LUT: self.TAIL_RUN})
        ignored = [p for p in res.ignored_predictions if p["camera"] == LUT]
        self.assertTrue(ignored)
        self.assertEqual(ignored[0]["reason"], AR.REASON_CLOSED)
        self.assertEqual(res.to_dict()["structure_authority"]
                         ["would_have_moved_end"], 1)

    def test_both_ends_at_once_are_both_ignored(self):
        res = resolve(state_from(canonical_window()),
                      {RUT: top_says_wagon_everywhere()})
        a = res.to_dict()["structure_authority"]
        self.assertEqual(a["would_have_moved_start"], 1)
        self.assertEqual(a["would_have_moved_end"], 1)
        self.assertEqual(a["per_top_camera"][RUT]["total_ignored"], 2)

    def test_agreement_reports_zero_rather_than_silence(self):
        """Zero has to mean "checked and agreed", not "never looked"."""
        agree = [Cls(0, int(2.5 * UNIT), int(7.5 * UNIT) - 1, W, 0.95)]
        res = resolve(state_from(canonical_window()), {RUT: agree})
        a = res.to_dict()["structure_authority"]
        self.assertEqual(a["top_predictions_ignored_total"], 0)
        self.assertIn(RUT, res.start.corroborated_by + res.end.corroborated_by)


# ===========================================================================
# 4. Top predictions cannot move START or END
# ===========================================================================

class TestTopCameraCannotMoveTheBoundaries(unittest.TestCase):

    def test_the_boundaries_are_identical_with_and_without_top_evidence(self):
        st = state_from(canonical_window())
        bare = resolve(st)
        loud = resolve(st, {RUT: top_says_wagon_everywhere(),
                            LUT: top_says_wagon_everywhere()})
        self.assertEqual((bare.start.frame, bare.end.frame),
                         (loud.start.frame, loud.end.frame))

    def test_the_start_is_the_first_canonical_wagon(self):
        win = canonical_window()
        res = resolve(state_from(win), {RUT: top_says_wagon_everywhere()})
        self.assertEqual(res.start.frame, win.wagon_units[0].start_frame_master)

    def test_the_end_is_the_last_canonical_wagon(self):
        win = canonical_window()
        res = resolve(state_from(win), {LUT: top_says_wagon_everywhere()})
        self.assertEqual(res.end.frame, win.wagon_units[-1].end_frame_master)

    def test_a_dissenting_top_camera_is_recorded_as_not_applied(self):
        far = [Cls(0, 0, int(1.0 * UNIT), W, 0.9)]
        res = resolve(state_from(canonical_window()), {RUT: far})
        d = res.start.dissent.get(RUT) or res.end.dissent.get(RUT)
        if d is not None:
            self.assertFalse(d["applied"])

    def test_the_boundary_source_is_always_the_master(self):
        res = resolve(state_from(canonical_window()),
                      {RUT: top_says_wagon_everywhere()})
        self.assertEqual(res.start.source_camera, RU)
        self.assertEqual(res.end.source_camera, RU)

    def test_the_module_never_writes_to_the_state(self):
        """AST: no assignment to any attribute of `state` anywhere in it."""
        src = open(os.path.join(V4_ROOT, "core/active_region.py"),
                   encoding="utf-8").read()
        writes = []
        for n in ast.walk(ast.parse(src)):
            if not isinstance(n, (ast.Assign, ast.AugAssign)):
                continue
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if (isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "state"):
                    writes.append(t.attr)
        self.assertEqual(writes, [], f"active_region writes state.{writes}")


# ===========================================================================
# 5/6. RIGHT_UP is master; LEFT_UP is side support
# ===========================================================================

class TestCameraRoles(unittest.TestCase):

    def test_right_up_is_the_master_constant(self):
        self.assertEqual(C.MASTER_CAMERA, RU)
        self.assertEqual(gf.MASTER_CAMERA, RU)

    def test_the_side_authority_is_the_two_side_cameras(self):
        self.assertEqual(VT.SIDE_AUTHORITY, (RU, LU))
        self.assertEqual(C.SIDE_CAMERAS, (RU, LU))

    def test_the_top_cameras_are_declared_non_authoritative(self):
        self.assertEqual(tuple(VT.NON_AUTHORITATIVE_CAMERAS),
                         tuple(C.TOP_CAMERAS))
        self.assertNotIn(RUT, VT.SIDE_AUTHORITY)
        self.assertNotIn(LUT, VT.SIDE_AUTHORITY)

    def test_left_up_corroborates_but_is_not_the_master(self):
        res = resolve(state_from(canonical_window()))
        a = res.to_dict()["structure_authority"]
        self.assertEqual(a["timeline_master"], RU)
        self.assertIn(LU, a["side_support"])
        self.assertNotIn(LU, a["non_authoritative_cameras"])

    def test_the_master_classification_uses_the_side_model(self):
        self.assertEqual(ts.classification_model_for(RU),
                         ts.SIDE_CLASSIFICATION_MODEL)
        self.assertEqual(ts.classification_model_for(LU),
                         ts.SIDE_CLASSIFICATION_MODEL)

    def test_the_top_cameras_use_their_own_models_not_the_side_one(self):
        for cam in C.TOP_CAMERAS:
            self.assertNotEqual(ts.classification_model_for(cam),
                                ts.SIDE_CLASSIFICATION_MODEL)

    def test_support_evidence_cannot_reorder_the_gap_sequence(self):
        doc = inspect.getdoc(gf.attach_support_evidence) or ""
        self.assertIn("cannot add, remove or reorder", doc)
        # Checked on the RECEIVER, not on the method name: the function does
        # call `.append` -- on its own diagnostic lists -- and a bare method-name
        # scan would flag those and prove nothing about the gap sequence.
        tree = ast.parse(inspect.getsource(gf.attach_support_evidence))
        mutations = []
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)):
                continue
            recv = n.func.value
            if (isinstance(recv, ast.Name) and recv.id == "global_gaps"
                    and n.func.attr in ("append", "insert", "remove", "pop",
                                        "sort", "reverse", "extend", "clear")):
                mutations.append(n.func.attr)
        self.assertEqual(mutations, [],
                         f"global_gaps structurally mutated: {mutations}")

    def test_support_evidence_only_writes_the_evidence_fields(self):
        """What it MAY write on a gap: support_observations, missing_cameras,
        unavailable_cameras, time_residuals, weighted_time, flags. Anything
        else would be a support camera reaching into the canonical sequence."""
        allowed = {"support_observations", "missing_cameras",
                   "unavailable_cameras", "time_residuals", "weighted_time",
                   "flags"}
        tree = ast.parse(inspect.getsource(gf.attach_support_evidence))
        written = set()
        for n in ast.walk(tree):
            targets = []
            if isinstance(n, ast.Assign):
                targets = n.targets
            elif isinstance(n, ast.AugAssign):
                targets = [n.target]
            for t in targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name):
                    if t.value.id in ("g", "gap"):
                        written.add(t.attr)
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Attribute)
                    and isinstance(n.func.value.value, ast.Name)
                    and n.func.value.value.id in ("g", "gap")):
                written.add(n.func.value.attr)
        self.assertTrue(written <= allowed,
                        f"support evidence writes {written - allowed}")


# ===========================================================================
# 9/10. Interior anomalies and leading/trailing non-wagons
# ===========================================================================

class TestRegionMembershipRules(unittest.TestCase):

    def test_an_interior_anomaly_does_not_delete_a_wagon(self):
        spec = [(E, 2.5), (W, 1.0), (W, 1.0), (B, 1.0), (W, 1.0), (W, 1.0),
                (B, 1.2)]
        win = canonical_window(spec)
        self.assertEqual(win.master_wagon_count, 5)
        self.assertEqual(len(win.interior_non_wagon_objects), 1)
        self.assertEqual([u.global_id for u in win.wagon_units],
                         [f"GW_{i}" for i in range(1, 6)])

    def test_the_interior_anomaly_is_reported_as_still_counted(self):
        spec = [(E, 2.5), (W, 1.0), (E, 1.0), (W, 1.0), (B, 1.2)]
        res = resolve(state_from(canonical_window(spec)))
        d = res.to_dict()
        self.assertTrue(d["interior_anomalies_are_still_counted"])

    def test_leading_and_trailing_non_wagons_get_no_gw_id(self):
        win = canonical_window()
        ids = {u.global_id for u in win.wagon_units}
        for o in (win.leading_non_wagon_objects
                  + win.trailing_non_wagon_objects):
            self.assertNotIn(o.classification, ("",))
            # A non-wagon object carries frames, never an identity.
            self.assertFalse(hasattr(o, "global_id"))
        self.assertEqual(ids, {f"GW_{i}" for i in range(1, 6)})

    def test_the_excluded_objects_are_reported_by_position(self):
        res = resolve(state_from(canonical_window()))
        d = res.to_dict()
        self.assertEqual(len(d["excluded_leading"]), 1)
        self.assertEqual(len(d["excluded_trailing"]), 2)

    def test_the_gate_holds_nothing_outside_carries_an_id(self):
        res = resolve(state_from(canonical_window()),
                      {RUT: top_says_wagon_everywhere()})
        self.assertTrue(res.gate_held, res.gate_violations)
        self.assertEqual(res.gate_violations, [])


# ===========================================================================
# 7/8. Gaps stay bound to the right wagons, per camera
# ===========================================================================

class TestGapsRemainBoundToTheirWagons(unittest.TestCase):

    BOUNDARY = {255: ("GW_1", "GW_2"), 256: ("GW_1", "GW_2"),
                317: ("GW_2", "GW_3"), 318: ("GW_2", "GW_3")}

    def test_a_gap_names_the_wagons_either_side_of_it(self):
        import rendering.feature_overlay_renderer as R
        gap = {"track_id": 2, "global_gap_id": 2,
               "start_frame": 249, "end_frame": 262}
        self.assertEqual(R._gap_neighbours(self.BOUNDARY, gap),
                         "GW_1 | GAP_2 | GW_2")

    def test_the_next_gap_names_the_next_pair(self):
        import rendering.feature_overlay_renderer as R
        gap = {"track_id": 3, "global_gap_id": 3,
               "start_frame": 311, "end_frame": 324}
        self.assertEqual(R._gap_neighbours(self.BOUNDARY, gap),
                         "GW_2 | GAP_3 | GW_3")

    def test_adjacency_comes_from_the_roster_not_from_arithmetic(self):
        """`GAP_9` between GW_1 and GW_2 must still say GW_1 | GAP_9 | GW_2 --
        the pairing is looked up, never computed from the gap number."""
        import rendering.feature_overlay_renderer as R
        gap = {"track_id": 9, "global_gap_id": 9,
               "start_frame": 249, "end_frame": 262}
        self.assertEqual(R._gap_neighbours(self.BOUNDARY, gap),
                         "GW_1 | GAP_9 | GW_2")

    def test_a_cameras_gap_geometry_is_never_copied_from_another(self):
        import rendering.feature_overlay_renderer as R
        seen = {}
        for cam, x0 in ((RU, 100.0), (LU, 40.0), (RUT, 250.0), (LUT, 20.0)):
            g = {"track_id": 2, "start_frame": 25, "end_frame": 27,
                 "hit_frames": [25, 26, 27],
                 "bbox_history": [[x0 + i, 80.0, x0 + i + 40, 200.0]
                                  for i in range(3)]}
            tracks = R._gap_tracks_for({cam: {"gaps": [g]}}, cam)
            seen[cam] = tracks[0]["bbox_history"][0][0]
        self.assertEqual(len(set(seen.values())), 4, seen)

    def test_one_cameras_tracking_is_invisible_to_another(self):
        import rendering.feature_overlay_renderer as R
        tracking = {RU: {"gaps": [{"track_id": 2, "start_frame": 25,
                                   "end_frame": 27, "hit_frames": [25],
                                   "bbox_history": [[1, 2, 3, 4]]}]}}
        self.assertEqual(R._gap_tracks_for(tracking, LUT), [])


# ===========================================================================
# 11. Batch and sequential build the roster the same way
# ===========================================================================

class TestBothModesShareTheConstruction(unittest.TestCase):

    @staticmethod
    def _src(path):
        return open(os.path.join(V4_ROOT, path), encoding="utf-8").read()

    def test_both_modes_call_the_one_assembly_function(self):
        for path in ("wagon_count/run_global_count.py",
                     "orchestrator/global_assembler.py"):
            self.assertIn("assemble_global_train_state_master_fixed",
                          self._src(path), path)

    def test_neither_mode_passes_a_top_camera_classification_to_assembly(self):
        """The canonical input is the MASTER's classifications. A top camera's
        list reaching this argument is the whole failure mode."""
        for path, expect in (
            ("wagon_count/run_global_count.py", "initial_classifications"),
            ("orchestrator/global_assembler.py", "_load_master_classifications"),
        ):
            src = self._src(path)
            for n in ast.walk(ast.parse(src)):
                if not (isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr ==
                        "assemble_global_train_state_master_fixed"):
                    continue
                kw = {k.arg: k.value for k in n.keywords}
                self.assertIn("initial_classifications", kw, path)
                rendered = ast.dump(kw["initial_classifications"])
                for banned in ("RIGHT_UP_TOP", "LEFT_UP_TOP", "TOP_CAMERAS",
                               "support_classifications"):
                    self.assertNotIn(banned, rendered,
                                     f"{path}: {banned} reaches the canonical "
                                     f"classification input")
                self.assertIn(expect, rendered, path)

    def test_the_master_classification_is_keyed_on_the_master_camera(self):
        src = self._src("orchestrator/global_assembler.py")
        self.assertIn("_load_master_classifications(\n            "
                      "bundles[master_camera])", src)

    def test_both_modes_produce_the_same_roster_from_the_same_segments(self):
        """The construction is one function, so identical segments in must give
        identical wagons out -- that is what makes the two modes equivalent
        rather than merely similar."""
        a = ts.get_master_wagon_window(segments(SPEC), verbose=False)
        b = ts.get_master_wagon_window(segments(SPEC), verbose=False)
        self.assertEqual(a.summary(), b.summary())
        self.assertEqual([u.global_id for u in a.wagon_units],
                         [u.global_id for u in b.wagon_units])

    def test_the_equivalence_test_for_the_two_pipelines_still_exists(self):
        self.assertTrue(os.path.exists(os.path.join(
            V4_ROOT, "tests/test_camera_pipeline_equivalence.py")))


# ===========================================================================
# 12. Downstream behaviour is untouched
# ===========================================================================

class TestDownstreamUnchanged(unittest.TestCase):

    def test_stage3_sampling_defaults_are_the_production_ones(self):
        self.assertEqual((C.STAGE3_DOOR_MODE, C.STAGE3_DOOR_STRIDE),
                         ("sampled", 3))
        self.assertEqual((C.STAGE3_DAMAGE_MODE, C.STAGE3_DAMAGE_STRIDE),
                         ("sampled", 3))
        self.assertEqual((C.STAGE3_LOAD_MODE, C.STAGE3_LOAD_STRIDE),
                         ("sampled", 2))

    def test_the_cli_still_exposes_the_stride_overrides(self):
        src = open(os.path.join(V4_ROOT, "orchestrator/master_runner.py"),
                   encoding="utf-8").read()
        for flag in ("--door-sample-stride", "--damage-sample-stride",
                     "--load-sample-stride", "--legacy-inference"):
            self.assertIn(flag, src, flag)

    def test_top_cameras_are_still_available_for_evidence(self):
        """Read-only for identity does not mean excluded: their frames and
        detections are exactly what the PDF and the videos are made of."""
        from reporting import wagon_evidence_grid as WEG
        self.assertEqual(tuple(WEG.TOP_ORDER), tuple(C.TOP_CAMERAS))
        self.assertTrue(set(C.TOP_CAMERAS) <= set(WEG.CAMERA_ORDER))

    def test_damage_association_still_uses_the_canonical_ids(self):
        from core import canonical_association as CA
        self.assertTrue(hasattr(CA, "CanonicalTimeline"))
        src = open(os.path.join(V4_ROOT, "core/canonical_association.py"),
                   encoding="utf-8").read()
        self.assertIn("global_wagon_id", src)

    def test_the_audit_module_adds_no_second_counter(self):
        src = open(os.path.join(V4_ROOT, "core/active_region.py"),
                   encoding="utf-8").read()
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                self.assertNotEqual(n.func.attr, "get_master_wagon_window")
                self.assertNotEqual(n.func.attr, "build_global_wagons")


if __name__ == "__main__":
    unittest.main(verbosity=2)
