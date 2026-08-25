"""Vehicle TYPE comes from the side cameras; top cameras never decide it.

Global Wagon IDENTITY and vehicle TYPE are separate questions. Identity -- which
objects exist and in what order -- belongs to the RIGHT_UP master
reconstruction. This suite covers only the second question and, just as
importantly, proves the first is untouched by it.

Two properties of the existing architecture make this tractable, and both are
asserted here rather than assumed:

  * type is already assigned at ONE place, `global_alignment`, from
    `initial_classifications`, which is built from the MASTER alone;
  * `assemble_global_train_state_master_fixed` states the invariant --
    "total_wagons == the WAGON units of the master's wagon window. ENGINE and
    BRAKE_VAN are preserved as metadata but never receive a GW id and never
    extend the wagon timeline. Support cameras contribute association +
    evidence + diagnostics only."

So the resolver's job is narrow: bring LEFT_UP in as corroboration and as an
explicit fallback, record provenance, and record what the top cameras said so an
audit can show it carried no weight.

RIGHT_UP always wins. That is the conservative choice and the reason the count
cannot move: `total_wagons` counts WAGON units, so letting LEFT_UP flip a wagon
to ENGINE would change the count from a support camera's opinion. Where RIGHT_UP
has an opinion the resolved type equals the existing one, and a test asserts the
timeline is byte-identical even when the top predictions are deliberately
corrupted.

One resolver, both pipelines: these drive `core.vehicle_type` directly, which is
the same function sequential and batch call.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "wagon_count"))

from core import constants as C                                   # noqa: E402
from core import vehicle_type as VT                                # noqa: E402
from core.global_state_loader import parse_global_train_state      # noqa: E402

FPS = 15.0
W, E, B = C.CLASS_WAGON, C.CLASS_ENGINE, C.CLASS_BRAKE_VAN
RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP
RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP

ALL_FPS = {c: FPS for c in C.ALL_CAMERAS}


class _Rec:
    """A classification record, in one camera's own local frames."""

    def __init__(self, idx, start_frame, end_frame, label, confidence):
        self.segment_index = idx
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.label = label
        self.confidence = confidence


def _state(n, *, types=None):
    """`n` canonical wagons, 4 s each, typed as the master saw them."""
    t = types or {}
    return parse_global_train_state({
        "total_wagons": n, "master_camera": RU, "master_fps": FPS,
        "master_total_frames": 60 * n,
        "wagons": [
            {"global_id": f"GW_{i}", "wagon_index": i,
             "start_frame_master": 60 * (i - 1), "end_frame_master": 60 * i - 1,
             "start_time": 4.0 * (i - 1), "end_time": 4.0 * i,
             "classification": t.get(i, W), "classification_confidence": 0.9,
             "supporting_cameras": list(C.ALL_CAMERAS)}
            for i in range(1, n + 1)],
    })


def _recs(n, labels, conf=0.95):
    """One record per wagon window, in local frames matching `_state`."""
    return [_Rec(i, 60 * (i - 1), 60 * i - 1, labels.get(i, W), conf)
            for i in range(1, n + 1)]


def _resolve(state, side, top=None, **kw):
    return VT.resolve_train(state, side_classifications=side,
                            camera_fps=ALL_FPS, top_classifications=top,
                            verbose=False, **kw)


def _types(state):
    return [w.classification for w in state.wagons]


def _ids(state):
    return [w.global_id for w in state.wagons]


# ---------------------------------------------------------------------------
# The eleven cases, in order
# ---------------------------------------------------------------------------

class TestTheSpecifiedCases(unittest.TestCase):

    def test_1_both_sides_and_both_tops_say_wagon(self):
        st = _state(3)
        res = _resolve(st, {RU: _recs(3, {}), LU: _recs(3, {})},
                       {RUT: _recs(3, {}), LUT: _recs(3, {})})
        self.assertEqual(_types(st), [W, W, W])
        self.assertEqual(res["primary"], 3)
        self.assertEqual(res["corroborated"], 3)
        self.assertEqual(res["changed"], [])

    def test_2_sides_say_engine_while_tops_wrongly_say_wagon(self):
        """The headline case. ENGINE must survive, and no wagon is created."""
        st = _state(3, types={1: W, 2: W, 3: W})
        before = _ids(st)
        res = _resolve(st,
                       {RU: _recs(3, {1: E}), LU: _recs(3, {1: E})},
                       {RUT: _recs(3, {}), LUT: _recs(3, {})})
        self.assertEqual(_types(st)[0], E, "the engine was called a wagon")
        self.assertEqual(_ids(st), before, "the timeline changed")
        d = res["decisions"][0]
        self.assertEqual(d["source_camera"], RU)
        self.assertEqual(d["top_predictions"][RUT]["type"], W)
        self.assertEqual(d["top_predictions"][RUT]["ignored_reason"],
                         VT.TOP_IGNORED_REASON)

    def test_3_right_up_says_engine_left_up_says_wagon(self):
        """RIGHT_UP wins; LEFT_UP's dissent is recorded, not applied."""
        st = _state(2)
        res = _resolve(st, {RU: _recs(2, {1: E}), LU: _recs(2, {1: W})})
        self.assertEqual(_types(st)[0], E)
        d = res["decisions"][0]
        self.assertEqual(d["decision"], VT.PRIMARY)
        self.assertEqual(d["dissent"][LU]["type"], W)
        self.assertFalse(d["dissent"][LU]["applied"])

    def test_4_right_up_unavailable_left_up_identifies_engine(self):
        st = _state(2)
        res = _resolve(st, {LU: _recs(2, {1: E})})       # no RIGHT_UP at all
        self.assertEqual(_types(st)[0], E)
        d = res["decisions"][0]
        self.assertEqual(d["decision"], VT.FALLBACK)
        self.assertEqual(d["source_camera"], LU)

    def test_5_both_sides_miss_while_tops_say_wagon(self):
        """No side evidence: the top cameras must NOT become the authority."""
        st = _state(2, types={1: E, 2: W})
        res = _resolve(st, {}, {RUT: _recs(2, {}), LUT: _recs(2, {})})
        self.assertEqual(_types(st), [E, W], "a top camera set the type")
        for d in res["decisions"]:
            self.assertEqual(d["decision"], VT.UNRESOLVED)
            self.assertEqual(d["source_camera"], "")
        self.assertEqual(res["changed"], [])

    def test_6_a_genuine_wagon_with_noisy_top_classification_stays(self):
        st = _state(3)
        before = _ids(st)
        _resolve(st, {RU: _recs(3, {}), LU: _recs(3, {})},
                 {RUT: _recs(3, {2: E}), LUT: _recs(3, {2: B})})
        self.assertEqual(_types(st), [W, W, W], "a top camera deleted a wagon")
        self.assertEqual(_ids(st), before)

    def test_8_a_top_camera_engine_creates_no_new_wagon(self):
        st = _state(4)
        before = _ids(st)
        _resolve(st, {RU: _recs(4, {})},
                 {RUT: _recs(4, {1: E, 4: B}), LUT: _recs(4, {1: E})})
        self.assertEqual(_ids(st), before)
        self.assertEqual(len(st.wagons), 4)

    def test_9_corrupting_every_top_prediction_changes_nothing(self):
        """Identity and order must be invariant under top-camera corruption."""
        clean = _state(6)
        _resolve(clean, {RU: _recs(6, {})})
        baseline_ids, baseline_types = _ids(clean), _types(clean)

        for corruption in ({i: E for i in range(1, 7)},
                           {i: B for i in range(1, 7)},
                           {1: E, 3: B, 5: E}):
            st = _state(6)
            _resolve(st, {RU: _recs(6, {})},
                     {RUT: _recs(6, corruption), LUT: _recs(6, corruption)})
            self.assertEqual(_ids(st), baseline_ids)
            self.assertEqual(_types(st), baseline_types)

    def test_10_the_same_evidence_gives_the_same_answer_every_time(self):
        """One resolver, so sequential and batch cannot diverge -- the same
        function on the same evidence must be deterministic."""
        side = {RU: _recs(5, {2: E, 5: B}), LU: _recs(5, {2: E})}
        top = {RUT: _recs(5, {2: W}), LUT: _recs(5, {2: W})}
        runs = []
        for _ in range(3):
            st = _state(5)
            res = _resolve(st, side, top)
            runs.append((_ids(st), _types(st),
                         [d["source_camera"] for d in res["decisions"]]))
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[1], runs[2])
        self.assertEqual(runs[0][1][1], E)


# ---------------------------------------------------------------------------
# Identity is never touched
# ---------------------------------------------------------------------------

class TestIdentityIsNeverTouched(unittest.TestCase):

    def test_the_resolver_can_neither_add_nor_drop_a_wagon(self):
        for n in (1, 5, 58):
            st = _state(n)
            _resolve(st, {RU: _recs(n, {1: E})}, {RUT: _recs(n, {})})
            self.assertEqual(len(st.wagons), n)
            self.assertEqual(_ids(st), [f"GW_{i}" for i in range(1, n + 1)])

    def test_order_is_preserved(self):
        st = _state(8)
        _resolve(st, {RU: _recs(8, {4: E})})
        self.assertEqual([w.wagon_index for w in st.wagons], list(range(1, 9)))

    def test_engine_wagons_are_annotated_not_removed(self):
        """ENGINE is preserved for diagnostics and engine-frame extraction."""
        st = _state(4)
        _resolve(st, {RU: _recs(4, {1: E, 4: B})})
        self.assertEqual(len(st.wagons), 4)
        self.assertEqual(_types(st), [E, W, W, B])

    def test_apply_false_reports_without_writing(self):
        st = _state(3)
        res = _resolve(st, {RU: _recs(3, {1: E})}, apply=False)
        self.assertEqual(_types(st), [W, W, W], "apply=False still wrote")
        self.assertEqual(res["decisions"][0]["resolved_type"], E)


# ---------------------------------------------------------------------------
# Evidence handling
# ---------------------------------------------------------------------------

class TestEvidenceHandling(unittest.TestCase):

    def test_the_longest_overlapping_record_wins_not_the_centre_frame(self):
        """A one-frame burst at the wagon's centre must not take the label."""
        st = _state(1)
        recs = [_Rec(0, 0, 59, W, 0.95), _Rec(1, 29, 30, E, 0.99)]
        _resolve(st, {RU: recs})
        self.assertEqual(_types(st), [W],
                         "a 2-frame burst outvoted a full-wagon record")

    def test_confidence_breaks_an_overlap_tie(self):
        st = _state(1)
        recs = [_Rec(0, 0, 59, W, 0.60), _Rec(1, 0, 59, E, 0.95)]
        _resolve(st, {RU: recs})
        self.assertEqual(_types(st), [E])

    def test_a_record_that_does_not_overlap_is_ignored(self):
        st = _state(2)
        _resolve(st, {RU: [_Rec(0, 500, 560, E, 0.99)]})
        self.assertEqual(_types(st), [W, W])

    def test_a_camera_offset_is_applied_to_the_local_clock(self):
        """LEFT_UP's records are in ITS clock; the offset maps them."""
        st = _state(2)
        shifted = [_Rec(i, 60 * (i - 1) + 150, 60 * i - 1 + 150,
                        E if i == 1 else W, 0.95) for i in (1, 2)]
        VT.resolve_train(st, side_classifications={LU: shifted},
                         camera_fps=ALL_FPS,
                         camera_offsets={LU: -10.0},   # local = global + 10 s
                         verbose=False)
        self.assertEqual(_types(st)[0], E,
                         "the offset was not applied to the local window")

    def test_a_zero_fps_camera_contributes_nothing(self):
        st = _state(2)
        VT.resolve_train(st, side_classifications={RU: _recs(2, {1: E})},
                         camera_fps={RU: 0.0}, verbose=False)
        self.assertEqual(_types(st), [W, W])

    def test_a_broken_record_does_not_raise(self):
        st = _state(2)
        bad = [_Rec(0, None, None, E, 0.9), object()]
        res = _resolve(st, {RU: bad})
        self.assertEqual(len(res["decisions"]), 2)


# ---------------------------------------------------------------------------
# Provenance and the audit trail
# ---------------------------------------------------------------------------

class TestProvenance(unittest.TestCase):

    def test_every_field_the_audit_needs_is_present(self):
        st = _state(2)
        res = _resolve(st, {RU: _recs(2, {1: E}), LU: _recs(2, {1: W})},
                       {RUT: _recs(2, {1: W})})
        d = res["decisions"][0]
        for key in ("global_id", "resolved_type", "source_camera",
                    "source_track_id", "confidence", "decision",
                    "corroborated_by", "dissent", "top_predictions",
                    "previous_type", "changed", "reason"):
            self.assertIn(key, d)
        self.assertTrue(d["reason"], "no reason was recorded")

    def test_primary_and_corroborated_are_distinguishable(self):
        st = _state(2)
        res = _resolve(st, {RU: _recs(2, {}), LU: _recs(2, {})})
        d = res["decisions"][0]
        self.assertEqual(d["decision"], VT.PRIMARY)
        self.assertEqual(d["corroborated_by"], [LU])

    def test_the_top_ignored_reason_is_one_shared_constant(self):
        """The log line and the report field must not drift apart."""
        st = _state(1)
        res = _resolve(st, {RU: _recs(1, {})}, {RUT: _recs(1, {1: E})})
        self.assertEqual(res["decisions"][0]["top_ignored_reason"],
                         VT.TOP_IGNORED_REASON)
        self.assertEqual(VT.TOP_IGNORED_REASON, "TOP_CAMERA_NON_AUTHORITATIVE")

    def test_the_diagnostic_line_names_type_source_and_confidence(self):
        d = VT.TypeDecision(global_id="GW_25", resolved_type=E,
                            source_camera=RU, confidence=0.91,
                            decision=VT.PRIMARY)
        line = d.render()
        for bit in ("[TYPE-RESOLUTION]", "GW_25", f"type={E}",
                    f"source={RU}", "confidence="):
            self.assertIn(bit, line)


# ---------------------------------------------------------------------------
# The architecture this rests on
# ---------------------------------------------------------------------------

class TestTheUnderlyingInvariants(unittest.TestCase):
    """If these ever stop holding, the resolver's guarantees stop holding."""

    def _src(self, *parts):
        return open(os.path.join(ROOT, *parts), encoding="utf-8").read()

    def test_type_is_assigned_from_the_master_classifications_only(self):
        src = self._src("wagon_count", "run_global_count.py")
        self.assertIn("camera_id=CAMERA_RIGHT_UP", src,
                      "master classification is no longer RIGHT_UP-scoped")

    def test_the_fixed_master_invariant_is_still_documented(self):
        src = self._src("wagon_count", "global_fusion.py")
        self.assertIn("never receive a GW id", src)
        self.assertIn("Support cameras contribute association", src)

    def test_top_cameras_are_not_in_the_type_authority(self):
        for cam in C.TOP_CAMERAS:
            self.assertNotIn(cam, VT.SIDE_AUTHORITY)
        self.assertEqual(set(VT.NON_AUTHORITATIVE_CAMERAS), set(C.TOP_CAMERAS))

    def test_right_up_is_first_in_the_authority_order(self):
        self.assertEqual(VT.SIDE_AUTHORITY[0], C.CAMERA_RIGHT_UP)


# ---------------------------------------------------------------------------
# Both pipelines, one resolver
# ---------------------------------------------------------------------------

class TestBothPipelinesUseTheOneResolver(unittest.TestCase):
    """The requirement is not "both implement the rule" but "both call the same
    function". Two implementations of a rule diverge; one function cannot."""

    @staticmethod
    def _calls(*parts):
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

    def _src(self, *parts):
        return open(os.path.join(ROOT, *parts), encoding="utf-8").read()

    def test_sequential_calls_resolve_train(self):
        self.assertIn("resolve_train",
                      self._calls("orchestrator", "global_assembler.py"))

    def test_batch_calls_resolve_train(self):
        self.assertIn("resolve_train",
                      self._calls("wagon_count", "run_global_count.py"))

    def test_neither_pipeline_implements_its_own_rule(self):
        """No second resolver: the decision words must appear only in core."""
        for parts in (("orchestrator", "global_assembler.py"),
                      ("wagon_count", "run_global_count.py")):
            calls = self._calls(*parts)
            self.assertNotIn("resolve_wagon", calls,
                             f"{parts} resolves wagons itself")

    def test_both_pass_side_and_top_separately(self):
        """Top evidence must arrive as `top_classifications`, never as a side
        camera -- passing it in the side slot would make it authoritative."""
        for parts in (("orchestrator", "global_assembler.py"),
                      ("wagon_count", "run_global_count.py")):
            src = self._src(*parts)
            self.assertIn("side_classifications=", src)
            self.assertIn("top_classifications=", src)

    def test_sequential_reads_side_cameras_for_the_side_slot(self):
        src = self._src("orchestrator", "global_assembler.py")
        self.assertIn("C.SIDE_CAMERAS", src)
        self.assertIn("C.TOP_CAMERAS", src)

    def test_the_resolver_lives_in_core_so_both_can_import_it(self):
        self.assertTrue(os.path.isfile(os.path.join(ROOT, "core",
                                                    "vehicle_type.py")))


class TestResolutionDoesNotDisturbTheReport(unittest.TestCase):
    """Case 11: the combined report is driven by the canonical timeline, and
    type resolution must not add, drop or reorder a row."""

    def test_every_canonical_wagon_survives_resolution(self):
        from reporting import combined_train_report as CTR
        st = _state(58)
        before = _ids(st)
        _resolve(st, {RU: _recs(58, {1: E, 58: B})},
                 {RUT: _recs(58, {i: W for i in range(1, 59)})})
        wagons, _synth = CTR.canonical_wagons(st, {})
        self.assertEqual([u.global_id for u in wagons], before)
        self.assertEqual(len(wagons), 58)

    def test_the_integrity_audit_still_passes_after_resolution(self):
        from reporting import combined_train_report as CTR
        st = _state(20)
        _resolve(st, {RU: _recs(20, {1: E})}, {LUT: _recs(20, {1: W})})
        wagons, synth = CTR.canonical_wagons(st, {})
        audit = CTR.audit_report_integrity(
            state=st, wagons_in_order=wagons, synthesized=synth, verbose=False)
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["missing_from_report"], [])
        self.assertEqual(audit["duplicated_in_report"], [])

    def test_engine_rows_are_still_present_for_engine_frame_extraction(self):
        """ENGINE must be preserved, not deleted -- engine frames need it."""
        from reporting import combined_train_report as CTR
        st = _state(5)
        _resolve(st, {RU: _recs(5, {1: E, 5: B})})
        wagons, _ = CTR.canonical_wagons(st, {})
        self.assertEqual(len(wagons), 5)
        self.assertEqual([u.classification for u in wagons],
                         [E, W, W, W, B])
