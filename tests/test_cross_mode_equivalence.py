"""Sequential and batch must reach the same answer from the same evidence.

The two pipelines schedule and serialize differently -- sequential seals each
camera to disk as it arrives, batch keeps Stage-1 output in memory -- so the only
durable guarantee is that both call the SAME functions on equivalent inputs.
These tests assemble the arguments the way each pipeline assembles them and
compare the results.

The three shared components:

    core.active_region.resolve        active_start / active_end / eligible GW_n
    core.vehicle_type.resolve_train   resolved type + provenance per GW_n
    rendering.feature_overlay_renderer.render_all_cameras   processed videos

and the canonical roster itself comes from `WagonWindow`, whose derivation the
existing `tests/test_camera_pipeline_equivalence.py` already asserts is the same
function in both modes.

What this suite adds is the end-of-pipeline comparison: same evidence in, same
boundaries, same roster, same order, same types out.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "wagon_count"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from core import constants as C                                   # noqa: E402
from core import active_region as AR                              # noqa: E402
from core import vehicle_type as VT                                # noqa: E402
from test_active_region import (_state_from, _recs, ALL_FPS,       # noqa: E402
                                RUT, LUT, W, E, B, SEG)

RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP

#: One train, one set of camera evidence, fed to both pipelines' argument
#: shapes. Top cameras deliberately call EVERY object a WAGON -- the worst
#: realistic misclassification -- so a mode that trusted them would diverge.
LABELS = [E, B, W, W, W, W, W, B, E]
TOP_WRONG = [W] * len(LABELS)


def _sequential_view():
    """How `global_assembler` assembles the arguments: side and top
    classifications read back from each camera's sealed bundle."""
    st, _win = _state_from(LABELS)
    side = {RU: _recs(LABELS), LU: _recs(LABELS)}
    top = {RUT: _recs(TOP_WRONG), LUT: _recs(TOP_WRONG)}
    return st, side, top


def _batch_view():
    """How `run_global_count` assembles them: RIGHT_UP from
    `initial_classifications`, the rest from `support_classifications`."""
    st, _win = _state_from(LABELS)
    initial_classifications = _recs(LABELS)
    support = {LU: _recs(LABELS), RUT: _recs(TOP_WRONG), LUT: _recs(TOP_WRONG)}
    side = {RU: list(initial_classifications)}
    if LU in support:
        side[LU] = support[LU]
    top = {c: support[c] for c in (RUT, LUT) if c in support}
    return st, side, top


def _run(view):
    st, side, top = view()
    ar = AR.resolve(st, top_classifications=top, camera_fps=ALL_FPS,
                    verbose=False)
    tr = VT.resolve_train(st, side_classifications=side, camera_fps=ALL_FPS,
                          top_classifications=top, verbose=False)
    return st, ar, tr


class TestCrossModeEquivalence(unittest.TestCase):

    def setUp(self):
        self.seq_state, self.seq_ar, self.seq_tr = _run(_sequential_view)
        self.bat_state, self.bat_ar, self.bat_tr = _run(_batch_view)

    # ---- the active region ------------------------------------------------

    def test_the_active_region_boundaries_match(self):
        self.assertEqual(self.seq_ar.start.frame, self.bat_ar.start.frame)
        self.assertEqual(self.seq_ar.end.frame, self.bat_ar.end.frame)
        self.assertEqual(self.seq_ar.start.time, self.bat_ar.start.time)
        self.assertEqual(self.seq_ar.end.time, self.bat_ar.end.time)

    def test_the_lifecycle_matches(self):
        self.assertEqual(self.seq_ar.transitions, self.bat_ar.transitions)
        self.assertEqual(self.seq_ar.transitions,
                         [AR.BEFORE, AR.ACTIVE, AR.AFTER])

    def test_the_excluded_non_wagons_match(self):
        for attr in ("excluded_leading", "excluded_trailing"):
            a = [o["classification"] for o in getattr(self.seq_ar, attr)]
            b = [o["classification"] for o in getattr(self.bat_ar, attr)]
            self.assertEqual(a, b, f"{attr} differs between modes")
        self.assertEqual(
            [o["classification"] for o in self.seq_ar.excluded_leading],
            [E, B])
        self.assertEqual(
            [o["classification"] for o in self.seq_ar.excluded_trailing],
            [B, E])

    def test_both_ignore_the_same_out_of_region_predictions(self):
        def _norm(r):
            return sorted((p["camera"], p["reason"]) for p in r.ignored_predictions)
        self.assertEqual(_norm(self.seq_ar), _norm(self.bat_ar))
        self.assertTrue(_norm(self.seq_ar), "no top prediction was even seen")

    def test_the_gate_holds_in_both(self):
        self.assertTrue(self.seq_ar.gate_held, self.seq_ar.gate_violations)
        self.assertTrue(self.bat_ar.gate_held, self.bat_ar.gate_violations)

    # ---- the canonical roster --------------------------------------------

    def test_the_canonical_roster_matches_exactly(self):
        self.assertEqual(self.seq_ar.eligible_global_ids,
                         self.bat_ar.eligible_global_ids)
        self.assertEqual(self.seq_ar.eligible_global_ids,
                         ["GW_1", "GW_2", "GW_3", "GW_4", "GW_5"])

    def test_the_count_and_order_match(self):
        seq = [w.global_id for w in self.seq_state.wagons]
        bat = [w.global_id for w in self.bat_state.wagons]
        self.assertEqual(seq, bat)
        self.assertEqual(len(seq), sum(1 for lb in LABELS if lb == W))

    def test_the_wagon_indices_match(self):
        self.assertEqual([w.wagon_index for w in self.seq_state.wagons],
                         [w.wagon_index for w in self.bat_state.wagons])

    # ---- vehicle type ----------------------------------------------------

    def test_the_resolved_types_match(self):
        seq = [(d["global_id"], d["resolved_type"])
               for d in self.seq_tr["decisions"]]
        bat = [(d["global_id"], d["resolved_type"])
               for d in self.bat_tr["decisions"]]
        self.assertEqual(seq, bat)

    def test_the_type_provenance_matches(self):
        for a, b in zip(self.seq_tr["decisions"], self.bat_tr["decisions"]):
            self.assertEqual(a["source_camera"], b["source_camera"])
            self.assertEqual(a["decision"], b["decision"])
            self.assertEqual(a["corroborated_by"], b["corroborated_by"])

    def test_neither_mode_let_a_top_camera_set_a_type(self):
        for tr in (self.seq_tr, self.bat_tr):
            for d in tr["decisions"]:
                self.assertNotIn(d["source_camera"], C.TOP_CAMERAS)
                if d["top_predictions"]:
                    self.assertEqual(d["top_ignored_reason"],
                                     VT.TOP_IGNORED_REASON)

    # ---- the report ------------------------------------------------------

    def test_both_modes_report_the_same_wagons(self):
        from reporting import combined_train_report as CTR
        seq, _ = CTR.canonical_wagons(self.seq_state, {})
        bat, _ = CTR.canonical_wagons(self.bat_state, {})
        self.assertEqual([u.global_id for u in seq], [u.global_id for u in bat])
        for st, w in ((self.seq_state, seq), (self.bat_state, bat)):
            audit = CTR.audit_report_integrity(state=st, wagons_in_order=w,
                                               verbose=False)
            self.assertTrue(audit["ok"], audit)

    def test_no_leading_or_trailing_non_wagon_became_a_report_row(self):
        from reporting import combined_train_report as CTR
        for st, ar in ((self.seq_state, self.seq_ar),
                       (self.bat_state, self.bat_ar)):
            rows, _ = CTR.canonical_wagons(st, {})
            self.assertEqual(len(rows), len(ar.eligible_global_ids))
            self.assertEqual(len(rows), 5)


class TestTheModesCannotDriftApart(unittest.TestCase):
    """Structural guards: two implementations of a rule diverge, one cannot."""

    @staticmethod
    def _calls(*parts):
        import ast
        tree = ast.parse(open(os.path.join(ROOT, *parts), encoding="utf-8").read())
        out = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                f = n.func
                out.add(f.id if isinstance(f, ast.Name)
                        else getattr(f, "attr", ""))
        return out

    def test_both_call_the_shared_active_region_resolver(self):
        for parts in (("orchestrator", "global_assembler.py"),
                      ("wagon_count", "run_global_count.py")):
            src = open(os.path.join(ROOT, *parts), encoding="utf-8").read()
            self.assertIn("active_region", src, f"{parts} misses the resolver")

    def test_both_call_the_shared_type_resolver(self):
        for parts in (("orchestrator", "global_assembler.py"),
                      ("wagon_count", "run_global_count.py")):
            self.assertIn("resolve_train", self._calls(*parts))

    def test_both_v4_pipelines_share_the_renderer(self):
        for parts in (("orchestrator", "global_assembler.py"),
                      ("orchestrator", "master_runner.py")):
            src = open(os.path.join(ROOT, *parts), encoding="utf-8").read()
            self.assertIn("feature_overlay_renderer.render_all_cameras", src)

    def test_the_window_derivation_is_already_asserted_equivalent(self):
        """The roster's own source is guarded by an existing test; this records
        the dependency so removing that guard is visible here too."""
        p = os.path.join(ROOT, "tests", "test_camera_pipeline_equivalence.py")
        self.assertTrue(os.path.isfile(p))
        src = open(p, encoding="utf-8").read()
        self.assertIn("derive_wagon_window", src)
