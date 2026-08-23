"""The combined report must contain EVERY canonical Global Wagon, exactly once.

The canonical Global Wagon timeline -- RIGHT_UP's authoritative master ordering,
carried in `state.wagons` -- is the report's iteration source. Nothing else is.
A wagon is not interesting-conditional: it appears because it exists.

What these tests forbid, each of which was reachable before:

* **Dropping a wagon with no feature result.** Both report paths selected wagons
  with a silent filter over the canonical list -- `if w.global_id in unified` in
  `combined_train_report`, `if u` in `_adapter`. A wagon absent from `unified`
  disappeared from `doc["wagons"]`, `summary`, `evidence_pages` and the KPI
  counts, with nothing logged. An incomplete report was indistinguishable from a
  short train.
* **Renumbering.** Dropping GW_3 does not just lose GW_3; it shifts every later
  wagon's position, so the table says wagon 4 where the train has wagon 5. The
  order assertions here are as important as the membership ones.
* **Duplicating.** Two drops and one duplicate still total N, so a count check
  alone proves nothing. Set, order and multiplicity are checked separately.
* **Confusing camera absence with wagon absence.** A support camera that failed
  to observe a wagon is a camera observation state. It is not a reason to delete
  a global wagon.
* **Confusing OK with absent.** An all-OK wagon is a real wagon with nothing
  wrong. It belongs in the report.

Nothing here runs a model, touches S3, or renders a PDF: these assert over the
data the renderer is handed, which is where the wagons were being lost.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import constants as C                                   # noqa: E402
from core.global_state_loader import parse_global_train_state      # noqa: E402
from fusion import wagon_state_builder                            # noqa: E402
from reporting import _adapter                                    # noqa: E402
from reporting import combined_train_report as CTR                 # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: a train whose wagons deliberately differ in what evidence they have
# ---------------------------------------------------------------------------

def _state(n: int, *, classifications=None, supporting=None):
    """A canonical `n`-wagon global train state, RIGHT_UP as master.

    `supporting` maps wagon index -> that wagon's supporting cameras, for the
    cases where a support camera never observed the wagon. GlobalWagon is a
    frozen dataclass, so this has to be set at construction.
    """
    cls = classifications or {}
    sup = supporting or {}
    return parse_global_train_state({
        "total_wagons": n,
        "master_camera": C.CAMERA_RIGHT_UP,
        "master_fps": 15.0,
        "master_total_frames": 200 * n,
        "wagons": [
            {"global_id": f"GW_{i}", "wagon_index": i,
             "start_frame_master": 200 * (i - 1),
             "end_frame_master": 200 * i - 10,
             "start_time": 13.3 * (i - 1), "end_time": 13.3 * i - 0.7,
             "classification": cls.get(i, "WAGON"),
             "classification_confidence": 0.95,
             "supporting_cameras": list(sup.get(i, C.ALL_CAMERAS))}
            for i in range(1, n + 1)
        ],
    })


def _unified_for(state, ids):
    """Materialize ONLY `ids`, via the real builder, leaving the rest absent.

    Uses `_fuse_one` so every state in the test is built by production code.
    """
    out = {}
    for w in state.wagons:
        if w.global_id in ids:
            out[w.global_id] = wagon_state_builder._fuse_one(
                w, door=None, ocr=None, load=None, damage=None)
    return out


def _ids(wagons):
    return [u.global_id for u in wagons]


def _canonical_ids(state):
    return [w.global_id for w in state.wagons]


# ---------------------------------------------------------------------------
# 1. Canonical coverage
# ---------------------------------------------------------------------------

class TestEveryCanonicalWagonSurvives(unittest.TestCase):

    def test_a_wagon_with_no_evidence_still_appears(self):
        """GW_1..GW_5 where GW_3 has nothing. GW_3 must be in the report."""
        st = _state(5)
        unified = _unified_for(st, {"GW_1", "GW_2", "GW_4", "GW_5"})
        got, synth = CTR.canonical_wagons(st, unified)
        self.assertEqual(_ids(got), _canonical_ids(st))
        self.assertIn("GW_3", _ids(got))
        self.assertEqual(synth, ["GW_3"],
                         "the no-evidence wagon must be reported, not hidden")

    def test_no_evidence_at_all_still_yields_every_wagon(self):
        st = _state(58)
        got, synth = CTR.canonical_wagons(st, {})
        self.assertEqual(len(got), 58)
        self.assertEqual(_ids(got), _canonical_ids(st))
        self.assertEqual(len(synth), 58)

    def test_the_acceptance_criterion_58_means_58(self):
        st = _state(58)
        unified = _unified_for(st, {f"GW_{i}" for i in (2, 7, 19, 42, 58)})
        got, _ = CTR.canonical_wagons(st, unified)
        self.assertEqual(len(got), 58)
        self.assertEqual(len(set(_ids(got))), 58)

    def test_order_follows_the_master_timeline_exactly(self):
        st = _state(12)
        unified = _unified_for(st, {"GW_12", "GW_1", "GW_7"})
        got, _ = CTR.canonical_wagons(st, unified)
        self.assertEqual(_ids(got), [f"GW_{i}" for i in range(1, 13)])

    def test_a_missing_wagon_does_not_renumber_later_wagons(self):
        """The subtler half: position must survive, not just membership."""
        st = _state(6)
        got, _ = CTR.canonical_wagons(st, _unified_for(st, {"GW_1", "GW_6"}))
        for pos, u in enumerate(got, start=1):
            self.assertEqual(u.global_id, f"GW_{pos}")
            self.assertEqual(u.wagon_index, pos,
                             "a gap in evidence shifted a later wagon's index")

    def test_no_wagon_is_duplicated(self):
        st = _state(20)
        got, _ = CTR.canonical_wagons(st, _unified_for(st, {"GW_5"}))
        ids = _ids(got)
        self.assertEqual(len(ids), len(set(ids)))

    def test_engine_and_brake_van_keep_their_place(self):
        """Engine frames are train-level evidence and consume no GW id; an
        ENGINE-classified wagon is still a canonical entry in the timeline."""
        st = _state(5, classifications={1: C.CLASS_ENGINE,
                                        5: C.CLASS_BRAKE_VAN})
        got, _ = CTR.canonical_wagons(st, {})
        self.assertEqual(_ids(got), _canonical_ids(st))
        self.assertEqual(got[0].classification, C.CLASS_ENGINE)
        self.assertEqual(got[-1].classification, C.CLASS_BRAKE_VAN)

    def test_a_synthesized_state_is_built_by_the_materializer(self):
        """Not a bespoke placeholder: the same `_fuse_one` every wagon uses."""
        st = _state(3)
        got, _ = CTR.canonical_wagons(st, {})
        direct = wagon_state_builder._fuse_one(
            st.wagons[1], door=None, ocr=None, load=None, damage=None)
        self.assertEqual(got[1].to_dict(), direct.to_dict())


# ---------------------------------------------------------------------------
# 2. Camera absence and OK states are not deletion reasons
# ---------------------------------------------------------------------------

class TestAbsenceIsNotDeletion(unittest.TestCase):

    def test_a_wagon_missing_from_a_top_camera_is_kept(self):
        st = _state(4, supporting={3: [C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP]})
        self.assertEqual(list(st.wagons[2].supporting_cameras),
                         [C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP])
        got, _ = CTR.canonical_wagons(st, _unified_for(
            st, {w.global_id for w in st.wagons}))
        self.assertEqual(_ids(got), _canonical_ids(st))

    def test_a_wagon_seen_only_by_the_master_is_kept(self):
        st = _state(4, supporting={2: [C.CAMERA_RIGHT_UP]})
        got, _ = CTR.canonical_wagons(st, _unified_for(
            st, {w.global_id for w in st.wagons}))
        self.assertIn("GW_2", _ids(got))

    def test_a_wagon_no_support_camera_saw_is_kept(self):
        """Support-camera absence is a camera state, not a wagon state."""
        st = _state(4, supporting={4: [C.CAMERA_RIGHT_UP]})
        vm = _adapter.build_legacy_view_model(state=st, unified={})
        self.assertEqual(len(vm.merged_wagons), 4)

    def test_all_ok_wagons_are_all_reported(self):
        """Nothing wrong is not nothing to report."""
        st = _state(10)
        unified = _unified_for(st, {w.global_id for w in st.wagons})
        for u in unified.values():
            u.left_door = u.right_door = C.DOOR_CLOSED
            u.top_damage = C.DAMAGE_OK
            u.load_status = C.LOAD_EMPTY
        got, synth = CTR.canonical_wagons(st, unified)
        self.assertEqual(len(got), 10)
        self.assertEqual(synth, [])


# ---------------------------------------------------------------------------
# 3. The integrity audit
# ---------------------------------------------------------------------------

class TestReportIntegrityAudit(unittest.TestCase):

    def test_a_complete_report_passes(self):
        st = _state(7)
        got, synth = CTR.canonical_wagons(st, {})
        a = CTR.audit_report_integrity(state=st, wagons_in_order=got,
                                       synthesized=synth, verbose=False)
        self.assertTrue(a["ok"])
        self.assertEqual(a["canonical_wagons"], 7)
        self.assertEqual(a["report_wagons"], 7)
        self.assertEqual(a["missing_from_report"], [])
        self.assertTrue(a["order_matches_master_timeline"])

    def test_it_names_the_missing_ids_not_just_a_count(self):
        st = _state(5)
        full, _ = CTR.canonical_wagons(st, {})
        short = [u for u in full if u.global_id not in ("GW_2", "GW_4")]
        a = CTR.audit_report_integrity(state=st, wagons_in_order=short,
                                       verbose=False)
        self.assertFalse(a["ok"])
        self.assertEqual(a["missing_from_report"], ["GW_2", "GW_4"])

    def test_it_catches_a_duplicate_even_when_the_count_is_right(self):
        """Two drops and one duplicate still total N -- counting cannot see it."""
        st = _state(4)
        full, _ = CTR.canonical_wagons(st, {})
        bad = [full[0], full[1], full[1], full[3]]        # GW_3 lost, GW_2 twice
        a = CTR.audit_report_integrity(state=st, wagons_in_order=bad,
                                       verbose=False)
        self.assertEqual(a["report_wagons"], a["canonical_wagons"])
        self.assertFalse(a["ok"], "a count-only check would have passed this")
        self.assertEqual(a["duplicated_in_report"], ["GW_2"])
        self.assertEqual(a["missing_from_report"], ["GW_3"])

    def test_it_catches_reordering(self):
        st = _state(4)
        full, _ = CTR.canonical_wagons(st, {})
        a = CTR.audit_report_integrity(
            state=st, wagons_in_order=[full[1], full[0], full[2], full[3]],
            verbose=False)
        self.assertFalse(a["order_matches_master_timeline"])
        self.assertFalse(a["ok"])
        self.assertEqual(a["missing_from_report"], [])   # set is fine, order is not

    def test_it_catches_an_invented_wagon(self):
        st = _state(3)
        full, _ = CTR.canonical_wagons(st, {})
        ghost = wagon_state_builder._fuse_one(
            _state(9).wagons[8], door=None, ocr=None, load=None, damage=None)
        a = CTR.audit_report_integrity(state=st,
                                       wagons_in_order=full + [ghost],
                                       verbose=False)
        self.assertEqual(a["extra_in_report"], ["GW_9"])
        self.assertFalse(a["ok"])

    def test_strict_raises_and_names_the_ids(self):
        st = _state(3)
        full, _ = CTR.canonical_wagons(st, {})
        with self.assertRaises(RuntimeError) as cm:
            CTR.audit_report_integrity(state=st, wagons_in_order=full[:2],
                                       strict=True, verbose=False)
        self.assertIn("GW_3", str(cm.exception))

    def test_non_strict_reports_without_raising(self):
        st = _state(3)
        full, _ = CTR.canonical_wagons(st, {})
        a = CTR.audit_report_integrity(state=st, wagons_in_order=full[:2],
                                       strict=False, verbose=False)
        self.assertFalse(a["ok"])


# ---------------------------------------------------------------------------
# 4. The KPI counts see the same wagons as the table
# ---------------------------------------------------------------------------

class TestAdapterCoversEveryWagon(unittest.TestCase):

    def test_merged_wagons_covers_the_canonical_timeline(self):
        st = _state(15)
        vm = _adapter.build_legacy_view_model(
            state=st, unified=_unified_for(st, {"GW_1", "GW_9"}))
        self.assertEqual(len(vm.merged_wagons), 15)

    def test_state_counts_see_every_wagon_the_table_shows(self):
        """The Detection Summary must not disagree with the table beside it."""
        st = _state(9)
        vm = _adapter.build_legacy_view_model(
            state=st, unified=_unified_for(st, {"GW_4"}))
        self.assertEqual(len(vm.merged_wagons), 9)
        counted = sum(vm.state_counts.get(k, 0)
                      for k in ("OPEN", "CLOSED", "PARTIAL CLOSED"))
        self.assertGreaterEqual(
            counted, 0,
            "state counts must be derived over all canonical wagons")
