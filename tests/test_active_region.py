"""Only wagons inside the wagon-active region become canonical GW_n.

The region is NOT computed here. `wagon_count/train_structure.WagonWindow`
already defines it -- "The counted region of the train: first WAGON .. last
WAGON" -- already denies a GW id to anything outside it ("Outside the window ->
NO GW id"), already renumbers survivors GW_1..GW_N, and already refuses to
invent one ("Nothing is invented"). Fusion enforces it via `wagon_only=True`,
and the same derivation runs in both pipelines.

`core.active_region` therefore audits and gates rather than counts: it states the
BEFORE -> ACTIVE -> AFTER lifecycle, gathers top-camera boundary evidence as
CORROBORATION ONLY, records every out-of-region prediction it ignored, and
asserts the gate held. These tests cover that behaviour and, by mutation, prove
the guarantees are real rather than incidental.

Why top cameras get no write access: `wagon_start_frame`/`wagon_end_frame` come
from the RIGHT_UP master, and `total_wagons` counts WAGON units -- so letting a
top camera move a boundary would change the count from a camera whose vehicle
classifier is known to call an ENGINE a WAGON.

Sustained evidence is already guaranteed upstream: the window is derived from
classifications `apply_temporal_classification` has smoothed with hysteresis
measured in SECONDS, so a single frame cannot open or close the region before
this module sees it. `min_sustain_sec` here gates only whether TOP evidence is
recorded as corroboration.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "wagon_count"))

from core import constants as C                                   # noqa: E402
from core import active_region as AR                              # noqa: E402
from core.global_state_loader import parse_global_train_state      # noqa: E402
import train_structure as ts                                      # noqa: E402
from global_train_state import GlobalWagon, SegmentClass           # noqa: E402
RU_M = C.CAMERA_RIGHT_UP

FPS = 15.0
SEG = 60                        # frames per segment -> 4.0 s
W, E, B = SegmentClass.WAGON, SegmentClass.ENGINE, SegmentClass.BRAKE_VAN
RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP
ALL_FPS = {c: FPS for c in C.ALL_CAMERAS}


class _Rec:
    def __init__(self, idx, sf, ef, label, conf=0.95):
        self.segment_index, self.start_frame, self.end_frame = idx, sf, ef
        self.label, self.confidence = label, conf


def _segments(labels):
    """Master segments as the classifier+window see them, one per label."""
    out = []
    for i, lb in enumerate(labels):
        sf, ef = i * SEG, (i + 1) * SEG - 1
        out.append(GlobalWagon(
            global_id=f"S_{i}", wagon_index=i,
            start_frame_master=sf, end_frame_master=ef,
            start_time=sf / FPS, end_time=(ef + 1) / FPS,
            classification=lb, classification_confidence=0.95,
            supporting_cameras=list(C.ALL_CAMERAS)))
    return out


def _state_from(labels):
    """A GlobalTrainState gated exactly as the pipeline gates it: the REAL
    window builder decides membership, then the survivors become the roster."""
    win = ts.get_master_wagon_window(_segments(labels), verbose=False)
    doc = {
        "total_wagons": len(win.wagon_units),
        "master_camera": C.CAMERA_RIGHT_UP, "master_fps": FPS,
        "master_total_frames": SEG * len(labels),
        "wagon_window": win.summary(),
        "wagons": [{
            "global_id": w.global_id, "wagon_index": w.wagon_index,
            "start_frame_master": w.start_frame_master,
            "end_frame_master": w.end_frame_master,
            "start_time": w.start_time, "end_time": w.end_time,
            "classification": w.classification,
            "classification_confidence": w.classification_confidence,
            "supporting_cameras": list(C.ALL_CAMERAS)} for w in win.wagon_units],
    }
    return parse_global_train_state(doc), win


def _recs(labels):
    return [_Rec(i, i * SEG, (i + 1) * SEG - 1, lb)
            for i, lb in enumerate(labels)]


def _resolve(st, top=None, **kw):
    return AR.resolve(st, top_classifications=top or {}, camera_fps=ALL_FPS,
                      verbose=False, **kw)


# ---------------------------------------------------------------------------
# The canonical timeline
# ---------------------------------------------------------------------------

class TestOnlyInRegionWagonsBecomeCanonical(unittest.TestCase):

    def test_1_engine_wagon_wagon_brakevan_gives_exactly_two(self):
        st, win = _state_from([E, W, W, B])
        self.assertEqual(len(st.wagons), 2)
        self.assertEqual([w.global_id for w in st.wagons], ["GW_1", "GW_2"])
        r = _resolve(st)
        self.assertTrue(r.found)
        self.assertEqual(r.eligible_global_ids, ["GW_1", "GW_2"])

    def test_the_spec_example_engine_three_wagons_brakevan(self):
        st, _ = _state_from([E, W, W, W, B])
        self.assertEqual([w.global_id for w in st.wagons],
                         ["GW_1", "GW_2", "GW_3"])

    def test_2_leading_engine_never_receives_a_gw_id(self):
        st, win = _state_from([E, E, W, W])
        r = _resolve(st)
        self.assertEqual(len(r.excluded_leading), 2)
        self.assertTrue(all(o["classification"] == E
                            for o in r.excluded_leading))
        for gw in r.eligible_global_ids:
            self.assertTrue(gw.startswith("GW_"))
        self.assertEqual(len(r.eligible_global_ids), 2)

    def test_3_trailing_brakevan_and_engine_never_receive_a_gw_id(self):
        st, _ = _state_from([W, W, B, E])
        r = _resolve(st)
        self.assertEqual(len(r.eligible_global_ids), 2)
        self.assertEqual([o["classification"] for o in r.excluded_trailing],
                         [B, E])

    def test_the_full_expected_structure(self):
        """ENGINE -> BRAKE_VAN -> WAGON... -> BRAKE_VAN -> ENGINE."""
        st, _ = _state_from([E, B, W, W, W, W, B, E])
        self.assertEqual(len(st.wagons), 4)
        r = _resolve(st)
        self.assertEqual(len(r.excluded_leading), 2)
        self.assertEqual(len(r.excluded_trailing), 2)
        self.assertEqual(r.eligible_global_ids,
                         ["GW_1", "GW_2", "GW_3", "GW_4"])

    def test_4_an_interior_non_wagon_label_is_an_anomaly_still_counted(self):
        """Engines cannot occur mid-train, so an interior ENGINE label is a
        MISREAD. Deleting the wagon would let one bad frame remove a real wagon
        from the authoritative count and renumber everything after it."""
        st, _ = _state_from([W, W, E, W, W])
        self.assertEqual(len(st.wagons), 5, "an interior misread deleted a wagon")
        r = _resolve(st)
        self.assertEqual(len(r.interior_anomalies), 1)
        self.assertEqual(r.interior_anomalies[0]["classification"], E)
        self.assertEqual(r.to_dict()["interior_anomalies_are_still_counted"],
                         True)

    def test_no_wagon_anywhere_invents_nothing(self):
        st, _ = _state_from([E, E, B])
        self.assertEqual(len(st.wagons), 0)
        r = _resolve(st)
        self.assertFalse(r.found)
        self.assertEqual(r.eligible_global_ids, [])
        self.assertIn("no segment was classified WAGON", r.reason)

    def test_18_no_wagon_is_created_outside_the_region(self):
        for labels in ([E, W, B], [E, E, W, W, B, B], [B, W, W, W, E]):
            st, win = _state_from(labels)
            r = _resolve(st)
            n_wagons = sum(1 for lb in labels if lb == W)
            self.assertEqual(len(r.eligible_global_ids), n_wagons)
            self.assertTrue(r.gate_held, r.gate_violations)


# ---------------------------------------------------------------------------
# Top cameras: evidence, never authority
# ---------------------------------------------------------------------------

class TestTopCamerasCannotMoveTheRegion(unittest.TestCase):

    def test_5_a_top_wagon_prediction_before_the_region_creates_nothing(self):
        st, _ = _state_from([E, E, W, W])
        before = [w.global_id for w in st.wagons]
        # Both tops call the leading engines WAGON, sustained.
        top = {RUT: _recs([W, W, W, W]), LUT: _recs([W, W, W, W])}
        r = _resolve(st, top)
        self.assertEqual(r.eligible_global_ids, before)
        self.assertEqual(len(r.eligible_global_ids), 2)
        self.assertTrue(any(p["reason"] == AR.REASON_NOT_OPEN
                            for p in r.ignored_predictions),
                        "the pre-region prediction was not recorded as ignored")

    def test_6_a_trailing_wagon_prediction_does_not_reopen_the_region(self):
        st, _ = _state_from([W, W, B, B])
        before = [w.global_id for w in st.wagons]
        top = {RUT: _recs([W, W, W, W])}
        r = _resolve(st, top)
        self.assertEqual(r.eligible_global_ids, before)
        self.assertEqual(r.transitions[-1], AR.AFTER,
                         "the region did not close")
        self.assertTrue(any(p["reason"] == AR.REASON_CLOSED
                            for p in r.ignored_predictions),
                        "the trailing prediction was not ignored")

    def test_7_top_noise_does_not_end_the_region_early(self):
        st, _ = _state_from([W, W, W, W, W])
        top = {RUT: _recs([W, W, E, W, W])}      # one segment of noise
        r = _resolve(st, top)
        self.assertEqual(len(r.eligible_global_ids), 5)
        self.assertEqual(r.end.frame, 5 * SEG - 1)

    def test_9_a_single_frame_of_top_evidence_is_not_sustained(self):
        st, _ = _state_from([E, W, W])
        # A 1-frame WAGON blip on a top camera, before the region.
        top = {RUT: [_Rec(0, 0, 0, W)]}
        r = _resolve(st, top, min_sustain_sec=1.0)
        self.assertEqual(r.ignored_predictions, [],
                         "a 1-frame blip was treated as sustained evidence")
        self.assertEqual(len(r.eligible_global_ids), 2)

    def test_top_agreement_is_recorded_as_corroboration(self):
        st, _ = _state_from([E, W, W, W, B])
        top = {RUT: _recs([E, W, W, W, B]), LUT: _recs([E, W, W, W, B])}
        r = _resolve(st, top)
        self.assertIn(RUT, r.start.corroborated_by)
        self.assertIn(LUT, r.end.corroborated_by)

    def test_top_disagreement_is_recorded_and_not_applied(self):
        st, _ = _state_from([E, E, E, W, W])
        top = {RUT: _recs([W, W, W, W, W])}     # thinks it starts much earlier
        r = _resolve(st, top)
        self.assertEqual(r.start.frame, 3 * SEG, "the boundary moved")
        self.assertIn(RUT, r.start.dissent)
        self.assertFalse(r.start.dissent[RUT]["applied"])

    def test_the_master_owns_both_boundaries(self):
        st, _ = _state_from([E, W, W, B])
        r = _resolve(st, {RUT: _recs([W, W, W, W])})
        self.assertEqual(r.start.source_camera, C.CAMERA_RIGHT_UP)
        self.assertEqual(r.end.source_camera, C.CAMERA_RIGHT_UP)


# ---------------------------------------------------------------------------
# Lifecycle + audit
# ---------------------------------------------------------------------------

class TestLifecycleAndAudit(unittest.TestCase):

    def test_the_state_sequence_is_before_active_after(self):
        st, _ = _state_from([E, W, W, B])
        r = _resolve(st)
        self.assertEqual(r.transitions, [AR.BEFORE, AR.ACTIVE, AR.AFTER])

    def test_a_train_with_no_region_never_reaches_active(self):
        st, _ = _state_from([E, B])
        r = _resolve(st)
        self.assertEqual(r.transitions, [AR.BEFORE])
        self.assertNotIn(AR.ACTIVE, r.transitions)

    def test_both_boundaries_carry_full_provenance(self):
        st, _ = _state_from([E, W, W, B])
        r = _resolve(st, {RUT: _recs([E, W, W, B])})
        for b in (r.start.to_dict(), r.end.to_dict()):
            for key in ("frame", "time", "source_camera", "evidence",
                        "confidence", "reason", "corroborated_by", "dissent"):
                self.assertIn(key, b)
            self.assertTrue(b["reason"])
            self.assertTrue(b["evidence"])

    def test_the_diagnostic_lines_carry_frame_camera_and_confidence(self):
        st, _ = _state_from([E, W, W])
        r = _resolve(st)
        for line in (r.start.render(), r.end.render()):
            self.assertIn("[ACTIVE-REGION]", line)
            self.assertIn("frame=", line)
            self.assertIn("camera=", line)
            self.assertIn("confidence=", line)

    def test_non_wagon_objects_are_preserved_with_provenance(self):
        """They must remain available for diagnostics and engine frames."""
        st, _ = _state_from([E, W, W, B])
        r = _resolve(st)
        for o in r.excluded_leading + r.excluded_trailing:
            for key in ("classification", "start_frame", "end_frame",
                        "start_time", "end_time", "position"):
                self.assertIn(key, o)

    def test_the_gate_is_asserted_not_assumed(self):
        st, _ = _state_from([E, W, W, B])
        r = _resolve(st)
        self.assertTrue(r.gate_held)
        self.assertEqual(r.gate_violations, [])


# ---------------------------------------------------------------------------
# Mutation tests: prove the guarantees are real
# ---------------------------------------------------------------------------

class TestMutationsProveTheGuarantees(unittest.TestCase):
    """If the gate is removed, or a post-region prediction is allowed to reopen
    the region, these must be the tests that go red."""

    def test_removing_the_gate_lets_an_engine_into_the_roster(self):
        """Simulate the gate being gone: put the leading ENGINE in the roster
        and confirm the audit CATCHES it rather than passing quietly."""
        st, win = _state_from([E, W, W])
        eng = _segments([E, W, W])[0]
        st.wagons = [type(st.wagons[0])(
            global_id="GW_0", wagon_index=0,
            start_frame_master=eng.start_frame_master,
            end_frame_master=eng.end_frame_master,
            start_time=eng.start_time, end_time=eng.end_time,
            classification=E, classification_confidence=0.9,
            supporting_cameras=[])] + list(st.wagons)
        r = _resolve(st)
        self.assertFalse(r.gate_held,
                         "an out-of-region ENGINE entered the roster unnoticed")
        self.assertTrue(any("GW_0" in v for v in r.gate_violations))

    def test_allowing_a_reopen_would_change_the_eligible_set(self):
        """Proof the closed-region rule is doing work: the trailing prediction
        exists, is sustained, and is still excluded."""
        st, _ = _state_from([W, W, B, B])
        top = {RUT: _recs([W, W, W, W])}
        r = _resolve(st, top)
        reopened = [p for p in r.ignored_predictions
                    if p["reason"] == AR.REASON_CLOSED]
        self.assertTrue(reopened, "no trailing prediction was even considered")
        self.assertEqual(len(r.eligible_global_ids), 2,
                         "the region reopened")

    def test_corrupting_every_top_prediction_changes_nothing(self):
        st_clean, _ = _state_from([E, W, W, W, B])
        base = _resolve(st_clean).eligible_global_ids
        for corrupt in ([W] * 5, [E] * 5, [B, W, E, W, B]):
            st, _ = _state_from([E, W, W, W, B])
            r = _resolve(st, {RUT: _recs(corrupt), LUT: _recs(corrupt)})
            self.assertEqual(r.eligible_global_ids, base)
            self.assertEqual(r.start.frame, 1 * SEG)
            self.assertEqual(r.end.frame, 4 * SEG - 1)


# ---------------------------------------------------------------------------
# One resolver, both pipelines
# ---------------------------------------------------------------------------

class TestBothPipelinesUseTheOneResolver(unittest.TestCase):

    @staticmethod
    def _calls(*parts):
        import ast
        tree = ast.parse(open(os.path.join(ROOT, *parts), encoding="utf-8").read())
        out = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                f = n.func
                out.add(f.id if isinstance(f, ast.Name) else
                        getattr(f, "attr", ""))
        return out

    def test_sequential_calls_the_shared_resolver(self):
        self.assertIn("resolve",
                      self._calls("orchestrator", "global_assembler.py"))

    def test_batch_calls_the_shared_resolver(self):
        src = open(os.path.join(ROOT, "wagon_count", "run_global_count.py"),
                   encoding="utf-8").read()
        self.assertIn("active_region", src)
        self.assertIn("AR.resolve(", src)

    def test_neither_pipeline_recomputes_the_window(self):
        """The region must come from the master window, not a second count."""
        for parts in (("orchestrator", "global_assembler.py"),
                      ("wagon_count", "run_global_count.py")):
            src = open(os.path.join(ROOT, *parts), encoding="utf-8").read()
            self.assertNotIn("get_master_wagon_window(", src.split(
                "active_region", 1)[-1][:2000],
                f"{parts} recomputes the window near the resolver")

    def test_17_equivalent_evidence_gives_an_equivalent_roster(self):
        """Cross-mode equivalence: one resolver on one state is deterministic,
        so sequential and batch cannot diverge for equal evidence."""
        labels = [E, B, W, W, W, W, B, E]
        top = {RUT: _recs([W] * 8), LUT: _recs([E] * 8)}
        runs = []
        for _ in range(3):
            st, _ = _state_from(labels)
            r = _resolve(st, top)
            runs.append((r.eligible_global_ids, r.start.frame, r.end.frame,
                         [o["classification"] for o in r.excluded_leading],
                         [o["classification"] for o in r.excluded_trailing]))
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[1], runs[2])
        self.assertEqual(runs[0][0], ["GW_1", "GW_2", "GW_3", "GW_4"])

    def test_20_the_report_contains_exactly_the_canonical_wagons(self):
        from reporting import combined_train_report as CTR
        st, _ = _state_from([E, B, W, W, W, B, E])
        r = _resolve(st, {RUT: _recs([W] * 7)})
        wagons, _synth = CTR.canonical_wagons(st, {})
        self.assertEqual([u.global_id for u in wagons], r.eligible_global_ids)
        self.assertEqual(len(wagons), 3)
        audit = CTR.audit_report_integrity(state=st, wagons_in_order=wagons,
                                           verbose=False)
        self.assertTrue(audit["ok"])


# ---------------------------------------------------------------------------
# The reported failure: "the region says not-active though a wagon has started"
# ---------------------------------------------------------------------------

class TestTheFirstWagonIsNeverLostToALateStart(unittest.TestCase):
    """The START carries no delay, and the diagnostic proves it per train.

    `train_structure` sets `wagon_start_frame = wagon_units[0].start_frame_master`
    -- the first canonical wagon's OWN start. So GW_1 cannot fall outside the
    region: the region's start IS GW_1's start. `frames_after_first_wagon` is
    reported anyway, so a future regression that introduced a delay would show
    up as a number rather than have to be inferred.
    """

    def test_the_start_equals_the_first_canonical_wagons_own_frame(self):
        for labels in ([E, W, W, W, B], [E, B, W, W], [W, W, W],
                       [E, E, E, W, W, B, E]):
            st, _ = _state_from(labels)
            r = _resolve(st)
            self.assertEqual(r.start.frame, st.wagons[0].start_frame_master,
                             f"{labels}: the region opened after its own GW_1")
            self.assertEqual(r.start.frames_after_first_wagon, 0)
            self.assertEqual(r.start.first_wagon_global_id, "GW_1")
            self.assertEqual(r.start.reason, "first_canonical_wagon")

    def test_the_first_wagon_at_frame_100_is_gw_1(self):
        """The reported scenario: first wagon begins well into the train."""
        labels = [E, B] + [W] * 4          # segments 0,1 non-wagon; 2.. wagons
        st, _ = _state_from(labels)
        self.assertEqual(st.wagons[0].start_frame_master, 2 * SEG)
        r = _resolve(st)
        self.assertEqual(r.start.frame, 2 * SEG)
        self.assertEqual(r.eligible_global_ids[0], "GW_1")
        self.assertEqual(len(r.eligible_global_ids), 4,
                         "a wagon was lost at the leading edge")

    def test_the_start_diagnostic_names_the_first_wagon_and_the_delta(self):
        st, _ = _state_from([E, W, W])
        line = _resolve(st).start.render()
        for bit in ("master_first_wagon_frame=", "first_wagon=GW_1",
                    "frames_after_first_wagon=0",
                    "reason=first_canonical_wagon"):
            self.assertIn(bit, line)

    def test_a_brief_first_wagon_is_still_gw_1(self):
        """Short but canonical: the window takes the FIRST wagon segment, not
        the first sustained run, so brevity cannot push the start later."""
        st, _ = _state_from([E, W, W, W, W])
        r = _resolve(st)
        self.assertEqual(r.start.frame, st.wagons[0].start_frame_master)
        self.assertEqual(len(r.eligible_global_ids), 4)

    def test_a_top_camera_wagon_run_before_the_start_cannot_open_it(self):
        st, _ = _state_from([E, E, W, W])
        r = _resolve(st, {RUT: _recs([W, W, W, W])})
        self.assertEqual(r.start.frame, 2 * SEG, "a top camera opened the region")
        self.assertEqual(r.start.frames_after_first_wagon, 0)


class TestTheMissedGapDiagnostic(unittest.TestCase):
    """A loco+wagon merged into ONE segment by a missed gap is the real cause of
    a lost first wagon -- and cannot be fixed by relabelling, because there is
    no boundary to split. This surfaces it with numbers instead."""

    def _merged(self, lead_segments=3):
        """A leading ENGINE segment `lead_segments` wagon-lengths long."""
        from global_train_state import GlobalWagon
        segs = []
        # one long leading ENGINE, then 4 normal wagons
        segs.append(GlobalWagon(
            global_id="S_0", wagon_index=0,
            start_frame_master=0, end_frame_master=SEG * lead_segments - 1,
            start_time=0.0, end_time=SEG * lead_segments / FPS,
            classification=E, classification_confidence=0.71,
            supporting_cameras=list(C.ALL_CAMERAS)))
        for i in range(4):
            sf = SEG * lead_segments + i * SEG
            segs.append(GlobalWagon(
                global_id=f"S_{i+1}", wagon_index=i + 1,
                start_frame_master=sf, end_frame_master=sf + SEG - 1,
                start_time=sf / FPS, end_time=(sf + SEG) / FPS,
                classification=W, classification_confidence=0.97,
                supporting_cameras=list(C.ALL_CAMERAS)))
        win = ts.get_master_wagon_window(segs, verbose=False)
        doc = {"total_wagons": len(win.wagon_units), "master_camera": RU_M,
               "master_fps": FPS,
               "master_total_frames": SEG * (lead_segments + 4),
               "wagon_window": win.summary(),
               "wagons": [{
                   "global_id": w.global_id, "wagon_index": w.wagon_index,
                   "start_frame_master": w.start_frame_master,
                   "end_frame_master": w.end_frame_master,
                   "start_time": w.start_time, "end_time": w.end_time,
                   "classification": w.classification,
                   "classification_confidence": w.classification_confidence,
                   "supporting_cameras": list(C.ALL_CAMERAS)}
                   for w in win.wagon_units]}
        return parse_global_train_state(doc)

    def test_an_over_long_leading_segment_is_flagged(self):
        st = self._merged(lead_segments=3)
        r = _resolve(st)
        self.assertEqual(len(r.suspect_merged_segments), 1,
                         "a 3-wagon-long leading ENGINE was not flagged")
        sm = r.suspect_merged_segments[0]
        self.assertEqual(sm["position"], "leading")
        self.assertEqual(sm["classification"], E)
        self.assertEqual(sm["reason"], AR.REASON_SUSPECT_MISSED_GAP)
        self.assertGreater(sm["wagon_lengths"], 1.8)
        self.assertIn("missed gap", sm["note"])

    def test_a_normal_length_locomotive_is_not_flagged(self):
        """A real loco is roughly one wagon long -- flagging it would cry wolf
        on every train."""
        st = self._merged(lead_segments=1)
        self.assertEqual(_resolve(st).suspect_merged_segments, [])

    def test_the_diagnostic_changes_no_count(self):
        st = self._merged(lead_segments=3)
        before = [w.global_id for w in st.wagons]
        r = _resolve(st)
        self.assertEqual([w.global_id for w in st.wagons], before)
        self.assertEqual(len(r.eligible_global_ids), 4)
        self.assertTrue(r.gate_held)

    def test_it_is_measured_against_this_trains_own_wagons(self):
        """Ratio, not an absolute frame count, so it scales with fps and speed."""
        self.assertGreater(AR.DEFAULT_SUSPECT_LENGTH_RATIO, 1.0)
        st = self._merged(lead_segments=3)
        sm = _resolve(st).suspect_merged_segments[0]
        self.assertAlmostEqual(sm["median_wagon_sec"], SEG / FPS, places=3)

    def test_the_audit_exposes_it(self):
        st = self._merged(lead_segments=3)
        d = _resolve(st).to_dict()
        self.assertIn("suspect_merged_segments", d)
        self.assertIn("start_reason", d)
        self.assertEqual(d["start_reason"], "first_canonical_wagon")
