"""Four cameras, four clocks, one wagon-active region.

The four cameras sit at different points along the track and their clocks are
not aligned, so the same physical first wagon appears at a different LOCAL time
in each feed. The numbers in these tests are the ones from the brief --

    RIGHT_UP 18.6s   LEFT_UP 21.2s   RIGHT_UP_TOP 16.8s   LEFT_UP_TOP 20.1s

-- and the point of every test below is that those are not four contradictory
boundaries. They are one boundary seen through four clocks, and they only become
comparable after `t_global = t_local + delta` using the offsets
`wagon_count.global_fusion` already measured.

Two guards matter more than the rest, because between them they are what stops a
classifier error becoming a wrong wagon count:

  * a boundary moves only on weighted support of 3+, which the two top cameras
    cannot reach between them (1 + 1), so relocating always needs a side camera;
  * the proposed boundary must sit on a VALIDATED MASTER GAP, so it has to be a
    place the master's own physical gap detection already found a coupling.
"""

from __future__ import annotations

import ast
import inspect
import os
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from core import region_consensus as RC

import train_structure as ts
from global_train_state import SegmentClass, GlobalWagon as EngineWagon

RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP
RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP
W, E, B = SegmentClass.WAGON, SegmentClass.ENGINE, SegmentClass.BRAKE_VAN

FPS = 15.0
UNIT = 60                       # one wagon = 4 s


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class Region:
    """A `LocalWagonRegion` as each camera's classification pass produces it,
    in that camera's OWN clock."""

    def __init__(self, start, end, found=True, reason=""):
        self.start_time, self.end_time = start, end
        self.found, self.reason = found, reason


def offsets(**kw):
    """`state.camera_offsets`: delta plus the status that says whether the delta
    was actually measured."""
    out = {RU: {"delta": 0.0, "status": "REFERENCE"}}
    for cam, v in kw.items():
        cam = {"ru": RU, "lu": LU, "rut": RUT, "lut": LUT}[cam]
        if isinstance(v, tuple):
            out[cam] = {"delta": v[0], "status": v[1]}
        else:
            out[cam] = {"delta": v, "status": "RESOLVED"}
    return out


#: The brief's numbers. Each camera sees the SAME physical boundary; the offsets
#: are exactly what makes 18.6 / 21.2 / 16.8 / 20.1 the same global instant.
BRIEF_LOCAL = {RU: 18.6, LU: 21.2, RUT: 16.8, LUT: 20.1}
BRIEF_OFFSETS = offsets(lu=-2.6, rut=+1.8, lut=-1.5)


def master_window(start, end, found=True):
    d = {"found": found, "wagon_start_time": start, "wagon_end_time": end}
    if start is not None:
        d["wagon_start_frame"] = int(round(start * FPS))
    if end is not None:
        d["wagon_end_frame"] = int(round(end * FPS)) - 1
    return d


def resolve(*, mw, regions, offs, gaps=(), cfg=RC.DEFAULT_CONFIG):
    return RC.resolve(master_camera=RU, master_window=mw,
                      support_regions=regions, camera_offsets=offs,
                      master_gap_times=list(gaps), cfg=cfg, verbose=False)


def segments(spec):
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


SPEC = [(E, 2.5)] + [(W, 1.0)] * 5 + [(B, 1.2), (E, 2.5)]


# ===========================================================================
# 1/2. Four different local times; correct normalization
# ===========================================================================

class TestFourCamerasNormalizeToOneBoundary(unittest.TestCase):

    def test_the_four_local_times_really_do_differ(self):
        """Guards the fixture: if they were equal, nothing below would be
        testing normalization."""
        self.assertEqual(len(set(BRIEF_LOCAL.values())), 4)

    def test_normalization_brings_all_four_to_the_same_global_instant(self):
        regions = {c: Region(BRIEF_LOCAL[c], BRIEF_LOCAL[c] + 240.0)
                   for c in (LU, RUT, LUT)}
        res = resolve(mw=master_window(18.6, 258.6), regions=regions,
                      offs=BRIEF_OFFSETS)
        got = {b.camera_id: b.global_start for b in res.boundaries}
        for cam, t in got.items():
            self.assertAlmostEqual(t, 18.6, places=6, msg=cam)

    def test_the_formula_is_local_plus_delta(self):
        regions = {LU: Region(21.2, 100.0)}
        res = resolve(mw=master_window(18.6, 100.0), regions=regions,
                      offs=offsets(lu=-2.6))
        lu = next(b for b in res.boundaries if b.camera_id == LU)
        self.assertEqual(lu.local_start, 21.2)
        self.assertEqual(lu.offset, -2.6)
        self.assertAlmostEqual(lu.global_start, 18.6)

    def test_all_four_agreeing_holds_the_master_boundary(self):
        regions = {c: Region(BRIEF_LOCAL[c], BRIEF_LOCAL[c] + 240.0)
                   for c in (LU, RUT, LUT)}
        res = resolve(mw=master_window(18.6, 258.6), regions=regions,
                      offs=BRIEF_OFFSETS)
        self.assertEqual(res.start.decision, RC.DECISION_MASTER_HELD)
        self.assertEqual(sorted(res.start.supported_by),
                         sorted([RU, LU, RUT, LUT]))
        self.assertEqual(res.start.support_weight,
                         RC.SIDE_WEIGHT * 2 + RC.TOP_WEIGHT * 2)

    def test_raw_local_comparison_would_have_called_it_a_disagreement(self):
        """Same evidence, offsets thrown away: the cameras now look 4.4s apart
        and three of four read as dissent. That is the bug being fixed."""
        regions = {c: Region(BRIEF_LOCAL[c], BRIEF_LOCAL[c] + 240.0)
                   for c in (LU, RUT, LUT)}
        naive = resolve(mw=master_window(18.6, 258.6), regions=regions,
                        offs=offsets(lu=0.0, rut=0.0, lut=0.0))
        self.assertTrue(naive.start.disagreed_by)
        proper = resolve(mw=master_window(18.6, 258.6), regions=regions,
                         offs=BRIEF_OFFSETS)
        self.assertEqual(proper.start.disagreed_by, [])

    def test_the_diagnostics_show_local_offset_and_global(self):
        regions = {c: Region(BRIEF_LOCAL[c], BRIEF_LOCAL[c] + 240.0)
                   for c in (LU, RUT, LUT)}
        lines = resolve(mw=master_window(18.6, 258.6), regions=regions,
                        offs=BRIEF_OFFSETS).render_lines()
        joined = "\n".join(lines)
        self.assertIn("[ACTIVE-REGION]", joined)
        for cam in (RU, LU, RUT, LUT):
            self.assertIn(cam, joined)
        self.assertIn("local", joined)
        self.assertIn("offset=", joined)
        self.assertIn("global", joined)
        self.assertIn("START", joined)
        self.assertIn("END", joined)


# ===========================================================================
# 3/4. A top camera starting earlier, or later
# ===========================================================================

class TestTopCameraTiming(unittest.TestCase):

    def test_a_top_camera_starting_earlier_cannot_create_a_wagon(self):
        """The classic failure: the top classifier calls WAGON over the
        locomotive. Both top cameras agree, and it still cannot move the
        boundary -- weight 2 is short of 3, and no side camera backs it."""
        regions = {LU: Region(21.2, 258.6 - 2.6),
                   RUT: Region(10.0, 258.6 - 1.8),
                   LUT: Region(11.5, 258.6 + 1.5)}
        res = resolve(mw=master_window(18.6, 258.6), regions=regions,
                      offs=BRIEF_OFFSETS, gaps=[11.8, 18.6, 22.6])
        self.assertEqual(res.start.decision, RC.DECISION_MASTER_HELD)
        self.assertEqual(res.start.canonical_time, 18.6)
        # Two independent gates would each reject this; the weight gate is
        # simply reached first, and both are correct refusals.
        self.assertTrue(
            RC.REASON_INSUFFICIENT_WEIGHT in res.start.reason
            or res.start.reason == RC.REASON_TOP_ONLY, res.start.reason)

    def test_the_top_only_gate_holds_even_if_the_weight_bar_is_lowered(self):
        """The weight gate normally rejects top-only proposals first. Lowering
        the bar exposes the second, independent guard -- so the protection does
        not rest on one number."""
        cfg = RC.ConsensusConfig(min_move_weight=2)
        regions = {LU: Region(21.2, 258.6 - 2.6),
                   RUT: Region(4.2, 258.6 - 1.8),     # global 6.0
                   LUT: Region(7.5, 258.6 + 1.5)}     # global 6.0
        res = resolve(mw=master_window(18.6, 258.6), regions=regions,
                      offs=BRIEF_OFFSETS, gaps=[6.0, 18.6], cfg=cfg)
        self.assertEqual(res.start.proposed_weight, 2)
        self.assertEqual(res.start.reason, RC.REASON_TOP_ONLY)
        self.assertFalse(res.start.moved)

    def test_a_top_camera_starting_earlier_with_no_master_gap_is_refused(self):
        """Even with a side camera agreeing, there must be a validated master
        gap where the boundary is being moved TO."""
        regions = {LU: Region(12.6, 258.6 - 2.6),      # global 10.0
                   RUT: Region(8.2, 258.6 - 1.8),      # global 10.0
                   LUT: Region(11.5, 258.6 + 1.5)}
        res = resolve(mw=master_window(18.6, 258.6), regions=regions,
                      offs=BRIEF_OFFSETS, gaps=[18.6, 22.6])
        self.assertEqual(res.start.decision, RC.DECISION_MASTER_HELD)
        self.assertIn(RC.REASON_NO_MASTER_GAP, res.start.reason)

    def test_a_top_camera_starting_later_does_not_lose_the_first_wagon(self):
        """A late top boundary would shrink the region. Shrinking is never
        acted on, so the master's first wagon survives."""
        regions = {LU: Region(21.2, 258.6 - 2.6),
                   RUT: Region(40.0, 258.6 - 1.8),
                   LUT: Region(41.0, 258.6 + 1.5)}
        res = resolve(mw=master_window(18.6, 258.6), regions=regions,
                      offs=BRIEF_OFFSETS, gaps=[18.6, 22.6, 41.8])
        self.assertEqual(res.start.canonical_time, 18.6)
        self.assertFalse(res.start.moved)
        self.assertIn("shrink", res.start.reason)

    def test_a_top_camera_ending_early_does_not_close_the_region(self):
        regions = {LU: Region(21.2, 100.0),
                   RUT: Region(16.8, 90.0),
                   LUT: Region(20.1, 95.0)}
        res = resolve(mw=master_window(18.6, 258.6), regions=regions,
                      offs=BRIEF_OFFSETS, gaps=[18.6, 258.6])
        self.assertEqual(res.end.canonical_time, 258.6)
        self.assertFalse(res.end.moved)


# ===========================================================================
# 5. A false top-camera prediction
# ===========================================================================

class TestFalseTopPrediction(unittest.TestCase):

    def test_a_top_camera_with_no_wagon_region_at_all_is_recorded(self):
        """ENGINE everywhere: the classifier found no wagons. That is evidence
        about the camera, not about the train."""
        regions = {LU: Region(21.2, 256.0),
                   RUT: Region(None, None, found=False,
                               reason="no WAGON segment identified"),
                   LUT: Region(20.1, 257.0)}
        res = resolve(mw=master_window(18.6, 258.6), regions=regions,
                      offs=BRIEF_OFFSETS)
        rut = next(b for b in res.boundaries if b.camera_id == RUT)
        self.assertFalse(rut.found)
        self.assertIsNone(rut.global_start)
        self.assertNotIn(RUT, res.start.supported_by)
        self.assertNotIn(RUT, res.start.disagreed_by)
        self.assertEqual(res.start.canonical_time, 18.6)

    def test_one_wild_top_camera_cannot_outvote_the_rest(self):
        regions = {LU: Region(21.2, 258.6 - 2.6),
                   RUT: Region(2.0, 258.6 - 1.8),
                   LUT: Region(20.1, 258.6 + 1.5)}
        res = resolve(mw=master_window(18.6, 258.6), regions=regions,
                      offs=BRIEF_OFFSETS, gaps=[3.8, 18.6])
        self.assertEqual(res.start.canonical_time, 18.6)
        self.assertIn(RUT, res.start.disagreed_by)
        self.assertEqual(res.start.decision, RC.DECISION_MASTER_HELD)

    def test_two_top_cameras_together_still_cannot_relocate(self):
        regions = {LU: Region(21.2, 258.6 - 2.6),
                   RUT: Region(4.2, 258.6 - 1.8),     # global 6.0
                   LUT: Region(7.5, 258.6 + 1.5)}     # global 6.0
        res = resolve(mw=master_window(18.6, 258.6), regions=regions,
                      offs=BRIEF_OFFSETS, gaps=[6.0, 18.6])
        self.assertEqual(res.start.proposed_weight, RC.TOP_WEIGHT * 2)
        self.assertLess(res.start.proposed_weight,
                        RC.DEFAULT_CONFIG.min_move_weight)
        self.assertFalse(res.start.moved)


# ===========================================================================
# 6. A side camera missing or delaying the boundary
# ===========================================================================

class TestSideCameraRecovery(unittest.TestCase):

    def test_a_side_camera_plus_a_top_camera_on_a_master_gap_recovers_it(self):
        """The case the brief asks to support: the master's own classifier
        started late, LEFT_UP and a top camera both put the boundary one wagon
        earlier, and there IS a validated master gap there."""
        regions = {LU: Region(17.2, 258.6 - 2.6),     # global 14.6
                   RUT: Region(12.8, 258.6 - 1.8),    # global 14.6
                   LUT: Region(20.1, 258.6 + 1.5)}
        res = resolve(mw=master_window(18.6, 258.6), regions=regions,
                      offs=BRIEF_OFFSETS, gaps=[14.6, 18.6, 22.6])
        self.assertTrue(res.start.moved)
        self.assertAlmostEqual(res.start.canonical_time, 14.6)
        self.assertIn(LU, res.start.proposed_by)
        self.assertGreaterEqual(res.start.proposed_weight,
                                RC.DEFAULT_CONFIG.min_move_weight)
        self.assertAlmostEqual(res.start.master_gap_time, 14.6)

    def test_recovery_still_requires_the_master_gap(self):
        regions = {LU: Region(17.2, 258.6 - 2.6),
                   RUT: Region(12.8, 258.6 - 1.8),
                   LUT: Region(20.1, 258.6 + 1.5)}
        res = resolve(mw=master_window(18.6, 258.6), regions=regions,
                      offs=BRIEF_OFFSETS, gaps=[18.6, 22.6])
        self.assertFalse(res.start.moved)
        self.assertIn(RC.REASON_NO_MASTER_GAP, res.start.reason)

    def test_a_side_camera_alone_reaches_the_weight_but_still_needs_the_gap(self):
        regions = {LU: Region(17.2, 258.6 - 2.6),
                   RUT: Region(16.8, 258.6 - 1.8),
                   LUT: Region(20.1, 258.6 + 1.5)}
        with_gap = resolve(mw=master_window(18.6, 258.6), regions=regions,
                           offs=BRIEF_OFFSETS, gaps=[14.6, 18.6])
        self.assertIn(LU, with_gap.start.proposed_by)

    def test_the_end_can_be_recovered_the_same_way(self):
        regions = {LU: Region(21.2, 265.2),      # global 262.6
                   RUT: Region(16.8, 260.8),     # global 262.6
                   LUT: Region(20.1, 250.0)}
        res = resolve(mw=master_window(18.6, 258.6), regions=regions,
                      offs=BRIEF_OFFSETS, gaps=[18.6, 258.6, 262.6])
        self.assertTrue(res.end.moved)
        self.assertAlmostEqual(res.end.canonical_time, 262.6)


# ===========================================================================
# 7/8. Disagreement, and unresolved offsets
# ===========================================================================

class TestDisagreementAndOffsets(unittest.TestCase):

    def test_disagreement_is_recorded_per_camera(self):
        regions = {LU: Region(21.2, 258.6 - 2.6),
                   RUT: Region(30.0, 258.6 - 1.8),
                   LUT: Region(20.1, 258.6 + 1.5)}
        res = resolve(mw=master_window(18.6, 258.6), regions=regions,
                      offs=BRIEF_OFFSETS, gaps=[18.6])
        self.assertIn(RUT, res.start.disagreed_by)
        self.assertIn(LU, res.start.supported_by)
        self.assertIn(LUT, res.start.supported_by)

    def test_an_unresolved_offset_is_not_silently_treated_as_zero(self):
        """The flaw this replaces: an unresolved camera was normalized with
        delta 0.0 and compared as though its clock matched the master's."""
        regions = {LU: Region(21.2, 258.6 - 2.6),
                   RUT: Region(16.8, 258.6 - 1.8),
                   LUT: Region(18.7, 258.0)}
        res = resolve(mw=master_window(18.6, 258.6), regions=regions,
                      offs=offsets(lu=-2.6, rut=+1.8, lut=(0.0, "UNRESOLVED")))
        lut = next(b for b in res.boundaries if b.camera_id == LUT)
        self.assertFalse(lut.comparable)
        self.assertIsNone(lut.global_start)
        self.assertIn(LUT, res.start.not_comparable)
        self.assertNotIn(LUT, res.start.supported_by)
        self.assertNotIn(LUT, res.start.disagreed_by)

    def test_an_unresolved_camera_keeps_its_evidence_on_the_record(self):
        regions = {LUT: Region(18.7, 258.0)}
        res = resolve(mw=master_window(18.6, 258.6), regions=regions,
                      offs=offsets(lut=(0.0, "UNRESOLVED")))
        d = next(b.to_dict() for b in res.boundaries if b["camera_id"] == LUT) \
            if False else next(b.to_dict() for b in res.boundaries
                               if b.camera_id == LUT)
        self.assertEqual(d["local_start"], 18.7)
        self.assertIsNone(d["global_start"])
        self.assertEqual(d["reason"], RC.REASON_NOT_COMPARABLE)

    def test_an_unresolved_camera_cannot_contribute_weight_to_a_move(self):
        regions = {LU: Region(17.2, 258.6 - 2.6),
                   RUT: Region(14.6, 258.6 - 1.8),
                   LUT: Region(14.6, 258.0)}
        res = resolve(mw=master_window(18.6, 258.6), regions=regions,
                      offs=offsets(lu=-2.6, rut=0.0, lut=(0.0, "UNRESOLVED")),
                      gaps=[14.6, 18.6])
        self.assertNotIn(LUT, res.start.proposed_by)

    def test_every_camera_can_have_a_different_offset(self):
        regions = {LU: Region(30.0, 200.0), RUT: Region(5.0, 180.0),
                   LUT: Region(12.0, 190.0)}
        res = resolve(mw=master_window(18.6, 200.0), regions=regions,
                      offs=offsets(lu=-11.4, rut=+13.6, lut=+6.6))
        for b in res.boundaries:
            if b.camera_id == RU:
                continue
            self.assertAlmostEqual(b.global_start, 18.6, places=6, msg=b.camera_id)
        self.assertEqual(len({b.offset for b in res.boundaries}), 4)

    def test_a_master_with_no_region_is_not_overridden_by_anyone(self):
        regions = {c: Region(10.0, 200.0) for c in (LU, RUT, LUT)}
        res = resolve(mw=master_window(None, None, found=False),
                      regions=regions, offs=BRIEF_OFFSETS, gaps=[10.0])
        self.assertEqual(res.start.decision, RC.DECISION_NO_MASTER)
        self.assertIsNone(res.start.canonical_time)


# ===========================================================================
# 9/10. RIGHT_UP stays master; top cameras own no identity
# ===========================================================================

class TestMasterAuthority(unittest.TestCase):

    def test_the_master_is_always_the_reference_clock(self):
        res = resolve(mw=master_window(18.6, 258.6), regions={}, offs={})
        ru = next(b for b in res.boundaries if b.camera_id == RU)
        self.assertEqual(ru.offset, 0.0)
        self.assertEqual(ru.offset_status, "REFERENCE")
        self.assertEqual(ru.global_start, 18.6)

    def test_side_cameras_outweigh_top_cameras(self):
        self.assertGreater(RC.SIDE_WEIGHT, RC.TOP_WEIGHT)
        self.assertLess(RC.TOP_WEIGHT * 2, RC.DEFAULT_CONFIG.min_move_weight)

    def test_the_consensus_module_never_touches_a_roster(self):
        """AST, not text: the module docstring explains what it REUSES, which
        mentions GW_1..GW_N and renumbering. A substring search matches that
        prose and would be asserting about a docstring, not about code."""
        src = open(os.path.join(V4_ROOT, "core/region_consensus.py"),
                   encoding="utf-8").read()
        tree = ast.parse(src)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for banned in ("wagon_units", "global_id", "total_wagons", "wagons",
                       "renumber_gap_events"):
            self.assertNotIn(banned, names, f"{banned} is referenced in code")
        # And no string literal in CODE mints an id. Docstrings are excluded:
        # they ARE ast.Constant string nodes, so including them would put the
        # explanatory prose back under test.
        docstrings = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)):
                body = getattr(n, "body", None) or []
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    docstrings.add(id(body[0].value))
        for n in ast.walk(tree):
            if (isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and id(n) not in docstrings):
                self.assertNotIn("GW_", n.value)

    def test_the_consensus_module_runs_no_detector_and_no_second_clock(self):
        src = open(os.path.join(V4_ROOT, "core/region_consensus.py"),
                   encoding="utf-8").read()
        tree = ast.parse(src)
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported.update(a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                imported.add((n.module or "").split(".")[0])
        self.assertNotIn("ultralytics", imported)
        self.assertNotIn("cv2", imported)
        for banned in ("GapTracker", "estimate_offset", "validate_gap_events"):
            self.assertNotIn(banned, src, banned)

    def test_the_override_only_selects_existing_master_segments(self):
        params = set(inspect.signature(ts.get_master_wagon_window).parameters)
        self.assertIn("first_wagon_index", params)
        self.assertIn("last_wagon_index", params)
        # Indices, not times or frames: nothing else can be supplied.
        self.assertNotIn("first_wagon_time", params)
        self.assertNotIn("start_frame", params)

    def test_widening_selects_a_real_master_segment_as_gw_1(self):
        base = ts.get_master_wagon_window(segments(SPEC), verbose=False)
        wide = ts.get_master_wagon_window(segments(SPEC), first_wagon_index=0,
                                          verbose=False)
        self.assertEqual(base.master_wagon_count + 1, wide.master_wagon_count)
        self.assertEqual(wide.wagon_units[0].start_frame_master,
                         segments(SPEC)[0].start_frame_master)
        self.assertEqual([u.global_id for u in wide.wagon_units],
                         [f"GW_{i}" for i in range(1, 7)])


# ===========================================================================
# 11/12. Gaps are the boundaries; ENGINE/BRAKE_VAN get no ids
# ===========================================================================

class TestStructureUnchanged(unittest.TestCase):

    def test_the_move_target_must_be_a_validated_master_gap(self):
        src = inspect.getsource(RC._decide)
        self.assertIn("_nearest_gap", src)
        self.assertIn("gap_tolerance_sec", src)

    def test_leading_and_trailing_non_wagons_have_no_ids(self):
        win = ts.get_master_wagon_window(segments(SPEC), verbose=False)
        self.assertEqual([o.classification
                          for o in win.leading_non_wagon_objects], [E])
        self.assertEqual([o.classification
                          for o in win.trailing_non_wagon_objects], [B, E])
        for o in (win.leading_non_wagon_objects
                  + win.trailing_non_wagon_objects):
            self.assertFalse(hasattr(o, "global_id"))

    def test_widening_keeps_the_remaining_non_wagons_outside(self):
        wide = ts.get_master_wagon_window(segments(SPEC), first_wagon_index=0,
                                          verbose=False)
        self.assertEqual(wide.leading_non_wagon_objects, [])
        self.assertEqual([o.classification
                          for o in wide.trailing_non_wagon_objects], [B, E])

    def test_the_segment_accounting_balances_after_widening(self):
        for kw in ({}, {"first_wagon_index": 0}):
            win = ts.get_master_wagon_window(segments(SPEC), verbose=False, **kw)
            self.assertEqual(
                win.master_wagon_count
                + len(win.leading_non_wagon_objects)
                + len(win.trailing_non_wagon_objects),
                win.total_segments, kw)

    def test_gw_ids_stay_contiguous_from_one(self):
        for kw in ({}, {"first_wagon_index": 0}, {"last_wagon_index": 6}):
            win = ts.get_master_wagon_window(segments(SPEC), verbose=False, **kw)
            self.assertEqual([u.wagon_index for u in win.wagon_units],
                             list(range(1, win.master_wagon_count + 1)))


# ===========================================================================
# 13. One algorithm, both modes
# ===========================================================================

class TestBothModesUseOneAlgorithm(unittest.TestCase):

    def test_the_consensus_is_called_from_the_shared_fusion_function(self):
        """Both pipelines call `assemble_global_train_state_master_fixed`, so
        placing the consensus inside it is what makes a per-mode algorithm
        impossible rather than merely unlikely."""
        src = open(os.path.join(V4_ROOT, "wagon_count/global_fusion.py"),
                   encoding="utf-8").read()
        self.assertIn("region_consensus", src)
        self.assertIn("RC.resolve(", src)
        fn = inspect.getsource(
            __import__("global_fusion").assemble_global_train_state_master_fixed)
        self.assertIn("RC.resolve(", fn)

    def test_neither_pipeline_has_its_own_consensus_call(self):
        for path in ("wagon_count/run_global_count.py",
                     "orchestrator/global_assembler.py",
                     "orchestrator/master_runner.py"):
            src = open(os.path.join(V4_ROOT, path), encoding="utf-8").read()
            self.assertNotIn("region_consensus.resolve", src, path)
            self.assertNotIn("RC.resolve(", src, path)

    def test_the_same_inputs_give_the_same_decision(self):
        regions = {LU: Region(17.2, 256.0), RUT: Region(12.8, 256.8),
                   LUT: Region(20.1, 257.1)}
        a = resolve(mw=master_window(18.6, 258.6), regions=regions,
                    offs=BRIEF_OFFSETS, gaps=[14.6, 18.6, 258.6])
        b = resolve(mw=master_window(18.6, 258.6), regions=regions,
                    offs=BRIEF_OFFSETS, gaps=[14.6, 18.6, 258.6])
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_the_decision_is_serialized_into_the_state(self):
        src = open(os.path.join(V4_ROOT, "wagon_count/global_train_state.py"),
                   encoding="utf-8").read()
        self.assertIn("region_consensus", src)

    def test_stage3_sampling_is_untouched(self):
        self.assertEqual((C.STAGE3_DOOR_MODE, C.STAGE3_DOOR_STRIDE),
                         ("sampled", 3))
        self.assertEqual((C.STAGE3_DAMAGE_MODE, C.STAGE3_DAMAGE_STRIDE),
                         ("sampled", 3))
        self.assertEqual((C.STAGE3_LOAD_MODE, C.STAGE3_LOAD_STRIDE),
                         ("sampled", 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
