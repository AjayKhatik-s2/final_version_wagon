"""A damage detection's wagon comes from normalized time and canonical gaps.

The defect these tests pin: the damage processor never chose a wagon number. It
read `wagon_cache/GW_n/<camera>/*.jpg` and inherited `GW_n` from the directory,
and those directories are cut at `round((GW.time - delta) * local_fps)` with
`delta = 0.0` for every camera whose clock offset the counter could not resolve.
On a displaced camera that files one wagon's frames under its neighbour, and
every damage seen in them is reported against the wrong wagon -- silently,
because a directory name carries no provenance.

The rule under test, on the worked example the spec names::

        GW_25            GAP_25            GW_26
    ...............|=================|...............
       BEFORE_GAP        ambiguous        AFTER_GAP
        -> GW_25         -> neither        -> GW_26

Every fixture here therefore works in SECONDS on the master clock, gives each
camera a different offset and lets the resolver normalize -- because a local
frame number means four different things on four cameras, which is the whole
reason the association exists.
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest
from dataclasses import dataclass

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from core import canonical_association as CA
from core.global_state_loader import GlobalTrainState, GlobalWagon
from orchestrator import damage_association as DA

RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP
RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP

FPS = 15.0
WAGON_S = 4.0            # each wagon occupies 4 s of master time
N_WAGONS = 30

#: The boundary the spec names. With 4 s wagons starting at t=0, GW_25 spans
#: 96-100 s and GW_26 spans 100-104 s, so GAP_25 sits at t=100.
GAP_25_TIME = 25 * WAGON_S
GW_25, GW_26 = "GW_25", "GW_26"

#: Offsets chosen so no two cameras agree and none is zero except the master:
#: a test where every camera shares a clock cannot fail the way production did.
OFFSETS = {
    RU:  ("REFERENCE", 0.0),
    LU:  ("RESOLVED", 1.25),
    RUT: ("RESOLVED", 2.0),
    LUT: ("RESOLVED", -3.5),
}


# ---------------------------------------------------------------------------
# Fixtures -- canonical roster and canonical gaps, nothing else
# ---------------------------------------------------------------------------

def _wagons(n: int = N_WAGONS):
    return tuple(
        GlobalWagon(
            global_id=f"GW_{i}", wagon_index=i,
            start_frame_master=int(round((i - 1) * WAGON_S * FPS)),
            end_frame_master=int(round(i * WAGON_S * FPS)) - 1,
            start_time=(i - 1) * WAGON_S, end_time=i * WAGON_S,
            classification=C.CLASS_WAGON, classification_confidence=0.95,
        )
        for i in range(1, n + 1)
    )


def _state(wagons=None, offsets=None) -> GlobalTrainState:
    ws = wagons if wagons is not None else _wagons()
    offs = offsets if offsets is not None else OFFSETS
    return GlobalTrainState(
        total_wagons=len(ws), wagons=ws, master_camera=RU, master_fps=FPS,
        master_total_frames=int(round(N_WAGONS * WAGON_S * FPS)),
        camera_offsets={cam: {"status": st, "delta": d}
                        for cam, (st, d) in offs.items()},
    )


def _global_gaps(n: int = N_WAGONS - 1, *, span_frames: int = 12,
                 without_timing=()):
    """The engine's canonical gap sequence: gap k sits at the GW_k/GW_k+1 join."""
    out = []
    for k in range(1, n + 1):
        t = None if k in without_timing else k * WAGON_S
        out.append({
            "global_gap_id": k,
            "master_camera": RU,
            "master_frame": (None if t is None else int(round(t * FPS))),
            "master_time": t,
            "master_track_id": 100 + k,
            "support_observations": {
                cam: {"camera_id": cam, "local_track_id": 200 + k,
                      "local_frame": (None if t is None
                                      else int(round((t - d) * FPS))),
                      "local_time": (None if t is None else t - d),
                      "fps": FPS, "span_frames": span_frames}
                for cam, (_st, d) in OFFSETS.items()
            },
        })
    return out


def _timeline(*, state=None, gaps=None, fps=None, cfg=CA.DEFAULT_CONFIG):
    return CA.CanonicalTimeline.build(
        state=state or _state(),
        global_gaps=gaps if gaps is not None else _global_gaps(),
        per_camera_fps=fps if fps is not None else {c: FPS for c in C.ALL_CAMERAS},
        cfg=cfg,
    )


def _local_frame(camera_id: str, t_global: float) -> int:
    """The frame on `camera_id` that shows master-clock time `t_global`.

    Inverse of the resolver's own normalization, so a test states the physical
    moment it means and the fixture works out each camera's frame number -- not
    the other way round.
    """
    delta = OFFSETS[camera_id][1]
    return int(round((t_global - delta) * FPS))


def _assign(camera_id: str, t_global: float, *, timeline=None, **kw):
    tl = timeline or _timeline(**kw)
    return tl.assign(CA.Detection(
        camera_id=camera_id, local_frame=_local_frame(camera_id, t_global),
        feature="damage", detection_id=f"{camera_id}@{t_global}"))


# ===========================================================================
# 1. BEFORE_GAP -> GW_25
# ===========================================================================

class TestBeforeTheGapBelongsToTheEarlierWagon(unittest.TestCase):

    def test_a_detection_before_gap_25_is_gw_25(self):
        a = _assign(RUT, GAP_25_TIME - 1.5)
        self.assertEqual(a.status, CA.STATUS_RESOLVED)
        self.assertEqual(a.method, CA.METHOD_BEFORE_GAP)
        self.assertEqual(a.global_wagon_id, GW_25)
        self.assertEqual(a.associated_global_gap_id, 25)

    def test_the_assignment_equals_the_previous_wagon_of_that_gap(self):
        """A BEFORE_GAP verdict must be self-consistent: the wagon chosen IS the
        one the provenance names on the near side of the gap."""
        a = _assign(RUT, GAP_25_TIME - 1.5)
        self.assertEqual(a.previous_wagon_id, GW_25)
        self.assertEqual(a.next_wagon_id, GW_26)
        self.assertEqual(a.global_wagon_id, a.previous_wagon_id)

    def test_every_camera_agrees_before_the_gap(self):
        for cam in C.ALL_CAMERAS:
            a = _assign(cam, GAP_25_TIME - 1.5)
            self.assertEqual(a.global_wagon_id, GW_25, cam)
            self.assertEqual(a.method, CA.METHOD_BEFORE_GAP, cam)

    def test_it_holds_at_another_boundary_too(self):
        """Not a coincidence of the number 25."""
        for k in (3, 12, 25, 29):
            a = _assign(LUT, k * WAGON_S - 1.5)
            self.assertEqual(a.global_wagon_id, f"GW_{k}")
            self.assertEqual(a.associated_global_gap_id, k)
            self.assertEqual(a.method, CA.METHOD_BEFORE_GAP)


# ===========================================================================
# 2. AFTER_GAP -> GW_26
# ===========================================================================

class TestAfterTheGapBelongsToTheNextWagon(unittest.TestCase):

    def test_a_detection_after_gap_25_is_gw_26(self):
        a = _assign(RUT, GAP_25_TIME + 1.5)
        self.assertEqual(a.status, CA.STATUS_RESOLVED)
        self.assertEqual(a.method, CA.METHOD_AFTER_GAP)
        self.assertEqual(a.global_wagon_id, GW_26)
        self.assertEqual(a.associated_global_gap_id, 25)

    def test_the_assignment_equals_the_next_wagon_of_that_gap(self):
        a = _assign(RUT, GAP_25_TIME + 1.5)
        self.assertEqual(a.previous_wagon_id, GW_25)
        self.assertEqual(a.next_wagon_id, GW_26)
        self.assertEqual(a.global_wagon_id, a.next_wagon_id)

    def test_the_same_gap_id_separates_both_verdicts(self):
        """Before and after must name the SAME physical gap -- that is what
        makes the pair a boundary rather than two unrelated findings."""
        before = _assign(RUT, GAP_25_TIME - 1.5)
        after = _assign(RUT, GAP_25_TIME + 1.5)
        self.assertEqual(before.associated_global_gap_id,
                         after.associated_global_gap_id)
        self.assertEqual((before.global_wagon_id, after.global_wagon_id),
                         (GW_25, GW_26))

    def test_every_camera_agrees_after_the_gap(self):
        for cam in C.ALL_CAMERAS:
            a = _assign(cam, GAP_25_TIME + 1.5)
            self.assertEqual(a.global_wagon_id, GW_26, cam)


# ===========================================================================
# 3. Cross-camera: different offsets, same canonical wagon
# ===========================================================================

class TestDifferentCamerasNormalizeToTheSameWagon(unittest.TestCase):
    """The point of normalizing. One physical defect, four cameras, four
    different local frame numbers, one `GW_n`."""

    def test_one_physical_moment_gives_one_wagon_on_all_four_cameras(self):
        t = GAP_25_TIME - 2.0            # mid-GW_25
        tl = _timeline()
        got = {}
        for cam in C.ALL_CAMERAS:
            a = tl.assign(CA.Detection(camera_id=cam,
                                       local_frame=_local_frame(cam, t),
                                       feature="damage"))
            got[cam] = a.global_wagon_id
        self.assertEqual(set(got.values()), {GW_25}, got)

    def test_the_local_frame_numbers_really_do_differ(self):
        """Guards the fixture: if every camera used the same frame number the
        test above would pass without exercising normalization at all."""
        t = GAP_25_TIME - 2.0
        frames = {cam: _local_frame(cam, t) for cam in C.ALL_CAMERAS}
        self.assertEqual(len(set(frames.values())), len(C.ALL_CAMERAS), frames)

    def test_using_the_local_frame_unnormalized_would_get_it_wrong(self):
        """Proves the bug is real: feed LEFT_UP_TOP's frame number through as if
        it were the master's and a different wagon comes out."""
        t = GAP_25_TIME - 2.0
        tl = _timeline()
        naive = tl.wagon_at(_local_frame(LUT, t) / FPS)   # no offset applied
        correct = tl.assign(CA.Detection(camera_id=LUT,
                                         local_frame=_local_frame(LUT, t),
                                         feature="damage")).global_wagon_id
        self.assertEqual(correct, GW_25)
        self.assertNotEqual(naive, correct)

    def test_two_cameras_seeing_across_the_gap_split_correctly(self):
        tl = _timeline()
        a = tl.assign(CA.Detection(camera_id=RUT, feature="damage",
                                   local_frame=_local_frame(RUT, GAP_25_TIME - 1.6)))
        b = tl.assign(CA.Detection(camera_id=LUT, feature="damage",
                                   local_frame=_local_frame(LUT, GAP_25_TIME + 1.6)))
        self.assertEqual((a.global_wagon_id, b.global_wagon_id), (GW_25, GW_26))

    def test_an_unresolved_offset_is_flagged_not_silently_assumed(self):
        st = _state(offsets={**{c: OFFSETS[c] for c in (RU, LU, RUT)},
                             LUT: ("UNRESOLVED", 0.0)})
        tl = _timeline(state=st)
        a = tl.assign(CA.Detection(camera_id=LUT, feature="damage",
                                   local_frame=int(round((GAP_25_TIME - 2.0) * FPS))))
        self.assertEqual(a.status, CA.STATUS_RESOLVED_ASSUMED_OFFSET)
        self.assertEqual(a.global_wagon_id, GW_25)
        self.assertEqual(a.camera_time_offset, 0.0)
        self.assertIn("shared-t=0", a.reason)

    def test_an_unresolved_offset_widens_the_ambiguity_band(self):
        """A less certain time base must fall to AMBIGUOUS more readily, not
        less."""
        resolved = _timeline()
        st = _state(offsets={**{c: OFFSETS[c] for c in (RU, LU, RUT)},
                             LUT: ("UNRESOLVED", 0.0)})
        assumed = _timeline(state=st)
        f = int(round((GAP_25_TIME - 2.0) * FPS))
        d = CA.Detection(camera_id=LUT, local_frame=f, feature="damage")
        self.assertGreater(assumed.assign(d).tolerance_s,
                           resolved.assign(CA.Detection(
                               camera_id=LUT, feature="damage",
                               local_frame=_local_frame(LUT, GAP_25_TIME - 2.0),
                           )).tolerance_s)


# ===========================================================================
# 4. Boundary tolerance -> BOUNDARY_AMBIGUOUS
# ===========================================================================

class TestNearTheBoundaryIsAmbiguousNotGuessed(unittest.TestCase):

    def test_exactly_on_the_gap_is_ambiguous(self):
        a = _assign(RUT, GAP_25_TIME)
        self.assertEqual(a.status, CA.STATUS_BOUNDARY_AMBIGUOUS)
        self.assertIsNone(a.global_wagon_id)

    def test_an_ambiguous_verdict_still_names_both_candidates(self):
        """"We could not tell" must say between WHICH two -- otherwise the
        finding is unusable downstream."""
        a = _assign(RUT, GAP_25_TIME)
        self.assertEqual(a.previous_wagon_id, GW_25)
        self.assertEqual(a.next_wagon_id, GW_26)
        self.assertEqual(a.associated_global_gap_id, 25)
        self.assertEqual(a.confidence, 0.0)

    def test_just_inside_the_tolerance_is_ambiguous_on_both_sides(self):
        tol = _assign(RUT, GAP_25_TIME).tolerance_s
        for signed in (-tol * 0.9, +tol * 0.9):
            a = _assign(RUT, GAP_25_TIME + signed)
            self.assertEqual(a.status, CA.STATUS_BOUNDARY_AMBIGUOUS,
                             f"offset {signed:+.3f}s")
            self.assertIsNone(a.global_wagon_id)

    def test_just_outside_the_tolerance_resolves(self):
        tol = _assign(RUT, GAP_25_TIME).tolerance_s
        self.assertEqual(_assign(RUT, GAP_25_TIME - tol * 1.5).global_wagon_id,
                         GW_25)
        self.assertEqual(_assign(RUT, GAP_25_TIME + tol * 1.5).global_wagon_id,
                         GW_26)

    def test_the_tolerance_comes_from_the_gaps_own_visible_span(self):
        """Not a magic number: a gap visible for twice as long is ambiguous over
        twice the band."""
        narrow = _timeline(gaps=_global_gaps(span_frames=6))
        wide = _timeline(gaps=_global_gaps(span_frames=24))
        d = CA.Detection(camera_id=RUT, feature="damage",
                         local_frame=_local_frame(RUT, GAP_25_TIME))
        self.assertLess(narrow.assign(d).tolerance_s, wide.assign(d).tolerance_s)

    def test_the_tolerance_is_clamped_to_the_shared_fusion_bounds(self):
        """Same floor and cap as `global_fusion`'s own timing sigma, so the
        repository has one convention for this quantity rather than two."""
        cfg = CA.DEFAULT_CONFIG
        tiny = _timeline(gaps=_global_gaps(span_frames=1))
        huge = _timeline(gaps=_global_gaps(span_frames=100000))
        d = CA.Detection(camera_id=RUT, feature="damage",
                         local_frame=_local_frame(RUT, GAP_25_TIME))
        self.assertGreaterEqual(tiny.assign(d).tolerance_s, cfg.tolerance_floor_s)
        self.assertLessEqual(huge.assign(d).tolerance_s,
                             cfg.tolerance_cap_s
                             * cfg.assumed_offset_tolerance_factor)

    def test_confidence_grows_with_distance_from_the_boundary(self):
        near = _assign(RUT, GAP_25_TIME - 0.5)
        far = _assign(RUT, GAP_25_TIME - 2.0)
        self.assertLessEqual(near.confidence, far.confidence)
        self.assertEqual(far.confidence, 1.0)


# ===========================================================================
# 5. Missing gap timing -> unresolved, never guessed
# ===========================================================================

class TestMissingGapTimingIsUnresolved(unittest.TestCase):

    def test_a_gap_without_timing_makes_its_interval_unresolved(self):
        tl = _timeline(gaps=_global_gaps(without_timing=(25,)))
        for t in (GAP_25_TIME - 1.5, GAP_25_TIME + 1.5):
            a = tl.assign(CA.Detection(camera_id=RUT, feature="damage",
                                       local_frame=_local_frame(RUT, t)))
            self.assertEqual(a.status, CA.STATUS_UNRESOLVED_NO_GAP_TIMING)
            self.assertIsNone(a.global_wagon_id)

    def test_the_reason_names_the_offending_gap(self):
        tl = _timeline(gaps=_global_gaps(without_timing=(25,)))
        a = tl.assign(CA.Detection(camera_id=RUT, feature="damage",
                                   local_frame=_local_frame(RUT, GAP_25_TIME - 1.5)))
        self.assertIn("GAP_25", a.reason)

    def test_the_gap_is_kept_in_the_audit_not_dropped_silently(self):
        tl = _timeline(gaps=_global_gaps(without_timing=(25,)))
        self.assertEqual(tl.summary()["gaps_without_timing"], [25])
        self.assertEqual(tl.summary()["canonical_gaps"], N_WAGONS - 1)
        self.assertEqual(tl.summary()["usable_gap_boundaries"], N_WAGONS - 2)

    def test_other_wagons_are_unaffected_by_one_bad_gap(self):
        tl = _timeline(gaps=_global_gaps(without_timing=(25,)))
        a = tl.assign(CA.Detection(camera_id=RUT, feature="damage",
                                   local_frame=_local_frame(RUT, 5 * WAGON_S - 1.5)))
        self.assertEqual(a.global_wagon_id, "GW_5")
        self.assertEqual(a.status, CA.STATUS_RESOLVED)

    def test_a_non_numeric_master_time_is_treated_as_missing(self):
        gaps = _global_gaps()
        gaps[24]["master_time"] = "not a number"
        tl = _timeline(gaps=gaps)
        a = tl.assign(CA.Detection(camera_id=RUT, feature="damage",
                                   local_frame=_local_frame(RUT, GAP_25_TIME - 1.5)))
        self.assertEqual(a.status, CA.STATUS_UNRESOLVED_NO_GAP_TIMING)

    def test_no_usable_gaps_at_all_is_unresolved_not_gw_1(self):
        tl = _timeline(gaps=[])
        a = tl.assign(CA.Detection(camera_id=RUT, local_frame=100,
                                   feature="damage"))
        self.assertEqual(a.status, CA.STATUS_UNRESOLVED_NO_CANONICAL_GAPS)
        self.assertIsNone(a.global_wagon_id)

    def test_a_camera_without_fps_cannot_be_normalized(self):
        tl = _timeline(fps={RU: FPS})
        a = tl.assign(CA.Detection(camera_id=RUT, local_frame=100,
                                   feature="damage"))
        self.assertEqual(a.status, CA.STATUS_UNRESOLVED_NO_CAMERA_FPS)
        self.assertIsNone(a.global_wagon_id)

    def test_a_time_outside_the_counted_train_is_not_pinned_to_the_last_wagon(self):
        """The trailing interval is open-ended, so on the bare interval rule a
        detection minutes past the brake van would still land in the last
        wagon."""
        a = _assign(RUT, N_WAGONS * WAGON_S + 60.0)
        self.assertEqual(a.status, CA.STATUS_UNRESOLVED_OUTSIDE_WAGON_REGION)
        self.assertIsNone(a.global_wagon_id)

    def test_no_unresolved_status_ever_carries_a_wagon(self):
        for status in CA.UNRESOLVED_STATUSES:
            self.assertNotIn(status, CA.RESOLVED_STATUSES)


# ===========================================================================
# 6. The canonical roster is never touched
# ===========================================================================

class TestTheCanonicalRosterIsImmutable(unittest.TestCase):

    @staticmethod
    def _fingerprint(state):
        return [(w.global_id, w.wagon_index, w.start_time, w.end_time,
                 w.classification) for w in state.wagons]

    def test_resolving_a_whole_train_changes_no_wagon(self):
        st = _state()
        before = self._fingerprint(st)
        DA.resolve_train(
            state=st, global_gaps=_global_gaps(),
            per_camera_fps={c: FPS for c in C.ALL_CAMERAS},
            damage_by_wagon={GW_25: [_rec(RUT, GAP_25_TIME + 1.5)]},
            verbose=False)
        self.assertEqual(self._fingerprint(st), before)
        self.assertEqual(st.total_wagons, N_WAGONS)

    def test_the_wagon_count_is_unchanged_by_a_reassignment(self):
        st = _state()
        res = DA.resolve_train(
            state=st, global_gaps=_global_gaps(),
            per_camera_fps={c: FPS for c in C.ALL_CAMERAS},
            damage_by_wagon={GW_25: [_rec(RUT, GAP_25_TIME + 1.5)]},
            verbose=False)
        self.assertEqual(res.moved, 1)
        self.assertEqual(len(st.wagons), N_WAGONS)
        self.assertEqual([w.global_id for w in st.wagons][:3],
                         ["GW_1", "GW_2", "GW_3"])

    def test_the_resolver_can_only_name_wagons_that_already_exist(self):
        st = _state()
        ids = {w.global_id for w in st.wagons}
        tl = _timeline(state=st)
        for k in range(0, N_WAGONS * 4):
            a = tl.assign(CA.Detection(camera_id=RUT, feature="damage",
                                       local_frame=int(k * FPS)))
            if a.global_wagon_id is not None:
                self.assertIn(a.global_wagon_id, ids)

    def test_the_global_wagon_dataclass_is_still_frozen(self):
        """A resolver that could mutate a wagon would not need to be audited --
        it would just be wrong. The type prevents it."""
        with self.assertRaises(Exception):
            _wagons()[0].start_time = 1.0

    def test_applying_assignments_does_not_write_the_roster(self):
        with tempfile.TemporaryDirectory() as tmp:
            st, res = _apply_in_tmp(tmp, {GW_25: [_rec(RUT, GAP_25_TIME + 1.5)]})
            self.assertEqual(st.total_wagons, N_WAGONS)
            self.assertEqual(len(st.wagons), N_WAGONS)
            self.assertFalse(os.path.exists(
                os.path.join(tmp, "global_state", "global_train_state.json")))
            self.assertGreaterEqual(res.moved, 1)


# ---------------------------------------------------------------------------
# Damage-record fixtures for the apply-side tests
# ---------------------------------------------------------------------------

def _rec(camera_id: str, t_global: float, *, track_id: int = 7,
         class_name: str = "floor_damage", track_idx: int = 1):
    """One damage track record, exactly as the damage processor writes it."""
    return {
        "track_idx": track_idx, "camera_id": camera_id, "track_id": track_id,
        "class_name": class_name, "confidence": 0.81, "best_confidence": 0.88,
        "best_frame_idx": _local_frame(camera_id, t_global),
        "bbox": [420.0, 210.0, 520.0, 300.0],
    }


def _apply_in_tmp(tmp: str, damage_by_wagon, *, gaps=None, state=None):
    """Lay out a real wagon_states/evidence tree, then resolve and apply."""
    st = state or _state()
    states_root = os.path.join(tmp, "wagon_states")
    evidence_root = os.path.join(tmp, "evidence")
    dmg_dir = os.path.join(states_root, "damage")
    os.makedirs(dmg_dir, exist_ok=True)
    for w in st.wagons:
        recs = list(damage_by_wagon.get(w.global_id) or [])
        doc = {"global_id": w.global_id, "feature": "damage",
               "status": C.STATUS_OK,
               "top_damage": C.DAMAGE_PRESENT if recs else C.DAMAGE_OK,
               "top_damage_details": recs, "per_camera": {},
               "supporting_cameras": sorted({r["camera_id"] for r in recs}),
               "frame_count": 60, "evidence": {}}
        with open(os.path.join(dmg_dir, f"{w.global_id}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(doc, f)
        if recs:
            ev = os.path.join(evidence_root, w.global_id, "damage")
            os.makedirs(ev, exist_ok=True)
            with open(os.path.join(ev, "metadata.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"global_id": w.global_id, "feature": "damage",
                           "top_damage": C.DAMAGE_PRESENT,
                           "tracks": [dict(r) for r in recs]}, f)
            for r in recs:
                from core.evidence_identity import damage_track_slot
                slot = damage_track_slot(int(r["track_idx"]), r["camera_id"])
                with open(os.path.join(ev, f"{slot}.jpg"), "wb") as f:
                    f.write(b"\xff\xd8\xff\xd9")

    res = DA.run(state=st, global_gaps=gaps if gaps is not None else _global_gaps(),
                 per_camera_fps={c: FPS for c in C.ALL_CAMERAS},
                 states_root=states_root, evidence_root=evidence_root,
                 diagnostics_dir=os.path.join(tmp, "global_state_diag"),
                 verbose=False)
    return st, res


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# 7. The PDF and the processed video see the resolved GW_n
# ===========================================================================

class TestReportAndVideoUseTheResolvedWagon(unittest.TestCase):
    """An association nobody downstream can see is not an association. Both
    readers are keyed by the wagon DIRECTORY, so the apply step has to move the
    record and the evidence metadata together."""

    def test_the_record_lands_in_the_resolved_wagons_state_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            _apply_in_tmp(tmp, {GW_25: [_rec(RUT, GAP_25_TIME + 1.5)]})
            dmg = os.path.join(tmp, "wagon_states", "damage")
            self.assertEqual(_read(os.path.join(dmg, "GW_25.json"))
                             ["top_damage_details"], [])
            moved = _read(os.path.join(dmg, "GW_26.json"))["top_damage_details"]
            self.assertEqual(len(moved), 1)
            self.assertEqual(moved[0]["moved_from_global_id"], GW_25)
            self.assertEqual(moved[0]["moved_by"], "canonical_association")

    def test_top_damage_follows_the_move_on_both_wagons(self):
        with tempfile.TemporaryDirectory() as tmp:
            _apply_in_tmp(tmp, {GW_25: [_rec(RUT, GAP_25_TIME + 1.5)]})
            dmg = os.path.join(tmp, "wagon_states", "damage")
            self.assertEqual(_read(os.path.join(dmg, "GW_25.json"))["top_damage"],
                             C.DAMAGE_OK)
            self.assertEqual(_read(os.path.join(dmg, "GW_26.json"))["top_damage"],
                             C.DAMAGE_PRESENT)

    def test_the_combined_pdf_reads_the_damage_on_the_resolved_wagon(self):
        """Through `reporting.wagon_evidence_grid.damage_from_evidence`, the
        function the combined PDF's damage section actually calls."""
        from reporting.wagon_evidence_grid import damage_from_evidence
        with tempfile.TemporaryDirectory() as tmp:
            st, _ = _apply_in_tmp(tmp, {GW_25: [_rec(RUT, GAP_25_TIME + 1.5)]})
            rows = damage_from_evidence(evidence_root=os.path.join(tmp, "evidence"),
                                        state=st, verbose=False)
            self.assertFalse(rows.get(GW_25), "the PDF still shows it on GW_25")
            self.assertTrue(rows.get(GW_26), "the PDF does not show it on GW_26")

    def test_the_moved_snapshot_is_reachable_from_the_new_owner(self):
        """A damage row whose picture cannot be found publishes with no image --
        a failure this pipeline has already had in production."""
        with tempfile.TemporaryDirectory() as tmp:
            _apply_in_tmp(tmp, {GW_25: [_rec(RUT, GAP_25_TIME + 1.5)]})
            ev26 = os.path.join(tmp, "evidence", GW_26, "damage")
            jpgs = [f for f in os.listdir(ev26) if f.endswith(".jpg")]
            self.assertTrue(jpgs, "no snapshot in the resolved wagon's evidence")
            self.assertTrue(any(RUT in f for f in jpgs),
                            f"snapshot is not camera-scoped: {jpgs}")

    def test_the_moved_track_keeps_a_usable_track_idx(self):
        with tempfile.TemporaryDirectory() as tmp:
            _apply_in_tmp(tmp, {GW_25: [_rec(RUT, GAP_25_TIME + 1.5)]})
            meta = _read(os.path.join(tmp, "evidence", GW_26, "damage",
                                      "metadata.json"))
            for t in meta["tracks"]:
                self.assertIsNotNone(t.get("track_idx"))

    def test_the_processed_video_labels_the_frame_with_the_same_wagon(self):
        """The renderer names a frame's wagon with
        `round((GW.time - delta) * fps)`, the inverse of the resolver's
        normalization. Outside the ambiguity band the two must agree exactly --
        otherwise the video and the report would caption one defect with two
        different wagons."""
        from rendering.feature_overlay_renderer import _map_wagon_to_local_frames
        st = _state()
        tl = _timeline(state=st)
        total = int(round(N_WAGONS * WAGON_S * FPS)) + 200
        for cam in C.ALL_CAMERAS:
            delta = OFFSETS[cam][1]
            frame_to_wagon = {}
            for w in st.wagons:
                sf, ef = _map_wagon_to_local_frames(w, FPS, total, delta)
                if ef >= sf:
                    for f in range(sf, ef + 1):
                        frame_to_wagon[f] = w.global_id
            checked = 0
            for t in [k * WAGON_S + 2.0 for k in range(N_WAGONS)]:
                f = _local_frame(cam, t)
                a = tl.assign(CA.Detection(camera_id=cam, local_frame=f,
                                           feature="damage"))
                if not a.resolved or f not in frame_to_wagon:
                    continue
                self.assertEqual(a.global_wagon_id, frame_to_wagon[f],
                                 f"{cam} frame {f}: video says "
                                 f"{frame_to_wagon[f]}, association says "
                                 f"{a.global_wagon_id}")
                checked += 1
            self.assertGreater(checked, 5, f"{cam}: nothing compared")

    def test_an_ambiguous_detection_is_not_moved_but_is_annotated(self):
        with tempfile.TemporaryDirectory() as tmp:
            _apply_in_tmp(tmp, {GW_25: [_rec(RUT, GAP_25_TIME)]})
            dmg = os.path.join(tmp, "wagon_states", "damage")
            recs = _read(os.path.join(dmg, "GW_25.json"))["top_damage_details"]
            self.assertEqual(len(recs), 1, "an ambiguous detection was moved")
            prov = recs[0][DA.PROVENANCE_KEY]
            self.assertEqual(prov["status"], CA.STATUS_BOUNDARY_AMBIGUOUS)
            self.assertEqual(prov["previous_wagon_id"], GW_25)
            self.assertEqual(prov["next_wagon_id"], GW_26)

    def test_an_unresolved_detection_is_preserved_where_it_was(self):
        """Evidence is never discarded because its wagon could not be decided."""
        with tempfile.TemporaryDirectory() as tmp:
            _apply_in_tmp(tmp, {GW_25: [_rec(RUT, GAP_25_TIME + 1.5)]},
                          gaps=_global_gaps(without_timing=(25,)))
            recs = _read(os.path.join(tmp, "wagon_states", "damage",
                                      "GW_25.json"))["top_damage_details"]
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0][DA.PROVENANCE_KEY]["status"],
                             CA.STATUS_UNRESOLVED_NO_GAP_TIMING)


# ===========================================================================
# 8. Provenance completeness
# ===========================================================================

class TestProvenanceIsComplete(unittest.TestCase):

    #: Every field the spec requires on an assignment.
    REQUIRED = ("global_wagon_id", "camera_id", "local_frame", "local_time",
                "global_time", "associated_global_gap_id", "previous_wagon_id",
                "next_wagon_id", "method", "confidence", "status")

    def test_every_required_field_is_present(self):
        d = _assign(RUT, GAP_25_TIME - 1.5).to_dict()
        for k in self.REQUIRED:
            self.assertIn(k, d)

    def test_the_three_clocks_are_all_recorded(self):
        a = _assign(RUT, GAP_25_TIME - 1.5)
        self.assertEqual(a.local_frame, _local_frame(RUT, GAP_25_TIME - 1.5))
        self.assertAlmostEqual(a.local_time, a.local_frame / FPS, places=6)
        self.assertAlmostEqual(a.global_time,
                               a.local_time + OFFSETS[RUT][1], places=6)
        self.assertEqual(a.camera_time_offset, OFFSETS[RUT][1])

    def test_the_damage_camera_is_named_on_the_stored_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            _apply_in_tmp(tmp, {GW_25: [_rec(RUT, GAP_25_TIME - 1.5)]})
            recs = _read(os.path.join(tmp, "wagon_states", "damage",
                                      "GW_25.json"))["top_damage_details"]
            prov = recs[0][DA.PROVENANCE_KEY]
            self.assertEqual(prov["damage_camera_id"], RUT)
            self.assertEqual(prov["bucketed_global_wagon_id"], GW_25)

    def test_the_audit_json_is_written_with_worked_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            _apply_in_tmp(tmp, {
                GW_25: [_rec(RUT, GAP_25_TIME - 1.5)],
                GW_26: [_rec(LUT, GAP_25_TIME + 1.5, track_id=9)],
            })
            doc = _read(os.path.join(tmp, "global_state_diag",
                                     "damage_association.json"))
            self.assertEqual(doc["detections"], 2)
            self.assertTrue(doc["examples"][CA.METHOD_BEFORE_GAP])
            self.assertTrue(doc["examples"][CA.METHOD_AFTER_GAP])
            self.assertIn("canonical_timeline", doc)

    def test_the_method_vocabulary_is_the_one_the_spec_names(self):
        self.assertEqual(CA.METHOD_BEFORE_GAP, "BEFORE_GAP")
        self.assertEqual(CA.METHOD_AFTER_GAP, "AFTER_GAP")
        self.assertEqual(CA.STATUS_BOUNDARY_AMBIGUOUS, "BOUNDARY_AMBIGUOUS")


# ===========================================================================
# 9. One shared resolver, both modes, reusable for the other features
# ===========================================================================

class TestBothModesUseTheOneResolver(unittest.TestCase):

    @staticmethod
    def _calls(path, func):
        src = open(os.path.join(V4_ROOT, path), encoding="utf-8").read()
        tree = ast.parse(src)
        out = []
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            name = (f.attr if isinstance(f, ast.Attribute)
                    else getattr(f, "id", None))
            if name == func:
                out.append(n)
        return out

    def test_sequential_mode_calls_the_resolver(self):
        self.assertTrue(self._calls("orchestrator/global_assembler.py", "run"))
        src = open(os.path.join(V4_ROOT, "orchestrator/global_assembler.py"),
                   encoding="utf-8").read()
        self.assertIn("damage_association as DASSOC", src)
        self.assertIn("DASSOC.run(", src)

    def test_batch_mode_calls_the_same_resolver(self):
        src = open(os.path.join(V4_ROOT, "orchestrator/master_runner.py"),
                   encoding="utf-8").read()
        self.assertIn("damage_association as DASSOC", src)
        self.assertIn("DASSOC.run(", src)
        self.assertIn("DASSOC.load_global_gaps(", src)

    @staticmethod
    def _call_line(path, module, attr):
        """Line of the first real CALL of `module.attr(...)`.

        Purely AST, matched on BOTH halves of the dotted name. Both files
        describe the stage order in prose -- `master_runner`'s module docstring
        lists "Stage 4  fusion.wagon_state_builder.build" on line 11 -- so any
        text scan compares a comment against code and the ordering assertion
        means nothing.
        """
        tree = ast.parse(open(os.path.join(V4_ROOT, path),
                              encoding="utf-8").read())
        lines = [
            n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == attr
            and isinstance(n.func.value, ast.Name) and n.func.value.id == module
        ]
        return min(lines) if lines else None

    def test_both_modes_run_it_before_fusion(self):
        """Stage 4 reads the per-wagon damage JSONs, so the association has to
        have finished moving them by then."""
        for path in ("orchestrator/global_assembler.py",
                     "orchestrator/master_runner.py"):
            assoc = self._call_line(path, "DASSOC", "run")
            fuse = self._call_line(path, "wagon_state_builder", "build")
            self.assertIsNotNone(assoc, f"{path}: no DASSOC.run() call")
            self.assertIsNotNone(fuse, f"{path}: no fusion call")
            self.assertLess(assoc, fuse,
                            f"{path}: association at line {assoc} runs after "
                            f"fusion at line {fuse}")

    def test_the_two_modes_produce_identical_assignments(self):
        """Same canonical inputs in, same wagons out -- the property that makes
        "identically in both modes" a fact rather than an intention."""
        st = _state()
        gaps = _global_gaps()
        recs = {GW_25: [_rec(RUT, GAP_25_TIME - 1.5)],
                GW_26: [_rec(LUT, GAP_25_TIME + 1.5, track_id=9)]}
        seq = DA.resolve_train(state=st, global_gaps=gaps,
                               per_camera_fps={c: FPS for c in C.ALL_CAMERAS},
                               damage_by_wagon=recs, verbose=False)
        # Batch reaches the same gap list through the state file on disk.
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "global_train_state.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"global_gaps": gaps}, f)
            batch = DA.resolve_train(
                state=st, global_gaps=DA.load_global_gaps(p),
                per_camera_fps={c: FPS for c in C.ALL_CAMERAS},
                damage_by_wagon=recs, verbose=False)
        self.assertEqual([a.to_dict() for a in seq.assignments],
                         [a.to_dict() for a in batch.assignments])

    def test_load_global_gaps_survives_a_missing_or_broken_file(self):
        self.assertEqual(DA.load_global_gaps("/nonexistent/state.json"), [])
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "s.json")
            with open(p, "w", encoding="utf-8") as f:
                f.write("{not json")
            self.assertEqual(DA.load_global_gaps(p), [])

    def test_the_resolver_is_not_damage_specific(self):
        """It has to serve door / load / ocr next, so nothing in the shared
        module may mention damage."""
        src = open(os.path.join(V4_ROOT, "core/canonical_association.py"),
                   encoding="utf-8").read()
        tree = ast.parse(src)
        names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
        names += [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
        names += [n.name for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        offenders = sorted({n for n in names if "damage" in n.lower()})
        self.assertEqual(offenders, [], f"damage-specific names: {offenders}")

    def test_another_feature_can_use_it_unchanged(self):
        """A door detection, associated by the identical call path."""
        tl = _timeline()
        a = tl.assign(CA.Detection(camera_id=LU, feature="door",
                                   local_frame=_local_frame(LU, GAP_25_TIME - 1.5),
                                   detection_id="door-1"))
        self.assertEqual(a.feature, "door")
        self.assertEqual(a.global_wagon_id, GW_25)
        self.assertEqual(a.method, CA.METHOD_BEFORE_GAP)

    def test_the_resolver_runs_no_inference(self):
        src = open(os.path.join(V4_ROOT, "core/canonical_association.py"),
                   encoding="utf-8").read()
        for banned in ("YOLO", "load_yolo", "cv2", "predict("):
            self.assertNotIn(banned, src, f"{banned} in the resolver")

    def test_the_adapter_reads_existing_evidence_and_runs_no_inference(self):
        src = open(os.path.join(V4_ROOT, "orchestrator/damage_association.py"),
                   encoding="utf-8").read()
        for banned in ("YOLO", "load_yolo", "iter_wagon_frames", "cv2"):
            self.assertNotIn(banned, src, f"{banned} in the adapter")
        self.assertIn("read_damage_by_wagon", src)


# ===========================================================================
# 10. Stage-3 sampling defaults are untouched
# ===========================================================================

class TestStage3SamplingDefaultsArePreserved(unittest.TestCase):
    """Association happens after inference; it must not have moved a stride."""

    def test_damage_is_still_sampled_stride_3(self):
        self.assertEqual(C.STAGE3_DAMAGE_MODE, "sampled")
        self.assertEqual(C.STAGE3_DAMAGE_STRIDE, 3)

    def test_the_other_features_strides_are_untouched(self):
        self.assertEqual((C.STAGE3_DOOR_MODE, C.STAGE3_DOOR_STRIDE),
                         ("sampled", 3))
        self.assertEqual((C.STAGE3_LOAD_MODE, C.STAGE3_LOAD_STRIDE),
                         ("sampled", 2))

    def test_neither_new_module_mentions_a_stride_or_mode(self):
        for path in ("core/canonical_association.py",
                     "orchestrator/damage_association.py"):
            src = open(os.path.join(V4_ROOT, path), encoding="utf-8").read()
            for banned in ("STAGE3_DAMAGE_STRIDE", "STAGE3_DAMAGE_MODE",
                           "sample_stride", "inference_mode"):
                self.assertNotIn(banned, src, f"{banned} in {path}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
