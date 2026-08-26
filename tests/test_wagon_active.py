"""The timeline is WAGON activity only, and the engine must never start it.

The failure being fixed: at the head of the rake the classifier is least
reliable. A locomotive looks a lot like a wagon, and the gap detector fires on
its leading face, so one early WAGON frame beside a gap was enough to open the
timeline on the engine -- minting GW_1 out of the locomotive and shifting every
later id by one.

Two defences, tested separately and together:

    sustained evidence   a WAGON run must last `min_active_duration` before it
                         opens a region, so a blip cannot start the train
    four cameras         the common interval is the MEDIAN of the per-camera
                         starts, so one camera misreading the engine cannot
                         drag the boundary onto it

ENGINE, BRAKE_VAN and UNKNOWN are never regions of this timeline and can never
receive a GW_n.
"""

from __future__ import annotations

import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from core.train_window import LabelledSpan
from core.wagon_active import (
    ACTIVE_CLASS, METHOD_MEDIAN, NON_WAGON_CLASSES, ActivationPolicy,
    audit_payload, build_wagon_timeline, camera_wagon_activity,
    common_wagon_window, gaps_inside_window,
)

RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP
RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP
ENGINE, WAGON, BRAKE, UNKNOWN = (C.CLASS_ENGINE, C.CLASS_WAGON,
                                 C.CLASS_BRAKE_VAN, C.CLASS_UNKNOWN)


def _spans(cam, items):
    return [LabelledSpan(cam, s, e, l, conf)
            for (s, e, l, conf) in
            ((a, b, c, d if len(x) > 3 else 0.9)
             for x in items
             for a, b, c, d in [(x[0], x[1], x[2],
                                 x[3] if len(x) > 3 else 0.9)])]


def _clean(cam=RU, shift=0.0):
    """ENGINE 10-18, then wagons 18-42, then BRAKE_VAN 42-48."""
    return _spans(cam, [(10.0 + shift, 18.0 + shift, ENGINE),
                        (18.0 + shift, 26.0 + shift, WAGON),
                        (26.0 + shift, 34.0 + shift, WAGON),
                        (34.0 + shift, 42.0 + shift, WAGON),
                        (42.0 + shift, 48.0 + shift, BRAKE)])


class TestEngineNeverStartsTheTimeline(unittest.TestCase):
    """The reported failure, from every angle."""

    def test_a_brief_wagon_misread_on_the_engine_is_rejected(self):
        spans = _spans(RU, [(10.0, 10.4, WAGON),      # 0.4s misread
                            (10.4, 18.0, ENGINE),
                            (18.0, 26.0, WAGON),
                            (26.0, 34.0, WAGON)])
        a = camera_wagon_activity(spans, RU)
        self.assertAlmostEqual(a.wagon_active_start, 18.0,
                               msg="a 0.4s blip on the engine opened the region")
        self.assertEqual(len(a.rejected_blips), 1)
        self.assertIn("under the", a.rejected_blips[0].reason)

    def test_a_gap_beside_the_engine_cannot_create_gw_1(self):
        """A gap at the engine's face is outside the interval, so it is dropped."""
        a = camera_wagon_activity(_clean(), RU)
        win = common_wagon_window({RU: a})
        gaps = [11.0, 26.0, 34.0, 45.0]        # 11.0 = engine face, 45.0 = brake
        inside = gaps_inside_window(gaps, win)
        self.assertEqual(inside, [26.0, 34.0])
        roster = build_wagon_timeline(win, gaps)
        self.assertEqual([w["global_id"] for w in roster],
                         ["GW_1", "GW_2", "GW_3"])
        self.assertGreaterEqual(roster[0]["start_time"], 18.0,
                                "GW_1 must not begin on the engine")

    def test_engine_classification_never_becomes_a_wagon(self):
        a = camera_wagon_activity(_clean(), RU)
        for iv in a.intervals:
            self.assertGreaterEqual(iv.start_time, 18.0)
        self.assertIn(ENGINE, a.non_wagon_before)

    def test_one_camera_misreading_the_engine_cannot_move_the_common_start(self):
        """The median defence: three cameras outvote the one that is wrong."""
        wrong = _spans(RU, [(10.0, 18.0, WAGON),       # engine read as WAGON
                            (18.0, 26.0, WAGON), (26.0, 34.0, WAGON),
                            (34.0, 42.0, WAGON)])
        acts = {RU: camera_wagon_activity(wrong, RU),
                LU: camera_wagon_activity(_clean(LU), LU),
                RUT: camera_wagon_activity(_clean(RUT), RUT),
                LUT: camera_wagon_activity(_clean(LUT), LUT)}
        self.assertAlmostEqual(acts[RU].wagon_active_start, 10.0,
                               msg="fixture should have the master wrong")
        win = common_wagon_window(acts)
        self.assertAlmostEqual(win.start_time, 18.0,
                               msg="one bad camera moved the common start")
        self.assertEqual(win.method, METHOD_MEDIAN)
        self.assertTrue(any("moved the start" in n for n in win.notes))

    def test_the_masters_own_figure_is_still_recorded(self):
        """Corroboration must be auditable, not silent."""
        wrong = _spans(RU, [(10.0, 42.0, WAGON)])
        acts = {RU: camera_wagon_activity(wrong, RU),
                LU: camera_wagon_activity(_clean(LU), LU),
                RUT: camera_wagon_activity(_clean(RUT), RUT)}
        win = common_wagon_window(acts)
        self.assertAlmostEqual(win.master_start, 10.0)
        self.assertNotAlmostEqual(win.start_time, 10.0)

    def test_a_gap_alone_cannot_open_the_timeline(self):
        """No classification, no interval -- however many gaps there are."""
        acts = {RU: camera_wagon_activity(
            _spans(RU, [(0.0, 60.0, UNKNOWN)]), RU)}
        win = common_wagon_window(acts)
        self.assertFalse(win.found)
        self.assertEqual(build_wagon_timeline(win, [5.0, 10.0, 20.0]), [])


class TestSustainedEvidence(unittest.TestCase):
    def test_sustained_wagon_opens_a_region(self):
        a = camera_wagon_activity(_clean(), RU)
        self.assertTrue(a.usable)
        self.assertAlmostEqual(a.wagon_active_start, 18.0)
        self.assertAlmostEqual(a.wagon_active_end, 42.0)

    def test_the_threshold_is_configurable(self):
        spans = _spans(RU, [(0.0, 1.0, WAGON), (1.0, 10.0, ENGINE),
                            (10.0, 20.0, WAGON)])
        lax = camera_wagon_activity(
            spans, RU, policy=ActivationPolicy(min_active_duration=0.5))
        self.assertAlmostEqual(lax.wagon_active_start, 0.0)
        strict = camera_wagon_activity(
            spans, RU, policy=ActivationPolicy(min_active_duration=3.0))
        self.assertAlmostEqual(strict.wagon_active_start, 10.0)

    def test_a_short_non_wagon_stretch_does_not_close_the_region(self):
        """An inter-wagon gap reads as non-WAGON; closing there splits the rake."""
        spans = _spans(RU, [(10.0, 18.0, WAGON), (18.0, 19.0, UNKNOWN),
                            (19.0, 27.0, WAGON)])
        a = camera_wagon_activity(spans, RU)
        self.assertEqual(len(a.intervals), 1, "the rake was split by a gap")
        self.assertAlmostEqual(a.wagon_active_end, 27.0)

    def test_sustained_absence_does_close_the_region(self):
        spans = _spans(RU, [(10.0, 18.0, WAGON), (18.0, 40.0, UNKNOWN),
                            (40.0, 48.0, WAGON)])
        a = camera_wagon_activity(
            spans, RU, policy=ActivationPolicy(min_inactive_duration=6.0))
        self.assertEqual(len(a.intervals), 2)
        self.assertAlmostEqual(a.wagon_active_start, 10.0)
        self.assertAlmostEqual(a.wagon_active_end, 48.0)

    def test_a_trailing_blip_does_not_extend_the_end(self):
        spans = _spans(RU, [(10.0, 34.0, WAGON), (34.0, 48.0, BRAKE),
                            (48.0, 48.3, WAGON)])
        a = camera_wagon_activity(spans, RU)
        self.assertAlmostEqual(a.wagon_active_end, 34.0)
        self.assertEqual(len(a.rejected_blips), 1)

    def test_no_wagon_evidence_at_all_is_reported(self):
        a = camera_wagon_activity(_spans(RU, [(0.0, 30.0, ENGINE)]), RU)
        self.assertFalse(a.usable)
        self.assertIn("no WAGON run reached", a.reason)


class TestNonWagonClassesAreExcluded(unittest.TestCase):
    def test_the_active_class_is_wagon_alone(self):
        self.assertEqual(ACTIVE_CLASS, C.CLASS_WAGON)
        self.assertEqual(set(NON_WAGON_CLASSES), {ENGINE, BRAKE, UNKNOWN})

    def test_engine_and_brakevan_are_outside_every_interval(self):
        a = camera_wagon_activity(_clean(), RU)
        for iv in a.intervals:
            self.assertGreaterEqual(iv.start_time, 18.0)
            self.assertLessEqual(iv.end_time, 42.0)
        self.assertIn(ENGINE, a.non_wagon_before)
        self.assertIn(BRAKE, a.non_wagon_after)

    def test_no_gw_id_is_produced_for_a_non_wagon_region(self):
        a = camera_wagon_activity(_clean(), RU)
        win = common_wagon_window({RU: a})
        roster = build_wagon_timeline(win, [26.0, 34.0])
        for w in roster:
            self.assertEqual(w["classification"], WAGON)
            self.assertGreaterEqual(w["start_time"], 18.0)
            self.assertLessEqual(w["end_time"], 42.0)

    def test_unknown_does_not_extend_the_timeline(self):
        spans = _spans(RU, [(10.0, 18.0, UNKNOWN), (18.0, 30.0, WAGON),
                            (30.0, 40.0, UNKNOWN)])
        a = camera_wagon_activity(spans, RU)
        self.assertAlmostEqual(a.wagon_active_start, 18.0)
        self.assertAlmostEqual(a.wagon_active_end, 30.0)


class TestCameraCorroboration(unittest.TestCase):
    def test_offsets_are_applied_before_combining(self):
        """A camera 3s behind, projected, agrees with the master."""
        local = _clean(LU, shift=-3.0)
        acts = {RU: camera_wagon_activity(_clean(RU), RU),
                LU: camera_wagon_activity(local, LU, offset=3.0)}
        self.assertAlmostEqual(acts[LU].wagon_active_start, 18.0)
        win = common_wagon_window(acts)
        self.assertAlmostEqual(win.start_time, 18.0)

    def test_all_four_cameras_contribute(self):
        acts = {c: camera_wagon_activity(_clean(c), c)
                for c in (RU, LU, RUT, LUT)}
        win = common_wagon_window(acts)
        self.assertEqual(sorted(win.contributing), sorted([RU, LU, RUT, LUT]))
        self.assertTrue(win.found)

    def test_a_camera_with_no_wagons_is_reported_not_dropped(self):
        acts = {RU: camera_wagon_activity(_clean(RU), RU),
                LUT: camera_wagon_activity(_spans(LUT, [(0.0, 60.0, UNKNOWN)]),
                                           LUT)}
        win = common_wagon_window(acts)
        self.assertIn(LUT, win.per_camera)
        self.assertFalse(win.per_camera[LUT].usable)
        self.assertNotIn(LUT, win.contributing)

    def test_a_single_camera_is_flagged_as_uncorroborated(self):
        win = common_wagon_window({RU: camera_wagon_activity(_clean(), RU)})
        self.assertTrue(win.found)
        self.assertTrue(any("uncorroborated" in n for n in win.notes))

    def test_prefer_master_overrides_the_median(self):
        acts = {RU: camera_wagon_activity(_clean(RU), RU),
                LU: camera_wagon_activity(_clean(LU, shift=5.0), LU)}
        win = common_wagon_window(
            acts, policy=ActivationPolicy(prefer_master=True))
        self.assertAlmostEqual(win.start_time, 18.0)
        self.assertEqual(win.contributing, [RU])


class TestRosterFromMasterGaps(unittest.TestCase):
    def test_gw_ids_come_from_gaps_inside_the_interval(self):
        win = common_wagon_window(
            {c: camera_wagon_activity(_clean(c), c) for c in (RU, LU, RUT)})
        roster = build_wagon_timeline(win, [26.0, 34.0])
        self.assertEqual([w["global_id"] for w in roster],
                         ["GW_1", "GW_2", "GW_3"])
        self.assertAlmostEqual(roster[0]["start_time"], 18.0)
        self.assertAlmostEqual(roster[-1]["end_time"], 42.0)

    def test_gaps_outside_the_interval_are_excluded(self):
        win = common_wagon_window({RU: camera_wagon_activity(_clean(), RU)})
        self.assertEqual(gaps_inside_window([11.0, 26.0, 45.0], win), [26.0])

    def test_gap_before_and_after_are_recorded_per_wagon(self):
        win = common_wagon_window({RU: camera_wagon_activity(_clean(), RU)})
        roster = build_wagon_timeline(win, [26.0, 34.0])
        self.assertIsNone(roster[0]["gap_before"], "first wagon has no gap before")
        self.assertAlmostEqual(roster[0]["gap_after"], 26.0)
        self.assertIsNone(roster[-1]["gap_after"], "last wagon has no gap after")

    def test_support_cameras_contribute_no_gaps(self):
        import ast
        import inspect
        from core import wagon_active
        src = inspect.getsource(wagon_active.gaps_inside_window)
        self.assertNotIn("support", src.split('"""')[-1])
        called = {ast.unparse(n.func)
                  for n in ast.walk(ast.parse(inspect.getsource(wagon_active)))
                  if isinstance(n, ast.Call)}
        for banned in ("build_global_gap_sequence", "GapTracker",
                       "validate_gap_events", "renumber_gap_events"):
            self.assertFalse([c for c in called if banned in c])


class TestAuditPayload(unittest.TestCase):
    def test_the_payload_explains_the_boundary(self):
        acts = {c: camera_wagon_activity(_clean(c), c)
                for c in (RU, LU, RUT, LUT)}
        win = common_wagon_window(acts)
        p = audit_payload(win, [11.0, 26.0, 34.0, 45.0])
        self.assertEqual(p["wagon_active_start"], 18.0)
        self.assertEqual(p["wagon_active_end"], 42.0)
        self.assertEqual(p["method"], METHOD_MEDIAN)
        self.assertEqual(p["canonical_gaps_total"], 4)
        self.assertEqual(p["canonical_gaps_inside"], 2)
        self.assertEqual(p["canonical_gaps_excluded"], 2)
        self.assertEqual(set(p["non_wagon_classes_excluded"]),
                         {ENGINE, BRAKE, UNKNOWN})
        for cam in (RU, LU, RUT, LUT):
            self.assertIn(cam, p["per_camera"])
            self.assertIn("wagon_active_start", p["per_camera"][cam])
            self.assertIn("rejected_blips", p["per_camera"][cam])

    def test_rejected_blips_are_visible_in_the_audit(self):
        spans = _spans(RU, [(10.0, 10.3, WAGON), (10.3, 18.0, ENGINE),
                            (18.0, 34.0, WAGON)])
        win = common_wagon_window({RU: camera_wagon_activity(spans, RU)})
        p = audit_payload(win, [26.0])
        blips = p["per_camera"][RU]["rejected_blips"]
        self.assertEqual(len(blips), 1)
        self.assertAlmostEqual(blips[0]["start_time"], 10.0)


if __name__ == "__main__":
    unittest.main()
