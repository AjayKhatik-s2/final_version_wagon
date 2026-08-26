"""The processed video must audit the canonical timeline, not decorate it.

The property that matters is DETECTED versus PROJECTED. A boundary drawn on a
support camera is usually a master boundary projected onto that camera's clock;
the local detector may never have seen it. Drawing both identically would let a
viewer read agreement into a picture that shows only arithmetic.

These test the overlay PLAN -- placement, labels, status, boundaries -- because
that is where a wrong answer is a real bug. A rectangle two pixels off is not.
"""

from __future__ import annotations

import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from core.master_timeline import CameraClock
from core.train_window import (
    LabelledSpan, build_train_timeline, detect_train_window,
)
from rendering.timeline_overlay import (
    DETECTED, OUT_OF_COVERAGE, PROJECTED, build_overlay_plan, draw_overlay,
    overlay_at,
)

RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP
LUT = C.CAMERA_LEFT_UP_TOP
ENGINE, WAGON, BRAKE = C.CLASS_ENGINE, C.CLASS_WAGON, C.CLASS_BRAKE_VAN
FPS = 15.0

#: ENGINE 10-16, WAGON 16-24, WAGON 24-32, BRAKE_VAN 32-40.
#: Canonical gaps therefore sit at 24.0 (GW_1|GW_2) -- the boundary between the
#: two counted wagons -- plus the engine/brake edges at 16.0 and 32.0.
SPANS = [LabelledSpan(RU, 10.0, 16.0, ENGINE, 0.95),
         LabelledSpan(RU, 16.0, 24.0, WAGON, 0.91),
         LabelledSpan(RU, 24.0, 32.0, WAGON, 0.88),
         LabelledSpan(RU, 32.0, 40.0, BRAKE, 0.83)]
GAPS = [16.0, 24.0, 32.0]


def _timeline(spans=None):
    spans = spans if spans is not None else SPANS
    return build_train_timeline(detect_train_window(master_spans=spans), spans)


def _clock(cam=RU, fps=FPS, frames=900, offset=0.0):
    return CameraClock(camera_id=cam, fps=fps, total_frames=frames,
                       offset=offset, offset_status="RESOLVED")


def _plan(clock=None, detected=(), centers=None, spans=None):
    return build_overlay_plan(
        timeline=_timeline(spans), canonical_gap_times=GAPS,
        clock=clock or _clock(), detected_gap_times=detected,
        detected_center_x=centers)


class TestGapMarkerPlacement(unittest.TestCase):
    def test_a_marker_lands_on_the_projected_local_frame(self):
        p = _plan()
        by_time = {m.master_time: m for m in p.markers}
        self.assertEqual(by_time[24.0].local_frame, int(round(24.0 * FPS)))

    def test_a_camera_offset_shifts_the_marker(self):
        """t_local = t_master - delta, so a +2s camera draws 30 frames earlier."""
        p = _plan(_clock(LU, offset=2.0))
        by_time = {m.master_time: m for m in p.markers}
        self.assertEqual(by_time[24.0].local_frame,
                         int(round((24.0 - 2.0) * FPS)))

    def test_a_different_fps_still_lands_on_the_same_instant(self):
        p = _plan(_clock(LU, fps=25.0, frames=1500))
        by_time = {m.master_time: m for m in p.markers}
        self.assertEqual(by_time[24.0].local_frame, int(round(24.0 * 25.0)))

    def test_a_gap_outside_the_footage_is_not_drawn(self):
        """A short camera must not have a boundary invented at its last frame."""
        p = _plan(_clock(LUT, frames=int(20.0 * FPS)))   # 20s only
        late = [m for m in p.markers if m.master_time > 20.0]
        self.assertTrue(late)
        for m in late:
            self.assertEqual(m.status, OUT_OF_COVERAGE)
            self.assertIsNone(m.local_frame)
            self.assertFalse(m.drawable)

    def test_no_marker_appears_outside_its_own_frame(self):
        p = _plan()
        target = int(round(24.0 * FPS))
        far = overlay_at(p, target + 30)
        self.assertEqual([m.master_time for m in far.markers_here], [])
        near = overlay_at(p, target)
        self.assertIn(24.0, [m.master_time for m in near.markers_here])

    def test_every_canonical_gap_gets_exactly_one_marker(self):
        p = _plan()
        self.assertEqual(len(p.markers), len(GAPS))
        self.assertEqual(sorted(m.master_time for m in p.markers), sorted(GAPS))
        self.assertEqual([m.global_gap_id for m in p.markers], [1, 2, 3])

    def test_the_renderer_creates_no_gap_of_its_own(self):
        """Only the canonical sequence may produce a marker."""
        p = build_overlay_plan(timeline=_timeline(), canonical_gap_times=[],
                               clock=_clock(), detected_gap_times=[5.0, 9.0])
        self.assertEqual(p.markers, [], "a local detection minted a boundary")


class TestDetectedVersusProjected(unittest.TestCase):
    def test_a_local_detection_marks_the_gap_detected(self):
        p = _plan(detected=[24.05])
        m = {x.master_time: x for x in p.markers}[24.0]
        self.assertEqual(m.status, DETECTED)
        self.assertAlmostEqual(m.detected_local_time, 24.05)

    def test_no_local_detection_marks_it_projected(self):
        p = _plan(detected=[])
        for m in p.markers:
            self.assertEqual(m.status, PROJECTED)
            self.assertIsNone(m.detected_local_time)

    def test_a_distant_detection_does_not_count_as_the_same_gap(self):
        p = _plan(detected=[20.0])          # 4s away from any canonical gap
        for m in p.markers:
            self.assertEqual(m.status, PROJECTED)

    def test_the_two_statuses_are_visibly_different_in_the_label(self):
        det = _plan(detected=[24.0]).markers[1]
        pro = _plan(detected=[]).markers[1]
        self.assertIn(DETECTED, det.label)
        self.assertIn(PROJECTED, pro.label)
        self.assertNotEqual(det.label, pro.label)

    def test_tracked_geometry_is_used_when_the_camera_detected_the_gap(self):
        p = _plan(detected=[24.0], centers={24.0: 640.0})
        m = {x.master_time: x for x in p.markers}[24.0]
        self.assertEqual(m.center_x, 640.0)

    def test_a_projected_marker_has_no_geometry_to_borrow(self):
        p = _plan(detected=[], centers={24.0: 640.0})
        m = {x.master_time: x for x in p.markers}[24.0]
        self.assertIsNone(m.center_x,
                          "projected boundary must not claim tracked geometry")


class TestClassificationAndGwLabels(unittest.TestCase):
    def test_engine_frames_show_the_class_and_no_gw(self):
        ov = overlay_at(_plan(), int(12.0 * FPS))
        self.assertTrue(ov.in_train)
        self.assertEqual(ov.region_kind, ENGINE)
        self.assertIsNone(ov.global_id)
        self.assertIn("not counted", ov.wagon_label)

    def test_brake_van_frames_show_the_class_and_no_gw(self):
        ov = overlay_at(_plan(), int(36.0 * FPS))
        self.assertEqual(ov.region_kind, BRAKE)
        self.assertIsNone(ov.global_id)

    def test_wagon_frames_show_the_canonical_gw(self):
        first = overlay_at(_plan(), int(20.0 * FPS))
        second = overlay_at(_plan(), int(28.0 * FPS))
        self.assertEqual(first.global_id, "GW_1")
        self.assertEqual(second.global_id, "GW_2")
        self.assertIn("GW_1", first.wagon_label)
        self.assertIn(WAGON, first.wagon_label)

    def test_confidence_is_carried_from_the_classification(self):
        ov = overlay_at(_plan(), int(20.0 * FPS))
        self.assertAlmostEqual(ov.region_confidence, 0.91)
        self.assertTrue(any("conf" in ln for ln in ov.lines()))

    def test_the_camera_name_is_always_shown(self):
        for cam in (RU, LU, LUT):
            ov = overlay_at(_plan(_clock(cam)), int(20.0 * FPS))
            self.assertIn(cam, ov.lines()[0])

    def test_frames_outside_the_train_say_so(self):
        ov = overlay_at(_plan(), int(2.0 * FPS))
        self.assertFalse(ov.in_train)
        self.assertEqual(ov.wagon_label, "OUTSIDE TRAIN")
        self.assertIsNone(ov.global_id)

    def test_the_text_block_is_stable_across_frames(self):
        """Same line count and order, so the overlay does not jitter."""
        p = _plan()
        counts = {len(overlay_at(p, f).lines())
                  for f in range(int(17 * FPS), int(23 * FPS), 5)}
        self.assertEqual(len(counts), 1, f"line count varies: {counts}")


class TestGapBeforeAndAfter(unittest.TestCase):
    def test_a_wagon_reports_both_canonical_boundaries(self):
        ov = overlay_at(_plan(), int(20.0 * FPS))       # inside GW_1
        self.assertIsNotNone(ov.gap_before)
        self.assertIsNotNone(ov.gap_after)
        self.assertAlmostEqual(ov.gap_before.master_time, 16.0)
        self.assertAlmostEqual(ov.gap_after.master_time, 24.0)

    def test_the_boundary_times_appear_in_the_text(self):
        text = " ".join(overlay_at(_plan(), int(20.0 * FPS)).lines())
        self.assertIn("GAP_BEFORE", text)
        self.assertIn("GAP_AFTER", text)
        self.assertIn("16.00s", text)
        self.assertIn("24.00s", text)

    def test_before_the_first_gap_there_is_no_gap_before(self):
        ov = overlay_at(_plan(), int(12.0 * FPS))       # inside the ENGINE
        self.assertIsNone(ov.gap_before)
        self.assertIn("train end", " ".join(ov.lines()))

    def test_after_the_last_gap_there_is_no_gap_after(self):
        ov = overlay_at(_plan(), int(36.0 * FPS))       # inside the BRAKE_VAN
        self.assertIsNone(ov.gap_after)

    def test_the_boundary_status_travels_with_it(self):
        ov = overlay_at(_plan(detected=[24.0]), int(20.0 * FPS))
        self.assertEqual(ov.gap_after.status, DETECTED)
        self.assertIn(DETECTED, " ".join(ov.lines()))

    def test_a_boundary_beside_the_engine_reports_no_wagon_on_that_side(self):
        """The engine has no GW id, so the marker must not borrow GW_1's."""
        m = {x.master_time: x for x in _plan().markers}[16.0]
        self.assertIsNone(m.gw_before)
        self.assertEqual(m.gw_after, "GW_1")

    def test_the_boundary_between_two_wagons_names_both(self):
        m = {x.master_time: x for x in _plan().markers}[24.0]
        self.assertEqual((m.gw_before, m.gw_after), ("GW_1", "GW_2"))


class TestSourceOfTruth(unittest.TestCase):
    def test_the_overlay_recomputes_nothing(self):
        import ast
        import inspect
        from rendering import timeline_overlay
        src = inspect.getsource(timeline_overlay)
        called = {ast.unparse(n.func) for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.Call)}
        for banned in ("GapTracker", "validate_gap_events", "detect_train_window",
                       "build_global_gap_sequence", "build_train_timeline",
                       "renumber_gap_events", "load_yolo"):
            with self.subTest(token=banned):
                self.assertFalse([c for c in called if banned in c],
                                 f"{banned} is invoked inside the renderer")

    def test_projection_uses_the_canonical_timeline_api(self):
        import inspect
        from rendering import timeline_overlay
        src = inspect.getsource(timeline_overlay)
        self.assertIn("from core.master_timeline import", src)
        self.assertIn("master_time_to_local_frame", src)

    def test_drawing_a_real_frame_does_not_raise(self):
        import numpy as np
        frame = np.zeros((240, 640, 3), dtype=np.uint8)
        p = _plan(detected=[24.0], centers={24.0: 300.0})
        draw_overlay(frame, overlay_at(p, int(24.0 * FPS)))
        self.assertTrue(frame.any(), "nothing was drawn")

    def test_the_plan_summarises_its_own_statuses(self):
        text = " ".join(_plan(detected=[24.0]).summary_lines())
        self.assertIn(DETECTED, text)
        self.assertIn(PROJECTED, text)
        self.assertIn("region(s)", text)


if __name__ == "__main__":
    unittest.main()
