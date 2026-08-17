"""Tests for the sequential-mode camera evidence bundle and local->global map.

No models, no video: this is the bookkeeping contract that lets each camera
finish independently and be reconciled later.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core.camera_evidence import (
    FAILED, LIFECYCLE, MAP_EXACT, MAP_MANY_TO_ONE, MAP_ONE_TO_MANY,
    MAP_UNMATCHED, MAP_UNRESOLVED, CameraEvidenceBundle, CameraEvidenceError,
    LocalSegment, can_advance, local_segment_id, map_segments_to_global,
    mapping_summary, next_state, ready_for_global_assembly,
)

CAMS = ("RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP", "LEFT_UP_TOP")
MASTER = "RIGHT_UP"


@dataclass
class _GW:
    global_id: str
    start_time: float
    end_time: float


def _roster(n=5, dur=4.0, t0=10.0):
    return [_GW(f"GW_{i}", t0 + (i - 1) * dur, t0 + i * dur)
            for i in range(1, n + 1)]


def _seg(i, s, e, cam=MASTER):
    return LocalSegment(local_id=local_segment_id(cam, i), index=i,
                        start_frame=int(s * 15), end_frame=int(e * 15),
                        start_time=s, end_time=e, label="WAGON",
                        confidence=0.9)


class TestLifecycle(unittest.TestCase):
    def test_order_is_forward_only_one_step(self):
        for a, b in zip(LIFECYCLE, LIFECYCLE[1:]):
            self.assertTrue(can_advance(a, b))
        self.assertFalse(can_advance("PENDING", "FEATURES"), "no skipping")
        self.assertFalse(can_advance("FEATURES", "TRACKING"), "no going back")

    def test_terminal_states_are_terminal(self):
        self.assertFalse(can_advance("SEALED", "REPORTED"))
        self.assertFalse(can_advance(FAILED, "TRACKING"))
        with self.assertRaises(CameraEvidenceError):
            next_state("SEALED")
        with self.assertRaises(CameraEvidenceError):
            next_state(FAILED)

    def test_failure_reachable_from_any_live_state(self):
        for s in LIFECYCLE[:-1]:
            self.assertTrue(can_advance(s, FAILED))


class TestBundleRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.b = CameraEvidenceBundle(self.tmp.name, MASTER)

    def tearDown(self):
        self.tmp.cleanup()

    def test_new_bundle_is_pending(self):
        self.assertEqual(self.b.load_manifest().state, "PENDING")
        self.assertFalse(self.b.is_sealed)

    def test_advance_persists_and_resumes(self):
        self.b.advance("TRACKING", fps=15.0, total_frames=3555)
        reopened = CameraEvidenceBundle(self.tmp.name, MASTER).load_manifest()
        self.assertEqual(reopened.state, "TRACKING")
        self.assertEqual(reopened.fps, 15.0)
        self.assertEqual(reopened.total_frames, 3555)

    def test_illegal_transition_rejected(self):
        with self.assertRaises(CameraEvidenceError):
            self.b.advance("FEATURES")

    def test_full_walk_to_sealed(self):
        for s in LIFECYCLE[1:]:
            self.b.advance(s)
        self.assertTrue(self.b.is_sealed)
        with self.assertRaises(CameraEvidenceError):
            self.b.advance("REPORTED")

    def test_failure_records_reason(self):
        self.b.advance("TRACKING")
        m = self.b.fail("cv2 could not open video")
        self.assertEqual(m.state, FAILED)
        self.assertIn("cv2", m.failure_reason)
        self.assertTrue(self.b.is_terminal)

    def test_segments_round_trip(self):
        segs = [_seg(1, 0.0, 4.0), _seg(2, 4.0, 8.0)]
        self.b.write_segments(segs)
        got = self.b.read_segments()
        self.assertEqual([s.local_id for s in got], ["L_RIGHT_UP_1", "L_RIGHT_UP_2"])
        self.assertEqual(got[1].start_frame, 60)

    def test_local_ids_are_not_global_ids(self):
        self.assertFalse(local_segment_id("LEFT_UP", 3).startswith("GW_"))
        self.assertEqual(local_segment_id("LEFT_UP", 3), "L_LEFT_UP_3")


class TestMapping(unittest.TestCase):
    def setUp(self):
        self.roster = _roster(5)          # GW_1..GW_5, 10..30s, 4s each

    def test_exact_one_to_one(self):
        segs = [_seg(i, 10.0 + (i - 1) * 4, 10.0 + i * 4) for i in range(1, 6)]
        got = map_segments_to_global(segs, self.roster, camera_id=MASTER)
        self.assertEqual([m.global_id for m in got],
                         [f"GW_{i}" for i in range(1, 6)])
        self.assertTrue(all(m.kind == MAP_EXACT for m in got))

    def test_offset_shift_is_applied(self):
        """Support camera 2s behind: local+2 must land on the same GWs."""
        segs = [_seg(i, 8.0 + (i - 1) * 4, 8.0 + i * 4, cam="LEFT_UP")
                for i in range(1, 6)]
        got = map_segments_to_global(segs, self.roster, camera_id="LEFT_UP",
                                     offset=2.0)
        self.assertEqual([m.global_id for m in got],
                         [f"GW_{i}" for i in range(1, 6)])
        self.assertTrue(all(m.offset_applied == 2.0 for m in got))

    def test_one_to_many_when_support_missed_a_gap(self):
        """A local segment covering GW_2+GW_3 must be flagged, not silently
        assigned."""
        segs = [_seg(1, 14.0, 22.0, cam="LEFT_UP")]
        got = map_segments_to_global(segs, self.roster, camera_id="LEFT_UP")
        self.assertEqual(got[0].kind, MAP_ONE_TO_MANY)
        self.assertGreaterEqual(len(got[0].candidates), 2)
        self.assertIn(got[0].global_id, ("GW_2", "GW_3"))

    def test_many_to_one_when_support_saw_an_extra_gap(self):
        """Two locals inside one GW must both be flagged, neither discarded."""
        segs = [_seg(1, 14.1, 16.0, cam="LEFT_UP"),
                _seg(2, 16.0, 17.9, cam="LEFT_UP")]
        got = map_segments_to_global(segs, self.roster, camera_id="LEFT_UP")
        self.assertTrue(all(m.global_id == "GW_2" for m in got))
        self.assertTrue(all(m.kind == MAP_MANY_TO_ONE for m in got))

    def test_unmatched_segment_is_reported_not_dropped(self):
        segs = [_seg(1, 900.0, 904.0, cam="LEFT_UP")]
        got = map_segments_to_global(segs, self.roster, camera_id="LEFT_UP")
        self.assertEqual(len(got), 1, "unmatched segment must still appear")
        self.assertEqual(got[0].kind, MAP_UNMATCHED)
        self.assertIsNone(got[0].global_id)

    def test_unresolved_offset_flagged_and_never_guessed(self):
        segs = [_seg(1, 10.0, 14.0, cam="LEFT_UP_TOP")]
        got = map_segments_to_global(segs, self.roster, camera_id="LEFT_UP_TOP",
                                     offset=7.5, offset_resolved=False)
        self.assertEqual(got[0].kind, MAP_UNRESOLVED)
        self.assertEqual(got[0].offset_applied, 0.0,
                         "unresolved offset must fall back to 0.0, not 7.5")

    def test_every_segment_yields_exactly_one_record(self):
        segs = [_seg(i, 10.0 + (i - 1) * 4, 10.0 + i * 4) for i in range(1, 6)]
        segs.append(_seg(99, 900.0, 904.0))
        got = map_segments_to_global(segs, self.roster, camera_id=MASTER)
        self.assertEqual(len(got), len(segs), "no segment may be dropped")

    def test_summary_counts_ambiguity(self):
        segs = [_seg(1, 14.0, 22.0), _seg(2, 900.0, 904.0)]
        s = mapping_summary(map_segments_to_global(segs, self.roster,
                                                   camera_id=MASTER))
        self.assertEqual(s["total"], 2)
        self.assertEqual(s["ambiguous"], 2)
        self.assertEqual(s["unmatched_local_ids"], ["L_RIGHT_UP_2"])


class TestAssemblyTrigger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _seal(self, cam):
        b = CameraEvidenceBundle(self.root, cam)
        for s in LIFECYCLE[1:]:
            b.advance(s)

    def test_blocked_until_master_sealed(self):
        ok, why = ready_for_global_assembly(self.root, MASTER, CAMS)
        self.assertFalse(ok)
        self.assertIn("RIGHT_UP", why)

    def test_blocked_while_a_support_camera_is_live(self):
        self._seal(MASTER)
        CameraEvidenceBundle(self.root, "LEFT_UP").advance("TRACKING")
        for c in ("RIGHT_UP_TOP", "LEFT_UP_TOP"):
            self._seal(c)
        ok, why = ready_for_global_assembly(self.root, MASTER, CAMS)
        self.assertFalse(ok)
        self.assertIn("LEFT_UP", why)

    def test_failed_support_camera_does_not_block(self):
        self._seal(MASTER)
        CameraEvidenceBundle(self.root, "LEFT_UP").fail("video missing")
        for c in ("RIGHT_UP_TOP", "LEFT_UP_TOP"):
            self._seal(c)
        ok, _ = ready_for_global_assembly(self.root, MASTER, CAMS)
        self.assertTrue(ok, "a failed SUPPORT camera must not block assembly")

    def test_failed_master_aborts(self):
        CameraEvidenceBundle(self.root, MASTER).fail("gap model missing")
        for c in CAMS[1:]:
            self._seal(c)
        ok, why = ready_for_global_assembly(self.root, MASTER, CAMS)
        self.assertFalse(ok)
        self.assertIn("FAILED", why)

    def test_ready_when_all_sealed(self):
        for c in CAMS:
            self._seal(c)
        ok, _ = ready_for_global_assembly(self.root, MASTER, CAMS)
        self.assertTrue(ok)

    def test_cameras_are_independent(self):
        """The core promise: one camera reaching SEALED requires no other."""
        self._seal("LEFT_UP_TOP")
        self.assertTrue(CameraEvidenceBundle(self.root, "LEFT_UP_TOP").is_sealed)
        for c in ("RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP"):
            self.assertEqual(
                CameraEvidenceBundle(self.root, c).load_manifest().state,
                "PENDING")


if __name__ == "__main__":
    unittest.main()
