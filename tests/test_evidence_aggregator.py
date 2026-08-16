"""Unit tests for the sampled-frame evidence aggregator.

Covers the behaviours the sampled Door/Damage path depends on: spatial
grouping of a moving object, repeated-evidence acceptance, rejection of
isolated false positives, conflicting states, NO_DATA, snapshot selection,
and stride invariance.

No models, no video, no cv2 -- the aggregator is pure stdlib by design.
"""

from __future__ import annotations

import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from features.evidence_aggregator import (
    AggregationConfig, Candidate, EvidenceAggregator, Observation, iou,
)

W, H = 960, 540


def obs(frame, state, conf, cx, w=120, h=200, score=None):
    """A detection centred at cx on the given frame."""
    return Observation(
        frame_idx=frame, state=state, confidence=conf,
        bbox=(cx - w / 2, 150.0, cx + w / 2, 150.0 + h),
        score=conf if score is None else score,
    )


def feed(pairs, stride=1, cfg=None):
    agg = EvidenceAggregator(frame_width=W, frame_height=H, stride=stride,
                             config=cfg or AggregationConfig())
    for frame, dets in pairs:
        agg.add_frame(frame, dets)
    return agg.finalize()


class TestIoU(unittest.TestCase):
    def test_identical_boxes(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_disjoint_boxes(self):
        self.assertEqual(iou((0, 0, 10, 10), (50, 50, 60, 60)), 0.0)

    def test_partial_overlap(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3, places=5)


class TestSpatialGrouping(unittest.TestCase):
    def test_moving_object_stays_one_candidate(self):
        """A door translating across the frame must not split into many."""
        pairs = [(f, [obs(f, "CLOSED", 0.9, 150 + f * 25)]) for f in range(0, 10)]
        r = feed(pairs)
        self.assertEqual(len(r["groups"]), 1)
        self.assertEqual(r["groups"][0]["frame_support"], 10)

    def test_two_separated_objects_stay_separate(self):
        pairs = []
        for f in range(0, 6):
            pairs.append((f, [obs(f, "CLOSED", 0.9, 100 + f * 10),
                              obs(f, "CLOSED", 0.9, 800 + f * 10)]))
        r = feed(pairs)
        self.assertEqual(len(r["groups"]), 2)

    def test_teleport_beyond_gate_starts_new_candidate(self):
        pairs = [(0, [obs(0, "CLOSED", 0.9, 100)]),
                 (1, [obs(1, "CLOSED", 0.9, 900)])]   # 0.83 frame-widths in 1 frame
        r = feed(pairs)
        self.assertEqual(len(r["groups"]), 2)

    def test_two_boxes_on_same_frame_never_merge(self):
        r = feed([(0, [obs(0, "CLOSED", 0.9, 300), obs(0, "CLOSED", 0.9, 320)])])
        self.assertEqual(len(r["groups"]), 2)


class TestRepeatedEvidence(unittest.TestCase):
    def test_repeated_closed_is_accepted(self):
        pairs = [(f, [obs(f, "CLOSED", 0.9, 200 + f * 20)]) for f in range(6)]
        r = feed(pairs)
        self.assertEqual(len(r["accepted"]), 1)
        self.assertEqual(r["accepted"][0]["state"], "CLOSED")

    def test_repeated_open_is_accepted(self):
        pairs = [(f, [obs(f, "OPEN", 0.85, 200 + f * 20)]) for f in range(6)]
        r = feed(pairs)
        self.assertEqual(r["accepted"][0]["state"], "OPEN")

    def test_single_frame_detection_is_rejected(self):
        """One isolated hit is not evidence -- this is the false-positive guard."""
        r = feed([(3, [obs(3, "OPEN", 0.99, 400)])])
        self.assertEqual(len(r["accepted"]), 0)
        self.assertEqual(r["groups"][0]["detail"]["reason"],
                         "insufficient_frame_support")

    def test_partial_is_preserved_as_its_own_state(self):
        pairs = [(f, [obs(f, "PARTIAL", 0.8, 200 + f * 20)]) for f in range(5)]
        self.assertEqual(feed(pairs)["accepted"][0]["state"], "PARTIAL")


class TestConflictingStates(unittest.TestCase):
    def test_single_high_confidence_outlier_loses_to_repeated_evidence(self):
        """The scenario from the brief: one OPEN 0.94 among many CLOSED."""
        pairs = []
        for f in range(8):
            state, conf = ("OPEN", 0.94) if f == 3 else ("CLOSED", 0.89)
            pairs.append((f, [obs(f, state, conf, 200 + f * 20)]))
        r = feed(pairs)
        self.assertEqual(len(r["accepted"]), 1)
        self.assertEqual(r["accepted"][0]["state"], "CLOSED",
                         "a lone high-confidence OPEN must not override "
                         "repeated CLOSED evidence")

    def test_consistent_open_beats_sparse_closed(self):
        pairs = []
        for f in range(8):
            state, conf = ("CLOSED", 0.95) if f in (0, 7) else ("OPEN", 0.80)
            pairs.append((f, [obs(f, state, conf, 200 + f * 20)]))
        r = feed(pairs)
        self.assertEqual(r["accepted"][0]["state"], "OPEN",
                         "consistent OPEN evidence must win on frame support "
                         "even at lower confidence")

    def test_even_split_resolves_by_confidence_not_arbitrarily(self):
        pairs = []
        for f in range(6):
            state, conf = ("OPEN", 0.95) if f % 2 == 0 else ("CLOSED", 0.70)
            pairs.append((f, [obs(f, state, conf, 200 + f * 20)]))
        r = feed(pairs)
        self.assertEqual(r["accepted"][0]["state"], "OPEN")


class TestNoData(unittest.TestCase):
    def test_no_detections_yields_no_groups(self):
        r = feed([(f, []) for f in range(10)])
        self.assertEqual(r["groups"], [])
        self.assertEqual(r["accepted"], [])
        self.assertEqual(r["sampled_frame_count"], 10)

    def test_all_candidates_rejected_yields_no_accepted(self):
        r = feed([(0, [obs(0, "OPEN", 0.9, 100)]),
                  (5, [obs(5, "OPEN", 0.9, 800)])])
        self.assertEqual(len(r["accepted"]), 0)

    def test_support_fraction_rejects_one_stray_in_a_long_run(self):
        cfg = AggregationConfig(min_support_frames=1, min_support_fraction=0.5)
        pairs = [(f, [obs(f, "CLOSED", 0.9, 200 + f * 20)]) for f in range(10)]
        pairs.append((10, [obs(10, "OPEN", 0.99, 400)]))
        r = feed(pairs, cfg=cfg)
        self.assertEqual(r["accepted"][0]["state"], "CLOSED")


class TestSnapshotSelection(unittest.TestCase):
    def test_best_observation_uses_caller_score(self):
        pairs = [(0, [obs(0, "CLOSED", 0.80, 200, score=0.1)]),
                 (1, [obs(1, "CLOSED", 0.70, 220, score=9.9)]),
                 (2, [obs(2, "CLOSED", 0.90, 240, score=0.2)])]
        best = feed(pairs)["accepted"][0]["best"]
        self.assertEqual(best.frame_idx, 1,
                         "snapshot must follow the caller's score, not raw conf")

    def test_best_observation_matches_resolved_state(self):
        pairs = []
        for f in range(6):
            state, conf = ("OPEN", 0.99) if f == 2 else ("CLOSED", 0.85)
            pairs.append((f, [obs(f, state, conf, 200 + f * 20)]))
        g = feed(pairs)["accepted"][0]
        self.assertEqual(g["state"], "CLOSED")
        self.assertEqual(g["best"].state, "CLOSED",
                         "snapshot must depict the state actually reported")

    def test_one_snapshot_per_candidate(self):
        pairs = [(f, [obs(f, "CLOSED", 0.9, 150 + f * 20)]) for f in range(8)]
        acc = feed(pairs)["accepted"]
        self.assertEqual(len(acc), 1)
        self.assertIsNotNone(acc[0]["best"])


class TestStrideInvariance(unittest.TestCase):
    """The property that makes sampling safe."""

    def _pairs(self, stride):
        return [(f, [obs(f, "CLOSED", 0.9, 150 + f * 15)])
                for f in range(0, 20, stride)]

    def test_same_verdict_at_stride_1_and_2(self):
        a = feed(self._pairs(1), stride=1)
        b = feed(self._pairs(2), stride=2)
        self.assertEqual(len(a["accepted"]), len(b["accepted"]), 1)
        self.assertEqual(a["accepted"][0]["state"], b["accepted"][0]["state"])

    def test_stride_2_halves_sampled_frames(self):
        a = feed(self._pairs(1), stride=1)
        b = feed(self._pairs(2), stride=2)
        self.assertEqual(a["sampled_frame_count"], 20)
        self.assertEqual(b["sampled_frame_count"], 10)

    def test_original_frame_indices_are_preserved(self):
        r = feed(self._pairs(2), stride=2)
        self.assertEqual(r["sampled_frames"], list(range(0, 20, 2)))
        self.assertTrue(all(f % 2 == 0 for f in r["sampled_frames"]))

    def test_no_duplicate_sampled_frames(self):
        r = feed(self._pairs(2), stride=2)
        self.assertEqual(len(set(r["sampled_frames"])), len(r["sampled_frames"]))

    def test_moving_object_still_groups_at_stride_2(self):
        """Drift gate scales with the gap, so faster apparent motion is fine."""
        r = feed(self._pairs(2), stride=2)
        self.assertEqual(len(r["groups"]), 1)


class TestCandidateInternals(unittest.TestCase):
    def test_frame_support_counts_distinct_frames_only(self):
        c = Candidate()
        c.add(obs(0, "CLOSED", 0.9, 100))
        c.add(obs(0, "CLOSED", 0.9, 105))     # same frame
        c.add(obs(1, "CLOSED", 0.9, 120))
        self.assertEqual(c.frame_support, 2)

    def test_state_evidence_reports_per_state_stats(self):
        c = Candidate()
        for f in range(4):
            c.add(obs(f, "CLOSED", 0.8 + f * 0.01, 100 + f * 10))
        c.add(obs(4, "OPEN", 0.99, 140))
        ev = c.state_evidence()
        self.assertEqual(ev["CLOSED"]["frames"], 4)
        self.assertEqual(ev["OPEN"]["frames"], 1)
        self.assertAlmostEqual(ev["OPEN"]["max_conf"], 0.99)


if __name__ == "__main__":
    unittest.main()
