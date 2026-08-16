"""Wiring tests for the EXPERIMENTAL sampled inference modes.

These assert the safety property that matters most right now: **production
behaviour is unchanged by construction**.  The default is "legacy" on both
processors, and the orchestrator never passes the flag at all, so the sampled
path is unreachable unless a benchmark asks for it explicitly.

Model-dependent behaviour (detection quality, per-wagon states) is covered by
the benchmark harness, not here -- these tests must run without weights.
"""

from __future__ import annotations

import inspect
import os
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from features.damage import processor as damage_proc
from features.door import processor as door_proc
from features.evidence_aggregator import EvidenceAggregator, Observation

PROCESSORS = (("door", door_proc), ("damage", damage_proc))


class TestDefaultsAreLegacy(unittest.TestCase):
    def test_both_processors_default_to_legacy(self):
        for name, mod in PROCESSORS:
            with self.subTest(feature=name):
                self.assertEqual(
                    inspect.signature(mod.run).parameters["inference_mode"].default,
                    "legacy",
                    f"{name} must default to the known-good legacy path")

    def test_sample_stride_default_is_two(self):
        for name, mod in PROCESSORS:
            with self.subTest(feature=name):
                self.assertEqual(
                    inspect.signature(mod.run).parameters["sample_stride"].default, 2)

    def test_legacy_implementation_is_retained(self):
        """The tracker path must still exist -- sampled mode is additive."""
        for name, mod in PROCESSORS:
            with self.subTest(feature=name):
                self.assertTrue(hasattr(mod, "_run_tracker_one_camera"))
                self.assertTrue(hasattr(mod, "_run_sampled_one_camera"))

    def test_invalid_mode_is_rejected_loudly(self):
        for name, mod in PROCESSORS:
            with self.subTest(feature=name):
                with self.assertRaises(ValueError):
                    mod.run(state=None, cache_root="", feature_models_dir="",
                            output_dir="", inference_mode="turbo")


class TestOrchestratorDoesNotEnableSampling(unittest.TestCase):
    """Production must not be able to drift into sampled mode by accident."""

    def test_master_runner_never_passes_the_flag(self):
        p = os.path.join(V4_ROOT, "orchestrator", "master_runner.py")
        with open(p, "r", encoding="utf-8") as f:
            src = f.read()
        for token in ("inference_mode", "sample_stride", "sampled"):
            with self.subTest(token=token):
                self.assertNotIn(token, src)


class TestSampledPathContract(unittest.TestCase):
    """The sampled helpers must return the SAME tuple shape as legacy, so the
    surrounding run() -- JSON, evidence, snapshots -- needs no branching."""

    def test_door_sampled_returns_six_tuple_on_empty_cache(self):
        got = door_proc._run_sampled_one_camera(
            None, door_proc.TrackerConfig(), "/nonexistent", "GW_1",
            "RIGHT_UP", sample_stride=2)
        self.assertEqual(len(got), 6)
        decisions, used, w, h, cands, overlay = got
        self.assertEqual((decisions, used, w, h), ([], 0, 0, 0))
        self.assertEqual(cands, {})
        self.assertEqual(set(overlay), {"tracks", "events"})

    def test_damage_sampled_returns_five_tuple_on_empty_cache(self):
        got = damage_proc._run_sampled_one_camera(
            None, damage_proc.DamageTrackerConfig(), "/nonexistent", "GW_1",
            "RIGHT_UP_TOP", confidence_floor=0.55, sample_stride=2)
        self.assertEqual(len(got), 5)
        self.assertEqual(got, ([], 0, 0, 0, []))

    def test_door_sampled_tuple_matches_legacy_arity(self):
        empty_legacy = door_proc._run_tracker_one_camera(
            None, door_proc.TrackerConfig(), door_proc.MergeConfig(),
            "/nonexistent", "GW_1", "RIGHT_UP")
        empty_sampled = door_proc._run_sampled_one_camera(
            None, door_proc.TrackerConfig(), "/nonexistent", "GW_1", "RIGHT_UP")
        self.assertEqual(len(empty_legacy), len(empty_sampled))


class TestAggregatorDrivesTheDecision(unittest.TestCase):
    """The decision records the sampled path emits must carry the fields
    `_pick_side_state` relies on, with frame support standing in for hits."""

    def _agg(self, states):
        agg = EvidenceAggregator(frame_width=960, frame_height=540, stride=2)
        for i, st in enumerate(states):
            agg.add_frame(i * 2, [Observation(
                frame_idx=i * 2, state=st, confidence=0.9,
                bbox=(100.0, 100.0, 300.0, 250.0), score=1.0)])
        return agg.finalize()

    def test_repeated_evidence_produces_an_accepted_group(self):
        res = self._agg(["CLOSED"] * 6)
        self.assertTrue(res["accepted"])
        g = res["accepted"][0]
        for key in ("candidate_id", "state", "confidence", "frame_support",
                    "first_frame", "last_frame", "best"):
            self.assertIn(key, g)
        self.assertEqual(g["state"], "CLOSED")
        self.assertEqual(g["frame_support"], 6)

    def test_lone_outlier_does_not_win(self):
        res = self._agg(["CLOSED"] * 6 + ["OPEN"])
        self.assertEqual(res["accepted"][0]["state"], "CLOSED")

    def test_frame_support_is_stride_invariant_in_fraction(self):
        """The reason stride-2 broke the tracker but not the aggregator."""
        dense = self._agg(["CLOSED"] * 8)["accepted"][0]
        sparse = self._agg(["CLOSED"] * 4)["accepted"][0]
        self.assertEqual(dense["state"], sparse["state"])


if __name__ == "__main__":
    unittest.main()
