"""Inspection must never modify the finalized global wagon roster.

Not append, not remove, not renumber, not reorder, not re-time.  Enforced
three ways: the type system (frozen wagons, tuple roster), a fingerprint the
orchestrator checks after every inspection stage, and a structural validator.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import unittest

from _engine_harness import (
    CAMERA_LEFT_UP, V4_ROOT, as_v4_state, drifting_gap_times,
    run_counting_engine,
)
from core import constants as C
from core.global_state_loader import (
    GlobalTrainState, GlobalWagon, RosterImmutabilityError,
    assert_roster_unchanged, roster_fingerprint, verify_roster_integrity,
)


def _roster():
    times = drifting_gap_times(9, start=24.0)
    return as_v4_state(run_counting_engine(
        times, {CAMERA_LEFT_UP: [t - 0.8 for t in times]})[0])


class TestRosterIsStructurallyImmutable(unittest.TestCase):
    def setUp(self):
        self.state = _roster()

    def test_wagon_is_a_frozen_dataclass(self):
        self.assertTrue(dataclasses.fields(GlobalWagon))
        self.assertTrue(GlobalWagon.__dataclass_params__.frozen)

    def test_cannot_edit_a_wagon(self):
        w = self.state.wagons[0]
        for attr, value in (("global_id", "GW_999"), ("wagon_index", 42),
                            ("start_time", -1.0), ("classification", "ENGINE")):
            with self.subTest(attr=attr):
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    setattr(w, attr, value)

    def test_roster_container_is_a_tuple(self):
        self.assertIsInstance(self.state.wagons, tuple)
        for op in ("append", "insert", "pop", "remove", "sort", "reverse"):
            self.assertFalse(hasattr(self.state.wagons, op),
                             f"roster exposes mutating operation {op}()")

    def test_cannot_append_or_reorder(self):
        with self.assertRaises(AttributeError):
            self.state.wagons.append(self.state.wagons[0])
        with self.assertRaises(TypeError):
            self.state.wagons[0] = self.state.wagons[-1]

    def test_hand_built_state_is_coerced_to_an_immutable_roster(self):
        s = GlobalTrainState(total_wagons=1, wagons=[self.state.wagons[0]])
        self.assertIsInstance(s.wagons, tuple)


class TestFingerprintGuard(unittest.TestCase):
    def setUp(self):
        self.state = _roster()
        self.fp = roster_fingerprint(self.state)

    def test_unchanged_roster_passes(self):
        assert_roster_unchanged(self.state, self.fp, stage="unit test")

    def test_detects_removal(self):
        tampered = dataclasses.replace(self.state, wagons=self.state.wagons[:-1])
        with self.assertRaises(RosterImmutabilityError):
            assert_roster_unchanged(tampered, self.fp, stage="unit test")

    def test_detects_append(self):
        extra = dataclasses.replace(self.state.wagons[-1],
                                    global_id="GW_EXTRA", wagon_index=99)
        tampered = dataclasses.replace(
            self.state, wagons=self.state.wagons + (extra,))
        with self.assertRaises(RosterImmutabilityError):
            assert_roster_unchanged(tampered, self.fp, stage="unit test")

    def test_detects_reorder(self):
        rev = tuple(reversed(self.state.wagons))
        tampered = dataclasses.replace(self.state, wagons=rev)
        with self.assertRaises(RosterImmutabilityError):
            assert_roster_unchanged(tampered, self.fp, stage="unit test")

    def test_detects_renumber(self):
        first = dataclasses.replace(self.state.wagons[0], global_id="GW_0")
        tampered = dataclasses.replace(
            self.state, wagons=(first,) + self.state.wagons[1:])
        with self.assertRaises(RosterImmutabilityError):
            assert_roster_unchanged(tampered, self.fp, stage="unit test")

    def test_detects_boundary_edit(self):
        first = dataclasses.replace(self.state.wagons[0],
                                    end_time=self.state.wagons[0].end_time + 5.0)
        tampered = dataclasses.replace(
            self.state, wagons=(first,) + self.state.wagons[1:])
        with self.assertRaises(RosterImmutabilityError):
            assert_roster_unchanged(tampered, self.fp, stage="unit test")


class TestIntegrityValidatorCatchesCorruption(unittest.TestCase):
    def setUp(self):
        self.state = _roster()

    def test_duplicate_id_is_reported(self):
        dup = self.state.wagons[:2] + (self.state.wagons[0],)
        bad = dataclasses.replace(self.state, wagons=dup, total_wagons=3)
        self.assertTrue(any("duplicate" in p
                            for p in verify_roster_integrity(bad)))

    def test_gap_in_numbering_is_reported(self):
        skipped = self.state.wagons[:1] + self.state.wagons[2:]
        bad = dataclasses.replace(self.state, wagons=skipped,
                                  total_wagons=len(skipped))
        self.assertTrue(any("non-contiguous" in p
                            for p in verify_roster_integrity(bad)))

    def test_count_mismatch_is_reported(self):
        bad = dataclasses.replace(self.state,
                                  total_wagons=self.state.total_wagons + 1)
        self.assertTrue(any("total_wagons" in p
                            for p in verify_roster_integrity(bad)))


class TestRealInspectionLeavesRosterUntouched(unittest.TestCase):
    def test_stage4_fusion_does_not_touch_the_roster(self):
        """Run the REAL inspection fusion stage over the roster."""
        from fusion import wagon_state_builder

        state = _roster()
        fp = roster_fingerprint(state)
        with tempfile.TemporaryDirectory() as tmp:
            for feature in ("door", "ocr", "load", "damage"):
                d = os.path.join(tmp, feature)
                os.makedirs(d, exist_ok=True)
                for w in state.wagons:
                    with open(os.path.join(d, f"{w.global_id}.json"), "w",
                              encoding="utf-8") as f:
                        json.dump({"global_id": w.global_id,
                                   "feature": feature,
                                   "status": C.STATUS_OK,
                                   "supporting_cameras": [C.CAMERA_RIGHT_UP]}, f)
            unified = wagon_state_builder.build(
                state=state, wagon_states_root=tmp, verbose=False)

        assert_roster_unchanged(state, fp, stage="Stage 4 (fusion)")
        self.assertEqual(set(unified), {w.global_id for w in state.wagons})

    def test_orchestrator_guards_every_inspection_stage(self):
        with open(os.path.join(V4_ROOT, "orchestrator", "master_runner.py"),
                  "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("assert_roster_unchanged", src)
        for stage in ("Stage 2 (materializer)", "Stage 3 (feature inference)",
                      "Stage 4 (fusion)", "Stage 5 (reporting)"):
            with self.subTest(stage=stage):
                self.assertIn(stage, src,
                              f"no roster immutability guard around {stage}")


if __name__ == "__main__":
    unittest.main()
