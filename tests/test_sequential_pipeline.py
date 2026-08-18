"""Sequential-mode pipeline tests.

Model-free by construction: they exercise the real lifecycle, real persistence
and the real assembler against synthetic bundles on a real filesystem. The
only stubbing is a counting fake standing in for a YOLO model where the point
IS to count calls.

The four-arrival simulation with real weights lives in
benchmarks/run_sequential_arrivals.py -- it needs the models and ~40 minutes,
so it is not part of the default suite.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from core.camera_evidence import (
    FAILED, LIFECYCLE, CameraEvidenceBundle, LocalSegment, local_segment_id,
    map_segments_to_global, ready_for_global_assembly,
)

CAMS = ("RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP", "LEFT_UP_TOP")
MASTER = "RIGHT_UP"


def _parse(argv):
    from orchestrator.master_runner import _build_parser
    return _build_parser().parse_args(argv)


def _seal(root, cam, *, segments=None, features=None):
    """Create a fully SEALED bundle on disk, as camera_runner would."""
    b = CameraEvidenceBundle(root, cam)
    os.makedirs(b.dir, exist_ok=True)
    for s in LIFECYCLE[1:]:
        b.advance(s)
    segs = segments if segments is not None else [
        LocalSegment(local_id=local_segment_id(cam, i), index=i,
                     start_frame=i * 60, end_frame=i * 60 + 59,
                     start_time=float((i - 1) * 4), end_time=float(i * 4),
                     label="WAGON", confidence=0.9)
        for i in range(1, 4)
    ]
    b.write_segments(segs)
    with open(os.path.join(b.dir, f"{cam}_report.pdf"), "wb") as f:
        f.write(b"%PDF-1.4 stub")
    for feat, payload in (features or {}).items():
        d = os.path.join(b.dir, "features", feat)
        os.makedirs(d, exist_ok=True)
        for s in segs:
            with open(os.path.join(d, f"{s.local_id}.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"global_id": s.local_id, "feature": feat,
                           "status": "OK", **payload}, f)
    return b, segs


class TestCliDispatch(unittest.TestCase):
    def test_batch_is_the_default(self):
        self.assertEqual(_parse(["--local-only"]).mode, "batch")
        self.assertEqual(_parse([]).mode, "batch")

    def test_sequential_parses(self):
        self.assertEqual(_parse(["--mode", "sequential"]).mode, "sequential")

    def test_only_two_modes_accepted(self):
        with self.assertRaises(SystemExit):
            _parse(["--mode", "turbo"])

    def test_sequential_dispatches_before_process_batch(self):
        from orchestrator import master_runner
        src = inspect.getsource(master_runner.main)
        self.assertLess(src.index('args.mode == "sequential"'),
                        src.index("run_local("),
                        "sequential must be checked before the batch path")

    def test_sequential_path_does_not_call_process_batch(self):
        from orchestrator.master_runner import run_sequential
        src = inspect.getsource(run_sequential)
        self.assertNotIn("process_batch", src)
        self.assertIn("camera_runner", src)
        self.assertIn("global_assembler", src)

    def test_batch_entry_point_still_exists_unchanged_in_shape(self):
        from orchestrator.master_runner import process_batch
        p = inspect.signature(process_batch).parameters
        self.assertIn("batch", p)
        self.assertIn("workspace_root", p)
        self.assertEqual(p["door_sample_stride"].default, 3)
        self.assertEqual(p["damage_sample_stride"].default, 3)
        self.assertEqual(p["load_sample_stride"].default, 2)


class TestStrideConfigUnchanged(unittest.TestCase):
    def test_camera_runner_uses_3_3_2(self):
        from orchestrator import camera_runner as cr
        self.assertEqual((cr.DOOR_STRIDE, cr.DAMAGE_STRIDE, cr.LOAD_STRIDE),
                         (3, 3, 2))


class TestArrivalPersistence(unittest.TestCase):
    """Genuine filesystem state, one arrival at a time."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_camera_completes_without_the_others(self):
        _seal(self.root, MASTER)
        self.assertEqual(
            CameraEvidenceBundle(self.root, MASTER).load_manifest().state,
            "SEALED")
        for cam in CAMS[1:]:
            self.assertFalse(os.path.exists(os.path.join(self.root, cam)),
                             f"{cam} must not exist yet")

    def test_state_persists_across_reopen(self):
        """A new object reading the same directory sees the sealed state --
        this is what makes separate processes work."""
        _seal(self.root, MASTER)
        reopened = CameraEvidenceBundle(self.root, MASTER)
        self.assertTrue(reopened.is_sealed)
        self.assertEqual(len(reopened.read_segments()), 3)

    def test_each_report_exists_before_the_next_arrival(self):
        arrived = []
        for cam in CAMS:
            _seal(self.root, cam)
            arrived.append(cam)
            for done in arrived:
                self.assertTrue(
                    os.path.isfile(os.path.join(self.root, done,
                                                f"{done}_report.pdf")),
                    f"{done}_report.pdf must exist before the next arrival")
            for pending in [c for c in CAMS if c not in arrived]:
                self.assertFalse(os.path.exists(os.path.join(self.root, pending)))

    def test_all_four_local_pdfs_after_four_arrivals(self):
        for cam in CAMS:
            _seal(self.root, cam)
        for cam in CAMS:
            self.assertTrue(os.path.isfile(
                os.path.join(self.root, cam, f"{cam}_report.pdf")))


class TestAssemblyGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_blocked_until_the_fourth_camera(self):
        for i, cam in enumerate(CAMS):
            if i:
                ok, _ = ready_for_global_assembly(self.root, MASTER, CAMS)
                self.assertFalse(ok, f"assembly ran with only {i} camera(s)")
            _seal(self.root, cam)
        ok, _ = ready_for_global_assembly(self.root, MASTER, CAMS)
        self.assertTrue(ok)

    def test_failed_support_does_not_block(self):
        _seal(self.root, MASTER)
        CameraEvidenceBundle(self.root, "LEFT_UP").fail("video missing")
        for cam in ("RIGHT_UP_TOP", "LEFT_UP_TOP"):
            _seal(self.root, cam)
        ok, _ = ready_for_global_assembly(self.root, MASTER, CAMS)
        self.assertTrue(ok)

    def test_failed_master_blocks(self):
        CameraEvidenceBundle(self.root, MASTER).fail("gap model missing")
        for cam in CAMS[1:]:
            _seal(self.root, cam)
        ok, why = ready_for_global_assembly(self.root, MASTER, CAMS)
        self.assertFalse(ok)
        self.assertIn("FAILED", why)

    def test_assembler_refuses_when_not_ready(self):
        from orchestrator.global_assembler import assemble
        r = assemble(evidence_root=self.root, output_root=self.root,
                     batch_key="t", verbose=False)
        self.assertFalse(r.ready)
        self.assertEqual(r.total_wagons, 0)


class TestRelabelling(unittest.TestCase):
    """L_<CAM>_<n> -> GW_n on real files, with no inference."""

    class _GW:
        def __init__(self, gid, s, e):
            self.global_id, self.start_time, self.end_time = gid, s, e

    def test_feature_evidence_is_relabelled_to_global_ids(self):
        from orchestrator.global_assembler import _relabel_feature_evidence
        with tempfile.TemporaryDirectory() as root:
            b, segs = _seal(root, MASTER,
                            features={"door": {"left_door": "CLOSED"}})
            roster = [self._GW(f"GW_{i}", float((i - 1) * 4), float(i * 4))
                      for i in range(1, 4)]
            maps = map_segments_to_global(segs, roster, camera_id=MASTER)
            states = os.path.join(root, "wagon_states")
            n = _relabel_feature_evidence(b, maps, states)
            self.assertEqual(n, 3)
            for i in range(1, 4):
                p = os.path.join(states, "door", f"GW_{i}.json")
                self.assertTrue(os.path.isfile(p), f"GW_{i}.json missing")
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                self.assertEqual(d["global_id"], f"GW_{i}")
                self.assertTrue(d["_sequential_audit"]["source_local_id"]
                                .startswith("L_RIGHT_UP_"))
                self.assertEqual(d["left_door"], "CLOSED",
                                 "payload must survive relabelling")

    def test_unmatched_segment_is_never_relabelled(self):
        from orchestrator.global_assembler import _relabel_feature_evidence
        with tempfile.TemporaryDirectory() as root:
            far = [LocalSegment(local_id=local_segment_id(MASTER, 1), index=1,
                                start_frame=0, end_frame=10,
                                start_time=900.0, end_time=904.0,
                                label="WAGON", confidence=0.9)]
            b, segs = _seal(root, MASTER, segments=far,
                            features={"door": {"left_door": "OPEN"}})
            roster = [self._GW("GW_1", 0.0, 4.0)]
            maps = map_segments_to_global(segs, roster, camera_id=MASTER)
            states = os.path.join(root, "wagon_states")
            self.assertEqual(_relabel_feature_evidence(b, maps, states), 0)
            self.assertFalse(os.path.exists(os.path.join(states, "door")))


class TestNoInferenceDuringAssembly(unittest.TestCase):
    def test_assembler_source_invokes_no_detector(self):
        from orchestrator import global_assembler as ga
        src = inspect.getsource(ga)
        for banned in ("door_proc", "damage_proc", "load_proc",
                       "load_yolo", "ultralytics", "from features.door",
                       "from features.damage", "from features.load"):
            with self.subTest(token=banned):
                self.assertNotIn(banned, src)

    def test_relabelling_only_moves_files(self):
        """No model object is ever constructed or called during relabelling."""
        from orchestrator.global_assembler import _relabel_feature_evidence
        src = inspect.getsource(_relabel_feature_evidence)
        parts = src.split('"""')          # drop the docstring: it says
        code = parts[0] + "".join(parts[2:])   # "No inference." in prose
        for banned in ("YOLO", "predict", "infer", "model("):
            with self.subTest(token=banned):
                self.assertNotIn(banned, code)


class TestProtectedAndAdditive(unittest.TestCase):
    def test_protected_packages_unmodified(self):
        import subprocess
        r = subprocess.run(["git", "diff", "--name-only", "HEAD",
                            "wagon_count", "reconstruction", "fusion"],
                           cwd=V4_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("git unavailable")
        self.assertEqual([x for x in r.stdout.split() if x], [])

    def test_reporting_gains_only_the_new_camera_local_file(self):
        import subprocess
        r = subprocess.run(["git", "diff", "--name-only", "HEAD", "reporting"],
                           cwd=V4_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("git unavailable")
        self.assertEqual([x for x in r.stdout.split() if x], [],
                         "existing reporting builders must be unmodified")
        self.assertTrue(os.path.isfile(
            os.path.join(V4_ROOT, "reporting", "camera_local_report.py")))


if __name__ == "__main__":
    unittest.main()
