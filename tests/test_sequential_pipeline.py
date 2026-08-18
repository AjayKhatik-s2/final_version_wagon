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

from _engine_harness import V4_ROOT, changed_paths  # noqa: F401

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


class TestFeatureAuthorityAndOcr(unittest.TestCase):
    """Each camera runs exactly the features it is authoritative for."""

    def _plan(self, cam, enabled=("door", "damage", "load", "ocr")):
        from orchestrator.camera_runner import _feature_plan
        return [name for name, _mod, _extra in _feature_plan(cam, set(enabled))]

    def test_side_cameras_run_door_only(self):
        for cam in C.SIDE_CAMERAS:
            with self.subTest(camera=cam):
                self.assertEqual(self._plan(cam), ["door"])

    def test_top_cameras_run_load_then_damage(self):
        """Order matters: damage's loaded-wagon filter reads load's JSON."""
        for cam in C.TOP_CAMERAS:
            with self.subTest(camera=cam):
                self.assertEqual(self._plan(cam), ["load", "damage"])

    def test_ocr_is_never_scheduled_even_if_requested(self):
        for cam in CAMS:
            with self.subTest(camera=cam):
                self.assertNotIn("ocr", self._plan(cam))

    def test_no_ocr_anywhere_in_the_sequential_path(self):
        import ast
        from orchestrator import (camera_report_adapter, camera_runner,
                                  global_assembler)
        for mod in (camera_runner, global_assembler, camera_report_adapter):
            tree = ast.parse(inspect.getsource(mod))
            names = {a.name for n in ast.walk(tree)
                     if isinstance(n, (ast.Import, ast.ImportFrom))
                     for a in n.names}
            if isinstance(getattr(mod, "__name__", ""), str):
                for n in ast.walk(tree):
                    if isinstance(n, ast.ImportFrom) and n.module:
                        names.add(n.module)
            with self.subTest(module=mod.__name__):
                self.assertFalse([x for x in names if "ocr" in x.lower()],
                                 f"{mod.__name__} imports an OCR module")

    def test_strides_are_the_approved_values(self):
        from orchestrator.camera_runner import _feature_plan
        got = {}
        for cam in CAMS:
            for name, _mod, extra in _feature_plan(cam, {"door", "damage",
                                                         "load"}):
                got[name] = extra.get("sample_stride")
        self.assertEqual(got, {"door": 3, "damage": 3, "load": 2})

    def test_every_feature_uses_the_sampled_mode(self):
        from orchestrator.camera_runner import _feature_plan
        for cam in CAMS:
            for name, _mod, extra in _feature_plan(cam, {"door", "damage",
                                                         "load"}):
                with self.subTest(camera=cam, feature=name):
                    self.assertEqual(extra.get("inference_mode"), "sampled")


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


class TestCrossCameraFeatureMerge(unittest.TestCase):
    """Each camera reports only its own half; assembly must combine them.

    In batch mode one processor invocation read every relevant camera. In
    sequential mode RIGHT_UP fills right_door and leaves left_door NO_DATA,
    LEFT_UP the reverse, and each top camera reports only what it saw. Keeping
    the first writer would silently drop half the readings.
    """

    def _merge(self, feature, base, new):
        from orchestrator.global_assembler import _merge_payloads
        return _merge_payloads(feature, base, new)

    def _door(self, cam, *, left="NO_DATA", right="NO_DATA",
              lc=0.0, rc=0.0, fl=0, fr=0):
        return {"global_id": "GW_1", "feature": "door", "status": "OK",
                "left_door": left, "left_door_confidence": lc,
                "right_door": right, "right_door_confidence": rc,
                "frames_left": fl, "frames_right": fr,
                "frame_count": fl + fr, "tracks": [],
                "supporting_cameras": [cam]}

    def test_both_door_sides_survive(self):
        r = self._door("RIGHT_UP", right="OPEN", rc=0.88, fr=9)
        l = self._door("LEFT_UP", left="CLOSED", lc=0.91, fl=7)
        m = self._merge("door", r, l)
        self.assertEqual(m["right_door"], "OPEN")
        self.assertEqual(m["left_door"], "CLOSED")
        self.assertAlmostEqual(m["left_door_confidence"], 0.91)
        self.assertAlmostEqual(m["right_door_confidence"], 0.88)
        self.assertEqual(sorted(m["supporting_cameras"]),
                         ["LEFT_UP", "RIGHT_UP"])
        self.assertEqual(m["frame_count"], 16)

    def test_merge_order_does_not_matter_for_doors(self):
        r = self._door("RIGHT_UP", right="OPEN", rc=0.88, fr=9)
        l = self._door("LEFT_UP", left="CLOSED", lc=0.91, fl=7)
        a = self._merge("door", r, l)
        b = self._merge("door", l, r)
        for k in ("left_door", "right_door", "left_door_confidence",
                  "right_door_confidence", "frame_count"):
            self.assertEqual(a[k], b[k], k)

    def test_a_real_reading_never_loses_to_no_data(self):
        r = self._door("RIGHT_UP", right="OPEN", rc=0.88)
        l = self._door("LEFT_UP", left="CLOSED", lc=0.91)
        self.assertEqual(self._merge("door", r, l)["left_door"], "CLOSED")
        self.assertEqual(self._merge("door", l, r)["right_door"], "OPEN")

    def test_conflicting_same_side_keeps_higher_confidence(self):
        a = self._door("RIGHT_UP", right="CLOSED", rc=0.55)
        b = self._door("RIGHT_UP", right="OPEN", rc=0.93)
        self.assertEqual(self._merge("door", a, b)["right_door"], "OPEN")
        self.assertEqual(self._merge("door", b, a)["right_door"], "OPEN")

    def _load(self, cam, status, conf):
        return {"global_id": "GW_1", "feature": "load", "status": "OK",
                "load_status": status, "load_confidence": conf,
                "supporting_cameras": [cam]}

    def test_right_up_top_is_authoritative_for_load(self):
        """features/load/processor.py: RIGHT_UP_TOP authoritative when present."""
        r = self._load("RIGHT_UP_TOP", "LOADED", 0.7)
        l = self._load("LEFT_UP_TOP", "EMPTY", 0.99)
        self.assertEqual(self._merge("load", r, l)["load_status"], "LOADED")
        self.assertEqual(self._merge("load", l, r)["load_status"], "LOADED")

    def test_left_up_top_is_the_fallback_for_load(self):
        r = self._load("RIGHT_UP_TOP", "NO_DATA", 0.0)
        l = self._load("LEFT_UP_TOP", "EMPTY", 0.8)
        self.assertEqual(self._merge("load", r, l)["load_status"], "EMPTY")

    def _dmg(self, cam, status, details=None, conf=0.0):
        return {"global_id": "GW_1", "feature": "damage", "status": "OK",
                "top_damage": status, "top_damage_confidence": conf,
                "top_damage_details": list(details or []),
                "supporting_cameras": [cam]}

    def test_any_top_camera_damage_wins(self):
        """features/damage/processor.py combines with `any_damage`."""
        ok = self._dmg("RIGHT_UP_TOP", "OK")
        bad = self._dmg("LEFT_UP_TOP", "DAMAGE", ["floor_damage"], 0.8)
        self.assertEqual(self._merge("damage", ok, bad)["top_damage"], "DAMAGE")
        self.assertEqual(self._merge("damage", bad, ok)["top_damage"], "DAMAGE")

    def test_damage_details_from_both_cameras_are_kept(self):
        a = self._dmg("RIGHT_UP_TOP", "DAMAGE", ["floor_damage"], 0.7)
        b = self._dmg("LEFT_UP_TOP", "DAMAGE", ["inner_wall_damage"], 0.6)
        m = self._merge("damage", a, b)
        self.assertEqual(sorted(m["top_damage_details"]),
                         ["floor_damage", "inner_wall_damage"])

    def test_damage_no_data_is_replaced_by_a_real_ok(self):
        nd = self._dmg("RIGHT_UP_TOP", "NO_DATA")
        ok = self._dmg("LEFT_UP_TOP", "OK")
        self.assertEqual(self._merge("damage", nd, ok)["top_damage"], "OK")

    def test_merge_invents_no_reading(self):
        nd1 = self._dmg("RIGHT_UP_TOP", "NO_DATA")
        nd2 = self._dmg("LEFT_UP_TOP", "NO_DATA")
        self.assertEqual(self._merge("damage", nd1, nd2)["top_damage"],
                         "NO_DATA")

    def test_merge_runs_no_inference(self):
        """It only chooses between values already computed on disk."""
        import ast
        from orchestrator import global_assembler as ga
        fn = next(n for n in ast.walk(ast.parse(inspect.getsource(ga)))
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_merge_payloads")
        calls = {ast.unparse(n.func) for n in ast.walk(fn)
                 if isinstance(n, ast.Call)}
        self.assertTrue(calls, "found no calls -- did the parse work?")
        for c in calls:
            for banned in ("YOLO", "predict", "model", "infer"):
                self.assertNotIn(banned, c, f"{c} looks like inference")

    def test_on_disk_relabelling_merges_two_cameras(self):
        """End to end through the real file path, not just the merge helper."""
        from orchestrator.global_assembler import _relabel_feature_evidence

        class _GW:
            def __init__(self, gid, s, e):
                self.global_id, self.start_time, self.end_time = gid, s, e

        with tempfile.TemporaryDirectory() as root:
            states = os.path.join(root, "wagon_states")
            roster = [_GW(f"GW_{i}", float((i - 1) * 4), float(i * 4))
                      for i in range(1, 4)]
            for cam, feats in (
                ("RIGHT_UP", {"door": {"right_door": "OPEN",
                                       "right_door_confidence": 0.88,
                                       "left_door": "NO_DATA",
                                       "left_door_confidence": 0.0,
                                       "supporting_cameras": ["RIGHT_UP"],
                                       "frames_right": 9, "frames_left": 0}}),
                ("LEFT_UP", {"door": {"left_door": "CLOSED",
                                      "left_door_confidence": 0.91,
                                      "right_door": "NO_DATA",
                                      "right_door_confidence": 0.0,
                                      "supporting_cameras": ["LEFT_UP"],
                                      "frames_left": 7, "frames_right": 0}}),
            ):
                b, segs = _seal(root, cam, features=feats)
                maps = map_segments_to_global(segs, roster, camera_id=cam)
                _relabel_feature_evidence(b, maps, states)
            for i in range(1, 4):
                with open(os.path.join(states, "door", f"GW_{i}.json"),
                          encoding="utf-8") as f:
                    d = json.load(f)
                self.assertEqual(d["right_door"], "OPEN", f"GW_{i}")
                self.assertEqual(d["left_door"], "CLOSED", f"GW_{i}")
                self.assertEqual(sorted(d["supporting_cameras"]),
                                 ["LEFT_UP", "RIGHT_UP"])
                self.assertTrue(d["_sequential_audit"]["merged_from"])

    def test_fused_state_reports_both_doors_after_merge(self):
        """The payoff: wagon_state_builder sees both sides, not one."""
        from orchestrator.global_assembler import _relabel_feature_evidence
        from fusion import wagon_state_builder
        from core.global_state_loader import GlobalTrainState, GlobalWagon

        class _GW:
            def __init__(self, gid, s, e):
                self.global_id, self.start_time, self.end_time = gid, s, e

        with tempfile.TemporaryDirectory() as root:
            states = os.path.join(root, "wagon_states")
            roster = [_GW("GW_1", 0.0, 4.0)]
            for cam, feats in (
                ("RIGHT_UP", {"door": {"right_door": "OPEN",
                                       "right_door_confidence": 0.88,
                                       "left_door": "NO_DATA",
                                       "left_door_confidence": 0.0,
                                       "supporting_cameras": ["RIGHT_UP"],
                                       "frames_right": 9, "frames_left": 0}}),
                ("LEFT_UP", {"door": {"left_door": "CLOSED",
                                      "left_door_confidence": 0.91,
                                      "right_door": "NO_DATA",
                                      "right_door_confidence": 0.0,
                                      "supporting_cameras": ["LEFT_UP"],
                                      "frames_left": 7, "frames_right": 0}}),
            ):
                segs = [LocalSegment(local_id=local_segment_id(cam, 1),
                                     index=1, start_frame=0, end_frame=59,
                                     start_time=0.0, end_time=4.0,
                                     label="WAGON", confidence=0.9)]
                b, segs = _seal(root, cam, segments=segs, features=feats)
                maps = map_segments_to_global(segs, roster, camera_id=cam)
                _relabel_feature_evidence(b, maps, states)
            state = GlobalTrainState(
                total_wagons=1,
                wagons=(GlobalWagon(global_id="GW_1", wagon_index=1,
                                    start_frame_master=0, end_frame_master=59,
                                    start_time=0.0, end_time=4.0,
                                    classification="WAGON",
                                    classification_confidence=1.0),),
                master_camera=MASTER)
            unified = wagon_state_builder.build(
                state=state, wagon_states_root=states,
                write_per_wagon_json=False, verbose=False)
            u = unified["GW_1"]
            self.assertEqual(u.right_door, "OPEN")
            self.assertEqual(u.left_door, "CLOSED")
            self.assertIn("RIGHT_DOOR_OPEN", u.anomalies)


class TestArrivalHarnessProcessIsolation(unittest.TestCase):
    """Four OS processes, not one in-process loop.

    A single-process loop could pass every filesystem assertion while quietly
    sharing tracker state in memory -- which is exactly the bug the sequential
    architecture exists to rule out. So the harness is checked structurally.
    """

    def _harness(self):
        import importlib
        import sys as _sys
        bench = os.path.join(V4_ROOT, "benchmarks")
        if bench not in _sys.path:
            _sys.path.insert(0, bench)
        return importlib.import_module("run_sequential_arrivals")

    def test_arrival_order_is_the_four_cameras_master_first(self):
        h = self._harness()
        self.assertEqual(h.ARRIVAL_ORDER[0], MASTER)
        self.assertEqual(set(h.ARRIVAL_ORDER), set(CAMS))
        self.assertEqual(len(h.ARRIVAL_ORDER), 4)

    def test_each_arrival_is_a_separate_interpreter(self):
        h = self._harness()
        src = inspect.getsource(h.run_one_arrival)
        self.assertIn("subprocess.run", src)
        self.assertIn("sys.executable", src)
        self.assertIn("--camera", src)

    def test_harness_never_calls_run_sequential_in_process(self):
        import ast
        h = self._harness()
        tree = ast.parse(inspect.getsource(h))
        # The module docstring explains that it deliberately does NOT call
        # run_sequential(), so scanning raw source would match its own prose.
        called = {ast.unparse(n.func) for n in ast.walk(tree)
                  if isinstance(n, ast.Call)}
        self.assertNotIn("run_sequential", called)
        self.assertNotIn("process_batch", called)
        for name in called:
            self.assertNotIn("process_batch", name)

    def test_only_the_worker_branch_runs_a_camera(self):
        """run_camera() is reachable only under --camera, i.e. in the child."""
        h = self._harness()
        src = inspect.getsource(h.main)
        self.assertLess(src.index("if args.camera:"),
                        src.index("camera_runner.run_camera"),
                        "the parent must not call run_camera directly")
        self.assertEqual(src.count("camera_runner.run_camera"), 1)

    def test_parent_loads_no_video_frames(self):
        """The orchestrator only resolves paths; decoding happens in children."""
        h = self._harness()
        src = inspect.getsource(h)
        for banned in ("VideoCapture", "cv2.imread", "read_frames",
                       "materializ"):
            with self.subTest(token=banned):
                self.assertNotIn(banned, src)

    def test_assembly_happens_after_the_arrival_loop(self):
        h = self._harness()
        src = inspect.getsource(h.main)
        self.assertLess(src.index("run_one_arrival("),
                        src.index("global_assembler.assemble("),
                        "assembly must come after the arrivals")

    def test_assembly_is_gated_on_all_four_being_sealed(self):
        h = self._harness()
        src = inspect.getsource(h.assert_after_arrival)
        self.assertIn("SEALED", src)
        self.assertIn("does NOT exist yet", src)

    def test_disk_is_reported_around_every_arrival(self):
        h = self._harness()
        src = inspect.getsource(h.main)
        self.assertIn('disk_report(f"before {cam}"', src)
        self.assertIn('disk_report(f"after  {cam}"', src)

    def test_tree_bytes_counts_a_hardlink_once(self):
        """Otherwise the global view would look like duplicated frames."""
        h = self._harness()
        with tempfile.TemporaryDirectory() as root:
            a = os.path.join(root, "a")
            os.makedirs(a)
            src = os.path.join(a, "f.bin")
            with open(src, "wb") as f:
                f.write(b"x" * 4096)
            one = h.tree_bytes(root)
            try:
                os.link(src, os.path.join(a, "f_link.bin"))
            except (OSError, AttributeError):
                self.skipTest("no hardlink support")
            if os.stat(src).st_ino == 0:
                self.skipTest("inode numbers unavailable")
            self.assertEqual(h.tree_bytes(root), one,
                             "hardlink double-counted as new disk")


class TestMediaAssociation(unittest.TestCase):
    """Image evidence must gain its GLOBAL name, or the combined report is blank.

    The JSON relabelling alone is not enough: the existing global report
    resolves `<evidence_root>/<GW_n>/<feature>/<slot>.jpg` and
    `<cache_root>/<GW_n>/<camera_folder>/*.jpg`.
    """

    class _GW:
        def __init__(self, gid, s, e):
            self.global_id, self.start_time, self.end_time = gid, s, e

    def _fixture(self, root, cam=MASTER):
        import numpy as np
        import cv2
        b, segs = _seal(root, cam)
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        for sg in segs:
            ed = os.path.join(b.dir, "evidence", sg.local_id, "door")
            os.makedirs(ed, exist_ok=True)
            cv2.imwrite(os.path.join(ed, "right_best.jpg"), img)
            cd = os.path.join(b.dir, "camera_cache", sg.local_id,
                              C.CAMERA_FOLDER[cam])
            os.makedirs(cd, exist_ok=True)
            cv2.imwrite(os.path.join(cd, "frame_000001.jpg"), img)
        roster = [self._GW(f"GW_{i}", float((i - 1) * 4), float(i * 4))
                  for i in range(1, 4)]
        maps = map_segments_to_global(segs, roster, camera_id=cam)
        return b, segs, maps

    def test_existing_report_lookups_resolve_under_global_ids(self):
        from orchestrator.global_assembler import _associate_media
        from reporting import _evidence_lookup as ev
        with tempfile.TemporaryDirectory() as root:
            b, _segs, maps = self._fixture(root)
            ed = os.path.join(root, "out", "evidence")
            cd = os.path.join(root, "out", "camera_cache")
            n = _associate_media(b, maps, evidence_dst=ed, cache_dst=cd)
            self.assertEqual(n["evidence"], 3)
            self.assertEqual(n["cache"], 3)
            for i in range(1, 4):
                self.assertTrue(
                    ev.evidence_snapshot(ed, f"GW_{i}", "door", "right_best"),
                    f"GW_{i} door evidence unresolved")
                self.assertTrue(os.path.isdir(os.path.join(
                    cd, f"GW_{i}", C.CAMERA_FOLDER[MASTER])))

    def test_camera_local_evidence_is_not_moved_away(self):
        """The camera bundles stay intact -- assembly only adds a global view."""
        from orchestrator.global_assembler import _associate_media
        with tempfile.TemporaryDirectory() as root:
            b, segs, maps = self._fixture(root)
            _associate_media(b, maps,
                             evidence_dst=os.path.join(root, "o", "evidence"),
                             cache_dst=os.path.join(root, "o", "cache"))
            for sg in segs:
                self.assertTrue(os.path.isfile(os.path.join(
                    b.dir, "evidence", sg.local_id, "door", "right_best.jpg")))

    def test_no_extra_disk_for_the_global_view(self):
        """Hardlinks, not copies -- the previous run filled the root volume."""
        from orchestrator.global_assembler import _associate_media
        with tempfile.TemporaryDirectory() as root:
            b, segs, maps = self._fixture(root)
            ed = os.path.join(root, "out", "evidence")
            _associate_media(b, maps, evidence_dst=ed,
                             cache_dst=os.path.join(root, "out", "cache"))
            src = os.path.join(b.dir, "evidence", segs[0].local_id, "door",
                              "right_best.jpg")
            dst = os.path.join(ed, "GW_1", "door", "right_best.jpg")
            if not hasattr(os, "link"):
                self.skipTest("no hardlink support")
            a, c = os.stat(src), os.stat(dst)
            if a.st_ino == 0:            # some Windows filesystems report 0
                self.skipTest("inode numbers unavailable")
            self.assertEqual(a.st_ino, c.st_ino,
                             "evidence was copied instead of linked")

    def test_unmatched_segment_media_is_never_given_a_global_name(self):
        from orchestrator.global_assembler import _associate_media
        with tempfile.TemporaryDirectory() as root:
            b, _segs, maps = self._fixture(root)
            for m in maps:                       # simulate all UNMATCHED
                m.global_id = ""
            ed = os.path.join(root, "out", "evidence")
            n = _associate_media(b, maps, evidence_dst=ed,
                                 cache_dst=os.path.join(root, "out", "cache"))
            self.assertEqual(n, {"evidence": 0, "cache": 0})
            self.assertFalse(os.path.exists(ed))

    def test_assembler_points_the_report_at_the_populated_roots(self):
        """A regression guard: cache_root=None rendered blank evidence pages."""
        from orchestrator import global_assembler as ga
        src = inspect.getsource(ga.assemble)
        self.assertIn("evidence_root=global_evidence", src)
        self.assertIn("cache_root=global_cache", src)
        self.assertNotIn("cache_root=None", src)


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
        modified = changed_paths("wagon_count", "reconstruction", "fusion")
        if modified is None:
            self.skipTest("git unavailable")
        self.assertEqual(modified, [])

    def test_reporting_adds_no_second_renderer(self):
        """Sequential mode reuses the existing renderer -- it adds none.

        The camera-local PDF is built by reporting/camera_reports.py via
        orchestrator/camera_report_adapter.py, so nothing under reporting/
        changes at all.
        """
        modified = changed_paths("reporting")
        if modified is None:
            self.skipTest("git unavailable")
        self.assertEqual(modified, [],
                         "existing reporting builders must be unmodified")
        self.assertFalse(os.path.exists(
            os.path.join(V4_ROOT, "reporting", "camera_local_report.py")),
            "the from-scratch camera renderer must stay deleted")


if __name__ == "__main__":
    unittest.main()
