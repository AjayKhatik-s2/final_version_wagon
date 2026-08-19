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
    """Sequential is the DEFAULT architecture for foreground runs.

    Batch is not removed -- `--mode batch` still reaches the unchanged
    process_batch() path.
    """

    def test_sequential_is_the_default(self):
        self.assertEqual(_parse([]).mode, "sequential")
        self.assertEqual(_parse(["--local-only"]).mode, "sequential")

    def test_explicit_sequential_parses(self):
        self.assertEqual(_parse(["--mode", "sequential"]).mode, "sequential")

    def test_explicit_batch_still_parses(self):
        self.assertEqual(_parse(["--mode", "batch"]).mode, "batch")

    def test_only_two_modes_accepted(self):
        for bad in ("turbo", "fast", "legacy"):
            with self.subTest(mode=bad), self.assertRaises(SystemExit):
                _parse(["--mode", bad])

    def test_help_documents_sequential_as_the_default(self):
        from orchestrator.master_runner import _build_parser
        action = next(a for a in _build_parser()._actions
                      if "--mode" in (a.option_strings or []))
        self.assertEqual(action.default, "sequential")
        self.assertEqual(tuple(action.choices), ("batch", "sequential"))
        self.assertIn("DEFAULT", action.help)

    def test_sequential_dispatches_before_process_batch(self):
        from orchestrator import master_runner
        src = inspect.getsource(master_runner.main)
        # anchor on the CALL -- the guard's comment mentions run_local() above
        self.assertLess(src.index('args.mode == "sequential"'),
                        src.index("return run_local("),
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


class TestMainDispatchRouting(unittest.TestCase):
    """Which entry point main() actually calls, per invocation.

    Parsing the right value is not the same as dispatching to the right
    function: the sequential branch returns unconditionally, so before the
    live guard existed, defaulting to sequential would have swallowed --auto
    and stopped production polling. These route the real main() with the three
    entry points mocked.
    """

    ENTRIES = ("run_sequential", "run_local", "run_auto")

    def _route(self, argv):
        from unittest import mock
        from orchestrator import master_runner as mr
        with mock.patch.object(mr, "run_sequential", return_value=0) as seq, \
             mock.patch.object(mr, "run_local", return_value=0) as loc, \
             mock.patch.object(mr, "run_auto", return_value=0) as auto, \
             mock.patch.object(mr, "resolve_feature_config",
                               return_value=mr.FeatureConfig.all_on()):
            rc = mr.main(list(argv))
        called = [n for n, m in zip(self.ENTRIES, (seq, loc, auto)) if m.called]
        self.assertLessEqual(len(called), 1, f"{argv} called {called}")
        return (called[0] if called else None), rc

    def test_no_mode_runs_sequential(self):
        self.assertEqual(self._route([])[0], "run_sequential")

    def test_local_only_runs_sequential(self):
        self.assertEqual(self._route(["--local-only"])[0], "run_sequential")

    def test_explicit_sequential_runs_sequential(self):
        self.assertEqual(self._route(["--mode", "sequential"])[0],
                         "run_sequential")

    def test_explicit_batch_local_only_runs_the_batch_path(self):
        self.assertEqual(self._route(["--mode", "batch", "--local-only"])[0],
                         "run_local")

    def test_explicit_batch_without_a_target_still_errors(self):
        """Unchanged: bare --mode batch has never been a runnable invocation."""
        who, rc = self._route(["--mode", "batch"])
        self.assertIsNone(who)
        self.assertEqual(rc, 2)

    def test_local_only_batch_key_names_the_output_not_an_s3_batch(self):
        self.assertEqual(self._route(["--local-only", "--batch", "k"])[0],
                         "run_sequential")


class TestLiveDispatchIsProtected(unittest.TestCase):
    """--auto/--once/--batch must reach run_auto() whatever --mode says.

    This is the regression that flipping the default could have caused: the
    polling daemon silently replaced by a one-shot local run.
    """

    def _route(self, argv):
        return TestMainDispatchRouting._route(self, argv)

    ENTRIES = TestMainDispatchRouting.ENTRIES

    def test_auto_still_reaches_run_auto(self):
        self.assertEqual(self._route(["--auto"])[0], "run_auto")

    def test_once_still_reaches_run_auto(self):
        self.assertEqual(self._route(["--once"])[0], "run_auto")

    def test_discovered_batch_still_reaches_run_auto(self):
        self.assertEqual(self._route(["--batch", "20260408_032134"])[0],
                         "run_auto")

    def test_mode_cannot_divert_the_live_daemon(self):
        for argv in (["--mode", "sequential", "--auto"],
                     ["--mode", "batch", "--auto"],
                     ["--mode", "sequential", "--once"]):
            with self.subTest(argv=argv):
                self.assertEqual(self._route(argv)[0], "run_auto")

    def test_the_guard_is_documented_at_the_branch(self):
        from orchestrator import master_runner
        src = inspect.getsource(master_runner.main)
        self.assertIn("live_dispatch", src)
        # Anchor on the guarded DISPATCH, not the first mention of the mode --
        # the historical branch also tests args.mode == "sequential", earlier.
        self.assertLess(src.index("live_dispatch = "),
                        src.index('args.mode == "sequential" and not '
                                  'live_dispatch'),
                        "the live guard must be computed before the branch")

    def test_run_auto_is_still_called_with_the_same_arguments(self):
        """The live contract is the CALL SITE.

        run_auto() itself takes (*args, **kwargs) and forwards to the legacy
        train_batch_manager, so its own signature asserts nothing. What must
        not drift is what main() hands it.
        """
        import ast
        from orchestrator import master_runner
        fn = next(n for n in ast.walk(ast.parse(
            inspect.getsource(master_runner.main)))
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "run_auto")
        passed = {kw.arg for kw in fn.keywords}
        for name in ("workspace", "recon_models_dir", "feat_models_dir",
                     "poll_interval", "partial_wait_minutes", "run_once",
                     "force_batch_key", "skip_upload", "skip_email",
                     "feature_config", "inference_opts"):
            self.assertIn(name, passed, f"run_auto no longer receives {name}")

    def test_polling_arguments_keep_their_sources(self):
        """--poll-interval / --partial-wait still feed the daemon unchanged."""
        import ast
        from orchestrator import master_runner
        call = next(n for n in ast.walk(ast.parse(
            inspect.getsource(master_runner.main)))
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "run_auto")
        by_name = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
        self.assertEqual(by_name["poll_interval"], "args.poll_interval")
        self.assertEqual(by_name["partial_wait_minutes"], "args.partial_wait")
        self.assertEqual(by_name["force_batch_key"], "args.batch")


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
        """The orchestrator only resolves paths; decoding happens in children.

        Scans CODE only -- prose in a comment may legitimately mention the
        materializer without the parent ever calling it.
        """
        import io as _io
        import tokenize
        h = self._harness()
        code = []
        src = inspect.getsource(h)
        for tok in tokenize.generate_tokens(_io.StringIO(src).readline):
            if tok.type not in (tokenize.COMMENT, tokenize.STRING):
                code.append(tok.string)
        code = " ".join(code)
        for banned in ("VideoCapture", "imread", "read_frames",
                       "wagon_cache_builder", "build_camera_local"):
            with self.subTest(token=banned):
                self.assertNotIn(banned, code)

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


class TestAssemblyIsTheBatchSequence(unittest.TestCase):
    """Assembly must BE the old pipeline from Stage 2 on, run late.

    `run_global_count.py` STEP 3 then `master_runner.process_batch` Stages 2-5:
    fusion(with wagon_regions) -> materializer -> features(load first) ->
    wagon_state_builder -> combined_train_report. Feature inference belongs
    HERE, because a support camera's clock offset -- and therefore its feature
    windows -- cannot be known while that camera is being processed alone.
    """

    def _src(self, fn=None):
        from orchestrator import global_assembler as ga
        return inspect.getsource(fn or ga)

    def test_wagon_regions_are_passed_to_fusion(self):
        """The confirmed omission: run_global_count.py:932 passes these."""
        from orchestrator import global_assembler as ga
        src = self._src(ga.assemble)
        self.assertIn("wagon_regions=support_regions", src)
        self.assertIn("_load_wagon_region(", src)

    def test_support_order_follows_all_cameras(self):
        """Batch builds support from ALL_CAMERAS, not dict insertion order."""
        src = self._src()
        self.assertIn("for c in all_cameras", src)

    def test_stage_order_matches_batch(self):
        from orchestrator import global_assembler as ga
        src = self._src(ga.assemble)
        seq = ["assemble_global_train_state_master_fixed",
               "wagon_cache_builder.build",
               "_FEATURE_ORDER",
               "wagon_state_builder.build",
               "combined_train_report"]
        pos = [src.index(tok) for tok in seq]
        self.assertEqual(pos, sorted(pos), f"stages out of order: {seq}")

    def test_load_runs_before_damage(self):
        """damage reads the sibling load JSON to drop floor_damage on LOADED."""
        from orchestrator.global_assembler import _FEATURE_ORDER
        names = [n for n, _e in _FEATURE_ORDER]
        self.assertEqual(names[0], "load")
        self.assertLess(names.index("load"), names.index("damage"))

    def test_strides_match_the_approved_values(self):
        from orchestrator.global_assembler import _FEATURE_ORDER
        got = {n: e["sample_stride"] for n, e in _FEATURE_ORDER}
        self.assertEqual(got, {"door": 3, "damage": 3, "load": 2})
        for _n, e in _FEATURE_ORDER:
            self.assertEqual(e["inference_mode"], "sampled")

    def test_no_ocr_in_assembly(self):
        from orchestrator.global_assembler import _FEATURE_ORDER
        self.assertNotIn("ocr", [n for n, _e in _FEATURE_ORDER])

    def test_features_run_over_the_global_state(self):
        """Not over camera-local segments -- that was the divergence."""
        from orchestrator import global_assembler as ga
        src = self._src(ga.assemble)
        self.assertIn("feature_kwargs = dict(state=state", src)
        self.assertIn("cache_root=global_cache", src)

    def test_no_stage1_runs_during_assembly(self):
        """Tracking, stitching, validation and classification are READ BACK.

        Feature detectors are expected here now; Stage-1 ones never are.
        """
        src = self._src()
        for banned in ("GapTracker", "reassemble_fragments",
                       "validate_gap_events", "renumber_gap_events",
                       "recover_wagon_active_candidates",
                       "apply_temporal_classification", "classify_segments",
                       "MasterClassifier", "segments_from_gaps",
                       "process_video", "ultralytics"):
            with self.subTest(token=banned):
                self.assertNotIn(banned, src)

    def test_the_overlap_mapper_is_diagnostic_only(self):
        """It must not decide where any feature frame goes."""
        from orchestrator import global_assembler as ga
        src = self._src(ga.assemble)
        i = src.index("map_segments_to_global(")
        j = src.index("wagon_cache_builder.build(")
        self.assertLess(i, j)
        # the block is labelled as diagnostic in the comment above the call
        self.assertIn("diagnostic", src[max(0, i - 700):j].lower())
        # nothing derived from the mapping may reach the feature stage
        after = src[j:]
        for tok in ("maps", "mapping_by_camera["):
            self.assertNotIn(f"{tok})", after)

    def test_relabelling_machinery_is_gone(self):
        from orchestrator import global_assembler as ga
        for name in ("_relabel_feature_evidence", "_associate_media",
                     "_merge_payloads", "_link_or_copy"):
            with self.subTest(symbol=name):
                self.assertFalse(hasattr(ga, name),
                                 f"{name} should have been removed")

    def test_materializer_is_called_with_resolved_offsets(self):
        from orchestrator import global_assembler as ga
        src = self._src(ga.assemble)
        self.assertIn("camera_offsets=resolved", src)
        self.assertIn("per_camera_fps=per_camera_fps", src)
        self.assertIn("video_paths=video_paths", src)


class TestCameraLocalFeaturesDoNotReachTheGlobalResult(unittest.TestCase):
    """Camera-time inference exists for the camera PDFs and nothing else.

    It runs by default -- otherwise the camera-local reports have no snapshots
    to embed -- but global assembly recomputes every feature over the global
    wagons and never reads what a camera wrote locally.
    """

    def test_it_runs_by_default(self):
        from orchestrator.camera_runner import run_camera
        p = inspect.signature(run_camera).parameters
        self.assertIn("camera_local_features", p)
        self.assertIs(p["camera_local_features"].default, True)

    def test_it_can_be_switched_off(self):
        from orchestrator import camera_runner
        src = inspect.getsource(camera_runner.run_camera)
        self.assertIn("if camera_local_features else []", src)

    def test_assembly_never_reads_the_camera_local_feature_tree(self):
        """The bundle's own features/ dir must not feed the global answer.

        `models/features` is the weights directory and is unrelated.
        """
        from orchestrator import global_assembler as ga
        src = inspect.getsource(ga.assemble)
        self.assertNotIn('bundle.dir, "features"', src)
        self.assertNotIn('b.dir, "features"', src)
        self.assertIn("output_dir=states_root", src)

    def test_assembly_reads_only_stage1_artefacts_from_the_bundles(self):
        from orchestrator import global_assembler as ga
        src = inspect.getsource(ga)
        read = {"tracking_full.json", "classification.json",
                "wagon_region.json", "segments.json"}
        for name in read:
            if name == "segments.json":
                continue            # read via bundle.read_segments()
            self.assertIn(name, src)
        self.assertNotIn('bundle.dir, "camera_cache"', src,
                         "assembly must materialize its own cache, not reuse "
                         "the camera-local one")
        self.assertIn('batch_root, "wagon_cache"', src,
                      "the global cache should use the batch pipeline's name")

    def test_assembly_writes_features_to_the_batch_tree(self):
        from orchestrator import global_assembler as ga
        src = inspect.getsource(ga.assemble)
        self.assertIn("feature_kwargs = dict(state=state", src)
        self.assertIn("evidence_root=global_evidence", src)


class TestHarnessUsesRealFields(unittest.TestCase):
    """The harness crashed on asm.media_linked after that field was removed."""

    def _harness_src(self):
        import importlib
        import sys as _sys
        bench = os.path.join(V4_ROOT, "benchmarks")
        if bench not in _sys.path:
            _sys.path.insert(0, bench)
        return inspect.getsource(importlib.import_module(
            "run_sequential_arrivals"))

    def test_harness_does_not_touch_media_linked(self):
        self.assertNotIn("media_linked", self._harness_src())

    def test_media_linked_was_not_reintroduced(self):
        import dataclasses
        from orchestrator.global_assembler import AssemblyResult
        names = {f.name for f in dataclasses.fields(AssemblyResult)}
        self.assertNotIn("media_linked", names)
        self.assertNotIn("relabelled", names)

    def test_every_asm_attribute_the_harness_reads_exists(self):
        """Catches the next stale field before a 40-minute run does."""
        import ast
        import dataclasses
        from orchestrator.global_assembler import AssemblyResult
        names = {f.name for f in dataclasses.fields(AssemblyResult)}
        used = set()
        for n in ast.walk(ast.parse(self._harness_src())):
            if (isinstance(n, ast.Attribute)
                    and isinstance(n.value, ast.Name) and n.value.id == "asm"):
                used.add(n.attr)
        self.assertTrue(used, "found no asm.* reads -- did the parse work?")
        self.assertEqual(sorted(used - names), [],
                         f"harness reads fields that do not exist: "
                         f"{sorted(used - names)}")


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
