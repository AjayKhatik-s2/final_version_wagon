"""A camera's report must reach the dashboard when THAT camera finishes.

The point of sequential mode is that an operator sees RIGHT_UP's result about a
quarter of the way into a train rather than 30-40 minutes later, and these tests
pin that: per-camera reporting happens per camera, the combined report still
waits for the whole train, and an older train's combined report is never held
behind a newer train's inference.

`run_camera` and `assemble` are stubbed here -- they need models and video, and
what is under test is the ORDER of calls, not what they compute.
"""

from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from unittest import mock

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from orchestrator.camera_runner import CameraRunResult

RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP
RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP
ALL = (RU, LU, RUT, LUT)


class _Asm:
    def __init__(self, wagons=7):
        self.ready = True
        self.total_wagons = wagons
        self.report_pdf_path = "/tmp/combined.pdf"
        self.mapping_by_camera = {}
        self.delivery = None


def _videos(train):
    return {cam: f"/v/{train}_{cam}.mp4" for cam in ALL}


class _Harness:
    """Records every camera run and every assembly, in order."""

    def __init__(self, fail: tuple = ()):
        self.calls = []
        self.kwargs = []
        self.fail = set(fail)

    def run_camera(self, **kw):
        self.kwargs.append(kw)
        cam, train = kw["camera_id"], kw.get("train_id", "")
        self.calls.append(("camera", train, cam))
        failed = (train, cam) in self.fail or cam in self.fail
        return CameraRunResult(
            camera_id=cam, state="FAILED" if failed else "SEALED",
            sealed=not failed,
            failure_reason="stub failure" if failed else "",
            local_segments=0 if failed else 5)

    def assemble(self, **kw):
        self.calls.append(("assemble", kw["batch_key"], ""))
        return _Asm()

    @property
    def order(self):
        return [(k, t, c) for k, t, c in self.calls]


def _run(sessions, harness, **kw):
    from orchestrator import master_runner as mr
    from orchestrator import camera_runner, global_assembler
    with tempfile.TemporaryDirectory() as ws:
        with mock.patch.object(camera_runner, "run_camera",
                               side_effect=harness.run_camera), \
             mock.patch.object(global_assembler, "assemble",
                               side_effect=harness.assemble):
            rc = mr.run_sessions(sessions=sessions, workspace=ws,
                                 verbose=False, **kw)
    return rc


# ---------------------------------------------------------------------------
# Per-camera reporting is per camera
# ---------------------------------------------------------------------------

class TestIncrementalCameraReports(unittest.TestCase):

    def test_run_camera_renders_and_publishes_inside_the_camera_run(self):
        """No orchestration layer is needed for this: `run_camera` owns it."""
        from orchestrator import camera_runner
        src = inspect.getsource(camera_runner.run_camera)
        self.assertIn("build_local_camera_pdf", src)
        self.assertIn("camera_inspection.publish", src)

    def test_the_camera_report_is_rendered_before_the_camera_seals(self):
        from orchestrator import camera_runner
        src = inspect.getsource(camera_runner.run_camera)
        self.assertLess(src.index("build_local_camera_pdf"),
                        src.index('bundle.advance("SEALED")'))

    def test_the_dashboard_publish_happens_after_the_seal(self):
        """A receiver outage must never be able to un-seal a camera."""
        from orchestrator import camera_runner
        src = inspect.getsource(camera_runner.run_camera)
        self.assertLess(src.index('bundle.advance("SEALED")'),
                        src.index("camera_inspection.publish"))

    def test_publish_is_reached_once_per_camera_not_once_per_train(self):
        from orchestrator import camera_runner
        src = inspect.getsource(camera_runner.run_camera)
        self.assertEqual(src.count("camera_inspection.publish"), 1)

    def test_scheduled_run_asks_for_per_camera_delivery_by_default(self):
        h = _Harness()
        _run([("T1", _videos("T1"))], h)
        for kw in h.kwargs:
            self.assertTrue(kw["deliver_per_camera"],
                            "per-camera dashboard delivery must be on in the "
                            "scheduled sequential path")

    def test_every_camera_is_run_exactly_once(self):
        h = _Harness()
        _run([("T1", _videos("T1"))], h)
        cams = [c for k, _t, c in h.order if k == "camera"]
        self.assertEqual(sorted(cams), sorted(ALL))

    def test_each_camera_run_carries_its_train_id(self):
        h = _Harness()
        _run([("T1", _videos("T1"))], h)
        for kw in h.kwargs:
            self.assertEqual(kw["train_id"], "T1")

    def test_no_second_renderer_is_introduced(self):
        """`run_sessions` must not build a report itself."""
        from orchestrator import master_runner as mr
        src = inspect.getsource(mr.run_sessions)
        for banned in ("reportlab", "SimpleDocTemplate", "build_camera_report",
                       "combined_train_report", "build_local_camera_pdf"):
            with self.subTest(token=banned):
                self.assertNotIn(banned, src)


# ---------------------------------------------------------------------------
# The combined report still waits for the train
# ---------------------------------------------------------------------------

class TestCombinedReportTiming(unittest.TestCase):

    def test_assembly_runs_only_after_the_last_camera(self):
        h = _Harness()
        _run([("T1", _videos("T1"))], h)
        kinds = [k for k, _t, _c in h.order]
        self.assertEqual(kinds, ["camera"] * 4 + ["assemble"])

    def test_assembly_runs_exactly_once_per_train(self):
        h = _Harness()
        _run([("T1", _videos("T1")), ("T2", _videos("T2"))], h)
        self.assertEqual(sum(1 for k, _t, _c in h.order if k == "assemble"), 2)

    def test_a_partial_train_is_never_assembled(self):
        h = _Harness()
        _run([("T1", {RU: "/v/a.mp4", LU: "/v/b.mp4"})], h)
        self.assertEqual([k for k, _t, _c in h.order], ["camera", "camera"])

    def test_a_failed_camera_still_lets_the_train_assemble(self):
        """Existing semantics: assembly itself decides if a failure is fatal."""
        h = _Harness(fail=(LUT,))
        _run([("T1", _videos("T1"))], h)
        self.assertIn("assemble", [k for k, _t, _c in h.order])

    def test_report_generation_failure_does_not_fail_the_camera(self):
        """`run_camera` swallows a PDF failure -- pinned, because the scheduler
        keys its state transitions off `sealed`."""
        from orchestrator import camera_runner
        src = inspect.getsource(camera_runner.run_camera)
        block = src[src.index("build_local_camera_pdf"):
                    src.index('bundle.advance("REPORTED")')]
        self.assertIn("except Exception", block)


# ---------------------------------------------------------------------------
# Multi-train ordering, end to end through run_sessions
# ---------------------------------------------------------------------------

class TestMultiTrainOrdering(unittest.TestCase):

    def test_older_train_is_fully_processed_and_assembled_first(self):
        h = _Harness()
        _run([("T_0200", _videos("T_0200")), ("T_0100", _videos("T_0100"))], h)
        first_assemble = next(i for i, (k, _t, _c) in enumerate(h.order)
                              if k == "assemble")
        self.assertEqual(h.order[first_assemble][1], "T_0100")
        self.assertTrue(all(t == "T_0100" for k, t, _c in
                            h.order[:first_assemble] if k == "camera"))

    def test_an_older_trains_combined_report_is_not_held_behind_a_newer_train(self):
        """T_0100 assembles before T_0200's first camera even starts."""
        h = _Harness()
        _run([("T_0100", _videos("T_0100")), ("T_0200", _videos("T_0200"))], h)
        idx_assemble_1 = next(i for i, (k, t, _c) in enumerate(h.order)
                              if k == "assemble" and t == "T_0100")
        idx_first_t2 = next(i for i, (k, t, _c) in enumerate(h.order)
                            if k == "camera" and t == "T_0200")
        self.assertLess(idx_assemble_1, idx_first_t2)

    def test_a_blocked_older_train_does_not_stall_a_newer_one(self):
        h = _Harness()
        _run([("T_0100", {RU: "/v/x.mp4"}), ("T_0200", _videos("T_0200"))], h)
        kinds = [(k, t) for k, t, _c in h.order]
        self.assertIn(("assemble", "T_0200"), kinds)
        self.assertNotIn(("assemble", "T_0100"), kinds)
        self.assertEqual(sum(1 for k, t in kinds
                             if k == "camera" and t == "T_0100"), 1)

    def test_trains_are_never_merged(self):
        """Each train gets its own evidence root and its own assembly key."""
        h = _Harness()
        _run([("T_0100", _videos("T_0100")), ("T_0200", _videos("T_0200"))], h)
        roots = {kw["train_id"]: kw["evidence_root"] for kw in h.kwargs}
        self.assertEqual(len(set(roots.values())), 2,
                         "two trains shared one evidence root")
        for train, root in roots.items():
            self.assertIn(train, root)
        keys = [t for k, t, _c in h.order if k == "assemble"]
        self.assertEqual(sorted(keys), ["T_0100", "T_0200"])

    def test_a_straggler_feed_reprioritises_the_older_train(self):
        """The documented 10:00 / 10:05 / 10:06 scenario, end to end."""
        from orchestrator import master_runner as mr
        from orchestrator import camera_runner, global_assembler
        from orchestrator.train_scheduler import TrainScheduler

        h = _Harness()
        sched = TrainScheduler(verbose=False)
        # A arrives at 10:00 missing both top cameras; B at 10:05 complete.
        for cam in (RU, LU):
            sched.submit_camera_video(camera_id=cam, video_path=f"/v/a_{cam}",
                                      train_id="A_1000",
                                      train_timestamp="A_1000")
        for cam in ALL:
            sched.submit_camera_video(camera_id=cam, video_path=f"/v/b_{cam}",
                                      train_id="B_1005",
                                      train_timestamp="B_1005")

        # A's stragglers land after B's first camera has been handed out.
        original = h.run_camera
        state = {"n": 0}

        def run_camera(**kw):
            state["n"] += 1
            if state["n"] == 1:
                for cam in (RUT, LUT):
                    sched.submit_camera_video(
                        camera_id=cam, video_path=f"/v/a_{cam}",
                        train_id="A_1000", train_timestamp="A_1000")
            return original(**kw)

        with tempfile.TemporaryDirectory() as ws:
            with mock.patch.object(camera_runner, "run_camera",
                                   side_effect=run_camera), \
                 mock.patch.object(global_assembler, "assemble",
                                   side_effect=h.assemble):
                mr.run_sessions(sessions=[], workspace=ws, scheduler=sched,
                                verbose=False)

        seq = [(k, t) for k, t, _c in h.order]
        self.assertEqual(seq[0], ("camera", "B_1005"),
                         "B should have got one job while A was blocked")
        a_done = seq.index(("assemble", "A_1000"))
        b_done = seq.index(("assemble", "B_1005"))
        self.assertLess(a_done, b_done,
                        "A became processable, so it must finish first")
        # B must not have been driven further than the single job it got before
        # A unblocked.
        b_before_a = [i for i, s in enumerate(seq[:a_done]) if s == ("camera",
                                                                    "B_1005")]
        self.assertEqual(len(b_before_a), 1,
                         "B kept running after A became processable")


class TestBackwardCompatibility(unittest.TestCase):

    def test_run_sequential_still_exists_with_its_signature(self):
        from orchestrator.master_runner import run_sequential
        p = inspect.signature(run_sequential).parameters
        for name in ("local_inputs", "workspace", "recon_models_dir",
                     "feat_models_dir", "batch_key", "feature_config",
                     "arrival_order", "deliver", "deliver_per_camera",
                     "send_email", "verbose"):
            with self.subTest(param=name):
                self.assertIn(name, p)

    def test_run_sequential_orders_its_cameras_through_the_scheduler(self):
        from orchestrator.master_runner import run_sequential
        src = inspect.getsource(run_sequential)
        self.assertIn("TrainScheduler", src)
        self.assertIn("sched.next_job()", src)
        # `for cam in order:` still exists -- it SUBMITS the arrived feeds. What
        # matters is that execution is driven by the scheduler's answer, so the
        # camera that runs comes from `job`, never from the submission loop.
        exec_block = src[src.index("while True:"):]
        self.assertIn("camera_id=job.camera_id", exec_block)
        self.assertNotIn("for cam in order:", exec_block)

    def test_run_sequential_keeps_using_the_existing_stages(self):
        from orchestrator.master_runner import run_sequential
        src = inspect.getsource(run_sequential)
        self.assertIn("camera_runner", src)
        self.assertIn("global_assembler", src)
        self.assertNotIn("process_batch", src)

    def test_run_camera_defaults_keep_old_callers_working(self):
        """The two new parameters must both be optional."""
        from orchestrator.camera_runner import run_camera
        p = inspect.signature(run_camera).parameters
        self.assertEqual(p["train_id"].default, "")
        self.assertIs(p["collect_engine_frames"].default, True)

    def test_batch_mode_is_untouched(self):
        from orchestrator.master_runner import process_batch
        src = inspect.getsource(process_batch)
        for banned in ("TrainScheduler", "train_scheduler", "next_job"):
            with self.subTest(token=banned):
                self.assertNotIn(banned, src)

    def test_auto_mode_still_uses_the_existing_batch_selector(self):
        from orchestrator.master_runner import run_auto
        src = inspect.getsource(run_auto)
        self.assertIn("select_runnable_batch", src)
        self.assertNotIn("TrainScheduler", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
