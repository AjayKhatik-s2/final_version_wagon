"""The scheduler's priority rule, proven case by case.

Every test here drives `TrainScheduler` directly.  It needs no video, no model
and no S3, because the scheduler deliberately contains no I/O and no inference
-- which is itself the property that makes it safe to put in front of the
pipeline.

The rule under test:

    1. oldest session whose every feed has arrived -> run it to completion
    2. otherwise ONE job from the newest session that has arrived work
    3. re-evaluate after every completion
"""

from __future__ import annotations

import os
import threading
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from orchestrator.train_scheduler import (
    JOB_COMPLETED, JOB_FAILED, JOB_PROCESSING, JOB_RECEIVED, JOB_WAITING,
    SESSION_COMPLETED, SESSION_PROCESSING, SESSION_WAITING,
    CameraJob, SchedulerError, TrainScheduler, TrainSession,
)

RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP
RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP
ALL = (RU, LU, RUT, LUT)


def _sched(**kw) -> TrainScheduler:
    return TrainScheduler(verbose=False, **kw)


def _submit(s: TrainScheduler, train: str, *cams: str) -> None:
    for cam in cams:
        s.submit_camera_video(camera_id=cam, video_path=f"/v/{train}_{cam}.mp4",
                              train_id=train, train_timestamp=train)


def _drain(s: TrainScheduler, limit: int = 100):
    """Run every job the scheduler offers, completing each.  Returns the order."""
    order = []
    for _ in range(limit):
        job = s.next_job()
        if job is None:
            break
        order.append((job.train_id, job.camera_id))
        s.mark_camera_completed(job.train_id, job.camera_id)
    return order


# ---------------------------------------------------------------------------
# Baseline: one train
# ---------------------------------------------------------------------------

class TestSingleTrain(unittest.TestCase):

    def test_all_cameras_arriving_normally(self):
        s = _sched()
        _submit(s, "T1", *ALL)
        self.assertEqual(_drain(s), [("T1", c) for c in ALL])
        self.assertTrue(s.session("T1").is_complete())
        self.assertEqual(s.session("T1").state(), SESSION_COMPLETED)

    def test_camera_order_is_the_configured_order(self):
        """Feeds submitted backwards still run in `expected_cameras` order."""
        s = _sched()
        _submit(s, "T1", *reversed(ALL))
        self.assertEqual([c for _, c in _drain(s)], list(ALL))

    def test_session_is_not_complete_when_one_camera_finishes(self):
        s = _sched()
        _submit(s, "T1", *ALL)
        job = s.next_job()
        s.mark_camera_completed(job.train_id, job.camera_id)
        self.assertFalse(s.session("T1").is_complete())

    def test_partial_train_is_never_reported_complete(self):
        """A missing feed keeps the session WAITING no matter what finishes."""
        s = _sched()
        _submit(s, "T1", RU, LU, RUT)
        _drain(s)
        sess = s.session("T1")
        self.assertFalse(sess.is_complete())
        self.assertEqual(sess.state(), SESSION_WAITING)
        self.assertEqual(sess.missing_cameras(), [LUT])
        self.assertEqual(sess.jobs[LUT].state, JOB_WAITING)

    def test_idle_when_nothing_submitted(self):
        s = _sched()
        self.assertIsNone(s.next_job())
        self.assertEqual(s.decisions[-1].rule, "idle")


# ---------------------------------------------------------------------------
# The core rule: chronological priority without idling
# ---------------------------------------------------------------------------

class TestChronologicalPriority(unittest.TestCase):

    def test_older_complete_train_beats_newer_complete_train(self):
        s = _sched()
        _submit(s, "T2_newer", *ALL)
        _submit(s, "T1_older", *ALL)
        order = _drain(s)
        self.assertEqual([t for t, _ in order[:4]], ["T1_older"] * 4)
        self.assertEqual([t for t, _ in order[4:]], ["T2_newer"] * 4)

    def test_blocked_older_train_yields_work_to_newer(self):
        """The documented scenario: A waiting on RUT/LUT, B complete.

        A blocked train must not stall B, and A must be recorded as the reason
        B was reachable at all.
        """
        s = _sched()
        _submit(s, "A_1000", RU, LU)          # missing RUT + LUT
        _submit(s, "B_1005", *ALL)

        job = s.next_job()
        self.assertEqual((job.train_id, job.camera_id), ("B_1005", RU))
        d = s.decisions[-1]
        self.assertIn("A_1000", d.blocked_sessions)
        self.assertEqual(d.older_sessions_checked[0], "A_1000",
                         "the older session must be inspected first")

    def test_one_job_from_newest_when_nothing_is_processable(self):
        """Rule 2 proper: no session has all its feeds, so don't idle."""
        s = _sched()
        _submit(s, "A_1000", RU)              # incomplete
        _submit(s, "B_1005", RU, LU)          # also incomplete, but newer
        job = s.next_job()
        self.assertEqual(job.train_id, "B_1005")
        self.assertEqual(s.decisions[-1].rule, "newest-partial")
        self.assertEqual(sorted(s.decisions[-1].blocked_sessions),
                         ["A_1000", "B_1005"])

    def test_oldest_processable_wins_over_a_newer_processable_one(self):
        """Where the brief's 'newest' wording and its priority rule disagree.

        A is blocked; B and C are BOTH processable. Draining C first would
        invert chronological order between two ready trains, so the priority
        requirement decides: B before C. See the scheduler docstring.
        """
        s = _sched()
        _submit(s, "A_1000", RU)              # permanently blocked
        _submit(s, "B_1005", *ALL)
        _submit(s, "C_1010", *ALL)
        order = _drain(s)
        b_last = max(i for i, (t, _) in enumerate(order) if t == "B_1005")
        c_first = min(i for i, (t, _) in enumerate(order) if t == "C_1010")
        self.assertLess(b_last, c_first)

    def test_older_train_takes_over_the_moment_its_feed_arrives(self):
        s = _sched()
        _submit(s, "A_1000", RU, LU)
        _submit(s, "B_1005", *ALL)

        j1 = s.next_job()                      # one job from B
        self.assertEqual(j1.train_id, "B_1005")
        s.mark_camera_completed(*[j1.train_id, j1.camera_id])

        _submit(s, "A_1000", RUT, LUT)         # A's stragglers land

        j2 = s.next_job()
        self.assertEqual(j2.train_id, "A_1000",
                         "an older processable train must preempt at the "
                         "next job boundary")
        self.assertEqual(s.decisions[-1].rule, "oldest-processable")

    def test_older_train_runs_to_completion_before_newer_resumes(self):
        s = _sched()
        _submit(s, "A_1000", RU, LU)
        _submit(s, "B_1005", *ALL)
        j = s.next_job()
        s.mark_camera_completed(j.train_id, j.camera_id)
        _submit(s, "A_1000", RUT, LUT)

        rest = _drain(s)
        a_span = [i for i, (t, _) in enumerate(rest) if t == "A_1000"]
        b_after = [i for i, (t, _) in enumerate(rest) if t == "B_1005"]
        self.assertEqual(len(a_span), 4)
        self.assertTrue(all(i < min(b_after) for i in a_span),
                        "B must not interleave into A's remaining work")

    def test_newer_train_never_fully_processes_while_older_is_processable(self):
        s = _sched()
        _submit(s, "A_1000", *ALL)
        _submit(s, "B_1005", *ALL)
        seen_b = 0
        for _ in range(3):
            job = s.next_job()
            if job.train_id == "B_1005":
                seen_b += 1
            s.mark_camera_completed(job.train_id, job.camera_id)
        self.assertEqual(seen_b, 0)

    def test_blocked_older_train_is_re_examined_before_every_single_job(self):
        """The guarantee that makes 'exactly one job, then return' unnecessary.

        A is never allowed to wait for a whole train to finish before being
        looked at again: it is the first session inspected on EVERY decision,
        so the moment it unblocks it wins.
        """
        s = _sched()
        _submit(s, "A_1000", RU)               # permanently short 3 feeds
        _submit(s, "B_1005", *ALL)
        for _ in range(4):
            job = s.next_job()
            d = s.decisions[-1]
            self.assertEqual(d.older_sessions_checked[0], "A_1000")
            self.assertIn("A_1000", d.blocked_sessions)
            s.mark_camera_completed(job.train_id, job.camera_id)

    def test_no_deadlock_while_waiting_on_an_older_feed(self):
        """A blocked older train must not stall available newer work."""
        s = _sched()
        _submit(s, "A_1000", RU, LU, RUT)      # LUT never arrives
        _submit(s, "B_1005", *ALL)
        order = _drain(s)
        self.assertEqual(len(order), 7, "all arrived feeds must still run")
        self.assertIsNone(s.next_job())

    def test_oldest_wins_when_two_older_trains_become_processable_together(self):
        s = _sched()
        _submit(s, "A_1000", RU, LU)
        _submit(s, "B_1002", RU, LU)
        _submit(s, "C_1005", *ALL)
        j = s.next_job()                       # rule 2 -> C
        s.mark_camera_completed(j.train_id, j.camera_id)

        _submit(s, "A_1000", RUT, LUT)         # both complete at once
        _submit(s, "B_1002", RUT, LUT)

        nxt = s.next_job()
        self.assertEqual(nxt.train_id, "A_1000", "oldest of the two must win")
        s.mark_camera_completed(nxt.train_id, nxt.camera_id)
        rest = _drain(s)
        first_b = next(i for i, (t, _) in enumerate(rest) if t == "B_1002")
        last_a = max(i for i, (t, _) in enumerate(rest) if t == "A_1000")
        self.assertLess(last_a, first_b)

    def test_three_trains_arriving_out_of_order(self):
        """Submission order is irrelevant; timestamp order decides."""
        s = _sched()
        _submit(s, "T_0300", *ALL)
        _submit(s, "T_0100", *ALL)
        _submit(s, "T_0200", *ALL)
        order = _drain(s)
        self.assertEqual([t for t, _ in order[0:4]], ["T_0100"] * 4)
        self.assertEqual([t for t, _ in order[4:8]], ["T_0200"] * 4)
        self.assertEqual([t for t, _ in order[8:12]], ["T_0300"] * 4)

    def test_a_newest_train_cannot_jump_an_older_processable_one(self):
        """Train C arrives while B is part-processed and A still waits."""
        s = _sched()
        _submit(s, "A_1000", RU)               # blocked
        _submit(s, "B_1005", *ALL)
        j = s.next_job()                       # one from B
        s.mark_camera_completed(j.train_id, j.camera_id)

        _submit(s, "C_1010", *ALL)             # newest, complete
        _submit(s, "A_1000", LU, RUT, LUT)     # A now processable

        nxt = s.next_job()
        self.assertEqual(nxt.train_id, "A_1000",
                         "C must not jump ahead of a newly-processable A")


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------

class TestNoStarvation(unittest.TestCase):

    def test_permanently_incomplete_old_train_does_not_starve_the_rest(self):
        s = _sched()
        _submit(s, "A_1000", RU)               # never completes
        _submit(s, "B_1005", *ALL)
        _submit(s, "C_1010", *ALL)
        order = _drain(s)
        by_train = {}
        for t, _ in order:
            by_train[t] = by_train.get(t, 0) + 1
        self.assertEqual(by_train.get("B_1005"), 4,
                         "the middle train must still be fully served")
        self.assertEqual(by_train.get("C_1010"), 4)
        self.assertEqual(by_train.get("A_1000"), 1)

    def test_rule_two_is_bounded_by_feeds_per_session(self):
        """Why 'newest' cannot loop forever: eligibility is exhaustible."""
        s = _sched()
        _submit(s, "A_1000", RU)
        _submit(s, "B_1005", *ALL)
        picks = []
        for _ in range(4):
            job = s.next_job()
            picks.append(job.train_id)
            s.mark_camera_completed(job.train_id, job.camera_id)
        self.assertEqual(picks, ["B_1005"] * 4)
        # B is exhausted -> eligibility must pass to A's one arrived feed.
        job = s.next_job()
        self.assertEqual(job.train_id, "A_1000")


# ---------------------------------------------------------------------------
# Duplicates, failures, retries
# ---------------------------------------------------------------------------

class TestDuplicatesAndFailures(unittest.TestCase):

    def test_duplicate_arrival_is_ignored(self):
        s = _sched()
        _submit(s, "T1", RU)
        _submit(s, "T1", RU)                   # same camera again
        _submit(s, "T1", RU)
        self.assertEqual(len(_drain(s)), 1, "a camera must run exactly once")

    def test_duplicate_arrival_does_not_reset_a_finished_job(self):
        s = _sched()
        _submit(s, "T1", RU)
        job = s.next_job()
        s.mark_camera_completed(job.train_id, job.camera_id)
        _submit(s, "T1", RU)                   # late duplicate
        self.assertIsNone(s.next_job())
        self.assertEqual(s.session("T1").jobs[RU].state, JOB_COMPLETED)

    def test_a_job_cannot_be_handed_out_twice(self):
        s = _sched()
        _submit(s, "T1", RU, LU, RUT, LUT)
        first = s.next_job()
        second = s.next_job()
        self.assertNotEqual(first.camera_id, second.camera_id)
        self.assertEqual(first.state, JOB_PROCESSING)

    def test_processing_jobs_hold_priority_rather_than_leaking_to_a_newer_train(self):
        s = _sched()
        _submit(s, "A_1000", RU)
        _submit(s, "B_1005", *ALL)
        # Give A its last feed so it is processable, then take its only job out.
        _submit(s, "A_1000", LU, RUT, LUT)
        job = s.next_job()
        self.assertEqual(job.train_id, "A_1000")
        # Three of A's jobs remain RECEIVED, so work is still offered from A.
        self.assertEqual(s.next_job().train_id, "A_1000")

    def test_failure_is_terminal_by_default(self):
        s = _sched()
        _submit(s, "T1", RU, LU, RUT, LUT)
        job = s.next_job()
        s.mark_camera_failed(job.train_id, job.camera_id, "boom")
        self.assertEqual(s.session("T1").jobs[job.camera_id].state, JOB_FAILED)
        self.assertEqual(s.session("T1").cameras_failed(), [RU])
        self.assertNotIn(RU, s.session("T1").pending_work())

    def test_failed_camera_does_not_block_the_others(self):
        s = _sched()
        _submit(s, "T1", *ALL)
        job = s.next_job()
        s.mark_camera_failed(job.train_id, job.camera_id, "boom")
        self.assertEqual(len(_drain(s)), 3)
        self.assertTrue(s.session("T1").is_complete(),
                        "every feed reached a terminal state")

    def test_retry_returns_the_job_to_the_queue(self):
        s = _sched()
        _submit(s, "T1", RU)
        job = s.next_job()
        s.mark_camera_failed("T1", RU, "transient", retry=True)
        self.assertEqual(s.session("T1").jobs[RU].state, JOB_RECEIVED)
        again = s.next_job()
        self.assertEqual(again.camera_id, RU)
        self.assertEqual(again.attempts, 2)

    def test_duplicate_completion_is_ignored(self):
        s = _sched()
        _submit(s, "T1", RU)
        job = s.next_job()
        s.mark_camera_completed("T1", RU)
        s.mark_camera_completed("T1", RU)       # must not raise or change state
        self.assertEqual(s.session("T1").jobs[RU].state, JOB_COMPLETED)

    def test_unknown_train_or_camera_raises(self):
        s = _sched()
        with self.assertRaises(SchedulerError):
            s.mark_camera_completed("nope", RU)
        with self.assertRaises(SchedulerError):
            s.submit_camera_video(camera_id="NOT_A_CAMERA",
                                  video_path="/v/x.mp4", train_id="T1")


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class TestConcurrency(unittest.TestCase):

    def test_parallel_workers_never_get_the_same_camera_twice(self):
        s = _sched()
        for i in range(6):
            _submit(s, f"T_{i:04d}", *ALL)

        handed: list = []
        lock = threading.Lock()

        def worker():
            while True:
                job = s.next_job()
                if job is None:
                    return
                with lock:
                    handed.append((job.train_id, job.camera_id))
                s.mark_camera_completed(job.train_id, job.camera_id)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(handed), 24)
        self.assertEqual(len(set(handed)), 24, "a camera was handed out twice")

    def test_concurrent_submission_is_atomic(self):
        s = _sched()

        def submit(i):
            _submit(s, f"T_{i:04d}", *ALL)

        threads = [threading.Thread(target=submit, args=(i,))
                   for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(len(s.sessions), 10)
        for sess in s.sessions:
            self.assertEqual(len(sess.present_cameras()), 4)

    def test_concurrent_duplicate_submission_yields_one_job(self):
        s = _sched()

        def submit():
            _submit(s, "T1", RU)

        threads = [threading.Thread(target=submit) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(len(_drain(s)), 1)


# ---------------------------------------------------------------------------
# Arbitrary camera counts
# ---------------------------------------------------------------------------

class TestArbitraryCameraCounts(unittest.TestCase):

    def test_two_camera_configuration(self):
        s = TrainScheduler(expected_cameras=(RU, LU), verbose=False)
        _submit(s, "T1", RU, LU)
        self.assertEqual(_drain(s), [("T1", RU), ("T1", LU)])
        self.assertTrue(s.session("T1").is_complete())

    def test_six_camera_configuration(self):
        cams = tuple(f"CAM_{i}" for i in range(6))
        s = TrainScheduler(expected_cameras=cams, verbose=False)
        for c in cams:
            s.submit_camera_video(camera_id=c, video_path=f"/v/{c}.mp4",
                                  train_id="T1", train_timestamp="T1")
        self.assertEqual([c for _, c in _drain(s)], list(cams))

    def test_priority_rule_holds_for_a_single_camera_deployment(self):
        s = TrainScheduler(expected_cameras=(RU,), verbose=False)
        _submit(s, "T_0200", RU)
        _submit(s, "T_0100", RU)
        self.assertEqual([t for t, _ in _drain(s)], ["T_0100", "T_0200"])

    def test_empty_camera_set_is_rejected(self):
        with self.assertRaises(SchedulerError):
            TrainScheduler(expected_cameras=(), verbose=False)


# ---------------------------------------------------------------------------
# Persistence + observability
# ---------------------------------------------------------------------------

class TestPersistenceAndLogging(unittest.TestCase):

    def test_state_is_written_and_reloadable(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sched.json")
            s = TrainScheduler(state_path=path, verbose=False)
            _submit(s, "T1", RU, LU)
            job = s.next_job()
            s.mark_camera_completed(job.train_id, job.camera_id)

            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["schema"], "wagon_eye.train_scheduler.v1")
            sess = doc["sessions"][0]
            self.assertEqual(sess["train_id"], "T1")
            self.assertEqual(sess["missing_cameras"], [RUT, LUT])
            self.assertEqual(sess["completed_cameras"], [RU])
            self.assertEqual(sess["jobs"][RU]["state"], JOB_COMPLETED)
            self.assertEqual(sess["jobs"][LUT]["state"], JOB_WAITING)

    def test_a_broken_state_path_does_not_stop_scheduling(self):
        s = TrainScheduler(state_path="/proc/definitely/not/writable/x.json",
                           verbose=False)
        _submit(s, "T1", RU)
        self.assertIsNotNone(s.next_job())

    def test_every_decision_records_its_reason(self):
        s = _sched()
        _submit(s, "A_1000", RU)
        _submit(s, "B_1005", *ALL)
        s.next_job()
        d = s.decisions[-1]
        self.assertEqual(d.rule, "oldest-processable")
        self.assertIn("A_1000", d.blocked_sessions)
        self.assertIn("all 4 feed(s) in", d.reason)
        self.assertIn("B_1005", d.render())

    def test_a_rule_two_decision_says_it_will_re_evaluate(self):
        s = _sched()
        _submit(s, "A_1000", RU)
        _submit(s, "B_1005", RU, LU)
        s.next_job()
        d = s.decisions[-1]
        self.assertEqual(d.rule, "newest-partial")
        self.assertIn("re-evaluating", d.reason)

    def test_decision_names_the_older_sessions_it_checked(self):
        s = _sched()
        _submit(s, "A_1000", RU)
        _submit(s, "B_1002", LU)
        _submit(s, "C_1005", *ALL)
        s.next_job()
        d = s.decisions[-1]
        self.assertEqual(d.older_sessions_checked, ["A_1000", "B_1002",
                                                    "C_1005"])

    def test_status_table_lists_every_session(self):
        s = _sched()
        _submit(s, "A_1000", RU)
        _submit(s, "B_1005", *ALL)
        out = s.render_status()
        self.assertIn("A_1000", out)
        self.assertIn("B_1005", out)


# ---------------------------------------------------------------------------
# Assembly gating
# ---------------------------------------------------------------------------

class TestAssemblyGating(unittest.TestCase):

    def test_a_train_is_not_assemblable_until_every_feed_is_terminal(self):
        s = _sched()
        _submit(s, "T1", *ALL)
        for _ in range(3):
            job = s.next_job()
            s.mark_camera_completed(job.train_id, job.camera_id)
            self.assertEqual(s.sessions_ready_to_assemble(), [])
        job = s.next_job()
        s.mark_camera_completed(job.train_id, job.camera_id)
        self.assertEqual([x.train_id for x in s.sessions_ready_to_assemble()],
                         ["T1"])

    def test_a_partial_train_is_never_offered_for_assembly(self):
        s = _sched()
        _submit(s, "T1", RU, LU, RUT)
        _drain(s)
        self.assertEqual(s.sessions_ready_to_assemble(), [])

    def test_assembled_sessions_are_not_offered_again(self):
        s = _sched()
        _submit(s, "T1", *ALL)
        _drain(s)
        self.assertEqual(len(s.sessions_ready_to_assemble()), 1)
        s.mark_assembled("T1")
        self.assertEqual(s.sessions_ready_to_assemble(), [])

    def test_assembly_queue_is_oldest_first(self):
        s = _sched()
        _submit(s, "T_0200", *ALL)
        _submit(s, "T_0100", *ALL)
        _drain(s)
        self.assertEqual([x.train_id for x in s.sessions_ready_to_assemble()],
                         ["T_0100", "T_0200"])


# ---------------------------------------------------------------------------
# The scheduler must not become a second pipeline
# ---------------------------------------------------------------------------

class TestSchedulerIsOrchestrationOnly(unittest.TestCase):

    #: Checked against CODE, never against the file text: the module docstring
    #: names `camera_runner` and `global_assembler` on purpose, to say who is
    #: allowed to run a stage. A substring scan would flag that prose and
    #: pressure the docstring into being less clear.
    BANNED = ("camera_runner", "global_assembler", "camera_pipeline",
              "reconstruction", "global_fusion", "wagon_cache_builder",
              "cv2", "ultralytics", "YOLO", "boto3",
              "process_batch", "GapTracker", "run_camera", "assemble")

    def _code_identifiers(self):
        """Every imported module and referenced name, docstrings excluded."""
        import ast
        src = open(os.path.join(V4_ROOT, "orchestrator", "train_scheduler.py"),
                   encoding="utf-8").read()
        tree = ast.parse(src)
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    names.update(a.name.split("."))
            elif isinstance(n, ast.ImportFrom):
                names.update((n.module or "").split("."))
                names.update(a.name for a in n.names)
            elif isinstance(n, ast.Name):
                names.add(n.id)
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
        return names

    def test_module_imports_no_pipeline_stage(self):
        """A scheduler that cannot run inference cannot duplicate a stage."""
        names = self._code_identifiers()
        for banned in self.BANNED:
            with self.subTest(token=banned):
                self.assertNotIn(banned, names)

    def test_module_defines_no_stage_function(self):
        import ast
        src = open(os.path.join(V4_ROOT, "orchestrator", "train_scheduler.py"),
                   encoding="utf-8").read()
        names = {n.name for n in ast.walk(ast.parse(src))
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for banned in ("run", "run_camera", "assemble", "process_batch",
                       "build", "main"):
            self.assertNotIn(banned, names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
