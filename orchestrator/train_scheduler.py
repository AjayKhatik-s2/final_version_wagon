"""Priority-aware Train Session scheduler -- an ORCHESTRATION LAYER ONLY.

What this module decides
-----------------------
Which existing camera job runs next.  Nothing else.  It never opens a video,
never loads a model, never writes evidence and never renders a report.  Stage 1
through Stage 5 are reached exclusively through the functions that already own
them (`orchestrator.camera_runner.run_camera` for a camera,
`orchestrator.global_assembler.assemble` for a train), and this file does not
import either of them -- the caller does.  A scheduler that cannot run
inference cannot accidentally become a second pipeline.

The rule it implements
----------------------
Chronological priority, but never idle:

1. Walk incomplete sessions oldest-first.  The first one whose every expected
   camera feed has ARRIVED is *processable*: hand out its jobs until the train
   is done.  A newer train never gets a job while an older train is
   processable.
2. If no session is processable -- every incomplete one is still missing a feed
   -- do NOT block.  Hand out exactly ONE job from the NEWEST session that has
   an unprocessed arrived feed, then let the caller come straight back here.
3. After every completed job, step 1 runs again from scratch.  The moment a
   blocked older train's missing feed arrives, it takes priority.

Why this cannot starve a middle-aged train
------------------------------------------
Rule 2 says "newest", which looks like it could feed the newest train forever
while a middle one waits.  It cannot: a session stops being *eligible* for rule
2 once it has no unprocessed arrived feed left, and a session has at most
`len(expected_cameras)` feeds.  So each session can absorb at most that many
rule-2 picks before eligibility passes to the next-newest.  Progress is
therefore bounded and every eligible session is reached.  This is a property of
the rule, not an extra fairness heuristic layered on top.

One point where two requirements pulled apart
---------------------------------------------
The brief says, of a blocked oldest train, "process only one additional
eligible camera job from the NEWEST available train".  It also says "older
processable trains always take priority over newer trains" and "maintain
chronological priority among processable sessions".

Those disagree when the oldest train is blocked AND two younger trains are
both processable.  Taken literally, "newest" would drain the youngest train
first and only then the middle one -- inverting chronological order between two
trains that were both ready.

This implementation follows the priority requirement: among *processable*
sessions the oldest always wins (rule 1), and "one job from the newest" applies
where it was actually needed -- when NO session is processable at all (rule 2),
so the choice is between doing some available work and idling.

Nothing is lost by that reading.  The purpose of "exactly one, then return" is
that a blocked older train must be re-checked constantly rather than after a
whole train has run, and that still holds exactly: `decide()` re-walks every
session from the oldest on EVERY call, so a blocked train is re-examined
between every pair of camera jobs, and takes over the instant its last feed
lands.

Preemption
----------
Decisions happen at job boundaries only.  A job that has been handed out runs
to completion; nothing here can interrupt inference mid-operation.  An older
train that becomes processable is picked at the NEXT boundary.

Camera count
------------
`expected_cameras` is per-session and defaults to `C.ALL_CAMERAS`.  Nothing in
this file assumes four, assumes their names, or assumes which is master.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C
from core.logging_setup import get_logger

log = get_logger("orchestrator.train_scheduler")

SCHEMA = "wagon_eye.train_scheduler.v1"


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

#: A camera job's state.  Deliberately COARSER than
#: `core.camera_evidence.LIFECYCLE`: that state machine (PENDING -> TRACKING ->
#: ... -> SEALED) is the persisted truth about what a camera has actually
#: produced, and it stays the authority.  The scheduler only needs to know
#: whether a job is runnable, running, or finished, so it keeps its own five
#: states and never writes to a CameraEvidenceBundle.
JOB_WAITING = "WAITING"        # expected, feed has not arrived
JOB_RECEIVED = "RECEIVED"      # feed on disk, not yet handed out
JOB_PROCESSING = "PROCESSING"  # handed out to a worker
JOB_COMPLETED = "COMPLETED"
JOB_FAILED = "FAILED"

JOB_TERMINAL = (JOB_COMPLETED, JOB_FAILED)

SESSION_WAITING = "WAITING"        # some expected feed has not arrived
SESSION_RECEIVED = "RECEIVED"      # every feed arrived, no job started
SESSION_PROCESSING = "PROCESSING"  # at least one job started, work remains
SESSION_COMPLETED = "COMPLETED"    # every arrived feed reached a terminal state
SESSION_FAILED = "FAILED"          # every arrived feed FAILED


class SchedulerError(RuntimeError):
    """An illegal scheduler operation -- never silenced."""


# ---------------------------------------------------------------------------
# Camera job
# ---------------------------------------------------------------------------

@dataclass
class CameraJob:
    """One camera of one train.  Carries Train ID + camera ID, as required."""
    train_id: str
    camera_id: str
    state: str = JOB_WAITING
    video_path: str = ""
    received_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    attempts: int = 0
    failure_reason: str = ""

    @property
    def has_feed(self) -> bool:
        return self.state != JOB_WAITING

    @property
    def is_runnable(self) -> bool:
        return self.state == JOB_RECEIVED

    @property
    def is_terminal(self) -> bool:
        return self.state in JOB_TERMINAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_id": self.train_id, "camera_id": self.camera_id,
            "state": self.state, "video_path": self.video_path,
            "received_at": self.received_at, "started_at": self.started_at,
            "finished_at": self.finished_at, "attempts": self.attempts,
            "failure_reason": self.failure_reason,
        }


# ---------------------------------------------------------------------------
# Train session
# ---------------------------------------------------------------------------

@dataclass
class TrainSession:
    """One physical train pass, and the state of each of its camera feeds.

    `train_id` is the EXISTING batch key -- see `TrainScheduler.submit_camera_video`
    -- so a session is the same thing the rest of the pipeline already calls a
    batch.  No second identity scheme is introduced.
    """
    train_id: str
    train_timestamp: str
    expected_cameras: Tuple[str, ...] = C.ALL_CAMERAS
    discovered_at: float = 0.0
    arrival_seq: int = 0
    jobs: Dict[str, CameraJob] = field(default_factory=dict)
    assembled: bool = False

    def __post_init__(self) -> None:
        for cam in self.expected_cameras:
            self.jobs.setdefault(cam, CameraJob(train_id=self.train_id,
                                                camera_id=cam))

    # ---- feed inventory ------------------------------------------------
    # Named to match `core.batch.TrainBatch.present_cameras/missing_cameras`
    # so the two read the same way at a call site.

    def present_cameras(self) -> List[str]:
        return [c for c in self.expected_cameras if self.jobs[c].has_feed]

    def missing_cameras(self) -> List[str]:
        return [c for c in self.expected_cameras if not self.jobs[c].has_feed]

    def cameras_in(self, *states: str) -> List[str]:
        return [c for c in self.expected_cameras if self.jobs[c].state in states]

    def runnable_cameras(self) -> List[str]:
        """Arrived, not handed out, not finished -- in the configured order."""
        return self.cameras_in(JOB_RECEIVED)

    # ---- lifecycle predicates ------------------------------------------

    def feeds_complete(self) -> bool:
        """Every EXPECTED camera feed has arrived."""
        return not self.missing_cameras()

    def is_processable(self) -> bool:
        """Can be driven to completion right now: all feeds in, work left."""
        return self.feeds_complete() and bool(self.pending_work())

    def pending_work(self) -> List[str]:
        """Cameras with a feed that have not reached a terminal state."""
        return [c for c in self.present_cameras()
                if not self.jobs[c].is_terminal]

    def is_eligible_for_partial(self) -> bool:
        """Has arrived-but-unprocessed work while still missing a feed."""
        return bool(self.runnable_cameras())

    def cameras_done(self) -> List[str]:
        return self.cameras_in(JOB_COMPLETED)

    def cameras_failed(self) -> List[str]:
        return self.cameras_in(JOB_FAILED)

    def is_complete(self) -> bool:
        """Every ARRIVED feed reached a terminal state AND no feed is missing.

        A session is NOT complete merely because one camera finished, and not
        complete while a feed is still expected -- that is what keeps a
        3-camera train from being assembled as if it were whole.  Whether a
        given set of sealed cameras may actually be assembled remains the
        existing pipeline's decision (`camera_evidence.ready_for_global_assembly`
        requires the master); the scheduler does not second-guess it.
        """
        return self.feeds_complete() and not self.pending_work()

    def state(self) -> str:
        if not self.feeds_complete():
            return SESSION_WAITING
        if self.pending_work():
            started = self.cameras_in(JOB_PROCESSING, JOB_COMPLETED, JOB_FAILED)
            return SESSION_PROCESSING if started else SESSION_RECEIVED
        done, failed = self.cameras_done(), self.cameras_failed()
        if failed and not done:
            return SESSION_FAILED
        return SESSION_COMPLETED

    def age_seconds(self, now: Optional[float] = None) -> float:
        return max(0.0, (now if now is not None else time.time())
                   - self.discovered_at)

    # ---- ordering ------------------------------------------------------

    @property
    def sort_key(self) -> Tuple[str, int, str]:
        """Chronological, with deterministic tie-breaks.

        `train_timestamp` is `YYYYMMDD_HHMMSS`, so lexicographic order IS
        chronological order.  Two trains sharing a timestamp fall back to
        arrival sequence and then train_id, so the ordering is total and
        reproducible regardless of dict iteration or thread interleaving.
        """
        return (self.train_timestamp, self.arrival_seq, self.train_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_id": self.train_id,
            "train_timestamp": self.train_timestamp,
            "expected_cameras": list(self.expected_cameras),
            "discovered_at": self.discovered_at,
            "arrival_seq": self.arrival_seq,
            "state": self.state(),
            "assembled": self.assembled,
            "present_cameras": self.present_cameras(),
            "missing_cameras": self.missing_cameras(),
            "completed_cameras": self.cameras_done(),
            "failed_cameras": self.cameras_failed(),
            "pending_work": self.pending_work(),
            "jobs": {c: j.to_dict() for c, j in self.jobs.items()},
        }


# ---------------------------------------------------------------------------
# Decision record -- so a log line can be asserted on in a test
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    """Why the scheduler returned what it returned."""
    job: Optional[CameraJob]
    reason: str
    rule: str                        # "oldest-processable" | "newest-partial" | "idle"
    older_sessions_checked: List[str] = field(default_factory=list)
    blocked_sessions: List[str] = field(default_factory=list)

    def render(self) -> str:
        if self.job is None:
            return f"[SCHED] no job: {self.reason}"
        return (f"[SCHED] {self.job.train_id}/{self.job.camera_id} "
                f"rule={self.rule} {self.reason}")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class TrainScheduler:
    """Single source of truth for which camera job runs next.

    Thread-safe: every public method takes one re-entrant lock, and `next_job`
    transitions the chosen job to PROCESSING *inside* that lock, so the same
    camera can never be handed to two workers.
    """

    def __init__(
        self,
        *,
        expected_cameras: Sequence[str] = C.ALL_CAMERAS,
        state_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        self.expected_cameras: Tuple[str, ...] = tuple(expected_cameras)
        if not self.expected_cameras:
            raise SchedulerError("expected_cameras must not be empty")
        self._sessions: Dict[str, TrainSession] = {}
        self._lock = threading.RLock()
        self._seq = 0
        self._decisions: List[Decision] = []
        self.state_path = state_path
        self.verbose = verbose

    # ---- introspection -------------------------------------------------

    @property
    def sessions(self) -> List[TrainSession]:
        """All sessions, oldest first."""
        with self._lock:
            return sorted(self._sessions.values(), key=lambda s: s.sort_key)

    def session(self, train_id: str) -> Optional[TrainSession]:
        with self._lock:
            return self._sessions.get(train_id)

    @property
    def decisions(self) -> List[Decision]:
        with self._lock:
            return list(self._decisions)

    def incomplete_sessions(self) -> List[TrainSession]:
        return [s for s in self.sessions if not s.is_complete()]

    # ---- ingestion -----------------------------------------------------

    def submit_camera_video(
        self,
        *,
        camera_id: str,
        video_path: str,
        train_id: str,
        train_timestamp: Optional[str] = None,
        discovered_at: Optional[float] = None,
    ) -> CameraJob:
        """Register one arrived camera feed against its Train Session.

        `train_id` comes from the CALLER, which must derive it with the
        pipeline's existing identification logic -- `core.batch.parse_train_timestamp`
        plus the clustering in `orchestrator.train_batch_manager.poll_for_batches`,
        or a `TrainBatch.batch_key` directly.  Deciding which train a clip
        belongs to is emphatically not this module's job; re-deriving it here
        would create a second, divergent notion of train identity.

        Duplicate submissions are IGNORED (logged, not raised): the same clip
        may legitimately be re-discovered by a poll, and a camera must never be
        processed twice.
        """
        if camera_id not in self.expected_cameras:
            raise SchedulerError(
                f"{camera_id} is not one of expected_cameras "
                f"{list(self.expected_cameras)}")
        with self._lock:
            sess = self._sessions.get(train_id)
            if sess is None:
                self._seq += 1
                sess = TrainSession(
                    train_id=train_id,
                    train_timestamp=train_timestamp or train_id,
                    expected_cameras=self.expected_cameras,
                    discovered_at=(discovered_at if discovered_at is not None
                                   else time.time()),
                    arrival_seq=self._seq,
                )
                self._sessions[train_id] = sess
                self._log("[SCHED] NEW SESSION %s (ts=%s, expects %d camera(s))",
                          train_id, sess.train_timestamp,
                          len(self.expected_cameras))

            job = sess.jobs[camera_id]
            if job.has_feed:
                self._log("[SCHED] DUPLICATE feed ignored: %s/%s already %s",
                          train_id, camera_id, job.state)
                return job

            job.state = JOB_RECEIVED
            job.video_path = video_path
            job.received_at = time.time()
            self._log("[SCHED] RECEIVED %s/%s  (%d/%d feeds in, missing=%s)",
                      train_id, camera_id, len(sess.present_cameras()),
                      len(self.expected_cameras), sess.missing_cameras())
            self._persist()
            return job

    # ---- the decision --------------------------------------------------

    def next_job(self) -> Optional[CameraJob]:
        """Return the next camera job to run, or None if there is nothing to do.

        Atomically marks the returned job PROCESSING.
        """
        return self.decide().job

    def decide(self) -> Decision:
        """`next_job` plus the reasoning, for logs and tests."""
        with self._lock:
            ordered = sorted(self._sessions.values(), key=lambda s: s.sort_key)
            checked: List[str] = []
            blocked: List[str] = []

            # ---- RULE 1: oldest processable session wins, always ----------
            # Walking oldest-first means a session that is already mid-flight
            # keeps being chosen (it is still the oldest processable one), so a
            # newer train cannot interleave into it.  An OLDER train that only
            # just became processable does take over at this boundary -- that
            # is chronological priority working as specified, and it is safe
            # because we are between jobs, never inside one.
            for s in ordered:
                if s.is_complete():
                    continue
                checked.append(s.train_id)
                if not s.feeds_complete():
                    blocked.append(s.train_id)
                    continue
                runnable = s.runnable_cameras()
                if not runnable:
                    # All feeds in, but every remaining job is already out with
                    # a worker.  Nothing to hand out; do NOT fall through to a
                    # newer train, or we would violate chronological priority
                    # while this train is still finishing.
                    if s.pending_work():
                        d = Decision(
                            job=None, rule="oldest-processable",
                            reason=(f"{s.train_id} is processable and its "
                                    f"remaining job(s) {s.pending_work()} are "
                                    f"already PROCESSING -- holding priority"),
                            older_sessions_checked=checked,
                            blocked_sessions=blocked)
                        return self._record(d)
                    continue
                cam = runnable[0]
                job = self._hand_out(s, cam)
                d = Decision(
                    job=job, rule="oldest-processable",
                    reason=(f"oldest processable session (age "
                            f"{s.age_seconds():.0f}s, ts={s.train_timestamp}); "
                            f"all {len(self.expected_cameras)} feed(s) in; "
                            f"remaining={s.pending_work()}"),
                    older_sessions_checked=checked,
                    blocked_sessions=blocked)
                return self._record(d)

            # ---- RULE 2: nothing processable -- do NOT idle ---------------
            # Every incomplete session is still waiting on a feed.  Take
            # exactly one job from the NEWEST session that has arrived work,
            # then the caller re-enters here and rule 1 is retried from the
            # top.  Bounded by feeds-per-session, so no session starves.
            for s in reversed(ordered):
                if s.is_complete() or not s.is_eligible_for_partial():
                    continue
                cam = s.runnable_cameras()[0]
                job = self._hand_out(s, cam)
                d = Decision(
                    job=job, rule="newest-partial",
                    reason=(f"no older session is processable "
                            f"(blocked={blocked}); running ONE job from the "
                            f"newest eligible session (ts={s.train_timestamp}, "
                            f"missing={s.missing_cameras()}), then "
                            f"re-evaluating"),
                    older_sessions_checked=checked,
                    blocked_sessions=blocked)
                return self._record(d)

            d = Decision(
                job=None, rule="idle",
                reason=(f"nothing runnable; {len(blocked)} session(s) waiting "
                        f"on feeds: {blocked}" if blocked
                        else "no sessions with work"),
                older_sessions_checked=checked, blocked_sessions=blocked)
            return self._record(d)

    def _hand_out(self, sess: TrainSession, camera_id: str) -> CameraJob:
        """Transition RECEIVED -> PROCESSING.  Caller holds the lock."""
        job = sess.jobs[camera_id]
        if job.state != JOB_RECEIVED:
            raise SchedulerError(
                f"{sess.train_id}/{camera_id} is {job.state}, not "
                f"{JOB_RECEIVED} -- refusing to hand it out twice")
        job.state = JOB_PROCESSING
        job.started_at = time.time()
        job.attempts += 1
        self._persist()
        return job

    # ---- completion ----------------------------------------------------

    def mark_camera_completed(self, train_id: str, camera_id: str) -> TrainSession:
        return self._finish(train_id, camera_id, JOB_COMPLETED, "")

    def mark_camera_failed(
        self, train_id: str, camera_id: str, reason: str = "",
        *, retry: bool = False,
    ) -> TrainSession:
        """Mark a camera FAILED, or return it to the queue when `retry`.

        `retry=False` is the default because it matches the existing semantics:
        `camera_runner.run_camera` never raises -- it seals the camera FAILED
        and returns -- and global assembly then decides on its own whether a
        failed support camera is tolerable.  The scheduler does not invent a
        retry policy on top of that.
        """
        if retry:
            with self._lock:
                sess = self._require(train_id)
                job = sess.jobs[camera_id]
                job.state = JOB_RECEIVED
                job.failure_reason = reason
                job.started_at = 0.0
                self._log("[SCHED] RETRY queued %s/%s (attempt %d): %s",
                          train_id, camera_id, job.attempts, reason)
                self._persist()
                return sess
        return self._finish(train_id, camera_id, JOB_FAILED, reason)

    def _finish(self, train_id: str, camera_id: str, state: str,
                reason: str) -> TrainSession:
        with self._lock:
            sess = self._require(train_id)
            if camera_id not in sess.jobs:
                raise SchedulerError(f"{train_id} has no camera {camera_id}")
            job = sess.jobs[camera_id]
            if job.is_terminal:
                self._log("[SCHED] %s/%s already %s -- ignoring duplicate "
                          "completion", train_id, camera_id, job.state)
                return sess
            job.state = state
            job.failure_reason = reason
            job.finished_at = time.time()
            self._log("[SCHED] %s %s/%s  (session now %s; done=%s failed=%s "
                      "pending=%s missing=%s)",
                      state, train_id, camera_id, sess.state(),
                      sess.cameras_done(), sess.cameras_failed(),
                      sess.pending_work(), sess.missing_cameras())
            self._persist()
            return sess

    def mark_assembled(self, train_id: str) -> TrainSession:
        """Record that the train's combined/global step has run."""
        with self._lock:
            sess = self._require(train_id)
            sess.assembled = True
            self._log("[SCHED] ASSEMBLED %s", train_id)
            self._persist()
            return sess

    def sessions_ready_to_assemble(self) -> List[TrainSession]:
        """Complete sessions whose combined step has not run yet, oldest first.

        Whether such a session may ACTUALLY be assembled stays the existing
        pipeline's call -- `global_assembler.assemble` re-checks
        `ready_for_global_assembly` and refuses if the master is not sealed.
        """
        return [s for s in self.sessions if s.is_complete() and not s.assembled]

    def _require(self, train_id: str) -> TrainSession:
        sess = self._sessions.get(train_id)
        if sess is None:
            raise SchedulerError(f"unknown train_id {train_id}")
        return sess

    # ---- persistence / logging ----------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "schema": SCHEMA,
                "expected_cameras": list(self.expected_cameras),
                "sessions": [s.to_dict() for s in self.sessions],
            }

    def _persist(self) -> None:
        if not self.state_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.state_path))
                        or ".", exist_ok=True)
            tmp = f"{self.state_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.snapshot(), f, indent=2)
            os.replace(tmp, self.state_path)
        except OSError as e:
            # Persistence is for operator visibility and recovery reasoning.
            # Losing it must never stop a train from being processed.
            log.warning("[SCHED] could not persist state to %s: %s",
                        self.state_path, e)

    def _record(self, d: Decision) -> Decision:
        self._decisions.append(d)
        if self.verbose:
            if d.job is not None:
                self._log("[SCHED] -> %s/%s  rule=%s  checked=%s  blocked=%s\n"
                          "        reason: %s",
                          d.job.train_id, d.job.camera_id, d.rule,
                          d.older_sessions_checked, d.blocked_sessions,
                          d.reason)
            else:
                self._log("[SCHED] -> (none)  rule=%s  reason: %s",
                          d.rule, d.reason)
        return d

    def _log(self, fmt: str, *args: Any) -> None:
        log.info(fmt, *args)
        if self.verbose:
            print(fmt % args if args else fmt)

    def render_status(self) -> str:
        """Operator-readable table of every session."""
        lines = [f"  {'TRAIN':<20}{'STATE':<12}{'IN':<6}{'DONE':<6}"
                 f"{'MISSING'}"]
        for s in self.sessions:
            lines.append(
                f"  {s.train_id:<20}{s.state():<12}"
                f"{len(s.present_cameras()):<6}{len(s.cameras_done()):<6}"
                f"{s.missing_cameras() or '-'}")
        return "\n".join(lines)
