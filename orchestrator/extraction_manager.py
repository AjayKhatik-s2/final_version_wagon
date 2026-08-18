"""ExtractionManager -- the raw->trimmed production stage of the pipeline.

Owned by the orchestrator when the pipeline source is RAW (see
core.pipeline_source).  Its responsibilities are exactly, and only:

    * raw S3 discovery              -- list not-yet-handled raw clips per camera
    * train-completion detection    -- delegated to the extractor's segment
                                       finder (a clip that only holds the leading
                                       part of a train is held as "ongoing" until
                                       its continuation arrives)
    * invoking the train extractor  -- the single vendored implementation in
                                       train_extraction/ (no second copy)
    * producing trimmed train videos-- uploaded to the trimmed input location the
                                       orchestrator consumes

It performs NO inspection and knows NOTHING about GlobalTrainState, fusion, or
reports.  master_runner owns its lifecycle (`start()` / `stop()`), and keeps
consuming trimmed clips from S3 exactly as in the pure-consumer path -- the two
halves stay decoupled through S3 (the extractor's trimmed bucket == the
orchestrator's WAGONEYE_S3_INPUT_*).  This keeps master_runner a high-level
coordinator: it wires the source and owns the manager's lifecycle, but contains
no extraction-orchestration logic itself.

Implementation note: the actual discovery+extract+upload for one camera is the
SAME `sweep_camera` the standalone `train_extraction.run_extraction_service`
runs, imported and reused here -- so the codebase has exactly one extraction
implementation and one place that talks to the extractor.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Sequence

from core.logging_setup import get_logger

log = get_logger("extraction_manager")


class ExtractionManager:
    """Continuously (or once) turn raw CCTV into trimmed train clips.

    Parameters
    ----------
    cameras:
        Restrict to a subset of cameras; ``None`` (default) uses all four.
    poll_interval:
        Seconds between full raw->trimmed sweeps in continuous mode.
    logger:
        Optional logger override (defaults to the module logger).
    """

    def __init__(
        self,
        *,
        cameras: Optional[Sequence[str]] = None,
        poll_interval: int = 60,
        logger=None,
    ) -> None:
        self._cameras = tuple(cameras) if cameras else None  # resolved lazily
        self._poll_interval = max(1, int(poll_interval))
        self._log = logger or log
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Spawn the background production loop.  Idempotent."""
        if self.is_running():
            return
        self._stop.clear()
        self._reset_sweep_flag(False)
        self._thread = threading.Thread(
            target=self._loop, name="extraction-manager", daemon=True)
        self._thread.start()
        self._log.info("[EXTRACT-MGR] started: cameras=%s poll=%ss",
                       self._resolve_cameras(), self._poll_interval)

    def stop(self, timeout: float = 30.0) -> None:
        """Signal the loop to stop and join it.  Breaks an in-flight sweep at
        the next raw-key boundary (the extractor finishes the current key)."""
        self._stop.set()
        self._reset_sweep_flag(True)   # bridge stop into the reused sweep
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._log.info("[EXTRACT-MGR] stopped")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # -- single sweep (also used for --once / tests) -----------------------

    def run_once(self) -> Dict[str, int]:
        """Run exactly one raw->trimmed sweep of all selected cameras and return
        aggregate counts.  Used by `--once` and by tests."""
        self._stop.clear()          # a prior stop() must not suppress this sweep
        self._reset_sweep_flag(False)
        return self._sweep_all()

    # -- internals ---------------------------------------------------------

    def _resolve_cameras(self):
        # Imported lazily so importing this module never pulls in ultralytics.
        from train_extraction import driver as EXD
        return self._cameras or EXD.ALL_CAMERAS

    @staticmethod
    def _reset_sweep_flag(value: bool) -> None:
        """Set the reused producer module's stop flag (best-effort).  Setting it
        True lets an in-progress `sweep_camera` break between raw keys; setting
        it False re-arms a fresh run."""
        try:
            from train_extraction import run_extraction_service as RES
            RES._STOP = value
        except Exception:
            pass

    def _sweep_all(self) -> Dict[str, int]:
        from train_extraction import run_extraction_service as RES
        agg = {"listed": 0, "new": 0, "trains": 0, "errors": 0}
        for cam in self._resolve_cameras():
            if self._stop.is_set():
                break
            try:
                r = RES.sweep_camera(cam)
                for k in agg:
                    agg[k] += int(r.get(k, 0))
                # Quiet when idle; speak up only on new work / errors.
                if r.get("new") or r.get("errors") or r.get("trains"):
                    self._log.info(
                        "[EXTRACT-MGR/%s] listed=%d new=%d trains=%d errors=%d",
                        cam, r["listed"], r["new"], r["trains"], r["errors"])
            except Exception as e:
                agg["errors"] += 1
                self._log.error("[EXTRACT-MGR/%s] sweep crashed: %s", cam, e,
                                exc_info=True)
        return agg

    def _loop(self) -> None:
        self._log.info("[EXTRACT-MGR] production loop running")
        while not self._stop.is_set():
            self._sweep_all()
            # interruptible sleep between sweeps
            self._stop.wait(self._poll_interval)
        self._log.info("[EXTRACT-MGR] production loop exiting")
