"""Continuous train-extraction producer service.

Polls each camera's RAW CCTV bucket, cuts out the train pass with the vendored
V4 extractor, and uploads the trimmed clip(s) to that camera's TRIMMED bucket.
It does NO inspection -- point the trimmed buckets at the prefixes that
`wagon_eye_v4_new`'s `--auto` orchestrator polls and the two run independently.

    python -m train_extraction.run_extraction_service                 # all 4 cameras
    python -m train_extraction.run_extraction_service --camera RIGHT_UP
    python -m train_extraction.run_extraction_service --once          # one sweep, exit
    python -m train_extraction.run_extraction_service --dry-run       # list only, no extract

Processed raw keys are remembered in a small local JSON ledger under
WAGONEYE_EXTRACTION_STATE_DIR (default <root>/logs/extraction_state) so a
restart never re-extracts an already-handled raw clip.  (The extractor's own
S3 state store additionally preserves cross-clip ongoing-train continuity.)

Graceful shutdown: SIGTERM/SIGINT finish the current key, then exit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from typing import Dict, List, Optional, Set

from . import driver as D

log = logging.getLogger("extraction.service")

_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".ts")

_STOP = False


def _handle_stop(signum, _frame):
    global _STOP
    _STOP = True
    log.info("signal %s received -- finishing current key then exiting", signum)


# -----------------------------------------------------------------------------
# local processed-key ledger (per camera)
# -----------------------------------------------------------------------------

def _state_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.environ.get("WAGONEYE_EXTRACTION_STATE_DIR",
                       os.path.join(root, "logs", "extraction_state"))
    os.makedirs(d, exist_ok=True)
    return d


def _ledger_path(camera: str) -> str:
    return os.path.join(_state_dir(), f"processed_{camera.lower()}.json")


def _load_ledger(camera: str) -> Set[str]:
    p = _ledger_path(camera)
    if not os.path.isfile(p):
        return set()
    try:
        with open(p, "r", encoding="utf-8") as f:
            return set(json.load(f).get("processed", []))
    except (OSError, json.JSONDecodeError):
        return set()


def _save_ledger(camera: str, processed: Set[str]) -> None:
    p = _ledger_path(camera)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"processed": sorted(processed)}, f, indent=2)
    os.replace(tmp, p)


# -----------------------------------------------------------------------------
# one sweep of one camera
# -----------------------------------------------------------------------------

def lookback_minutes() -> float:
    """How far back to consider raw clips, in minutes (0 = no limit).

    A raw bucket holds months of CCTV.  Without a window, a fresh install (or any
    install whose dedup ledger is incomplete) starts at the OLDEST clip and grinds
    forward through the entire history before it reaches anything current -- which
    is exactly what happened on first run: it began with a February clip.

    The window makes "start the pipeline" mean "process trains from RIGHT NOW",
    which is what an operator expects.  Default 10 minutes.

    TRADE-OFF: clips older than the window are skipped PERMANENTLY, so if the
    service is down longer than this, video from the gap is never extracted.
    Raise it to cover your expected downtime, or set 0 to process everything the
    dedup ledger hasn't already handled (the old behaviour).
    """
    raw = os.environ.get("WAGONEYE_EXTRACTION_LOOKBACK_MINUTES")
    if raw is None or raw == "":
        return 10.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 10.0


def _list_raw_keys(ex, raw_bucket: str) -> List[str]:
    """Sorted video keys under the camera's raw bucket/prefix, newest-first window.

    `raw_bucket` carries the camera's prefix (``bucket/camera_CCTV_...``), which
    `S3Client.list_objects` now applies -- so this lists ONLY that camera's
    folder, not the whole bucket.

    Objects older than `lookback_minutes()` are dropped, judged by the S3 object's
    own LastModified (authoritative; no filename parsing needed).  An object with
    no LastModified is KEPT rather than silently discarded.
    """
    objs = ex.s3.list_objects(raw_bucket)
    vids = [o for o in objs
            if str(o.get("Key", "")).lower().endswith(_VIDEO_EXTS)]

    cutoff, window_desc = _raw_cutoff()
    if cutoff is None:
        return sorted(o["Key"] for o in vids)

    from datetime import timezone
    fresh, stale = [], 0
    for o in vids:
        lm = o.get("LastModified")
        if lm is None:
            fresh.append(o["Key"])
            continue
        if getattr(lm, "tzinfo", None) is None:
            lm = lm.replace(tzinfo=timezone.utc)
        if lm >= cutoff:
            fresh.append(o["Key"])
        else:
            stale += 1
    if stale:
        _log_skip_once(raw_bucket, window_desc, stale, len(vids))
    return sorted(fresh)


#: Last skip-summary logged per raw bucket, so an idle sweep stays silent.
_LAST_SKIP: Dict[str, tuple] = {}

#: Last dedup summary logged per camera, for the same reason.
_LAST_DEDUP: Dict[str, tuple] = {}


def _log_skip_once(raw_bucket: str, window_desc: str, stale: int,
                   total: int) -> None:
    """Log the skip summary only when it CHANGES for this camera.

    With no feed, every 60s sweep skipped an identical ~35k clips and said so --
    four cameras x 1440 sweeps = ~11.5k identical lines a day, which buries the
    one line that matters (a train arriving) and fills the disk with nothing.

    The count changes the moment new raw video lands or the operational day
    rolls, so a real event still logs immediately.  This gates the MESSAGE only;
    the filtering above is untouched.
    """
    key = (window_desc, stale, total)
    if _LAST_SKIP.get(raw_bucket) == key:
        return
    _LAST_SKIP[raw_bucket] = key
    log.info("%s: %d of %d raw clip(s) are older than the window and were "
             "skipped", window_desc, stale, total)


def _raw_cutoff():
    """`(cutoff_or_None, description)` for raw-clip discovery.

    DEFAULT: the operational-day anchor (05:00 IST) -- the previous production
    rule, so a restart at any hour still sees today's whole operational day.
    An explicit WAGONEYE_EXTRACTION_LOOKBACK_MINUTES switches to a sliding window.
    """
    from datetime import datetime, timedelta, timezone
    raw = os.environ.get("WAGONEYE_EXTRACTION_LOOKBACK_MINUTES")
    if raw is not None and raw != "":
        mins = lookback_minutes()
        if mins <= 0:
            return None, "no window"
        return (datetime.now(timezone.utc) - timedelta(minutes=mins),
                f"lookback {mins:.0f}min")
    try:
        from core import config as CFG
        cutoff = CFG.discovery_cutoff_utc()
        return cutoff, (f"operational day from "
                        f"{cutoff.astimezone(CFG.IST):%Y-%m-%d %H:%M} IST")
    except Exception:
        # Standalone use without the inspection package: fall back to the same
        # 05:00 IST rule computed locally.
        ist = timezone(timedelta(hours=5, minutes=30))
        now_ist = datetime.now(ist)
        start = now_ist.replace(hour=5, minute=0, second=0, microsecond=0)
        if now_ist.hour < 5:
            start -= timedelta(days=1)
        return start.astimezone(timezone.utc), "operational day (05:00 IST)"


def _s3_processed(ex) -> Set[str]:
    """Already-cut raw keys from the extractor's S3 state store (best-effort)."""
    try:
        return set(getattr(ex.state, "processed_videos", None) or set())
    except Exception:
        return set()


def sweep_camera(camera: str, *, dry_run: bool = False) -> Dict[str, int]:
    """Extract every not-yet-processed raw clip for one camera."""
    result = {"listed": 0, "new": 0, "trains": 0, "errors": 0, "foreign": 0}
    try:
        ex = D.get_extractor(camera)
    except FileNotFoundError as e:
        log.error("[%s] cannot start (missing model): %s", camera, e)
        result["errors"] += 1
        return result
    raw_bucket = D.raw_bucket_for(camera)

    # Dedup set = the LOCAL ledger UNION the extractor's S3 state store.
    #
    # The local ledger lives in the checkout (logs/extraction_state/), so a fresh
    # clone or a moved install starts empty -- and because `extract()` only ever
    # ADDS to the S3 state and never consults it, an empty ledger meant the sweep
    # re-extracted the entire raw history from the oldest clip forward.  Folding
    # the S3 state in makes the dedup survive a rebuild: it is the authoritative
    # record of what this pipeline has already cut.
    processed = _load_ledger(camera)
    s3_seen = _s3_processed(ex)
    if s3_seen:
        before = len(processed)
        processed |= s3_seen
        if len(processed) > before:
            # Same reasoning as _log_skip_once: with a dry feed the local ledger
            # never grows, so this summary is byte-identical every sweep.
            key = (before, len(s3_seen), len(processed))
            if _LAST_DEDUP.get(camera) != key:
                _LAST_DEDUP[camera] = key
                log.info("[%s] dedup: %d local + %d from the S3 state store -> "
                         "%d keys", camera, before, len(s3_seen), len(processed))
    keys = _list_raw_keys(ex, raw_bucket)
    result["listed"] = len(keys)

    # The camera's own prefix within the raw bucket ("" when the bucket has none).
    expected = raw_bucket.split("/", 1)[1] if "/" in raw_bucket else ""
    for key in keys:
        if _STOP:
            break
        if key in processed:
            continue
        # Defence in depth: never hand one camera's video to another camera's
        # extractor.  A listing that escapes its prefix (or a mis-set bucket) would
        # otherwise run the SIDE classifier over a TOP view and cut nonsense.
        if expected and not key.startswith(expected):
            result["foreign"] = result.get("foreign", 0) + 1
            log.warning("[%s] SKIP foreign key not under %s/: %s",
                        camera, expected, key)
            continue
        result["new"] += 1
        if dry_run:
            log.info("[%s] DRY-RUN would extract: %s", camera, key)
            continue
        try:
            trains = D.extract_trains(camera, key)
            result["trains"] += len(trains)
            for t in trains:
                log.info("[%s] trimmed -> %s", camera,
                         getattr(t, "trimmed_video_url", "?"))
            # mark processed only after a successful extract call
            processed.add(key)
            _save_ledger(camera, processed)
        except Exception as e:
            result["errors"] += 1
            log.error("[%s] extract failed for %s: %s", camera, key, e,
                      exc_info=True)
    return result


# -----------------------------------------------------------------------------
# main loop
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="train_extraction.run_extraction_service",
        description="Continuous train-extraction producer (raw -> trimmed).")
    p.add_argument("--camera", action="append", choices=list(D.ALL_CAMERAS),
                   help="limit to one or more cameras (default: all four)")
    p.add_argument("--once", action="store_true",
                   help="run a single sweep of all selected cameras, then exit")
    p.add_argument("--dry-run", action="store_true",
                   help="list what WOULD be extracted; upload/extract nothing")
    p.add_argument("--poll-interval", type=int,
                   default=int(os.environ.get("WAGONEYE_EXTRACTION_POLL_INTERVAL", "60")),
                   help="seconds between sweeps in continuous mode (default 60)")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("WAGONEYE_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cameras = tuple(args.camera) if args.camera else D.ALL_CAMERAS

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    log.info("train-extraction producer starting: cameras=%s once=%s dry_run=%s "
             "poll=%ss", cameras, args.once, args.dry_run, args.poll_interval)

    while not _STOP:
        for camera in cameras:
            if _STOP:
                break
            r = sweep_camera(camera, dry_run=args.dry_run)
            log.info("[%s] sweep: listed=%d new=%d trains=%d foreign=%d errors=%d",
                     camera, r["listed"], r["new"], r["trains"],
                     r.get("foreign", 0), r["errors"])
        if args.once or _STOP:
            break
        # interruptible sleep between sweeps
        for _ in range(max(1, args.poll_interval)):
            if _STOP:
                break
            time.sleep(1)

    log.info("train-extraction producer exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
