"""Stage-0 batch acquisition for continuous (`--auto`) mode.

This module is the S3-facing half of the orchestrator: it discovers source
videos in S3, clusters them into per-train batches by filename timestamp,
decides which batch is runnable, and persists a small processed-batches state
file so a restarted service never reprocesses a batch.

It owns NO detection / fusion / reporting logic -- once a `TrainBatch` is
handed back to `orchestrator.master_runner.process_batch`, the batch is
processed exactly as before.  Nothing here changes how a batch is analysed.

Call contract (consumed verbatim by master_runner.run_auto):

    poll_for_batches(s3_client, processed_batches, start_time, tolerance_sec)
        -> List[TrainBatch]
    select_runnable_batch(batches, partial_wait_minutes) -> Optional[TrainBatch]
    load_batch_state(s3_client, state_loc) -> Dict[str, str]
    save_batch_state(s3_client, state_loc, processed) -> None
    DEFAULT_BATCH_TOLERANCE_SEC : int

Configuration (all via core.constants, i.e. WAGONEYE_* env overrides):
    S3_INPUT_BUCKET        bucket holding the source videos
    S3_INPUT_PREFIXES      comma-separated key prefixes to scan (one per
                           camera rig, or a single shared prefix).  If empty,
                           polling finds nothing and --auto idles (safe
                           default: the operator must point it at real data).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from core import constants as C
from core.batch import CameraVideo, TrainBatch, parse_train_timestamp
from core.logging_setup import get_logger

log = get_logger("batch_manager")

# Two source videos of the same train pass may carry filename timestamps that
# differ by a few seconds (each camera's trimmer stamps independently).  Videos
# whose timestamps fall within this window are clustered into one batch.
DEFAULT_BATCH_TOLERANCE_SEC = 120

_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# Warn only once if the operator hasn't configured any input prefixes.
_WARNED_NO_PREFIXES = False


# -----------------------------------------------------------------------------
# processed_batches state file (JSON on S3)
# -----------------------------------------------------------------------------

def _split_state_loc(state_loc: str) -> Tuple[str, str]:
    """`"bucket/key/with/slashes.json"` -> ("bucket", "key/with/slashes.json")."""
    parts = state_loc.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid state_loc (expected 'bucket/key'): {state_loc!r}")
    return parts[0], parts[1]


def load_batch_state(s3_client, state_loc: str) -> Dict[str, str]:
    """Read the {batch_key -> final_status} map from S3.  Missing -> {}."""
    bucket, key = _split_state_loc(state_loc)
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read()
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        log.warning("[BATCH] state file %s is not a dict; starting empty", state_loc)
        return {}
    except Exception as e:
        # NoSuchKey on first run is normal; anything else we log and start fresh
        # rather than crash the service.
        name = type(e).__name__
        if name in ("NoSuchKey", "ClientError"):
            log.info("[BATCH] no existing state file at %s (%s) -- starting empty",
                     state_loc, name)
        else:
            log.warning("[BATCH] could not read state file %s: %s", state_loc, e)
        return {}


def save_batch_state(s3_client, state_loc: str, processed: Dict[str, str]) -> None:
    """Persist the {batch_key -> final_status} map back to S3 (best-effort)."""
    bucket, key = _split_state_loc(state_loc)
    try:
        body = json.dumps(processed, indent=2, sort_keys=True).encode("utf-8")
        s3_client.put_object(
            Bucket=bucket, Key=key, Body=body,
            ContentType="application/json",
        )
    except Exception as e:
        # Non-fatal: the batch was still processed; we just failed to checkpoint.
        # Worst case the batch is re-evaluated next poll and skipped as already
        # present in the in-memory `processed` dict for this process lifetime.
        log.error("[BATCH] failed to persist state file %s: %s", state_loc, e)


# -----------------------------------------------------------------------------
# S3 listing + camera / timestamp resolution
# -----------------------------------------------------------------------------

def _camera_for_key(key: str) -> Optional[str]:
    """Resolve an S3 key to a camera id.

    Delegates to `constants.camera_from_key`, which checks the camera FOLDER
    first and only then filename tokens.  Basename-only matching against the
    canonical ids used to drop both TOP cameras on the floor: the site writes
    `RIGHT_TOP`/`LEFT_TOP`, which contains neither `right_up_top` nor
    `right_up`, so those clips resolved to no camera and never joined a batch.
    """
    base = key.rsplit("/", 1)[-1].lower()
    if not base.endswith(_VIDEO_EXTS):
        return None
    return C.camera_from_key(key)


def _clean_etag(raw) -> Optional[str]:
    """S3 ETags come wrapped in quotes; normalize to a bare hex string."""
    if not raw:
        return None
    return str(raw).strip().strip('"')


def _list_input_objects(s3_client) -> List[Tuple[str, str, object, Optional[str], int]]:
    """Return [(bucket, key, last_modified, etag, size), ...] for every video
    under the configured input prefixes.  Handles pagination.

    `size` is carried so incremental gap extraction can use it as a second
    object-identity signal alongside the ETag."""
    global _WARNED_NO_PREFIXES
    prefixes = C.S3_INPUT_PREFIXES
    bucket = C.S3_INPUT_BUCKET
    if not prefixes:
        if not _WARNED_NO_PREFIXES:
            log.warning("[BATCH] WAGONEYE_S3_INPUT_PREFIXES is empty -- no source "
                        "videos will be discovered.  Set it to the S3 prefix(es) "
                        "holding the camera videos.")
            _WARNED_NO_PREFIXES = True
        return []

    out: List[Tuple[str, str, object, Optional[str], int]] = []
    for prefix in prefixes:
        token = None
        while True:
            kwargs = {"Bucket": bucket, "Prefix": prefix.strip("/")}
            if token:
                kwargs["ContinuationToken"] = token
            try:
                resp = s3_client.list_objects_v2(**kwargs)
            except Exception as e:
                log.error("[BATCH] list_objects_v2 failed (bucket=%s prefix=%s): %s",
                          bucket, prefix, e)
                break
            for item in resp.get("Contents", []):
                key = item["Key"]
                if key.lower().endswith(_VIDEO_EXTS):
                    out.append((bucket, key, item.get("LastModified"),
                                _clean_etag(item.get("ETag")),
                                int(item.get("Size") or 0)))
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
                if not token:
                    break
            else:
                break
    return out


def _discovery_cutoff():
    """`(cutoff_datetime_or_None, human_description)` for trimmed-clip discovery.

    DEFAULT: the operational-day anchor (05:00 IST), exactly as the previous
    production processor did -- so a restart at ANY hour still sees every train
    from today's operational day, and a service that was down overnight loses
    nothing when it comes back at 05:30.

    `WAGONEYE_CONSUMER_LOOKBACK_MINUTES`, when explicitly set, replaces the anchor
    with a sliding "last N minutes" window (0 = no bound at all).  That is only
    useful for narrow testing: a sliding window silently skips anything uploaded
    while the service was stopped, which is the trap the old code called out.
    """
    raw = os.getenv("WAGONEYE_CONSUMER_LOOKBACK_MINUTES")
    if raw is not None and raw != "":
        mins = consumer_lookback_minutes()
        if mins <= 0:
            return None, "no window"
        return (datetime.now(timezone.utc) - timedelta(minutes=mins),
                f"lookback {mins:.0f}min")
    from core import config as CFG
    cutoff = CFG.discovery_cutoff_utc()
    return cutoff, f"operational day from {cutoff.astimezone(CFG.IST):%Y-%m-%d %H:%M} IST"


def consumer_lookback_minutes() -> float:
    """How far back the CONSUMER considers trimmed clips, in minutes (0 = no limit).

    The trimmed bucket holds every clip the extractor has ever produced -- months
    of them.  Without a bound, a single poll opened a batch for EVERY historical
    clip (~17,600 on the first production run) and would have tried to inspect the
    entire archive: weeks of CPU, a full disk, and thousands of emails and
    dashboard posts.

    Default 60 minutes -- deliberately wider than the extraction window (10 min),
    because a trimmed clip appears only AFTER its raw footage is cut, and the four
    cameras arrive minutes apart.  It still must comfortably exceed
    FINAL_CAMERA_WAIT_MINUTES so a batch's late cameras are still discoverable.

    Set 0 to consider everything (the old behaviour) -- only safe when the
    terminal-batch set in processed_batches.json genuinely covers the archive.
    """
    raw = os.getenv("WAGONEYE_CONSUMER_LOOKBACK_MINUTES")
    if raw is None or raw == "":
        return 60.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 60.0


#: A clip the extractor could not complete is named `..._train_incomplete.mp4`.
#: It shares a train timestamp with the real `..._train.mp4`, so both classify to
#: the same (camera, timestamp) and each poll overwrote the other's ETag --
#: logging `ETag changed A -> B` then `B -> A` forever and re-triggering a camera
#: rebuild every tick.  The complete clip always wins.
_INCOMPLETE_MARKER = "_train_incomplete"


def _is_incomplete(key: str) -> bool:
    return _INCOMPLETE_MARKER in key.rsplit("/", 1)[-1].lower()


def list_candidate_videos(s3_client) -> List[CameraVideo]:
    """Classify every discoverable input video into a CameraVideo (camera id +
    train timestamp + ETag).  No clustering -- the manifest scheduler attaches
    each candidate to an active batch (or creates one).  Unclassifiable objects
    are dropped.

    Two filters keep this bounded and stable:
      * a recency window (`consumer_lookback_minutes`), so the whole archive is
        not re-queued on every poll;
      * one candidate per (camera, train timestamp) -- a complete `_train.mp4`
        beats an `_train_incomplete.mp4`, and otherwise the most recently modified
        object wins.  Without this, two objects for the same slot thrash each
        other's ETag forever.
    """
    cutoff, window_desc = _discovery_cutoff()

    best: Dict[tuple, CameraVideo] = {}
    stale = 0
    for bucket, key, last_modified, etag, size in _list_input_objects(s3_client):
        cam = _camera_for_key(key)
        ts = parse_train_timestamp(key)
        if not cam or not ts:
            continue
        if cutoff is not None:
            lm = last_modified
            if lm is not None:
                if getattr(lm, "tzinfo", None) is None:
                    lm = lm.replace(tzinfo=timezone.utc)
                if lm < cutoff:
                    stale += 1
                    continue
        cv = CameraVideo(
            camera_id=cam, bucket=bucket, s3_key=key,
            filename=key.rsplit("/", 1)[-1],
            s3_url=_https_url(bucket, key),
            train_timestamp=ts, last_modified=last_modified, etag=etag,
            file_size=size,
        )
        slot = (cam, ts)
        prev = best.get(slot)
        if prev is None or _prefer(cv, prev):
            best[slot] = cv

    if stale:
        _log_discovery_skip_once(window_desc, stale)
    out = list(best.values())
    # deterministic order: timestamp, camera, key
    out.sort(key=lambda cv: (cv.train_timestamp, cv.camera_id, cv.s3_key))
    return out


#: Last (window, count) logged, so an idle poll stays silent.  See
#: `train_extraction.run_extraction_service._log_skip_once` for the rationale.
_LAST_DISCOVERY_SKIP: list = [None]


def _log_discovery_skip_once(window_desc: str, stale: int) -> None:
    """Log the skip summary only when it CHANGES.

    An idle poll skipped the same 4298 clips every 60s and said so each time.
    The count moves as soon as a trimmed clip lands or the day rolls, so a real
    event is never suppressed.  Gates the MESSAGE only, not the filtering.
    """
    key = (window_desc, stale)
    if _LAST_DISCOVERY_SKIP[0] == key:
        return
    _LAST_DISCOVERY_SKIP[0] = key
    log.info("[DISCOVERY] %s: skipped %d trimmed clip(s) older than the window",
             window_desc, stale)


def _prefer(new: CameraVideo, old: CameraVideo) -> bool:
    """Should `new` replace `old` for the same (camera, train timestamp)?"""
    new_inc, old_inc = _is_incomplete(new.s3_key), _is_incomplete(old.s3_key)
    if new_inc != old_inc:
        return old_inc            # a COMPLETE clip always beats an incomplete one
    nl, ol = new.last_modified, old.last_modified
    if nl is not None and ol is not None:
        if getattr(nl, "tzinfo", None) is None:
            nl = nl.replace(tzinfo=timezone.utc)
        if getattr(ol, "tzinfo", None) is None:
            ol = ol.replace(tzinfo=timezone.utc)
        if nl != ol:
            return nl > ol        # newest upload wins
    return new.s3_key > old.s3_key    # last resort: deterministic, not arbitrary


#: The site's zone.  Filename timestamps are IST wall-clock, not UTC -- the
#: producer writes them that way (`train_extraction/time_utils.
#: parse_timestamp_from_filename`).
_IST = timezone(timedelta(hours=5, minutes=30))


def _ts_to_dt(ts: str) -> Optional[datetime]:
    """`YYYYMMDD_HHMMSS` -> tz-aware datetime in IST.

    This function is used ONLY for relative comparisons -- the greedy clustering
    gate (`abs(dt - anchor) <= tolerance_sec`) and the candidate sort -- so the
    label it carried made no behavioural difference: every consumer subtracts two
    of these from each other, and a constant offset cancels.  It said UTC anyway,
    which contradicted the producer and set a trap for the next reader who
    compared one of these against a real instant.  `TrainBatch.age_seconds()` had
    exactly that bug and held partial batches back ~5.4 h.  Aligned here so there
    is one answer to "what timezone is a filename timestamp" in this package.
    """
    try:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S").replace(tzinfo=_IST)
    except ValueError:
        return None


# -----------------------------------------------------------------------------
# Batch discovery
# -----------------------------------------------------------------------------

def poll_for_batches(
    *,
    s3_client,
    processed_batches: Dict[str, str],
    start_time: Optional[datetime] = None,
    tolerance_sec: int = DEFAULT_BATCH_TOLERANCE_SEC,
    apply_cutoff: bool = True,
) -> List[TrainBatch]:
    """Discover candidate TrainBatches from the S3 input prefixes.

    Videos are grouped into batches by filename timestamp: each new video
    either joins an existing open cluster (if its timestamp is within
    `tolerance_sec` of that cluster's anchor and that camera slot is free) or
    starts a new cluster.  Batches already present in `processed_batches`
    (the persisted state map) are excluded.

    `start_time` is accepted for API compatibility and used only for logging;
    de-duplication is done via `processed_batches`, which lets `--batch <key>`
    replay an older batch and lets a restarted service resume cleanly.

    `apply_cutoff` bounds discovery to `_discovery_cutoff()` -- the same
    operational-day / lookback window `list_candidate_videos` applies.  It
    defaults to True because without it this function returned a batch for EVERY
    clip in the trimmed bucket: that bucket holds every clip the extractor has
    ever produced, so a first production poll queued the entire archive (the
    ~17,600-batch flood `consumer_lookback_minutes` documents) and would have
    inspected months of history train by train.  Pass False to deliberately
    reach past the window -- `run_auto` does exactly that for `--batch <key>`,
    so replaying a specific older batch still works.
    """
    objects = _list_input_objects(s3_client)
    if not objects:
        return []

    cutoff, window_desc = _discovery_cutoff() if apply_cutoff else (None, "no window")
    stale = 0

    # Build (camera, timestamp) candidates, dropping anything we can't classify.
    #
    # `_list_input_objects` yields 5-tuples (bucket, key, last_modified, etag,
    # size).  This loop used to unpack FOUR, so `poll_for_batches` raised
    # `ValueError: too many values to unpack (expected 4)` on the first
    # non-empty listing -- i.e. every real poll.  `size` is not needed here, so
    # it is bound and ignored.
    candidates = []
    for bucket, key, last_modified, etag, _size in objects:
        cam = _camera_for_key(key)
        ts = parse_train_timestamp(key)
        if not cam or not ts:
            continue
        dt = _ts_to_dt(ts)
        if dt is None:
            continue
        if cutoff is not None and last_modified is not None:
            lm = last_modified
            if getattr(lm, "tzinfo", None) is None:
                lm = lm.replace(tzinfo=timezone.utc)
            if lm < cutoff:
                stale += 1
                continue
        candidates.append((dt, ts, cam, bucket, key, last_modified, etag))

    if stale:
        _log_discovery_skip_once(window_desc, stale)

    # Deterministic order: by timestamp, then camera, then key.
    candidates.sort(key=lambda c: (c[0], c[2], c[4]))

    # Greedy temporal clustering.
    clusters: List[Dict] = []
    for dt, ts, cam, bucket, key, last_modified, etag in candidates:
        placed = False
        for cl in clusters:
            if cam in cl["videos"]:
                continue  # camera slot already filled for this cluster
            if abs((dt - cl["anchor"]).total_seconds()) <= tolerance_sec:
                cl["videos"][cam] = CameraVideo(
                    camera_id=cam, bucket=bucket, s3_key=key,
                    filename=key.rsplit("/", 1)[-1],
                    s3_url=_https_url(bucket, key),
                    train_timestamp=cl["batch_key"],
                    last_modified=last_modified,
                )
                placed = True
                break
        if not placed:
            clusters.append({
                "anchor": dt,
                "batch_key": ts,
                "videos": {cam: CameraVideo(
                    camera_id=cam, bucket=bucket, s3_key=key,
                    filename=key.rsplit("/", 1)[-1],
                    s3_url=_https_url(bucket, key),
                    train_timestamp=ts,
                    last_modified=last_modified,
                )},
            })

    batches: List[TrainBatch] = []
    for cl in clusters:
        if cl["batch_key"] in processed_batches:
            continue
        batches.append(TrainBatch(
            batch_key=cl["batch_key"],
            train_timestamp=cl["batch_key"],
            videos=cl["videos"],
        ))

    if batches:
        log.info("[BATCH] discovered %d unprocessed batch(es): %s",
                 len(batches), [b.batch_key for b in batches])
    return batches


def _https_url(bucket: str, key: str) -> str:
    return f"https://{bucket}.s3.{C.S3_REGION}.amazonaws.com/{key}"


# -----------------------------------------------------------------------------
# Batch selection
# -----------------------------------------------------------------------------

def select_runnable_batch(
    batches: List[TrainBatch],
    partial_wait_minutes: float = 30.0,
) -> Optional[TrainBatch]:
    """Pick the batch to run next.

    Priority:
        1. The OLDEST complete batch (all 4 cameras present) -- run immediately.
        2. Otherwise the oldest PARTIAL batch, but only once it has aged past
           `partial_wait_minutes` (giving stragglers time to upload).  A
           younger partial batch is held back (returns None) so we don't run a
           3-camera batch that would have been complete 30 s later.
    """
    if not batches:
        return None

    # Oldest first == earliest train_timestamp.
    ordered = sorted(batches, key=lambda b: b.train_timestamp)

    complete = [b for b in ordered if b.is_complete()]
    if complete:
        return complete[0]

    wait_sec = partial_wait_minutes * 60.0
    for b in ordered:
        if b.age_seconds() >= wait_sec:
            log.info("[BATCH] %s partial (cameras=%s), aged %.0fs >= %.0fs wait "
                     "-- running partial", b.batch_key, b.present_cameras(),
                     b.age_seconds(), wait_sec)
            return b
    return None
