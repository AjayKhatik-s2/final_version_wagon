"""Historical (time-range) execution mode -- an INPUT-SELECTION layer ONLY.

    CLI -> resolve_window -> discover_batches -> master_runner.process_batch

Everything after batch construction is the EXISTING pipeline, called through the
exact same entry point (`master_runner.process_batch`) that `--auto`, `--once`
and `--local-only` use.  There is no second Stage 1/2/3/4/5 here: this module
selects which historical S3 objects belong to the requested window, groups them
into per-train batches, and hands each one over unchanged.

Isolation from the live pipeline
--------------------------------
* `run_auto`'s polling loop and `train_batch_manager`'s live discovery
  (`list_candidate_videos`, `poll_for_batches`, `select_runnable_batch`,
  `_discovery_cutoff`) are never entered.  Historical mode calls `process_batch`
  directly.  (This package has no lifecycle runner or active-batch scheduler --
  those live in the sibling `global_train` repo -- so there is no scheduler state
  machine here for historical mode to disturb.)
* `processed_batches.json` on S3 is never read or written, so a historical run
  can neither mark a live batch terminal nor be blocked by one.
* Output goes under `<workspace_root>/historical/<batch_key>/`, so a historical
  re-run of a train that already ran live cannot overwrite the live batch tree.
* Delivery (S3 upload + email) is OFF unless `--historical-deliver` is passed --
  reprocessing a week of history must not re-email the operators or overwrite
  the delivered artifacts of the live run.

  Note precisely what `--historical-deliver` DOES overwrite when you pass it.
  The LOCAL tree is isolated, but the delivered S3 objects are keyed by the
  TRAIN's own timestamp, not by the batch root:
  `<camera_folder>/<YYYY-MM-DD_HH-MM-SS>/inspection_data.json` and the
  `train_batch/<batch_key>/` report tree.  So delivering a historical re-run of a
  train that already ran live REPLACES that train's dashboard document and
  re-POSTs it to the ingest receivers.  For a genuine reprocess that is usually
  the point -- the corrected report should supersede the old one -- but it is not
  an isolated operation, so use `--dry-run` first and do not pass
  `--historical-deliver` for a bulk window unless you intend every train in it
  to be re-delivered.
* No environment variable is read that `--auto` does not already read, and none
  is written.  `WAGONEYE_PIPELINE_SOURCE` is irrelevant here: historical mode is
  always a pure consumer of already-trimmed clips.

Timestamp semantics (NOT guessed -- taken from the code that writes the names)
-----------------------------------------------------------------------------
A trimmed clip is named `<raw basename>_train.mp4`, and the raw basename carries
`..._YYYYMMDD_HHMMSS`.  `train_extraction/time_utils.parse_timestamp_from_filename`
states those digits are **IST wall-clock** and attaches the tzinfo without
shifting; `train_extraction/extractor._emit_segments` derives the trimmed name
from that raw basename.  So a clip's filename timestamp is the START OF THE RAW
CLIP in IST -- NOT the moment the train passed.  The train passed somewhere in

    [filename_ts, filename_ts + clip_span]

because the extractor cuts the pass out of a raw clip (300 s in production) and,
for a train that spans several raw clips, names the result after the FIRST one.
Selection therefore treats each clip as covering `[T, T + lookahead]` and keeps
it when that overlaps the requested window (`--pad-minutes`, default 15).
`--dry-run` exists precisely so this can be eyeballed before any inference runs.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import config as CFG
from core import constants as C
from core.batch import CameraVideo, TrainBatch, parse_train_timestamp
from core.logging_setup import get_logger

from . import train_batch_manager as TBM

log = get_logger("historical")

#: Default zone for `--date/--start-time/--end-time` when `--timezone` is absent.
#: The site runs on IST and every filename timestamp is IST wall-clock.
DEFAULT_TIMEZONE = "Asia/Kolkata"

#: How far past its filename timestamp a clip may still hold its train, in
#: minutes.  A raw clip is 300 s; an "ongoing train" merged from several raw
#: clips keeps the FIRST clip's name, so 15 min covers a 3-clip merge with room
#: to spare.  Widen it for a site with longer raw clips.
DEFAULT_PAD_MINUTES = 15.0

#: Subdirectory of the workspace that holds every historical batch, keeping the
#: live `batch_outputs/<key>/` tree untouched.
HISTORICAL_SUBDIR = "historical"

MANIFEST_NAME = "historical_manifest.json"


# -----------------------------------------------------------------------------
# Window resolution
# -----------------------------------------------------------------------------

@dataclass
class HistoricalWindow:
    start: datetime            # tz-aware
    end: datetime              # tz-aware
    tz_name: str
    tz: Any
    rolled_overnight: bool = False

    def describe(self) -> str:
        return (f"{self.start.strftime('%Y-%m-%d %H:%M:%S')} -> "
                f"{self.end.strftime('%Y-%m-%d %H:%M:%S')} {self.tz_name}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_local": self.start.isoformat(),
            "end_local": self.end.isoformat(),
            "start_utc": self.start.astimezone(_UTC).isoformat(),
            "end_utc": self.end.astimezone(_UTC).isoformat(),
            "timezone": self.tz_name,
            "rolled_overnight": self.rolled_overnight,
        }


_UTC = timezone.utc


def resolve_timezone(name: Optional[str]):
    """Return a tzinfo for `name`.

    Uses stdlib `zoneinfo` when the platform has tzdata.  Falls back to the
    fixed +05:30 offset the pipeline already hardcodes (`core.config.IST`) for
    the site's own zone, so a minimal container without tzdata still works.
    """
    name = (name or DEFAULT_TIMEZONE).strip()
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name), name
    except Exception:
        if name.lower() in ("asia/kolkata", "asia/calcutta", "ist", "+05:30"):
            return CFG.IST, "Asia/Kolkata"
        raise ValueError(
            f"unknown timezone {name!r} and no tzdata available; install the "
            f"`tzdata` package or pass --timezone Asia/Kolkata")


def _parse_hhmm(value: str, field_name: str) -> Tuple[int, int, int]:
    parts = str(value).strip().split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        raise ValueError(f"{field_name} must be HH:MM or HH:MM:SS (got {value!r})")
    h, m = int(parts[0]), int(parts[1])
    s = int(parts[2]) if len(parts) == 3 else 0
    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
        raise ValueError(f"{field_name} out of range (got {value!r})")
    return h, m, s


def resolve_window(
    *,
    date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    timezone_name: Optional[str] = None,
    start_iso: Optional[str] = None,
    end_iso: Optional[str] = None,
) -> HistoricalWindow:
    """Build the requested window from either form of CLI input.

    Form A: --date YYYY-MM-DD --start-time HH:MM --end-time HH:MM [--timezone Z]
    Form B: --start <ISO8601> --end <ISO8601>   (offset in the string wins)

    An end that is not after the start is rolled to the NEXT DAY in form A (a
    22:00 -> 02:00 night window), and reported so the interpretation is never
    silent.  In form B it is an error -- an explicit ISO timestamp means what it
    says.
    """
    tz, tz_name = resolve_timezone(timezone_name)

    if start_iso or end_iso:
        if not (start_iso and end_iso):
            raise ValueError("--start and --end must be given together")
        if date or start_time or end_time:
            raise ValueError(
                "use EITHER --start/--end (ISO) OR --date/--start-time/--end-time")
        try:
            start = datetime.fromisoformat(str(start_iso).strip())
            end = datetime.fromisoformat(str(end_iso).strip())
        except ValueError as e:
            raise ValueError(f"could not parse ISO timestamp: {e}") from e
        if start.tzinfo is None:
            start = start.replace(tzinfo=tz)
        if end.tzinfo is None:
            end = end.replace(tzinfo=tz)
        if end <= start:
            raise ValueError(f"--end ({end.isoformat()}) must be after "
                             f"--start ({start.isoformat()})")
        return HistoricalWindow(start=start, end=end, tz_name=tz_name, tz=tz)

    missing = [n for n, v in (("--date", date), ("--start-time", start_time),
                              ("--end-time", end_time)) if not v]
    if missing:
        raise ValueError(
            f"historical mode needs {', '.join(missing)} "
            f"(or --start/--end as ISO timestamps)")

    try:
        day = datetime.strptime(str(date).strip(), "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"--date must be YYYY-MM-DD (got {date!r})") from e

    sh, sm, ss = _parse_hhmm(start_time, "--start-time")
    eh, em, es = _parse_hhmm(end_time, "--end-time")
    start = datetime(day.year, day.month, day.day, sh, sm, ss, tzinfo=tz)
    end = datetime(day.year, day.month, day.day, eh, em, es, tzinfo=tz)
    rolled = False
    if end <= start:
        end = end + timedelta(days=1)
        rolled = True
    return HistoricalWindow(start=start, end=end, tz_name=tz_name, tz=tz,
                            rolled_overnight=rolled)


# -----------------------------------------------------------------------------
# Object selection
# -----------------------------------------------------------------------------

def filename_timestamp_local(ts: str) -> Optional[datetime]:
    """`YYYYMMDD_HHMMSS` -> tz-aware datetime in IST.

    IST -- not UTC -- because that is what the producer writes; see the module
    docstring and `train_extraction/time_utils.parse_timestamp_from_filename`,
    whose agreement with this function is asserted by the test suite.
    """
    try:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S").replace(tzinfo=CFG.IST)
    except (TypeError, ValueError):
        return None


@dataclass
class SelectedObject:
    camera_id: str
    bucket: str
    key: str
    train_timestamp: str
    clip_start_local: datetime
    covers_until_local: datetime
    last_modified: Optional[datetime]
    etag: Optional[str]
    size: int
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera": self.camera_id,
            "s3_uri": f"s3://{self.bucket}/{self.key}",
            "bucket": self.bucket,
            "key": self.key,
            "train_timestamp": self.train_timestamp,
            "clip_start_ist": self.clip_start_local.isoformat(),
            "covers_until_ist": self.covers_until_local.isoformat(),
            "last_modified_utc": (self.last_modified.isoformat()
                                  if self.last_modified else None),
            "etag": self.etag,
            "size_bytes": self.size,
            "selected_because": self.reason,
        }


@dataclass
class DiscoveryResult:
    window: HistoricalWindow
    pad_minutes: float
    bucket: str
    prefixes: List[str]
    listed: int = 0
    classified: int = 0
    selected: List[SelectedObject] = field(default_factory=list)
    batches: List[TrainBatch] = field(default_factory=list)
    duplicates_dropped: int = 0
    tolerance_sec: int = TBM.DEFAULT_BATCH_TOLERANCE_SEC


def select_objects(
    *, s3_client, window: HistoricalWindow, pad_minutes: float = DEFAULT_PAD_MINUTES,
    tolerance_sec: int = TBM.DEFAULT_BATCH_TOLERANCE_SEC,
) -> DiscoveryResult:
    """List the configured input prefixes and keep the clips whose coverage
    window overlaps the requested range.

    Reuses `train_batch_manager._list_input_objects` verbatim, so the bucket,
    the prefixes, the pagination and the video-extension filter are exactly the
    ones `--auto` uses.  The operational-day / lookback cutoff that
    `list_candidate_videos` applies is deliberately NOT used here -- that cutoff
    exists to stop the live poller re-queueing the archive, and skipping the
    archive is the one thing historical mode must not do.
    """
    res = DiscoveryResult(
        window=window, pad_minutes=pad_minutes,
        bucket=C.S3_INPUT_BUCKET, prefixes=list(C.S3_INPUT_PREFIXES),
    )
    pad = timedelta(minutes=max(0.0, pad_minutes))

    best: Dict[Tuple[str, str], SelectedObject] = {}
    best_cv: Dict[Tuple[str, str], CameraVideo] = {}

    for bucket, key, last_modified, etag, size in TBM._list_input_objects(s3_client):
        res.listed += 1
        cam = TBM._camera_for_key(key)
        ts = parse_train_timestamp(key)
        if not cam or not ts:
            continue
        clip_start = filename_timestamp_local(ts)
        if clip_start is None:
            continue
        res.classified += 1

        covers_until = clip_start + pad
        # Overlap test: [clip_start, clip_start+pad] ∩ [window.start, window.end]
        if clip_start > window.end or covers_until < window.start:
            continue

        if window.start <= clip_start <= window.end:
            reason = "clip starts inside the requested window"
        else:
            reason = (f"clip starts {(window.start - clip_start).total_seconds() / 60.0:.1f} "
                      f"min before the window but can still hold a train inside it "
                      f"(pad {pad_minutes:g} min)")

        cv = CameraVideo(
            camera_id=cam, bucket=bucket, s3_key=key,
            filename=key.rsplit("/", 1)[-1],
            s3_url=TBM._https_url(bucket, key),
            train_timestamp=ts, last_modified=last_modified, etag=etag,
            file_size=int(size or 0),
        )
        sel = SelectedObject(
            camera_id=cam, bucket=bucket, key=key, train_timestamp=ts,
            clip_start_local=clip_start, covers_until_local=covers_until,
            last_modified=last_modified, etag=etag, size=int(size or 0),
            reason=reason,
        )
        slot = (cam, ts)
        prev = best_cv.get(slot)
        if prev is None:
            best_cv[slot], best[slot] = cv, sel
        else:
            res.duplicates_dropped += 1
            # Same dedup rule as the live path: a complete clip beats an
            # `_train_incomplete` one, else newest upload wins.
            if TBM._prefer(cv, prev):
                best_cv[slot], best[slot] = cv, sel

    ordered = sorted(best_cv.values(),
                     key=lambda cv: (cv.train_timestamp, cv.camera_id, cv.s3_key))
    res.selected = [best[(cv.camera_id, cv.train_timestamp)] for cv in ordered]
    res.tolerance_sec = int(tolerance_sec)
    res.batches = cluster_into_batches(ordered, tolerance_sec=int(tolerance_sec))
    return res


def cluster_into_batches(
    videos: Sequence[CameraVideo],
    *,
    tolerance_sec: int = TBM.DEFAULT_BATCH_TOLERANCE_SEC,
) -> List[TrainBatch]:
    """Group per-camera clips into one TrainBatch per train pass.

    Same rule the live path uses (`train_batch_manager.poll_for_batches` /
    `master_runner._attach_candidate`): greedy temporal clustering with the
    shared `DEFAULT_BATCH_TOLERANCE_SEC`, one slot per camera per cluster, and
    the earliest timestamp in a cluster becomes its batch key.  Two trains in
    the window therefore stay two batches -- they are never merged into one
    Global Train.
    """
    clusters: List[Dict[str, Any]] = []
    for cv in sorted(videos, key=lambda v: (v.train_timestamp, v.camera_id, v.s3_key)):
        dt = filename_timestamp_local(cv.train_timestamp)
        if dt is None:
            continue
        placed = False
        for cl in clusters:
            if cv.camera_id in cl["videos"]:
                continue
            if abs((dt - cl["anchor"]).total_seconds()) <= tolerance_sec:
                cl["videos"][cv.camera_id] = cv
                # Chain from the LATEST member, not the first: the four cameras
                # are stamped from their own raw clips and can arrive in steps.
                # Anchoring on the first member caps a train's total span at
                # `tolerance_sec`, which split real 4-camera trains whose clips
                # spanned ~5 min (2026-08-01).  Successive gaps are still each
                # bounded by `tolerance_sec`.
                cl["anchor"] = max(cl["anchor"], dt)
                placed = True
                break
        if not placed:
            clusters.append({"anchor": dt, "batch_key": cv.train_timestamp,
                             "videos": {cv.camera_id: cv}})

    out: List[TrainBatch] = []
    for cl in sorted(clusters, key=lambda c: c["batch_key"]):
        out.append(TrainBatch(batch_key=cl["batch_key"],
                              train_timestamp=cl["batch_key"],
                              videos=dict(cl["videos"])))
    return out


# -----------------------------------------------------------------------------
# Manifest
# -----------------------------------------------------------------------------

def build_manifest(
    res: DiscoveryResult, *, workspace_root: str, dry_run: bool,
) -> Dict[str, Any]:
    batches = []
    for i, b in enumerate(res.batches, start=1):
        by_cam = {}
        for cam in C.ALL_CAMERAS:
            cv = b.videos.get(cam)
            if cv is None:
                continue
            sel = next((s for s in res.selected
                        if s.camera_id == cam and s.key == cv.s3_key), None)
            by_cam[cam] = sel.to_dict() if sel else {
                "camera": cam, "s3_uri": f"s3://{cv.bucket}/{cv.s3_key}"}
        batches.append({
            "index": i,
            "batch_key": b.batch_key,
            "train_timestamp": b.train_timestamp,
            "train_time_ist": (filename_timestamp_local(b.train_timestamp) or "").__str__(),
            "cameras": by_cam,
            "present_cameras": b.present_cameras(),
            "missing_cameras": b.missing_cameras(),
            "batch_root": os.path.join(workspace_root, b.batch_key),
            "staged_inputs": os.path.join(workspace_root, b.batch_key,
                                          CFG.DIR_DOWNLOADS),
        })
    return {
        "mode": "historical",
        "dry_run": bool(dry_run),
        "generated_at": datetime.now(_UTC).isoformat(),
        "requested_window": res.window.to_dict(),
        "pad_minutes": res.pad_minutes,
        "clustering": {
            "tolerance_sec": res.tolerance_sec,
            "live_default_sec": TBM.DEFAULT_BATCH_TOLERANCE_SEC,
        },
        "search": {
            "bucket": res.bucket,
            "prefixes": res.prefixes,
            "objects_listed": res.listed,
            "objects_classified": res.classified,
            "objects_selected": len(res.selected),
            "duplicates_dropped": res.duplicates_dropped,
        },
        "batches_discovered": len(res.batches),
        "batches": batches,
        "workspace_root": workspace_root,
    }


def log_manifest(res: DiscoveryResult, manifest: Dict[str, Any]) -> None:
    """Print the operator-facing manifest before anything is downloaded."""
    w = res.window
    log.info("[HISTORICAL] requested window: %s", w.describe())
    if w.rolled_overnight:
        log.info("[HISTORICAL] end-time was not after start-time -- interpreted "
                 "as an overnight window ending the NEXT day")
    log.info("[HISTORICAL] searching bucket=%s prefixes=%s",
             res.bucket, res.prefixes or "<none configured>")
    log.info("[HISTORICAL] listed %d object(s), %d classified, %d selected "
             "(clip-coverage pad %g min)",
             res.listed, res.classified, len(res.selected), res.pad_minutes)
    n_complete = sum(1 for b in res.batches if b.is_complete())
    log.info("[HISTORICAL] batches discovered: %d  (%d complete, %d partial; "
             "clustering tolerance %ds)",
             len(res.batches), n_complete, len(res.batches) - n_complete,
             res.tolerance_sec)
    if res.batches and n_complete * 2 < len(res.batches):
        log.warning("[HISTORICAL] most batches are PARTIAL.  On some days the "
                    "four cameras' clips for ONE train are stamped minutes "
                    "apart, so a %ds tolerance splits them.  Re-run --dry-run "
                    "with a larger --tolerance-sec (e.g. 300) and compare.",
                    res.tolerance_sec)

    for entry in manifest["batches"]:
        log.info("[HISTORICAL] --- batch %d/%d  %s  (%s IST) ---",
                 entry["index"], len(manifest["batches"]), entry["batch_key"],
                 entry["train_time_ist"])
        for cam in C.ALL_CAMERAS:
            info = entry["cameras"].get(cam)
            if info is None:
                log.info("[HISTORICAL] %-13s MISSING -- no clip in this window",
                         cam + ":")
                continue
            log.info("[HISTORICAL] %-13s %s", cam + ":", info["s3_uri"])
            log.info("[HISTORICAL] %-13s   ts=%s size=%s bytes  %s",
                     "", info.get("train_timestamp"), info.get("size_bytes"),
                     info.get("selected_because"))
        if entry["missing_cameras"]:
            log.warning("[HISTORICAL] batch %s is PARTIAL -- missing %s "
                        "(processed with the existing partial-camera behaviour; "
                        "no substitute video is used)",
                        entry["batch_key"], entry["missing_cameras"])


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def _write_manifest(manifest: Dict[str, Any], path: str) -> Optional[str]:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        log.info("[HISTORICAL] manifest written: %s", path)
        return path
    except OSError as e:
        log.error("[HISTORICAL] could not write manifest %s: %s", path, e)
        return None


def _no_match_message(res: DiscoveryResult) -> str:
    return (
        f"no video matched the requested window.\n"
        f"  window    : {res.window.describe()}\n"
        f"  bucket    : {res.bucket or '<unset>'}\n"
        f"  prefixes  : {res.prefixes or '<none configured -- set '
                         'WAGONEYE_S3_INPUT_PREFIXES>'}\n"
        f"  listed    : {res.listed} object(s), {res.classified} classified to a "
        f"camera + timestamp\n"
        f"  pad       : {res.pad_minutes:g} min of clip coverage past each "
        f"filename timestamp\n"
        f"  note      : filename timestamps are IST wall-clock; widen "
        f"--pad-minutes if the train started well after its raw clip did."
    )


def stage_clips(batch: TrainBatch, batch_root: str, s3_client,
                *, verbose: bool = True) -> Dict[str, str]:
    """Download this batch's clips into `<batch_root>/downloads/`.

    The batch path gets this for free -- `process_batch` downloads inline -- but
    the sequential path needs real local files before a camera can run, so
    staging is explicit here.  Mirrors `process_batch`'s naming
    (`<CAMERA>_<filename>`) so the two trees are interchangeable when debugging.
    """
    dl = os.path.join(batch_root, CFG.DIR_DOWNLOADS)
    os.makedirs(dl, exist_ok=True)
    out: Dict[str, str] = {}
    for cam in C.ALL_CAMERAS:
        cv = batch.videos.get(cam)
        if cv is None:
            continue
        dest = os.path.join(dl, f"{cam}_{cv.filename}")
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            out[cam] = dest          # already staged: a resumed run re-uses it
            continue
        try:
            s3_client.download_file(cv.bucket, cv.s3_key, dest)
            out[cam] = dest
            if verbose:
                log.info("[HISTORICAL] staged %s -> %s", cam, dest)
        except Exception as e:  # noqa: BLE001 - one camera must not stop the rest
            log.error("[HISTORICAL] could not stage %s (%s): %s",
                      cam, cv.s3_key, e)
    return out


def process_batch_sequential(
    batch: TrainBatch,
    *,
    hist_root: str,
    s3_client,
    recon_models_dir: str,
    feat_models_dir: str,
    feature_config=None,
    deliver: bool = False,
    deliver_per_camera: bool = False,
    send_email: bool = False,
    verbose: bool = True,
):
    """Run ONE historical batch through the SEQUENTIAL architecture.

    Each camera is processed and sealed independently -- and, with
    `deliver_per_camera`, publishes to the dashboard the moment it seals -- then
    global assembly fuses the persisted evidence and delivers the canonical
    documents.

    This calls exactly the same two entry points `run_sequential` calls, in the
    same order and with the same arguments, so historical and foreground
    sequential runs cannot diverge:

        orchestrator.camera_runner.run_camera(...)   per camera
        orchestrator.global_assembler.assemble(...)  once

    The only thing added is staging the clips out of S3 first, because a
    historical window's inputs are not already on disk.
    """
    from orchestrator import camera_runner, global_assembler

    batch_root = os.path.join(hist_root, batch.batch_key)
    evidence_root = os.path.join(batch_root, "camera_evidence")
    os.makedirs(evidence_root, exist_ok=True)

    videos = stage_clips(batch, batch_root, s3_client, verbose=verbose)
    if not videos:
        log.error("[HISTORICAL] %s: no clip could be staged -- skipping",
                  batch.batch_key)
        return None

    from core.feature_config import FeatureConfig
    enabled = (feature_config or FeatureConfig.all_on()).enabled_keys()

    results = []
    for cam in C.ALL_CAMERAS:
        vp = videos.get(cam)
        if not vp:
            log.info("[HISTORICAL/%s] no clip staged -- camera skipped", cam)
            continue
        log.info("[HISTORICAL/%s] --- ARRIVAL (sequential) ---", cam)
        results.append(camera_runner.run_camera(
            camera_id=cam, video_path=vp,
            recon_models_dir=recon_models_dir,
            feat_models_dir=feat_models_dir,
            evidence_root=evidence_root,
            enabled_features=enabled,
            deliver_per_camera=deliver_per_camera,
            # The train's key, so a per-camera post lands on the object
            # assembly later replaces (delivery.camera_inspection._train_ts).
            # Without it the provisional document keys off this camera's own
            # clip stamp, which differs from the batch anchor for every camera
            # but the one that set it.
            train_id=batch.batch_key,
            s3_client=s3_client,
            verbose=verbose))

    for r in results:
        line = f"[HISTORICAL/{r.camera_id}] {r.state} segments={r.local_segments}"
        pci = getattr(r, "per_camera_ingest", None)
        if pci is not None:
            line += f"  {pci.render()}"
        log.info("%s", line)

    log.info("[HISTORICAL] --- GLOBAL ASSEMBLY %s ---", batch.batch_key)
    asm = global_assembler.assemble(
        evidence_root=evidence_root, output_root=hist_root,
        batch_key=batch.batch_key, feat_models_dir=feat_models_dir,
        deliver=deliver, send_email=send_email, s3_client=s3_client,
        # Assembly only sees the STAGED local copies, so the S3 objects they
        # came from have to be handed over for `trimmed_video_url` /
        # `raw_video_urls` to be populated. Same map `process_batch` passes to
        # its Stage-5 call.
        source_video_urls={cam: cv.s3_url
                           for cam, cv in batch.videos.items()
                           if getattr(cv, "s3_url", "")},
        enabled_features=enabled,
        verbose=verbose)
    return asm


def run(
    *,
    s3_client,
    window: HistoricalWindow,
    workspace_root: str,
    recon_models_dir: str,
    feat_models_dir: str,
    feature_config=None,
    pad_minutes: float = DEFAULT_PAD_MINUTES,
    tolerance_sec: int = TBM.DEFAULT_BATCH_TOLERANCE_SEC,
    dry_run: bool = False,
    keep_inputs: bool = False,
    deliver: bool = False,
    send_email: bool = True,
    manifest_out: Optional[str] = None,
    inference_opts: Optional[Dict[str, Any]] = None,
    mode: str = "batch",
    deliver_per_camera: bool = False,
    verbose: bool = True,
) -> int:
    """Discover, stage and process every train batch in `window`.

    Returns a process exit code: 0 all good, 2 nothing to do / bad input,
    3 at least one batch failed.
    """
    hist_root = os.path.join(workspace_root, HISTORICAL_SUBDIR)

    res = select_objects(s3_client=s3_client, window=window,
                         pad_minutes=pad_minutes, tolerance_sec=tolerance_sec)
    manifest = build_manifest(res, workspace_root=hist_root, dry_run=dry_run)
    log_manifest(res, manifest)

    out_path = manifest_out or os.path.join(hist_root, MANIFEST_NAME)
    _write_manifest(manifest, out_path)

    if not res.batches:
        log.error("[HISTORICAL] %s", _no_match_message(res))
        return 2

    if dry_run:
        log.info("[HISTORICAL] --dry-run: %d batch(es) would be processed; "
                 "nothing downloaded, no inference run", len(res.batches))
        return 0

    if not deliver:
        log.info("[HISTORICAL] delivery DISABLED (no S3 upload, no dashboard "
                 "ingest, no email) -- pass --historical-deliver to enable")
    else:
        log.info("[HISTORICAL] delivery ENABLED: S3 upload + dashboard ingest%s",
                 "" if send_email else " (email suppressed by --skip-email)")

    from orchestrator.master_runner import process_batch  # lazy: avoids a cycle

    total = len(res.batches)
    failures: List[str] = []
    for i, batch in enumerate(res.batches, start=1):
        log.info("[HISTORICAL] processing batch %d/%d: %s (cameras=%s)",
                 i, total, batch.batch_key, batch.present_cameras())
        batch_root = os.path.join(hist_root, batch.batch_key)
        # Before committing to another train, reclaim ALREADY-DELIVERED ones if
        # the disk is low. `active_batch_key` keeps this batch untouched, and a
        # sibling is only reclaimed when its finalization marker proves it was
        # delivered -- an in-flight or failed batch has no marker.
        try:
            from delivery import cleanup as CU
            CU.ensure_free_space(workspace_root=hist_root,
                                 active_batch_key=batch.batch_key,
                                 cfg=_cleanup_config(keep_inputs),
                                 verbose=verbose)
        except Exception as e:  # noqa: BLE001 - never block a train on cleanup
            log.warning("[CLEANUP] pre-train sweep failed (non-fatal): %s", e)
        log.info("[HISTORICAL] staging inputs -> %s",
                 os.path.join(batch_root, CFG.DIR_DOWNLOADS))
        # ---- sequential architecture ---------------------------------------
        # Same two entry points the foreground sequential path uses; the only
        # difference is that the clips are staged out of S3 first.
        if mode == "sequential":
            log.info("[HISTORICAL] invoking SEQUENTIAL pipeline "
                     "(per-camera -> assembly)")
            t0 = time.time()
            try:
                asm = process_batch_sequential(
                    batch,
                    hist_root=hist_root,
                    s3_client=s3_client,
                    recon_models_dir=recon_models_dir,
                    feat_models_dir=feat_models_dir,
                    feature_config=feature_config,
                    deliver=deliver,
                    deliver_per_camera=deliver_per_camera,
                    send_email=send_email,
                    verbose=verbose,
                )
            except Exception as e:  # noqa: BLE001 - one batch must not stop the rest
                log.error("[HISTORICAL] batch %s raised %s: %s",
                          batch.batch_key, type(e).__name__, e, exc_info=True)
                failures.append(batch.batch_key)
                continue
            ok = bool(asm is not None and asm.total_wagons > 0)
            log.info("[HISTORICAL] batch %d/%d %s: wagons=%s (%.1fs)",
                     i, total, "completed" if ok else "FAILED",
                     getattr(asm, "total_wagons", 0), time.time() - t0)
            if ok:
                # Cleanup is gated on DELIVERY, not on assembly. A train whose
                # S3 upload or dashboard post failed keeps every byte, because
                # those are exactly the artifacts a retry needs. When this run
                # is not delivering at all (`--historical-deliver` absent) the
                # old input-only reclaim still applies -- there is nothing to
                # verify, and `downloads/` is re-stageable regardless.
                if deliver:
                    # Settle BEFORE reclaiming, so the rake grouping is updated
                    # while the train's artifacts are still on disk -- if
                    # settlement reveals a problem, everything needed to look at
                    # it is still there. Settlement never fails a train: a
                    # delivered train is delivered, and settle is re-runnable
                    # with the command its own failure line prints.
                    from delivery import rake_settle as RS
                    RS.settle_batch(batch_key=batch.batch_key, verbose=verbose)

                    from delivery import cleanup as CU
                    CU.cleanup_batch(
                        batch_root=batch_root, batch_key=batch.batch_key,
                        delivery=getattr(asm, "delivery", None),
                        cfg=_cleanup_config(keep_inputs), verbose=verbose)
                else:
                    _cleanup_inputs(batch_root, keep_inputs=keep_inputs)
            else:
                failures.append(batch.batch_key)
            continue

        log.info("[HISTORICAL] invoking existing pipeline (process_batch)")
        t0 = time.time()
        try:
            outcome = process_batch(
                batch=batch,
                workspace_root=hist_root,
                recon_models_dir=recon_models_dir,
                feat_models_dir=feat_models_dir,
                s3_client=s3_client,
                skip_upload=not deliver,
                skip_email=(not deliver) or (not send_email),
                verbose=verbose,
                feature_config=feature_config,
                # Stage-3 sampling modes (door/damage/load).  Passed through so a
                # historical batch is inspected with the SAME settings as a live
                # one; omitting them would silently fall back to the function
                # defaults and make historical output non-comparable.
                **(inference_opts or {}),
            )
        except Exception as e:  # noqa: BLE001 -- one bad batch must not stop the rest
            log.error("[HISTORICAL] batch %s raised %s: %s",
                      batch.batch_key, type(e).__name__, e, exc_info=True)
            failures.append(batch.batch_key)
            continue

        ok = outcome.final_status in (C.BATCH_COMPLETED, C.BATCH_COMPLETED_PARTIAL)
        log.info("[HISTORICAL] batch %d/%d %s: %s (%.1fs)",
                 i, total, "completed" if ok else "FAILED",
                 outcome.final_status, time.time() - t0)
        if outcome.report_pdf_path:
            log.info("[HISTORICAL] report: %s", outcome.report_pdf_path)
        if ok:
            # NO explicit dashboard-ingest call here, deliberately.  In THIS
            # package `process_batch` runs the per-camera dashboard feed itself
            # as Stage 6b, so calling it again would POST every historical batch
            # twice.  And when `deliver` is False, `skip_upload=True` makes
            # `process_batch` return before Stage 6 / 6b, so a non-delivering
            # historical run reaches no external endpoint at all.
            #
            # Settlement is the exception, and only when delivering: it is not
            # part of `process_batch`, it acts on runs ingest has ALREADY
            # written, and it is what leaves the rake grouping current instead
            # of waiting for someone to run it by hand. Same call, same
            # never-fails-a-train contract, as the sequential branch above.
            if deliver:
                from delivery import rake_settle as RS
                RS.settle_batch(batch_key=batch.batch_key, verbose=verbose)
            _cleanup_inputs(batch_root, keep_inputs=keep_inputs)
        else:
            failures.append(batch.batch_key)
            log.info("[HISTORICAL] inputs RETAINED at %s for diagnosis",
                     os.path.join(batch_root, CFG.DIR_DOWNLOADS))

    done = total - len(failures)
    log.info("[HISTORICAL] finished: %d/%d batch(es) completed%s",
             done, total, f", failed: {failures}" if failures else "")
    return 3 if failures else 0


def _cleanup_config(keep_inputs: bool):
    """Cleanup settings for this run.

    `--keep-inputs` stays the master off switch, so the existing flag keeps its
    existing meaning. Every other setting is read by `CleanupConfig.from_env`,
    NOT here: this module reads no environment variable directly, which is what
    guarantees it cannot redefine an existing variable's meaning.
    """
    from delivery.cleanup import CleanupConfig
    return CleanupConfig.from_env(enabled=not keep_inputs)


def _cleanup_inputs(batch_root: str, *, keep_inputs: bool) -> None:
    """Reclaim a SUCCESSFUL historical batch's reconstructible intermediates.

    Scope is deliberately narrow: only `<batch_root>/downloads/`, which is the
    one directory historical mode caused to exist.  Reports, evidence, processed
    videos, the wagon cache and the sealed state are left exactly as the pipeline
    wrote them -- pruning pipeline artifacts is a retention policy, and this
    package has no retention module whose behaviour could be reused.  (The
    sibling `global_train` repo prunes more via `delivery.retention`; that module
    does not exist here and is NOT reimplemented.)

    A FAILED batch is never cleaned -- its staged inputs stay for diagnosis.
    `--keep-inputs` keeps them for a successful batch too.

    Two directories are reclaimed, and only these two:

      `downloads/`    the clips this batch staged.
      `wagon_cache/`  the per-wagon JPEG cache.  It is the bulk of a batch's
                      several GB, and a bulk window is tens of trains back to
                      back, so without this a 12-hour re-run fills the disk after
                      a few trains.  Safe at this point: the cache is read DURING
                      report generation, which has already finished -- the
                      outcome carries a written PDF before we get here.

    Everything the run produced is KEPT: reports, evidence, processed videos, the
    sealed global state, per-wagon states and the archive.  Those are the
    artifacts of the run; the two above are reconstructible intermediates.
    """
    if keep_inputs:
        log.info("[HISTORICAL] --keep-inputs: intermediates left under %s",
                 batch_root)
        return
    freed = {}
    for name in (CFG.DIR_DOWNLOADS, CFG.DIR_WAGON_CACHE):
        d = os.path.join(batch_root, name)
        if not os.path.isdir(d):
            continue
        size = 0
        for root_dir, _dirs, files in os.walk(d):
            for fn in files:
                try:
                    size += os.path.getsize(os.path.join(root_dir, fn))
                except OSError:
                    pass
        try:
            shutil.rmtree(d)
            freed[name] = size
        except OSError as e:
            log.warning("[HISTORICAL] could not reclaim %s: %s", d, e)
    if freed:
        log.info("[HISTORICAL] reclaimed %.2f GB from %s (%s); reports, evidence "
                 "and processed videos kept",
                 sum(freed.values()) / 1e9, os.path.basename(batch_root),
                 ", ".join(sorted(freed)))
