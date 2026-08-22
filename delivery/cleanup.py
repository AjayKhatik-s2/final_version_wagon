"""Reclaim a train's reconstructible intermediates AFTER it is delivered.

Sequential mode writes several GB per train that exist only to be read once.
On a 58 GB root volume a bulk historical window fills the disk after a handful
of trains -- measured: `camera_cache` alone reached ~18 GB and the volume hit
100%, which breaks the pipeline and `apt` alike.

The rule this module enforces
-----------------------------
Nothing is deleted until the train is DELIVERED, not merely processed. A train
whose S3 upload or dashboard post failed keeps every byte, because those are
exactly the artifacts a retry needs.

    camera processing -> assembly -> combined report -> S3 -> dashboard
                                                              |
                                                    verify delivery
                                                              |
                                                        cleanup HERE

`is_delivered()` is the single gate. Everything else in this file is bookkeeping
around it.

What is temporary, and why each one is safe
------------------------------------------
`downloads/`
    The clips this batch staged from S3. The originals are untouched in the
    input bucket, so this is a cache of a cache.

`wagon_cache/`
    The global per-wagon JPEGs. Read DURING report generation, which has
    finished by the time delivery succeeds.

`camera_evidence/<CAM>/camera_cache/`
    The per-camera JPEGs, and the bulk of the problem -- four of them per train.
    Written by `camera_runner`, read only by the camera-local feature pass and
    `camera_report_adapter.adapt()`, both of which run at camera-seal time.
    Assembly materializes its OWN `wagon_cache/` and, as
    `camera_runner` documents, "recomputes all three features over the global
    wagons and ignores everything written here". Nothing reads it after
    assembly.

`processed_videos/`
    The four overlay videos. OPT-IN only (`delete_processed_videos`), and only
    once `archived["processed_videos"]` proves they reached S3 -- the published
    documents link to the S3 copies, so the local ones are redundant, but they
    are also the most expensive thing to regenerate.

What is always kept
-------------------
`reports/`, `global_state/`, `wagon_states/`, `evidence/`, `archive/`,
`delivery/`, and inside every camera bundle the manifest, `tracking_full.json`,
`tracking.json`, `camera_report.json`, the camera PDF, `segments.json`,
`engine_frames/`, `features/` and `evidence/`. Those are the run's output and
its audit trail. `tracking_full.json` in particular must survive: it is the only
place the full-fidelity gap trajectories live, and the damage-boundary resolver
reads them.

Safety
------
* every target path is verified to sit INSIDE the batch root before removal;
* the batch root itself is never removed, and neither is any sibling batch;
* no S3 object is ever touched -- this module has no S3 client;
* a missing directory is a no-op, so running twice is safe;
* nothing raises: a cleanup failure logs and returns, because a delivered train
  must not be reported as failed just because the disk did not shrink.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import config as CFG
from core import constants as C
from core.logging_setup import get_logger

log = get_logger("delivery.cleanup")

#: Per-camera cache, relative to a batch root.  The one that actually fills the
#: disk in sequential mode.
CAMERA_EVIDENCE_DIR = "camera_evidence"
CAMERA_CACHE_DIR = "camera_cache"

#: Never removed, whatever else happens.  Listed positively so a future reader
#: can see the intent rather than infer it from what is absent.
RETAINED_DIRS: Tuple[str, ...] = (
    CFG.DIR_REPORTS, CFG.DIR_GLOBAL_STATE, CFG.DIR_WAGON_STATES,
    CFG.DIR_EVIDENCE, CFG.DIR_ARCHIVE, CFG.DIR_DELIVERY,
)

RETAINED_IN_CAMERA_BUNDLE: Tuple[str, ...] = (
    "manifest.json", "tracking_full.json", "tracking.json",
    "camera_report.json", "segments.json", "classification.json",
    "gap_validation.json", "fragments.json", "wagon_region.json",
    "wagon_active_recovery.json", "run_result.json",
    "engine_frames", "features", "evidence", "delivery",
)


@dataclass
class CleanupConfig:
    """Tunables.  Defaults are the conservative choice everywhere."""

    dry_run: bool = False
    """Report what WOULD be removed and remove nothing."""

    enabled: bool = True
    """Master switch.  `--keep-inputs` / `WAGONEYE_KEEP_INPUTS` turns it off."""

    delete_camera_cache: bool = True
    """The per-camera JPEG caches -- the bulk of a sequential train."""

    delete_wagon_cache: bool = True
    delete_downloads: bool = True

    delete_processed_videos: bool = False
    """OFF by default.  The overlay videos are expensive to regenerate, and
    they are only redundant once S3 holds them -- which is checked separately."""

    min_free_gb: float = 8.0
    """Below this, the pre-train sweep reclaims already-DELIVERED batches."""

    require_dashboard: Optional[bool] = None
    """Whether dashboard ingest must have succeeded before cleanup.

    `None` (default) means "required only if dashboard ingest was enabled for
    this run" -- a run with the dashboard switched off should still be able to
    reclaim disk. Set True to demand it regardless.
    """


    @classmethod
    def from_env(cls, *, enabled: bool = True) -> "CleanupConfig":
        """Build from the environment.

        Lives HERE, not in `historical_runner`, because that module is required
        to read no environment variable directly -- that is what guarantees it
        cannot redefine the meaning of an existing variable, and there is a test
        enforcing it. Config belongs with the code it configures anyway.

            WAGONEYE_CLEANUP_DRY_RUN=1            report, delete nothing
            WAGONEYE_CLEANUP_MIN_FREE_GB=8        pre-train sweep threshold
            WAGONEYE_CLEANUP_PROCESSED_VIDEOS=1   also drop the overlay videos
        """
        def _flag(name: str, default: bool = False) -> bool:
            raw = os.getenv(name)
            if raw is None or raw == "":
                return default
            return raw.strip().lower() in ("1", "true", "yes", "on")

        def _num(name: str, default: float) -> float:
            try:
                return float(os.getenv(name) or default)
            except ValueError:
                return default

        return cls(
            enabled=enabled,
            dry_run=_flag("WAGONEYE_CLEANUP_DRY_RUN"),
            min_free_gb=_num("WAGONEYE_CLEANUP_MIN_FREE_GB", 8.0),
            delete_processed_videos=_flag("WAGONEYE_CLEANUP_PROCESSED_VIDEOS"),
        )


DEFAULT_CONFIG = CleanupConfig()


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def is_delivered(batch_root: str, delivery: Any,
                 cfg: CleanupConfig = DEFAULT_CONFIG) -> Tuple[bool, str]:
    """`(ok, reason)` -- may this batch's intermediates be reclaimed?

    Deliberately strict and deliberately explicit: the reason is logged either
    way, so an operator can see WHY a train kept its artifacts.

    On what can and cannot be verified: `finalize` records `archived` as
    per-subtree FILE COUNTS, not per-artifact receipts. So "the JSON and PDF
    were uploaded" is checked as "the reports subtree uploaded at least one file
    and no error was recorded" -- which is what the data supports. Claiming a
    per-file guarantee here would be a guarantee this pipeline cannot make.
    """
    if delivery is None:
        return False, "no delivery result -- train was processed but not delivered"

    errors = list(getattr(delivery, "errors", None) or [])
    if errors:
        return False, f"delivery reported {len(errors)} error(s): {errors[:3]}"

    if not bool(getattr(delivery, "uploaded", False)):
        return False, "delivery.uploaded is False -- nothing reached S3"

    # 1. the combined report must exist locally
    reports = os.path.join(batch_root, CFG.DIR_REPORTS)
    combined = os.path.join(reports, "combined_train_report.json")
    if not os.path.isfile(combined):
        return False, "combined_train_report.json missing -- Stage 5 incomplete"

    # 2/3/4. the subtrees that carry the report, its evidence and the videos
    archived = dict(getattr(delivery, "archived", None) or {})
    for subtree in ("reports", "evidence"):
        if int(archived.get(subtree) or 0) <= 0:
            return False, (f"no file uploaded from '{subtree}' "
                           f"(archived={archived or '{}'})")

    # 5. dashboard, when it was enabled for this run
    dash = dict(getattr(delivery, "dashboard", None) or {})
    dash_enabled = bool(dash.get("enabled"))
    need_dash = cfg.require_dashboard
    if need_dash is None:
        need_dash = dash_enabled
    if need_dash:
        if not dash_enabled:
            return False, "dashboard delivery required but was not enabled"
        cams = dict(dash.get("cameras") or {})
        good = [c for c, i in cams.items()
                if str((i or {}).get("status") or "") in
                ("ingested", "already_ingested")]
        if not good:
            return False, (f"dashboard ingest succeeded for no camera "
                           f"(statuses={{{', '.join(f'{c}:{(i or {}).get('status')}' for c, i in sorted(cams.items()))}}})")

    return True, "delivered"


def is_delivered_marker(batch_root: str) -> Tuple[bool, str]:
    """Same question for a batch we no longer hold a DeliveryResult for.

    Used by the pre-train disk sweep, which looks at OTHER batches on disk. The
    finalization marker (`delivery.finalization`) is written only after Stage 6
    completes, so its presence -- together with the combined report -- is the
    durable record that a train was delivered.
    """
    from delivery import finalization

    combined = os.path.join(batch_root, CFG.DIR_REPORTS,
                            "combined_train_report.json")
    if not os.path.isfile(combined):
        return False, "no combined report"
    marker = finalization.load(batch_root)
    if not marker:
        return False, "no finalization marker -- delivery never completed"
    return True, "delivered (marker present)"


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def _dir_size(path: str) -> int:
    total = 0
    for root_dir, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root_dir, fn))
            except OSError:
                pass
    return total


def _inside(child: str, parent: str) -> bool:
    """True only if `child` really sits under `parent`.

    Guards against a crafted or mis-joined path escaping the batch: every
    removal is checked, so no bug elsewhere can turn this into an `rm -rf` of
    something outside the batch tree.
    """
    c = os.path.realpath(child)
    p = os.path.realpath(parent)
    return c != p and os.path.commonpath([c, p]) == p


def _videos_reached_s3(batch_root: str, delivery: Any) -> bool:
    """Did the overlay videos actually reach S3?

    Only then are the local copies redundant, so this gates
    `delete_processed_videos`. It is the most expensive artifact to regenerate:
    if the answer is not a clear yes, the videos stay.

    Two sources, because there are two callers. The per-train path holds a live
    `DeliveryResult`. The pre-train SWEEP does not -- it passes `delivery=None`
    and establishes delivery from the finalization marker instead. Reading only
    the live object silently made the sweep's `delete_processed_videos` a no-op:
    it reported "freed 0.00 GB from 0 path(s)" on batches whose markers plainly
    recorded four uploaded videos. The marker is the same evidence the sweep
    already trusts to call the batch delivered, so it is the right source here.
    """
    archived = dict(getattr(delivery, "archived", None) or {})
    if archived:
        try:
            return int(archived.get("processed_videos") or 0) > 0
        except (TypeError, ValueError):
            return False

    # Sweep path: no live result. `archived` is the upload COUNT and the
    # strongest proof there is, so newer batches carry it in the marker (see
    # delivery/finalize.py). Batches finalized before that do not, and for them
    # the durable evidence is the published report's own
    # `processed_video_urls`: the combined report and every dashboard document
    # link the viewer at those S3 objects, so if they are absent from S3 the
    # delivery is already broken in a way deleting a local copy cannot worsen.
    # Weaker than a count, and deliberately the LAST resort.
    try:
        from delivery import finalization
        marker = finalization.load(batch_root) or {}
    except Exception:                                            # noqa: BLE001
        return False
    if not marker:
        return False
    try:
        counts = dict(marker.get("archived") or {})
        if counts:
            return int(counts.get("processed_videos") or 0) > 0
        import json as _json
        rp = os.path.join(batch_root, CFG.DIR_REPORTS,
                          "combined_train_report.json")
        with open(rp, "r", encoding="utf-8") as fh:
            doc = _json.load(fh)
        urls = ((doc.get("train_metadata") or {}
                 ).get("processed_video_urls") or {})
        return bool([u for u in urls.values() if u])
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def plan(batch_root: str, cfg: CleanupConfig = DEFAULT_CONFIG,
         *, videos_uploaded: bool = False) -> List[Tuple[str, int]]:
    """`[(path, bytes), ...]` this configuration would remove, largest first.

    Pure: touches nothing. `cleanup_batch` and the dry-run path both use it, so
    what a dry run prints is exactly what a real run would delete.
    """
    targets: List[str] = []
    if cfg.delete_downloads:
        targets.append(os.path.join(batch_root, CFG.DIR_DOWNLOADS))
    if cfg.delete_wagon_cache:
        targets.append(os.path.join(batch_root, CFG.DIR_WAGON_CACHE))
    if cfg.delete_camera_cache:
        cam_root = os.path.join(batch_root, CAMERA_EVIDENCE_DIR)
        if os.path.isdir(cam_root):
            for entry in sorted(os.listdir(cam_root)):
                targets.append(os.path.join(cam_root, entry, CAMERA_CACHE_DIR))
    if cfg.delete_processed_videos and videos_uploaded:
        targets.append(os.path.join(batch_root, CFG.DIR_PROCESSED_VIDEOS))

    out: List[Tuple[str, int]] = []
    for t in targets:
        if not os.path.isdir(t):
            continue                      # idempotent: already gone
        if not _inside(t, batch_root):
            log.error("[CLEANUP] refusing %s -- outside %s", t, batch_root)
            continue
        if os.path.basename(t) in RETAINED_DIRS:
            log.error("[CLEANUP] refusing %s -- retained artifact", t)
            continue
        out.append((t, _dir_size(t)))
    out.sort(key=lambda x: -x[1])
    return out


# ---------------------------------------------------------------------------
# Doing it
# ---------------------------------------------------------------------------

@dataclass
class CleanupResult:
    batch_key: str = ""
    batch_root: str = ""
    performed: bool = False
    dry_run: bool = False
    skipped_reason: str = ""
    removed: List[str] = field(default_factory=list)
    freed_bytes: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def freed_gb(self) -> float:
        return self.freed_bytes / 1e9

    def render(self) -> str:
        if not self.performed:
            return (f"[CLEANUP] {self.batch_key or '?'} SKIPPED -- "
                    f"{self.skipped_reason}")
        verb = "would free" if self.dry_run else "freed"
        names = ", ".join(sorted({os.path.basename(os.path.dirname(p))
                                  + "/" + os.path.basename(p)
                                  if os.path.basename(p) == CAMERA_CACHE_DIR
                                  else os.path.basename(p)
                                  for p in self.removed})) or "(nothing)"
        line = (f"[CLEANUP] {self.batch_key} {verb} {self.freed_gb:.2f} GB "
                f"from {len(self.removed)} path(s): {names}")
        if self.errors:
            line += f"  errors={self.errors}"
        return line

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_key": self.batch_key, "batch_root": self.batch_root,
            "performed": self.performed, "dry_run": self.dry_run,
            "skipped_reason": self.skipped_reason,
            "removed": list(self.removed), "freed_bytes": self.freed_bytes,
            "freed_gb": round(self.freed_gb, 3), "errors": list(self.errors),
        }


def cleanup_batch(
    *,
    batch_root: str,
    batch_key: str = "",
    delivery: Any = None,
    cfg: CleanupConfig = DEFAULT_CONFIG,
    require_delivery: bool = True,
    verbose: bool = True,
) -> CleanupResult:
    """Reclaim ONE delivered batch's intermediates.  Never raises.

    `require_delivery=False` is for the pre-train sweep, which has already
    established delivery from the finalization marker and has no
    `DeliveryResult` to pass.
    """
    res = CleanupResult(batch_key=batch_key or os.path.basename(batch_root),
                        batch_root=batch_root, dry_run=bool(cfg.dry_run))

    if not cfg.enabled:
        res.skipped_reason = "cleanup disabled"
        if verbose:
            log.info("%s", res.render())
        return res
    if not batch_root or not os.path.isdir(batch_root):
        res.skipped_reason = f"no such batch directory: {batch_root}"
        if verbose:
            log.info("%s", res.render())
        return res

    if require_delivery:
        ok, why = is_delivered(batch_root, delivery, cfg)
        if not ok:
            res.skipped_reason = why
            log.warning("[CLEANUP] %s KEPT -- %s", res.batch_key, why)
            return res

    videos_uploaded = _videos_reached_s3(batch_root, delivery)
    try:
        targets = plan(batch_root, cfg, videos_uploaded=videos_uploaded)
    except Exception as e:  # noqa: BLE001
        res.skipped_reason = f"could not plan cleanup: {type(e).__name__}: {e}"
        log.warning("[CLEANUP] %s", res.skipped_reason)
        return res

    res.performed = True
    for path, size in targets:
        if cfg.dry_run:
            res.removed.append(path)
            res.freed_bytes += size
            if verbose:
                log.info("[CLEANUP] --dry-run: would remove %s (%.2f GB)",
                         path, size / 1e9)
            continue
        try:
            shutil.rmtree(path)
            res.removed.append(path)
            res.freed_bytes += size
        except OSError as e:
            res.errors.append(f"{path}: {e}")
            log.warning("[CLEANUP] could not remove %s: %s", path, e)

    if verbose:
        log.info("%s", res.render())
    return res


# ---------------------------------------------------------------------------
# Disk pressure, before the next train
# ---------------------------------------------------------------------------

def free_gb(path: str = "/") -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return float("inf")        # unknown: never block a run on a bad stat


def ensure_free_space(
    *,
    workspace_root: str,
    active_batch_key: str = "",
    cfg: CleanupConfig = DEFAULT_CONFIG,
    verbose: bool = True,
) -> List[CleanupResult]:
    """Before starting a train, reclaim ALREADY-DELIVERED batches if disk is low.

    Three guards, all necessary:

    * only runs when free space is below `min_free_gb`, so a healthy disk is
      never touched;
    * `active_batch_key` is skipped -- the batch about to be processed, or being
      processed, keeps everything;
    * a sibling batch is reclaimed only if its finalization marker says it was
      delivered. An in-flight or failed batch has no marker and is left alone.

    Oldest batches go first, so the most recent output survives longest.
    """
    out: List[CleanupResult] = []
    if not cfg.enabled:
        return out
    before = free_gb(workspace_root if os.path.isdir(workspace_root) else "/")
    if before >= float(cfg.min_free_gb):
        return out

    log.warning("[CLEANUP] free space %.1f GB is below the %.1f GB threshold "
                "-- reclaiming delivered batches", before, cfg.min_free_gb)
    try:
        entries = sorted(d for d in os.listdir(workspace_root)
                         if os.path.isdir(os.path.join(workspace_root, d)))
    except OSError as e:
        log.warning("[CLEANUP] cannot scan %s: %s", workspace_root, e)
        return out

    for name in entries:
        if active_batch_key and name == active_batch_key:
            continue                       # never the batch we are about to run
        root = os.path.join(workspace_root, name)
        ok, why = is_delivered_marker(root)
        if not ok:
            log.info("[CLEANUP] %s kept -- %s", name, why)
            continue
        r = cleanup_batch(batch_root=root, batch_key=name, delivery=None,
                          cfg=cfg, require_delivery=False, verbose=verbose)
        out.append(r)
        if free_gb(workspace_root) >= float(cfg.min_free_gb):
            break

    after = free_gb(workspace_root)
    log.info("[CLEANUP] disk sweep: %.1f GB -> %.1f GB after %d batch(es)",
             before, after, len(out))
    return out
