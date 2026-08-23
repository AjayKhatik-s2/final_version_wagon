"""Settle a delivered train's rake grouping, right after it is delivered.

Settlement is a SEPARATE step from ingestion and runs after it. Ingest writes
one InspectionRun per camera; settle then clusters those runs into 7-minute rake
windows, assigns each a `rake_id`, and pairs loaded against empty clusters by
their shared 11-digit wagon numbers. Until it runs, a freshly ingested train has
runs in the database but no rake grouping, so the dashboard's rake views are
stale -- which is why this was being done by hand after every run.

    per-camera ingest  ->  InspectionRun rows  ->  settle  ->  rake_id / pairing
                                                     ^
                                              this module, once
                                              the train is delivered

Three things this module is careful about
-----------------------------------------
**`dry_run` is sent explicitly, always.** The API defaults it to TRUE. A call
that omits it returns 200, reports what it *would* change, and writes nothing --
the worst kind of failure, because it looks exactly like success. Every request
here carries `dry_run` as a literal.

**The train's date, not today's.** A historical run processes July dates in
August; settling "today" would cluster the wrong day's runs, or none. The date
comes from the batch key.

**It cannot fail a train.** A delivered train is delivered. Settlement is a
downstream grouping that can be re-run at any time with the same command, so
every failure here is logged and swallowed. Nothing in this module raises.

Off by default (`WAGONEYE_AUTO_SETTLE=1` to enable): it writes to a shared
production database, and that is not a thing to start doing implicitly.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core import constants as C
from core.logging_setup import get_logger

log = get_logger("delivery.settle")

#: `20260726_071641` -> `2026-07-26`. The batch key's digits are IST wall-clock
#: from the producer's filename, so the date is read straight off them and never
#: derived from the clock of the machine doing the processing.
_BATCH_KEY_DATE = re.compile(r"^(\d{4})(\d{2})(\d{2})_\d{6}$")

DEFAULT_TIMEOUT_SEC = 120.0


def batch_key_date(batch_key: str) -> Optional[str]:
    """`YYYY-MM-DD` for a batch key, or None if it is not a batch key.

    None is a refusal, not a fallback: settling the wrong date is worse than not
    settling, because it silently regroups another day's runs.
    """
    m = _BATCH_KEY_DATE.match((batch_key or "").strip())
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


@dataclass
class SettleConfig:
    enabled: bool = False
    """Off by default. Enabling it starts writing to a production database."""

    dry_run: bool = False
    """Sent EXPLICITLY on every request. See the module docstring: the API's own
    default is True, so omitting it silently turns a real settle into a preview."""

    url: str = ""
    timeout_sec: float = DEFAULT_TIMEOUT_SEC

    @classmethod
    def from_env(cls) -> "SettleConfig":
        """Read the environment HERE, so callers need not.

        `orchestrator/historical_runner.py` is required to read no environment
        variable directly -- that is what guarantees it cannot redefine an
        existing variable's meaning, and a test enforces it.

            WAGONEYE_AUTO_SETTLE=1        enable (default: off)
            WAGONEYE_SETTLE_DRY_RUN=1     preview instead of writing
            WAGONEYE_SETTLE_API_URL=...   override the endpoint
            WAGONEYE_SETTLE_TIMEOUT=120   seconds
        """
        def _flag(name: str) -> bool:
            raw = (os.getenv(name) or "").strip().lower()
            return raw in ("1", "true", "yes", "on")

        try:
            timeout = float(os.getenv("WAGONEYE_SETTLE_TIMEOUT")
                            or DEFAULT_TIMEOUT_SEC)
        except ValueError:
            timeout = DEFAULT_TIMEOUT_SEC

        return cls(
            enabled=_flag("WAGONEYE_AUTO_SETTLE"),
            dry_run=_flag("WAGONEYE_SETTLE_DRY_RUN"),
            url=C.SETTLE_API_URL,
            timeout_sec=timeout,
        )


@dataclass
class SettleResult:
    batch_key: str = ""
    date: str = ""
    attempted: bool = False
    ok: bool = False
    skipped_reason: str = ""
    status_code: Optional[int] = None
    dry_run: bool = False
    clusters: Optional[int] = None
    changes: Optional[int] = None
    error: str = ""
    request_url: str = ""

    def render(self) -> str:
        if not self.attempted:
            return f"[SETTLE] {self.batch_key or '?'} skipped -- {self.skipped_reason}"
        if not self.ok:
            return (f"[SETTLE] {self.date} FAILED (http={self.status_code}) "
                    f"-- {self.error}; rake grouping unchanged, the train is "
                    f"still delivered. Retry: {self.request_url}")
        mode = "DRY RUN (nothing written)" if self.dry_run else "committed"
        bits = [f"[SETTLE] {self.date} {mode}"]
        if self.clusters is not None:
            bits.append(f"clusters={self.clusters}")
        if self.changes is not None:
            bits.append(f"changes={self.changes}")
        return "  ".join(bits)


def _count_changes(doc: Any) -> Optional[int]:
    """Total run-level changes across the response, if it is shaped as expected.

    The response schema (`RangeSettlementResponse`) is the API's to change, so
    this reads defensively and returns None rather than guessing a number.
    """
    if not isinstance(doc, dict):
        return None
    total = 0
    found = False
    for key in ("dates", "results", "settlements"):
        seq = doc.get(key)
        if not isinstance(seq, list):
            continue
        for entry in seq:
            if not isinstance(entry, dict):
                continue
            for ck in ("changes", "updated_runs", "run_changes"):
                v = entry.get(ck)
                if isinstance(v, list):
                    total += len(v)
                    found = True
                elif isinstance(v, int):
                    total += v
                    found = True
    return total if found else None


def _count_clusters(doc: Any) -> Optional[int]:
    if not isinstance(doc, dict):
        return None
    for key in ("clusters", "rake_clusters"):
        v = doc.get(key)
        if isinstance(v, list):
            return len(v)
        if isinstance(v, int):
            return v
    return None


def settle_batch(
    *,
    batch_key: str,
    cfg: Optional[SettleConfig] = None,
    requests_mod: Any = None,
    verbose: bool = True,
) -> SettleResult:
    """Settle the rake grouping for one delivered train.  Never raises.

    `requests_mod` is injectable so tests exercise the request without network.
    """
    cfg = cfg or SettleConfig.from_env()
    res = SettleResult(batch_key=batch_key, dry_run=cfg.dry_run)

    if not cfg.enabled:
        res.skipped_reason = "auto-settle disabled (WAGONEYE_AUTO_SETTLE=1 to enable)"
        if verbose:
            log.info("%s", res.render())
        return res

    date = batch_key_date(batch_key)
    if not date:
        # Refusing beats guessing: settling the wrong date regroups another
        # day's runs, and that is not obviously wrong from the outside.
        res.skipped_reason = (f"cannot read a date from batch_key "
                             f"{batch_key!r} -- refusing to guess one")
        log.warning("%s", res.render())
        return res
    res.date = date

    url = cfg.url or C.SETTLE_API_URL
    params = {"start_date": date, "end_date": date,
              "dry_run": "true" if cfg.dry_run else "false"}
    res.request_url = f"{url}?start_date={date}&end_date={date}&dry_run={params['dry_run']}"
    res.attempted = True

    if requests_mod is None:
        try:
            import requests as requests_mod  # type: ignore
        except Exception as e:  # noqa: BLE001
            res.error = f"requests unavailable: {e}"
            log.warning("%s", res.render())
            return res

    try:
        resp = requests_mod.post(url, params=params, timeout=cfg.timeout_sec)
        res.status_code = getattr(resp, "status_code", None)
        if res.status_code is not None and 200 <= int(res.status_code) < 300:
            res.ok = True
            try:
                doc = resp.json()
            except Exception:                                    # noqa: BLE001
                doc = None
            res.clusters = _count_clusters(doc)
            res.changes = _count_changes(doc)
        else:
            body = ""
            try:
                body = (getattr(resp, "text", "") or "")[:300]
            except Exception:                                    # noqa: BLE001
                pass
            res.error = f"non-2xx response{': ' + body if body else ''}"
    except Exception as e:  # noqa: BLE001
        # A delivered train stays delivered. Settlement is re-runnable, so the
        # command to retry is logged rather than the train being failed.
        res.error = f"{type(e).__name__}: {e}"

    if verbose:
        (log.info if res.ok else log.warning)("%s", res.render())
    return res
