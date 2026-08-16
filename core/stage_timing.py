"""Per-stage wall-clock instrumentation for the orchestrator.

Measurement only: this module never changes what a stage computes, only how
long it took.  It is driven entirely from `orchestrator/master_runner.py`, so
no stage implementation -- and in particular nothing under `wagon_count/` --
imports or is modified by it.

Wall clock is the right metric here.  Stage 3 runs features concurrently, so
comparing the sum of the individual feature times against the Stage-3 total is
what reveals whether they are genuinely overlapping:

    sum(features) >> stage3_total   ->  real parallelism
    sum(features) ~= stage3_total   ->  serialized on the CPU

`overlap_factor` in the emitted JSON reports exactly that ratio.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple


class StageTimer:
    """Thread-safe ordered collection of named wall-clock spans.

    Feature processors are timed from worker threads, so both `record()` and
    the `stage()` context manager take a lock.  Ordering is first-touch, which
    keeps the printed table in pipeline order.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: Dict[str, float] = {}
        self._order: List[str] = []
        self._t0 = time.perf_counter()

    # -- recording --------------------------------------------------------

    def record(self, name: str, seconds: float) -> None:
        with self._lock:
            if name not in self._spans:
                self._order.append(name)
                self._spans[name] = 0.0
            # Accumulate, so a stage entered more than once sums correctly.
            self._spans[name] += float(seconds)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block. The span is recorded even if the block raises."""
        t = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - t)

    # -- reading ----------------------------------------------------------

    def get(self, name: str) -> float:
        with self._lock:
            return self._spans.get(name, 0.0)

    def elapsed_total(self) -> float:
        return time.perf_counter() - self._t0

    def items(self) -> List[Tuple[str, float]]:
        with self._lock:
            return [(n, self._spans[n]) for n in self._order]

    # -- reporting --------------------------------------------------------

    def overlap_factor(self, total_name: str, part_names: List[str]) -> Optional[float]:
        """sum(parts) / total -- >1.0 means the parts genuinely overlapped."""
        total = self.get(total_name)
        if total <= 0:
            return None
        parts = sum(self.get(p) for p in part_names)
        if parts <= 0:
            return None
        return round(parts / total, 3)

    def to_dict(self, *, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        doc: Dict[str, Any] = {
            "schema": "wagon_eye.stage_timings.v1",
            "wall_clock_seconds": {n: round(v, 3) for n, v in self.items()},
            "total_seconds": round(self.elapsed_total(), 3),
        }
        if extra:
            doc.update(extra)
        return doc

    def write(self, path: str, *, extra: Optional[Dict[str, Any]] = None) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(extra=extra), f, indent=2)
        return path

    def render_table(self, *, title: str = "STAGE TIMINGS") -> str:
        rows = self.items()
        if not rows:
            return ""
        width = max(len(n) for n, _ in rows)
        total = self.elapsed_total()
        out = [f"\n{'=' * (width + 26)}", f"  {title}", f"{'=' * (width + 26)}"]
        for name, secs in rows:
            pct = (secs / total * 100.0) if total > 0 else 0.0
            out.append(f"  {name:<{width}}  {secs:9.2f}s  {pct:5.1f}%")
        out.append(f"  {'-' * (width + 20)}")
        out.append(f"  {'TOTAL':<{width}}  {total:9.2f}s")
        out.append("=" * (width + 26))
        return "\n".join(out)
