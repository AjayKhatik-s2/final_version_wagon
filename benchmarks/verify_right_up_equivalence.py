#!/usr/bin/env python3
"""RIGHT_UP equivalence: sequential run_master_camera() vs a GOLDEN batch.

Proves the sequential single-camera runner reproduces the proven Stage-1
per-camera chain exactly, by re-running RIGHT_UP and diffing against a
known-good batch produced by the normal (batch-mode) pipeline.

    python benchmarks/verify_right_up_equivalence.py \
        --golden-batch batch_outputs/20260817_064608 \
        --video        local_inputs/right_up.mp4

Reads only. Runs no fusion, writes nothing into the golden batch, and does not
touch production code or wagon_count/.

Exit code 0 = equivalent, 1 = differences found (listed), 2 = setup error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def say(*a: Any) -> None:
    print(*a, flush=True)


def _fmt(seq: List[Any], n: int = 8) -> str:
    """Compact preview so a 58-element list stays readable."""
    if len(seq) <= n * 2:
        return str(seq)
    head = ", ".join(map(str, seq[:n]))
    tail = ", ".join(map(str, seq[-n:]))
    return f"[{head}, ... ({len(seq) - 2 * n} more) ..., {tail}]"


class Comparison:
    """Ordered checks with an exact first-difference report."""

    def __init__(self) -> None:
        self.diffs: List[Tuple[str, Any, Any]] = []

    def check(self, label: str, new: Any, ref: Any) -> bool:
        ok = new == ref
        if ok:
            say(f"  PASS  {label}")
        else:
            say(f"  DIFF  {label}")
            say(f"          new = {_fmt(new) if isinstance(new, list) else new!r}")
            say(f"          ref = {_fmt(ref) if isinstance(ref, list) else ref!r}")
            if isinstance(new, list) and isinstance(ref, list):
                if len(new) != len(ref):
                    say(f"          length differs: {len(new)} vs {len(ref)}")
                for i, (a, b) in enumerate(zip(new, ref)):
                    if a != b:
                        say(f"          FIRST DIFFERENCE at index {i}: "
                            f"new={a!r}  ref={b!r}")
                        break
            self.diffs.append((label, new, ref))
        return ok

    def note(self, label: str, new: Any, ref: Any) -> None:
        """Informational -- reported, but not an equivalence failure."""
        flag = "same" if new == ref else "DIFFERS"
        say(f"  note  {label}: new={new!r} ref={ref!r}  ({flag})")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden-batch", required=True,
                    help="known-good batch_outputs/<key>/ (read-only)")
    ap.add_argument("--video", required=True, help="RIGHT_UP source video")
    ap.add_argument("--models-dir",
                    default=os.path.join(_ROOT, "models", "reconstruction"))
    ap.add_argument("--camera", default="RIGHT_UP")
    args = ap.parse_args(argv)

    golden = os.path.abspath(args.golden_batch)
    state_p = os.path.join(golden, "global_state", "global_train_state.json")
    track_p = os.path.join(golden, "global_state", "per_camera_tracking.json")
    for p in (state_p, track_p, args.video):
        if not os.path.exists(p):
            say(f"ERROR: missing {p}")
            return 2
    gap_model = os.path.join(args.models_dir, "right_up_wagon_gap.pt")
    cls_model = os.path.join(args.models_dir, "side_classification.pt")
    for p in (gap_model, cls_model):
        if not os.path.exists(p):
            say(f"ERROR: missing model {p}")
            return 2

    from orchestrator import camera_pipeline as cp

    state = json.load(open(state_p, encoding="utf-8"))
    track = json.load(open(track_p, encoding="utf-8"))
    if args.camera not in track:
        say(f"ERROR: {args.camera} not in per_camera_tracking.json")
        return 2
    ref_cam = track[args.camera]

    say("=" * 74)
    say(f"RIGHT_UP EQUIVALENCE   sequential run_master_camera()  vs  GOLDEN")
    say(f"  golden : {golden}")
    say(f"  video  : {args.video}")
    say("=" * 74)
    say(f"  golden total_wagons        : {state.get('total_wagons')}")
    say(f"  golden right_up_final_gaps : {state.get('right_up_final_gap_count')}")
    say(f"  golden fusion_mode         : {state.get('fusion_mode')}")

    t0 = time.perf_counter()
    res = cp.run_master_camera(
        camera_id=args.camera, video_path=os.path.abspath(args.video),
        gap_model_path=gap_model, side_cls_path=cls_model, verbose=False)
    say(f"\nsequential run completed in {time.perf_counter() - t0:.1f}s")

    # ---- stage diagnostics, side by side ------------------------------
    sres, vres, rec = res.stitch, res.validation, res.recovery
    g_st = (state.get("fragment_stitching") or {}).get(args.camera, {})
    g_gv = (state.get("gap_validation_statistics") or {}).get(args.camera, {})
    g_rc = state.get("wagon_active_recovery") or {}

    say("\n--- STAGE DIAGNOSTICS ---")
    say(f"  {'stage':<28}{'new':>10}{'golden':>10}")
    rows = [
        ("stitch input candidates", getattr(sres, "input_count", None),
         g_st.get("input_candidates")),
        ("stitch output candidates", len(getattr(sres, "events", []) or []),
         g_st.get("output_candidates")),
        ("validation accepted", len(getattr(vres, "accepted", []) or []),
         g_gv.get("valid_gap_events")),
        ("validation rejected", len(getattr(vres, "rejected", []) or []),
         g_gv.get("rejected_total")),
        ("tracked candidates", getattr(vres, "tracked_candidate_count", None),
         g_gv.get("tracked_candidates")),
        ("recovery recovered", len(getattr(rec, "recovered", []) or []) if rec else 0,
         len(g_rc.get("recovered") or []) if isinstance(g_rc.get("recovered"), list)
         else g_rc.get("recovered")),
    ]
    for name, a, b in rows:
        say(f"  {name:<28}{str(a):>10}{str(b):>10}")

    cmp = Comparison()
    say("\n--- EQUIVALENCE CHECKS ---")

    new_gaps = list(res.tracks.gaps)
    ref_gaps = list(ref_cam.get("gaps") or [])

    cmp.check("4. gap count", len(new_gaps), len(ref_gaps))
    cmp.check("4b. vs right_up_final_gap_count", len(new_gaps),
              state.get("right_up_final_gap_count"))
    cmp.check("3. track IDs", [g.track_id for g in new_gaps],
              [g["track_id"] for g in ref_gaps])
    cmp.check("2. gap start frames", [g.start_frame for g in new_gaps],
              [g["start_frame"] for g in ref_gaps])
    cmp.check("2b. gap end frames", [g.end_frame for g in new_gaps],
              [g["end_frame"] for g in ref_gaps])
    cmp.check("1. final accepted gaps (id,start,end)",
              [(g.track_id, g.start_frame, g.end_frame) for g in new_gaps],
              [(g["track_id"], g["start_frame"], g["end_frame"])
               for g in ref_gaps])

    n_rec = len(rec.recovered) if rec and getattr(rec, "recovered", None) else 0
    g_rec_raw = g_rc.get("recovered")
    n_gref = (len(g_rec_raw) if isinstance(g_rec_raw, list)
              else int(g_rec_raw or 0))
    cmp.check("8. recovery-added gaps", n_rec, n_gref)

    say("\n--- SEGMENTS ---")
    idx = [s.index for s in res.segments]
    cmp.check("7. local ordering contiguous", idx,
              list(range(1, len(res.segments) + 1)))
    cmp.check("5. segment count == gaps+1", len(res.segments), len(new_gaps) + 1)
    if res.segments:
        say(f"  6. first segment : frames {res.segments[0].start_frame}-"
            f"{res.segments[0].end_frame}  label={res.segments[0].label}")
        say(f"     last  segment : frames {res.segments[-1].start_frame}-"
            f"{res.segments[-1].end_frame}  label={res.segments[-1].label}")
    # The golden roster is the WAGON-window subset; not directly comparable.
    cmp.note("golden total_wagons vs new segment count",
             len(res.segments), state.get("total_wagons"))
    say("      (golden total_wagons is the post-wagon-window subset -- the "
        "window filter runs in FUSION, outside this module, so a difference "
        "here is expected and is NOT an equivalence failure)")

    say("\n--- 9. CLASSIFICATIONS ---")
    ref_cls = ref_cam.get("pre_fusion_classifications") or []
    cmp.check("classification count", len(res.classifications), len(ref_cls))
    if ref_cls:
        cmp.check("label sequence", [c.label for c in res.classifications],
                  [c["label"] for c in ref_cls])
    else:
        say("  note  golden has no pre_fusion_classifications block; "
            "label sequence not comparable")

    say("\n" + "=" * 74)
    if not cmp.diffs:
        say("RESULT: EQUIVALENT -- no differences on any compared dimension")
        say("=" * 74)
        return 0
    say(f"RESULT: NOT EQUIVALENT -- {len(cmp.diffs)} difference(s)")
    for label, _n, _r in cmp.diffs:
        say(f"  - {label}")
    say("\nUse the STAGE DIAGNOSTICS above to attribute the divergence:")
    say("  stitch counts differ    -> reassemble_fragments arguments")
    say("  validation counts differ-> GapValidationConfig construction")
    say("  recovery differs        -> derive_wagon_window / recovery args")
    say("  gaps match, labels don't-> classification or temporal smoothing")
    say("=" * 74)
    return 1


if __name__ == "__main__":
    sys.exit(main())
