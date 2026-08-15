"""Per-candidate rejection table for a completed run.

WHY THIS EXISTS. When a train under-counts, the question is always "which real
gap was thrown away, and by which gate?". The pipeline already records every
rejected candidate with its measured evidence; this script renders that record
as a table you can read, so a failure can be diagnosed on a machine that has the
video without shipping the video anywhere.

It reads ONLY the run's own JSON. It re-derives nothing and re-runs no models, so
it cannot disagree with the pipeline that produced the numbers -- if the table
says a candidate was rejected on speed, that is the decision the pipeline made.

    python rejection_report.py results/global_train_state.json
    python rejection_report.py results/global_train_state.json --csv rejections.csv
    python rejection_report.py results/global_train_state.json --camera RIGHT_UP --soft-only

Read the table like this:

  * A SOFT rejection inside WAGON_ACTIVE that was NOT recovered is the shape of
    an under-count. The `recovery` column says what blocked it.
  * A HARD rejection is a static / reversed / duplicate / blind-track artefact.
    Those are meant to die, in every train state.
  * `speed` well below `ref_speed` with state=WAGON_ACTIVE is the classic
    deceleration false-negative: the candidate is real, the reference is stale.
  * `nearest_prev` / `nearest_next` give the neighbouring ACCEPTED gaps, in
    seconds. A rejected candidate sitting midway between two accepted gaps that
    are ~2x the normal spacing apart is almost certainly a lost wagon boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from bisect import bisect_left
from typing import Any, Dict, List, Optional, Sequence

# Fall back gracefully: the table must render even if the validator module is
# absent (e.g. someone copies this script next to a JSON on another machine).
try:
    from gap_validation import HARD_REJECTION_REASONS, SOFT_REJECTION_REASONS
except Exception:                                            # pragma: no cover
    HARD_REJECTION_REASONS = frozenset()
    SOFT_REJECTION_REASONS = frozenset()


# ---------------------------------------------------------------------------
# train state
# ---------------------------------------------------------------------------

PRE_WAGON, WAGON_ACTIVE, POST_WAGON, UNKNOWN_STATE = (
    "PRE_WAGON", "WAGON_ACTIVE", "POST_WAGON", "UNKNOWN")


def wagon_window_frames(state: Dict[str, Any]) -> tuple:
    """(start, end) frame of the counted wagon region, or (None, None).

    Accepts the several key spellings the window has carried, so an older run's
    JSON still renders instead of silently reporting UNKNOWN for every row.
    """
    win = state.get("wagon_window") or {}
    for a, b in (("start_frame", "end_frame"),
                 ("wagon_start_frame", "wagon_end_frame"),
                 ("first_wagon_start_frame", "last_wagon_end_frame")):
        if win.get(a) is not None and win.get(b) is not None:
            return int(win[a]), int(win[b])
    return None, None


def train_state_at(frame: int, start: Optional[int], end: Optional[int]) -> str:
    if start is None or end is None:
        return UNKNOWN_STATE
    if frame < start:
        return PRE_WAGON
    if frame > end:
        return POST_WAGON
    return WAGON_ACTIVE


# ---------------------------------------------------------------------------
# reading the run
# ---------------------------------------------------------------------------

def accepted_times_by_camera(state: Dict[str, Any]) -> Dict[str, List[float]]:
    """Sorted accept times per camera, for the nearest-valid-gap columns.

    The master camera's accepted gaps are the global gaps (fixed-master
    architecture), so they are read from the global sequence; support cameras
    come from their own validation statistics where available.
    """
    out: Dict[str, List[float]] = {}
    master = state.get("master_camera") or "RIGHT_UP"
    for g in state.get("global_gaps") or []:
        cam = g.get("master_camera") or g.get("camera_id") or master
        # 'master_time' is the fixed-master field name; the others are older
        # spellings kept so an earlier run's JSON still resolves neighbours.
        t = g.get("master_time", g.get("time_start", g.get("start_time")))
        if t is None:
            continue
        out.setdefault(cam, []).append(float(t))
    stats = state.get("gap_validation_statistics") or {}
    for cam, s in stats.items():
        if not isinstance(s, dict):
            continue
        times = [float(t) for t in (s.get("accepted_times") or [])]
        if times and cam not in out:
            out[cam] = times
    for cam in out:
        out[cam].sort()
    return out


def recovery_index(state: Dict[str, Any]) -> Dict[tuple, Dict[str, Any]]:
    """(camera, track_id) -> recovery detail, so each row can say what happened."""
    idx: Dict[tuple, Dict[str, Any]] = {}
    rec = state.get("wagon_active_recovery") or {}
    blocks = rec.values() if isinstance(rec, dict) else rec
    for block in blocks:
        if not isinstance(block, dict):
            continue
        cam = block.get("camera_id", "")
        for d in block.get("details") or []:
            tid = d.get("track_id")
            if tid is not None:
                idx[(cam, int(tid))] = d
    return idx


def rejection_rows(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten every rejected candidate into one row per candidate."""
    start, end = wagon_window_frames(state)
    accepts = accepted_times_by_camera(state)
    recov = recovery_index(state)
    details = state.get("gap_rejection_details") or {}

    rows: List[Dict[str, Any]] = []
    for cam, entries in sorted(details.items()):
        for r in entries or []:
            f = r.get("features") or {}
            fs = int(f.get("frame_start", -1))
            reason = r.get("reason", "")
            hard = r.get("hard")
            if hard is None:                     # older JSON: classify by name
                hard = reason in HARD_REJECTION_REASONS
            soft = r.get("soft")
            if soft is None:
                soft = reason in SOFT_REJECTION_REASONS

            times = accepts.get(cam, [])
            t0 = float(f.get("time_start", 0.0))
            i = bisect_left(times, t0)
            prev_t = times[i - 1] if i > 0 else None
            next_t = times[i] if i < len(times) else None

            det = recov.get((cam, int(f.get("track_id", -1))))
            if det is None:
                recovery = "-"
            else:
                outcome = det.get("outcome", "?")
                recovery = (outcome if outcome != "blocked"
                            else "blocked: " + str(det.get("note", ""))[:60])

            rows.append({
                "camera": cam,
                "track_id": f.get("track_id"),
                "frames": f"{fs}-{f.get('frame_end')}",
                "frame_start": fs,
                "time_s": round(t0, 2),
                "duration_s": f.get("duration_s"),
                "hits": f.get("hits"),
                "coverage": f.get("coverage"),
                "confidence": f.get("mean_confidence"),
                "min_conf": f.get("min_confidence"),
                "displacement_px": f.get("displacement_px"),
                "speed_px_s": f.get("velocity_px_per_sec"),
                "ref_speed_px_s": f.get("motion_reference_speed"),
                "ref_kind": f.get("motion_reference_kind") or "-",
                "motion_paused": f.get("motion_paused"),
                "direction": f.get("direction"),
                "monotonic": f.get("monotonic_fraction"),
                "path_efficiency": f.get("path_efficiency"),
                "max_det_gap": f.get("max_detection_gap"),
                "reason": reason,
                "class": "HARD" if hard else ("SOFT" if soft else "?"),
                "train_state": train_state_at(fs, start, end),
                "nearest_prev_s": None if prev_t is None else round(prev_t, 2),
                "nearest_next_s": None if next_t is None else round(next_t, 2),
                "recovery": recovery,
                "detail": r.get("detail", ""),
            })
    rows.sort(key=lambda r: (r["camera"], r["frame_start"]))
    return rows


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

COLUMNS = [
    ("track_id", "track", 6), ("frames", "frames", 13), ("time_s", "t(s)", 8),
    ("duration_s", "dur(s)", 7), ("confidence", "conf", 6),
    ("displacement_px", "disp(px)", 9), ("speed_px_s", "speed", 8),
    ("ref_speed_px_s", "ref_spd", 8), ("ref_kind", "ref", 7),
    ("direction", "dir", 4), ("monotonic", "mono", 6),
    ("path_efficiency", "patheff", 8), ("class", "class", 6),
    ("reason", "reason", 30), ("train_state", "train_state", 13),
    ("nearest_prev_s", "prev(s)", 8), ("nearest_next_s", "next(s)", 8),
    ("recovery", "recovery", 24),
]


def _cell(v: Any, w: int) -> str:
    if v is None:
        s = "-"
    elif isinstance(v, float):
        s = f"{v:.2f}" if abs(v) < 1000 else f"{v:.0f}"
    elif isinstance(v, bool):
        s = "yes" if v else "no"
    else:
        s = str(v)
    if len(s) > w:
        s = s[: w - 1] + "…"
    return s.ljust(w)


def render_table(rows: Sequence[Dict[str, Any]], out=sys.stdout) -> None:
    if not rows:
        out.write("No rejected candidates recorded -- every tracked candidate "
                  "was validated.\n")
        return
    header = " ".join(lbl.ljust(w) for _, lbl, w in COLUMNS)
    for cam in sorted({r["camera"] for r in rows}):
        cam_rows = [r for r in rows if r["camera"] == cam]
        out.write(f"\n{'=' * len(header)}\nCAMERA {cam} "
                  f"-- {len(cam_rows)} rejected candidate(s)\n"
                  f"{'=' * len(header)}\n{header}\n{'-' * len(header)}\n")
        for r in cam_rows:
            out.write(" ".join(_cell(r[k], w) for k, _, w in COLUMNS) + "\n")


def render_stitching(state: Dict[str, Any], out=sys.stdout) -> None:
    """What fragment reassembly did, per camera.

    Reported alongside the rejections because the two are complementary: a gap
    lost to fragmentation shows up here as a seam that was NOT joined, not as a
    rejection with an interesting reason.
    """
    blocks = state.get("fragment_stitching") or {}
    if not blocks:
        return
    out.write("\n" + "=" * 78 + "\nFRAGMENT REASSEMBLY\n" + "=" * 78 + "\n")
    for cam, b in sorted(blocks.items()):
        if not isinstance(b, dict):
            continue
        out.write(f"\n{cam}: {b.get('input_candidates')} candidate(s) -> "
                  f"{b.get('output_candidates')}   "
                  f"reassembled={b.get('reassembled_gaps')}  "
                  f"fragments_absorbed={b.get('fragments_absorbed')}\n")
        adv = b.get("reference_advance_px_per_frame")
        if adv:
            out.write(f"  reference advance: {adv} px/frame   "
                      f"dominant direction: {b.get('dominant_direction')}\n")
        for c in b.get("chains") or []:
            out.write(f"    tracks {c.get('member_track_ids')} -> frames "
                      f"{c.get('start_frame')}-{c.get('end_frame')} "
                      f"({c.get('hit_count')} hits)\n")
        refused = b.get("rejected_seams") or []
        if refused:
            out.write(f"  {len(refused)} seam(s) considered and refused; "
                      f"reasons:\n")
            counts: Dict[str, int] = {}
            for s in refused:
                why = str(s.get("refused_because", "?"))
                key = why.split(" -- ")[0][:70]
                counts[key] = counts.get(key, 0) + 1
            for why, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                out.write(f"      {n:>4}  {why}\n")


def render_summary(state: Dict[str, Any], rows: Sequence[Dict[str, Any]],
                   out=sys.stdout) -> None:
    start, end = wagon_window_frames(state)
    out.write("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78 + "\n")
    out.write(f"wagon window      : frames {start} .. {end}\n")
    out.write(f"global gaps       : {len(state.get('global_gaps') or [])}\n")
    out.write(f"global wagons     : {state.get('master_wagon_count')}\n")

    states = (PRE_WAGON, WAGON_ACTIVE, POST_WAGON, UNKNOWN_STATE)
    for cam in sorted({r["camera"] for r in rows}):
        cam_rows = [r for r in rows if r["camera"] == cam]
        out.write(f"\n{cam}\n")
        for st in states:
            sub = [r for r in cam_rows if r["train_state"] == st]
            if not sub:
                continue
            hard = sum(1 for r in sub if r["class"] == "HARD")
            soft = len(sub) - hard
            got = sum(1 for r in sub if r["recovery"] == "recovered")
            out.write(f"  {st:<12} rejected={len(sub):<4} hard={hard:<4} "
                      f"soft={soft:<4} recovered={got}\n")
        by_reason: Dict[str, int] = {}
        for r in cam_rows:
            by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
        for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            out.write(f"      {reason:<34} {n}\n")

    lost = [r for r in rows
            if r["class"] == "SOFT" and r["train_state"] == WAGON_ACTIVE
            and r["recovery"] != "recovered"]
    out.write("\n")
    if lost:
        out.write(f"!! {len(lost)} SOFT rejection(s) inside WAGON_ACTIVE were NOT "
                  f"recovered. These are the candidates to inspect first --\n"
                  f"   each one is a potential lost wagon boundary:\n")
        for r in lost:
            out.write(f"   {r['camera']} track {r['track_id']} @ {r['time_s']}s "
                      f"({r['reason']}) -> {r['recovery']}\n")
    else:
        out.write("No unrecovered SOFT rejections inside WAGON_ACTIVE: no "
                  "candidate was lost to a soft gate in the counted region.\n")


def write_csv(rows: Sequence[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Render the per-candidate rejection table for a run.")
    p.add_argument("json_path",
                   help="global_train_state.json from a completed run")
    p.add_argument("--csv", help="also write the rows to this CSV path")
    p.add_argument("--camera", action="append", default=None,
                   help="restrict to a camera (repeatable)")
    p.add_argument("--soft-only", action="store_true",
                   help="only SOFT rejections (the recoverable class)")
    p.add_argument("--wagon-active-only", action="store_true",
                   help="only candidates inside the counted wagon region")
    p.add_argument("--no-summary", action="store_true")
    a = p.parse_args(argv)

    try:
        with open(a.json_path, encoding="utf-8") as fh:
            state = json.load(fh)
    except FileNotFoundError:
        print(f"error: no such run JSON: {a.json_path}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"error: {a.json_path} is not valid JSON: {exc}", file=sys.stderr)
        return 2

    rows = rejection_rows(state)
    if a.camera:
        want = {c.upper() for c in a.camera}
        rows = [r for r in rows if r["camera"].upper() in want]
    if a.soft_only:
        rows = [r for r in rows if r["class"] == "SOFT"]
    if a.wagon_active_only:
        rows = [r for r in rows if r["train_state"] == WAGON_ACTIVE]

    render_table(rows)
    if not a.no_summary:
        render_stitching(state)
        render_summary(state, rows)
    if a.csv:
        write_csv(rows, a.csv)
        print(f"\nwrote {len(rows)} row(s) to {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
