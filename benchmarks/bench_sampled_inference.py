#!/usr/bin/env python3
"""Door/Damage sampled-inference benchmark.

REUSES AN EXISTING Stage-1 RESULT. It never runs Stage 1, never runs
reconstruction, and never touches wagon_count/. Point it at a completed batch
directory and it re-runs only Stage 3 for the combinations below.

    python benchmarks/bench_sampled_inference.py \
        --batch-dir batch_outputs/20260816_172727 \
        --feat-models-dir models/features \
        --out /tmp/bench

Combinations (per the review plan):

    baseline  door legacy            + damage legacy
    test1     door sampled stride=2  + damage legacy
    test2     door legacy            + damage sampled stride=2
    test3     door sampled stride=2  + damage sampled stride=2
    test4     door sampled stride=2  + damage sampled stride=3

Each arm writes its per-wagon JSON to its own output tree, so nothing
overwrites the baseline. A per-wagon diff against the baseline is printed at
the end -- every disagreement is listed, none are summarised away.

Load runs ONCE up front (legacy, unmodified) and its output tree is shared by
every arm, so the loaded-wagon floor-damage suppression behaves identically in
all of them and is never bypassed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import constants as C                                    # noqa: E402
from core.global_state_loader import load_global_train_state       # noqa: E402
from features.damage import processor as damage_proc               # noqa: E402
from features.door import processor as door_proc                   # noqa: E402
from features.load import processor as load_proc                   # noqa: E402

ARMS: List[Dict[str, Any]] = [
    {"name": "baseline", "door": ("legacy", 1),  "damage": ("legacy", 1)},
    {"name": "test1",    "door": ("sampled", 2), "damage": ("legacy", 1)},
    {"name": "test2",    "door": ("legacy", 1),  "damage": ("sampled", 2)},
    {"name": "test3",    "door": ("sampled", 2), "damage": ("sampled", 2)},
    {"name": "test4",    "door": ("sampled", 2), "damage": ("sampled", 3)},
]


def _read_states(root: str, feature: str) -> Dict[str, Dict[str, Any]]:
    d = os.path.join(root, feature)
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            with open(os.path.join(d, fn), "r", encoding="utf-8") as f:
                out[fn[:-5]] = json.load(f)
    return out


def _door_view(p: Dict[str, Any]) -> Tuple:
    return (p.get("status"), p.get("left_door"), p.get("right_door"),
            round(float(p.get("left_door_confidence") or 0.0), 3),
            round(float(p.get("right_door_confidence") or 0.0), 3))


def _damage_view(p: Dict[str, Any]) -> Tuple:
    return (p.get("status"), p.get("top_damage"),
            len(p.get("top_damage_details") or []))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-dir", required=True,
                    help="existing batch_outputs/<key>/ (must already contain "
                         "global_state/ and wagon_cache/ -- Stage 1 is NOT run)")
    ap.add_argument("--feat-models-dir", default=os.path.join(_ROOT, "models", "features"))
    ap.add_argument("--out", default=None, help="benchmark output root")
    ap.add_argument("--arms", default="", help="comma-separated subset of arm names")
    ap.add_argument("--limit-wagons", type=int, default=0,
                    help="benchmark only the first N wagons (0 = all)")
    args = ap.parse_args(argv)

    batch_dir = os.path.abspath(args.batch_dir)
    state_path = os.path.join(batch_dir, "global_state", "global_train_state.json")
    cache_root = os.path.join(batch_dir, "wagon_cache")
    for p in (state_path, cache_root):
        if not os.path.exists(p):
            print(f"ERROR: missing {p} -- run the pipeline once first; this "
                  f"script never runs Stage 1.", file=sys.stderr)
            return 2

    state = load_global_train_state(state_path)          # frozen Stage-1 roster
    if args.limit_wagons > 0:
        import dataclasses
        state = dataclasses.replace(
            state, wagons=state.wagons[:args.limit_wagons],
            total_wagons=min(args.limit_wagons, state.total_wagons))
    print(f"Stage-1 roster REUSED (not recomputed): {state.total_wagons} wagons "
          f"{state.wagons[0].global_id}..{state.wagons[-1].global_id}")

    out_root = os.path.abspath(args.out or os.path.join(batch_dir, "_bench"))
    os.makedirs(out_root, exist_ok=True)

    # ---- Load once, legacy, shared by every arm -------------------------
    load_root = os.path.join(out_root, "_shared_load")
    if not os.path.isdir(os.path.join(load_root, "load")):
        os.makedirs(load_root, exist_ok=True)
        print("\nRunning Load ONCE (legacy, unmodified) -- shared by all arms "
              "so loaded-wagon floor suppression is identical everywhere")
        t = time.perf_counter()
        load_proc.run(state=state, cache_root=cache_root,
                      feature_models_dir=args.feat_models_dir,
                      output_dir=load_root, evidence_root=None, verbose=False)
        print(f"  load: {time.perf_counter() - t:.1f}s")

    wanted = {a.strip() for a in args.arms.split(",") if a.strip()}
    results: Dict[str, Dict[str, Any]] = {}

    for arm in ARMS:
        if wanted and arm["name"] not in wanted:
            continue
        name = arm["name"]
        arm_root = os.path.join(out_root, name)
        shutil.rmtree(arm_root, ignore_errors=True)
        os.makedirs(arm_root, exist_ok=True)
        # every arm sees the same Load result
        shutil.copytree(os.path.join(load_root, "load"),
                        os.path.join(arm_root, "load"))
        ev_root = os.path.join(arm_root, "evidence")

        d_mode, d_stride = arm["door"]
        g_mode, g_stride = arm["damage"]
        print(f"\n=== {name}: door={d_mode}/{d_stride}  damage={g_mode}/{g_stride} ===")

        t0 = time.perf_counter()
        door_proc.run(state=state, cache_root=cache_root,
                      feature_models_dir=args.feat_models_dir,
                      output_dir=arm_root, evidence_root=ev_root,
                      verbose=False, inference_mode=d_mode,
                      sample_stride=d_stride)
        t_door = time.perf_counter() - t0

        t0 = time.perf_counter()
        damage_proc.run(state=state, cache_root=cache_root,
                        feature_models_dir=args.feat_models_dir,
                        output_dir=arm_root, evidence_root=ev_root,
                        verbose=False, inference_mode=g_mode,
                        sample_stride=g_stride)
        t_damage = time.perf_counter() - t0

        door_states = _read_states(arm_root, "door")
        dmg_states = _read_states(arm_root, "damage")
        door_calls = sum(int(p.get("frame_count") or 0) for p in door_states.values())
        dmg_calls = sum(int(p.get("frame_count") or 0) for p in dmg_states.values())
        no_data = sum(1 for p in door_states.values()
                      if p.get("left_door") == C.NO_DATA
                      or p.get("right_door") == C.NO_DATA)

        results[name] = {
            "door_s": t_door, "damage_s": t_damage,
            "stage3_serial_s": t_door + t_damage,
            "door_calls": door_calls, "damage_calls": dmg_calls,
            "door": door_states, "damage": dmg_states, "no_data": no_data,
        }
        print(f"  door   {t_door:8.1f}s  yolo_calls={door_calls}")
        print(f"  damage {t_damage:8.1f}s  yolo_calls={dmg_calls}")

    # ---- summary --------------------------------------------------------
    print(f"\n{'='*92}")
    print(f"{'arm':<10}{'door_s':>10}{'damage_s':>11}{'serial_s':>11}"
          f"{'door_calls':>12}{'dmg_calls':>11}{'NO_DATA':>9}")
    base = results.get("baseline")
    for name, r in results.items():
        print(f"{name:<10}{r['door_s']:>10.1f}{r['damage_s']:>11.1f}"
              f"{r['stage3_serial_s']:>11.1f}{r['door_calls']:>12}"
              f"{r['damage_calls']:>11}{r['no_data']:>9}")
    if base:
        print(f"\n{'arm':<10}{'door speedup':>14}{'damage speedup':>16}"
              f"{'door calls saved':>18}{'dmg calls saved':>17}")
        for name, r in results.items():
            if name == "baseline":
                continue
            print(f"{name:<10}{base['door_s']/max(r['door_s'],1e-9):>13.2f}x"
                  f"{base['damage_s']/max(r['damage_s'],1e-9):>15.2f}x"
                  f"{base['door_calls']-r['door_calls']:>18}"
                  f"{base['damage_calls']-r['damage_calls']:>17}")

    # ---- per-wagon diff -------------------------------------------------
    if base:
        for name, r in results.items():
            if name == "baseline":
                continue
            print(f"\n{'-'*92}\nPER-WAGON DIFF  {name}  vs baseline")
            ndiff = 0
            for gw in sorted(base["door"], key=lambda s: int(s.split("_")[1])):
                bd, nd = base["door"].get(gw, {}), r["door"].get(gw, {})
                bg, ng = base["damage"].get(gw, {}), r["damage"].get(gw, {})
                d_same = _door_view(bd) == _door_view(nd)
                g_same = _damage_view(bg) == _damage_view(ng)
                if d_same and g_same:
                    continue
                ndiff += 1
                print(f"  {gw}")
                if not d_same:
                    print(f"    Door LEFT :  {bd.get('left_door')} "
                          f"({float(bd.get('left_door_confidence') or 0):.3f})"
                          f"  ->  {nd.get('left_door')} "
                          f"({float(nd.get('left_door_confidence') or 0):.3f})")
                    print(f"    Door RIGHT:  {bd.get('right_door')} "
                          f"({float(bd.get('right_door_confidence') or 0):.3f})"
                          f"  ->  {nd.get('right_door')} "
                          f"({float(nd.get('right_door_confidence') or 0):.3f})")
                if not g_same:
                    print(f"    Damage    :  {bg.get('top_damage')} "
                          f"[{len(bg.get('top_damage_details') or [])} tracks]"
                          f"  ->  {ng.get('top_damage')} "
                          f"[{len(ng.get('top_damage_details') or [])} tracks]")
            print(f"  wagons differing: {ndiff}/{len(base['door'])}"
                  f"   ({'IDENTICAL' if ndiff == 0 else 'SEE ABOVE'})")

    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({k: {kk: vv for kk, vv in v.items()
                       if kk not in ("door", "damage")}
                   for k, v in results.items()}, f, indent=2)
    print(f"\nwrote {os.path.join(out_root, 'summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
