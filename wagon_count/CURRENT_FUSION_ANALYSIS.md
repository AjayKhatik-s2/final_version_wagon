# CURRENT_FUSION_ANALYSIS

**Phase 1 deliverable — analysis of the existing implementation. No code has been
changed.**

Scope: how the current system creates global gaps, exactly which code path lets
support cameras inflate the wagon count, and which functions must change versus
stay untouched.

Companion documents: `WAGON_COUNT_LOGIC.md` (full behavioural reference),
`GLOBAL_FUSION_DESIGN.md` (Phase 2 proposal).

All numbers below are measured from the completed run in `results/`.

---

## 1. FILES INSPECTED

| File | Relevance to counting |
|---|---|
[run_global_count.py](run_global_count.py) | orchestration, CLI, fusion config assembly |
[tracker_engine.py](tracker_engine.py) | detection + tracking + classification (produces the inputs) |
[global_alignment.py](global_alignment.py) | **the fusion stage — where the count is decided** |
[global_train_state.py](global_train_state.py) | data contracts, `total_wagons`, `supporting_cameras` |
[video_segmenter.py](video_segmenter.py) | master→local projection, overlay renderer |
[evidence_report.py](evidence_report.py) | PDF report (downstream, reporting only) |
[requirements.txt](requirements.txt), [README.md](README.md), [PIPELINE_WORKFLOW_WAGON_COUNT.txt](PIPELINE_WORKFLOW_WAGON_COUNT.txt), [WAGON_COUNT_LOGIC.md](WAGON_COUNT_LOGIC.md) | documentation / environment |

**Tests present in the repository: none.** There is no `tests/` directory, no
`test_*.py`, no pytest configuration. This is the first thing the new work adds.

---

## 2. WHERE EACH CONCERN LIVES (repository-wide search results)

| Concern | Location |
|---|---|
| Detects gaps | `tracker_engine.GapTracker._detect_gaps` (YOLO + 3 filters) |
| Tracks gaps | `tracker_engine.GapTracker.process_video` (`_KF1D` + greedy NN + hit/miss) |
| Emits per-camera gap list | same → `LocalCameraTracks.gaps: List[GapEvent]` |
| **Creates global gaps** | `global_alignment.fuse_master_timeline` [global_alignment.py:270-288](global_alignment.py#L270-L288) |
| **Merges camera gaps** | `global_alignment.match_support_to_master` [global_alignment.py:98](global_alignment.py#L98) |
| **Temporal matching** | same (±1.0 s window / temporal IoU ≥ 0.2) |
| **Camera synchronization** | **DOES NOT EXIST** — only `t = frame / fps` |
| **Support-camera insertion** | `global_alignment.decide_inserted_gaps` [global_alignment.py:175](global_alignment.py#L175) |
| **Quorum logic** | `insert_min_support` [global_alignment.py:74](global_alignment.py#L74) |
| **Creates corrections** | `decide_inserted_gaps` → `GapCorrection` |
| Creates GW IDs | `global_alignment.build_global_wagons` [global_alignment.py:296](global_alignment.py#L296) |
| Calculates `total_wagons` | `assemble_global_train_state` → `GlobalTrainState(total_wagons=len(wagons))` [global_alignment.py:475](global_alignment.py#L475) |
| Creates wagon segments | `build_global_wagons` (boundaries → segments) |
| Generates `combined_report.pdf` | `evidence_report.build_combined_report` |
| Generates processed videos | `video_segmenter.render_processed_video` |
| Master→local projection | `video_segmenter.map_global_wagon_to_local_frames` [video_segmenter.py:70](video_segmenter.py#L70) |

---

## 3. CURRENT FUSION ALGORITHM, STEP BY STEP

Entry point: `assemble_global_train_state(master_tracks, support_tracks,
initial_classifications, config, verbose)` [global_alignment.py:419](global_alignment.py#L419).

Config comes from `PHASE1_DEFAULTS` [global_alignment.py:68](global_alignment.py#L68);
`run_global_count.py` [run_global_count.py:412-417](run_global_count.py#L412-L417)
overrides exactly three keys (`insert_min_support`, `insert_max_spread_sec`,
`insert_min_confidence`). The other three are not reachable from the CLI.

### Step 1 — `match_support_to_master` (per support camera)

For each support gap, scan **every** master gap:

```python
iou        = temporal IoU of the two [start_time, end_time] intervals
dt         = abs(support.center_time - master.center_time)
time_score = max(0.0, 1.0 - dt / match_time_window_sec)      # window = 1.0 s
score      = max(iou, time_score) if (iou >= 0.2 or dt <= 1.0) else -1.0
```

Highest score wins; matched if `best_master_id >= 0 and best_score >= 0.0`.
Otherwise the support gap goes into `leftover`.

Three structural properties:

* **No monotonicity constraint.** Support gap *j* may match master *i* while
  *j+1* matches master *i−1*. Train order can be violated and nothing detects it.
* **No one-to-one constraint.** `matched` is a plain dict
  `support.track_id → master.track_id`; several support gaps may claim the same
  master gap.
* **A matched support gap is then discarded.** It never refines the master
  boundary. Matching exists *only* to decide what is left over.

### Step 2 — `cluster_unmatched_supports`

All leftovers from all cameras pooled, sorted by
`(center_time, camera_id, track_id)`, swept with a **running-mean** centre and a
radius of `insert_max_spread_sec` (1.5 s).

### Step 3 — `decide_inserted_gaps` — **the insertion rule**

A cluster becomes a new global gap if all four pass:

| # | Test | Constant | Default |
|---|---|---|---|
| 1 | `len({g.camera_id for g in cluster}) >= min_support` | `insert_min_support` | **2** |
| 2 | `centers[-1] - centers[0] <= max_spread_sec` | `insert_max_spread_sec` | 1.5 s |
| 3 | `mean(confidence) >= min_confidence` | `insert_min_confidence` | 0.4 |
| 4 | `min(|center − mc| for mc in master_centers) >= min_distance` | `insert_min_distance_to_master_sec` | 1.0 s |

Note test 4 compares only against **master** centres, never against
previously-accepted inserts.

### Step 4 — `fuse_master_timeline` — **synthetic gap creation**

[global_alignment.py:270-288](global_alignment.py#L270-L288):

```python
next_synth_id = -1
for c in inserts:
    f = c.inserted_at_master_frame
    synth_gaps.append(GapEvent(
        track_id=next_synth_id,                              # negative id
        camera_id=f"FUSED({'+'.join(c.supporting_cameras)})", # no RIGHT_UP source
        start_frame=max(0, f - 1), end_frame=f + 1,
        confidence=c.mean_confidence, fps=master_fps,
        class_label="gap_inserted",
    ))
    next_synth_id -= 1
fused = sorted(master_gaps + synth_gaps, key=lambda g: g.center_time)
```

**This is the exact line that violates the required invariant:**
`master_gaps + synth_gaps`. A `GapEvent` with no RIGHT_UP provenance is
concatenated into the authoritative timeline and is thereafter indistinguishable
from a real master gap to every downstream consumer.

### Step 5 — `build_global_wagons` — boundaries → wagons → GW IDs

```python
boundaries = sorted(clamp(round(g.center_frame), 0, total-1) for g in fused_gaps)
segs, prev = [], 0
for b in boundaries:
    if b <= prev: continue          # ONLY duplicate suppression in the pipeline
    segs.append((prev, b - 1)); prev = b
if prev <= total - 1: segs.append((prev, total - 1))
wagons = [GlobalWagon(global_id=f"GW_{i}", ...) for i, (sf, ef) in enumerate(segs, 1)]
```

`total_wagons = len(wagons)`. **Nothing filters this list afterwards.**

### Step 6 — Fallback

Two triggers: an exception inside fusion/build, or `if not wagons:`. Both call
`build_wagons_pure_master` — which is already exactly the behaviour the new
architecture wants as its *normal* path. Measured: `fallback_used: false`.

---

## 4. EXACT SOURCE OF THE OVERCOUNT

**The causal chain, with the measured numbers from `results/`:**

```
(1) No camera offset estimation exists  (t = frame / fps only)
        |
        v
(2) Support gap times are compared to master times on a false common clock
    within a +/-1.0 s window
        |
        v
(3) 60 of 111 support gaps (54%) fail to match  ->  `leftover`
        |
        v
(4) `leftover` is INTERPRETED AS EVIDENCE OF A GAP THE MASTER MISSED
        |
        v
(5) `decide_inserted_gaps`: any 2 leftovers from distinct cameras within 1.5 s
    and >= 1.0 s from a master gap are accepted     -> 45 clusters -> 11 accepted
        |
        v
(6) `fuse_master_timeline`: master_gaps + synth_gaps                (line 288)
        |
        v
(7) each synthetic gap adds one boundary -> one extra wagon
        |
        v
    52 master gaps + 11 synthetic = 63 boundaries -> 64 wagons
    (the invariant requires 52 -> 53)
```

**Step (4) is the conceptual error.** "This support detection does not line up
with any master gap" is being read as "the master missed a gap", when the far
more likely explanation — given no offset model exists — is "the clocks are not
aligned". The measured offsets are large: agreement with the master improves
sharply under a constant shift (LEFT_UP: 2/34 gaps within ±0.5 s of a master gap
at offset 0, 19/34 at ≈ +16 s). `match_time_window_sec` is 1.0 s — an order of
magnitude too small for that.

### Evidence that the inserted gaps are largely spurious

Classifying each of the 11 insertions by whether the interval it split was
genuinely too long — ratio of the interval to the **local** median interval
(the train's speed drifts from ~3.7 s to ~6.6 s spacing, so a local baseline is
required):

| insert t (s) | interval | length | local base | ratio | assessment |
|---|---|---|---|---|---|
| 26.72 | G5→G6 | 3.667 | 4.033 | 0.91 | **splits a normal wagon** |
| 82.17 | G14→G15 | 11.867 | 6.400 | 1.85 | interval genuinely long |
| 106.57 | G17→G18 | 8.533 | 6.233 | 1.37 | **borderline** |
| 110.17 | G17→G18 | 8.533 | 6.233 | 1.37 | **second insert in the same interval** |
| 114.10 | G18→G19 | 3.767 | 4.733 | 0.80 | **splits a normal wagon** |
| 122.20 | G20→G21 | 7.867 | 3.933 | 2.00 | interval genuinely long |
| 157.70 | G28→G29 | 7.933 | 3.867 | 2.05 | interval genuinely long |
| 168.55 | G30→G31 | 3.900 | 3.900 | 1.00 | **splits a normal wagon** |
| 180.45 | G33→G34 | 7.533 | 4.067 | 1.85 | interval genuinely long |
| 193.55 | G35→G36 | 7.800 | 4.167 | 1.87 | interval genuinely long |
| 203.02 | G37→G38 | 4.167 | 4.433 | 0.94 | **splits a normal wagon** |

**6 of 11 insertions split a normal-length wagon in two**, and one interval
received two insertions. Under the required architecture all 11 are
inadmissible regardless, because none has a RIGHT_UP source.

### Secondary defect: `supporting_cameras` is fabricated

[global_alignment.py:349](global_alignment.py#L349):

```python
supporting_cameras=[MASTER_CAMERA] + [c for c in support_camera_ids if c != MASTER_CAMERA]
```

This is a **static list**, identical for every wagon, computed without looking at
any observation. Measured: all 64 wagons claim all four cameras — including
wagons for which `LEFT_UP_TOP` has no footage at all. Any consumer treating this
field as evidence is being misled. It is the natural place to put the real
alignment result.

### Third defect: silent clamping fabricates evidence

`map_global_wagon_to_local_frames` [video_segmenter.py:70](video_segmenter.py#L70)
clamps an out-of-range projection to `local_total_frames - 1` and returns a
syntactically valid one-frame range. A caller cannot distinguish "the last frame"
from "this camera has no footage here". Only `evidence_report` guards against
this, with its own unclamped pre-check; the overlay renderer does not.

---

## 5. A FINDING THAT CONFLICTS WITH THE STATED INVARIANT

Reporting this because it is a measured fact, not to argue against the
architecture. The requirement states RIGHT_UP does not miss real gaps. RIGHT_UP's
own interval statistics suggest otherwise:

* **7 intervals are ≈ 1.7×–2.05× the local median** (G8→G9, G14→G15, G15→G16,
  G20→G21, G28→G29, G33→G34, G35→G36). A near-integer multiple of the local
  spacing is the signature of one missed boundary each.
* **1 interval is 0.833 s against a local base of 5.967 s (ratio 0.14)** —
  `G51→G52`. The next-smallest interval in the entire train is 3.667 s, so this
  is an extreme outlier and looks like a duplicate/false master boundary. It is
  what produces the two spurious 0.27 s and 0.80 s `UNKNOWN` tail wagons
  (`GW_63`, `GW_64`).

**How the design handles this:** it obeys the invariant exactly — these are
**reported as diagnostics only** and never alter the count. RIGHT_UP's 52 gaps
stay 52 gaps. The flags simply make the situation visible so the master's
detection can be improved at the detection layer, which is where the requirement
says such work belongs.

Also note the arithmetic implication: the requirement's worked example is
"59 gaps → 60 wagons". On the run currently in `results/`, RIGHT_UP has
**52 final gaps → 53 wagons**. The 59 figure must come from a different run;
whichever run is used, the invariant is what fixes the answer.

---

## 6. FUNCTIONS THAT MUST CHANGE

| Function | File | Required change |
|---|---|---|
| `decide_inserted_gaps` | `global_alignment.py` | **must never run in the counting path.** No support-camera quorum may create a gap. |
| `cluster_unmatched_supports` | `global_alignment.py` | **not used for counting.** Leftovers become EXTRA diagnostics, never insert candidates. |
| `fuse_master_timeline` | `global_alignment.py` | the `master_gaps + synth_gaps` concatenation must go. The global sequence *is* `master.gaps`. |
| `match_support_to_master` | `global_alignment.py` | replaced by order-preserving DP alignment against the fixed master sequence. |
| `assemble_global_train_state` | `global_alignment.py` | new path: build the sequence from RIGHT_UP, then attach evidence; assert the invariant. |
| `GlobalWagon.supporting_cameras` | `global_train_state.py` | must reflect real matched observations instead of a static list. |
| `GlobalTrainState` | `global_train_state.py` | must carry global gaps with provenance, offsets, and EXTRA/MISSING diagnostics. |
| `render_processed_video` | `video_segmenter.py` | needs optional offset + validity range so support overlays land correctly and are not drawn out of range. Backward-compatible defaults. |
| `run_global_count.py` fusion block | `run_global_count.py` | call the new path; retire the three `--fuse-*` insertion knobs. |
| `evidence_report.py` | — | consume the real per-wagon support/unavailable data instead of re-deriving it. Count still unaffected. |

## 7. FUNCTIONS THAT MUST NOT CHANGE

| Function / value | Why |
|---|---|
| `GapTracker.__init__`, `process_video`, `_detect_gaps` | detection + tracking are out of scope and declared final |
| `_KF1D` (`process_var` 4.0, `meas_var` 9.0) | tracker parameters frozen |
| `match_distance_px` 80.0, `min_hits` 3, `max_miss` 30 | tracker parameters frozen |
| detection confidence 0.4; height ratios 0.35 / 0.05 | thresholds frozen |
| YOLO weights, model classes, the `"gap" in name` filter | frozen |
| `MasterClassifier`, `_label_to_class` | classification must not affect the count, and does not |
| `segments_from_gaps` | pre-fusion segmentation for classification, unchanged |
| `build_global_wagons` | **reused as-is** — with `fused_gaps = master.gaps` it already produces exactly the required result; the `b <= prev` rule and `GW_{i}` numbering stay |
| `GapEvent`, `LocalCameraTracks`, `_MasterClassification` | contracts stay compatible |
| `map_global_wagon_to_local_frames` | left intact for the existing renderer; an explicit-validity variant is added alongside |
| `N gaps -> N+1 wagons` | convention confirmed in code and preserved |
| CLI default `python run_global_count.py` | must keep working |
| `--render-videos`, `--no-report`, `--report-dpi`, `--keep-evidence-frames` | preserved |
| evidence 20/40/60/80% sampling, temp-frame cleanup | preserved |

---

## 8. THE ONE-LINE SUMMARY

The current code computes

```
total_wagons = master_gaps + accepted_support_insertions + 1     # 52 + 11 + 1 = 64
```

because unmatched support gaps — mostly unmatched due to a *missing offset
model*, not a missing master gap — are promoted into the authoritative timeline
at [global_alignment.py:288](global_alignment.py#L288).

The required behaviour is

```
total_wagons = master_gaps + 1                                   # 52 + 1 = 53
```

with support cameras contributing **association and evidence only**. The
`build_wagons_pure_master` path already computes this; the work is to make it the
normal path, to align support observations to it correctly (which needs the
offset model and the order-preserving alignment), and to assert the invariant so
a regression fails loudly.

---

*Analysis only. No source file, model, input video, requirement or output was
modified in producing this document. Nothing was committed or pushed.*
