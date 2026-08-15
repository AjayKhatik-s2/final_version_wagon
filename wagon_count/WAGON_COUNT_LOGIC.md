# WAGON_COUNT_LOGIC

**Source of truth for how the Global Wagon Count pipeline actually computes its
number.** Documentation only — this file describes the code as it exists today.
It proposes no changes and reflects no intended redesign.

Everything below was read out of the current source:
`run_global_count.py`, `tracker_engine.py`, `global_alignment.py`,
`global_train_state.py`, `video_segmenter.py`, `evidence_report.py`, plus
`README.md` and `PIPELINE_WORKFLOW_WAGON_COUNT.txt`. Concrete numbers come from
the latest completed run in `results/`. See **DOCUMENTATION CONFIDENCE** at the
end for what was verified versus what could not be.

> ## ⚠ THE FUSION STAGE HAS SINCE BEEN REPLACED
>
> This document describes the **legacy** fusion path (still reachable with
> `--fusion legacy`). The default is now `--fusion master-fixed`, implemented in
> `global_fusion.py`. What changed, and what did not:
>
> | | Legacy (described below) | Current default (`master-fixed`) |
> |---|---|---|
> | Global gap sequence | master gaps **+ synthetic support gaps** | **exactly** the RIGHT_UP gaps |
> | Support cameras may insert a gap | **yes** (quorum ≥ 2 within 1.5 s) | **no mechanism exists** |
> | Camera clock offsets | none — `t = frame/fps`, shared `t=0` assumed | estimated per camera, `RESOLVED`/`UNRESOLVED` |
> | Support matching | independent nearest-neighbour, order can cross | order-preserving DP; crossing is unrepresentable |
> | Unmatched support gap | promoted to a global gap | recorded as `EXTRA`, creates nothing |
> | Raw YOLO gap | is a wagon boundary | is a **candidate**; must pass motion/temporal validation |
> | Engines / brake vans | receive `GW_n` ids and are counted | preserved as metadata, **never counted, never given an id** |
> | Counted region | the whole video | the **wagon window**: first WAGON .. last WAGON |
> | Count on the run in `results/` | 52 + 11 + 1 = **64** | validated master gaps → segments → wagon window only |
> | `supporting_cameras` | static all-four list on every wagon | only cameras with a real matched observation |
> | Out-of-range projection | clamped to the last frame | reported unavailable; never clamped |
>
> **Unchanged by that work**, and therefore still described accurately below:
> sections 1 (detection, tracking, `GapEvent` emission), 2 (classification),
> the `build_global_wagons` boundary/segment rule, `GW_{i}` numbering, the
> `N gaps -> N+1 wagons` convention, and every detection/tracking threshold.
>
> Read `GLOBAL_FUSION_DESIGN.md` for the current fusion architecture and
> `CURRENT_FUSION_ANALYSIS.md` for why the legacy path inflated the count.
> Sections 2, 3, 4, 6 and 9 below remain the best description of the **legacy**
> behaviour and of the failure modes that motivated the replacement.

---

## 0. THE PROBLEM

Four fixed CCTV cameras watch the same train pass a point on the track:

| Camera | Role | View |
|---|---|---|
| `RIGHT_UP` | **master** — authoritative | side view, right |
| `LEFT_UP` | support | side view, left |
| `RIGHT_UP_TOP` | support | top-down, right |
| `LEFT_UP_TOP` | support | top-down, left |

Each camera independently sees the same physical wagons. The goal is **one**
global train representation — a single ordered sequence `GW_1 … GW_N` where the
same `GW_n` means the same physical wagon in every camera — rather than four
independent per-camera counts that disagree.

The core inference is **indirect**: the models do not detect wagons. They detect
the **gaps between wagons**. A wagon is the span of video *between* two
consecutive gaps. So:

```
N gaps on a timeline  ->  N+1 segments  ->  N+1 wagons
```

Everything else — tracking, synchronization, fusion, classification — exists to
produce one trustworthy gap timeline on the master clock.

**Separate concepts, deliberately kept distinct in this document** (the code
keeps them in separate functions and they fail in different ways):

1. **Detection** — per frame, YOLO finds gap bounding boxes.
2. **Tracking** — per camera, detections across frames are linked into one
   `GapEvent` per physical gap.
3. **Local counting** — per camera, `gaps + 1` wagons.
4. **Temporal alignment** — mapping master time to each camera's frame numbers.
5. **Fusion** — comparing support gap timelines against the master's.
6. **Correction / insertion** — adding a gap the master missed.
7. **Classification** — labelling a segment ENGINE / WAGON / BRAKE_VAN / UNKNOWN.
8. **Global ID generation** — assigning `GW_n`.

Only steps 1, 2, 5, 6 and the segment construction affect **the count**.
Classification does **not** change the count. Reporting does **not** change the
count.

---

## 1. EXECUTION ORDER (what `python run_global_count.py` actually does)

`main()` in `run_global_count.py`, in order:

| # | Stage | Function(s) |
|---|---|---|
| 0 | Parse CLI, resolve input/model paths | `_build_arg_parser`, `_resolve_input`, `_resolve_model` |
| 1 | Per-camera gap tracking (×4) | `_process_side_camera`, `_process_top_camera` → `GapTracker.process_video` |
| 1a | **Fragment reassembly** — rebuild one physical gap from the pieces the tracker split it into | `fstitch.reassemble_fragments` |
| 1b | **Gap validation** — candidates → valid boundaries | `gval.validate_gap_events` |
| 2 | Master classification | `_classify_master_pre_fusion` → `MasterClassifier` |
| 2a | Temporal classification (segment hysteresis) | `tcls.apply_temporal_classification` |
| 2b | Support-camera classification | `ts.load_segment_classifier` |
| 2c | WAGON_ACTIVE recovery (second validation pass) | `gval.recover_wagon_active_candidates` |
| 3 | Cross-camera fusion + global build | `gf.assemble_global_train_state_master_fixed` (or `ga.assemble_global_train_state` under `--fusion legacy`) |
| 4 | Write JSON | inline |
| 5 | Overlay videos — **opt-in**, `--render-videos` | `vs.render_processed_video` |
| 6 | Evidence report — **reporting only** | `er.build_combined_report` |
| 7 | Console summary, re-write JSON | `summarize_state` |

**The count is fully determined at the end of stage 3.** Stages 4–7 only
serialize, draw and report it.

Stages 1a and 1b are where a *detected and tracked* gap becomes a *counted*
boundary, and they are ordered deliberately:

- **1a decides which observations belong to the same physical object.** It never
  accepts a gap and relaxes no threshold. Merging requires evidence; refusing to
  merge leaves the fragments exactly as the tracker emitted them.
- **1b decides whether that object is a wagon boundary.** Because 1a ran first,
  every gate in 1b sees the whole gap rather than a piece of one.

Getting this order wrong is what caused a measured under-count: judged
piece-by-piece, each fragment of one real gap failed the minimum-duration gate,
and three wagon boundaries were lost.

---

### STAGE 0 — Path resolution

**Input:** CLI args (all optional).
**Logic:** `_here()` returns the directory of `run_global_count.py`; defaults are
`./inputs`, `./models`, `./results` relative to that file, so the project is
location-independent. `_resolve_input` tries, per camera, the names in
`_INPUT_FALLBACK_PATTERNS` (e.g. `right_up.mp4`, `RIGHT_UP.mp4`,
`cam_right_up.mp4`) and raises `FileNotFoundError` if none exist.
`_resolve_model` requires the exact filename.
**Output:** eight absolute paths.
**Fixed model→camera binding** (`main()`):

| Camera | Gap model | Detection conf | Min height ratio |
|---|---|---|---|
| `RIGHT_UP` | `right_up_wagon_gap.pt` | `--side-confidence` 0.4 | `--side-min-height-ratio` 0.35 |
| `LEFT_UP` | `left_up_wagon_gap.pt` | `--side-confidence` 0.4 | 0.35 |
| `RIGHT_UP_TOP` | `top_gap.pt` | `--top-confidence` 0.4 | `--top-min-height-ratio` 0.05 |
| `LEFT_UP_TOP` | `top_gap.pt` *(same file)* | 0.4 | 0.05 |

`side_classification.pt` is used **only** on `RIGHT_UP`, in stage 2.

The two top cameras share one model instance path but get **two separate
`GapTracker` objects**, so their tracks never mix.

---

### STAGE 1 — Detection and tracking, per camera

`GapTracker.process_video(video_path, frame_limit=0, keep_raw_detections=True)`
in `tracker_engine.py`. Called four times, sequentially, independently. There is
no cross-camera information at this stage.

#### 1a. Opening the video

`cv2.VideoCapture(video_path)`; reads `CAP_PROP_FPS`, `FRAME_WIDTH`,
`FRAME_HEIGHT`, `FRAME_COUNT`. **If `fps <= 0` it raises** — fps is load-bearing,
every later time value derives from it.

#### 1b. Frame iteration

A plain `while True: ret, frame = cap.read()` loop over **every frame** — no
sampling, no skipping, no resizing. `frame_limit` exists but the CLI never sets
it (default `0` = unlimited). `frame_idx` counts from 0.

At the end:

```python
total_frames = max(effective_frames, total_frames_meta if total_frames_meta > 0 else 0)
```

so a container that under-reports its frame count cannot shorten the timeline.

#### 1c. Per-frame detection — `_detect_gaps(frame, frame_h)`

`self.model(frame, verbose=False)[0]` — full-resolution frame, no letterboxing
done by this code, no augmentation. Three filters run in order, each with a
diagnostic counter (`_diag_total_yolo_boxes` → `_diag_after_class` →
`_diag_after_conf` → `_diag_kept`):

1. **Class filter.**
   ```python
   self._is_single_class_model = (len(self.class_names) == 1)
   ...
   if not self._is_single_class_model and "gap" not in name: continue
   ```
   *Measured model classes:*
   * `right_up_wagon_gap.pt` / `left_up_wagon_gap.pt` → **3 classes**:
     `engine_head`, `gap`, `locono`. Not single-class, so **only `gap`
     survives**.
   * `top_gap.pt` → **1 class**: `gaps`. Single-class, so the filter is skipped
     and *every* box is accepted regardless of class name.

   > **How engine/locomotive detections are handled:** the side gap models can
   > emit `engine_head` and `locono`, but this line **discards them**. They never
   > reach tracking, never become gaps, and never influence the count. Engine
   > identification happens **only** through `side_classification.pt` in stage 2.

2. **Confidence filter.** `if float(conf) < self.confidence: continue` — 0.4 for
   all four cameras by default.

3. **Height-ratio filter.** `h = box[3]-box[1]; if h < frame_h * min_height_ratio: continue`.
   0.35 for side cameras (a gap is a tall vertical slot), 0.05 for top cameras
   (a gap is a thin horizontal strip). Purely a bbox-height test — width is never
   checked.

Surviving detections become dicts `{bbox, confidence, center_x, height}` where
`center_x = (x1+x2)/2`. **`center_x` is the only geometric quantity tracking
uses.**

#### 1d. Tracking — hand-rolled, 1-D

Not ByteTrack/SORT/DeepSORT. It is a **constant-velocity Kalman filter on
`center_x` only**, plus greedy nearest-neighbour association and a hit/miss rule.

`_KF1D` (`tracker_engine.py`): state `[x, vx]`, transition `x+=vx`,
`P0 = 100·I`, `Q = 4.0·I` (`process_var`), `R = [[9.0]]` (`meas_var`). One filter
per track. No vertical or size modelling.

Per frame:

1. `tr.predicted_center()` → `kf.predict()` for every active track (mutates state).
2. Detections sorted by `center_x` (determinism).
3. For each detection, scan unused active tracks for the smallest
   `|cx − tr.kf.cx|`, accepted only if `<= match_distance_px` (**80.0 px**).
   * match → `tr.update(frame_idx, cx, conf, bbox)`; when `hit_count >= min_hits`
     (**3**) the track is marked `confirmed`.
   * no match → new `_Track` with `track_id = next_track_id++`.
4. Unmatched active tracks get `mark_miss()`.
5. A track with `miss_count >= max_miss` (**30 frames = 2.0 s at 15 fps**) is
   closed: **kept only if `confirmed`, otherwise silently dropped.**

After the loop, still-active confirmed tracks are flushed.

#### 1e. Emitting `GapEvent`s

```python
completed_tracks.sort(key=lambda t: (t.first_frame, t.last_seen_frame))
for new_id, tr in enumerate(completed_tracks, start=1):   # track_id RENUMBERED 1..N
```

so `GapEvent.track_id` is a **temporal rank within that camera**, not the id used
during tracking. Each `GapEvent` (`global_train_state.py`) carries:

`track_id`, `camera_id`, `start_frame`, `end_frame`, `confidence` (mean over the
track), `hit_count`, `center_x_trajectory`, `fps`,
`temporal_consistency_score = min(1, hit_count/span)`, `hit_frames`,
`bbox_history`, `class_label` (`"gap"`).

Derived properties — **this is where frames become time**:

```python
center_frame = (start_frame + end_frame) / 2
center_time  = center_frame / fps          # 0.0 if fps <= 0
start_time   = start_frame / fps
end_time     = end_frame   / fps
```

**Output:** `LocalCameraTracks(camera_id, video_path, fps, total_frames, width,
height, gaps=[GapEvent...], raw_frame_detections)`.

#### 1f. Local wagon count

```python
@property
def local_wagon_count(self):
    if self.total_frames <= 0: return 0
    return len(self.gaps) + 1
```

This is **only a diagnostic**. It is reported per camera but is *not* the global
count and is *not* used to compute it.

**Measured, latest run:**

| Camera | frames | fps | duration | gaps | local wagons |
|---|---|---|---|---|---|
| RIGHT_UP | 4140 | 15.0 | 276.0 s | **52** | 53 |
| LEFT_UP | 3690 | 15.0 | 246.0 s | 34 | 35 |
| RIGHT_UP_TOP | 4110 | 15.0 | 274.0 s | 50 | 51 |
| LEFT_UP_TOP | 3120 | 15.0 | 208.0 s | 27 | 28 |

---

### STAGE 2 — Master classification (does not affect the count)

`_classify_master_pre_fusion(master_tracks, side_cls_path, num_samples, verbose)`.

**Input:** `RIGHT_UP`'s `LocalCameraTracks` — **pre-fusion**, before any inserted
gap exists.

`segments_from_gaps(gaps, total_frames)` converts the master gap list into
inclusive `(start, end)` segments:

```python
boundaries = [clamp(round(g.center_frame)) for g in sorted(gaps, key=center_frame)]
prev = 0
for b in boundaries:
    if b <= prev: continue          # boundary at/behind the cursor is skipped
    segments.append((prev, b - 1)); prev = b
if prev <= total_frames - 1: segments.append((prev, total_frames - 1))
```

*Measured:* 52 master gaps → **53 pre-fusion segments**.

`MasterClassifier.classify_segments` then, per segment:

* `span = end-start+1`, `margin = max(1, int(span*0.1))` — the outer 10% at each
  end is skipped so the sample avoids the gap itself.
* If `safe_end <= safe_start` → one sample at the segment midpoint; if
  `num_samples == 1` → the midpoint of the safe range; otherwise
  `num_samples` (**5**) evenly spaced indices.
* Each index is fetched with `cap.set(cv2.CAP_PROP_POS_FRAMES, fi)` (**seeking**,
  unlike stage 1's sequential read) and classified by `classify_frame`.
* `classify_frame` prefers `results.probs` (classification head) and falls back to
  the highest-confidence box; if neither exists it returns `("wagon", 0.0)`.
* Majority vote: `max(counts.items(), key=lambda kv: (kv[1], -ord(kv[0][0])))` —
  count first, then alphabetically-earliest first letter as the tiebreak.
  Confidence returned is the mean confidence **of the winning label only**.

`_label_to_class` maps the raw string:

| Raw label set | → `SegmentClass` |
|---|---|
| `engine`, `loco`, `engine_head`, `locono`, `locomotive` | `ENGINE` |
| `tail`, `brake_van`, `brakevan`, `guard_van`, `wagon_tail` | `BRAKE_VAN` |
| `track`, `background`, `empty_track`, `rail`, `tracks` | `UNKNOWN` |
| anything else | `WAGON` |

*Measured `side_classification.pt`:* task `classify`, 4 classes —
`brakevan` → BRAKE_VAN, `empty_track` → **UNKNOWN**, `engine` → ENGINE,
`wagon` → WAGON. Every class the model can emit maps cleanly; no label falls
through to the `WAGON` default by accident.

**Output:** `List[_MasterClassification(segment_index, start_frame, end_frame,
label, confidence)]`.

**If classification raises**, `main()` catches it, prints a warning and continues
with `initial_classifications = []`. The run still produces the same count; every
wagon simply becomes `UNKNOWN`. **Classification failure cannot change the
number.**

---

### STAGE 3 — Fusion and global construction

`ga.assemble_global_train_state(master_tracks, support_tracks,
initial_classifications, config, verbose)`. This is where the count is decided.

`config` starts as `PHASE1_DEFAULTS` and `main()` overrides exactly three keys
from the CLI: `insert_min_support`, `insert_max_spread_sec`,
`insert_min_confidence`. The other three defaults are **not CLI-reachable**.

The whole of fusion + global build sits inside one `try`. See **Fallback** below.

#### 3a. `match_support_to_master(master_gaps, support_gaps, match_time_window_sec=1.0, match_min_iou=0.2)`

Run once per support camera. For each support gap, against every master gap:

```python
iou        = temporal IoU of [s_start,s_end] vs [m_start,m_end]   # seconds
dt         = abs(s_center_time - m_center_time)
time_score = max(0.0, 1.0 - dt / match_time_window_sec)
score      = max(iou, time_score) if (iou >= match_min_iou or dt <= match_time_window_sec) else -1.0
```

The master gap with the highest score wins; the support gap is **matched** if
`best_master_id >= 0 and best_score >= 0.0`. Effectively: *a support gap is
matched if some master gap is within **1.0 s** of it, or overlaps it with
temporal IoU ≥ **0.2***. Otherwise it goes to `leftover`.

`compute_temporal_iou` returns `(0.0, 0.0)` for any non-overlapping or
zero-length interval pair.

Two properties worth knowing:

* Matching is **many-to-one and unenforced** — `matched` is a dict
  `support.track_id → master.track_id`; several support gaps may claim the *same*
  master gap, and nothing detects that.
* A matched support gap is then **discarded**. It never reinforces, re-times or
  corrects the master gap it matched. Matching exists purely to decide what is
  *left over*.

**Measured, latest run:**

| Support camera | gaps | matched | leftover |
|---|---|---|---|
| LEFT_UP | 34 | 12 | **22** |
| RIGHT_UP_TOP | 50 | 27 | **23** |
| LEFT_UP_TOP | 27 | 12 | **15** |
| total | 111 | 51 | **60** |

#### 3b. `cluster_unmatched_supports(leftovers_per_camera, spread_sec)`

`spread_sec` is passed `cfg["insert_max_spread_sec"]` (**1.5 s** — the same knob
later used as the acceptance test). All leftovers from all cameras are pooled and
sorted by `(center_time, camera_id, track_id)`, then swept:

```python
if abs(g.center_time - cluster_center) <= spread_sec:
    current.append(g)
    cluster_center = mean(center_time of current)     # running mean, drifts
else:
    start a new cluster
```

Because the centre is a **running mean**, a chain of gaps 1.4 s apart can grow a
cluster whose total spread exceeds 1.5 s; that cluster is then rejected in 3c.

*Measured:* 60 leftovers → **45 clusters**.

#### 3c. `decide_inserted_gaps(...)` — the correction/insertion rule

A cluster becomes an inserted gap **only if all four tests pass**:

| # | Test | Constant | Default | Overridable |
|---|---|---|---|---|
| 1 | `len({g.camera_id for g in cluster}) >= min_support` — **distinct cameras**, not gap count | `insert_min_support` | **2** | `--fuse-min-support` |
| 2 | `centers[-1] - centers[0] <= max_spread_sec` | `insert_max_spread_sec` | **1.5 s** | `--fuse-max-spread` |
| 3 | `mean(g.confidence) >= min_confidence` | `insert_min_confidence` | **0.4** | `--fuse-min-conf` |
| 4 | `min(|center − mc| for mc in master_centers) >= min_distance_to_master_sec` | `insert_min_distance_to_master_sec` | **1.0 s** | **no** |

Test 4 guards against inserting a duplicate next to a master gap the matcher
already accounted for.

An accepted cluster produces a `GapCorrection(inserted_at_master_time=mean of
centers, inserted_at_master_frame=round(center*master_fps), supporting_cameras,
mean_confidence, time_spread_sec, contributing_track_ids)`. Results are sorted by
time.

*Measured:* 45 clusters → **11 inserted**.

#### 3d. `fuse_master_timeline` — building the corrected timeline

Each accepted correction becomes a **synthetic `GapEvent`**:

```python
track_id   = -1, -2, -3, …            # negative marks "fused, not observed by master"
camera_id  = "FUSED(LEFT_UP+LEFT_UP_TOP)"
start_frame= max(0, f-1);  end_frame = f+1     # so center_frame == f
fps        = master_fps
class_label= "gap_inserted"
fused      = sorted(master_gaps + synth_gaps, key=lambda g: g.center_time)
```

**Output:** `fused_gaps` — the corrected master gap timeline.
*Measured:* 52 real + 11 synthetic = **63 fused gaps**.

#### 3e. `build_global_wagons(...)` — gaps become wagons

```python
boundaries = sorted(clamp(round(g.center_frame), 0, master_total_frames-1) for g in fused_gaps)

segs, prev = [], 0
for b in boundaries:
    if b <= prev: continue                     # dedup / skip
    segs.append((prev, b - 1)); prev = b
if prev <= master_total_frames - 1:
    segs.append((prev, master_total_frames - 1))
```

* `if b <= prev: continue` is the **only** duplicate-suppression in the whole
  pipeline. Two boundaries on the *same* frame collapse into one. Two boundaries
  **one frame apart** do **not** collapse — they create a legitimate 1-frame
  segment, i.e. a whole extra wagon.
* A boundary at frame 0 is skipped (`0 <= 0`), so a gap at the very start does not
  create an empty leading wagon.
* The final `if prev <= total-1` always fires unless a boundary landed exactly on
  the last frame, so the tail of the video is always a wagon.

Then per segment, in order:

```python
for i, (sf, ef) in enumerate(segs, start=1):
    gw = GlobalWagon(global_id=f"GW_{i}", wagon_index=i,
                     start_frame_master=sf, end_frame_master=ef,
                     start_time=sf/master_fps, end_time=(ef+1)/master_fps,
                     classification=label, classification_confidence=conf,
                     supporting_cameras=[MASTER] + support_camera_ids)
```

* `classification` comes from `label_for_frame((sf+ef)//2)` — the **pre-fusion**
  classification whose `[start_frame, end_frame]` contains that midpoint; if none
  contains it, the nearest by frame distance; if the list is empty, `UNKNOWN`.
  So when an inserted gap splits a pre-fusion segment, **both children inherit
  the parent's label** (an ENGINE stays ENGINE; a merged WAGON becomes two
  WAGONs).
* `leading_gap` / `trailing_gap` record provenance: the last fused gap with
  `center_frame <= sf`, and the first with `center_frame > ef`. `source` is
  `"master"` when `track_id > 0`, `"fused"` when negative, else `"video_start"` /
  `"video_end"`.
* If either boundary is synthetic, `split_from_global_id = f"PRE_SEG_{idx}"` of
  the pre-fusion segment containing `sf`. *Measured:* 21 wagons carry a
  `split_from_global_id`.

> **`supporting_cameras` is a static list, not evidence.** It is
> `[MASTER_CAMERA] + [every support camera id]` for **every** wagon,
> unconditionally. It does **not** mean those cameras actually observed that
> wagon. In the latest run all 64 wagons list all four cameras, including wagons
> for which `LEFT_UP_TOP` has no valid frames at all.

#### 3f. Fallback

Two independent triggers in `assemble_global_train_state`:

1. **Exception** anywhere in `fuse_master_timeline` or `build_global_wagons` →
   `fallback_used = True`, `fallback_reason = "fusion error: <Type>: <msg>"`,
   `corrections = []`, and `wagons = build_wagons_pure_master(...)`.
2. **Empty result** — `if not wagons:` after the try →
   `fallback_used = True`, reason `"no wagons produced; using pure RIGHT_UP
   build"`, `corrections = []`, rebuild from master only.

`build_wagons_pure_master` calls the same `build_global_wagons` with
`fused = sorted(master_tracks.gaps)` and `support_camera_ids=[MASTER_CAMERA]`, so
the fallback count is exactly `master_gaps + 1`.

*Measured latest run:* `fallback_used: false` — the fallback did **not** run.

#### 3g. The final state object

```python
state = GlobalTrainState(total_wagons=len(wagons), wagons=wagons, ...)
```

**`total_wagons` is literally `len(wagons)` — the number of segments produced in
3e.** Nothing filters, merges or de-duplicates that list afterwards.

Derived properties are pure counts over classification and are computed on
demand:

```python
regular_wagon_count = sum(1 for w in wagons if w.classification == "WAGON")
engine_count        = sum(1 for w in wagons if w.classification == "ENGINE")
brake_van_count     = sum(1 for w in wagons if w.classification == "BRAKE_VAN")
```

`UNKNOWN` wagons are counted in **none** of those three, but **are** included in
`total_wagons`. So `regular + engine + brake_van` can be **less** than
`total_wagons`.

*Measured:* `total_wagons = 64` = 60 WAGON + 1 ENGINE + 1 BRAKE_VAN + **2
UNKNOWN**.

---

### STAGES 4–7 — Output (none of it affects the count)

* **Stage 4** writes `results/global_train_state.json` (`state.to_json()`) and
  `results/per_camera_tracking.json`.
* **Stage 5** overlay MP4s — opt-in via `--render-videos`, off by default.
* **Stage 6** `evidence_report.build_combined_report(...)` — reads the finished
  `state` and `tracks`, writes `results/combined_report.pdf`, deletes its temp
  frames. **It is a pure consumer.** It never mutates `state`, never returns a
  count, and cannot alter `total_wagons`. Deleting `evidence_report.py` would not
  change a single number in `global_train_state.json`.
* **Stage 7** prints `summarize_state(state)` and re-writes the JSON so any
  warning notes appended during stages 5–6 are persisted.

---

## 2. TEMPORAL SYNCHRONIZATION — READ THIS SECTION CAREFULLY

### There is no automatic synchronization anywhere in this codebase.

Verified by reading every module: **no offset estimation, no cross-correlation,
no clock/timestamp parsing, no drift correction, no calibration file, no
per-camera offset parameter, and no CLI flag for any of it.** The word "sync" in
the docs refers to an *upstream* trimming step that is expected to have already
happened.

### The only alignment mechanism

Everything rests on this identity, applied independently per camera:

```
master time (seconds) = master_frame / master_fps
local  frame          = round(master_time × local_fps)
```

That is it. Two cameras are considered aligned **iff their frame 0 is the same
instant in the real world.** The code asserts this; it never checks it.

`README.md` and the header of `run_global_count.py` state the requirement
explicitly: *"The 4 videos must be **synchronized** — i.e., trimmed by an upstream
service to the same train pass so they share a `t=0` alignment."*

### `video_segmenter.map_global_wagon_to_local_frames(wagon, local_fps, local_total_frames)`

The single function that crosses from master time to a camera's frames:

```python
if local_fps <= 0 or local_total_frames <= 0:
    return (0, -1)                                   # sentinel: empty/invalid
sf = int(round(wagon.start_time * local_fps))
ef = int(round(wagon.end_time   * local_fps)) - 1
sf = max(0, min(local_total_frames - 1, sf))         # CLAMP
ef = max(0, min(local_total_frames - 1, ef))         # CLAMP
if ef < sf: ef = sf
return (sf, ef)
```

Assumptions baked in: shared `t=0`; constant, exact fps; no drift; no dropped
frames; direct proportionality between frame index and wall-clock time.

**Out-of-range handling is silent clamping.** An event at master t = 260 s
projected onto a 208 s camera yields `sf = ef = local_total_frames - 1` — a
*syntactically valid* one-frame range pointing at the last frame, which has
nothing to do with the event. The function returns no error and no flag.

Only `evidence_report.select_event_evidence` guards against this, and only for
its own purposes: it recomputes the **unclamped** projection and, if
`raw_end < 0 or raw_start > total_frames-1`, reports "event outside this camera's
video length" rather than using the clamped value. **That guard is in the
reporting layer only** — `build_camera_wagon_frame_map` and
`render_processed_video` use the raw clamped result.

### Unequal video durations — measured

| Camera | frames | fps | duration |
|---|---|---|---|
| RIGHT_UP (master) | 4140 | 15.0 | **276.0 s** |
| LEFT_UP | 3690 | 15.0 | **246.0 s** (−30.0 s) |
| RIGHT_UP_TOP | 4110 | 15.0 | **274.0 s** (−2.0 s) |
| LEFT_UP_TOP | 3120 | 15.0 | **208.0 s** (−68.0 s) |

All four share 15.00 fps and 848×480. **Equal fps is not synchronization** — it
only means one master second equals one local second in *duration*. It says
nothing about whether the two timelines start at the same instant.

### Measured evidence that `t=0` is NOT shared

This is a **diagnostic measurement made for this document**, not something the
pipeline computes. Method: take each camera's `center_time` list from the run's
own `per_camera_tracking.json`, shift it by a constant offset, and count how many
support gaps land within ±0.5 s of *some* master gap. Sweep the offset over
±30 s in 0.05 s steps.

| Camera | gaps | matched at offset 0 | best offset | matched at best offset |
|---|---|---|---|---|
| LEFT_UP | 34 | **2** | **+16.20 s** | 19 |
| RIGHT_UP_TOP | 50 | **13** | **−3.15 s** | 28 |
| LEFT_UP_TOP | 27 | **7** | **+28.80 s** | 12 |

Agreement improves dramatically under a constant shift — LEFT_UP goes from 2/34
to 19/34. A large, camera-specific constant offset is the simplest explanation.

This is corroborated independently by the pipeline's own fusion statistics
(**60 of 111 support gaps, 54%, failed to match any master gap** within the 1.0 s
window) and visually by page 4 of `combined_report.pdf`, where `GW_1` is
`ENGINE (conf 1.00)` and the RIGHT_UP frames show locomotive **42816** while the
`LEFT_UP_TOP` frames at the same master time show **empty track**.

Caveats, stated honestly: the sweep is a crude greedy count, offsets may not be
constant (start offset vs drift are not separated here), and this does **not**
establish the true wagon count. It establishes that the shared-`t=0` assumption
does not hold for these four files. The 1.0 s `match_time_window_sec` is one to
two orders of magnitude smaller than the apparent offsets.

---

## 3. THRESHOLD REFERENCE

| Threshold | Value | Defined in | Controls |
|---|---|---|---|
| `--side-confidence` | 0.4 | `run_global_count.py` `_build_arg_parser` | YOLO conf floor, side gap models |
| `--top-confidence` | 0.4 | same | YOLO conf floor, `top_gap.pt` |
| `--side-min-height-ratio` | 0.35 | same | bbox height ÷ frame height, side cameras |
| `--top-min-height-ratio` | 0.05 | same | same, top cameras (thin strips) |
| `--classification-samples` | 5 | same | frames voted per segment in stage 2 |
| `match_distance_px` | 80.0 | `GapTracker.__init__` (not CLI) | max horizontal px to associate a detection with a track |
| `min_hits` | 3 | `GapTracker.__init__` (not CLI) | hits before a track is `confirmed` and may be emitted |
| `max_miss` | 30 | `GapTracker.__init__` (not CLI) | consecutive misses before a track closes (2.0 s @15 fps) |
| `process_var` / `meas_var` | 4.0 / 9.0 | `_KF1D.__init__` | Kalman Q and R on `center_x` |
| `match_time_window_sec` | 1.0 | `PHASE1_DEFAULTS` (**not CLI**) | support↔master gap match window |
| `match_min_iou` | 0.2 | `PHASE1_DEFAULTS` (**not CLI**) | alternative temporal-IoU match route |
| `insert_min_support` | 2 | `PHASE1_DEFAULTS`, `--fuse-min-support` | distinct cameras required to insert a gap |
| `insert_max_spread_sec` | 1.5 | `PHASE1_DEFAULTS`, `--fuse-max-spread` | **both** the clustering radius and the accepted spread |
| `insert_min_confidence` | 0.4 | `PHASE1_DEFAULTS`, `--fuse-min-conf` | mean cluster confidence to insert |
| `insert_min_distance_to_master_sec` | 1.0 | `PHASE1_DEFAULTS` (**not CLI**) | keep-away distance from existing master gaps |
| `EVIDENCE_PERCENTAGES` | (20,40,60,80) | `evidence_report.py` | reporting only |
| `DEFAULT_REPORT_DPI` | 150 | `evidence_report.py` | reporting only |

No threshold anywhere is adaptive, learned, or data-dependent.

---

## 4. CURRENT COUNTING FORMULA / LOGIC

```text
# ---------- STAGE 1: per camera, independently ----------
for cam in (RIGHT_UP, LEFT_UP, RIGHT_UP_TOP, LEFT_UP_TOP):
    tracker = GapTracker(cam, gap_model[cam], confidence, min_height_ratio)
    tracks[cam] = tracker.process_video(video[cam])

    # inside process_video, per frame:
    #   dets = _detect_gaps(frame, H)
    #        = [d for d in YOLO(frame)
    #             if (single_class_model or "gap" in class_name)     # engine_head/locono DROPPED
    #            and d.conf >= confidence                            # 0.4
    #            and d.height >= H * min_height_ratio]               # 0.35 side / 0.05 top
    #   greedy 1-D nearest-neighbour on center_x, |dx| <= 80 px, Kalman-smoothed
    #   track confirmed at hit_count >= 3 ; closed after 30 consecutive misses
    # emit one GapEvent per confirmed track, renumbered 1..N by first_frame
    #   center_time = ((start_frame + end_frame) / 2) / fps
    # local_wagon_count = len(gaps) + 1        <-- DIAGNOSTIC ONLY

master  = tracks[RIGHT_UP]                      # master camera is HARD-CODED
support = [tracks[c] for c in ALL_CAMERAS if c != RIGHT_UP]

# ---------- STAGE 2: classification (NO effect on count) ----------
pre_segments            = segments_from_gaps(master.gaps, master.total_frames)
initial_classifications = MasterClassifier(side_classification.pt, num_samples=5) \
                              .classify_segments(master.video_path, pre_segments)
    # per segment: skip outer 10%, sample 5 frames, majority vote,
    # _label_to_class -> ENGINE | WAGON | BRAKE_VAN | UNKNOWN

# ---------- STAGE 3: temporal normalization + fusion ----------
# NOTE: "normalization" is ONLY  t = frame / fps  per camera.
#       No offset is estimated. Shared t=0 is ASSUMED.
for st in support:                                       # match_support_to_master
    for sg in st.gaps:
        best = argmax over master_gaps of
                 max(temporal_iou, 1 - dt/1.0)  if (iou >= 0.2 or dt <= 1.0) else -1
        if best exists: matched[sg] = best               # matched gaps are DISCARDED
        else:           leftover[st.camera_id].append(sg)

clusters = cluster_unmatched_supports(leftover, spread_sec=1.5)   # running-mean sweep

inserts = []                                             # decide_inserted_gaps
for cl in clusters:
    if len(distinct camera_ids in cl) < 2:            continue    # quorum
    if (max(centers) - min(centers)) > 1.5:           continue    # spread
    if mean(confidence) < 0.4:                        continue    # confidence
    if min(|mean(centers) - mc| for mc in master_centers) < 1.0: continue  # keep-away
    inserts.append(GapCorrection(at = mean(centers)))

synthetic  = [GapEvent(track_id = -k, center_frame = round(at * master_fps)) for k, at in inserts]
fused_gaps = sorted(master.gaps + synthetic, key=center_time)

# ---------- STAGE 4: ordered global events -> GW ids ----------
boundaries = sorted(clamp(round(g.center_frame), 0, master.total_frames - 1) for g in fused_gaps)
segs, prev = [], 0
for b in boundaries:
    if b <= prev: continue                 # ONLY duplicate suppression in the pipeline
    segs.append((prev, b - 1)); prev = b
if prev <= master.total_frames - 1:
    segs.append((prev, master.total_frames - 1))

wagons = [GlobalWagon(global_id = f"GW_{i}", wagon_index = i,
                      start_frame_master = sf, end_frame_master = ef,
                      classification = label_for_frame((sf + ef) // 2))
          for i, (sf, ef) in enumerate(segs, start=1)]

# ---------- FINAL COUNT ----------
total_wagons = len(wagons)                 # == len(segs) == accepted_boundaries + 1
                                           # classification NEVER filters this list
```

**In one line:**

```
LEGACY  (--fusion legacy, described above):
total_wagons = (number of RIGHT_UP gap tracks) + (number of inserted gaps) + 1
               ─ (boundaries collapsed by the `b <= prev` rule)

CURRENT DEFAULT  (--fusion master-fixed + wagon-only):
raw RIGHT_UP gap detections
   -> tracked candidates                      (existing Kalman tracker)
   -> VALIDATED gap events                    (gap_validation.py: motion,
                                               persistence, trajectory,
                                               direction, duplicates)
   -> segments = validated gaps + 1           (build_global_wagons, unchanged)
   -> classification per segment              (side_classification.pt)
   -> WAGON WINDOW = first WAGON .. last WAGON
total_wagons = WAGON units inside that window
               (ENGINE and BRAKE_VAN excluded -- they never get a GW id)
```

*Measured on the run in `results/`:* legacy gives `52 + 11 + 1 = 64`;
master-fixed gives `52 + 1 = 53`. Zero boundaries collapsed in either case.
Under master-fixed the support-camera term does not exist: `build_global_wagons`
is fed `list(master_tracks.gaps)` directly, so no support-derived boundary can
reach it.

---

## 5. ONE CONCRETE EXAMPLE

Real values from the latest run (`results/global_train_state.json`,
`results/per_camera_tracking.json`, `results/combined_report.pdf` page 4).

### `GW_1` — the leading engine

**Stage 1, RIGHT_UP.** YOLO fires on the gap behind the locomotive over frames
134–146. Greedy association links them into one track (13 hits, ≥ 3 → confirmed;
no 30-frame miss run until it leaves). Renumbered to `track_id = 1`:

```json
{"track_id": 1, "camera_id": "RIGHT_UP", "start_frame": 134, "end_frame": 146,
 "start_time": 8.9333, "end_time": 9.7333, "center_time": 9.3333,
 "confidence": 0.8117, "hit_count": 13, "temporal_consistency_score": 1.0,
 "class_label": "gap"}
```

`center_frame = (134+146)/2 = 140.0` → `center_time = 140/15 = 9.3333 s`.

**Stage 1, other cameras.** They produced their own gap lists (34 / 50 / 27
tracks). None of them is consulted for `GW_1`'s existence — `GW_1` exists purely
because RIGHT_UP's timeline starts with a gap at frame 140.

**Stage 2.** `segments_from_gaps` makes pre-fusion segment 0 = frames
`(0, 139)` — from video start to just before the first boundary. Five frames
inside the middle 80% are sampled and voted:

```json
{"segment_index": 0, "start_frame": 0, "end_frame": 139,
 "label": "ENGINE", "confidence": 1.0}
```

Raw label `engine` → `_ENGINE_LABELS` → `SegmentClass.ENGINE`.

**Stage 3, fusion.** No correction was inserted before t = 9.33 s (the earliest
of the 11 is at t = 26.72 s), so the head of the fused timeline is unchanged.
First boundary = `round(140.0) = 140`.

**Stage 3, segment build.** `prev = 0`, `b = 140`; `140 > 0` so
`segs.append((0, 139))`, `prev = 140`. First segment → `i = 1` → **`GW_1`**:

```json
{"global_id": "GW_1", "wagon_index": 1,
 "start_frame_master": 0, "end_frame_master": 139,
 "start_time": 0.0, "end_time": 9.3333, "duration": 9.3333,
 "classification": "ENGINE", "classification_confidence": 1.0,
 "supporting_cameras": ["RIGHT_UP","LEFT_UP","RIGHT_UP_TOP","LEFT_UP_TOP"],
 "split_from_global_id": null,
 "leading_gap":  {"source": "video_start"},
 "trailing_gap": {"source": "master", "camera_id": "RIGHT_UP",
                  "track_id": 1, "center_time": 9.3333}}
```

`start_time = 0/15 = 0.0`; `end_time = (139+1)/15 = 9.3333`. `leading_gap` is
`video_start` because no fused gap has `center_frame <= 0`.

**`GW_2` — a rounding detail.** RIGHT_UP gap 2 spans 191–206, so
`center_frame = 397/2 = 198.5`. Python's `round()` is banker's rounding, so
`round(198.5) = 198`, not 199 — and indeed `GW_2` is frames `140–197`. This
matters only for exact reproducibility, but it is real behaviour.

`GW_2` is classified `BRAKE_VAN` at confidence 0.9984, i.e. the second vehicle
behind the engine. A brake van in position 2 of 64 is physically unusual — see
**WHY THE COUNT CAN BE WRONG → classification error**.

### One inserted gap — correction #1

```json
{"inserted_at_master_time": 26.7167, "inserted_at_master_frame": 401,
 "supporting_cameras": ["LEFT_UP","LEFT_UP_TOP"], "mean_confidence": 0.8829,
 "time_spread_sec": 1.3667, "contributing_track_ids": {"LEFT_UP_TOP": 3, "LEFT_UP": 5}}
```

`LEFT_UP` track 5 and `LEFT_UP_TOP` track 3 both failed to match any master gap
within 1.0 s, landed in the same 1.5 s cluster, came from 2 distinct cameras
(quorum ✓), had spread 1.3667 s (≤ 1.5 ✓), mean confidence 0.8829 (≥ 0.4 ✓), and
sat ≥ 1.0 s from every master gap (✓). A synthetic `GapEvent(track_id=-1,
center_frame=401)` was added, creating one extra boundary and therefore **one
extra wagon**, and shifting every subsequent `GW_n` up by one.

Note the spread of **1.3667 s** — 91% of the 1.5 s budget. Given the offset
evidence in section 2, the more likely reading is that these two gaps are *not*
the same physical gap at all, but two unrelated gaps from two differently-shifted
timelines that happened to fall within 1.5 s of each other.

---

## 6. WHY THE COUNT CAN BE WRONG

No fixes proposed here — only the places in the *current* logic where a wrong
count can originate.

### 6.1 Detection error
* **Class filter drops most of the model's output.** `if not single_class_model
  and "gap" not in name` discards `engine_head` and `locono` from the side
  models. If the training data ever labels an inter-wagon gap as one of those,
  that gap is invisible → missed boundary → **undercount**.
* **`top_gap.pt` is single-class, so its class filter is bypassed entirely.** Any
  box it emits above 0.4 confidence and 5% frame height becomes a gap candidate.
  A noisier top model feeds more false leftovers into fusion → more insert
  candidates → **overcount**.
* **Confidence 0.4 is low and uniform** across four cameras with different
  lighting and geometry. It is not tuned per camera.
* **Height filter is a bbox-height test only.** A tall vertical shadow, pole,
  catenary mast or inter-coach shade passes the 0.35 side test; width is never
  checked.

### 6.2 False gap / missed gap
* Every accepted boundary is `+1` wagon. There is **no minimum wagon length**
  check anywhere. Two boundaries one frame apart produce a 1-frame "wagon".
* *Observed in the latest run:* `GW_63` = frames 4124–4135 (12 frames, 0.8 s) and
  `GW_64` = frames 4136–4139 (**4 frames, 0.27 s**), both classified `UNKNOWN`
  (raw label `empty_track`). These are almost certainly not wagons, yet both are
  in `total_wagons`. That alone is a probable **+2 overcount**.
* A missed master gap merges two wagons into one → **undercount** — recoverable
  only if ≥ 2 support cameras independently see it *and* pass all four insertion
  tests.

### 6.3 Tracking error
* **Association is 1-D on `center_x` with an 80 px gate.** Two gaps closer than
  80 px horizontally can be absorbed into one track (→ one boundary instead of
  two, **undercount**); a fast gap moving more than 80 px between frames breaks
  into two tracks (→ two boundaries, **overcount**).
* **`min_hits = 3`.** A gap detected on only 1–2 frames is never confirmed and is
  silently dropped → **undercount**. At 15 fps that is anything visible under
  ~0.2 s.
* **`max_miss = 30`** (2.0 s). If a gap is occluded longer than 2 s the track
  closes and a re-detection starts a new track → same gap counted twice →
  **overcount**.
* `center_frame` is the **midpoint of the whole track**, so a track that lingers
  (large `end_frame`) biases the boundary later in time, shifting the wagon
  split.

### 6.4 Camera timing misalignment  ← *most likely primary cause here*
* **No offset estimation exists.** The code assumes shared `t=0`.
* `match_time_window_sec = 1.0 s` is the entire tolerance. Measured apparent
  offsets are **≈ +16 s (LEFT_UP), ≈ −3 s (RIGHT_UP_TOP), ≈ +29 s
  (LEFT_UP_TOP)** — 3× to 29× the window.
* Consequence A: genuine support observations of a real master gap fail to match
  (**60 of 111 leftover, 54%**) and are treated as *evidence of a missed gap*
  rather than as confirmation of a known one.
* Consequence B: those leftovers cluster by coincidence, and any 2 of them from
  distinct cameras within 1.5 s and ≥ 1.0 s from a master gap become an
  **inserted boundary → +1 wagon each**. All **11** corrections in the latest run
  are suspect for this reason; 8 of the 11 are `LEFT_UP`+`LEFT_UP_TOP` pairs, the
  two most-shifted cameras.
* Consequence C: a support gap can match the *wrong* master gap — within 1.0 s
  the matcher takes the best score, and with a large offset the "best" master gap
  need not be the corresponding one.

### 6.5 Unequal video duration
* Durations differ by up to **68 s** (276 s vs 208 s). Master events after
  208 s have no `LEFT_UP_TOP` counterpart at all, so the effective quorum
  population shrinks toward the tail — late missed gaps are less recoverable.
* Conversely, if a shorter video is *truncated* rather than *offset*, its gaps
  correspond to a different portion of the train, which is exactly the failure in
  6.4.

### 6.6 Cross-camera matching error
* Matching is **many-to-one and unvalidated** — several support gaps may match
  the same master gap. Nothing flags this, and a genuinely missed master gap can
  therefore be masked: its support evidence gets absorbed by the neighbouring
  master gap instead of becoming a leftover. → **undercount**.
* **A matched support gap is discarded** — it never refines the master boundary's
  time. Master timing errors are never corrected by agreement.
* `score = max(iou, time_score)` mixes two different units (a ratio and a
  normalized time distance), so ranking between candidate masters can be
  dominated by whichever measure happens to be larger.

### 6.7 Duplicate global event creation
* The **only** duplicate suppression is `if b <= prev: continue` on the integer
  frame index. Boundaries collapse only when they round to the *same* frame.
* `insert_min_distance_to_master_sec = 1.0 s` prevents an insert *near a master
  gap*, but **nothing prevents two accepted inserts from being close to each
  other** — `decide_inserted_gaps` compares each cluster against
  `master_centers` only, never against previously accepted inserts. Two clusters
  0.2 s apart (both ≥ 1.0 s from any master gap) both insert → two boundaries
  3 frames apart → a spurious extra wagon.

### 6.8 Missing global event
* An insert needs **all four** tests. Realistic failure paths: only one camera
  saw it (quorum 2 fails); the two sightings are 1.6 s apart (spread fails); the
  cluster's running-mean sweep split them into separate clusters; or the gap is
  genuinely within 1.0 s of a master gap and is suppressed by the keep-away rule.

### 6.9 Classification error
* **Classification never changes `total_wagons`.** An `UNKNOWN`/`empty_track`
  segment is still one wagon. This is by design in `build_global_wagons`, and it
  is why the two tail `UNKNOWN` segments inflate the count (6.2).
* Sampling skips the outer 10% and votes 5 frames via `cap.set` seeking. On a
  short segment `safe_end <= safe_start` collapses to a **single** midpoint
  sample — one frame decides the label.
* Tiebreak `-ord(label[0])` is arbitrary (alphabetical), not confidence-based.
* *Observed:* `GW_2` = `BRAKE_VAN` at conf 0.9984 in position 2 of 64. Either the
  classifier is confidently wrong, or the segmentation at the head of the train
  is wrong. Either way it signals a real problem worth tracing — though it does
  not by itself change the count.

### 6.10 End-of-video clipping
* `build_global_wagons` **always** appends a final segment `(prev,
  total_frames-1)` unless a boundary landed exactly on the last frame. If the
  master video runs on after the train has passed, that trailing empty track
  becomes a wagon. `GW_64` (4 frames, `UNKNOWN`) looks exactly like this.
* `total_frames = max(effective_frames, meta)` — if the container over-reports
  `CAP_PROP_FRAME_COUNT`, the timeline is padded past the real footage and the
  tail segment is stretched.
* `map_global_wagon_to_local_frames` clamps out-of-range projections to the last
  frame instead of reporting them as invalid; consumers other than
  `evidence_report` cannot distinguish "last frame" from "no data".

---

## 7. WHAT THE CURRENT SYSTEM ASSUMES

Assumptions, each of which the code relies on and none of which it verifies:

1. **All four videos share a common `t=0`.** The single most load-bearing
   assumption. *Measured evidence says it does not hold for the current files.*
2. **Frame index ÷ fps is true wall-clock time**, with constant fps, no dropped
   frames, no variable frame rate and no drift over 276 s.
3. **`RIGHT_UP` is the master and is authoritative.** Hard-coded as
   `MASTER_CAMERA = CAMERA_RIGHT_UP` in `global_train_state.py`. Its gap list is
   the base timeline; support cameras may only *add*, never remove or re-time.
4. **A detected "gap" means an inter-wagon coupling gap** — not a shadow, a
   background gap between the train and the far side, or a gap between the train
   and another train.
5. **Every inter-wagon boundary is visible to at least the master**, or else to
   ≥ 2 support cameras with all four insertion tests satisfied.
6. **All four cameras cover the whole train pass.** `supporting_cameras` is set
   to all four for every wagon unconditionally, which encodes this assumption
   even when it is false.
7. **The train moves in one direction at roughly steady speed.** No direction is
   ever computed — `center_x_trajectory` is recorded but never used for
   direction; `vx` in the Kalman filter is internal only. Ordering is purely
   temporal (increasing master frame), so a reversing or shunting movement would
   be counted as additional wagons.
8. **Wagon order = time order on the master clock.** `GW_n` is positional in
   master frames.
9. **The videos are trimmed to exactly one train pass**, with no other train, no
   long empty lead-in and no long empty run-out.
10. **All four videos have the same fps** (measured true: 15.0 each) — but note
    the code does not require it; it converts per camera. Equal fps is *not*
    evidence of synchronization.
11. **`.pt` models load with `weights_only=False`** — `tracker_engine.py`
    monkey-patches `torch.load` to allow the pickled ultralytics checkpoints on
    torch ≥ 2.6.

**Measured facts (not assumptions), latest run:** durations 276.0 / 246.0 /
274.0 / 208.0 s; all 15.00 fps; all 848×480; master gaps 52; inserts 11;
`total_wagons` 64; `fallback_used` false.

---

## 8. DATA FLOW

```
inputs/right_up.mp4      models/right_up_wagon_gap.pt  ─┐
inputs/left_up.mp4       models/left_up_wagon_gap.pt   ─┤
inputs/right_up_top.mp4  models/top_gap.pt             ─┤
inputs/left_up_top.mp4   models/side_classification.pt ─┘
        │
        ▼  STAGE 1  GapTracker.process_video  (×4, independent)
   tracks: Dict[str, LocalCameraTracks]          [in memory]
        │      └── .gaps : List[GapEvent]   ← the atomic evidence unit
        │
        ▼  STAGE 2  MasterClassifier            (RIGHT_UP only)
   initial_classifications: List[_MasterClassification]   [in memory]
        │
        ▼  STAGE 3  assemble_global_train_state
   fused_gaps        = master.gaps + synthetic inserts    [in memory]
   corrections       : List[GapCorrection]
   wagons            : List[GlobalWagon]   ← GW_1..GW_N created here
   state             : GlobalTrainState    ← total_wagons FIXED HERE
        │
        ├─► results/global_train_state.json    COUNTS  (canonical contract)
        ├─► results/per_camera_tracking.json   DEBUG   (per-camera gap timelines)
        ├─► results/processed_videos/*.mp4     REPORT-ONLY, opt-in
        └─► results/combined_report.pdf        REPORT-ONLY
                 (+ results/.evidence_tmp/  created then deleted)
```

**Affects the count:** `LocalCameraTracks.gaps`, `fused_gaps`, `corrections`,
`wagons`, `GlobalTrainState.total_wagons`.

**Does not affect the count:** `initial_classifications` (labels only),
`raw_frame_detections` (drawing only), `per_camera_tracking.json`,
`processed_videos/`, and **`evidence_report.py` / `combined_report.pdf`**.

> **`evidence_report.py` cannot change the wagon count.** It is invoked at stage 6,
> after `state` is final and already serialized at stage 4. It only *reads*
> `state` and `tracks`, and its single write path is the PDF plus a temp directory
> it deletes. It never assigns to `state.total_wagons` or `state.wagons`; the only
> mutation it can trigger is `state.add_note(...)` on failure, which is a
> free-text list. It reuses `map_global_wagon_to_local_frames` for its
> 20/40/60/80% sampling but that is a read-only projection.

---

## 9. FINAL COUNT TRACE

How to walk a wrong count backwards. Every step is a real field in a real file.

**Step 1 — the number.**
`results/global_train_state.json` → `total_wagons` (latest: **64**).
It equals `len(wagons)` exactly. If it disagrees with the length of the `wagons`
array, the JSON is stale — re-run.

**Step 2 — the ids.**
`wagons[]` → `global_id` `GW_1 … GW_64`, each with `start_frame_master`,
`end_frame_master`, `start_time`, `end_time`, `classification`.
*First triage:* sort by `duration`. Anything under ~1 s is a suspect spurious
split (latest run: `GW_64` = 0.27 s, `GW_63` = 0.80 s, both `UNKNOWN`).

**Step 3 — the boundaries that created it.**
Each wagon's `leading_gap` / `trailing_gap` gives `source`:
* `"master"` + `camera_id` + `track_id` → a real RIGHT_UP gap; go to step 4.
* `"fused"` (negative `track_id`, `camera_id` like `FUSED(LEFT_UP+LEFT_UP_TOP)`)
  → an inserted gap; go to step 5.
* `"video_start"` / `"video_end"` → an edge of the master video, not a detection.
  A short wagon bounded by `video_end` is an end-of-video artefact (6.10).

Also check `split_from_global_id`: non-null means at least one boundary was
inserted, so this wagon exists *because of fusion*. Latest run: 21 such wagons.

**Step 4 — a master boundary back to source frames.**
`results/per_camera_tracking.json` → `RIGHT_UP.gaps[]`, find the entry with that
`track_id`. It gives `start_frame`, `end_frame`, `center_time`, `confidence`,
`hit_count`, `temporal_consistency_score`.
* `hit_count` near 3 → barely confirmed, possibly noise.
* `temporal_consistency_score` well under 1.0 → intermittent detection, possibly
  a broken/merged track.
* Source frames are `right_up.mp4` frames `start_frame … end_frame`; the boundary
  itself is `round((start+end)/2)`.

**Step 5 — an inserted boundary back to its evidence.**
`global_train_state.json` → `corrections_applied[]`, match on
`inserted_at_master_frame`. `contributing_track_ids` maps camera → that camera's
`track_id`; look each up in `per_camera_tracking.json` under that camera's
`gaps[]`.
*Check:* `time_spread_sec` near 1.5 and `supporting_cameras` being the two most
time-shifted cameras is the signature of a coincidental cluster rather than a
real missed gap (see 6.4).

**Step 6 — sanity-check the arithmetic.**
```
total_wagons  ==  RIGHT_UP gap_count  +  len(corrections_applied)  +  1
                  ─ (boundaries collapsed by `b <= prev`)
64            ==  52 + 11 + 1                                        ✓ (0 collapsed)
```
If this identity fails, boundaries collapsed — two gaps rounded to the same
master frame.

**Step 7 — visual confirmation.**
`results/combined_report.pdf` page for that `GW_n`: the RIGHT_UP row is
authoritative. **Do not trust the other three rows for identity until the timing
offsets in section 2 are resolved** — under a large offset they show a different
part of the train at the same master time.

**Step 8 — is the master itself wrong?**
Compare `per_camera_tracking.json` gap counts: 52 / 34 / 50 / 27. If the master
is much *lower* than the others, it is missing gaps (undercount risk). Here the
master is the **highest**, and `LEFT_UP` (34) and `LEFT_UP_TOP` (27) are far
lower — consistent either with weaker detection on those cameras, or with those
cameras covering a *different span of the train* because of the timing offset.

---

## 10. FILES AND FUNCTIONS REFERENCE

### `run_global_count.py` — orchestration
| Symbol | Role |
|---|---|
| `_here` | project root = directory of this file; makes all defaults relative |
| `_resolve_input` / `_resolve_model` | locate the 4 videos and 4 weights, with filename fallbacks |
| `_process_side_camera` / `_process_top_camera` | build a `GapTracker` and run it (differ only in which conf/height ratio they receive) |
| `_classify_master_pre_fusion` | pre-fusion master segments → `MasterClassifier` |
| `_build_arg_parser` | every CLI threshold and its default |
| `main` | stage order; owns `tracks`, `state`, JSON writing |

### `tracker_engine.py` — detection + tracking (per camera)
| Symbol | Role |
|---|---|
| `_KF1D` | constant-velocity Kalman filter on bbox `center_x` only |
| `_Track` | in-progress track: hits, misses, centers, bboxes, `confirmed` |
| `GapTracker.__init__` | patches `torch.load(weights_only=False)`, loads YOLO, sets `_is_single_class_model` |
| `GapTracker.process_video` | sequential frame loop, association, track lifecycle, emits `GapEvent`s |
| `GapTracker._detect_gaps` | per-frame YOLO + class/confidence/height filters (+ diagnostics) |
| `MasterClassifier.classify_segments` / `_classify_one` | sample 5 frames per segment, majority vote |
| `MasterClassifier.classify_frame` | `probs` head, else best box, else `("wagon", 0.0)` |
| `MasterClassifier._label_to_class` | raw label → ENGINE / BRAKE_VAN / UNKNOWN / WAGON |
| `segments_from_gaps` | gap list → inclusive `(start,end)` segments (pre-fusion) |

### `global_alignment.py` — fusion (**where the count is decided**)
| Symbol | Role |
|---|---|
| `PHASE1_DEFAULTS` | the six fusion thresholds |
| `compute_temporal_iou` | IoU + overlap of two time intervals |
| `match_support_to_master` | support gap → best master gap, or `leftover` |
| `cluster_unmatched_supports` | running-mean temporal sweep over pooled leftovers |
| `decide_inserted_gaps` | quorum / spread / confidence / keep-away → `GapCorrection` |
| `fuse_master_timeline` | master gaps + synthetic inserts (negative `track_id`) |
| `build_global_wagons` | **boundaries → segments → `GW_n`**; inherits classification |
| `build_wagons_pure_master` | fallback: master gaps only |
| `assemble_global_train_state` | end-to-end; owns both fallback triggers |

### `global_train_state.py` — data contracts
| Symbol | Role |
|---|---|
| `CAMERA_*`, `MASTER_CAMERA`, `ALL_CAMERAS` | camera ids; master is hard-coded `RIGHT_UP` |
| `SegmentClass` | ENGINE / WAGON / BRAKE_VAN / UNKNOWN |
| `GapEvent` | one tracked gap; `center_frame`/`center_time` convert frames→seconds |
| `_MasterClassification` | one pre-fusion segment label |
| `LocalCameraTracks` | one camera's whole result; `local_wagon_count = gaps+1` |
| `GlobalWagon` | one global wagon incl. `global_id`, master window, provenance |
| `GapCorrection` | audit record for one inserted gap |
| `GlobalTrainState` | `total_wagons`, `wagons`, per-camera counts, corrections, fallback |
| `summarize_state` | the console summary block |

### `video_segmenter.py` — projection + drawing
| Symbol | Role |
|---|---|
| `map_global_wagon_to_local_frames` | **master time → local frame range** (clamped); the only cross-camera projection |
| `build_camera_wagon_frame_map` | `global_id → (start,end)` for one camera |
| `render_processed_video` | overlay MP4 (opt-in, reporting only) |
| `extract_wagon_frames` | **deprecated, no longer called** by the pipeline |
| `_interp_gap_bbox` | interpolate a tracked bbox between hits (drawing only) |

### `evidence_report.py` — reporting only, zero effect on the count
| Symbol | Role |
|---|---|
| `EVIDENCE_PERCENTAGES` / `PDF_FILENAME` | `(20,40,60,80)` / `combined_report.pdf` |
| `select_event_evidence` | per event+camera interval via `map_global_wagon_to_local_frames`, plus an unclamped overlap guard |
| `_percentage_frame` | `start + round(p/100 × (end−start))` |
| `extract_evidence_frames` | one `grab()`/`retrieve()` pass per camera into the temp dir |
| `_ReportBuilder` | composes summary, roster and one 4×4 page per event |
| `build_combined_report` | orchestrates select → extract → compose → verify → delete temp |
| `verify_pdf` / `cleanup_temp_evidence` | `%PDF`+`%%EOF` check; delete temp frames only on success |

---

## 11. DOCUMENTATION CONFIDENCE

### Directly verified by reading the current source
All execution order, function names, data structures, filter order, tracking
parameters, Kalman constants, fusion tests, insertion rules, boundary/segment
construction, `GW_n` assignment, `total_wagons = len(wagons)`, the classification
label map, both fallback triggers, every threshold and where it is defined, the
CLI surface, and the claim that `evidence_report.py` cannot influence the count.

### Taken from actual run output (not invented)
From `results/global_train_state.json`, `results/per_camera_tracking.json`,
`results/combined_report.pdf` and the console log of the last full run:
per-camera frames/fps/duration/gaps/local wagons; `total_wagons` 64; the 60/1/1/2
classification split; 11 corrections including correction #1's exact values;
`fallback_used: false`; the `GW_1`, `GW_2` and RIGHT_UP `track_id` 1–3 records;
53 pre-fusion segments; 21 wagons with `split_from_global_id`; the fusion
matched/leftover counts (12/22, 27/23, 12/15), 45 clusters, 11 inserts; and the
model class lists (`engine_head/gap/locono`, `gaps`,
`brakevan/empty_track/engine/wagon`) from `validate_ec2.py`.

### Measured for this document, not produced by the pipeline
The **camera offset estimates** in section 2 (+16.20 s, −3.15 s, +28.80 s). These
come from a read-only sweep over the run's own `center_time` lists, counting
support gaps within ±0.5 s of any master gap. The method is deliberately crude.
It is strong evidence that shared `t=0` is violated; it is **not** a calibration
result, and the true offsets may be non-constant.

### Not verified / unknown
* **The ground-truth wagon count of this train.** Nothing in the repository
  records it, so no statement here about the count being "too high" or "too low"
  is proven — only that specific segments (`GW_63`, `GW_64`) are implausible as
  wagons.
* **Whether the offsets are constant** (start offset) or drift over the pass.
  Distinguishing the two needs a longer analysis than a single-offset sweep.
* **Why the videos differ in duration** — upstream trimming behaviour is outside
  this repository.
* **Model training data and labelling conventions** — what exactly was annotated
  as `gap`, `engine_head`, `locono`, `gaps`, `empty_track`. Only the class
  *names* are observable from the weights.
* **`GW_2 = BRAKE_VAN`** — whether this is a classifier error or a genuine
  brake van behind the locomotive could not be determined from the repository.
* **GPU behaviour.** The verified run was CPU-only (`torch 2.12.0+cpu`, no CUDA).
  Ultralytics would select CUDA automatically elsewhere; results should be
  equivalent but that was not tested here.
* Timings and sizes quoted anywhere in this document are from a 4-core CPU
  Windows run and are not representative of the EC2 instance.

---

*This document describes behaviour only. No algorithm, threshold, model, input or
output was modified in producing it.*
