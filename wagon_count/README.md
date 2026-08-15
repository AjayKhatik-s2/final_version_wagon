# Wagon Eye — Phase 1 (standalone wagon counter)

Self-contained 4-camera global wagon counting + classification.
Designed to zip and run on AWS (EC2 / SageMaker) with no other code.

> **Phase 1 only.** No doors, no damage, no OCR, no PDF, no email,
> no S3 upload. Just global wagon segmentation + classification +
> processed videos + per-wagon frame folders.

---

## Folder layout

```
wagon_count/
├── run_global_count.py          # entry point
├── global_train_state.py        # data classes
├── tracker_engine.py            # per-camera gap tracking + master classifier
├── gap_validation.py            # raw candidates -> validated gap boundaries
├── global_fusion.py             # fixed-master fusion (RIGHT_UP is authoritative)
├── global_alignment.py          # wagon construction from the gap sequence
├── train_structure.py           # wagon window: first WAGON .. last WAGON
├── temporal_classification.py   # segment-level classification hysteresis
├── video_segmenter.py           # overlay rendering + frame-range mapping
├── evidence_report.py           # 20/40/60/80% evidence + combined_report.pdf
├── rejection_report.py          # diagnostic: table of every rejected candidate
├── validate_ec2.py              # pre-flight env/asset check (runs no pipeline)
├── setup_ec2.sh                 # Ubuntu/EC2 environment bootstrap
├── requirements.txt
├── .gitignore                   # keeps videos/weights/results out of Git
├── inputs/                      # drop your 4 trimmed train videos here
│   ├── right_up.mp4
│   ├── left_up.mp4
│   ├── right_up_top.mp4
│   └── left_up_top.mp4
├── models/                      # drop the YOLO weights here
│   ├── right_up_wagon_gap.pt    # RIGHT_UP (master) gap model
│   ├── left_up_wagon_gap.pt     # LEFT_UP gap model
│   ├── top_gap.pt               # top cameras (RIGHT_UP_TOP, LEFT_UP_TOP)
│   ├── side_classification.pt   # RIGHT_UP + LEFT_UP: ENGINE / WAGON / BRAKE_VAN
│   └── top_classification.pt    # RIGHT_UP_TOP + LEFT_UP_TOP (optional)
└── results/                     # created on first run
```

---

## Quick start (AWS EC2 / SageMaker)

```bash
# 1) Get the code
git clone https://github.com/AjayKhatik-s2/wagon_count_global.git
cd wagon_count_global

# 2) Set up the Python environment (creates ./.venv, installs requirements)
chmod +x setup_ec2.sh && ./setup_ec2.sh --with-apt-deps
source .venv/bin/activate

# 3) Copy the 4 videos into ./inputs/ and the 4 .pt models into ./models/
#    (they are intentionally not in Git -- see "Runtime assets" below)

# 4) Pre-flight check -- verifies env, videos and weights. Runs no pipeline.
python validate_ec2.py

# 5) Run with defaults -- all 4 inputs and 4 models auto-discovered
python run_global_count.py
```

That's it. The pipeline writes everything to `./results/`.

No environment variables are required — see *Environment variables* below.
For the full local→GitHub→EC2 walkthrough, see
**GitHub → EC2 Deployment**.

---

## Inputs

The 4 videos must be **synchronized** — i.e., trimmed by an upstream
service to the same train pass so they share a `t=0` alignment.
RIGHT_UP is the **master camera**; its gap detections and
classifications are authoritative.

Default filenames (auto-discovered from `inputs/`):

| Camera           | Filename                                   | Model                                            |
|------------------|--------------------------------------------|--------------------------------------------------|
| RIGHT_UP (master)| `right_up.mp4`                             | `right_up_wagon_gap.pt` + `side_classification.pt` |
| LEFT_UP          | `left_up.mp4`                              | `left_up_wagon_gap.pt`                            |
| RIGHT_UP_TOP     | `right_up_top.mp4`                         | `top_gap.pt`                                      |
| LEFT_UP_TOP      | `left_up_top.mp4`                          | `top_gap.pt`                                      |

You can also override any path explicitly:

```bash
python run_global_count.py \
  --right_up     /data/cam_right_up_20260408.mp4 \
  --left_up      /data/cam_left_up_20260408.mp4 \
  --right_up_top /data/cam_right_up_top_20260408.mp4 \
  --left_up_top  /data/cam_left_up_top_20260408.mp4 \
  --models-dir   /opt/models \
  --output       /opt/results
```

---

## Outputs

After a run:

```
results/
├── combined_report.pdf              ← THE evidence deliverable
├── global_train_state.json          ← canonical Phase-1 output
└── per_camera_tracking.json         ← per-camera gap timelines for debug
```

That is the whole output. Three files.

All four cameras use the **same** `GW_n` ids — they refer to the same
physical wagon. This is the contract Phase-2 (door / damage / OCR /
loaded-empty) will consume.

### What gets counted

```
                RIGHT_UP  (master, the only counting authority)
                    |
   raw YOLO gap detections   -- candidates only
                    |
   temporal tracking          (existing Kalman tracker, unchanged)
                    |
   gap validation             motion / persistence / trajectory / duplicates
                    |
   valid gap events  ->  segments  ->  classification
                    |
   WAGON WINDOW:  first WAGON .. last WAGON
                    |
                GW_1 .. GW_N          <-- the wagon count
```

Two rules govern the number:

1. **Only wagons are counted.** `ENGINE` and `BRAKE_VAN` are real train objects,
   are kept in the JSON, the PDF and the overlay videos, and are visible in the
   train-structure summary — but they never receive a `GW_n` id and never extend
   the wagon timeline. `ENGINE + 3 WAGON + BRAKE_VAN` yields `GW_1 GW_2 GW_3`,
   not five ids. Use `--no-wagon-only` to count every segment (A/B comparison).
2. **A raw YOLO gap is a candidate, not a boundary.** The train moves, so a real
   inter-wagon gap sweeps across the image. A detection pinned to one pixel
   column is background — measured on this project's own footage, three such
   artefacts had confidence 0.93 and were detected on every one of 30 frames, so
   confidence and persistence alone cannot reject them. See *Gap validation*.

`total_wagons == master_wagon_count`, always. Support cameras contribute
evidence, synchronization and diagnostics; they can never change the number.

### Fragment reassembly

```
GapTracker completed fragments
    -> fragment reassembly          (fragment_stitching.py)
    -> reconstructed physical gaps
    -> gap validation, unchanged
```

One physical gap is one object, but the tracker can emit it as several short
tracks: when a detection is missed the gap reappears beyond the association gate,
so the track closes and a new id opens. Validation then judges each piece
separately, each is shorter than the minimum track duration, and the gap they
jointly prove is never counted.

This was measured on a real train, and confirmed against the video frames: one
gap crossing the frame arrived as three consecutive 3-frame tracks. Three wagon
boundaries were lost that way, leaving a single **184-frame "wagon"** where the
median is 54 frames.

Reassembly restores the object *before* any duration judgement, so every existing
gate then applies to the whole gap. Two fragments are joined only when all hold:

| Criterion | Why |
|---|---|
| strictly sequential, hole ≤ `--stitch-max-seam-sec` | overlapping tracks are duplicates or two simultaneous gaps, not one object |
| both move in the camera's **dominant direction** | derived per camera from its own candidates |
| the later fragment is **ahead** of the earlier one | a fragment behind its predecessor is a different object |
| seam jump ≤ `--stitch-seam-tolerance` × what the **local advance rate** predicts | local, so acceleration, deceleration and stops are followed rather than assumed away |
| seam jump ≤ `--stitch-max-seam-frac` of frame width | a long hole must not authorise a leap across the frame |
| each fragment has ≥ 2 detections | a one-frame detection is never eligible, so raw detections cannot bypass tracking |

Two properties make this safe rather than permissive:

- **Merging is what needs evidence; refusing is free.** Failing any criterion
  leaves the fragments exactly as the tracker emitted them, so the worst case is
  today's behaviour.
- **The reference comes from tracks long enough to trust.** A fragment cannot
  measure train speed — measured, 3-hit tracks advance 29.7–41.7 px/frame while
  tracks that pass validation on their own advance at a median of 47.7. Predicting
  a seam from neighbouring fragments under-predicts ~2× and refuses exactly the
  seams it should accept. So the advance rate is measured only from tracks of at
  least `min_track_seconds`, and with fewer than three of those nothing is
  stitched at all.

Reassembly resolves both observed fragment cases with one rule and no
special-casing: fragments whose neighbours are also fragments become **one new
gap**, while a fragment whose neighbour already validates is **absorbed** into it
— so the leading edge of an already-counted gap cannot double-count. It can only
ever *reduce* the candidate count; it never accepts a gap, and a reassembled
track still faces every gate below. Disable with `--no-fragment-stitching`.

### Gap validation

Detection and tracking are unchanged: the existing YOLO models, confidence
thresholds, Kalman parameters and association gates are all untouched. What is
new is a deterministic validation layer between the tracker and fusion, which
reads the motion the tracker already recorded on every `GapEvent`
(`center_x_trajectory`, `hit_frames`, `bbox_history`).

A raw detection is a **candidate**, never a boundary — high detection confidence
is not evidence of a physical gap (the three false positives below all score
0.93). Each candidate must survive every test below, and each rejection is either
**HARD** (physically impossible for a real gap: it is an artefact in any train
state) or **SOFT** (implausible *given its neighbours*, which is a judgement that
can be wrong):

| Test | Rejection reason | Class |
|---|---|---|
| pinned centre | `REJECTED_STATIC` | HARD |
| against the camera's flow | `REJECTED_WRONG_DIRECTION` | HARD |
| no trajectory at all | `REJECTED_NO_TRAJECTORY` | HARD |
| track extent, hit count | `REJECTED_TOO_SHORT` | HARD |
| blind runs, coverage | `REJECTED_DETECTION_GAP` | HARD |
| same physical gap tracked twice | `REJECTED_DUPLICATE` | HARD |
| too close in time to an accepted gap | `REJECTED_MIN_SEPARATION` | HARD |
| far from the local median speed | `REJECTED_TRAIN_MOTION_MISMATCH` | SOFT |
| apparent speed out of band | `REJECTED_IMPLAUSIBLE_SPEED` | SOFT |
| insufficient displacement | `REJECTED_LOW_MOTION` | SOFT |
| mean track confidence | `REJECTED_LOW_CONFIDENCE` | SOFT |
| direction flips | `REJECTED_INCONSISTENT_TRAJECTORY` | SOFT |

**WAGON_ACTIVE recovery.** Validation must run *before* classification, because
classification needs the segments that validated gaps define — so the first pass
cannot know whether a candidate sits inside the train's wagon run. Once the wagon
window is derived, a second pass re-examines the SOFT rejections that fell inside
it. This is deliberately not "accept everything inside the train": a recovered
candidate must still clear every HARD gate plus safety floors on displacement,
confidence, direction and **path efficiency** (`|net displacement| ÷ path
travelled`). Path efficiency is what separates *noisier than expected* (~0.71 on
real tracks) from *oscillating noise* (~0.13) — monotonic fraction cannot, since
a drifting oscillation scores 0.55 monotonic while travelling 8× its own net
displacement. Disable with `--no-wagon-recovery`.

Nothing is discarded silently: every rejection is written to
`global_train_state.json` with its reason **and** its measured features, so the
chain `raw detections → tracked candidates → valid gap events` is auditable per
camera. `rejection_report.py` renders that record as a table (see
**Diagnosing an under-count** below).

Thresholds are **camera-independent**: every one is declared in seconds, as a
fraction of frame width, or as a dimensionless ratio, then resolved to pixels and
frames per camera at runtime from that camera's own width and FPS. Nothing is
tuned to one train's geometry, speed or wagon count.

Thresholds were set from measurement, not guesswork. Running the existing tracker
over the first 1500 frames of three cameras gave:

| Camera | tracks | \|displacement\| px (min/med/max) | speed px/s | monotonic | coverage |
|---|---|---|---|---|---|
| RIGHT_UP | 16 | 401 / 468 / 547 | 317 / 470 / 554 | 1.00 | 1.00 |
| RIGHT_UP_TOP | 18 | 111 / 340 / 614 | 74 / 387 / 546 | ≥0.61 | ≥0.58 |
| LEFT_UP_TOP | 8 | 0 / 248 / 354 | 0 / 237 / 356 | ≥0.52 | ≥0.43 |

The three zero-displacement LEFT_UP_TOP tracks are genuine false positives —
**confidence 0.93, 30 hits each, coverage 1.00**, centres pinned at x=436, x=487
and x=283. Confidence, hit-count and coverage filters all pass them; only motion
rejects them. Gap direction is also per-camera (RIGHT_UP moves −x, LEFT_UP_TOP
moves +x), so the dominant direction is derived from each camera's own tracks
rather than assumed.

Every default therefore has a measured margin: the static ceiling
(`--gap-static-max-frac`, 0.0047 of frame width ≈ 4 px at 848 px) sits against a
smallest real displacement of 110.7 px, and the motion floor
(`--gap-min-motion-frac`, 0.0142 ≈ 12 px) keeps a ~9× margin. All are
CLI-configurable and recorded in the output JSON.

### Diagnosing an under-count

When a train counts low, the question is always *which real gap was discarded, and
by which gate?* The run already records every rejected candidate with its
measured evidence; `rejection_report.py` renders it:

```bash
python rejection_report.py results/global_train_state.json

# just the recoverable class, inside the counted region -- the under-count shape
python rejection_report.py results/global_train_state.json \
    --soft-only --wagon-active-only

# for sharing / spreadsheets
python rejection_report.py results/global_train_state.json --csv rejections.csv
```

Each row gives the track id, frame range, duration, confidence, displacement,
its speed against the local reference speed, direction, monotonicity, path
efficiency, the rejection reason and HARD/SOFT class, the train state
(`PRE_WAGON` / `WAGON_ACTIVE` / `POST_WAGON`), the nearest accepted gap on either
side, and what recovery did with it.

Read it like this:

- A **SOFT** rejection inside `WAGON_ACTIVE` that was **not** recovered is the
  signature of a lost wagon boundary. The summary block lists these explicitly.
- A rejected candidate sitting midway between two accepted gaps roughly **2×** the
  normal spacing apart is almost certainly a real boundary that was dropped.
- **HARD** rejections are static / reversed / duplicate / blind-track artefacts.
  Those are meant to die, in every train state.

The script reads only the run's JSON — it re-runs no models and re-derives
nothing, so it cannot disagree with the pipeline that produced the numbers.

### `combined_report.pdf` — the evidence report

The pipeline no longer writes one JPEG per frame per wagon per camera
(which produced tens of thousands of files and 1–2 GB per run). Instead it
selects a small set of **representative evidence frames** and packages them
into a single PDF:

* **Page 1 — summary.** Master camera, master FPS, master total frames,
  final global wagon count, regular / engine / brake-van counts,
  per-camera local wagon and gap counts with status, the global
  corrections/insertions table, and the evidence policy. Every value is
  read straight from the pipeline's own `GlobalTrainState`; nothing is
  recomputed or invented.
* **Global wagon roster.** Every `GW_n` with its classification,
  confidence, master frame range, master time window, duration and how
  many evidence frames were captured. Paginated for long trains.
* **One page per global event.** A `4 cameras × 4 percentages` grid — up to
  **16 evidence images per event** — above a metadata block carrying the
  global wagon id, wagon index, master frame range, master time window and
  duration, supporting cameras, and the leading/trailing gap provenance
  (source, camera, track id, centre time).

Each image is captioned with its camera and percentage, exactly:

```
RIGHT_UP - 20%      RIGHT_UP - 40%      RIGHT_UP - 60%      RIGHT_UP - 80%
LEFT_UP - 20%       LEFT_UP - 40%       LEFT_UP - 60%       LEFT_UP - 80%
RIGHT_UP_TOP - 20%  RIGHT_UP_TOP - 40%  RIGHT_UP_TOP - 60%  RIGHT_UP_TOP - 80%
LEFT_UP_TOP - 20%   LEFT_UP_TOP - 40%   LEFT_UP_TOP - 60%   LEFT_UP_TOP - 80%
```

so the same physical event can be compared across all four camera
perspectives at a glance. Aspect ratio is always preserved — frames are
scaled, never stretched.

### How the 20 / 40 / 60 / 80 % frames are chosen

The percentages are relative to **that camera's own valid evidence interval
for that one event** — never to the whole video. The interval is obtained
from the project's existing mapping,
`video_segmenter.map_global_wagon_to_local_frames()`, which projects the
event's master time window onto the camera and clips it to that camera's
length. This is the same interval the old full-frame extraction walked, so
no new timing or alignment method was introduced.

Given a valid interval `[start, end]`:

```
frame(p) = start + round(p/100 × (end − start))       p ∈ {20, 40, 60, 80}
```

### When there are fewer than 16 images

Missing evidence is reported, never fabricated. A cell shows
**"No valid evidence available"** plus the reason:

| Situation | What the report shows |
|---|---|
| Event lies beyond this camera's video length (the four videos differ in duration) | `event outside this camera's video length` |
| Camera was not processed in this run | `camera not processed` |
| Interval shorter than 4 frames | the frame is used **once**, and the duplicate percentages say `interval too short (N frame(s)); 60% resolves to the same frame as 40%` |
| Frame could not be decoded / written | `frame could not be decoded` |

Partially overlapping events are **not** discarded — the interval is
clipped to the valid portion and the percentages are computed inside it.

### Temporary frames are deleted

Evidence frames are extracted at full resolution into
`results/.evidence_tmp/`, composed into the PDF, and **that directory is
deleted once the PDF has been written and verified** (`%PDF` header and
`%%EOF` trailer present, non-trivial size). If the PDF fails verification
the temporary frames are deliberately *kept* so the failure can be
diagnosed, and the reason is printed. Use `--keep-evidence-frames` to keep
them on a successful run too.

Temporary filenames are deterministic and collision-free:
`{CAMERA}__{GW_id}__p{pct}__f{frame:06d}.jpg`.

### Overlay videos are now opt-in

`results/processed_videos/*.mp4` were ~100 MB per camera and purely visual.
They are **off by default**; pass `--render-videos` if you want them.
`--no-videos` and `--every-nth-frame` are still accepted but ignored (they
print a deprecation note), so existing invocations keep working.

If an earlier run left a heavyweight `results/frames/` or
`results/processed_videos/` behind, the pipeline points it out but **never
deletes it** — remove it yourself when you are ready:

```bash
rm -rf results/frames results/processed_videos
```

`global_train_state.json` shape:

```json
{
  "schema": "wagon_eye.global_train_state.v1",
  "master_camera": "RIGHT_UP",
  "master_fps": 25.0,
  "master_total_frames": 7321,
  "total_wagons": 47,
  "regular_wagon_count": 45,
  "engine_count": 1,
  "brake_van_count": 1,
  "wagons": [
    {
      "global_id": "GW_1",
      "wagon_index": 1,
      "start_frame_master": 0,
      "end_frame_master": 312,
      "start_time": 0.0,
      "end_time": 12.52,
      "classification": "ENGINE",
      "classification_confidence": 0.94,
      "supporting_cameras": ["RIGHT_UP","LEFT_UP","RIGHT_UP_TOP","LEFT_UP_TOP"],
      "split_from_global_id": null,
      "leading_gap":  {"source": "video_start"},
      "trailing_gap": {"source": "master", "camera_id": "RIGHT_UP", "track_id": 1, "center_time": 12.51}
    },
    ...
  ],
  "per_camera_local_counts": { "RIGHT_UP": 46, "LEFT_UP": 45, "RIGHT_UP_TOP": 47, "LEFT_UP_TOP": 47 },
  "per_camera_gap_counts":   { "RIGHT_UP": 45, "LEFT_UP": 44, "RIGHT_UP_TOP": 46, "LEFT_UP_TOP": 46 },
  "corrections_applied":     [ {"inserted_at_master_time": 134.4, "supporting_cameras": ["LEFT_UP_TOP","RIGHT_UP_TOP"], ...} ],
  "fallback_used": false
}
```

---

## How the fusion works (one paragraph)

1. **Per-camera gap tracking** — each camera runs its own YOLO gap model.
   RIGHT_UP uses `right_up_wagon_gap.pt`; LEFT_UP uses
   `left_up_wagon_gap.pt`; both top cameras share `top_gap.pt`. A
   constant-velocity Kalman filter on the gap bounding-box center plus a
   hit/miss persistence rule emits one `GapEvent` per stable track.

2. **Master classification** — RIGHT_UP's pre-fusion segments (the spans
   between consecutive RIGHT_UP gaps) are labeled ENGINE / WAGON / BRAKE_VAN
   by `side_classification.pt` via majority vote on sampled frames.

3. **Cross-camera fusion** — each support camera's gaps are matched to
   the master gap timeline by **temporal IoU**. Unmatched support gaps
   are clustered across cameras. A cluster becomes an **inserted gap**
   only when it has **≥2 supporting cameras**, time spread ≤ 1.5 s,
   and mean confidence ≥ 0.4 — and is ≥ 1.0 s from any existing
   master gap. The fused master gap list is the original RIGHT_UP gaps
   plus accepted inserts.

4. **Global wagon rebuild** — segments between consecutive fused gaps
   become `GW_1 .. GW_N`. If an inserted gap splits a RIGHT_UP segment,
   children inherit the parent's classification (so ENGINE and BRAKE_VAN
   stay stable; a merged WAGON splits into two WAGONs).

5. **Fallback** — if support fusion throws or produces zero wagons,
   the system falls back to pure RIGHT_UP wagon counting and sets
   `fallback_used: true` in the JSON.

---

## Tuning knobs

All optional; defaults are usually fine.

| Flag                         | Default | Meaning                                      |
|------------------------------|---------|----------------------------------------------|
| `--side-confidence`          | 0.4     | YOLO conf threshold for side gap models (`right_up_wagon_gap.pt`, `left_up_wagon_gap.pt`) |
| `--top-confidence`           | 0.4     | YOLO conf threshold for `top_gap.pt`         |
| `--classification-samples`   | 5       | Frames per segment voted in classification   |
| `--fuse-min-support`         | 2       | Min cameras needed to insert a missed gap    |
| `--fuse-max-spread`          | 1.5     | Max time spread (s) inside a fusion cluster  |
| `--fuse-min-conf`            | 0.4     | Min mean confidence to insert a fused gap    |
| `--no-wagon-only`            | off     | Count every segment, including engines/brake vans |
| `--no-gap-validation`        | off     | Treat every tracked candidate as a boundary  |
| `--gap-min-motion-px`        | 12.0    | Min centre displacement over a track (real gaps measured 110–615 px) |
| `--gap-static-max-px`        | 4.0     | At/below this ⇒ `REJECTED_STATIC` (measured FPs moved ≤0.2 px) |
| `--gap-min-motion-px-sec`    | 8.0     | Min apparent speed                           |
| `--gap-max-motion-px-sec`    | 2000.0  | Max apparent speed                           |
| `--gap-motion-tolerance`     | 4.0     | Max factor from the camera's median gap speed |
| `--gap-min-confidence`       | 0.45    | Min mean track confidence (per-frame detector threshold unchanged) |
| `--gap-min-track-frames`     | 4       | Min track extent in frames                   |
| `--gap-max-track-gap`        | 20      | Longest blind run tolerated inside a track   |
| `--gap-min-monotonic`        | 0.60    | Min fraction of steps sharing the direction  |
| `--no-support-classification`| off     | Skip top/side classification of support cameras |
| `--render-videos`            | off     | Also render the overlay MP4s (~100 MB/camera) |
| `--no-report`                | off     | Skip evidence selection + `combined_report.pdf` |
| `--report-dpi`               | 150     | PDF page raster resolution; raise for bigger evidence images |
| `--keep-evidence-frames`     | off     | Keep the temporary evidence JPEGs (debugging) |
| `--no-raw-detections`        | off     | Save RAM by not storing per-frame bboxes     |
| `--quiet`                    | off     | Reduce log verbosity                         |

Deprecated but still accepted (ignored, with a note): `--no-videos`
(overlays are already off), `--every-nth-frame` (no full sequences are
written any more), and `--no-frames` (kept as an alias for `--no-report`).

---

## Environment variables

**No environment variables are currently required.**

The project reads nothing from the environment — there is no `os.environ`
or `os.getenv` call anywhere in the codebase, no `.env` loading, no config
file, no API keys, no credentials, and no network or S3 access. Everything
is controlled by CLI flags (see *Tuning knobs*) and by the two conventional
directories `inputs/` and `models/`.

There is therefore **no `.env.example` to copy**. If a future change does
introduce a variable, add it to a new `.env.example`; `.env` is already
listed in `.gitignore` so a real one can never be committed by accident.

*(Optional, for debugging only: setting `VALIDATE_VERBOSE=1` makes
`validate_ec2.py` print a full traceback when a model fails to load. It is
not required and affects nothing in the pipeline.)*

---

## Runtime assets (videos + model weights)

The 4 input videos (~30–40 MB each) and 4 model weights (~2–22 MB each) are
**deliberately excluded from Git** by `.gitignore`. The repository ships
source code, configuration and deployment scripts only; the runtime assets
are supplied separately on each machine.

That means a fresh clone gives you this, with the directories present (each
keeps its own `README.md`) but empty of assets:

```
wagon_count/
├── inputs/     ← empty; copy the 4 .mp4 files in
└── models/     ← empty; copy the 4 .pt files in
```

Required exact filenames:

| `inputs/`            | `models/`                  | Used by |
|----------------------|----------------------------|---------|
| `right_up.mp4`       | `right_up_wagon_gap.pt`    | RIGHT_UP gap detection |
| `left_up.mp4`        | `left_up_wagon_gap.pt`     | LEFT_UP gap detection |
| `right_up_top.mp4`   | `top_gap.pt`               | both TOP cameras, gap detection |
| `left_up_top.mp4`    | `side_classification.pt`   | RIGHT_UP (master) + LEFT_UP classification |
|                      | `top_classification.pt`    | RIGHT_UP_TOP + LEFT_UP_TOP classification — **optional** |

### `top_classification.pt` (optional)

Classifies the two overhead cameras into ENGINE / WAGON / BRAKE_VAN. It improves
support-camera classification, synchronization and PDF/video labels. It is
**never a counting authority** — RIGHT_UP alone determines the wagon count — so
if the file is absent the run still completes, the top cameras are simply not
classified, and `validate_ec2.py` reports the missing capability as a FAIL.

If you keep it in S3, copy it in as a deployment step (the pipeline itself
contains no S3 code, by design):

```bash
aws s3 cp s3://<your-bucket>/top_classification.pt models/top_classification.pt
python validate_ec2.py          # confirms it loads and prints its real class names
```

The semantic mapping is derived from the model's **actual** `model.names` at
runtime — class indices are never assumed, and an unrecognised class is mapped to
`UNKNOWN` and reported, never silently treated as a WAGON.

The model paths in the Python code are unchanged — the code still expects
these names in `./models`. Excluding them from Git changes nothing at
runtime; it only means you copy them in once per machine.

### Copying them onto EC2

Pick whichever you already have access to. From your **local machine**:

```bash
# scp (from the directory above wagon_count/)
scp -i /path/to/key.pem wagon_count/inputs/*.mp4 \
    ubuntu@<EC2_PUBLIC_IP>:~/wagon_count_global/inputs/
scp -i /path/to/key.pem wagon_count/models/*.pt \
    ubuntu@<EC2_PUBLIC_IP>:~/wagon_count_global/models/
```

```powershell
# Windows PowerShell equivalent
scp -i C:\path\to\key.pem C:\Users\Ajay\Desktop\other\wagon_count\inputs\*.mp4 `
    ubuntu@<EC2_PUBLIC_IP>:~/wagon_count_global/inputs/
scp -i C:\path\to\key.pem C:\Users\Ajay\Desktop\other\wagon_count\models\*.pt `
    ubuntu@<EC2_PUBLIC_IP>:~/wagon_count_global/models/
```

```bash
# rsync -- resumable, better for the ~150 MB of video
rsync -avzP -e "ssh -i /path/to/key.pem" \
    wagon_count/inputs/ ubuntu@<EC2_PUBLIC_IP>:~/wagon_count_global/inputs/
rsync -avzP -e "ssh -i /path/to/key.pem" \
    wagon_count/models/ ubuntu@<EC2_PUBLIC_IP>:~/wagon_count_global/models/
```

Total transfer: ~146 MB of video + ~67 MB of weights ≈ **213 MB**.

> This project contains no S3 code and none was added. If you later want an
> S3-based transfer, the only requirement is that the files end up at
> `<project>/inputs/*.mp4` and `<project>/models/*.pt` with the exact names
> above — or pass explicit paths with `--inputs-dir` / `--models-dir`.

---

## GitHub → EC2 Deployment

```
LOCAL (Windows / VS Code)
   ↓  verify videos + models      python validate_ec2.py
   ↓  git status / add / commit
   ↓  git push
GITHUB   github.com/AjayKhatik-s2/wagon_count_global
   ↓  git clone   (or git pull)
EC2 (Ubuntu)
   ↓  ./setup_ec2.sh              venv + dependencies
   ↓  copy the 4 .mp4 + 4 .pt into inputs/ and models/
   ↓  python validate_ec2.py      PASS / FAIL pre-flight
   ↓  python run_global_count.py
results/
```

### Step 1 — Local: verify before you commit

```powershell
cd C:\Users\Ajay\Desktop\other\wagon_count
python validate_ec2.py
```

Expect `RESULT: PASS`. A `WARN` about no CUDA GPU is normal on a laptop.

### Step 2 — Local: commit the source

```powershell
git status                       # confirm NO .mp4, NO .pt, NO results/
git add .
git status                       # review the staged list once more
git commit -m "Standalone Global Wagon Count: EC2 deployment preparation"
```

### Step 3 — Local: push to GitHub

```powershell
git branch -M main
git remote -v                    # confirm 'origin' points at the repo below
git push -u origin main
```

Remote: `https://github.com/AjayKhatik-s2/wagon_count_global.git`

### Step 4 — EC2: launch and connect

Recommended instance:

| Use case | Instance | Storage | Notes |
|----------|----------|---------|-------|
| CPU only | `t3.xlarge` (4 vCPU, 16 GB) | 60 GB gp3 | works; slowest |
| GPU      | `g4dn.xlarge` (T4, 16 GB) | 60 GB gp3 | markedly faster inference |

AMI: **Ubuntu Server 22.04 or 24.04 LTS**. 30 GB gp3 is ample for the code,
the virtualenv (torch is ~2.5 GB) and the runtime assets; a run's output is
only the JSON contracts plus a few-MB `combined_report.pdf`. Give it 60 GB if
you plan to use `--render-videos`, which adds ~100 MB per camera per run.

```bash
ssh -i /path/to/key.pem ubuntu@<EC2_PUBLIC_IP>
```

### Step 5 — EC2: clone the repository

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip
git clone https://github.com/AjayKhatik-s2/wagon_count_global.git
cd wagon_count_global
```

### Step 6 — EC2: set up the Python environment

```bash
chmod +x setup_ec2.sh
./setup_ec2.sh --with-apt-deps        # --with-apt-deps installs libgl1 + libglib2.0-0
source .venv/bin/activate
```

`setup_ec2.sh` creates `.venv`, installs `requirements.txt`, ensures
`inputs/ models/ results/` exist, verifies every import, and prints a
readiness report. It never deletes anything, never recreates an existing
`.venv`, and never downloads weights or videos.

### Step 7 — EC2: place the runtime assets

Copy the 4 videos and 4 weights in as described under
**Runtime assets** above, then confirm:

```bash
ls -lh inputs/ models/
```

### Step 8 — EC2: validate

```bash
python validate_ec2.py
```

This checks Python, numpy/OpenCV/torch/ultralytics, CUDA, the three
directories, all 4 videos (open + fps + frame count + resolution + first
frame decodes), all 4 weights (real ultralytics load + task + class names),
and write access — then prints PASS/FAIL and exits `0`/`1`. It does **not**
run the pipeline.

### Step 9 — EC2: run the pipeline

```bash
python run_global_count.py
```

For a long CPU run, keep it alive across a dropped SSH session:

```bash
sudo apt-get install -y tmux
tmux new -s wagon
python run_global_count.py 2>&1 | tee run.log
# detach: Ctrl-b then d      reattach: tmux attach -t wagon
```

Optional variants (all optional, algorithm unchanged):

```bash
python run_global_count.py --no-report                 # skip combined_report.pdf
python run_global_count.py --render-videos             # also write overlay MP4s
python run_global_count.py --report-dpi 300            # larger evidence images
python run_global_count.py --keep-evidence-frames      # keep the temp JPEGs
python run_global_count.py --no-raw-detections         # lower RAM
python run_global_count.py -o /mnt/data/results        # output to another volume
```

Results land in `results/` — see *Outputs* above. The evidence deliverable is
`results/combined_report.pdf`; the headline number is `total_wagons` in
`results/global_train_state.json`:

```bash
python -c "import json;d=json.load(open('results/global_train_state.json'));print('wagons:',d['total_wagons'],'fallback:',d['fallback_used'])"
```

### Step 10 — Updating EC2 after future code changes

```bash
cd ~/wagon_count_global
git pull
source .venv/bin/activate

# only if requirements.txt changed:
pip install -r requirements.txt

# re-run the pre-flight whenever code or dependencies changed:
python validate_ec2.py
python run_global_count.py
```

`git pull` never touches `inputs/`, `models/` or `results/` — they are
ignored by Git, so your assets and previous outputs survive every update.

---

## Recommended EC2 Python environment

| Component | Recommended on EC2 | Verified locally |
|-----------|--------------------|------------------|
| OS        | Ubuntu 22.04 / 24.04 LTS | Windows 11 |
| Python    | 3.10 – 3.12 (3.10 ships with 22.04, 3.12 with 24.04) | 3.14.5 |
| numpy     | ≥ 1.23 | 2.4.6 |
| opencv-python | ≥ 4.7 (or `opencv-python-headless`) | 4.13.0 |
| torch     | ≥ 2.0 | 2.12.0+cpu |
| torchvision | ≥ 0.15 (must match torch) | 0.27.0 |
| ultralytics | ≥ 8.0 | 8.4.53 |

Python **3.10 is the floor** (the code uses `from __future__ import
annotations` plus modern typing). Use the project virtualenv rather than
the system interpreter — Ubuntu 24.04 blocks system-wide `pip install`.

System packages needed by `opencv-python` on a headless server:

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

Or avoid them entirely — this project never calls `cv2.imshow()`:

```bash
pip uninstall -y opencv-python && pip install opencv-python-headless
```

### GPU (optional)

The pipeline runs correctly on CPU; a GPU only makes it faster. Ultralytics
selects CUDA automatically when `torch.cuda.is_available()` is true — no
code or flag change is needed. On a GPU instance (`g4dn`, `g5`, …), install
the CUDA build of torch **before** `requirements.txt` so pip keeps the CUDA
wheels:

```bash
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt        # loose floors -> CUDA wheels are kept
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

A Deep Learning AMI already ships CUDA drivers; on a plain Ubuntu AMI
install the NVIDIA driver first (`sudo apt-get install -y nvidia-driver-535`,
then reboot) and check with `nvidia-smi`.

---

## Packaging as a zip (alternative to Git)

From the directory **above** `wagon_count/`:

```powershell
# Windows PowerShell
Compress-Archive -Path wagon_count -DestinationPath wagon_count.zip -Force
```

```bash
# Linux/macOS
zip -r wagon_count.zip wagon_count -x 'wagon_count/results/*' \
                                   -x 'wagon_count/inputs/*' \
                                   -x 'wagon_count/models/*' \
                                   -x 'wagon_count/.venv/*'
```

The `-x` excludes keep the zip small — the instance gets videos and models
separately, exactly as in the Git workflow above.

---

## Phase-2 hook

The Phase-2 pipeline (door state, damage, OCR, loaded/empty, report)
consumes `results/global_train_state.json` — the canonical machine-readable
contract. For every `GW_n` it carries the master frame range and master time
window, so a downstream feature extractor can pull exactly the frames it
needs straight from the source videos using the same projection this project
uses (`video_segmenter.map_global_wagon_to_local_frames()`).

Same GW ids across cameras mean each downstream extractor can correlate
findings without re-running synchronization.

> Phase 1 no longer persists `results/frames/<CAMERA>/<GW_n>/` frame
> folders. They cost 1–2 GB per run, and the evidence they carried is now
> summarised in `combined_report.pdf`. A consumer that genuinely needs full
> frame sequences should derive them on demand from the JSON contract rather
> than have Phase 1 write them for every run.
