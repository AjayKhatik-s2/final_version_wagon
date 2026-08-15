# WagonEye v4 — Train-State-Native Production Pipeline

Single source of truth for wagon counting, numbering, and classification
is the **GlobalTrainState** produced by `wagon_count/`. Everything
downstream — frame extraction, feature inference, fusion, reporting —
consumes that state and never recounts wagons, never re-segments video,
never re-runs gap detection.

```
4 source videos
    │
    ▼
[Stage 1] reconstruction.runner  →  global_state/global_train_state.json
            SOLE counting authority (wagon_count/, fixed-master fusion):
              fragment reassembly → gap validation → WAGON_ACTIVE recovery
              → master classification + temporal smoothing
              → per-camera clock-offset estimation
              → global gaps == validated RIGHT_UP gaps
              → wagon window (ENGINE / BRAKE_VAN get no GW id)
            Emits one IMMUTABLE roster GW_1..GW_N.  No later stage may
            count, re-segment, renumber or reorder it.
            + wagon_count tracking overlay mp4s (debug artifacts)
            → global_state/processed_videos/<CAM>_processed.mp4
    │
    ▼
[Stage 2] materializer.wagon_cache_builder
            single-pass per video → wagon_cache/<GW_n>/<camera>/*.jpg
    │
    ▼
[Stage 3] features.load  runs FIRST, to completion
            (damage reads the load JSON to drop floor_damage on LOADED
             wagons; running it first makes that read deterministic)
          then features.{door,ocr,damage} in parallel
            pure YOLO inference on cached frames
            persists evidence snapshots + metadata under evidence/
            → wagon_states/<feature>/<GW_n>.json
            → evidence/<GW_n>/<feature>/{*.jpg,metadata.json}
    │
    ▼
[Stage 4] fusion.wagon_state_builder
            authority-rule merge → wagon_states/unified/<GW_n>.json
    │
    ▼
[Stage 4b] rendering.feature_overlay_renderer  (visualization-only,
            consumes state + evidence + tracking JSON; never reruns
            any detector model)
            → processed_videos/<CAM>_processed.mp4
    │
    ▼
[Stage 5a] reporting.camera_reports   (legacy camera-wise hierarchy)
            4 camera-wise PDFs, each showing ONLY what that camera is
            authoritative for:
              RIGHT_UP     → right door, OCR, classification
              LEFT_UP      → left door
              RIGHT_UP_TOP → load, top damage
              LEFT_UP_TOP  → top-damage support / validation
            Each report: Camera Summary page (visible wagons, anomalies,
            processing confidence, coverage %), per-wagon pages (2×2
            quartile overview + per-detection detail pages, snapshots
            from THAT camera only), Anomaly Summary grouped by severity,
            and a Camera Evidence section.
            → reports/{right_up,left_up,right_up_top,left_up_top}_report.pdf
    │
    ▼
[Stage 5b] reporting.combined_train_report
            Aggregates the 4 camera reports into the unified train view
            (legacy CombinedReportGenerator visual identity, rebuilt
            against v4 state via _adapter.LegacyViewModel): navy title
            banner, VIDEO EVIDENCE table, PARTIAL REPORT banner,
            DETAILED CAMERA REPORTS links (LEFT/RIGHT/R-TOP/L-TOP),
            10-col INSPECTION SUMMARY KPI row (LOCO, RAKE TYPE, STATUS),
            7-col wagon inspection table with issue-row highlighting,
            and the "Damaged Wagon Report" per-anomaly evidence grid
            (camera-priority ordered).  Schema v4 JSON includes
            legacy_view_model.
            → reports/combined_train_report.{pdf,json}
    │
    ▼
[Stage 6] delivery.s3_upload + delivery.notification
            S3 archive (incl. evidence + processed_videos) +
            one email per batch
```

## Package layout

```
wagon_eye_v4/
├── README.md                          (this file)
├── requirements.txt
├── pytest.ini                         test collection scope
├── orchestrator/master_runner.py      ★ entry point
├── reconstruction/runner.py           Stage 1 (subprocess wagon_count)
├── materializer/wagon_cache_builder.py Stage 2
├── features/
│   ├── _common.py                     YOLO loader cache + helpers
│   ├── _evidence.py                   evidence persistence helpers
│   │                                   (per-wagon JPEG + metadata)
│   ├── inference_lib/                 mature tracking intelligence ported
│   │                                   from the legacy system (door tracker,
│   │                                   identity merger, illumination, shape
│   │                                   prior, damage tracker, OCR + number
│   │                                   aggregator, temporal reasoning)
│   ├── door/processor.py              door_state.pt
│   ├── load/processor.py              loaded.pt
│   ├── damage/processor.py            damage.pt
│   └── ocr/processor.py               wagon_id_counting.pt + easyocr
├── fusion/wagon_state_builder.py      Stage 4
├── rendering/
│   └── feature_overlay_renderer.py    Stage 4b (visualization-only;
│                                       never reruns any detector)
├── reporting/
│   ├── _brand.py                      Legacy WagonEye palette, paragraph
│   │                                   styles, anomaly + state helpers,
│   │                                   page widgets (logo, warning banner,
│   │                                   camera links).
│   ├── _adapter.py                    v4 backend -> legacy report view-
│   │                                   model (merged_wagons + per-camera
│   │                                   doors + KPIs).
│   ├── _evidence_lookup.py            Quartile + midpoint cache frame
│   │                                   resolution + evidence snapshot
│   │                                   path helpers.
│   ├── _pages.py                      Shared reportlab page widgets
│   │                                   (doc maker, bordered image,
│   │                                   detection-summary table, wagon
│   │                                   quartile overview, detail page,
│   │                                   simple-state page).
│   ├── camera_reports.py              Stage 5a (4 camera-wise PDFs by
│   │                                   camera authority: RIGHT_UP /
│   │                                   LEFT_UP / RIGHT_UP_TOP /
│   │                                   LEFT_UP_TOP).
│   ├── combined_train_report.py       Stage 5b (aggregates the 4 camera
│   │                                   reports; legacy visual identity:
│   │                                   navy title banner, 10-col KPI
│   │                                   summary, 7-col wagon table,
│   │                                   Damaged Wagon Report evidence
│   │                                   grid; schema v4).
│   └── assets/Logo.jpeg               Per-page logo (copied from the
│                                       legacy product).
├── delivery/
│   ├── s3_upload.py                   Stage 6: PDF/JSON + tree upload
│   └── notification.py                Stage 6: one email per batch
├── core/
│   ├── constants.py                   camera ids, classes, statuses
│   ├── unified_wagon_state.py         UnifiedWagonState dataclass
│   ├── global_state_loader.py         ★ THE counting→inspection adapter:
│   │                                   parses the engine JSON, freezes the
│   │                                   roster, exposes camera offsets,
│   │                                   roster fingerprint + integrity check
│   ├── feature_config.py              Stage-3 feature registry + toggles
│   ├── frame_quality.py               evidence-frame scoring heuristics
│   └── batch.py                       CameraVideo / TrainBatch
├── models/
│   ├── reconstruction/                Stage-1 counting weights, EXACT names
│   │                                   (right_up_wagon_gap.pt,
│   │                                    left_up_wagon_gap.pt, top_gap.pt,
│   │                                    side_classification.pt,
│   │                                    top_classification.pt = optional)
│   └── features/                      Stage-3 inspection weights
│                                       (door_state.pt, loaded.pt,
│                                        damage.pt, wagon_id_counting.pt)
├── tests/                             v4 integration + counting-swap
│                                       regression suite (70 tests)
├── _legacy_wagon_count_removed/       the REPLACED counting modules, kept
│                                       for review only.  Not a package,
│                                       not importable, asserted unreferenced.
│                                       Safe to delete after review.
└── wagon_count/                       ★ the counting engine — adopted
    ├── run_global_count.py             byte-identical from the proven
    ├── tracker_engine.py               correct-count implementation
    ├── fragment_stitching.py           tracker fragments → physical gaps
    ├── gap_validation.py               candidate → validated boundary
    ├── temporal_classification.py      class-sequence hysteresis
    ├── train_structure.py              wagon window (no GW id for
    │                                    ENGINE / BRAKE_VAN)
    ├── global_fusion.py                fixed-master fusion + camera offsets
    ├── global_alignment.py             segment build (+ retained legacy path)
    ├── global_train_state.py           engine-side data contracts
    ├── video_segmenter.py              Stage-1 debug overlay videos
    ├── evidence_report.py              engine's own PDF (skipped by v4)
    ├── rejection_report.py             gap-rejection CSV/JSON tooling
    └── tests/                          the engine's own suite (283 tests)
```

There are NO `RIGHT_UP/`, `LEFT_UP/`, `RIGHT_UP_TOP/`, `LEFT_UP_TOP/`
folders inside this package. Camera-centric assumptions, legacy
`DoorProcessor` / `DamageProcessor` wrappers, `cv2.VideoCapture` calls
downstream of Stage 1, and mini-mp4 reconstruction are all removed.

## Output per batch

```
batch_outputs/<batch_key>/
├── downloads/                         raw videos (downloaded from S3, or
│                                       local-passthrough)
├── global_state/
│   ├── global_train_state.json
│   ├── per_camera_tracking.json
│   └── processed_videos/              wagon_count's debug tracking videos
│       └── <CAM>_processed.mp4         (4 cameras)
├── wagon_cache/
│   ├── GW_1/{right_up,left_up,right_up_top,left_up_top}/frame_*.jpg
│   ├── GW_2/...
│   └── ...
├── wagon_states/
│   ├── door/GW_*.json
│   ├── load/GW_*.json
│   ├── damage/GW_*.json
│   ├── ocr/GW_*.json
│   └── unified/GW_*.json
├── evidence/                          best-frame snapshots + metadata
│   ├── GW_1/door/{left_best,left_crop,right_best,right_crop}.jpg
│   │       + metadata.json
│   ├── GW_1/damage/{track_1,track_1_crop,...}.jpg + metadata.json
│   ├── GW_1/ocr/{best_frame,number_crop}.jpg + metadata.json
│   ├── GW_1/load/best_frame.jpg + metadata.json
│   └── GW_2/...
├── processed_videos/                  rich feature-overlay videos
│   └── <CAM>_processed.mp4             (4 cameras; rendered from
│                                        state+evidence — no detector rerun)
├── reports/
│   ├── combined_train_report.json     schema v4 (legacy_view_model +
│   │                                   evidence_pages)
│   ├── combined_train_report.pdf      legacy WagonEye visual identity:
│   │                                   title banner + KPIs + wagon table +
│   │                                   Damaged Wagon Report
│   ├── right_up_report.pdf            camera-wise reports (Stage 5a):
│   ├── left_up_report.pdf              one per camera, each scoped to
│   ├── right_up_top_report.pdf         that camera's authority
│   └── left_up_top_report.pdf
└── archive/                           run logs (future)
```

## Quick start

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Drop the .pt model files into (exact filenames — no aliases):
#       wagon_eye_v4/models/reconstruction/     (4 required + 1 optional)
#           right_up_wagon_gap.pt
#           left_up_wagon_gap.pt
#           top_gap.pt
#           side_classification.pt
#           top_classification.pt   OPTIONAL — refines TOP-camera
#                                   classification; never a counting
#                                   authority, so a missing file only
#                                   degrades labelling, never the count.
#       wagon_eye_v4/models/features/
#           door_state.pt
#           loaded.pt
#           damage.pt
#           wagon_id_counting.pt

# 3) Local single-batch (no S3):
mkdir -p wagon_eye_v4/local_inputs
# copy 4 trimmed train videos in -- filenames must contain
# 'right_up' / 'left_up' / 'right_up_top' / 'left_up_top'
cd wagon_eye_v4
python -m orchestrator.master_runner --local-only --local-inputs ./local_inputs

# 4) Continuous S3 polling (production):
python -m orchestrator.master_runner --auto

# 5) Single-batch replay:
python -m orchestrator.master_runner --batch 20260408_032134
```

## Tests

```bash
cd wagon_eye_v4
python -m pytest -q                      # 353 tests (v4 + counting engine)
python -m unittest discover -s tests     # 70 tests (v4 only, stdlib runner)
```

No model weights, no video decode and no GPU are needed: the counting modules
are pure stdlib, and the suites drive them with synthetic tracker output.
Nothing here runs production inference.

| Suite | Covers |
|---|---|
| `tests/test_counting_engine_swap.py` | the new engine is wired in; the old one is unreachable; no aliasing/download logic |
| `tests/test_global_roster_contract.py` | deterministic `GW_1..GW_N`, no duplicates/gaps, count independent of support cameras |
| `tests/test_roster_immutability.py` | inspection cannot append / remove / reorder / renumber / re-time the roster |
| `tests/test_downstream_contract.py` | door/load/damage/OCR + fusion + reporting still receive the expected structure |
| `tests/test_counting_integration.py` | four-camera input → engine → JSON → adapter → inspection; camera offsets reach materialization |
| `wagon_count/tests/` | the counting engine's own 283-test suite, adopted with it |

Tests that compare the engine against its source folder skip cleanly when
`wagon_count - Copy_correct_count/` is absent (it is gitignored).

## Run-mode flags

| Flag                        | Effect                                                |
|-----------------------------|-------------------------------------------------------|
| `--auto`                    | Continuous S3 polling.                                |
| `--once`                    | Process one batch then exit.                          |
| `--batch <key>`             | Replay / debug a specific batch_key.                  |
| `--local-only`              | Skip S3; use `--local-inputs` instead.                |
| `--local-inputs DIR`        | Folder to scan for the 4 videos.                      |
| `--workspace DIR`           | Output root (default: `./batch_outputs`).             |
| `--recon-models-dir DIR`    | Override `models/reconstruction/`.                    |
| `--feat-models-dir DIR`     | Override `models/features/`.                          |
| `--skip-upload`             | Don't upload PDF/JSON, don't archive to S3.           |
| `--skip-email`              | Don't send the combined email.                        |
| `--poll-interval N`         | Continuous-mode S3 poll interval (seconds).           |
| `--partial-wait N`          | Wait this many minutes for missing cameras.           |
| `--disable-features LIST`   | Comma-separated feature keys to turn OFF (`door,ocr,load,damage`); skips the interactive prompt. |
| `--no-interactive`          | Never prompt for feature config (force all features ON unless `--disable-features` is given). |

## Feature configuration (enable/disable)

Before Stage 3 runs, you can choose which feature processors execute. The
feature set is **registry-driven** (`core/feature_config.py`) — adding a
feature there makes it appear here automatically.

Three ways to configure, in precedence order:

1. **CLI (non-interactive, scriptable):**
   ```bash
   python -m orchestrator.master_runner --local-only --local-inputs ./local_inputs \
          --disable-features ocr,damage
   ```
2. **Interactive prompt** — shown automatically on foreground runs
   (`--local-only` / `--once` / `--batch`) **only when stdin is a real
   terminal**:
   ```
   Current Feature Configuration:
     [ON]  Door
     [ON]  OCR
     [ON]  Load
     [ON]  Damage
   Turn OFF any feature? (y/n): y

   Select feature(s) to turn OFF (comma-separated numbers, e.g. 2,4):
     1. Door
     2. OCR
     3. Load
     4. Damage
   Disable: 2,4

   Final Feature Configuration:
     [ON]  Door
     [OFF] OCR
     [ON]  Load
     [OFF] Damage
   ```
3. **Default** — every feature ON. Continuous `--auto` polling, piped, and
   cron runs **never prompt** (safe for unattended operation); pass
   `--disable-features` to change the set there.

A disabled feature is skipped in Stage 3 and its per-wagon state is marked
`DISABLED_BY_USER`. In the camera-wise and combined reports its fields read
**`DISABLED BY USER`** (never flagged as an anomaly), and its overlay is
never drawn in the processed videos.

## Failure handling

| Failure                                  | Outcome                                            |
|------------------------------------------|----------------------------------------------------|
| Stage 1 reconstruction errors / 0 wagons | Batch marked `failed_no_global_state`. Abort.      |
| Stage 1 required counting model missing  | Engine exits non-zero naming the exact expected path. Batch marked `failed_no_global_state`. Abort. |
| Stage 1 malformed roster (duplicate / non-contiguous `GW_n`, count mismatch) | Rejected by `verify_roster_integrity()` at the Stage-1 boundary. Batch marked `failed_no_global_state`. Abort. |
| Inspection mutates the roster            | `RosterImmutabilityError` at the next stage guard. Loud failure, never silent corruption. |
| Stage 2 cannot open one video            | That camera's wagon_cache subtree empty. Continue. Batch is `completed_partial`. |
| Stage 3 one feature processor crashes    | Its per-wagon JSONs marked `FAILED`. Other features continue. |
| Stage 3 one wagon fails in one feature   | That `(feature, GW_n)` JSON marked `FAILED`. Rest unaffected. |
| Stage 4 fusion error per wagon           | Unified state for that wagon partial. Report still generated. |
| Stage 4b overlay renderer fails per cam  | That camera's mp4 missing; other cameras continue. Combined PDF still generated. |
| Stage 5a one camera report crashes       | That camera's PDF missing; other 3 camera reports + combined report unaffected. |
| Stage 5b combined PDF crashes            | Batch marked `report_failed`. JSON still written.  |
| PDF microservice down                    | S3 direct-upload fallback URL used.                |
| Email API down                           | Logged; batch outcome still persisted.             |

## Authority rules (fusion)

| Field                  | Authority                       |
|------------------------|---------------------------------|
| classification         | GlobalTrainState (RIGHT_UP)     |
| wagon_identifier (OCR) | RIGHT_UP only                   |
| right_door             | RIGHT_UP                        |
| left_door              | LEFT_UP                         |
| load_status            | RIGHT_UP_TOP (LEFT_UP_TOP fall) |
| top_damage             | any TOP camera reporting DAMAGE |

## Counting authority (Stage 1)

`wagon_count/` is the **only** component allowed to detect gaps, count, assign
wagon ids or define wagon boundaries.  It runs fixed-master fusion:

| Property | Guarantee |
|---|---|
| Global gap sequence | **exactly** the validated RIGHT_UP gaps |
| Support cameras | evidence + synchronization only; they can never create, delete, split or merge a global gap |
| Raw YOLO gap | a *candidate* — must pass motion / persistence / trajectory / duplicate validation |
| Camera clocks | per-camera offset estimated (`t_global = t_local + delta`), `RESOLVED` or `UNRESOLVED`; unresolved contributes 0.0, i.e. the shared-`t=0` assumption |
| Roster contents | **WAGON units only.**  ENGINE and BRAKE_VAN are preserved as `wagon_window` metadata but never receive a `GW_n` id and never extend the wagon timeline |
| Immutability | the roster is frozen the moment Stage 1 returns; `orchestrator` re-checks `roster_fingerprint()` after Stages 2, 3, 4 and 5 |

`core/global_state_loader.py` is the single adapter between this engine and
everything downstream — nothing else parses the counting engine's JSON.

### What "WAGON units only" changes in the output

`total_wagons` now counts **wagons**, not wagons + loco + brake van, so it
reads lower than a pre-swap run on the same train. Where the engine/brake-van
counts surface:

| Consumer | Value | Source |
|---|---|---|
| `GlobalTrainState.engine_count` / `.brake_van_count` | preserved | adapter reads `wagon_window` structure metadata |
| `reports/combined_train_report.json` → `legacy_view_model.summary_kpis.engine_count` | preserved | via the adapter above |
| `reports/combined_train_report.json` → `summary.engine_count` | **0** | `summarize_wagons()` counts over the roster, which is wagon-only |
| PDF "LOCO NUMBER" column | `Not Detected` | reads `summary_kpis.loco_numbers`, hard-coded `[]` in `_adapter.py` — unchanged by this swap, it behaved this way before |

The `summary` / `legacy_view_model` disagreement is pre-existing plumbing, left
as-is deliberately: reconciling it would mean editing the inspection/report
layer, which is out of scope for a counting-engine swap.

A camera that cannot see a given global wagon contributes **no frames** for it
(rather than clamping onto an unrelated frame); the feature processors then
report that camera's existing `NO_FRAMES` status.  No camera ever invents a
local wagon number.

## Constraints honored

- ❌ No `DoorProcessor.process_video()` or `DamageProcessor.process_video()`
  wrappers anywhere.
- ❌ No mini-mp4 reconstruction.
- ❌ No `cv2.VideoCapture` for **inference** outside Stage 2 (materializer)
  and Stage 1 (wagon_count subprocess).  Stage 4b's overlay renderer uses
  `cv2.VideoCapture` strictly for **visualization** — it never invokes any
  detector / YOLO / OCR / tracking model; everything it draws comes from
  already-persisted `GlobalTrainState`, `UnifiedWagonState`, evidence
  metadata, and `per_camera_tracking.json`.
- ❌ No per-camera folders (`RIGHT_UP/` etc.) in this package.
- ❌ No `wagon_gap.pt` recompute downstream of Stage 1.
- ✅ All models in a single centralized tree (`models/{reconstruction,features}/`).
- ✅ Frames extracted exactly once.
- ✅ GlobalTrainState is the immutable backbone — enforced, not just intended
  (frozen wagons, tuple roster, per-stage fingerprint guard).
- ❌ No per-camera independent wagon count is ever the final authority.
