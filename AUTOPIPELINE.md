# Production auto-pipeline runbook

One command, one service: raw CCTV in → reports + V4 dashboard feed out.

```bash
python -m orchestrator.master_runner --auto --source raw --no-interactive
```

```
RAW CCTV (S3)
    │   biro-wagon-raw-video-copy/camera_CCTV_HZBN_DHN_*/
    ▼
[ExtractionManager]  detect a train pass → trim → upload      (--source raw only)
    │   biro-wagon-pre-processed-video-copy/camera_CCTV_HZBN_DHN_*/
    ▼
[train_batch_manager]  cluster the 4 cameras into one TrainBatch
    ▼
Stage 1   seal GlobalTrainState (RIGHT_UP canonical count + numbering)
Stage 2   materialize wagon_cache
Stage 3   features: load → {door, ocr, damage}   (ocr = AWS Rekognition)
Stage 4   fusion → unified wagon states
Stage 4b  overlay videos
Stage 5   4 camera PDFs + combined PDF/JSON
Stage 6   S3 archive · ONE email
Stage 6b  4 exact-V4 *_inspection.json → dashboard ingest + ML API callback
```

Both halves are decoupled through S3 — the extractor's trimmed bucket **is** the
consumer's `WAGONEYE_S3_INPUT_BUCKET`. That is why one process can own both
without any code coupling, and why `--source trimmed` (the default) still runs as
the pure consumer it always was.

---

## What changed in this package

| Area | Before | Now |
|---|---|---|
| `--auto` / `--once` / `--batch` | **dead** — imported `train_batch_manager` from the repo's *parent*, which does not exist here, so it always printed "continuous polling unavailable" and returned 3 | works — `orchestrator/train_batch_manager.py` lives in the package |
| Train extraction | none (trimmed clips had to arrive some other way) | `train_extraction/` + `orchestrator/extraction_manager.py`, opt-in via `--source raw` |
| Wagon-number OCR | easyocr only | **AWS Rekognition by default** (V4's algorithm); easyocr retained via `--ocr-engine easyocr` |
| Dashboard | nothing left the box but the PDF/JSON + email | 4 exact-V4 per-camera `inspection_data.json`, uploaded and POSTed to the V4 ingest receivers + the ML API |
| Model weights | had to be placed by hand | auto-synced from `s3://wagon-eye-models/` when missing (two ambiguous filenames excepted, see below) |
| Config | constants hardcoded in `core/constants.py` | every value env-overridable; `core/config.py` adds paths, device, the operational-day anchor |

Unchanged: Stage 1–5 behaviour, the counting engine, the sampled-inference
defaults (door/damage/load = 3/3/2), `--mode sequential`, and every existing
test.

---

## The V4 API set

Taken from V4's **committed** `Train-Inspection-Engine/configs/config.json`.

| Call | Endpoint |
|---|---|
| Ingest PROD | `https://ms-pnr-location-notification-api.suvidhaen.com/cctv-receiver/inspections/ingest` |
| Ingest UAT | `https://cctv-wagon-uat-api.suvidhaen.com/inspections/ingest` |
| ML API | `https://ms-pnr-location-notification-api.suvidhaen.com/cctv-receiver/api/v1/ml` |
| Email | `https://ms-pnr-location-notification-api.suvidhaen.com/notification_microservice/send-email` |
| Inspection-JSON bucket | `test-inspection-artifacts-sarva` (V4's `ARTIFACT_BUCKET`) |

> **Do not copy the endpoints out of V4's `core/config.py` dataclass.** Those
> defaults are stale — they point at `cctv-wagon-api.suvidhaen.com`, and V4's own
> commit *"Match notebook artifact + JSON contract; fix flush-emit + endpoint
> URLs"* replaced them with the values above. Posting to the stale host returns
> 2xx and the report never appears. `tests/test_v4_dashboard_feed.py` asserts
> that no endpoint here uses that host.

Each of the four documents is POSTed to **both** receivers, exactly as V4's
`trigger_db_ingestion_dual` does, with exactly V4's three body fields:

```json
{"inspection_s3_uri": "s3://test-inspection-artifacts-sarva/<camera_folder>/<ts>/inspection_data.json",
 "camera_id": "camera_CCTV_HZBN_DHN_2_RIGHT_UP",
 "version": "v1"}
```

The idempotency key travels as an `Idempotency-Key` **header**, never as a body
field — an unknown body field is a payload divergence this receiver can reject
with 422.

### `version` selects the dashboard tab

The PROD ingest URL is byte-identical for the V1 and V4 dashboards; the **only**
thing choosing the view is the document's `version`. It defaults to `v1`, so
reports land in the **V1 tab**.

`version` also selects the JSON **dialect**, because "exact V4 JSON" and "renders
in the V1 tab" genuinely disagree on five nested details:

| detail | `v1` (default) | `v4` |
|---|---|---|
| `bounding_box` | `{bounding_box_coordinates, confidence, class_name}` | `[x1,y1,x2,y2]` |
| open-door `problem_type` | `door_open` | `open_door` |
| `problem_frames_by_type` | `{damage, door_open}` | + `closed_door`, `partially_closed` |
| problem `segment_number` | the wagon count | `null` (side) |
| side `rake_status` | right-to-left = Loaded | left-to-right = Loaded |
| `camera_id` | keeps `camera_` prefix | prefix stripped |
| `damage_model_active`, `doors_partially_closed` | absent | present |

The last row is why a v1 document is not byte-identical to a v4 one: emitting V4
shapes at a V1 consumer breaks it. Set `WAGONEYE_INSPECTION_VERSION=v4` to emit
the V4 dialect (and land in the V4 tab) instead.

---

## Buckets — identical to the V4 engine

Every bucket is a built-in default taken from V4's `configs/cameras/*.yaml` and
`configs/combiner.yaml`, so an empty env file already points at the existing
production topology.

| V4 config key | Bucket | Used by |
|---|---|---|
| `raw_video_bucket` | `biro-wagon-raw-video-copy` | extraction input (`--source raw`) |
| `trimmed_video_bucket` | `biro-wagon-pre-processed-video-copy` | extraction output **and** `--auto` input |
| `detected_video_bucket` | `biro-wagon-processed-video-copy` | overlay-video mirror |
| `inspection_output_bucket` | `biro-wagon-report-biro-copy` | reports, evidence, archive |
| `combined_output_bucket` | `biro-combined-report-copy` | combined report |
| `ARTIFACT_BUCKET` | `test-inspection-artifacts-sarva` | the 4 inspection JSONs |
| *(models)* | `wagon-eye-models` (flat, bucket root) | missing-model auto-sync |
| `region` | `ap-south-1` | everything |

`<camera_folder>` is one of `camera_CCTV_HZBN_DHN_{2_RIGHT_UP, 1_LEFT_UP,
5_RIGHT_TOP, 6_LEFT_TOP}`, defined once in `core.constants.CAMERA_S3_FOLDER` and
shared by extraction, discovery, the report layout and the dashboard feed — so a
rig rename is a one-line edit and producer and consumer can never drift apart.

> The site names its top rigs `RIGHT_TOP` / `LEFT_TOP`, but the canonical camera
> ids are `RIGHT_UP_TOP` / `LEFT_UP_TOP`. `C.camera_from_key()` maps both, for S3
> keys and local filenames alike. Matching only the canonical ids silently
> dropped every top-camera clip, and batches formed with two cameras.

---

## Models

```
models/reconstruction/   right_up_wagon_gap.pt  left_up_wagon_gap.pt  top_gap.pt
                         side_classification.pt  top_classification.pt (optional)
models/features/         door_state.pt  loaded.pt  damage.pt
                         wagon_number_update.pt        ← V4's plate detector
models/extraction/       side_classification.pt  top_classification.pt
                                                       ← ONLY for --source raw
```

`wagon_number_update.pt` is the canonical OCR detector; if only the older
`wagon_id_counting.pt` is present it is used automatically.

> **Filename collision.** `side_classification.pt` and `top_classification.pt`
> exist in **both** `models/reconstruction/` and `models/extraction/` with
> **different weights** (Stage-1 segment classifier vs. train-presence
> classifier). A flat bucket cannot tell them apart, so these two are never
> auto-downloaded — startup names them and you place them yourself. Never copy
> one over the other.

---

## Flags

| Flag | Effect |
|---|---|
| `--source trimmed` \| `raw` | what this deployment consumes. `raw` also runs extraction. Default `trimmed` (or `WAGONEYE_PIPELINE_SOURCE`). |
| `--ocr-engine rekognition` \| `easyocr` | wagon-number reader. Default `rekognition`. |
| `--skip-model-sync` | skip the startup availability check / S3 sync. |
| `--auto` / `--once` / `--batch <key>` | continuous / one batch / replay. |
| `--partial-wait N` | minutes to wait for missing cameras before running partial. |
| `--skip-upload` / `--skip-email` | `--skip-upload` also makes the dashboard feed a **dry run**: documents are written under `<batch>/delivery/dashboard/` and nothing is uploaded or POSTed. |

## Key environment variables

```bash
WAGONEYE_PIPELINE_SOURCE=raw            # or trimmed (default)
WAGONEYE_OCR_ENGINE=rekognition         # or easyocr
WAGONEYE_DASHBOARD_INGEST_ENABLED=true  # default true -- posts to LIVE receivers
WAGONEYE_INSPECTION_VERSION=v1          # dashboard tab + JSON dialect
WAGONEYE_INSPECTION_INGEST_API_URLS=v4  # v4 | prod | uat | explicit CSV
WAGONEYE_INSPECTION_KEY_LAYOUT=v4       # v4 (V4's key) | v1 (legacy key)
WAGONEYE_INSPECTION_JSON_BUCKET=...     # default test-inspection-artifacts-sarva
WAGONEYE_ML_API_ENABLED=true            # the V4 ML callback
WAGONEYE_ML_API_SECRET=...              # sent as X-ML-SECRET
WAGONEYE_PROCESSOR_START_UTC=...        # one-time backlog skip (ISO 8601)
WAGONEYE_EXTRACTION_POLL_INTERVAL=60
```

> The dashboard feed is **enabled by default and posts to the live UAT + PROD
> receivers**. Use `--skip-upload` for a dry run, or
> `WAGONEYE_DASHBOARD_INGEST_ENABLED=false` to disable it.

---

## Discovery is anchored to the operational day

Everything discovered — raw clips and trimmed clips — is bounded by the start of
the current operational day: **05:00 IST**, rolling back a day before 05:00.

* A restart at any hour still sees the whole operational day, so stopping
  overnight and starting at 05:30 loses nothing — a sliding "last N minutes"
  window would skip every train uploaded while the service was down.
* It is inherently bounded to one day, so it can never reach back into months of
  archive and queue thousands of batches.
* It matches the 05:00 boundary the dashboard already uses for its date folders,
  so a train and its report always agree about which day they belong to.

`WAGONEYE_PROCESSOR_START_UTC` raises the anchor for a one-time backlog skip;
it can never lower it, and it self-expires at the next day's anchor.

---

## Startup behaviour

`--auto` validates the effective configuration and **refuses to poll** on any
error, rather than failing once a minute forever:

```
[ORCH] refusing to start -- configuration errors:
  * PIPELINE_SOURCE=raw but the extraction models dir does not exist: .../models/extraction
    (set WAGONEYE_EXTRACTION_MODELS_DIR, or use --source trimmed)
```

Then it prints a redacted effective-config summary (recipients as counts only)
and verifies every model the run needs, naming each missing file and the exact
`s3://` key it looked for.

## Failure isolation

| Failure | Outcome |
|---|---|
| Extraction sweep crashes for one camera | logged; other cameras sweep; retried next cycle |
| Rekognition unavailable / throttled | that wagon's OCR is `NO_DATA`; every other feature still reports. `--ocr-engine easyocr` runs with no network |
| Ingest receiver down / 4xx / 5xx | recorded per endpoint in `delivery/finalization.json`; a document counts as ingested if **either** receiver accepts it; retried on a later run |
| ML API down | logged; batch outcome still persisted |
| Re-run of an already-delivered batch | skipped — per-camera status is keyed by the document's sha256, so no duplicate uploads or POSTs |

## Tests

```bash
python -m pytest -q       # 674 passed, 2 skipped
```

No weights, no video, no AWS credentials, no network: the receiver, S3 client and
Rekognition client are all stubbed.

| Suite | Covers |
|---|---|
| `tests/test_v4_dashboard_feed.py` | endpoints are V4's (and not the stale host), the 3-field payload, dual receivers, `version`/dialect rules, V4's S3 key layout, per-camera narrowing of fused feature files, flat/nested evidence, idempotency, end-to-end run |
| `tests/test_v4_rekognition_ocr.py` | digit reading, two-row plates, sheet assembly, loaded/empty triplet order, banding, call budget, engine selection, detector/load-state resolution |
| `tests/test_auto_pipeline_wiring.py` | source resolution, the batch-manager call contract, the operational-day anchor, config validation, ExtractionManager lifecycle, bucket/model inventory |

## Not yet proven

Every stage above is unit-tested and the delivery contract is asserted against
V4's committed config, but **this has not been run end-to-end against live S3 or
a real four-camera train** from this package. The first live run should use
`--skip-upload` (dry run: builds the four documents locally, posts nothing) and
the documents under `<batch>/delivery/dashboard/` should be diffed against a
known-good V4 `inspection_data.json` before enabling live delivery.
