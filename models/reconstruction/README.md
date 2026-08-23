# models/reconstruction/

Stage-1 (global wagon counting) YOLO weights.

The counting engine resolves each weight by its **exact** filename under this
directory (overridable with `--recon-models-dir`).  There is no alias map and
no fallback name: a missing required file aborts the batch with a clear error
naming the exact path it looked at.

## How the files get here

They are **not** in Git — that is why a clean checkout has only this README.
`core/model_sync.py` fetches whatever is missing at startup from the configured
model store:

```
s3://$WAGONEYE_MODELS_S3_BUCKET/$WAGONEYE_MODELS_S3_PREFIX/<file>
   default: s3://complete-train/new_local/<file>   (ap-south-1)
   flat layout: every .pt side by side under the prefix
```

Required models missing from the store fail the run; optional ones log a
capability note and continue.  A file that is present locally is never
re-downloaded, so updating a model means replacing it on the box or deleting it
so the next run pulls a fresh copy.  To check the store against what a run needs
without downloading anything:

```
python -m core.model_sync
```

## Required (4)

| Filename                   | Task     | Used by                          |
|----------------------------|----------|----------------------------------|
| `right_up_wagon_gap.pt`    | detect   | RIGHT_UP (master) gap detection  |
| `left_up_wagon_gap.pt`     | detect   | LEFT_UP gap detection            |
| `top_gap.pt`               | detect   | RIGHT_UP_TOP + LEFT_UP_TOP gaps  |
| `side_classification.pt`   | classify | RIGHT_UP (counting authority) + LEFT_UP |

## Optional (2)

| Filename                   | Task     | Used by                          |
|----------------------------|----------|----------------------------------|
| `top_classification.pt`    | classify | RIGHT_UP_TOP                     |
| `ltop.pt`                  | classify | LEFT_UP_TOP                      |

Both let a TOP camera identify its own engine / brake-van regions so those
observations stay out of wagon synchronization, and let the overlay videos label
them.  Neither is **ever a counting authority** — RIGHT_UP alone decides the
count — so if one is absent the run continues with a note and the wagon count is
unaffected.

**The two top cameras do not share a classifier.**  They did until `ltop.pt` was
trained for LEFT_UP_TOP's own overhead view.  Which camera loads which file is
decided in exactly one place:

```
wagon_count/train_structure.py :: CAMERA_CLASSIFICATION_MODEL
```

Both the sequential (`orchestrator/camera_runner.py`) and batch
(`wagon_count/run_global_count.py`) paths read that table, so they cannot
disagree.  Do not add a second mapping, and do not "replace the top model" —
that would move RIGHT_UP_TOP too.

An absent classifier is never substituted with the other top camera's.  A
classifier run on the wrong camera still returns confident `engine` / `wagon` /
`brakevan` labels — just wrong ones — and those labels decide which segments are
excluded from wagon synchronization, so silent substitution is worse than no
classification at all.  Each mode logs its choice:

```
[MODEL] LEFT_UP_TOP  classification -> s3://complete-train/new_local/ltop.pt
                                    -> models/reconstruction/ltop.pt
[MODEL] RIGHT_UP_TOP classification -> .../top_classification.pt
```

Note the gap models are unaffected: **both** top cameras still use `top_gap.pt`.

## Do not substitute by filename

These are counting models.  Two of them carry class names that look like
inspection concerns but are not:

* `top_classification.pt` (and `ltop.pt`, its LEFT_UP_TOP counterpart) exposes a
  **`wagon_loaded`** class.  It is **not** a load-detection model — the counting
  engine maps `wagon_loaded -> WAGON` and never reads load status from it.  Load
  status comes from the inspection model in `models/features/`.
* `right_up_wagon_gap.pt` / `left_up_wagon_gap.pt` expose **`locono`** and
  **`engine_head`**.  These are **not** the OCR model — wagon-number reading
  is a separate inspection model in `models/features/`.

Verify a weight by its real `model.names`, never by its filename.
