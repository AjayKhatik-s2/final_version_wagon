# models/reconstruction/

Stage-1 (global wagon counting) YOLO weights.

The counting engine resolves each weight by its **exact** filename under this
directory (overridable with `--recon-models-dir`).  There is no alias map and
no fallback name: a missing required file aborts the batch with a clear error
naming the exact path it looked at.  Place the files below verbatim.

## Required (4)

| Filename                   | Task     | Used by                          |
|----------------------------|----------|----------------------------------|
| `right_up_wagon_gap.pt`    | detect   | RIGHT_UP (master) gap detection  |
| `left_up_wagon_gap.pt`     | detect   | LEFT_UP gap detection            |
| `top_gap.pt`               | detect   | RIGHT_UP_TOP + LEFT_UP_TOP gaps  |
| `side_classification.pt`   | classify | RIGHT_UP (counting authority) + LEFT_UP |

## Optional (1)

| Filename                   | Task     | Used by                          |
|----------------------------|----------|----------------------------------|
| `top_classification.pt`    | classify | RIGHT_UP_TOP + LEFT_UP_TOP       |

`top_classification.pt` lets the TOP cameras identify their own engine /
brake-van regions so those observations stay out of wagon synchronization, and
lets the overlay videos label them.  It is **never a counting authority** —
RIGHT_UP alone decides the count — so if it is absent the run continues with a
note and the wagon count is unaffected.

## Do not substitute by filename

These are counting models.  Two of them carry class names that look like
inspection concerns but are not:

* `top_classification.pt` exposes a **`wagon_loaded`** class.  It is **not** a
  load-detection model — the counting engine maps `wagon_loaded -> WAGON` and
  never reads load status from it.  Load status comes from the inspection
  model in `models/features/`.
* `right_up_wagon_gap.pt` / `left_up_wagon_gap.pt` expose **`locono`** and
  **`engine_head`**.  These are **not** the OCR model — wagon-number reading
  is a separate inspection model in `models/features/`.

Verify a weight by its real `model.names`, never by its filename.
