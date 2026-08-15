# GLOBAL_FUSION_DESIGN

**STATUS: IMPLEMENTED** (Phases 3–7 complete; no full production run yet).

Companion documents: `CURRENT_FUSION_ANALYSIS.md` (Phase 1, what was wrong),
`WAGON_COUNT_LOGIC.md` (behavioural reference).

## IMPLEMENTATION SUMMARY

| | |
|---|---|
| New modules | `global_fusion.py`, `gap_validation.py`, `train_structure.py` |
| New tests | `tests/` — **92 tests, all passing** (fusion 31, gap validation 27, train structure 34) |
| Modified | `run_global_count.py`, `global_train_state.py`, `video_segmenter.py`, `evidence_report.py`, `validate_ec2.py` |
| Unchanged | `tracker_engine.py` (verified byte-identical), `global_alignment.build_global_wagons` (reused as-is) |
| Default fusion | `--fusion master-fixed`; `--fusion legacy` retained for A/B only |

### The counting chain, as built

```
raw YOLO gap detections        (candidates only)
        |   tracker_engine.GapTracker  -- UNCHANGED
tracked gap candidates
        |   gap_validation.validate_gap_events        <-- NEW
valid gap events               (motion / persistence / trajectory / duplicates)
        |   build_global_wagons  -- REUSED UNCHANGED
segments + classification
        |   train_structure.get_master_wagon_window   <-- NEW
WAGON WINDOW: first WAGON .. last WAGON
        |
GW_1 .. GW_N                   == total_wagons == master_wagon_count
```

Two invariants now hold together:

```
global_gaps  == validated RIGHT_UP gaps          (fixed-master)
total_wagons == WAGON units of the wagon window  (wagon-only)
```

ENGINE and BRAKE_VAN are preserved in `wagon_window`, the PDF's train-structure
block and the overlay videos, but never receive a GW id and never extend the
wagon timeline. Assertion 13/14 fails loudly if one ever does.

### Measured result on the real dataset

| Quantity | Value |
|---|---|
| RIGHT_UP final gaps | **52** |
| Global gaps | **52** — invariant holds |
| Total wagons | **53** |
| Legacy result from the same input | 64 (52 + 11 synthetic support gaps) |
| Support gaps across all three cameras | 111 → **created 0 global gaps** |
| `corrections_applied` | `[]` — no insertion mechanism exists |
| Wagon count across offset margins 0.00 → 0.30 | **53 at every setting** (count is offset-independent) |

### Estimated camera offsets, corroborated by an independent method

| Camera | DP + interval-pattern | Independent hit-count sweep | Margin | Status at default |
|---|---|---|---|---|
| LEFT_UP | **+16.63 s** | +16.20 s | 10.2% | RESOLVED |
| RIGHT_UP_TOP | **−3.32 s** | −3.15 s | 11.5% | RESOLVED |
| LEFT_UP_TOP | +28.50 s | +28.80 s | 2.2% | **UNRESOLVED** (refused) |

Visual confirmation: with the −3.32 s offset applied, `RIGHT_UP_TOP`'s evidence
for `GW_1` now shows the **roof of locomotive 42816** (pantograph, insulators)
matching RIGHT_UP's side view of the same engine. Before the offset model
existed it showed empty track. Evidence coverage rose from 210/848 frames to
**605/848** with no change to the count.

---

## 0. THE HARD INVARIANT

```
                RIGHT_UP
                   |
                   v
          COMPLETE FINAL GAPS          <-- authoritative, immutable
                   |
                   v
            GLOBAL GAPS                <-- identical set, 1:1
                   |
                   v
             GLOBAL WAGONS             <-- N gaps -> N+1 wagons
                   |
       +-----------+-----------+
       |           |           |
       v           v           v
    LEFT_UP    RIGHT_UP_TOP  LEFT_UP_TOP
       |           |           |
       +-----------+-----------+
                   |
                   v
        SUPPORT / EVIDENCE ONLY        <-- may never change the structure
```

```
global_gap_count == right_up_final_gap_count            ALWAYS
total_wagons     == right_up_final_gap_count + 1        ALWAYS
```

Fusion is **cross-camera association**, not count combination. The only question
a support camera answers is: *"which RIGHT_UP global gap does this observation of
mine belong to?"*

Measured consequence on the run in `results/`: RIGHT_UP has **52** final gaps, so
the system must report **52 global gaps and 53 wagons** (today it reports 64).
For the worked example in the requirement, 59 gaps → 59 global gaps → 60 wagons.

---

## 1. WHY THIS IS A SMALL CHANGE

Because the global sequence *is* RIGHT_UP's gap list, the count path collapses to
something the codebase already contains:

```python
build_global_wagons(fused_gaps=master.gaps, ...)   # == build_wagons_pure_master
```

`build_global_wagons` is reused **unmodified** — its `b <= prev` collapse rule,
its `N gaps -> N+1` segmentation and its `GW_{i}` numbering all stay. The
`total_wagons = len(wagons)` line stays.

So no new counting mathematics is needed. All the new work is in the two things
the current system genuinely lacks:

1. **A camera offset model** — so support observations can be compared on a
   common clock at all.
2. **Order-preserving alignment** — so each support observation is attributed to
   the correct RIGHT_UP gap, with MISSING and EXTRA outcomes that create nothing.

Plus assertions, provenance, and diagnostics. Nothing else in the project moves.

---

## 2. ARCHITECTURE

```
 LocalCameraTracks x4                            (UNCHANGED upstream)
        |
        +-- RIGHT_UP -------------------------------+
        |                                           |
        |   2.1 build_global_gap_sequence()          |  IMMUTABLE FROM HERE ON
        |   one GlobalGap per RIGHT_UP GapEvent      |
        |   global_gap_id = 1..N, master order       |
        |                                           v
        |                                   GlobalGapSequence  (frozen)
        |                                           |
        +-- LEFT_UP / RIGHT_UP_TOP / LEFT_UP_TOP    |
                    |                               |
                    v  2.2 estimate_camera_offset()  |
              CameraOffset(delta, status)            |
                    |                               |
                    v  2.3 align_to_master()  monotonic DP
              SupportAlignment: MATCH / MISSING / EXTRA
                    |                               |
                    +---------------> 2.4 attach_support_evidence()
                                                    |
                                                    v
                                 GlobalGap.support_observations   (children)
                                 GlobalGap.missing_cameras
                                 SupportAlignment.extra_observations (diagnostic)
                                                    |
                                                    v  2.5 assert_invariants()
                                                    |
                                                    v  2.6 build_global_wagons()  REUSED
                                              GW_1 .. GW_(N+1)
                                                    |
                                    +---------------+---------------+
                                    v                               v
                          combined_report.pdf              processed_videos/
                          (reporting only)                 (visualization only)
```

**Direction of information flow is one-way.** After step 2.1 the global sequence
is frozen; steps 2.2–2.4 can only attach children to existing gaps.

### New module

All of this lands in **one new file, `global_fusion.py`**. `global_alignment.py`
stays on disk so the legacy path remains available for A/B comparison behind
`--fusion legacy`, but the **default becomes `--fusion master-fixed`**. Nothing
in `global_alignment.py` is deleted; `build_global_wagons` is imported and reused
from it.

---

## 3. DATA STRUCTURES

### 3.1 `GapObservation` — one local detection, fully traceable

```python
@dataclass(frozen=True)
class GapObservation:
    camera_id: str
    local_track_id: int          # GapEvent.track_id (temporal rank in that camera)
    local_frame: float           # GapEvent.center_frame
    local_time: float            # GapEvent.center_time  (local clock)
    confidence: float
    start_frame: int             # provenance back to source video frames
    end_frame: int
    span_frames: int             # end - start   -> timing tolerance (see 5.2)
    fps: float
    hit_count: int
    temporal_consistency_score: float
    center_x: Optional[float]    # last/mean of center_x_trajectory, if present
    global_time: Optional[float] = None   # local_time + delta_c, set in 2.3
```

Produced by `to_gap_observations(local_tracks) -> List[GapObservation]`, a pure
adapter over the existing `GapEvent`s. Nothing is discarded.

### 3.2 `GlobalGap` — a master gap plus its evidence children

```python
@dataclass
class GlobalGap:
    global_gap_id: int                 # 1..N, from RIGHT_UP order
    master_camera: str                 # always "RIGHT_UP"
    master_observation: GapObservation # THE source. Never None.
    master_frame: int                  # authoritative coordinate
    master_time: float                 # authoritative coordinate
    support_observations: Dict[str, GapObservation]  # camera_id -> matched obs
    missing_cameras: List[str]         # in range, should have seen it, did not
    unavailable_cameras: Dict[str, str] # camera_id -> reason (out of range / offset unresolved)
    support_count: int                 # len(support_observations)
    time_residuals: Dict[str, float]   # (support.global_time - master_time)
    weighted_time: Optional[float]     # confidence-weighted; DIAGNOSTIC ONLY
    alignment_quality: Optional[float]
```

`master_observation` is non-optional **by construction** — a `GlobalGap` cannot
be built without a RIGHT_UP source. That is the invariant expressed in the type.

### 3.3 `CameraOffset`, `SupportAlignment`

```python
@dataclass
class CameraOffset:
    camera_id: str
    delta: float                 # t_global = t_local + delta
    status: str                  # "REFERENCE" | "RESOLVED" | "UNRESOLVED"
    cost: float
    margin_ratio: float          # decisiveness vs the runner-up (see 4.3)
    runner_up_delta: Optional[float]
    n_match: int; n_missing: int; n_extra: int
    reason: str                  # populated when UNRESOLVED

@dataclass
class SupportAlignment:
    camera_id: str
    offset: CameraOffset
    matches: Dict[int, GapObservation]        # global_gap_id -> observation
    missing_global_gap_ids: List[int]
    extra_observations: List[GapObservation]  # EXTRA: diagnostic only, creates nothing
    total_cost: float
```

`delta_RIGHT_UP = 0.0`, `status = "REFERENCE"`, by definition.

---

## 4. CAMERA TEMPORAL OFFSET

### 4.1 The model

```
t_global = t_local + delta_c          delta_RIGHT_UP = 0
```

A single constant per camera. Estimated **only** to associate evidence; it can
never alter the global sequence, because the sequence was frozen in step 2.1.

### 4.2 Objective — the whole ordered sequence, never one detection

```
delta_c* = argmin_delta  AlignmentCost( master_times, [t + delta for t in support_times] )
```

where `AlignmentCost` is exactly the DP total cost of §5 — so the objective
inherently accounts for temporal mismatch, match count, MISSING, EXTRA and train
order. There is no separate hand-written cost function to keep in sync.

Procedure: coarse sweep over `[-OFFSET_SEARCH_S, +OFFSET_SEARCH_S]` at
`OFFSET_COARSE_STEP_S`, collect local minima, disambiguate (§4.3), refine the
winner at `OFFSET_FINE_STEP_S`.

### 4.3 CRITICAL: offset aliasing, and why an offset may be refused

A prototype sweep against the real data (disposable script, scratchpad only) gave:

| Camera | best delta | cost | matched | cost at delta=0 | runner-up | margin |
|---|---|---|---|---|---|---|
| LEFT_UP | +16.50 s | 43.22 | 24 | 53.29 | **+9.00 s @ 43.3** | **0.2% — not decisive** |
| RIGHT_UP_TOP | +7.75 s | 41.49 | 38 | 57.96 | −3.25 s @ 45.0 | 8% — usable |
| LEFT_UP_TOP | +28.50 s | 46.07 | 21 | 53.76 | **+32.50 s @ 46.8** | **1.5% — not decisive** |

Two facts matter:

* The optimum is **not unique**. `16.50 − 9.00 = 7.5 s ≈ 2×` the local wagon
  spacing; `32.5 − 28.5 = 4.0 s ≈ 1×` the spacing. Because wagon spacing is
  quasi-periodic (~3.9 s), **shifting by a whole number of wagons yields an
  almost equally good alignment.** This is cycle-slip aliasing.
* Two methods disagree on RIGHT_UP_TOP (a crude ±0.5 s hit-count sweep said
  −3.15 s; the DP says +7.75 s, with −3.25 s as its runner-up).

An offset wrong by exactly *k* wagon spacings is the **most dangerous** failure
available: the alignment looks internally excellent (tight matches, few extras)
while attributing every observation to the wagon *k* positions away. It would
corrupt evidence attribution while appearing more convincing than today's
behaviour.

**Three mandatory mitigations:**

* **M1 — disambiguate on interval *pattern*, not absolute time.** The interval
  sequence is distinctive (a smooth speed ramp from ~3.7 s to ~6.6 s spacing plus
  7 unusually long intervals). Comparing interval *shape* breaks the periodic tie
  in a way raw timestamps cannot.
* **M2 — require a decisive margin.** Accept only if
  `(cost_runner_up − cost_best) / cost_best >= OFFSET_MIN_MARGIN_RATIO`, where
  the runner-up is the best minimum more than one local spacing away. Otherwise
  → `UNRESOLVED`.
* **M3 — `UNRESOLVED` degrades safely.** That camera contributes no evidence; all
  its gaps are recorded as `unavailable_cameras[cam] = "offset unresolved"`. The
  run completes normally. **The count is unaffected — it never depended on the
  offset.** This is the structural benefit of the fixed-master invariant: a
  synchronization failure degrades evidence quality, never the number.

No offset value from this document may be hard-coded.

---

## 5. ORDER-PRESERVING SEQUENCE ALIGNMENT

### 5.1 The recurrence

Needleman–Wunsch over the frozen master sequence `R1..RN` and one support
camera's offset-corrected sequence `S1..SM`:

```
D[i][j] = min(
    D[i-1][j-1] + match_cost(R_i, S_j),   # MATCH
    D[i-1][j]   + MISSING_PENALTY,        # R_i unobserved by this camera
    D[i][j-1]   + EXTRA_PENALTY,          # S_j corresponds to no master gap
)
```

**Train order is guaranteed structurally, not checked afterwards.** Indices only
ever advance, so `R_i ↔ S_j` and `R_k ↔ S_l` with `i < k` forces `j < l`.
Crossing matches are *unrepresentable*. The requirement's invalid example
(`R1→S3, R2→S1`) cannot be produced.

Duplicate support detections around one master gap are resolved automatically:
the DP advances `i` once, so one becomes `MATCH` and the other `EXTRA`. One
physical gap stays one global gap.

### 5.2 Match cost and timing tolerance

`GapEvent` carries no uncertainty field, so none is invented. Two options, both
documented and configurable, defaulting to (a):

**(a) Fixed configurable tolerance** — simplest, no derived quantities:

```python
d = abs(R_i.global_time - S_j.global_time)
match_cost = d / MATCH_TOLERANCE_S   if d <= MATCH_GATE_S else INFINITY
```

**(b) Span-derived tolerance** — uses information that genuinely exists. A gap is
*visible* for a measurable number of frames, so its centre time has an inherent
uncertainty:

| Camera | median span (frames) | implied half-span (s) |
|---|---|---|
| RIGHT_UP | 13.0 | 0.433 |
| LEFT_UP | 9.0 | 0.300 |
| RIGHT_UP_TOP | 11.0 | 0.367 |
| LEFT_UP_TOP | 16.0 | 0.533 |

```python
sigma_i = clamp(obs.span_frames / 2 / obs.fps, SIGMA_FLOOR_S, SIGMA_CAP_S)
d_norm  = abs(t_i - t_j) / sqrt(sigma_i**2 + sigma_j**2)
```

The cap is required: LEFT_UP has a 78-frame (5.2 s) span outlier from a merged
track. This is measured, not invented — but it is *derived*, so (a) is the
default and (b) is opt-in via `MATCH_COST_MODE`.

Optional small confidence term: `match_cost *= (1 + CONF_WEIGHT * (1 - mean_conf))`.

Because `n_missing − n_extra = N − M` is invariant, each accepted match trades
`MISSING_PENALTY + EXTRA_PENALTY` for one `match_cost`. So the gate and the two
penalties **jointly** define the effective tolerance and must be calibrated and
documented together — they are not independent knobs.

### 5.3 Worked examples from the requirement

**Missing (§7):**
```
master : R1 R2 R3 R4 R5
support: S1 S2    S4 S5
result : R1->S1  R2->S2  R3->MISSING  R4->S4  R5->S5
global : G1 G2 G3 G4 G5                          (5 gaps, unchanged)
```

**Extra (§10):**
```
master : R1 R2    R3 R4 R5
support: L1 L2 X  L3 L4 L5
result : R1->L1  R2->L2  X=EXTRA  R3->L3  R4->L4  R5->L5
global : G1 G2 G3 G4 G5                          (5 gaps, NOT 6)
```

**Duplicate (§4):**
```
master : R25
support: S40 S41
result : R25->S40 (closer)   S41=EXTRA
global : one gap                                 (never GW_26)
```

---

## 6. SUPPORT EVIDENCE ATTACHMENT

For every `GlobalGap`, from each support camera's `SupportAlignment`:

* **MATCH** → `support_observations[cam] = obs`; record
  `time_residuals[cam] = obs.global_time - master_time`.
* **MISSING**, camera footage covers `master_time` → `missing_cameras.append(cam)`.
* **MISSING**, footage does *not* cover it → `unavailable_cameras[cam] = "out of range"`.
* Camera offset `UNRESOLVED` → `unavailable_cameras[cam] = "offset unresolved"`.
* **EXTRA** → appended to `SupportAlignment.extra_observations`. **Creates nothing.**

`missing` and `unavailable` are deliberately distinct: the first is a detection
failure, the second is an absence of footage. Conflating them is what allows
fabricated evidence.

### 6.1 Confidence-weighted timestamp — diagnostic only

```python
weighted_time = sum(w_i * t_i) / sum(w_i),  w_i = confidence_i   (or conf/sigma^2 in mode (b))
```

It **never** moves the boundary. `master_frame` / `master_time` remain
authoritative, exactly as required. `weighted_time` and `time_residuals` exist to
measure consistency — a large systematic residual on one camera is precisely the
signal that its `delta` is wrong or aliased (§4.3).

### 6.2 Out-of-range handling — no fabrication

`map_global_wagon_to_local_frames` clamps out-of-range projections to the last
frame, which fabricates evidence. It is **left untouched** (the existing renderer
depends on it) and a new explicit-validity function is added alongside:

```python
def project_global_time_to_local(t_global, offset, fps, total_frames) -> Optional[int]:
    """Return the local frame index, or None if the instant is outside this
    camera's real footage. Never clamps."""
```

Fusion and evidence attachment use only this version.

Measured durations that make this necessary: RIGHT_UP 276.0 s, LEFT_UP 246.0 s,
RIGHT_UP_TOP 274.0 s, LEFT_UP_TOP 208.0 s.

---

## 7. GLOBAL IDS, WAGONS, AND THE COUNT

### 7.1 Global gap IDs — from RIGHT_UP only

```python
for i, obs in enumerate(sorted(master_observations, key=lambda o: o.local_frame), start=1):
    GlobalGap(global_gap_id=i, master_observation=obs, ...)
```

Nothing else can mint a `global_gap_id`. Support cameras are never enumerated.

### 7.2 Wagon segments and GW IDs — existing code, reused unmodified

```python
fused_gaps = [g.master_observation.source_gap_event for g in global_gaps]  # == master.gaps
wagons     = build_global_wagons(fused_gaps, master_total_frames=..., master_fps=...,
                                 initial_classifications=..., support_camera_ids=...)
total_wagons = len(wagons)
```

Preserved exactly: the `b <= prev` boundary-collapse rule, `N gaps -> N+1`
segments, `GW_{i}` numbering by master frame order, and classification
inheritance from the pre-fusion segments.

`GlobalWagon.supporting_cameras` is **corrected** to list the cameras that
actually have a matched observation on a bounding gap, replacing today's static
all-four list. This is a semantic fix to an existing field, not a count change.

### 7.3 The count

```
total_wagons = len(build_global_wagons(master.gaps, ...))
             = master_gap_count + 1        (minus any boundaries that collapse to the same frame)
```

52 → 53 on the current data. 59 → 60 in the requirement's example. Support
cameras appear nowhere in this expression.

### 7.4 Classification

Unchanged, and still incapable of changing the count. It labels segments after
they exist. If it fails, every wagon becomes `UNKNOWN` and the count is identical.

---

## 8. ASSERTIONS (requirement §18)

A single `assert_invariants(state)` called at the end of assembly, raising
`FusionInvariantError` — loud, not silent:

```python
1  len(global_gaps) == len(master.gaps)                       # the hard invariant
2  every g.master_observation is not None
3  every g.master_observation.camera_id == "RIGHT_UP"
4  {g.global_gap_id} == {1..N}, strictly increasing            # unique, ordered
5  master_frame strictly non-decreasing across the sequence    # train order
6  no GlobalGap constructed from a support observation
7  total_wagons == len(wagons) == len(segments)
8  total_wagons == len(global_gaps) + 1 - collapsed_boundaries
9  every support observation is MATCH xor MISSING xor EXTRA (exactly one)
10 sum(matches) + len(extras) == that camera's observation count
11 GW ids are GW_1..GW_total_wagons in order
12 no support camera_id appears as the source of any global gap
```

Assertion 8 tolerates the `b <= prev` collapse (which is existing, intended
behaviour) by comparing against the count of collapsed boundaries, and logs when
it is non-zero rather than hiding it.

Behaviour on violation: raise in tests and by default; `--fusion-strict false`
downgrades to a loud warning plus a `state.add_note`, for field diagnosis only.

---

## 9. PHYSICAL INTERVAL DIAGNOSTICS — REPORT ONLY

Requirement §16: report suspicious intervals, never silently change the count.

The train's speed drifts (measured: ~3.7 s spacing early, ~6.6 s late), so a
single global "minimum wagon duration" would be wrong. A **local** baseline is
used:

```python
interval_i   = t(G_{i+1}) - t(G_i)
local_base_i = median(interval_j for j in window(i, LOCAL_BASE_WINDOW))
ratio_i      = interval_i / local_base_i
```

* `ratio <= SHORT_INTERVAL_RATIO` → flag `SUSPICIOUSLY_SHORT`
* `ratio >= LONG_INTERVAL_RATIO` → flag `POSSIBLE_MISSING_GAP` (+ implied count)

Measured on the current data — these are what the flags would surface:

| Finding | Detail |
|---|---|
| 1 × `SUSPICIOUSLY_SHORT` | `G51→G52` = 0.833 s vs local base 5.967 s (**ratio 0.14**); next-smallest interval in the whole train is 3.667 s. Produces the 0.27 s and 0.80 s `UNKNOWN` tail wagons. |
| 7 × `POSSIBLE_MISSING_GAP` | ratios 1.67–2.05 at G8→G9, G14→G15, G15→G16, G20→G21, G28→G29, G33→G34, G35→G36 |

**These flags change nothing.** RIGHT_UP's 52 gaps remain 52 gaps and the count
remains 53. The flags are written into `GlobalTrainState`, printed in the console
summary, and shown in the PDF, so the master's detection quality is visible and
can be improved at the detection layer — which is where the requirement places
such work. Whether any flagged interval should ever be acted upon is a separate
decision requiring explicit approval.

---

## 10. REPORTING AND PROCESSED VIDEOS

Both are strictly downstream and both must show the **same** GW IDs.

### 10.1 `combined_report.pdf` (requirement §19)

Preserved: summary page, wagon roster, one page per global wagon, 4 cameras × 4
percentages, 20/40/60/80% sampling, temporary-frame extraction and cleanup after
a verified PDF.

Changes: the report reads the **real** per-wagon support/`unavailable` data from
`GlobalGap` instead of re-deriving availability itself, and can additionally
apply each camera's `delta` so the sampled frames land on the correct footage.
Page count is `1 + roster + total_wagons` — driven by the RIGHT_UP-derived
roster, so extra support detections cannot add pages.

`evidence_report.py` still cannot influence the count: it runs after the state is
final and serialized, only reads it, and mutates nothing but the free-text
`notes` list.

### 10.2 Processed videos (requirement §20)

`--render-videos` continues to work and output to `results/processed_videos/`.
The **existing** `render_processed_video` is reused, with two new optional,
backward-compatible parameters:

```python
def render_processed_video(*, local_tracks, state, output_path,
                           draw_raw_detections=True, verbose=True,
                           time_offset: float = 0.0,          # NEW, default = today's behaviour
                           extra_observation_frames=None):    # NEW, optional
```

* `time_offset` — the camera's estimated `delta`, so global wagon boundaries are
  projected onto the correct local frames instead of assuming a shared `t=0`.
  Default `0.0` reproduces today's output exactly.
* Boundaries that project outside the camera's footage are **not drawn** (no
  clamping to the last frame).
* A support camera's EXTRA observations may be drawn in a visually distinct style
  labelled `EXTRA (local only)` — explicitly *not* a boundary and *not* a GW ID.
* RIGHT_UP renders the authoritative structure exactly as now.

No GUI, no `cv2.imshow`; headless and EC2-safe as today.

---

## 11. CONFIGURATION — CENTRALIZED, NOTHING HIDDEN

One documented `FusionConfig` dataclass in `global_fusion.py`. No magic numbers
inside functions.

| Parameter | Purpose | Suggested start | Basis |
|---|---|---|---|
| `OFFSET_SEARCH_S` | offset sweep half-range | 35.0 | max observed ≈ 29 s (§4.3) |
| `OFFSET_COARSE_STEP_S` | coarse step | 0.25 | ≈ 4 frames @ 15 fps |
| `OFFSET_FINE_STEP_S` | refinement step | 1/fps | one frame |
| `offset_min_margin_ratio` | decisive-margin test (M2) | **0.10** (calibrated, see below) | §4.3 |
| `MATCH_COST_MODE` | `"fixed"` or `"sigma"` | `"fixed"` | (b) is derived, so opt-in |
| `MATCH_TOLERANCE_S` | normalizer for fixed mode | 0.50 | median half-span 0.30–0.53 s (§5.2) |
| `MATCH_GATE_S` | never match beyond this | 1.50 | ≈ 3 × tolerance |
| `SIGMA_FLOOR_S` / `SIGMA_CAP_S` | sigma-mode bounds | 0.10 / 1.00 | ≈1.5 frames; LEFT_UP 78-frame outlier |
| `MATCH_GATE_SIGMAS` | sigma-mode gate | 3.0 | ~3σ |
| `MISSING_PENALTY` / `EXTRA_PENALTY` | DP penalties | 1.0 / 1.0 | calibrate jointly with the gate (§5.2) |
| `CONF_WEIGHT` | confidence term weight | 0.25 | small by design |
| `LOCAL_BASE_WINDOW` | intervals each side for the local median | 6 | speed drift (§9) |
| `SHORT_INTERVAL_RATIO` | flag suspiciously short | 0.50 | measured 0.14 vs next 0.62 |
| `LONG_INTERVAL_RATIO` | flag possible missing gap | 1.60 | all 7 measured cases ≥ 1.67 |
| `STRICT_INVARIANTS` | raise vs warn | `True` | §8 |

Retired: `insert_min_support`, `insert_max_spread_sec`, `insert_min_confidence`,
`insert_min_distance_to_master_sec` — there is no insertion. The CLI flags
`--fuse-min-support`, `--fuse-max-spread`, `--fuse-min-conf` are kept as accepted
no-ops that print a deprecation note (matching the precedent already set for
`--no-videos` / `--every-nth-frame`), so existing invocations do not break.

### Two calibration changes made during implementation

**1. The margin formula was corrected.** The design first normalized by the best
score, `(runner − best) / best`. That is unstable: a clean synthetic alignment
scores ≈ 0, making any rival look infinitely worse and masking genuine
ambiguity. It now normalizes by the larger of the two, floored at 1.0:

```python
margin_ratio = (cost_runner_up - cost_best) / max(|cost_best|, |cost_runner_up|, 1.0)
```

A clean alignment yields ≈ 1.0; the real-data pair scoring 43.22 vs 43.27
correctly yields ≈ 0.001.

**2. `offset_min_margin_ratio` was set to 0.10, not 0.15.** At 0.15 all three
support cameras were refused, losing every support evidence frame. But the
estimator and a fully independent method agree within 0.5 s on all three cameras
while their margins are only 10.2%, 11.5% and 2.2%. Rejecting offsets that two
independent methods corroborate buys nothing, because the count is provably
unaffected (verified: 53 wagons at every margin from 0.00 to 0.30). 0.10 accepts
the two corroborated cameras and still refuses the weakest. `--offset-min-margin`
raises it; the count will not move either way.

---

## 12. WHAT CANNOT HAPPEN (requirement §3, §10, §13)

| Forbidden | Why it is structurally impossible |
|---|---|
| support-only global gap | `GlobalGap.master_observation` is non-optional and must be `camera_id == "RIGHT_UP"`; assertions 2, 3, 6, 12 |
| `R1 R2 R3 X R4` | the sequence is built solely by enumerating master gaps (7.1); nothing appends |
| quorum → new global gap | `decide_inserted_gaps` is not on the counting path at all |
| unmatched support gap → synthetic gap | unmatched becomes `EXTRA`, a diagnostic list entry |
| support count > master count → count rises | `total_wagons` is a function of `master.gaps` only (7.3) |
| duplicate support detections → 2 wagons | DP advances `i` once → one MATCH, one EXTRA (5.1) |
| crossing/order-violating matches | unrepresentable in the DP recurrence (5.1) |
| fabricated evidence at a camera's last frame | `project_global_time_to_local` returns `None`; never clamps (6.2) |
| a camera ending early → new gap | out-of-range is `unavailable`, which creates nothing |
| classification changing the count | it labels existing segments only (7.4) |
| the PDF or videos changing the count | both run after the state is final and only read it (10) |

---

## 13. TEST MATRIX (requirement §23)

New `tests/` directory, pytest, all synthetic except TEST 7. No pipeline run is
required for tests 1–6 and 8.

| Test | Scenario | Assertion |
|---|---|---|
| 1 | 4 cameras, same 10 gaps, different constant offsets | `global_gaps == 10`; all offsets `RESOLVED` and ≈ the injected values |
| 2 | master 10, LEFT_UP 8 | `global_gaps == 10`; 2 `MISSING`; count unchanged |
| 3 | master 10, LEFT_UP 13 | `global_gaps == 10`; **3 `EXTRA`**; count unchanged |
| 4 | one master gap, two nearby support detections | 1 global gap; 1 MATCH + 1 EXTRA |
| 5 | one support camera ends early | out-of-range → `unavailable`; no fabricated evidence; count unchanged |
| 6 | support order such that nearest-neighbour would cross | order preserved; the offender is `EXTRA` |
| 7 | **real data** from `results/per_camera_tracking.json` | `global_gaps == RIGHT_UP gap_count` (52 here); wagons `== 53`; **never 64/79/80** |
| 8 | report page count | `pages == 1 + roster + total_wagons`; roster == RIGHT_UP-derived list |
| 9 | `--render-videos` on a bounded clip | `results/processed_videos/` populated; GW IDs identical to the PDF's |
| 10 *(added)* | offset aliased by exactly *k* wagon spacings | either the true offset or `UNRESOLVED` — **never a confidently wrong k-shift** |
| 11 *(added)* | all three support offsets `UNRESOLVED` | degrades to master-only evidence; run completes; **count still `master+1`** |
| 12 *(added)* | property test: random master/support with injected miss+extra+offset | `global_gaps == len(master)` for every generated case |

Tests 10–12 exist because of §4.3: an aliased offset is the most dangerous
failure available, and test 12 makes the invariant a property rather than a set
of examples.

### As implemented — `tests/test_global_fusion.py`, 31 tests, all passing

Written against stdlib `unittest`, so it needs **no new dependency** and runs
under either `python -m unittest discover -s tests` or `python -m pytest tests`.

| Class | Covers | Notable assertions |
|---|---|---|
| `Test01PerfectAlignment` | test 1 | 4 cameras, injected offsets +7 / −3 / +12.5 s all recovered within 0.30 s; 10 gaps → 11 wagons |
| `Test02SupportMissing` | test 2 | 8-of-10 support → still 10 global gaps; the exact skipped ids (3, 7) are reported `MISSING` |
| `Test03SupportExtra` | test 3 | 13 support gaps → 10 global; 3 `EXTRA`. Plus a 20-vs-10 case on all three cameras |
| `Test04DuplicateSupport` | test 4 | two detections 0.4 s apart → 1 MATCH + 1 EXTRA; no global gap matched twice |
| `Test05DifferentDurations` | test 5 | late gaps are `unavailable` not `missing`; `project_global_time_to_local` returns `None` and never clamps |
| `Test06OrderPreservation` | test 6 | crossing input yields monotonic pairs; 40 randomized trials keep order |
| `Test07RealData` | test 7 | real timelines: 52 → 52 → 53, asserts `!= 64`; plus the 59 → 60 illustrative case |
| `Test08OffsetAliasing` | test 10 | uniform spacing → true offset or `UNRESOLVED`; drifting spacing → resolved; pattern penalty punishes a k-shifted pairing; pure-noise camera cannot change the count |
| `Test09UnresolvedDegradesSafely` | test 11 | all three unresolved → still 53-style `master+1`; no evidence claimed |
| `Test10InvariantProperty` | test 12 | 60 randomized trials with misses, extras, offsets and truncation; plus "vary only support gap count (0→60 extras), total must not move" |
| `Test11Assertions` | §8 | a lost gap and a forged support-sourced gap both raise `FusionInvariantError`; non-strict mode warns |
| `Test12DiagnosticsAndMetadata` | §9, §7.2 | flagged intervals do not change the sequence; `supporting_cameras` excludes a camera with no observation; `weighted_time` never moves `master_time` |

Tests 8 and 9 from the matrix (report page count, `--render-videos`) are covered
by the bounded smoke test rather than unit tests, because both need real video
files: 56 pages = 1 summary + 2 roster + 53 wagons, and both outputs use
`GW_1..GW_53`.

---

## 14. IMPLEMENTATION PLAN

| Phase | Deliverable | Gate |
|---|---|---|
| 3a | `global_fusion.py`: `GapObservation`, `to_gap_observations`, `FusionConfig` | imports clean |
| 3b | `align_to_master` (monotonic DP) | tests 2, 3, 4, 6, 12 |
| 3c | `estimate_camera_offset` (+ M1–M3) | tests 1, 10, 11 |
| 3d | `build_global_gap_sequence`, `attach_support_evidence`, `assert_invariants` | test 7 on real data |
| 3e | `GlobalTrainState` extension + `assemble_global_train_state_v2`; wire `run_global_count.py` | count == master+1 |
| 4 | `tests/` complete | all pass |
| 5 | compile + import checks | clean |
| 6 | bounded smoke test (`frame_limit`, real videos/models) | invariant holds end to end |
| 7 | `--render-videos` bounded check | test 9 |
| 8 | full run **only with your approval** | — |

Estimated footprint: **1 new module (~600 lines), 1 new test package, and
localized edits to 4 existing files** (`run_global_count.py` fusion block,
`global_train_state.py` new fields, `video_segmenter.py` two optional
parameters, `evidence_report.py` reads real support data).
`global_alignment.py` keeps `build_global_wagons` and is otherwise bypassed.

---

## 15. DECISIONS TAKEN (all confirmed before implementation)

| # | Decision | Outcome |
|---|---|---|
| 1 | Fixed-master invariant approved; the 64 → 53 drop is **not** a regression | Implemented. 52 gaps → 52 global gaps → 53 wagons. |
| 2 | RIGHT_UP stays "complete and final"; **no recovery mechanism** | No insertion path exists. The 7 possible-missing and 1 suspiciously-short intervals are flagged as diagnostics only; the master sequence is untouched. |
| 3 | Use the real 52-gap dataset; treat 59 as illustrative | Test 7 asserts the invariant on the real 52-gap timelines, plus a synthetic 59 → 60 case. |
| 4 | Estimate offsets, but never let uncertainty touch the count | Verified: 53 wagons at every margin from 0.00 to 0.30, i.e. whether 0, 2 or 3 cameras resolve. |
| 5 | `--fusion master-fixed` is the default; keep `legacy` for A/B | Done. `legacy` prints a warning that it breaks the invariant. |

### A finding that remains open, by design

RIGHT_UP's own interval statistics still suggest its sequence is not complete:
7 intervals sit at 1.67–2.05× the local median spacing (the signature of one
missed boundary each) and 1 sits at 0.14× (the signature of a duplicate). Per
decision 2 these are **reported, never corrected**, so the reported 53 is
probably below the true wagon count and one of the 52 boundaries is probably
false. That is a detection-quality problem and belongs at the detection layer;
the fusion layer now obeys the master sequence exactly, which is what was asked
for. The flags exist so the issue stays visible instead of being silently
absorbed into the count as it was before.

---

## 16. WHAT THIS DESIGN DOES NOT TOUCH

YOLO weights and classes; the `"gap" in name` filter; detection confidence 0.4;
height ratios 0.35 / 0.05; `_KF1D` parameters; `match_distance_px` 80.0;
`min_hits` 3; `max_miss` 30; `GapTracker` behaviour; `MasterClassifier` and
`_label_to_class`; `segments_from_gaps`; `build_global_wagons`; the
`N gaps -> N+1 wagons` convention; `GW_{i}` numbering; camera filenames; input
handling; the default `python run_global_count.py`; `--render-videos`;
the 20/40/60/80% evidence logic; temporary-frame cleanup; and the promise that no
thousands of permanent JPEGs are written.

No GUI is introduced. Nothing is committed or pushed.

---

*Design only. No source file, model, input video, requirement or output was
modified in producing this document. The prototype behind §4.3 and §9 lives in a
scratchpad directory outside the project and is imported by no pipeline code.*
