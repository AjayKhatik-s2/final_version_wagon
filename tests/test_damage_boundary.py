"""One physical damage at a wagon boundary must belong to exactly one wagon.

The defect these tests pin: a dent near a coupling is visible in frames on both
sides of a reconstructed boundary, the materializer buckets those frames into
two different global wagons, and the report shows the same dent twice.

Ownership here is decided SPATIALLY -- damage centre versus the gap's x position
in the SAME camera frame -- so every test builds a real gap track with
`hit_frames` / `center_x_trajectory` / `bbox_history` and asserts on which
`GW_n` ends up owning the observation. Nothing is decided by time windows and
nothing by confidence.

The four cameras do not share an orientation: `gap_validation` records that
"RIGHT_UP_TOP gaps move in -x, LEFT_UP_TOP gaps move in +x". So each direction
is exercised explicitly -- a rule that is right on one top camera and backwards
on the other would pass a single-camera test.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from core.global_state_loader import GlobalTrainState, GlobalWagon
from orchestrator import damage_boundary as DB

RU, LU = C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP
RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP

FPS = 15.0
GAP_TRACK_ID = 7
MASTER_TRACK_ID = 3


# ---------------------------------------------------------------------------
# Fixtures: real GapEvent-shaped tracks, real wagon roster
# ---------------------------------------------------------------------------

class _Gap:
    """A gap track with the three parallel arrays the resolver reads.

    Deliberately a stand-in rather than the real `GapEvent`: the resolver only
    ever touches `track_id`, `start_frame`, `end_frame`, `hit_frames`,
    `center_x_trajectory` and `bbox_history`, and pinning that surface keeps the
    test honest about what it depends on.
    """

    def __init__(self, track_id, hit_frames, xs, *, half_w=15.0):
        self.track_id = track_id
        self.hit_frames = list(hit_frames)
        self.center_x_trajectory = [float(x) for x in xs]
        self.bbox_history = [[x - half_w, 100.0, x + half_w, 300.0]
                             for x in self.center_x_trajectory]
        self.start_frame = min(self.hit_frames)
        self.end_frame = max(self.hit_frames)


class _Tracks:
    def __init__(self, camera_id, gaps):
        self.camera_id = camera_id
        self.gaps = list(gaps)


def sweeping_gap(direction, *, track_id=GAP_TRACK_ID, first=1000, n=21,
                 x_from=None, x_to=None):
    """A gap crossing the frame in `direction` (+1 towards +x, -1 towards -x).

    Trajectory endpoints mirror what a real sweep looks like, so the derived
    dominant direction comes from the data rather than being declared.
    """
    if x_from is None or x_to is None:
        x_from, x_to = (150.0, 800.0) if direction > 0 else (800.0, 150.0)
    hits = [first + i * 2 for i in range(n)]
    xs = [x_from + (x_to - x_from) * i / (n - 1) for i in range(n)]
    return _Gap(track_id, hits, xs)


def tracks_for(camera_id, direction, *, extra=3):
    """One boundary gap plus `extra` decoys, all sweeping the same way.

    `camera_direction` needs several tracks before it will commit
    (min_tracks_for_direction), mirroring `gap_validation`'s refusal to infer a
    direction from too few survivors.
    """
    gaps = [sweeping_gap(direction)]
    for k in range(extra):
        gaps.append(sweeping_gap(direction, track_id=100 + k,
                                 first=2000 + k * 500))
    return _Tracks(camera_id, gaps)


def roster(n=3):
    """`n` wagons in a row, bounded by SHARED master gap tracks.

    Wagon i's trailing gap and wagon i+1's leading gap are the same physical
    boundary and must therefore carry the SAME track id -- that is how
    `global_alignment` writes them. Getting this wrong makes the boundary
    unfindable from one side and quietly turns an oscillation test into a
    no-op, so the sharing is explicit here.

    Boundary GW_1|GW_2 is MASTER_TRACK_ID: the one under test.
    """
    # boundary[k] separates wagon k+1 from wagon k+2
    boundary = [MASTER_TRACK_ID] + [90 + k for k in range(1, n)]
    wagons = []
    for i in range(1, n + 1):
        lead = ({"source": "video_start"} if i == 1 else
                {"source": "master", "camera_id": RU,
                 "track_id": boundary[i - 2], "center_time": 60.0 + i * 10})
        trail = ({"source": "master", "camera_id": RU,
                  "track_id": boundary[i - 1], "center_time": 70.0 + i * 10}
                 if i < n else {"source": "video_end"})
        wagons.append(GlobalWagon(
            global_id=f"GW_{i}", wagon_index=i,
            start_frame_master=(i - 1) * 100, end_frame_master=i * 100 - 1,
            start_time=(i - 1) * 6.0, end_time=i * 6.0,
            classification=C.CLASS_WAGON, classification_confidence=0.95,
            leading_gap=lead, trailing_gap=trail))
    return GlobalTrainState(total_wagons=n, wagons=tuple(wagons),
                            master_camera=RU, master_fps=FPS)


def global_gaps(cameras=(RUT,), *, master_track_id=MASTER_TRACK_ID,
                support_track_id=GAP_TRACK_ID):
    """One global boundary, observed by `cameras`."""
    return [{
        "global_gap_id": 1,
        "master_camera": RU,
        "master_track_id": master_track_id,
        "master_frame": 1020,
        "master_time": 70.0,
        "support_observations": {
            cam: {"camera_id": cam, "local_track_id": support_track_id,
                  "local_frame": 1020.0, "local_time": 68.0,
                  "confidence": 0.9, "start_frame": 1000, "end_frame": 1040}
            for cam in cameras
        },
        "missing_cameras": [],
    }]


def observation(camera_id, frame_idx, center_x, *, half_w=30.0,
                class_name="floor_damage", conf=0.62, track_idx=1,
                track_id=1):
    return {
        "track_idx": track_idx, "camera_id": camera_id, "track_id": track_id,
        "class_name": class_name, "confidence": conf, "best_confidence": conf,
        "best_frame_idx": frame_idx,
        "bbox": [center_x - half_w, 400.0, center_x + half_w, 500.0],
    }


def resolve_one(camera_id, direction, frame_idx, center_x, *,
                on="GW_2", cameras=(RUT,), cfg=None, n=3):
    st = roster(n)
    return DB.resolve_observation(
        gw_id=on,
        observation=observation(camera_id, frame_idx, center_x),
        wagons=list(st.wagons),
        global_gaps=global_gaps(cameras),
        tracks_by_camera={camera_id: tracks_for(camera_id, direction)},
        directions={camera_id: direction},
        cfg=cfg or DB.DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# The fixture must be able to fail
# ---------------------------------------------------------------------------

class TestFixtureIsHonest(unittest.TestCase):

    def test_direction_is_measured_from_the_trajectory(self):
        self.assertEqual(DB.track_direction(sweeping_gap(+1)), 1)
        self.assertEqual(DB.track_direction(sweeping_gap(-1)), -1)

    def test_camera_direction_needs_enough_tracks(self):
        one = _Tracks(RUT, [sweeping_gap(+1)])
        self.assertEqual(DB.camera_direction(one), 0,
                         "a single track cannot establish a direction")
        self.assertEqual(DB.camera_direction(tracks_for(RUT, +1)), 1)
        self.assertEqual(DB.camera_direction(tracks_for(RUT, -1)), -1)

    def test_gap_x_moves_with_the_frame(self):
        g = sweeping_gap(+1)
        early, _ = DB.gap_x_at_frame(g, g.hit_frames[0])
        late, _ = DB.gap_x_at_frame(g, g.hit_frames[-1])
        self.assertLess(early, late)


# ---------------------------------------------------------------------------
# Same-frame lookup
# ---------------------------------------------------------------------------

class TestSameFrameGapLookup(unittest.TestCase):

    def test_exact_hit_returns_the_recorded_position_unchanged(self):
        g = sweeping_gap(+1)
        i = 5
        f = g.hit_frames[i]
        x, bb = DB.gap_x_at_frame(g, f)
        self.assertAlmostEqual(x, g.center_x_trajectory[i], places=6)
        self.assertEqual(bb, g.bbox_history[i])

    def test_between_hits_interpolates_via_the_renderer(self):
        """Uses wagon_count.video_segmenter._interp_gap_bbox, not a copy."""
        g = sweeping_gap(+1)
        f0, f1 = g.hit_frames[4], g.hit_frames[5]
        mid = (f0 + f1) // 2
        self.assertNotIn(mid, g.hit_frames)
        x, _ = DB.gap_x_at_frame(g, mid)
        lo = min(g.center_x_trajectory[4], g.center_x_trajectory[5])
        hi = max(g.center_x_trajectory[4], g.center_x_trajectory[5])
        self.assertGreater(x, lo)
        self.assertLess(x, hi)

    def test_the_interpolator_is_the_renderers_own_function(self):
        import inspect
        from video_segmenter import _interp_gap_bbox
        src = inspect.getsource(DB._interp_bbox)
        self.assertIn("_interp_gap_bbox", src)
        self.assertTrue(callable(_interp_gap_bbox))

    def test_outside_the_track_span_yields_nothing(self):
        g = sweeping_gap(+1)
        self.assertEqual(DB.gap_x_at_frame(g, g.start_frame - 50), (None, None))
        self.assertEqual(DB.gap_x_at_frame(g, g.end_frame + 50), (None, None))

    def test_center_x_scalar_is_not_used(self):
        """`GapObservation.center_x` is the LAST hit only; using it would be wrong."""
        import inspect
        src = inspect.getsource(DB)
        self.assertNotIn("obs.center_x", src)
        self.assertNotIn('["center_x"]', src)
        self.assertNotIn('.get("center_x")', src)


# ---------------------------------------------------------------------------
# The spatial rule, per camera orientation
# ---------------------------------------------------------------------------

class TestSpatialOwnership(unittest.TestCase):
    """With dominant +1 the preceding wagon sits at LARGER x; with -1, smaller.

    Each case is asserted on the resolved `GW_n`, not on a side label alone.
    """

    def _gap_x_at(self, direction, frame):
        return DB.gap_x_at_frame(sweeping_gap(direction), frame)[0]

    def test_before_gap_goes_to_the_preceding_wagon_plus_x(self):
        f = 1020
        gx = self._gap_x_at(+1, f)
        v = resolve_one(RUT, +1, f, gx + 200.0)
        self.assertEqual(v.side, DB.SIDE_BEFORE)
        self.assertEqual(v.owner, "GW_1")
        self.assertEqual(v.reason, DB.REASON_RESOLVED)
        self.assertFalse(v.ambiguous)

    def test_after_gap_goes_to_the_following_wagon_plus_x(self):
        f = 1020
        gx = self._gap_x_at(+1, f)
        v = resolve_one(RUT, +1, f, gx - 200.0)
        self.assertEqual(v.side, DB.SIDE_AFTER)
        self.assertEqual(v.owner, "GW_2")
        self.assertFalse(v.ambiguous)

    def test_the_sign_flips_with_the_camera_direction(self):
        """Same geometry, opposite camera orientation -> opposite owner.

        This is the case a hardcoded `x < gap_x` rule gets wrong on one of the
        two top cameras.
        """
        f = 1020
        for direction, cam in ((+1, RUT), (-1, LUT)):
            gx = self._gap_x_at(direction, f)
            larger = resolve_one(cam, direction, f, gx + 200.0,
                                 cameras=(cam,))
            smaller = resolve_one(cam, direction, f, gx - 200.0,
                                  cameras=(cam,))
            with self.subTest(camera=cam, direction=direction):
                if direction > 0:
                    self.assertEqual(larger.owner, "GW_1")
                    self.assertEqual(smaller.owner, "GW_2")
                else:
                    self.assertEqual(larger.owner, "GW_2")
                    self.assertEqual(smaller.owner, "GW_1")

    def test_all_four_camera_orientations(self):
        f = 1020
        for cam in (RU, LU, RUT, LUT):
            for direction in (+1, -1):
                gx = self._gap_x_at(direction, f)
                v = resolve_one(cam, direction, f, gx + 250.0,
                                cameras=(cam,))
                with self.subTest(camera=cam, direction=direction):
                    self.assertEqual(v.reason, DB.REASON_RESOLVED)
                    self.assertIn(v.owner, ("GW_1", "GW_2"))
                    self.assertEqual(v.owner,
                                     "GW_1" if direction > 0 else "GW_2")

    def test_ownership_uses_an_interpolated_frame_too(self):
        g = sweeping_gap(+1)
        mid = (g.hit_frames[4] + g.hit_frames[5]) // 2
        gx = DB.gap_x_at_frame(g, mid)[0]
        v = resolve_one(RUT, +1, mid, gx + 200.0)
        self.assertEqual(v.reason, DB.REASON_RESOLVED)
        self.assertEqual(v.owner, "GW_1")
        self.assertEqual(v.frame_idx, mid)


# ---------------------------------------------------------------------------
# Ambiguity and unavailability
# ---------------------------------------------------------------------------

class TestAmbiguousAndUnavailable(unittest.TestCase):

    def test_centred_on_the_gap_is_ambiguous_but_owned_by_one_wagon(self):
        f = 1020
        gx = DB.gap_x_at_frame(sweeping_gap(+1), f)[0]
        v = resolve_one(RUT, +1, f, gx)
        self.assertEqual(v.reason, DB.REASON_WITHIN_TOLERANCE)
        self.assertTrue(v.ambiguous)
        self.assertEqual(v.owner, "GW_2", "keeps the bucketed owner")
        self.assertFalse(v.moved)

    def test_just_inside_the_tolerance_stays_ambiguous(self):
        f = 1020
        gx = DB.gap_x_at_frame(sweeping_gap(+1), f)[0]
        cfg = DB.BoundaryConfig(tolerance_px=60.0)
        v = resolve_one(RUT, +1, f, gx + 50.0, cfg=cfg)
        self.assertEqual(v.reason, DB.REASON_WITHIN_TOLERANCE)
        self.assertTrue(v.ambiguous)

    def test_just_outside_the_tolerance_resolves(self):
        f = 1020
        gx = DB.gap_x_at_frame(sweeping_gap(+1), f)[0]
        cfg = DB.BoundaryConfig(tolerance_px=20.0)
        v = resolve_one(RUT, +1, f, gx + 50.0, cfg=cfg)
        self.assertEqual(v.reason, DB.REASON_RESOLVED)
        self.assertFalse(v.ambiguous)

    def test_no_support_gap_for_this_camera(self):
        """A camera that never observed the boundary cannot use the rule."""
        st = roster()
        v = DB.resolve_observation(
            gw_id="GW_2",
            observation=observation(LUT, 1020, 600.0),
            wagons=list(st.wagons),
            global_gaps=global_gaps((RUT,)),          # LUT absent
            tracks_by_camera={LUT: tracks_for(LUT, +1)},
            directions={LUT: 1})
        self.assertEqual(v.reason, DB.REASON_NO_SUPPORT_GAP)
        self.assertTrue(v.ambiguous)
        self.assertEqual(v.owner, "GW_2")
        self.assertFalse(v.moved)

    def test_no_gap_position_when_the_track_has_no_geometry(self):
        st = roster()
        bare = _Gap(GAP_TRACK_ID, [1000, 1040], [400.0, 500.0])
        bare.bbox_history = []                        # geometry unrecoverable
        tracks = _Tracks(RUT, [bare] + [sweeping_gap(+1, track_id=200 + k)
                                        for k in range(3)])
        v = DB.resolve_observation(
            gw_id="GW_2", observation=observation(RUT, 1020, 600.0),
            wagons=list(st.wagons), global_gaps=global_gaps((RUT,)),
            tracks_by_camera={RUT: tracks}, directions={RUT: 1})
        self.assertEqual(v.reason, DB.REASON_NO_GAP_POSITION)
        self.assertEqual(v.owner, "GW_2")

    def test_indeterminate_camera_direction_is_not_guessed(self):
        f = 1020
        gx = DB.gap_x_at_frame(sweeping_gap(+1), f)[0]
        v = resolve_one(RUT, 0, f, gx + 200.0)
        self.assertEqual(v.reason, DB.REASON_NO_DIRECTION)
        self.assertTrue(v.ambiguous)
        self.assertEqual(v.owner, "GW_2")

    def test_an_edge_wagon_with_no_boundary(self):
        st = roster()
        v = DB.resolve_observation(
            gw_id="GW_1", observation=observation(RUT, 1020, 600.0),
            wagons=[st.wagons[0]],                    # no neighbours
            global_gaps=global_gaps((RUT,)),
            tracks_by_camera={RUT: tracks_for(RUT, +1)},
            directions={RUT: 1})
        self.assertEqual(v.reason, DB.REASON_NO_BOUNDARY)

    def test_an_observation_with_no_bbox(self):
        st = roster()
        obs = observation(RUT, 1020, 600.0)
        obs["bbox"] = None
        v = DB.resolve_observation(
            gw_id="GW_2", observation=obs, wagons=list(st.wagons),
            global_gaps=global_gaps((RUT,)),
            tracks_by_camera={RUT: tracks_for(RUT, +1)},
            directions={RUT: 1})
        self.assertEqual(v.reason, DB.REASON_NO_DAMAGE_BOX)

    def test_never_owned_by_two_wagons(self):
        """Whatever happens, a verdict names at most one owner."""
        f = 1020
        gx = DB.gap_x_at_frame(sweeping_gap(+1), f)[0]
        for dx in (-400.0, -60.0, -10.0, 0.0, 10.0, 60.0, 400.0):
            v = resolve_one(RUT, +1, f, gx + dx)
            with self.subTest(dx=dx):
                self.assertIsInstance(v.owner, str)
                self.assertIn(v.owner, ("GW_1", "GW_2"))


# ---------------------------------------------------------------------------
# Oscillation across a boundary, and multi-camera evidence
# ---------------------------------------------------------------------------

class TestWholeTrainPass(unittest.TestCase):

    def _run(self, damage_by_wagon, cameras=(RUT,), direction=+1, cfg=None):
        st = roster()
        cams = {c: tracks_for(c, direction) for c in cameras}
        return st, DB.resolve_train(
            state=st, engine_global_gaps=global_gaps(cameras),
            tracks_by_camera=cams, damage_by_wagon=damage_by_wagon,
            cfg=cfg or DB.DEFAULT_CONFIG, verbose=False)

    def test_oscillating_frames_do_not_damage_both_wagons(self):
        """The reported defect: one dent, frames either side of the boundary.

        Both halves resolve to the SAME wagon and the duplicate is collapsed, so
        exactly one wagon ends up owning it.
        """
        g = sweeping_gap(+1)
        f1, f2 = 1018, 1024
        gx1 = DB.gap_x_at_frame(g, f1)[0]
        gx2 = DB.gap_x_at_frame(g, f2)[0]
        _st, res = self._run({
            "GW_1": [observation(RUT, f1, gx1 + 220.0, track_idx=1)],
            "GW_2": [observation(RUT, f2, gx2 + 210.0, track_idx=1)],
        })
        owners = {v.owner for v in res.verdicts if v.owner}
        self.assertEqual(owners, {"GW_1"},
                         f"expected one owner, got {owners}")
        self.assertEqual(res.deduplicated, 1)
        self.assertEqual(sum(1 for v in res.verdicts if v.owner), 1)

    def test_two_genuinely_separate_defects_are_not_merged(self):
        g = sweeping_gap(+1)
        f1, f2 = 1004, 1038          # far apart in frames
        gx1 = DB.gap_x_at_frame(g, f1)[0]
        gx2 = DB.gap_x_at_frame(g, f2)[0]
        _st, res = self._run({
            "GW_1": [observation(RUT, f1, gx1 + 220.0, track_idx=1),
                     observation(RUT, f2, gx2 + 230.0, track_idx=2)],
        }, cfg=DB.BoundaryConfig(dedup_frame_window=10))
        self.assertEqual(res.deduplicated, 0)
        self.assertEqual(sum(1 for v in res.verdicts if v.owner), 2)

    def test_one_camera_only_is_still_valid(self):
        g = sweeping_gap(+1)
        f = 1020
        gx = DB.gap_x_at_frame(g, f)[0]
        _st, res = self._run({"GW_2": [observation(RUT, f, gx + 200.0)]})
        self.assertEqual(len(res.verdicts), 1)
        v = res.verdicts[0]
        self.assertEqual(v.reason, DB.REASON_RESOLVED)
        self.assertEqual(v.owner, "GW_1")

    def test_two_cameras_stay_two_observations_of_one_event(self):
        """Both cameras see the same defect; both keep their own provenance."""
        g = sweeping_gap(+1)
        f = 1020
        gx = DB.gap_x_at_frame(g, f)[0]
        _st, res = self._run({
            "GW_2": [observation(RUT, f, gx + 200.0, track_idx=1),
                     observation(LUT, f, gx + 205.0, track_idx=2)],
        }, cameras=(RUT, LUT))
        self.assertEqual(len(res.verdicts), 2)
        for v in res.verdicts:
            with self.subTest(camera=v.camera_id):
                self.assertEqual(v.reason, DB.REASON_RESOLVED)
                self.assertEqual(v.owner, "GW_1")
        cams = {v.camera_id for v in res.verdicts if v.owner}
        self.assertEqual(cams, {RUT, LUT},
                         "per-camera observations must both survive")

    def test_provenance_is_preserved_on_every_verdict(self):
        g = sweeping_gap(+1)
        f = 1020
        gx = DB.gap_x_at_frame(g, f)[0]
        _st, res = self._run({"GW_2": [observation(RUT, f, gx + 200.0)]})
        v = res.verdicts[0]
        self.assertEqual(v.camera_id, RUT)
        self.assertEqual(v.frame_idx, f)
        self.assertEqual(v.class_name, "floor_damage")
        self.assertIsNotNone(v.gap_bbox)
        self.assertIsNotNone(v.damage_center_x)

    def test_diagnostics_line_carries_every_required_field(self):
        g = sweeping_gap(+1)
        f = 1020
        gx = DB.gap_x_at_frame(g, f)[0]
        _st, res = self._run({"GW_2": [observation(RUT, f, gx + 200.0)]})
        line = res.verdicts[0].render()
        self.assertIn("[DAMAGE-BOUNDARY]", line)
        for token in (RUT, "f=1020", "pair=", "gap_x=", "dmg_x=", "dir=",
                      "side=", "owner=", "reason="):
            with self.subTest(token=token):
                self.assertIn(token, line)


# ---------------------------------------------------------------------------
# Applying the outcome to disk
# ---------------------------------------------------------------------------

class TestApplyVerdicts(unittest.TestCase):

    def _tree(self, root, damage_by_wagon):
        states = os.path.join(root, "wagon_states")
        ev = os.path.join(root, "evidence")
        os.makedirs(os.path.join(states, "damage"))
        for gw, recs in damage_by_wagon.items():
            with open(os.path.join(states, "damage", f"{gw}.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"global_id": gw, "feature": "damage",
                           "status": C.STATUS_OK,
                           "top_damage": C.DAMAGE_PRESENT,
                           "top_damage_details": recs,
                           "per_camera": {}, "supporting_cameras": [RUT],
                           "frame_count": 10, "evidence": {}}, f)
            d = os.path.join(ev, gw, "damage")
            os.makedirs(d, exist_ok=True)
            from core.evidence_identity import damage_track_slot
            for r in recs:
                slot = damage_track_slot(r["track_idx"], r["camera_id"])
                with open(os.path.join(d, f"{slot}.jpg"), "wb") as f:
                    f.write(b"\xff\xd8fake")
            with open(os.path.join(d, "metadata.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"global_id": gw, "feature": "damage",
                           "top_damage": C.DAMAGE_PRESENT,
                           "tracks": recs}, f)
        return states, ev

    def test_a_moved_observation_changes_both_wagons_verdicts(self):
        g = sweeping_gap(+1)
        f = 1020
        gx = DB.gap_x_at_frame(g, f)[0]
        st = roster()
        with tempfile.TemporaryDirectory() as root:
            # bucketed on GW_2 but spatially before the gap -> belongs to GW_1
            recs = {"GW_2": [observation(RUT, f, gx + 220.0, track_idx=1)]}
            states, ev = self._tree(root, recs)
            res = DB.resolve_train(
                state=st, engine_global_gaps=global_gaps((RUT,)),
                tracks_by_camera={RUT: tracks_for(RUT, +1)},
                damage_by_wagon=recs, verbose=False)
            DB.apply_verdicts(result=res, states_root=states, evidence_root=ev,
                              wagons=list(st.wagons), verbose=False)

            src = json.load(open(os.path.join(states, "damage", "GW_2.json"),
                                 encoding="utf-8"))
            dst = json.load(open(os.path.join(states, "damage", "GW_1.json"),
                                 encoding="utf-8"))
            self.assertEqual(src["top_damage"], C.DAMAGE_OK,
                             "the wagon that lost its only track is not damaged")
            self.assertEqual(src["top_damage_details"], [])
            self.assertEqual(len(dst["top_damage_details"]), 1)
            moved = dst["top_damage_details"][0]
            self.assertEqual(moved["camera_id"], RUT)
            self.assertEqual(moved["best_frame_idx"], f)
            self.assertEqual(moved["moved_from_global_id"], "GW_2")
            self.assertEqual(moved["boundary_side"], DB.SIDE_BEFORE)
            self.assertFalse(moved["boundary_ambiguous"])

    def test_the_snapshot_follows_the_damage(self):
        g = sweeping_gap(+1)
        f = 1020
        gx = DB.gap_x_at_frame(g, f)[0]
        st = roster()
        with tempfile.TemporaryDirectory() as root:
            recs = {"GW_2": [observation(RUT, f, gx + 220.0, track_idx=1)]}
            states, ev = self._tree(root, recs)
            res = DB.resolve_train(
                state=st, engine_global_gaps=global_gaps((RUT,)),
                tracks_by_camera={RUT: tracks_for(RUT, +1)},
                damage_by_wagon=recs, verbose=False)
            DB.apply_verdicts(result=res, states_root=states, evidence_root=ev,
                              wagons=list(st.wagons), verbose=False)
            dst_dir = os.path.join(ev, "GW_1", "damage")
            jpgs = [p for p in os.listdir(dst_dir) if p.endswith(".jpg")]
            self.assertTrue(jpgs, "the new owner has no snapshot")
            self.assertTrue(any(RUT in p for p in jpgs),
                            f"slot must keep the camera: {jpgs}")

    def test_an_ambiguous_observation_is_annotated_not_moved(self):
        g = sweeping_gap(+1)
        f = 1020
        gx = DB.gap_x_at_frame(g, f)[0]
        st = roster()
        with tempfile.TemporaryDirectory() as root:
            recs = {"GW_2": [observation(RUT, f, gx, track_idx=1)]}
            states, ev = self._tree(root, recs)
            res = DB.resolve_train(
                state=st, engine_global_gaps=global_gaps((RUT,)),
                tracks_by_camera={RUT: tracks_for(RUT, +1)},
                damage_by_wagon=recs, verbose=False)
            DB.apply_verdicts(result=res, states_root=states, evidence_root=ev,
                              wagons=list(st.wagons), verbose=False)
            doc = json.load(open(os.path.join(states, "damage", "GW_2.json"),
                                 encoding="utf-8"))
            self.assertEqual(doc["top_damage"], C.DAMAGE_PRESENT)
            self.assertEqual(len(doc["top_damage_details"]), 1)
            self.assertTrue(doc["top_damage_details"][0]["boundary_ambiguous"])
            self.assertEqual(doc["top_damage_details"][0]["boundary_reason"],
                             DB.REASON_WITHIN_TOLERANCE)


# ---------------------------------------------------------------------------
# Integration surface
# ---------------------------------------------------------------------------

class TestWiring(unittest.TestCase):

    def test_the_resolver_runs_between_stage3_and_stage4(self):
        import inspect
        from orchestrator import global_assembler as GA
        src = inspect.getsource(GA.assemble)
        self.assertLess(src.index('res.timings["stage3_features"]'),
                        src.index("damage_boundary"))
        self.assertLess(src.index("damage_boundary"),
                        src.index("wagon_state_builder.build"))

    def test_it_uses_the_engine_state_gaps_and_full_tracks(self):
        import inspect
        from orchestrator import global_assembler as GA
        src = inspect.getsource(GA.assemble)
        block = src[src.index("Stage 3b"):src.index("Stage 4: fuse")]
        self.assertIn("engine_state", block)
        self.assertIn("tracks_by_camera=tracks", block)

    def test_a_resolver_failure_cannot_fail_the_train(self):
        import inspect
        from orchestrator import global_assembler as GA
        src = inspect.getsource(GA.assemble)
        block = src[src.index("Stage 3b"):src.index("Stage 4: fuse")]
        self.assertIn("except Exception", block)

    def test_batch_mode_is_untouched(self):
        import inspect
        from orchestrator import master_runner as MR
        self.assertNotIn("damage_boundary",
                         inspect.getsource(MR.process_batch))

    def test_the_module_runs_no_detector_and_writes_no_segmentation(self):
        import ast
        src = open(os.path.join(V4_ROOT, "orchestrator",
                                "damage_boundary.py"), encoding="utf-8").read()
        names = set()
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Import):
                for a in n.names:
                    names.update(a.name.split("."))
            elif isinstance(n, ast.ImportFrom):
                names.update((n.module or "").split("."))
                names.update(a.name for a in n.names)
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
        for banned in ("YOLO", "ultralytics", "GapTracker", "segments_from_gaps",
                       "build_global_wagons", "wagon_cache_builder",
                       "assemble_global_train_state_master_fixed"):
            with self.subTest(token=banned):
                self.assertNotIn(banned, names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
