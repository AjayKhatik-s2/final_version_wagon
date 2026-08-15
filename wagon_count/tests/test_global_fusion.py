"""Unit tests for the fixed-master global fusion architecture.

THE PROPERTY UNDER TEST, above all others:

    global_gap_count == len(right_up_gaps)      for ANY support-camera input
    total_wagons     == global_gap_count + 1    (minus collapsed boundaries)

Stdlib `unittest` only -- no new dependency. Runs under either:

    python -m unittest discover -s tests -v
    python -m pytest tests -v
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import global_fusion as gf
from global_train_state import (
    ALL_CAMERAS, CAMERA_LEFT_UP, CAMERA_LEFT_UP_TOP, CAMERA_RIGHT_UP,
    CAMERA_RIGHT_UP_TOP, GapEvent, LocalCameraTracks, SegmentClass,
    _MasterClassification,
)

FPS = 15.0
GAP_SPAN = 12          # frames a gap stays visible (~0.8 s), like the real data


# ---------------------------------------------------------------------------
# synthetic builders
# ---------------------------------------------------------------------------

def make_tracks(camera_id, gap_times, duration_s=300.0, fps=FPS, confidence=0.85):
    """Build a LocalCameraTracks whose gaps sit at the given LOCAL times."""
    gaps = []
    for i, t in enumerate(sorted(gap_times), start=1):
        center = t * fps
        start = int(round(center - GAP_SPAN / 2))
        end = int(round(center + GAP_SPAN / 2))
        gaps.append(GapEvent(
            track_id=i, camera_id=camera_id,
            start_frame=max(0, start), end_frame=max(1, end),
            confidence=confidence, hit_count=GAP_SPAN, fps=fps,
            temporal_consistency_score=1.0, class_label="gap",
        ))
    return LocalCameraTracks(
        camera_id=camera_id, video_path=f"/synthetic/{camera_id}.mp4",
        fps=fps, total_frames=int(round(duration_s * fps)),
        width=848, height=480, gaps=gaps,
    )


def uneven_gap_times(n, start=10.0):
    """Gap times with a DRIFTING spacing, like the real train.

    Uniform spacing would make every whole-period offset a perfect alias, which
    is unrealistic and would make the anti-aliasing tests vacuous.
    """
    times, t, spacing = [], start, 4.0
    for i in range(n):
        times.append(t)
        spacing = 4.0 + 2.0 * (i / max(1, n - 1))      # 4.0 s -> 6.0 s
        t += spacing
    return times


def classifications_for(master):
    """One pre-fusion classification spanning the whole master video."""
    return [_MasterClassification(0, 0, max(0, master.total_frames - 1),
                                  SegmentClass.WAGON, 1.0)]


def assemble(master, supports, cfg=None, verbose=False):
    return gf.assemble_global_train_state_master_fixed(
        master_tracks=master, support_tracks=supports,
        initial_classifications=classifications_for(master),
        config=cfg or gf.FusionConfig(), verbose=verbose,
    )


# ===========================================================================
# TEST 1 -- perfect alignment, four cameras, different constant offsets
# ===========================================================================

class Test01PerfectAlignment(unittest.TestCase):
    def test_ten_gaps_four_cameras_with_offsets(self):
        # Start late enough that every shifted local time stays positive; a
        # camera whose footage starts before the train is a separate case
        # (covered by Test05).
        base = uneven_gap_times(10, start=30.0)
        master = make_tracks(CAMERA_RIGHT_UP, base)
        # Support cameras see the same train, but their clocks start elsewhere.
        # t_local = t_global - delta  =>  a camera whose delta is +7 s has
        # local times 7 s EARLIER than the master's.
        deltas = {CAMERA_LEFT_UP: 7.0, CAMERA_RIGHT_UP_TOP: -3.0,
                  CAMERA_LEFT_UP_TOP: 12.5}
        supports = [make_tracks(c, [t - d for t in base])
                    for c, d in deltas.items()]

        state = assemble(master, supports)

        self.assertEqual(len(state.global_gaps), 10)
        self.assertEqual(state.invariant_checks["right_up_final_gap_count"], 10)
        self.assertEqual(state.invariant_checks["global_gap_count"], 10)
        self.assertTrue(state.invariant_checks["invariant_holds"])
        self.assertEqual(state.total_wagons, 11)

        for cam, expected in deltas.items():
            off = state.camera_offsets[cam]
            self.assertEqual(off["status"], gf.OFFSET_RESOLVED,
                             f"{cam} should resolve: {off['reason']}")
            self.assertAlmostEqual(off["delta"], expected, delta=0.30,
                                   msg=f"{cam} delta {off['delta']} != {expected}")
            self.assertEqual(state.support_alignment_summary[cam]["n_match"], 10)
            self.assertEqual(state.support_alignment_summary[cam]["n_extra"], 0)

    def test_master_is_the_reference_clock(self):
        master = make_tracks(CAMERA_RIGHT_UP, uneven_gap_times(6))
        state = assemble(master, [])
        self.assertEqual(state.camera_offsets[CAMERA_RIGHT_UP]["delta"], 0.0)
        self.assertEqual(state.camera_offsets[CAMERA_RIGHT_UP]["status"],
                         gf.OFFSET_REFERENCE)


# ===========================================================================
# TEST 2 -- support camera MISSES gaps: count must not change
# ===========================================================================

class Test02SupportMissing(unittest.TestCase):
    def test_support_with_eight_of_ten(self):
        base = uneven_gap_times(10)
        master = make_tracks(CAMERA_RIGHT_UP, base)
        # LEFT_UP misses the 3rd and 7th gaps entirely.
        partial = [t for i, t in enumerate(base) if i not in (2, 6)]
        support = make_tracks(CAMERA_LEFT_UP, partial)

        state = assemble(master, [support])

        self.assertEqual(len(state.global_gaps), 10, "missing support must not delete gaps")
        self.assertEqual(state.total_wagons, 11)
        summary = state.support_alignment_summary[CAMERA_LEFT_UP]
        self.assertEqual(summary["n_match"], 8)
        self.assertEqual(summary["n_missing"], 2)
        self.assertEqual(summary["n_extra"], 0)
        # The two skipped gaps are the ones reported missing.
        self.assertEqual(summary["missing_global_gap_ids"], [3, 7])

    def test_master_alone_still_counts(self):
        master = make_tracks(CAMERA_RIGHT_UP, uneven_gap_times(10))
        empty = make_tracks(CAMERA_LEFT_UP, [])
        state = assemble(master, [empty])
        self.assertEqual(len(state.global_gaps), 10)
        self.assertEqual(state.total_wagons, 11)


# ===========================================================================
# TEST 3 -- support camera has EXTRA false gaps: they must not become global
# ===========================================================================

class Test03SupportExtra(unittest.TestCase):
    def test_thirteen_support_gaps_still_ten_global(self):
        base = uneven_gap_times(10)
        master = make_tracks(CAMERA_RIGHT_UP, base)
        # 3 false detections placed well away from any real gap.
        extras = [base[0] + 2.0, base[4] + 2.2, base[8] + 2.1]
        support = make_tracks(CAMERA_LEFT_UP, base + extras)

        state = assemble(master, [support])

        self.assertEqual(len(state.global_gaps), 10,
                         "EXTRA support observations must never create global gaps")
        self.assertEqual(state.total_wagons, 11)
        summary = state.support_alignment_summary[CAMERA_LEFT_UP]
        self.assertEqual(summary["n_match"], 10)
        self.assertEqual(summary["n_extra"], 3)
        self.assertEqual(len(state.extra_support_observations[CAMERA_LEFT_UP]), 3)

    def test_support_with_far_more_gaps_than_master(self):
        """The headline failure mode: support cameras must not inflate anything."""
        base = uneven_gap_times(10)
        master = make_tracks(CAMERA_RIGHT_UP, base)
        noisy = base + [t + 1.9 for t in base]          # 20 support gaps vs 10
        supports = [make_tracks(c, noisy) for c in
                    (CAMERA_LEFT_UP, CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP)]

        state = assemble(master, supports)

        self.assertEqual(len(state.global_gaps), 10)
        self.assertEqual(state.total_wagons, 11)
        self.assertEqual(state.corrections_applied, [],
                         "no insertion mechanism may exist")


# ===========================================================================
# TEST 4 -- duplicate support detections around ONE physical gap
# ===========================================================================

class Test04DuplicateSupport(unittest.TestCase):
    def test_two_nearby_detections_yield_one_global_gap(self):
        master = make_tracks(CAMERA_RIGHT_UP, [30.0])
        # Two detections 0.4 s apart, both plausibly the same physical gap.
        support = make_tracks(CAMERA_LEFT_UP, [29.8, 30.2])

        state = assemble(master, [support])

        self.assertEqual(len(state.global_gaps), 1, "one physical gap, one global gap")
        self.assertEqual(state.total_wagons, 2)
        summary = state.support_alignment_summary[CAMERA_LEFT_UP]
        self.assertEqual(summary["n_match"], 1, "only one detection may match")
        self.assertEqual(summary["n_extra"], 1, "the duplicate becomes EXTRA")

    def test_duplicates_across_a_longer_train(self):
        base = uneven_gap_times(8)
        master = make_tracks(CAMERA_RIGHT_UP, base)
        dupes = sorted(base + [t + 0.25 for t in base])
        support = make_tracks(CAMERA_LEFT_UP, dupes)

        state = assemble(master, [support])

        self.assertEqual(len(state.global_gaps), 8)
        summary = state.support_alignment_summary[CAMERA_LEFT_UP]
        self.assertEqual(summary["n_match"], 8)
        self.assertEqual(summary["n_extra"], 8)
        # No global gap may be matched twice.
        self.assertEqual(len(set(summary["matched_global_gap_ids"])),
                         len(summary["matched_global_gap_ids"]))


# ===========================================================================
# TEST 5 -- different video durations: no fabricated end-of-video evidence
# ===========================================================================

class Test05DifferentDurations(unittest.TestCase):
    def test_short_support_video_reports_unavailable_not_clamped(self):
        base = uneven_gap_times(12)                     # spans ~10 s .. ~70 s
        master = make_tracks(CAMERA_RIGHT_UP, base, duration_s=120.0)
        # LEFT_UP_TOP stops after 40 s and only saw the early gaps.
        early = [t for t in base if t < 38.0]
        short = make_tracks(CAMERA_LEFT_UP_TOP, early, duration_s=40.0)

        state = assemble(master, [short])

        self.assertEqual(len(state.global_gaps), 12, "an early-ending camera adds nothing")
        self.assertEqual(state.total_wagons, 13)

        # Late gaps must be 'unavailable' (no footage), never a fabricated match.
        late = [g for g in state.global_gaps if g["master_time"] > 45.0]
        self.assertTrue(late)
        for g in late:
            self.assertNotIn(CAMERA_LEFT_UP_TOP, g["support_observations"])
            self.assertIn(CAMERA_LEFT_UP_TOP, g["unavailable_cameras"])
            self.assertIn("out of range", g["unavailable_cameras"][CAMERA_LEFT_UP_TOP])
            self.assertNotIn(CAMERA_LEFT_UP_TOP, g["missing_cameras"],
                             "no footage is 'unavailable', not a detection failure")

    def test_projection_never_clamps(self):
        # 100 frames at 15 fps = valid local times [0, 6.6 s]
        self.assertIsNone(gf.project_global_time_to_local(50.0, 0.0, FPS, 100))
        self.assertIsNone(gf.project_global_time_to_local(-5.0, 0.0, FPS, 100))
        self.assertEqual(gf.project_global_time_to_local(2.0, 0.0, FPS, 100), 30)
        # With an offset, the same global instant maps elsewhere locally.
        self.assertEqual(gf.project_global_time_to_local(12.0, 10.0, FPS, 100), 30)
        self.assertFalse(gf.camera_covers(50.0, 0.0, FPS, 100))
        self.assertTrue(gf.camera_covers(2.0, 0.0, FPS, 100))


# ===========================================================================
# TEST 6 -- train order is preserved (no crossing matches)
# ===========================================================================

class Test06OrderPreservation(unittest.TestCase):
    def test_nearest_neighbour_would_cross_but_dp_does_not(self):
        # Master gaps 2 s apart; support observations deliberately arranged so a
        # greedy per-observation nearest match would pair them out of order.
        master_obs = gf.to_gap_observations(make_tracks(CAMERA_RIGHT_UP, [10.0, 12.0, 14.0]))
        support_obs = gf.to_gap_observations(
            make_tracks(CAMERA_LEFT_UP, [11.9, 10.1, 14.05]))
        # to_gap_observations sorts by time, so the support order is
        # [10.1, 11.9, 14.05] -- the DP must map them monotonically.
        cost, pairs, missing, extra = gf.align_to_master(master_obs, support_obs)

        m_idx = [p[0] for p in pairs]
        s_idx = [p[1] for p in pairs]
        self.assertEqual(m_idx, sorted(m_idx))
        self.assertEqual(s_idx, sorted(s_idx), "support indices must not cross")
        self.assertEqual(pairs, [(0, 0), (1, 1), (2, 2)])

    def test_extra_in_the_middle_does_not_shift_later_matches(self):
        master_obs = gf.to_gap_observations(
            make_tracks(CAMERA_RIGHT_UP, [10.0, 20.0, 30.0, 40.0, 50.0]))
        # X at 25 s corresponds to no master gap.
        support_obs = gf.to_gap_observations(
            make_tracks(CAMERA_LEFT_UP, [10.0, 20.0, 25.0, 30.0, 40.0, 50.0]))
        cost, pairs, missing, extra = gf.align_to_master(master_obs, support_obs)

        self.assertEqual(pairs, [(0, 0), (1, 1), (2, 3), (3, 4), (4, 5)])
        self.assertEqual(extra, [2], "the interloper is EXTRA")
        self.assertEqual(missing, [])

    def test_order_holds_under_random_perturbation(self):
        rng = random.Random(20260812)
        for trial in range(40):
            n = rng.randint(4, 18)
            base = uneven_gap_times(n)
            master_obs = gf.to_gap_observations(make_tracks(CAMERA_RIGHT_UP, base))
            noisy = [t + rng.uniform(-0.35, 0.35) for t in base]
            if rng.random() < 0.5:
                noisy += [rng.uniform(base[0], base[-1]) for _ in range(rng.randint(1, 4))]
            support_obs = gf.to_gap_observations(make_tracks(CAMERA_LEFT_UP, noisy))
            _, pairs, _, _ = gf.align_to_master(master_obs, support_obs)
            m_idx = [p[0] for p in pairs]
            s_idx = [p[1] for p in pairs]
            self.assertEqual(m_idx, sorted(set(m_idx)), f"trial {trial}")
            self.assertEqual(s_idx, sorted(set(s_idx)), f"trial {trial}")


# ===========================================================================
# TEST 7 -- the real dataset invariant (52 RIGHT_UP gaps -> 53 wagons)
# ===========================================================================

REAL_TRACKING = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "results", "per_camera_tracking.json")


class Test07RealData(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(REAL_TRACKING),
                         "results/per_camera_tracking.json not present")
    def test_real_gap_timelines_obey_the_invariant(self):
        import json
        with open(REAL_TRACKING, encoding="utf-8") as f:
            doc = json.load(f)

        def rebuild(cam):
            v = doc[cam]
            gaps = [GapEvent(
                track_id=g["track_id"], camera_id=cam,
                start_frame=g["start_frame"], end_frame=g["end_frame"],
                confidence=g["confidence"], hit_count=g["hit_count"],
                fps=v["fps"],
                temporal_consistency_score=g["temporal_consistency_score"],
            ) for g in v["gaps"]]
            return LocalCameraTracks(
                camera_id=cam, video_path=f"/real/{cam}.mp4", fps=v["fps"],
                total_frames=v["total_frames"], width=v["width"],
                height=v["height"], gaps=gaps)

        master = rebuild(CAMERA_RIGHT_UP)
        supports = [rebuild(c) for c in ALL_CAMERAS if c != CAMERA_RIGHT_UP]

        n_master = len(master.gaps)
        state = assemble(master, supports)

        # The invariant, on the real data. The old pipeline reported 64 wagons
        # from these same 52 master gaps by inserting 11 synthetic support gaps.
        self.assertEqual(len(state.global_gaps), n_master)
        self.assertEqual(state.invariant_checks["global_gap_count"], n_master)
        self.assertTrue(state.invariant_checks["invariant_holds"])
        self.assertEqual(state.total_wagons, n_master + 1)
        self.assertEqual(state.corrections_applied, [])

        # Support cameras genuinely have MORE gaps in total than the master, and
        # still cannot change the count.
        total_support = sum(len(s.gaps) for s in supports)
        self.assertGreater(total_support, n_master)
        self.assertNotEqual(state.total_wagons, 64,
                            "the old inflated result must not reappear")

        # Every global gap is master-sourced.
        for g in state.global_gaps:
            self.assertEqual(g["master_camera"], CAMERA_RIGHT_UP)
            self.assertIsNotNone(g["master_track_id"])

    def test_fifty_nine_gap_illustrative_case(self):
        """The scenario from the requirement: 59 master gaps -> 60 wagons,
        regardless of support-camera gap counts."""
        base = uneven_gap_times(59)
        master = make_tracks(CAMERA_RIGHT_UP, base, duration_s=base[-1] + 20)
        supports = [
            make_tracks(CAMERA_LEFT_UP, base[:55], duration_s=base[-1] + 20),
            make_tracks(CAMERA_RIGHT_UP_TOP,
                        sorted(base + [t + 1.8 for t in base[:4]]),
                        duration_s=base[-1] + 20),
            make_tracks(CAMERA_LEFT_UP_TOP,
                        sorted(base + [t + 1.9 for t in base[:11]]),
                        duration_s=base[-1] + 20),
        ]
        state = assemble(master, supports)
        self.assertEqual(len(state.global_gaps), 59)
        self.assertEqual(state.total_wagons, 60)
        self.assertNotIn(state.total_wagons, (70, 79, 80))


# ===========================================================================
# TEST 8 -- offset aliasing: never a confidently wrong whole-period shift
# ===========================================================================

class Test08OffsetAliasing(unittest.TestCase):
    def test_uniform_spacing_is_reported_unresolved(self):
        """With perfectly uniform spacing a k-period shift is indistinguishable.
        The estimator must admit that rather than guess."""
        spacing = 4.0
        base = [10.0 + i * spacing for i in range(12)]      # perfectly periodic
        master_obs = gf.to_gap_observations(make_tracks(CAMERA_RIGHT_UP, base))
        # Support is shifted by exactly 3 periods; timestamps align perfectly
        # under BOTH the true offset and the aliased one.
        support_obs = gf.to_gap_observations(
            make_tracks(CAMERA_LEFT_UP, [t - 3 * spacing for t in base]))

        off = gf.estimate_camera_offset(master_obs, support_obs,
                                       camera_id=CAMERA_LEFT_UP)
        if off.status == gf.OFFSET_RESOLVED:
            # Only acceptable if it happens to have found the true offset.
            self.assertAlmostEqual(off.delta, 3 * spacing, delta=0.3)
        else:
            self.assertEqual(off.status, gf.OFFSET_UNRESOLVED)
            self.assertIn("ambiguous", off.reason.lower())

    def test_drifting_spacing_breaks_the_alias(self):
        """Real trains change speed. That non-uniformity is what the pattern
        term exploits to pick the true offset over a whole-period alias."""
        base = uneven_gap_times(16)
        master_obs = gf.to_gap_observations(make_tracks(CAMERA_RIGHT_UP, base))
        true_delta = 9.0
        support_obs = gf.to_gap_observations(
            make_tracks(CAMERA_LEFT_UP, [t - true_delta for t in base]))

        off = gf.estimate_camera_offset(master_obs, support_obs,
                                       camera_id=CAMERA_LEFT_UP)
        self.assertEqual(off.status, gf.OFFSET_RESOLVED, off.reason)
        self.assertAlmostEqual(off.delta, true_delta, delta=0.3)

    def test_pattern_penalty_punishes_a_shifted_pairing(self):
        base = uneven_gap_times(14)
        m = gf.to_gap_observations(make_tracks(CAMERA_RIGHT_UP, base))
        s = gf.to_gap_observations(make_tracks(CAMERA_LEFT_UP, base))
        aligned = [(i, i) for i in range(len(base))]
        shifted = [(i + 2, i) for i in range(len(base) - 2)]
        self.assertLess(gf.interval_pattern_penalty(m, s, aligned),
                        gf.interval_pattern_penalty(m, s, shifted))

    def test_a_wrong_offset_cannot_change_the_count(self):
        """The central safety property: synchronization failure must cost
        evidence quality, never the number."""
        base = uneven_gap_times(10)
        master = make_tracks(CAMERA_RIGHT_UP, base)
        # A support camera whose gaps are pure noise -> nothing can align.
        rng = random.Random(7)
        noise = sorted(rng.uniform(0, 120) for _ in range(14))
        support = make_tracks(CAMERA_LEFT_UP, noise)

        state = assemble(master, [support])

        self.assertEqual(len(state.global_gaps), 10)
        self.assertEqual(state.total_wagons, 11)


# ===========================================================================
# TEST 9 -- UNRESOLVED offsets degrade safely
# ===========================================================================

class Test09UnresolvedDegradesSafely(unittest.TestCase):
    def test_all_supports_unresolved_still_counts(self):
        base = uneven_gap_times(10, start=30.0)
        master = make_tracks(CAMERA_RIGHT_UP, base)
        # Demand more matches than exist -> every camera must be UNRESOLVED.
        cfg = gf.FusionConfig(offset_min_match_fraction=1.5)
        supports = [make_tracks(c, [t - 5.0 for t in base])
                    for c in (CAMERA_LEFT_UP, CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP)]

        state = assemble(master, supports, cfg=cfg)

        self.assertEqual(len(state.global_gaps), 10)
        self.assertEqual(state.total_wagons, 11)
        for cam in (CAMERA_LEFT_UP, CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP):
            self.assertEqual(state.camera_offsets[cam]["status"], gf.OFFSET_UNRESOLVED)
        # No evidence claimed from an unsynchronized camera.
        for g in state.global_gaps:
            self.assertEqual(g["support_observations"], {})
            for cam in (CAMERA_LEFT_UP, CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP):
                self.assertIn(cam, g["unavailable_cameras"])

    def test_camera_with_no_metadata_is_handled(self):
        master = make_tracks(CAMERA_RIGHT_UP, uneven_gap_times(5))
        broken = LocalCameraTracks(camera_id=CAMERA_LEFT_UP, video_path="/x.mp4",
                                   fps=0.0, total_frames=0, gaps=[])
        state = assemble(master, [broken])
        self.assertEqual(len(state.global_gaps), 5)
        self.assertEqual(state.total_wagons, 6)


# ===========================================================================
# TEST 10 -- property test: the invariant under randomized support input
# ===========================================================================

class Test10InvariantProperty(unittest.TestCase):
    def test_invariant_holds_for_random_support_sequences(self):
        rng = random.Random(31337)
        for trial in range(60):
            n_master = rng.randint(1, 25)
            base = uneven_gap_times(n_master)
            duration = base[-1] + 20
            master = make_tracks(CAMERA_RIGHT_UP, base, duration_s=duration)

            supports = []
            for cam in (CAMERA_LEFT_UP, CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP):
                kept = [t for t in base if rng.random() > 0.25]           # misses
                delta = rng.uniform(-20, 20)
                obs = [t - delta + rng.uniform(-0.3, 0.3) for t in kept]
                obs += [rng.uniform(0, duration) for _ in range(rng.randint(0, 6))]
                supports.append(make_tracks(
                    cam, sorted(max(0.5, t) for t in obs),
                    duration_s=rng.choice([duration, duration * 0.6])))

            state = assemble(master, supports)

            msg = f"trial {trial} (master={n_master})"
            self.assertEqual(len(state.global_gaps), n_master, msg)
            self.assertEqual(state.invariant_checks["global_gap_count"], n_master, msg)
            self.assertTrue(state.invariant_checks["invariant_holds"], msg)
            self.assertLessEqual(state.total_wagons, n_master + 1, msg)
            self.assertEqual(state.corrections_applied, [], msg)
            for g in state.global_gaps:
                self.assertEqual(g["master_camera"], CAMERA_RIGHT_UP, msg)

    def test_support_gap_count_never_appears_in_the_total(self):
        """Vary ONLY the support cameras; the count must not move."""
        base = uneven_gap_times(9)
        master = make_tracks(CAMERA_RIGHT_UP, base)
        counts = set()
        for n_extra in (0, 1, 5, 20, 60):
            supports = [make_tracks(
                CAMERA_LEFT_UP,
                sorted(base + [10.0 + i * 0.7 for i in range(n_extra)]))]
            counts.add(assemble(master, supports).total_wagons)
        self.assertEqual(counts, {10}, "total_wagons must be support-independent")


# ===========================================================================
# TEST 11 -- assertions actually fire
# ===========================================================================

class Test11Assertions(unittest.TestCase):
    def test_invariant_violation_raises(self):
        master = make_tracks(CAMERA_RIGHT_UP, uneven_gap_times(5))
        global_gaps = gf.build_global_gap_sequence(master)
        global_gaps.pop()          # simulate a bug that loses a gap
        with self.assertRaises(gf.FusionInvariantError):
            gf.assert_invariants(
                global_gaps=global_gaps, master_tracks=master,
                wagons=[], alignments={}, support_tracks=[], strict=True)

    def test_support_sourced_gap_is_rejected(self):
        master = make_tracks(CAMERA_RIGHT_UP, uneven_gap_times(3))
        global_gaps = gf.build_global_gap_sequence(master)
        # Forge a support-sourced global gap -- exactly what must be impossible.
        bad = gf.to_gap_observations(make_tracks(CAMERA_LEFT_UP, [99.0]))[0]
        global_gaps[1].master_observation = bad
        with self.assertRaises(gf.FusionInvariantError):
            gf.assert_invariants(
                global_gaps=global_gaps, master_tracks=master,
                wagons=[], alignments={}, support_tracks=[], strict=True)

    def test_non_strict_mode_warns_instead(self):
        master = make_tracks(CAMERA_RIGHT_UP, uneven_gap_times(4))
        global_gaps = gf.build_global_gap_sequence(master)
        global_gaps.pop()
        rec = gf.assert_invariants(
            global_gaps=global_gaps, master_tracks=master, wagons=[],
            alignments={}, support_tracks=[], strict=False)
        self.assertFalse(rec["invariant_holds"])
        self.assertTrue(rec["violations"])

    def test_build_global_gap_sequence_ids_and_provenance(self):
        master = make_tracks(CAMERA_RIGHT_UP, uneven_gap_times(7))
        gaps = gf.build_global_gap_sequence(master)
        self.assertEqual([g.global_gap_id for g in gaps], list(range(1, 8)))
        for g in gaps:
            self.assertEqual(g.master_observation.camera_id, CAMERA_RIGHT_UP)
            self.assertEqual(g.master_camera, CAMERA_RIGHT_UP)


# ===========================================================================
# TEST 12 -- diagnostics are report-only, and metadata is truthful
# ===========================================================================

class Test12DiagnosticsAndMetadata(unittest.TestCase):
    def test_short_interval_is_flagged_but_not_removed(self):
        # A duplicate-looking boundary 0.4 s after a real one, among 4 s spacing.
        base = [10.0, 14.0, 18.0, 18.4, 22.0, 26.0, 30.0, 34.0, 38.0]
        master = make_tracks(CAMERA_RIGHT_UP, base)
        state = assemble(master, [])

        self.assertEqual(len(state.global_gaps), len(base),
                         "a flagged interval must NOT change the master sequence")
        flags = [d for d in state.interval_diagnostics
                 if d["flag"] == "SUSPICIOUSLY_SHORT"]
        self.assertTrue(flags, "the 0.4 s interval should be flagged")
        self.assertIn("diagnostic only", flags[0]["note"])

    def test_long_interval_is_flagged_but_nothing_inserted(self):
        base = [10.0, 14.0, 18.0, 22.0, 34.0, 38.0, 42.0, 46.0]   # 12 s hole
        master = make_tracks(CAMERA_RIGHT_UP, base)
        state = assemble(master, [])

        self.assertEqual(len(state.global_gaps), len(base))
        self.assertEqual(state.total_wagons, len(base) + 1)
        longs = [d for d in state.interval_diagnostics
                 if d["flag"] == "POSSIBLE_MISSING_GAP"]
        self.assertTrue(longs)
        self.assertGreaterEqual(longs[0]["implied_missing_gaps"], 1)
        self.assertEqual(state.corrections_applied, [],
                         "a possible missing gap must NOT be auto-inserted")

    def test_supporting_cameras_reflects_real_evidence(self):
        """The old code hard-coded all four cameras on every wagon."""
        base = uneven_gap_times(6)
        master = make_tracks(CAMERA_RIGHT_UP, base)
        # LEFT_UP sees everything; LEFT_UP_TOP has no gaps at all.
        supports = [make_tracks(CAMERA_LEFT_UP, base),
                    make_tracks(CAMERA_LEFT_UP_TOP, [])]

        state = assemble(master, supports)

        for w in state.wagons:
            self.assertIn(CAMERA_RIGHT_UP, w.supporting_cameras)
            self.assertNotIn(CAMERA_LEFT_UP_TOP, w.supporting_cameras,
                             "a camera with no observation must not be claimed")
        interior = [w for w in state.wagons
                    if (w.leading_gap or {}).get("source") == "master"]
        self.assertTrue(any(CAMERA_LEFT_UP in w.supporting_cameras for w in interior))

    def test_weighted_time_does_not_move_the_master_boundary(self):
        # Several gaps agree exactly, so the estimated offset is ~0 and is not
        # able to absorb the deliberate jitter on one single observation.
        base = uneven_gap_times(8, start=30.0)
        jittered = list(base)
        jittered[3] += 0.3                      # one observation 0.3 s late
        master = make_tracks(CAMERA_RIGHT_UP, base)
        support = make_tracks(CAMERA_LEFT_UP, jittered, confidence=0.99)

        state = assemble(master, [support])

        self.assertAlmostEqual(state.camera_offsets[CAMERA_LEFT_UP]["delta"],
                               0.0, delta=0.15)
        g = state.global_gaps[3]
        # Equal to within frame quantization (a gap centre is stored at
        # half-frame resolution, i.e. ~0.033 s at 15 fps).
        self.assertAlmostEqual(g["master_time"], base[3], delta=0.05,
                               msg="master coordinate must be untouched")
        self.assertEqual(g["master_frame"],
                         state.wagons[3].end_frame_master + 1,
                         "the boundary frame is the master's own")
        self.assertIsNotNone(g["weighted_time"])
        self.assertNotAlmostEqual(g["weighted_time"], g["master_time"], places=3)
        self.assertAlmostEqual(g["time_residuals"][CAMERA_LEFT_UP], 0.3, delta=0.08)
        # An exactly-agreeing gap has ~zero residual.
        self.assertAlmostEqual(state.global_gaps[0]["time_residuals"][CAMERA_LEFT_UP],
                               0.0, delta=0.08)


if __name__ == "__main__":
    unittest.main(verbosity=2)
