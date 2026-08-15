"""Fragment reassembly: rebuild one physical gap, never invent one.

These tests encode the two real EC2 cases that motivated the layer, plus the
safety properties that must hold for it to be trustworthy on trains this machine
has never seen.

The geometry below is SYNTHETIC. Where a number mirrors something measured
(a 3-frame fragment, a one-frame seam, a seam ~2.5x the per-frame advance) it is
reproducing the SHAPE of the real failure, not a constant from one train: every
test builds its own camera width, fps and speed, and the thresholds under test
are fractions, seconds and ratios that resolve against whatever those are.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fragment_stitching as fst  # noqa: E402
import gap_validation as gval
from global_train_state import CAMERA_LEFT_UP_TOP, CAMERA_RIGHT_UP, GapEvent

def _code_numbers(path):
    """Every numeric literal in a module's EXECUTABLE code.

    Strings and comments are dropped, so documented provenance ("measured 47.7
    px/frame") does not count while a real hardcoded constant does.
    """
    import io
    import tokenize
    out = set()
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(io.BytesIO(fh.read()).readline):
            if tok.type == tokenize.NUMBER:
                out.add(tok.string)
    return out


WIDTH = 960
FPS = 15.0
ADVANCE = 50.0          # px per frame interval: this camera's "train speed"


def frag(track_id, start_frame, n_hits, x_start, *, advance=ADVANCE, conf=0.92,
         camera_id=CAMERA_RIGHT_UP, fps=FPS, xs=None, frames=None):
    """One tracker fragment: `n_hits` consecutive detections advancing steadily."""
    if frames is None:
        frames = list(range(start_frame, start_frame + n_hits))
    if xs is None:
        xs = [x_start + advance * i for i in range(n_hits)]
    return GapEvent(
        track_id=track_id, camera_id=camera_id, start_frame=frames[0],
        end_frame=frames[-1], confidence=conf, hit_count=len(frames),
        center_x_trajectory=[float(x) for x in xs], fps=fps,
        temporal_consistency_score=1.0, hit_frames=list(frames),
        bbox_history=[[x - 30, 100, x + 30, 380] for x in xs])


def long_tracks(n=6, *, start=100, step=80, hits=12, x0=120.0,
                advance=ADVANCE, camera_id=CAMERA_RIGHT_UP, first_id=1):
    """A healthy population of full-length gaps.

    Reassembly refuses to act without enough long tracks to measure the advance
    rate from -- deliberately, since a fragment cannot measure train speed. So
    most tests need this backdrop to exercise anything at all.
    """
    out, f = [], start
    for i in range(n):
        out.append(frag(first_id + i, f, hits, x0, advance=advance,
                        camera_id=camera_id))
        f += step
    return out


def stitch(gaps, cfg=None, camera_id=CAMERA_RIGHT_UP, width=WIDTH, fps=FPS):
    return fst.reassemble_fragments(
        gaps, camera_id, cfg or fst.FragmentStitchConfig(),
        frame_width=width, fps=fps, verbose=False)


def chain_ids(res):
    return sorted((tuple(c.member_track_ids) for c in res.chains
                   if c.is_reassembled))


def validate(gaps, camera_id=CAMERA_RIGHT_UP, width=WIDTH, fps=FPS):
    return gval.validate_gap_events(gaps, camera_id, verbose=False,
                                    frame_width=width, fps=fps)


# ===========================================================================
# CASE 1 -- fragments of an uncounted gap reassemble, then validate normally
# ===========================================================================

class TestReassembleOneGap(unittest.TestCase):
    """The measured failure: 3 + 3 + 3 fragments of one physical gap."""

    def _fragmented_gap(self, base_frame, x0=150.0):
        # Three 3-hit fragments, one frame apart, each resuming ahead of the
        # last: the shape the tracker produces when a detection is missed and
        # the object reappears past the association gate.
        a = frag(90, base_frame, 3, x0)
        b = frag(91, base_frame + 4, 3, x0 + 4 * ADVANCE + 60)
        c = frag(92, base_frame + 8, 3, x0 + 8 * ADVANCE + 120)
        return [a, b, c]

    def test_three_fragments_become_one_physical_gap(self):
        gaps = long_tracks() + self._fragmented_gap(700)
        res = stitch(gaps)
        self.assertEqual(chain_ids(res), [(90, 91, 92)])

    def test_the_reassembled_gap_then_passes_validation(self):
        """Each piece is too short alone; the whole object is not."""
        pieces = self._fragmented_gap(700)
        before = validate(long_tracks() + pieces)
        lost = {r.features.track_id for r in before.rejected}
        self.assertEqual(lost, {90, 91, 92},
                         "every fragment must fail on its own -- that is the bug")
        self.assertTrue(all(r.reason == gval.REJECTED_TOO_SHORT
                            for r in before.rejected))

        after = validate(stitch(long_tracks() + pieces).events)
        self.assertEqual(after.rejected, [])
        self.assertEqual(len(after.accepted), len(before.accepted) + 1,
                         "exactly one gap gained, not three")

    def test_merged_track_carries_the_union_of_observations(self):
        res = stitch(long_tracks() + self._fragmented_gap(700))
        merged = next(c.merged for c in res.chains if c.is_reassembled)
        self.assertEqual(merged.hit_count, 9)
        self.assertEqual(merged.start_frame, 700)
        self.assertEqual(merged.end_frame, 710)
        self.assertEqual(len(merged.center_x_trajectory), 9)
        self.assertEqual(len(merged.hit_frames), 9)
        self.assertEqual(merged.hit_frames, sorted(merged.hit_frames))

    def test_merged_track_invents_no_observation(self):
        """Every merged sample must come from a member fragment."""
        pieces = self._fragmented_gap(700)
        source = {(f, round(x, 4)) for p in pieces
                  for f, x in zip(p.hit_frames, p.center_x_trajectory)}
        merged = next(c.merged for c in stitch(long_tracks() + pieces).chains
                      if c.is_reassembled)
        for f, x in zip(merged.hit_frames, merged.center_x_trajectory):
            self.assertIn((f, round(x, 4)), source)

    def test_two_separate_fragmented_gaps_stay_separate(self):
        """Multiple fragment groups -> multiple physical gaps, not one blob."""
        gaps = long_tracks() + self._fragmented_gap(700) + [
            frag(95, 900, 3, 150.0), frag(96, 904, 3, 150.0 + 4 * ADVANCE + 60),
            frag(97, 908, 3, 150.0 + 8 * ADVANCE + 120)]
        res = stitch(gaps)
        self.assertEqual(chain_ids(res), [(90, 91, 92), (95, 96, 97)])
        self.assertEqual(res.reassembled_count, 2)


# ===========================================================================
# CASE 2 -- a fragment of an ALREADY COUNTED gap must not add a gap
# ===========================================================================

class TestAbsorbIntoCountedGap(unittest.TestCase):
    """The double-count trap: the fragment's neighbour already validates."""

    def _leading_fragment_plus_real_track(self):
        lead = frag(80, 700, 3, 150.0)                     # 3 hits: too short alone
        real = frag(81, 704, 8, 150.0 + 4 * ADVANCE + 60)  # passes on its own
        return lead, real

    def test_fragment_is_absorbed_into_the_accepted_track(self):
        lead, real = self._leading_fragment_plus_real_track()
        res = stitch(long_tracks() + [lead, real])
        self.assertEqual(chain_ids(res), [(80, 81)])

    def test_absorption_does_not_change_the_gap_count(self):
        """The decisive property: the same physical gap stays ONE gap."""
        lead, real = self._leading_fragment_plus_real_track()
        base = long_tracks()
        without = validate(base + [real])
        with_frag = validate(stitch(base + [lead, real]).events)
        self.assertEqual(len(with_frag.accepted), len(without.accepted),
                         "absorbing a leading fragment must not add a wagon gap")
        self.assertEqual(with_frag.rejected, [])

    def test_absorbed_gap_starts_earlier_than_before(self):
        """Absorption is visible: the gap now spans its full observed extent."""
        lead, real = self._leading_fragment_plus_real_track()
        merged = next(c.merged for c in stitch(long_tracks() + [lead, real]).chains
                      if c.is_reassembled)
        self.assertEqual(merged.start_frame, lead.start_frame)
        self.assertEqual(merged.end_frame, real.end_frame)


# ===========================================================================
# never invent a gap: each refusal criterion
# ===========================================================================

class TestNeverInventsAGap(unittest.TestCase):
    def test_unrelated_nearby_fragments_are_not_stitched(self):
        """Close in time, but the second is nowhere near where the first left off."""
        a = frag(90, 700, 3, 150.0)
        b = frag(91, 704, 3, 150.0 + 4 * ADVANCE + 600)   # far beyond prediction
        res = stitch(long_tracks() + [a, b])
        self.assertEqual(chain_ids(res), [])
        self.assertTrue(any("seam jump" in s.get("refused_because", "")
                            for s in res.rejected_seams))

    def test_wrong_direction_fragment_is_not_stitched(self):
        """A fragment running against the camera's flow is a different object."""
        a = frag(90, 700, 3, 150.0)
        b = frag(91, 704, 3, 150.0 + 4 * ADVANCE + 60, advance=-ADVANCE)
        res = stitch(long_tracks() + [a, b])
        self.assertEqual(chain_ids(res), [])
        self.assertTrue(any("dominant direction" in s.get("refused_because", "")
                            for s in res.rejected_seams))

    def test_fragment_behind_its_predecessor_is_not_stitched(self):
        """Moving the right way, but positioned backwards: not a continuation."""
        a = frag(90, 700, 3, 500.0)
        b = frag(91, 704, 3, 200.0)
        res = stitch(long_tracks() + [a, b])
        self.assertEqual(chain_ids(res), [])
        self.assertTrue(any("not ahead" in s.get("refused_because", "")
                            for s in res.rejected_seams))

    def test_excessive_temporal_gap_is_not_stitched(self):
        """Two real adjacent gaps are far apart in time and must never merge."""
        cfg = fst.FragmentStitchConfig()
        far = int(cfg.max_seam_seconds * FPS) + 5
        a = frag(90, 700, 3, 150.0)
        b = frag(91, 700 + 3 + far, 3, 150.0 + 60)
        res = stitch(long_tracks() + [a, b], cfg)
        self.assertEqual(chain_ids(res), [])

    def test_excessive_seam_displacement_is_not_stitched(self):
        """Beyond the frame-width cap, regardless of what speed would allow."""
        cfg = fst.FragmentStitchConfig(seam_speed_tolerance=10_000.0)
        a = frag(90, 700, 3, 50.0)
        b = frag(91, 704, 3, 50.0 + cfg.max_seam_frac * WIDTH + 200)
        res = stitch(long_tracks() + [a, b], cfg)
        self.assertEqual(chain_ids(res), [])
        self.assertTrue(any("frame-width cap" in s.get("refused_because", "")
                            for s in res.rejected_seams))

    def test_overlapping_fragments_are_not_stitched(self):
        """Simultaneous tracks are duplicates or two gaps -- not one object."""
        a = frag(90, 700, 6, 150.0)
        b = frag(91, 702, 6, 150.0 + 2 * ADVANCE + 60)
        res = stitch(long_tracks() + [a, b])
        self.assertEqual(chain_ids(res), [])
        self.assertTrue(any("overlap in time" in s.get("refused_because", "")
                            for s in res.rejected_seams))

    def test_nothing_is_stitched_without_a_measurable_direction(self):
        """Too few candidates to know the camera's flow -> act conservatively."""
        res = stitch([frag(90, 700, 3, 150.0),
                      frag(91, 704, 3, 150.0 + 4 * ADVANCE + 60)])
        self.assertEqual(chain_ids(res), [])
        self.assertEqual(res.dominant_direction, 0)

    def test_nothing_is_stitched_without_enough_long_tracks(self):
        """A fragment cannot measure train speed, so no reference -> no stitch."""
        cfg = fst.FragmentStitchConfig(min_reference_tracks=99)
        gaps = long_tracks() + [frag(90, 700, 3, 150.0),
                                frag(91, 704, 3, 150.0 + 4 * ADVANCE + 60)]
        res = stitch(gaps, cfg)
        self.assertEqual(chain_ids(res), [])
        self.assertTrue(any("advance rate" in s.get("refused_because", "")
                            for s in res.rejected_seams))


# ===========================================================================
# raw-detection safety
# ===========================================================================

class TestRawDetectionSafety(unittest.TestCase):
    def test_isolated_short_detection_is_still_rejected(self):
        """No compatible neighbour -> untouched -> still fails validation."""
        gaps = long_tracks() + [frag(90, 700, 3, 150.0)]
        res = stitch(gaps)
        self.assertEqual(chain_ids(res), [])
        after = validate(res.events)
        self.assertIn(90, [r.features.track_id for r in after.rejected])
        self.assertEqual(next(r.reason for r in after.rejected
                              if r.features.track_id == 90),
                         gval.REJECTED_TOO_SHORT)

    def test_single_frame_detection_is_never_stitched(self):
        """A one-frame YOLO detection must never contribute to a wagon."""
        one = GapEvent(track_id=90, camera_id=CAMERA_RIGHT_UP, start_frame=700,
                       end_frame=700, confidence=0.99, hit_count=1,
                       center_x_trajectory=[150.0], fps=FPS, hit_frames=[700])
        gaps = long_tracks() + [one, frag(91, 702, 3, 150.0 + 2 * ADVANCE + 60)]
        res = stitch(gaps)
        self.assertEqual(chain_ids(res), [])
        self.assertNotIn(90, [g.track_id for g in validate(res.events).accepted])

    def test_a_pair_of_one_frame_detections_never_becomes_a_gap(self):
        one_a = GapEvent(track_id=90, camera_id=CAMERA_RIGHT_UP, start_frame=700,
                         end_frame=700, confidence=0.99, hit_count=1,
                         center_x_trajectory=[150.0], fps=FPS, hit_frames=[700])
        one_b = GapEvent(track_id=91, camera_id=CAMERA_RIGHT_UP, start_frame=702,
                         end_frame=702, confidence=0.99, hit_count=1,
                         center_x_trajectory=[250.0], fps=FPS, hit_frames=[702])
        res = stitch(long_tracks() + [one_a, one_b])
        self.assertEqual(chain_ids(res), [])
        accepted = {g.track_id for g in validate(res.events).accepted}
        self.assertNotIn(90, accepted)
        self.assertNotIn(91, accepted)

    def test_stitching_cannot_rescue_a_static_artefact(self):
        """Two stalled pieces at one position: not moving with the flow."""
        a = frag(90, 700, 3, 400.0, advance=0.0, conf=0.93)
        b = frag(91, 704, 3, 400.0, advance=0.0, conf=0.93)
        res = stitch(long_tracks() + [a, b])
        self.assertEqual(chain_ids(res), [])
        accepted = {g.track_id for g in validate(res.events).accepted}
        self.assertNotIn(90, accepted)
        self.assertNotIn(91, accepted)


# ===========================================================================
# speed changes: accelerate, decelerate, stop, resume
# ===========================================================================

class TestSpeedChanges(unittest.TestCase):
    def _population_then_fragments(self, pop_advance, frag_advance, seam_extra):
        """A population at one speed, then a fragmented gap at another."""
        pop = long_tracks(advance=pop_advance)
        base = 700
        a = frag(90, base, 3, 150.0, advance=frag_advance)
        b = frag(91, base + 4, 3, 150.0 + 3 * frag_advance + seam_extra,
                 advance=frag_advance)
        return pop, [a, b]

    def test_decelerating_train_still_reassembles(self):
        """The local reference must track the slowdown, not a stale median."""
        pop, pieces = self._population_then_fragments(ADVANCE, ADVANCE * 0.5,
                                                      ADVANCE * 0.5)
        res = stitch(pop + pieces)
        self.assertEqual(chain_ids(res), [(90, 91)])

    def test_accelerating_train_still_reassembles(self):
        pop, pieces = self._population_then_fragments(ADVANCE, ADVANCE * 1.6,
                                                      ADVANCE * 1.6)
        res = stitch(pop + pieces)
        self.assertEqual(chain_ids(res), [(90, 91)])

    def test_local_reference_follows_a_speed_change(self):
        """Reference near fast tracks must exceed reference near slow ones."""
        slow = long_tracks(n=5, start=100, step=80, advance=20.0, first_id=1)
        fast = long_tracks(n=5, start=2000, step=80, advance=90.0, first_id=50)
        frags = fst._describe(slow + fast)
        thr = fst.resolve_stitch_thresholds(fst.FragmentStitchConfig(), WIDTH, FPS)
        cfg = fst.FragmentStitchConfig()
        near_slow = fst._local_reference_advance(frags, 200.0, cfg, thr)
        near_fast = fst._local_reference_advance(frags, 2100.0, cfg, thr)
        self.assertLess(near_slow, near_fast)
        self.assertAlmostEqual(near_slow, 20.0, delta=6.0)
        self.assertAlmostEqual(near_fast, 90.0, delta=6.0)

    def test_stopped_train_fragments_can_still_rejoin(self):
        """During a stop a gap barely moves; the seam floor keeps stitching possible.

        Direction still has to hold, so this is drift, not a pinned artefact --
        and the reassembled track still faces the static/low-motion gates.
        """
        cfg = fst.FragmentStitchConfig()
        pop = long_tracks()
        creep = 2.0
        a = frag(90, 700, 3, 400.0, advance=creep)
        b = frag(91, 704, 3, 400.0 + 3 * creep + 1.0, advance=creep)
        res = stitch(pop + [a, b], cfg)
        self.assertEqual(chain_ids(res), [(90, 91)],
                         "a near-stationary seam must be inside the floor")

    def test_stop_then_resume_does_not_duplicate_the_gap(self):
        """Pieces before and after a resume make ONE gap, not two."""
        pop = long_tracks()
        a = frag(90, 700, 3, 300.0, advance=4.0)          # stalled
        b = frag(91, 704, 4, 300.0 + 3 * 4.0 + 30.0)      # resumed
        res = stitch(pop + [a, b])
        self.assertEqual(chain_ids(res), [(90, 91)])
        merged = next(c.merged for c in res.chains if c.is_reassembled)
        self.assertEqual(merged.hit_count, 7)
        # one physical gap in, one out
        self.assertEqual(len(res.events), len(pop) + 1)


# ===========================================================================
# determinism and multi-train isolation
# ===========================================================================

class TestDeterminismAndIsolation(unittest.TestCase):
    def _scenario(self):
        return long_tracks() + [frag(90, 700, 3, 150.0),
                                frag(91, 704, 3, 150.0 + 4 * ADVANCE + 60),
                                frag(92, 708, 3, 150.0 + 8 * ADVANCE + 120)]

    def test_result_is_independent_of_input_order(self):
        gaps = self._scenario()
        forward = stitch(gaps)
        backward = stitch(list(reversed(gaps)))
        self.assertEqual(chain_ids(forward), chain_ids(backward))
        self.assertEqual([g.start_frame for g in forward.events],
                         [g.start_frame for g in backward.events])

    def test_repeated_runs_agree_exactly(self):
        gaps = self._scenario()
        a, b = stitch(gaps), stitch(gaps)
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_input_events_are_not_mutated(self):
        gaps = self._scenario()
        snapshot = [(g.track_id, g.start_frame, g.end_frame, g.hit_count,
                     list(g.center_x_trajectory)) for g in gaps]
        stitch(gaps)
        self.assertEqual([(g.track_id, g.start_frame, g.end_frame, g.hit_count,
                           list(g.center_x_trajectory)) for g in gaps], snapshot)

    def test_two_trains_in_one_process_do_not_leak(self):
        """Sequential trains: differing geometry, speed and direction."""
        train_a = self._scenario()
        res_a1 = stitch(train_a)

        # A different camera: opposite flow, other resolution and frame rate.
        pop_b = long_tracks(n=6, start=50, step=60, hits=10, x0=1500.0,
                            advance=-70.0, camera_id=CAMERA_LEFT_UP_TOP)
        frags_b = [frag(90, 700, 3, 1500.0, advance=-70.0,
                        camera_id=CAMERA_LEFT_UP_TOP, fps=25.0),
                   frag(91, 704, 3, 1500.0 - 4 * 70.0 - 80.0, advance=-70.0,
                        camera_id=CAMERA_LEFT_UP_TOP, fps=25.0)]
        res_b = stitch(pop_b + frags_b, camera_id=CAMERA_LEFT_UP_TOP,
                       width=1920, fps=25.0)
        self.assertEqual(res_b.dominant_direction, -1)
        self.assertEqual(chain_ids(res_b), [(90, 91)])

        # Train A re-run afterwards must be byte-identical to its first run.
        self.assertEqual(stitch(train_a).to_dict(), res_a1.to_dict())

    def test_disabled_passes_everything_through_untouched(self):
        gaps = self._scenario()
        res = stitch(gaps, fst.FragmentStitchConfig(enabled=False))
        self.assertEqual([g.track_id for g in res.events],
                         [g.track_id for g in gaps])
        self.assertEqual(res.chains, [])


# ===========================================================================
# camera independence -- nothing in the config is tied to one geometry
# ===========================================================================

class TestCameraIndependence(unittest.TestCase):
    def test_config_declares_no_absolute_pixels_or_frames(self):
        cfg = fst.FragmentStitchConfig()
        for name in cfg.__dataclass_fields__:
            self.assertFalse(
                name.endswith("_px") or name.endswith("_frames"),
                f"{name} is an absolute quantity; the config must stay in "
                f"seconds, frame-width fractions, ratios and counts")

    def test_thresholds_scale_with_camera_geometry(self):
        cfg = fst.FragmentStitchConfig()
        small = fst.resolve_stitch_thresholds(cfg, 640, 10.0)
        large = fst.resolve_stitch_thresholds(cfg, 1920, 30.0)
        self.assertLess(small.max_seam_px, large.max_seam_px)
        self.assertLess(small.max_seam_frames, large.max_seam_frames)
        self.assertAlmostEqual(small.max_seam_px / 640,
                               large.max_seam_px / 1920, places=6)

    def test_the_same_scene_reassembles_at_any_resolution(self):
        """Scale the whole scene: the decision must not change."""
        for width, fps, scale in ((640, 15.0, 640 / 960), (1920, 30.0, 2.0)):
            adv = ADVANCE * scale
            pop = long_tracks(advance=adv, x0=120.0 * scale)
            pieces = [frag(90, 700, 3, 150.0 * scale, advance=adv),
                      frag(91, 704, 3, (150.0 + 60) * scale + 4 * adv,
                           advance=adv)]
            res = stitch(pop + pieces, width=width, fps=fps)
            self.assertEqual(chain_ids(res), [(90, 91)],
                             f"failed at width={width} fps={fps}")

    def test_no_train_specific_constants_in_the_module(self):
        """Only executable code is checked -- docstrings SHOULD cite measurements.

        Tokenising rather than string-splitting matters: a docstring that records
        where a threshold came from is exactly what we want, so a crude text
        filter would either flag legitimate provenance or miss a real constant.
        """
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "fragment_stitching.py")
        numbers = _code_numbers(path)
        for value in ("3107", "3252", "184", "654.76", "960", "341.76", "47.7"):
            self.assertNotIn(value, numbers,
                             f"train/camera-specific constant {value!r} appears "
                             f"in executable code, not just documentation")


# ===========================================================================
# the master camera keeps authority
# ===========================================================================

class TestMasterAuthority(unittest.TestCase):
    def test_reassembly_only_ever_reduces_the_candidate_count(self):
        """Merging cannot manufacture candidates."""
        gaps = long_tracks() + [frag(90, 700, 3, 150.0),
                                frag(91, 704, 3, 150.0 + 4 * ADVANCE + 60)]
        res = stitch(gaps)
        self.assertLessEqual(len(res.events), len(gaps))
        self.assertEqual(len(res.events),
                         len(gaps) - res.absorbed_fragment_count)

    def test_every_input_track_is_accounted_for(self):
        """No candidate silently disappears: each is a member of exactly one chain."""
        gaps = long_tracks() + [frag(90, 700, 3, 150.0),
                                frag(91, 704, 3, 150.0 + 4 * ADVANCE + 60),
                                frag(95, 1500, 3, 400.0, advance=0.0)]
        res = stitch(gaps)
        members = [t for c in res.chains for t in c.member_track_ids]
        self.assertEqual(sorted(members), sorted(g.track_id for g in gaps))
        self.assertEqual(len(members), len(set(members)),
                         "a track must belong to exactly one chain")

    def test_a_support_camera_reassembly_is_independent_of_the_master(self):
        master = long_tracks() + [frag(90, 700, 3, 150.0),
                                  frag(91, 704, 3, 150.0 + 4 * ADVANCE + 60)]
        m1 = stitch(master)
        support = long_tracks(camera_id=CAMERA_LEFT_UP_TOP, advance=-ADVANCE,
                              x0=900.0)
        stitch(support, camera_id=CAMERA_LEFT_UP_TOP)
        self.assertEqual(stitch(master).to_dict(), m1.to_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
