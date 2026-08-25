"""The wagon active region is anchored at the END and derived backwards.

The defect these tests pin, validated on real footage: the trailing
WAGON -> BRAKE_VAN transition is reliably detected, but the leading
ENGINE -> first-WAGON coupling is not. When that coupling is missed,
`build_global_wagons` emits the locomotive and the first wagon as ONE segment,
the segment inherits the ENGINE label, and the forward rule -- "the region starts
at the first segment labelled WAGON" -- starts it at the SECOND wagon. A real
wagon disappears, and nothing in the output says so.

So the reliable end becomes the anchor and the leading edge is walked backwards
from it. The walk is a boundary mechanism, not a second counting algorithm: it
can only retain a segment the master gaps already produced, and every test below
that expects an extension also checks the count, the GW ids and that the
retained thing is not a bare locomotive.

Durations are written in MEDIAN WAGONS rather than frames, because that is the
scale every threshold is expressed in -- a fixture in raw frames would silently
depend on the fps and stop testing what it claims to.
"""

from __future__ import annotations

import os
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

import gap_validation as gv
import train_structure as ts
from global_train_state import GlobalWagon, SegmentClass

W = SegmentClass.WAGON
E = SegmentClass.ENGINE
B = SegmentClass.BRAKE_VAN
U = SegmentClass.UNKNOWN

FPS = 15.0
WAGON_FRAMES = 60           # one median wagon = 4 s at 15 fps

#: A real soft rejection reason -- relaxable, per `SOFT_REJECTION_REASONS`.
SOFT = gv.REJECTED_LOW_CONFIDENCE
#: A real hard rejection reason -- one of the false-positive defences.
HARD = gv.REJECTED_STATIC


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def segments(spec):
    """Build a master segment list from `[(label, length_in_median_wagons)]`.

    Frames are INCLUSIVE (`start_frame_master..end_frame_master`) and
    `end_time == (end_frame + 1) / fps`, matching
    `global_alignment.build_global_wagons` exactly -- these tests assert on
    boundary frames, so the fixture must not invent its own convention.
    """
    out, f = [], 0
    for i, (label, mult) in enumerate(spec, start=1):
        span = int(round(WAGON_FRAMES * mult))
        out.append(GlobalWagon(
            global_id=f"SEG_{i}", wagon_index=i,
            start_frame_master=f, end_frame_master=f + span - 1,
            start_time=f / FPS, end_time=(f + span) / FPS,
            classification=label, classification_confidence=0.95))
        f += span
    return out


def soft_gap_at(center: int, *, half: int = 5, reason: str = SOFT):
    return ts.RejectedGapSpan(frame_start=center - half, frame_end=center + half,
                              reason=reason, soft=True, track_id=7)


def hard_gap_at(center: int, *, half: int = 5, reason: str = HARD):
    return ts.RejectedGapSpan(frame_start=center - half, frame_end=center + half,
                              reason=reason, soft=False, track_id=7)


def window(spec, rejected=(), **kw):
    return ts.get_master_wagon_window(
        segments(spec), rejected_gap_spans=list(rejected), verbose=False, **kw)


#: The normal train, and the same train with its leading coupling missed.
NORMAL = [(E, 2.5)] + [(W, 1.0)] * 5 + [(B, 1.2), (E, 2.5)]
MERGED = [(E, 3.5)] + [(W, 1.0)] * 4 + [(B, 1.2), (E, 2.5)]
#: In MERGED the locomotive ends 2.5 median wagons in -- the missed coupling.
MERGED_COUPLING = int(2.5 * WAGON_FRAMES)


# ===========================================================================
# 1. Normal train -- ENGINE, WAGON x N, BRAKE_VAN, ENGINE
# ===========================================================================

class TestNormalTrain(unittest.TestCase):

    def test_the_count_is_the_wagons_only(self):
        w = window(NORMAL)
        self.assertEqual(w.master_wagon_count, 5)
        self.assertEqual([u.global_id for u in w.wagon_units],
                         [f"GW_{i}" for i in range(1, 6)])

    def test_the_leading_engine_stays_outside(self):
        w = window(NORMAL)
        self.assertEqual([o.classification for o in w.leading_non_wagon_objects],
                         [E])
        self.assertEqual(w.wagon_start_frame,
                         int(2.5 * WAGON_FRAMES))

    def test_the_trailing_brake_van_and_engine_stay_outside(self):
        w = window(NORMAL)
        self.assertEqual([o.classification for o in w.trailing_non_wagon_objects],
                         [B, E])

    def test_the_forward_and_reverse_boundaries_agree(self):
        a = window(NORMAL).reverse_anchor
        self.assertTrue(a.boundaries_agree)
        self.assertEqual(a.reason, ts.REVERSE_REASON_AGREES)
        self.assertEqual(a.disagreement, "")
        self.assertEqual(a.forward_first_wagon_segment_index,
                         a.first_wagon_segment_index)

    def test_the_end_anchor_is_the_last_wagon(self):
        a = window(NORMAL).reverse_anchor
        self.assertEqual(a.last_wagon, "GW_5")
        self.assertEqual(a.last_wagon_segment_index, 5)
        self.assertEqual(a.end_frame, int(7.5 * WAGON_FRAMES) - 1)

    def test_a_soft_rejection_in_a_bare_locomotive_changes_nothing(self):
        """The engine here is 2.5 wagons long, so it passes the duration test;
        only the geometry keeps it out."""
        w = window(NORMAL, [soft_gap_at(int(0.3 * WAGON_FRAMES))])
        self.assertEqual(w.master_wagon_count, 5)
        self.assertTrue(w.reverse_anchor.boundaries_agree)

    def test_the_diagnostics_name_every_required_field(self):
        d = window(NORMAL).reverse_anchor.to_dict()
        for k in ("end_frame", "last_wagon", "start_frame", "first_wagon",
                  "wagons_retained", "leading_non_wagon", "trailing_non_wagon",
                  "reason", "boundaries_agree", "disagreement",
                  "forward_first_wagon_segment_index"):
            self.assertIn(k, d)

    def test_the_render_line_is_the_documented_one(self):
        line = window(NORMAL).reverse_anchor.render()
        self.assertIn("[ACTIVE-REGION] REVERSE-ANCHOR", line)
        for k in ("end_frame=", "last_wagon=", "start_frame=", "first_wagon=",
                  "wagons_retained=", "leading_non_wagon=",
                  "trailing_non_wagon=", "reason="):
            self.assertIn(k, line)


# ===========================================================================
# 2. The failure case: a missed leading ENGINE -> WAGON coupling
# ===========================================================================

class TestMissedLeadingCoupling(unittest.TestCase):
    """The wagon the forward method loses."""

    def test_the_forward_method_loses_the_first_wagon(self):
        """Baseline. Without the rejected-candidate evidence the walk has
        nothing to go on and reproduces the old answer exactly -- which is what
        makes the next test a real difference and not a fixture artefact."""
        w = window(MERGED)
        self.assertEqual(w.master_wagon_count, 4)
        self.assertTrue(w.reverse_anchor.boundaries_agree)

    def test_the_reverse_method_retains_it(self):
        w = window(MERGED, [soft_gap_at(MERGED_COUPLING)])
        self.assertEqual(w.master_wagon_count, 5)
        self.assertFalse(w.reverse_anchor.boundaries_agree)
        self.assertEqual(w.reverse_anchor.reason,
                         ts.REVERSE_REASON_EXTENDED)

    def test_the_region_now_starts_at_the_merged_segment(self):
        w = window(MERGED, [soft_gap_at(MERGED_COUPLING)])
        self.assertEqual(w.wagon_start_frame, 0)
        self.assertEqual(w.reverse_anchor.start_frame, 0)
        self.assertEqual(w.reverse_anchor.forward_start_frame,
                         int(3.5 * WAGON_FRAMES))

    def test_the_end_anchor_did_not_move(self):
        """Only the leading edge is in question; the anchor is the fixed point."""
        before = window(MERGED).reverse_anchor
        after = window(MERGED, [soft_gap_at(MERGED_COUPLING)]).reverse_anchor
        self.assertEqual(before.end_frame, after.end_frame)
        self.assertEqual(before.last_wagon_segment_index,
                         after.last_wagon_segment_index)

    def test_the_disagreement_is_recorded_not_hidden(self):
        a = window(MERGED, [soft_gap_at(MERGED_COUPLING)]).reverse_anchor
        self.assertTrue(a.disagreement)
        self.assertIn("forward boundary", a.disagreement)
        self.assertIn("end-anchored", a.disagreement)

    def test_the_retained_segment_is_recorded_with_its_evidence(self):
        w = window(MERGED, [soft_gap_at(MERGED_COUPLING)])
        ext = w.reverse_anchor.extended_segments
        self.assertEqual(len(ext), 1)
        self.assertEqual(ext[0]["segment_index"], 0)
        self.assertEqual(ext[0]["classification"], E)
        self.assertEqual(ext[0]["rejected_boundary"]["center_frame"],
                         MERGED_COUPLING)
        self.assertGreater(ext[0]["trailing_wagon_frames"], 0)

    def test_the_gw_ids_stay_contiguous_from_one(self):
        w = window(MERGED, [soft_gap_at(MERGED_COUPLING)])
        self.assertEqual([u.global_id for u in w.wagon_units],
                         [f"GW_{i}" for i in range(1, 6)])
        self.assertEqual([u.wagon_index for u in w.wagon_units],
                         list(range(1, 6)))

    def test_the_retained_segment_is_not_filed_as_interior(self):
        """It sits at the leading edge, not inside the run; calling it interior
        would misreport where the ambiguity is."""
        w = window(MERGED, [soft_gap_at(MERGED_COUPLING)])
        self.assertEqual(w.interior_non_wagon_objects, [])
        self.assertEqual(len(w.reverse_extended_objects), 1)
        self.assertEqual(w.reverse_extended_objects[0].position, "leading_merged")

    def test_the_segment_accounting_still_balances(self):
        """`wagons + leading + trailing == total_segments` is a fusion invariant;
        a retained segment must move between the buckets, not be duplicated."""
        for rejected in ([], [soft_gap_at(MERGED_COUPLING)]):
            w = window(MERGED, rejected)
            self.assertEqual(
                w.master_wagon_count
                + len(w.leading_non_wagon_objects)
                + len(w.trailing_non_wagon_objects),
                w.total_segments, f"rejected={bool(rejected)}")


# ===========================================================================
# 3. Evidence quality: what may and may not open the region
# ===========================================================================

class TestOnlyCanonicalEvidenceOpensTheRegion(unittest.TestCase):

    def test_a_hard_rejected_candidate_is_ignored(self):
        """`gap_validation` calls the hard reasons "the false-positive
        defences", and `recover_wagon_active_candidates` re-admits only soft
        failures. Treating a hard rejection as boundary evidence would smuggle
        back exactly what validation exists to discard."""
        w = window(MERGED, [hard_gap_at(MERGED_COUPLING)])
        self.assertEqual(w.master_wagon_count, 4)
        self.assertTrue(w.reverse_anchor.boundaries_agree)
        self.assertEqual(w.reverse_anchor.rejected_extensions[0]["reason"],
                         ts.REVERSE_REASON_NO_EVIDENCE)
        self.assertEqual(
            w.reverse_anchor.rejected_extensions[0]["hard_rejections_ignored"], 1)

    def test_no_evidence_at_all_keeps_the_forward_boundary(self):
        w = window(MERGED, [])
        self.assertTrue(w.reverse_anchor.boundaries_agree)
        self.assertEqual(w.master_wagon_count, 4)

    def test_a_candidate_at_the_segment_edge_is_not_a_missed_boundary(self):
        """The existing boundary re-detected, not a new one."""
        segs = MERGED
        end = int(3.5 * WAGON_FRAMES) - 1
        for c in (2, end - 2):
            w = window(segs, [soft_gap_at(c, half=1)])
            self.assertEqual(w.master_wagon_count, 4, f"center={c}")

    def test_a_candidate_leaving_no_room_for_a_wagon_is_refused(self):
        """A rejection just before the segment ends cannot be a coupling with a
        wagon behind it."""
        end = int(3.5 * WAGON_FRAMES) - 1
        w = window(MERGED, [soft_gap_at(end - int(0.2 * WAGON_FRAMES))])
        self.assertEqual(w.master_wagon_count, 4)

    def test_a_segment_too_short_for_two_vehicles_is_refused(self):
        """A one-wagon-long leading ENGINE cannot also contain a wagon, whatever
        was rejected inside it."""
        spec = [(E, 1.0)] + [(W, 1.0)] * 4 + [(B, 1.2)]
        w = window(spec, [soft_gap_at(int(0.5 * WAGON_FRAMES))])
        self.assertEqual(w.master_wagon_count, 4)
        self.assertEqual(w.reverse_anchor.rejected_extensions[0]["reason"],
                         "TOO_SHORT_TO_HOLD_ENGINE_AND_WAGON")

    def test_the_walk_never_makes_a_wagon_from_a_pure_engine_train(self):
        for spec in ([(E, 2.5)], [(E, 2.5), (E, 3.0)], [(E, 4.0), (B, 1.2)]):
            w = window(spec, [soft_gap_at(int(1.2 * WAGON_FRAMES))])
            self.assertEqual(w.master_wagon_count, 0, spec)
            self.assertFalse(w.found, spec)

    def test_the_walk_stops_after_one_merged_segment(self):
        """Behind the coupling is pure locomotive. Two ENGINE segments both
        carrying soft rejections must still yield exactly one extension --
        otherwise the walk eats its way up the train."""
        spec = [(E, 3.5), (E, 3.5)] + [(W, 1.0)] * 3 + [(B, 1.2)]
        rejected = [soft_gap_at(int(2.5 * WAGON_FRAMES)),
                    soft_gap_at(int(3.5 * WAGON_FRAMES) + int(2.5 * WAGON_FRAMES))]
        w = window(spec, rejected)
        self.assertEqual(len(w.reverse_anchor.extended_segments), 1)
        self.assertEqual(w.master_wagon_count, 4)
        self.assertEqual(len(w.leading_non_wagon_objects), 1)

    def test_the_median_is_measured_before_any_extension(self):
        """A retained segment must not widen the yardstick it was measured
        against -- that would let each extension justify the next."""
        w = window(MERGED, [soft_gap_at(MERGED_COUPLING)])
        self.assertAlmostEqual(w.reverse_anchor.median_wagon_frames,
                               float(WAGON_FRAMES))


# ===========================================================================
# 4. Missed trailing coupling
# ===========================================================================

class TestMissedTrailingCoupling(unittest.TestCase):
    """A merged last-WAGON + BRAKE_VAN. The end anchor is where it lands, and
    the reverse walk must not react by moving the leading edge."""

    SPEC = [(E, 2.5)] + [(W, 1.0)] * 4 + [(W, 2.2)]     # last wagon + van merged

    def test_the_leading_boundary_is_unaffected(self):
        w = window(self.SPEC, [soft_gap_at(0)])
        self.assertEqual(w.wagon_start_frame, int(2.5 * WAGON_FRAMES))
        self.assertTrue(w.reverse_anchor.boundaries_agree)

    def test_the_anchor_is_the_last_wagon_labelled_segment(self):
        w = window(self.SPEC)
        self.assertEqual(w.reverse_anchor.last_wagon_segment_index,
                         len(self.SPEC) - 1)
        self.assertEqual(w.reverse_anchor.end_frame, w.wagon_end_frame)
        self.assertEqual(w.trailing_non_wagon_objects, [])

    def test_a_trailing_brake_van_labelled_wagon_is_still_counted(self):
        """Retained by the master gaps, as before: classification never deletes
        an individual wagon."""
        w = window(self.SPEC)
        self.assertEqual(w.master_wagon_count, 5)

    def test_a_soft_rejection_in_the_merged_tail_does_not_extend_the_end(self):
        """The walk owns the LEADING edge only. The end anchor comes from the
        labels, and no rejected candidate may move it."""
        tail_center = int((2.5 + 4 + 1.1) * WAGON_FRAMES)
        a = window(self.SPEC, [soft_gap_at(tail_center)]).reverse_anchor
        self.assertEqual(a.end_frame, window(self.SPEC).reverse_anchor.end_frame)


# ===========================================================================
# 5. False interior ENGINE / BRAKE_VAN classification
# ===========================================================================

class TestFalseInteriorNonWagon(unittest.TestCase):
    """A misclassified wagon in the middle of the run. It is still counted --
    classification decides where the region starts and ends, never whether an
    individual wagon exists."""

    SPEC = [(E, 2.5), (W, 1.0), (W, 1.0), (B, 1.0), (W, 1.0), (W, 1.0),
            (B, 1.2), (E, 2.5)]

    def test_the_interior_anomaly_is_counted(self):
        w = window(self.SPEC)
        self.assertEqual(w.master_wagon_count, 5)
        self.assertEqual(len(w.interior_non_wagon_objects), 1)
        self.assertEqual(w.interior_non_wagon_objects[0].classification, B)

    def test_the_backward_walk_does_not_stop_at_it(self):
        """It sits between the first and last WAGON labels, so the walk never
        reaches it -- but a walk that scanned backwards blindly would stop
        there and truncate the region."""
        w = window(self.SPEC)
        self.assertEqual(w.wagon_start_frame, int(2.5 * WAGON_FRAMES))
        self.assertTrue(w.reverse_anchor.boundaries_agree)

    def test_the_gw_ids_are_contiguous_across_the_anomaly(self):
        w = window(self.SPEC)
        self.assertEqual([u.global_id for u in w.wagon_units],
                         [f"GW_{i}" for i in range(1, 6)])

    def test_a_false_interior_engine_behaves_the_same(self):
        spec = [(E, 2.5), (W, 1.0), (E, 1.0), (W, 1.0), (B, 1.2)]
        w = window(spec)
        self.assertEqual(w.master_wagon_count, 3)
        self.assertEqual(len(w.interior_non_wagon_objects), 1)


# ===========================================================================
# 6. Noisy WAGON predictions in the leading engine region
# ===========================================================================

class TestNoisyWagonInTheLeadingEngine(unittest.TestCase):
    """A misread frame run inside the locomotive must not open the region.

    Here the noise has already survived smoothing and shows up as a WAGON-
    labelled segment inside the engine -- the worst case, because the forward
    rule would start the region there.
    """

    def test_a_short_noisy_wagon_label_inside_the_engine_is_counted_as_before(self):
        """Not a regression this change introduces: an engine segment split by a
        WAGON label is already the forward rule's answer, and the backward walk
        deliberately does not second-guess a WAGON label."""
        spec = [(E, 1.2), (W, 0.15), (E, 1.2)] + [(W, 1.0)] * 4 + [(B, 1.2)]
        w = window(spec)
        a = w.reverse_anchor
        self.assertEqual(a.forward_first_wagon_segment_index, 1)
        self.assertEqual(a.first_wagon_segment_index, 1)
        self.assertTrue(a.boundaries_agree)

    def test_noise_plus_a_soft_rejection_does_not_extend_further_back(self):
        """The engine ahead of the noise must stay out."""
        spec = [(E, 1.2), (W, 0.15), (E, 1.2)] + [(W, 1.0)] * 4 + [(B, 1.2)]
        w = window(spec, [soft_gap_at(int(0.6 * WAGON_FRAMES))])
        self.assertEqual(w.reverse_anchor.first_wagon_segment_index, 1)
        self.assertEqual(len(w.reverse_anchor.extended_segments), 0)
        self.assertEqual(len(w.leading_non_wagon_objects), 1)

    def test_a_noisy_leading_engine_with_no_wagons_yields_no_wagons(self):
        w = window([(E, 2.0), (E, 2.0)], [soft_gap_at(int(1.0 * WAGON_FRAMES))])
        self.assertEqual(w.master_wagon_count, 0)


# ===========================================================================
# 7. Short and long wagon durations
# ===========================================================================

class TestWagonDurationExtremes(unittest.TestCase):
    """Thresholds are ratios of THIS train's median wagon, so a slow train and a
    fast one must behave the same."""

    def test_a_slow_train_with_long_wagons(self):
        scale = 3.0
        spec = [(E, 2.5 * scale)] + [(W, 1.0 * scale)] * 4 + [(B, 1.2 * scale)]
        w = window(spec, [soft_gap_at(int(2.5 * scale * WAGON_FRAMES))])
        self.assertEqual(w.master_wagon_count, 4)
        merged = [(E, 3.5 * scale)] + [(W, 1.0 * scale)] * 4 + [(B, 1.2 * scale)]
        w2 = window(merged, [soft_gap_at(int(2.5 * scale * WAGON_FRAMES))])
        self.assertEqual(w2.master_wagon_count, 5)
        self.assertFalse(w2.reverse_anchor.boundaries_agree)

    def test_a_fast_train_with_short_wagons(self):
        scale = 0.25
        merged = [(E, 3.5 * scale)] + [(W, 1.0 * scale)] * 4 + [(B, 1.2 * scale)]
        w = window(merged, [soft_gap_at(int(2.5 * scale * WAGON_FRAMES), half=1)])
        self.assertEqual(w.master_wagon_count, 5)
        self.assertFalse(w.reverse_anchor.boundaries_agree)

    def test_wagons_of_uneven_length_use_the_median_not_the_mean(self):
        """One very long wagon must not drag the yardstick up and stop the
        mechanism working on the rest."""
        spec = [(E, 3.5), (W, 1.0), (W, 1.0), (W, 4.0), (W, 1.0), (B, 1.2)]
        w = window(spec, [soft_gap_at(MERGED_COUPLING)])
        self.assertAlmostEqual(w.reverse_anchor.median_wagon_frames,
                               float(WAGON_FRAMES))
        self.assertEqual(w.master_wagon_count, 5)


# ===========================================================================
# 8. Exact boundary frames -- no off-by-one
# ===========================================================================

class TestExactBoundaryFrames(unittest.TestCase):
    """Segments are INCLUSIVE in frames and their `end_time` is exclusive
    (`(end_frame + 1) / fps`). The walk selects segments and must read those
    numbers off them, never recompute them."""

    def test_the_window_frames_are_the_retained_segments_own_frames(self):
        for spec, rejected in ((NORMAL, []),
                               (MERGED, [soft_gap_at(MERGED_COUPLING)])):
            w = window(spec, rejected)
            self.assertEqual(w.wagon_start_frame,
                             w.wagon_units[0].start_frame_master)
            self.assertEqual(w.wagon_end_frame,
                             w.wagon_units[-1].end_frame_master)

    def test_the_window_times_are_the_retained_segments_own_times(self):
        w = window(MERGED, [soft_gap_at(MERGED_COUPLING)])
        self.assertEqual(w.wagon_start_time, w.wagon_units[0].start_time)
        self.assertEqual(w.wagon_end_time, w.wagon_units[-1].end_time)

    def test_the_end_time_stays_exclusive(self):
        w = window(NORMAL)
        last = w.wagon_units[-1]
        self.assertAlmostEqual(w.wagon_end_time,
                               (last.end_frame_master + 1) / FPS)

    def test_the_anchor_reports_the_same_frames_as_the_window(self):
        for spec, rejected in ((NORMAL, []),
                               (MERGED, [soft_gap_at(MERGED_COUPLING)])):
            w = window(spec, rejected)
            self.assertEqual(w.reverse_anchor.start_frame, w.wagon_start_frame)
            self.assertEqual(w.reverse_anchor.end_frame, w.wagon_end_frame)

    def test_the_region_is_contiguous_with_no_dropped_frame(self):
        w = window(MERGED, [soft_gap_at(MERGED_COUPLING)])
        for a, b in zip(w.wagon_units, w.wagon_units[1:]):
            self.assertEqual(b.start_frame_master, a.end_frame_master + 1)

    def test_the_leading_region_ends_exactly_where_the_window_starts(self):
        w = window(NORMAL)
        last_lead = w.leading_non_wagon_objects[-1]
        self.assertEqual(last_lead.end_frame + 1, w.wagon_start_frame)

    def test_the_trailing_region_starts_exactly_after_the_window(self):
        w = window(NORMAL)
        first_trail = w.trailing_non_wagon_objects[0]
        self.assertEqual(first_trail.start_frame, w.wagon_end_frame + 1)


# ===========================================================================
# 9. One-wagon trains
# ===========================================================================

class TestSingleWagonTrain(unittest.TestCase):

    def test_one_wagon_between_an_engine_and_a_brake_van(self):
        w = window([(E, 2.5), (W, 1.0), (B, 1.2)])
        self.assertEqual(w.master_wagon_count, 1)
        self.assertEqual(w.wagon_units[0].global_id, "GW_1")
        a = w.reverse_anchor
        self.assertEqual(a.first_wagon, "GW_1")
        self.assertEqual(a.last_wagon, "GW_1")
        self.assertEqual(a.first_wagon_segment_index,
                         a.last_wagon_segment_index)

    def test_the_single_wagons_own_frames_bound_the_region(self):
        w = window([(E, 2.5), (W, 1.0), (B, 1.2)])
        self.assertEqual(w.wagon_start_frame, int(2.5 * WAGON_FRAMES))
        self.assertEqual(w.wagon_end_frame, int(3.5 * WAGON_FRAMES) - 1)

    def test_a_single_wagon_merged_into_the_engine_is_recovered(self):
        """The hardest version of the failure: lose the first wagon and the
        train has none at all. There is no other wagon to take a median from,
        so the scale comes from the one wagon the labels do give."""
        spec = [(E, 3.5), (W, 1.0), (B, 1.2)]
        w = window(spec, [soft_gap_at(MERGED_COUPLING)])
        self.assertEqual(w.master_wagon_count, 2)
        self.assertFalse(w.reverse_anchor.boundaries_agree)

    def test_a_train_with_no_wagon_label_counts_nothing(self):
        w = window([(E, 2.5), (B, 1.2)], [soft_gap_at(int(1.2 * WAGON_FRAMES))])
        self.assertFalse(w.found)
        self.assertEqual(w.master_wagon_count, 0)
        self.assertIsNone(w.reverse_anchor)

    def test_an_empty_segment_list_is_not_a_train(self):
        w = ts.get_master_wagon_window([], verbose=False)
        self.assertFalse(w.found)
        self.assertEqual(w.master_wagon_count, 0)


# ===========================================================================
# 10. The support cameras cannot touch the region
# ===========================================================================

class TestSupportCamerasCannotMoveTheRegion(unittest.TestCase):

    def test_the_walk_reads_no_camera_at_all(self):
        """It takes master segments and master rejections; there is no parameter
        through which a support camera could reach it."""
        import inspect
        params = set(inspect.signature(ts.get_master_wagon_window)
                     .parameters) | set(inspect.signature(ts._merge_evidence)
                                        .parameters)
        for banned in ("camera", "camera_id", "support", "regions",
                       "wagon_regions", "top_classifications"):
            self.assertNotIn(banned, params)

    def test_left_up_top_is_still_only_a_support_camera(self):
        from core import constants as C
        self.assertIn(C.CAMERA_LEFT_UP_TOP, C.TOP_CAMERAS)
        self.assertEqual(C.MASTER_CAMERA, C.CAMERA_RIGHT_UP)
        self.assertNotEqual(C.CAMERA_LEFT_UP_TOP, C.MASTER_CAMERA)

    def test_the_module_never_mentions_a_support_camera_in_the_walk(self):
        import ast
        src = open(os.path.join(V4_ROOT, "wagon_count/train_structure.py"),
                   encoding="utf-8").read()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "get_master_wagon_window")
        body = ast.dump(fn)
        for banned in ("LEFT_UP_TOP", "RIGHT_UP_TOP", "CAMERA_LEFT_UP",
                       "support_regions", "LocalWagonRegion"):
            self.assertNotIn(banned, body)

    def test_both_pipelines_pass_the_same_evidence_kind(self):
        """Batch reads the in-memory validation result, sequential reads the
        JSON that same pass wrote. Both must land on `RejectedGapSpan`."""
        import ast
        for path, fname in (
            ("wagon_count/run_global_count.py", "rejected_gap_spans_from_validation"),
            ("orchestrator/global_assembler.py", "rejected_gap_spans_from_json"),
        ):
            src = open(os.path.join(V4_ROOT, path), encoding="utf-8").read()
            names = [n.func.attr for n in ast.walk(ast.parse(src))
                     if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute)]
            names += [n.func.id for n in ast.walk(ast.parse(src))
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            self.assertIn(fname, names, path)

    def test_the_two_adapters_agree_on_the_same_evidence(self):
        """Same rejection, in memory and as JSON, must produce the same span --
        this is what keeps the canonical timeline identical across modes."""
        class _F:
            frame_start, frame_end, track_id = 140, 160, 4

        class _R:
            reason, features, is_soft = SOFT, _F(), True

        class _Res:
            rejected = [_R()]

        from_mem = ts.rejected_gap_spans_from_validation(_Res())
        from_json = ts.rejected_gap_spans_from_json({"rejections": [{
            "reason": SOFT, "soft": True,
            "features": {"frame_start": 140, "frame_end": 160, "track_id": 4}}]})
        self.assertEqual([s.to_dict() for s in from_mem],
                         [s.to_dict() for s in from_json])

    def test_a_malformed_rejection_is_skipped_not_guessed(self):
        self.assertEqual(ts.rejected_gap_spans_from_json(
            {"rejections": [{"reason": SOFT, "features": {}}]}), [])
        self.assertEqual(ts.rejected_gap_spans_from_json(None), [])
        self.assertEqual(ts.rejected_gap_spans_from_validation(None), [])


# ===========================================================================
# 11. Stage-3 sampling is untouched
# ===========================================================================

class TestStage3SamplingUnchanged(unittest.TestCase):

    def test_the_strides_are_the_production_ones(self):
        from core import constants as C
        self.assertEqual((C.STAGE3_DOOR_MODE, C.STAGE3_DOOR_STRIDE),
                         ("sampled", 3))
        self.assertEqual((C.STAGE3_DAMAGE_MODE, C.STAGE3_DAMAGE_STRIDE),
                         ("sampled", 3))
        self.assertEqual((C.STAGE3_LOAD_MODE, C.STAGE3_LOAD_STRIDE),
                         ("sampled", 2))

    def test_the_walk_mentions_no_stage3_setting(self):
        src = open(os.path.join(V4_ROOT, "wagon_count/train_structure.py"),
                   encoding="utf-8").read()
        for banned in ("STAGE3_", "sample_stride", "inference_mode",
                       "every_nth"):
            self.assertNotIn(banned, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ===========================================================================
# 12. The processed video follows the newly resolved boundary
# ===========================================================================

class TestProcessedVideoUsesTheResolvedRegion(unittest.TestCase):
    """The canonical region is what the video's ACTIVE-REGION marker draws.

    RIGHT_UP is the master and is deliberately excluded from `support_regions`
    ("master has no support region"), so the renderer projects
    `state.wagon_window` at offset 0 -- which is now the end-anchored boundary.
    These tests prove the marker actually moves when the anchor moves, rather
    than trusting that it does.
    """

    #: The HUD fixture's own scale (`tests/test_processed_video_hud`).
    VID_FPS = 10.0
    VID_FRAMES = 60

    def _state_from(self, win):
        """A core `GlobalTrainState` carrying this window, at the video's fps."""
        from core.global_state_loader import parse_global_train_state
        from core import constants as C
        # Compress the fixture's master frames onto the short test video.
        scale = self.VID_FRAMES / float(
            max(1, win.wagon_units[-1].end_frame_master + 1))
        wagons, sf_first, ef_last = [], None, None
        for n, u in enumerate(win.wagon_units, start=1):
            sf = int(round(u.start_frame_master * scale))
            ef = int(round((u.end_frame_master + 1) * scale)) - 1
            if sf_first is None:
                sf_first = sf
            ef_last = ef
            wagons.append({
                "global_id": f"GW_{n}", "wagon_index": n,
                "start_frame_master": sf, "end_frame_master": ef,
                "start_time": sf / self.VID_FPS,
                "end_time": (ef + 1) / self.VID_FPS,
                "classification": C.CLASS_WAGON,
                "classification_confidence": 0.9,
                "supporting_cameras": list(C.ALL_CAMERAS)})
        return parse_global_train_state({
            "total_wagons": len(wagons), "master_camera": C.CAMERA_RIGHT_UP,
            "master_fps": self.VID_FPS, "master_total_frames": self.VID_FRAMES,
            "wagons": wagons,
            "wagon_window": {
                "found": True,
                "wagon_start_frame": sf_first, "wagon_end_frame": ef_last,
                "wagon_start_time": sf_first / self.VID_FPS,
                "wagon_end_time": (ef_last + 1) / self.VID_FPS,
            },
        }), sf_first

    def _rendered_region(self, win):
        import tempfile
        from core import constants as C
        import test_processed_video_hud as HUD
        state, expected_start = self._state_from(win)
        with tempfile.TemporaryDirectory() as tmp:
            HUD._render(C.CAMERA_RIGHT_UP, tmp, state=state,
                        tracking={C.CAMERA_RIGHT_UP: {
                            "fps": self.VID_FPS,
                            "total_frames": self.VID_FRAMES, "gaps": []}})
        audit = HUD.R.RENDER_AUDITS[C.CAMERA_RIGHT_UP]["active_region"]
        return audit, expected_start

    def test_the_marker_uses_the_canonical_window(self):
        audit, expected = self._rendered_region(window(NORMAL))
        self.assertEqual(audit["source"], "master_window_projected")
        self.assertEqual(audit["start"], expected)

    def test_the_marker_moves_when_the_reverse_anchor_extends_the_region(self):
        """The whole point: recovering the first wagon has to be visible in the
        video, not just in the JSON."""
        fwd_audit, fwd_start = self._rendered_region(window(MERGED))
        rev_audit, rev_start = self._rendered_region(
            window(MERGED, [soft_gap_at(MERGED_COUPLING)]))
        self.assertLess(rev_start, fwd_start,
                        "the recovered wagon did not move the region start")
        self.assertEqual(rev_audit["start"], rev_start)
        self.assertEqual(fwd_audit["start"], fwd_start)

    def test_a_region_edge_is_still_not_a_physical_gap(self):
        audit, _ = self._rendered_region(window(NORMAL))
        self.assertFalse(audit["is_physical_wagon_gap"])


# ===========================================================================
# 13. Sequential and batch derive the same canonical timeline
# ===========================================================================

class TestBothModesDeriveTheSameRegion(unittest.TestCase):
    """Batch holds the master's validation result in memory; sequential reads the
    `gap_validation.json` that same pass wrote. Equivalent input has to give an
    identical canonical timeline, or the two modes would disagree about how many
    wagons a train has."""

    class _F:
        def __init__(self, s, e, t=7):
            self.frame_start, self.frame_end, self.track_id = s, e, t

    class _R:
        def __init__(self, f, reason=SOFT, soft=True):
            self.features, self.reason, self.is_soft = f, reason, soft

    class _Res:
        def __init__(self, rejected):
            self.rejected = rejected

    def _both(self, center):
        """The same rejection, reached the way each mode reaches it."""
        batch = ts.rejected_gap_spans_from_validation(
            self._Res([self._R(self._F(center - 5, center + 5))]))
        sequential = ts.rejected_gap_spans_from_json({"rejections": [{
            "reason": SOFT, "soft": True, "hard": False,
            "features": {"frame_start": center - 5, "frame_end": center + 5,
                         "track_id": 7}}]})
        return batch, sequential

    def test_the_windows_are_identical(self):
        batch, sequential = self._both(MERGED_COUPLING)
        wb = window(MERGED, batch)
        ws = window(MERGED, sequential)
        self.assertEqual(wb.summary(), ws.summary())

    def test_both_recover_the_same_first_wagon(self):
        batch, sequential = self._both(MERGED_COUPLING)
        for w in (window(MERGED, batch), window(MERGED, sequential)):
            self.assertEqual(w.master_wagon_count, 5)
            self.assertEqual(w.wagon_start_frame, 0)
            self.assertFalse(w.reverse_anchor.boundaries_agree)

    def test_both_agree_when_there_is_nothing_to_recover(self):
        batch, sequential = self._both(MERGED_COUPLING)
        self.assertEqual(window(NORMAL, batch).summary(),
                         window(NORMAL, sequential).summary())

    def test_a_mode_with_no_evidence_falls_back_not_forward(self):
        """If one mode somehow loses the diagnostic it must degrade to the
        forward boundary -- a safe, explainable answer -- never to a guess."""
        w = window(MERGED, [])
        self.assertTrue(w.reverse_anchor.boundaries_agree)
        self.assertEqual(w.reverse_anchor.first_wagon_segment_index,
                         w.reverse_anchor.forward_first_wagon_segment_index)
