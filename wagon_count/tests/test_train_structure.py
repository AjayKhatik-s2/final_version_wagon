"""Unit tests for wagon-only counting and the top classification model.

The hard rules under test:

    ENGINE is not a wagon.  BRAKE_VAN is not a wagon.
    Neither ever receives a GW id, and neither extends the wagon timeline.
    GLOBAL WAGON COUNT == validated RIGHT_UP WAGON count.
    Support cameras are evidence only.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import global_fusion as gf
import train_structure as ts
from global_train_state import (
    ALL_CAMERAS, CAMERA_LEFT_UP, CAMERA_LEFT_UP_TOP, CAMERA_RIGHT_UP,
    CAMERA_RIGHT_UP_TOP, GapEvent, GlobalWagon, LocalCameraTracks, SegmentClass,
    _MasterClassification,
)

FPS = 15.0
GAP_SPAN = 12


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def make_tracks(camera_id, gap_times, duration_s=400.0, fps=FPS):
    gaps = []
    for i, t in enumerate(sorted(gap_times), start=1):
        c = t * fps
        s, e = int(round(c - GAP_SPAN / 2)), int(round(c + GAP_SPAN / 2))
        xs = [600.0 - 35.0 * k for k in range(GAP_SPAN + 1)]
        gaps.append(GapEvent(
            track_id=i, camera_id=camera_id, start_frame=max(0, s),
            end_frame=max(1, e), confidence=0.9, hit_count=GAP_SPAN + 1,
            center_x_trajectory=xs, fps=fps, temporal_consistency_score=1.0,
            hit_frames=list(range(max(0, s), max(1, e) + 1)),
            bbox_history=[[x - 20, 100, x + 20, 300] for x in xs]))
    return LocalCameraTracks(
        camera_id=camera_id, video_path=f"/synthetic/{camera_id}.mp4", fps=fps,
        total_frames=int(round(duration_s * fps)), width=848, height=480, gaps=gaps)


def segments_and_labels(labels, spacing=4.0, start=10.0, fps=FPS,
                        duration_s=400.0):
    """Build a master with len(labels) segments carrying the given labels.

    n labels -> n-1 internal gaps, plus the leading/trailing video edges.
    """
    n = len(labels)
    gap_times = [start + i * spacing for i in range(1, n)]
    master = make_tracks(CAMERA_RIGHT_UP, gap_times, duration_s=duration_s, fps=fps)

    # Boundaries the way build_global_wagons computes them.
    bounds = [int(round(g.center_frame)) for g in master.gaps]
    segs, prev = [], 0
    for b in sorted(bounds):
        if b <= prev:
            continue
        segs.append((prev, b - 1)); prev = b
    if prev <= master.total_frames - 1:
        segs.append((prev, master.total_frames - 1))

    assert len(segs) == n, f"expected {n} segments, built {len(segs)}"
    cls = [_MasterClassification(i, s, e, labels[i], 1.0)
           for i, (s, e) in enumerate(segs)]
    return master, cls


def assemble(master, supports, cls, wagon_only=True):
    return gf.assemble_global_train_state_master_fixed(
        master_tracks=master, support_tracks=supports,
        initial_classifications=cls, config=gf.FusionConfig(),
        verbose=False, wagon_only=wagon_only)


E, W, B, U = (SegmentClass.ENGINE, SegmentClass.WAGON,
              SegmentClass.BRAKE_VAN, SegmentClass.UNKNOWN)


# ===========================================================================
# engine / brake-van exclusion
# ===========================================================================

class TestWagonOnlyCounting(unittest.TestCase):
    def test_engine_plus_five_wagons(self):
        master, cls = segments_and_labels([E, W, W, W, W, W])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 5)
        self.assertEqual([w.global_id for w in st.wagons],
                         ["GW_1", "GW_2", "GW_3", "GW_4", "GW_5"])
        self.assertTrue(all(w.classification == W for w in st.wagons))

    def test_five_wagons_plus_brake_van(self):
        master, cls = segments_and_labels([W, W, W, W, W, B])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 5)
        self.assertEqual(st.wagons[-1].global_id, "GW_5")

    def test_engine_five_wagons_brake_van(self):
        master, cls = segments_and_labels([E, W, W, W, W, W, B])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 5)
        self.assertEqual([w.global_id for w in st.wagons],
                         ["GW_1", "GW_2", "GW_3", "GW_4", "GW_5"])

    def test_three_engines_then_wagons_then_brake_van(self):
        master, cls = segments_and_labels([E, E, E, W, W, W, W, B])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 4)
        ww = st.wagon_window
        self.assertEqual(ww["leading_non_wagon_count"], 3)
        self.assertEqual(ww["trailing_non_wagon_count"], 1)
        self.assertEqual(ww["leading_non_wagon_classes"], {E: 3})
        self.assertEqual(ww["trailing_non_wagon_classes"], {B: 1})

    def test_the_worked_example_from_the_brief(self):
        """ENGINE WAGON WAGON WAGON BRAKE_VAN -> GW_1 GW_2 GW_3, not five ids."""
        master, cls = segments_and_labels([E, W, W, W, B])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 3)
        self.assertEqual([w.global_id for w in st.wagons], ["GW_1", "GW_2", "GW_3"])

    def test_no_engine_or_brakevan_ever_holds_a_gw_id(self):
        master, cls = segments_and_labels([E, E, W, W, B, B])
        st = assemble(master, [], cls)
        for w in st.wagons:
            self.assertNotIn(w.classification, (E, B))

    def test_interior_engine_anomaly_does_not_delete_a_wagon(self):
        """An interior ENGINE label must NOT remove a master wagon.

        The segment sits between two validated RIGHT_UP master gaps, so it is a
        wagon boundary region by construction. An engine label there is a
        classification anomaly; letting it delete a wagon would put
        classification in control of an individual wagon, which it must never be.
        """
        master, cls = segments_and_labels([E, W, W, E, W, W, B])
        st = assemble(master, [], cls)
        # 5 interior segments (indices 1..5), all counted
        self.assertEqual(st.total_wagons, 5)
        self.assertEqual([w.global_id for w in st.wagons],
                         ["GW_1", "GW_2", "GW_3", "GW_4", "GW_5"])
        # ...and the anomaly is reported, not hidden
        self.assertEqual(st.wagon_window["interior_classification_anomalies"], 1)
        self.assertTrue(st.wagon_window["interior_anomalies_are_still_counted"])
        self.assertTrue(st.invariant_checks["invariant_holds"])
        self.assertEqual(
            st.invariant_checks["interior_classification_anomalies_counted"], 1)

    def test_interior_brakevan_anomaly_does_not_delete_a_wagon(self):
        master, cls = segments_and_labels([E, W, W, B, W, W, B])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 5)
        self.assertEqual(st.wagon_window["interior_classification_anomalies"], 1)
        self.assertTrue(st.invariant_checks["invariant_holds"])

    def test_gw_ids_are_not_renumbered_by_an_interior_anomaly(self):
        """The same master gaps must yield the same ids with or without the
        interior anomaly."""
        clean, cls_clean = segments_and_labels([E, W, W, W, W, W, B])
        anom, cls_anom = segments_and_labels([E, W, W, E, W, W, B])
        a = assemble(clean, [], cls_clean)
        b = assemble(anom, [], cls_anom)
        self.assertEqual(a.total_wagons, b.total_wagons)
        self.assertEqual([w.global_id for w in a.wagons],
                         [w.global_id for w in b.wagons])
        self.assertEqual([w.start_frame_master for w in a.wagons],
                         [w.start_frame_master for w in b.wagons])

    def test_unknown_inside_the_window_is_counted_and_reported(self):
        """An unlabelled vehicle between two wagons is physically a wagon."""
        master, cls = segments_and_labels([E, W, U, W, B])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 3)
        self.assertIn(U, [w.classification for w in st.wagons])

    def test_unknown_outside_the_window_is_not_counted(self):
        master, cls = segments_and_labels([U, E, W, W, B, U])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 2)

    def test_no_wagons_at_all_yields_zero(self):
        master, cls = segments_and_labels([E, E, B])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 0)
        self.assertEqual(st.wagons, [])
        self.assertFalse(st.wagon_window["found"])

    def test_wagon_only_can_be_disabled_for_ab_comparison(self):
        master, cls = segments_and_labels([E, W, W, W, B])
        st = assemble(master, [], cls, wagon_only=False)
        self.assertEqual(st.total_wagons, 5, "legacy behaviour counts every segment")


# ===========================================================================
# the wagon window itself
# ===========================================================================

class TestWagonWindow(unittest.TestCase):
    def _segments(self, labels):
        out = []
        for i, lb in enumerate(labels):
            out.append(GlobalWagon(
                global_id=f"SEG_{i}", wagon_index=i,
                start_frame_master=i * 100, end_frame_master=i * 100 + 99,
                start_time=i * 100 / FPS, end_time=(i * 100 + 100) / FPS,
                classification=lb, classification_confidence=1.0))
        return out

    def test_first_and_last_wagon_bound_the_window(self):
        win = ts.get_master_wagon_window(self._segments([E, E, W, W, W, B]),
                                        verbose=False)
        self.assertTrue(win.found)
        self.assertEqual(win.first_wagon_segment_index, 2)
        self.assertEqual(win.last_wagon_segment_index, 4)
        self.assertEqual(win.master_wagon_count, 3)
        self.assertEqual(win.wagon_start_frame, 200)
        self.assertEqual(win.wagon_end_frame, 499)

    def test_temporal_order_and_timestamps_are_preserved(self):
        """Non-wagon frames are excluded from counting, never deleted or shifted."""
        segs = self._segments([E, W, W, B])
        win = ts.get_master_wagon_window(segs, verbose=False)
        self.assertEqual(win.wagon_units[0].start_frame_master, 100)
        self.assertEqual(win.wagon_units[0].start_time, 100 / FPS)
        self.assertEqual(win.leading_non_wagon_objects[0].start_frame, 0)
        self.assertEqual(win.trailing_non_wagon_objects[0].start_frame, 300)
        # the master frame numbers are untouched by renumbering
        self.assertEqual([w.global_id for w in win.wagon_units], ["GW_1", "GW_2"])

    def test_every_segment_is_accounted_for(self):
        for labels in ([E, W, B], [E, E, W, W, W, B], [W], [E, W, E, W, B],
                       [U, W, U], [E, B]):
            win = ts.get_master_wagon_window(self._segments(labels), verbose=False)
            # Interior anomalies are part of wagon_units now, so only
            # leading/trailing are outside the count.
            total = (win.master_wagon_count
                     + len(win.leading_non_wagon_objects)
                     + len(win.trailing_non_wagon_objects))
            self.assertEqual(total, len(labels), f"labels={labels}")

    def test_engine_and_brakevan_metadata_is_preserved(self):
        win = ts.get_master_wagon_window(self._segments([E, W, W, B]), verbose=False)
        lead = win.leading_non_wagon_objects[0]
        self.assertEqual(lead.classification, E)
        self.assertEqual(lead.position, "leading")
        self.assertIn("classification", lead.to_dict())
        trail = win.trailing_non_wagon_objects[0]
        self.assertEqual(trail.classification, B)
        self.assertEqual(trail.position, "trailing")

    def test_empty_input(self):
        win = ts.get_master_wagon_window([], verbose=False)
        self.assertFalse(win.found)
        self.assertEqual(win.master_wagon_count, 0)


# ===========================================================================
# support cameras cannot change the count
# ===========================================================================

class TestSupportCannotInflate(unittest.TestCase):
    def test_engine_and_brakevan_on_all_cameras_do_not_inflate(self):
        master, cls = segments_and_labels([E, W, W, W, W, B])
        supports = [make_tracks(c, [t / FPS for t in
                                    [g.center_frame for g in master.gaps]])
                    for c in ALL_CAMERAS if c != CAMERA_RIGHT_UP]
        st = assemble(master, supports, cls)
        self.assertEqual(st.total_wagons, 4)

    def test_support_extra_observations_cannot_increase_the_count(self):
        master, cls = segments_and_labels([E, W, W, W, B])
        base = [g.center_time for g in master.gaps]
        noisy = sorted(base + [b + 1.7 for b in base] + [b + 2.4 for b in base])
        supports = [make_tracks(CAMERA_LEFT_UP, noisy)]
        st = assemble(master, supports, cls)
        self.assertEqual(st.total_wagons, 3)
        self.assertEqual(st.corrections_applied, [])

    def test_support_missing_observations_do_not_change_the_count(self):
        master, cls = segments_and_labels([E, W, W, W, W, W, B])
        supports = [make_tracks(CAMERA_LEFT_UP, [master.gaps[0].center_time])]
        st = assemble(master, supports, cls)
        self.assertEqual(st.total_wagons, 5)

    def test_duplicate_support_detections_cannot_create_ids(self):
        master, cls = segments_and_labels([E, W, W, B])
        base = [g.center_time for g in master.gaps]
        supports = [make_tracks(CAMERA_LEFT_UP_TOP,
                                sorted(base + [b + 0.2 for b in base]))]
        st = assemble(master, supports, cls)
        self.assertEqual(st.total_wagons, 2)
        self.assertEqual([w.global_id for w in st.wagons], ["GW_1", "GW_2"])

    def test_count_equals_master_wagon_count(self):
        master, cls = segments_and_labels([E, E, W, W, W, W, W, W, B])
        supports = [make_tracks(c, [g.center_time for g in master.gaps][:2])
                    for c in ALL_CAMERAS if c != CAMERA_RIGHT_UP]
        st = assemble(master, supports, cls)
        self.assertEqual(st.total_wagons, 6)
        self.assertEqual(st.master_wagon_count, 6)
        self.assertEqual(st.invariant_checks["master_wagon_count"], 6)
        self.assertTrue(st.invariant_checks["invariant_holds"])

    def test_region_filtering_cannot_unresolve_a_good_offset(self):
        """Regression: wagon-region filtering used to remove a camera's evidence.

        The offset was estimated from the FILTERED observations, so dropping a
        few observations shrank the score margin below threshold and marked a
        demonstrably correct offset UNRESOLVED -- which silently blanked that
        camera on every PDF wagon page. A camera's clock offset is a property of
        the camera, not of which observations we chose to align, so it is now
        estimated BEFORE filtering.
        """
        master, cls = segments_and_labels([E] + [W] * 23)
        mtimes = [g.center_time for g in master.gaps]
        delta = 5.0
        sup = make_tracks(CAMERA_RIGHT_UP_TOP, [t - delta for t in mtimes])

        gaps = gf.build_global_gap_sequence(master)
        # Region excludes the first few observations, as classification does.
        region = ts.LocalWagonRegion(
            camera_id=CAMERA_RIGHT_UP_TOP, found=True,
            start_time=mtimes[3] - delta - 0.05, end_time=mtimes[-1] - delta + 5.0)

        al = gf.attach_support_evidence(
            gaps, [sup], gf.FusionConfig(), verbose=False,
            wagon_regions={CAMERA_RIGHT_UP_TOP: region})[CAMERA_RIGHT_UP_TOP]

        self.assertEqual(al.offset.status, gf.OFFSET_RESOLVED,
                         f"filtering must not unresolve a good offset: "
                         f"{al.offset.reason}")
        self.assertAlmostEqual(al.offset.delta, delta, delta=0.35)
        self.assertGreater(len(al.matches), 0, "evidence must be attached")
        n_with = sum(1 for g in gaps
                     if CAMERA_RIGHT_UP_TOP in g.support_observations)
        self.assertGreater(n_with, 0, "wagon pages must carry this camera")
        # and the count is untouched by any of it
        self.assertEqual(len(gaps), len(master.gaps))

    def test_offset_is_estimated_on_the_unfiltered_observation_set(self):
        """The offset must be identical with and without region filtering."""
        master, cls = segments_and_labels([E] + [W] * 19)
        mtimes = [g.center_time for g in master.gaps]
        sup = make_tracks(CAMERA_LEFT_UP, [t - 4.0 for t in mtimes])
        gaps_a = gf.build_global_gap_sequence(master)
        gaps_b = gf.build_global_gap_sequence(master)

        no_filter = gf.attach_support_evidence(
            gaps_a, [sup], gf.FusionConfig(), verbose=False)[CAMERA_LEFT_UP]
        region = ts.LocalWagonRegion(
            camera_id=CAMERA_LEFT_UP, found=True,
            start_time=mtimes[2] - 4.0 - 0.05, end_time=mtimes[-1] - 4.0 + 5.0)
        with_filter = gf.attach_support_evidence(
            gaps_b, [sup], gf.FusionConfig(), verbose=False,
            wagon_regions={CAMERA_LEFT_UP: region})[CAMERA_LEFT_UP]

        self.assertAlmostEqual(no_filter.offset.delta, with_filter.offset.delta,
                               places=6, msg="offset must not depend on filtering")
        self.assertEqual(no_filter.offset.margin_ratio,
                         with_filter.offset.margin_ratio,
                         "margin must not depend on filtering either")
        self.assertGreater(with_filter.excluded_observation_count, 0,
                           "the filter must actually have excluded something")

    def test_observation_bookkeeping_balances_after_region_filtering(self):
        """Regression: the EC2 run reported a false invariant violation because
        MATCH+EXTRA was compared against the RAW observation count instead of the
        post-filter count actually given to the DP.

        Reproduces the shape of that run: a support camera with the same number
        of observations as the master, two of which fall outside its wagon
        region, so 2 fewer reach alignment.

            raw = aligned + excluded
            aligned = MATCH + EXTRA
        """
        n = 20
        master, cls = segments_and_labels([E] + [W] * (n - 1))
        base = [g.center_time for g in master.gaps]
        sup = make_tracks(CAMERA_RIGHT_UP_TOP, base)
        # Region starts after the first two observations -> exactly 2 excluded.
        region = ts.LocalWagonRegion(
            camera_id=CAMERA_RIGHT_UP_TOP, found=True,
            start_time=base[2] - 0.05, end_time=base[-1] + 5.0)

        st = gf.assemble_global_train_state_master_fixed(
            master_tracks=master, support_tracks=[sup],
            initial_classifications=cls, config=gf.FusionConfig(), verbose=False,
            wagon_regions={CAMERA_RIGHT_UP_TOP: region})

        # The invariant must HOLD -- this is the bug being regression-tested.
        self.assertTrue(st.invariant_checks["invariant_holds"],
                        st.invariant_checks["violations"])

        s = st.support_alignment_summary[CAMERA_RIGHT_UP_TOP]
        self.assertEqual(s["raw_observations"], len(sup.gaps))
        self.assertEqual(s["excluded_outside_wagon_region"], 2)
        self.assertEqual(s["aligned_observations"], len(sup.gaps) - 2)
        self.assertEqual(s["MATCH"] + s["EXTRA"], s["aligned_observations"],
                         "MATCH + EXTRA must equal the ALIGNED count")
        self.assertEqual(s["aligned_observations"]
                         + s["excluded_outside_wagon_region"],
                         s["raw_observations"])
        self.assertTrue(s["bookkeeping_balances"])
        # and the master invariant is untouched by any of it
        self.assertEqual(st.invariant_checks["global_gap_count"], len(master.gaps))

    def test_bookkeeping_balances_with_no_filtering(self):
        """With no region filter, excluded is 0 and raw == aligned."""
        master, cls = segments_and_labels([E, W, W, W, W, B])
        sup = make_tracks(CAMERA_LEFT_UP, [g.center_time for g in master.gaps])
        st = assemble(master, [sup], cls)
        s = st.support_alignment_summary[CAMERA_LEFT_UP]
        self.assertEqual(s["excluded_outside_wagon_region"], 0)
        self.assertEqual(s["raw_observations"], s["aligned_observations"])
        self.assertEqual(s["MATCH"] + s["EXTRA"], s["aligned_observations"])
        self.assertTrue(st.invariant_checks["invariant_holds"])

    def test_bookkeeping_balances_when_offset_unresolved(self):
        """An unresolved camera still balances: every aligned obs is EXTRA."""
        master, cls = segments_and_labels([E, W, W, W, W, W, W, B])
        base = [g.center_time for g in master.gaps]
        sup = make_tracks(CAMERA_LEFT_UP_TOP, base)
        region = ts.LocalWagonRegion(
            camera_id=CAMERA_LEFT_UP_TOP, found=True,
            start_time=base[1] - 0.05, end_time=base[-1] + 5.0)
        st = gf.assemble_global_train_state_master_fixed(
            master_tracks=master, support_tracks=[sup],
            initial_classifications=cls,
            config=gf.FusionConfig(offset_min_match_fraction=1.5),  # force UNRESOLVED
            verbose=False, wagon_regions={CAMERA_LEFT_UP_TOP: region})
        s = st.support_alignment_summary[CAMERA_LEFT_UP_TOP]
        self.assertEqual(s["offset"]["status"], gf.OFFSET_UNRESOLVED)
        self.assertEqual(s["excluded_outside_wagon_region"], 1)
        self.assertEqual(s["MATCH"], 0)
        self.assertEqual(s["MATCH"] + s["EXTRA"], s["aligned_observations"])
        self.assertTrue(s["bookkeeping_balances"])
        self.assertTrue(st.invariant_checks["invariant_holds"])

    def test_a_genuine_accounting_error_is_still_caught(self):
        """The invariant must not have been weakened: corrupting the alignment
        bookkeeping still fails loudly."""
        master, cls = segments_and_labels([E, W, W, W, B])
        sup = make_tracks(CAMERA_LEFT_UP, [g.center_time for g in master.gaps])
        gaps = gf.build_global_gap_sequence(master)
        aligns = gf.attach_support_evidence(gaps, [sup], gf.FusionConfig(),
                                           verbose=False)
        al = aligns[CAMERA_LEFT_UP]
        al.extra_observations.append(al.matches[list(al.matches)[0]])  # phantom EXTRA
        with self.assertRaises(gf.FusionInvariantError):
            gf.assert_invariants(
                global_gaps=gaps, master_tracks=master, wagons=[],
                alignments=aligns, support_tracks=[sup], strict=True)

    def test_support_observations_outside_the_wagon_region_are_excluded(self):
        master, cls = segments_and_labels([E, W, W, W, B])
        base = [g.center_time for g in master.gaps]
        sup = make_tracks(CAMERA_RIGHT_UP_TOP, base)
        region = ts.LocalWagonRegion(
            camera_id=CAMERA_RIGHT_UP_TOP, found=True,
            start_time=base[1] - 0.1, end_time=base[-1] + 0.1)
        st = gf.assemble_global_train_state_master_fixed(
            master_tracks=master, support_tracks=[sup],
            initial_classifications=cls, config=gf.FusionConfig(), verbose=False,
            wagon_regions={CAMERA_RIGHT_UP_TOP: region})
        self.assertEqual(st.total_wagons, 3, "region filtering must not alter the count")
        summary = st.support_alignment_summary[CAMERA_RIGHT_UP_TOP]
        self.assertGreaterEqual(summary["n_non_wagon_excluded"], 1)


# ===========================================================================
# top classification model
# ===========================================================================

class TestTopClassificationMapping(unittest.TestCase):
    def test_camera_to_classifier_mapping(self):
        m = ts.CAMERA_CLASSIFICATION_MODEL
        self.assertEqual(m[CAMERA_RIGHT_UP], ts.SIDE_CLASSIFICATION_MODEL)
        self.assertEqual(m[CAMERA_RIGHT_UP_TOP], ts.TOP_CLASSIFICATION_MODEL)
        self.assertEqual(m[CAMERA_LEFT_UP_TOP], ts.TOP_CLASSIFICATION_MODEL)
        self.assertEqual(m[CAMERA_LEFT_UP], ts.SIDE_CLASSIFICATION_MODEL)

    def test_mapping_is_built_from_real_names_not_indices(self):
        """Class IDs are never assumed: 0 is not 'wagon' by fiat."""
        lm = ts.build_label_mapping({0: "brakevan", 1: "engine", 2: "wagon"})
        self.assertEqual(lm.semantic_for("wagon"), W)
        self.assertEqual(lm.semantic_for("engine"), E)
        self.assertEqual(lm.semantic_for("brakevan"), B)
        # a differently ordered model must map identically
        lm2 = ts.build_label_mapping({0: "wagon", 1: "brakevan", 2: "engine"})
        self.assertEqual(lm.mapping, lm2.mapping)

    def test_unexpected_class_maps_to_unknown_never_wagon(self):
        lm = ts.build_label_mapping({0: "wagon", 1: "engine", 2: "sheep",
                                     3: "flying_saucer"})
        self.assertEqual(lm.semantic_for("sheep"), U)
        self.assertEqual(lm.semantic_for("flying_saucer"), U)
        self.assertNotEqual(lm.semantic_for("sheep"), W)
        self.assertEqual(sorted(lm.unmapped), ["flying_saucer", "sheep"])

    def test_background_style_classes_map_to_unknown(self):
        lm = ts.build_label_mapping({0: "empty_track", 1: "background",
                                     2: "other", 3: "unknown"})
        for name in ("empty_track", "background", "other", "unknown"):
            self.assertEqual(lm.semantic_for(name), U)
        self.assertEqual(lm.unmapped, [], "these are recognised, not unexpected")

    def test_brakevan_variants_are_not_read_as_wagons(self):
        lm = ts.build_label_mapping({0: "wagon_tail", 1: "guard_van",
                                     2: "brake-van", 3: "tail"})
        for name in ("wagon_tail", "guard_van", "brake-van", "tail"):
            self.assertEqual(lm.semantic_for(name), B, name)

    def test_engine_variants(self):
        lm = ts.build_label_mapping({0: "loco", 1: "locomotive", 2: "engine_head",
                                     3: "locono"})
        for name in lm.names.values():
            self.assertEqual(lm.semantic_for(name), E, name)

    def test_case_and_whitespace_insensitive(self):
        lm = ts.build_label_mapping({0: " Wagon ", 1: "ENGINE"})
        self.assertEqual(lm.semantic_for(" Wagon "), W)
        self.assertEqual(lm.semantic_for("engine"), E)

    def test_unknown_label_at_lookup_time_is_unknown(self):
        lm = ts.build_label_mapping({0: "wagon"})
        self.assertEqual(lm.semantic_for("something_else_entirely"), U)

    def test_mapping_serializes_for_the_json_report(self):
        lm = ts.build_label_mapping({0: "wagon", 1: "mystery"}, "models/top.pt")
        d = lm.to_dict()
        self.assertEqual(d["class_count"], 2)
        self.assertEqual(d["names"], {0: "wagon", 1: "mystery"})
        self.assertEqual(d["unmapped_classes"], ["mystery"])


class TestLocalWagonRegion(unittest.TestCase):
    def test_region_from_labels(self):
        segs = [(0, 99), (100, 199), (200, 299), (300, 399)]
        reg = ts.build_local_wagon_region(
            CAMERA_RIGHT_UP_TOP, segs, [E, W, W, B], FPS, verbose=False)
        self.assertTrue(reg.found)
        self.assertEqual(reg.start_frame, 100)
        self.assertEqual(reg.end_frame, 299)
        self.assertTrue(reg.contains_time(150 / FPS))
        self.assertFalse(reg.contains_time(50 / FPS))
        self.assertFalse(reg.contains_time(350 / FPS))

    def test_unknown_region_accepts_everything(self):
        """A missing classification must not silently discard evidence."""
        reg = ts.build_local_wagon_region(
            CAMERA_LEFT_UP_TOP, [(0, 99)], [E], FPS, verbose=False)
        self.assertFalse(reg.found)
        self.assertTrue(reg.contains_time(0.0))
        self.assertTrue(reg.contains_time(9999.0))

    def test_top_engine_and_brakevan_produce_no_gw_ids(self):
        """A top camera seeing an engine cannot mint a wagon id."""
        master, cls = segments_and_labels([E, W, W, B])
        sup = make_tracks(CAMERA_RIGHT_UP_TOP,
                          [g.center_time for g in master.gaps])
        region = ts.build_local_wagon_region(
            CAMERA_RIGHT_UP_TOP, [(0, 100), (101, 200)], [E, B], FPS, verbose=False)
        st = gf.assemble_global_train_state_master_fixed(
            master_tracks=master, support_tracks=[sup],
            initial_classifications=cls, config=gf.FusionConfig(), verbose=False,
            wagon_regions={CAMERA_RIGHT_UP_TOP: region})
        self.assertEqual(st.total_wagons, 2)
        self.assertEqual([w.global_id for w in st.wagons], ["GW_1", "GW_2"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
