"""Regression tests for segment-level temporal classification.

The bug being regressed: one low-confidence misclassification must not move a
train-structure boundary, while a genuine class change must still be detected.

Synthetic durations mirror the MEASURED real data (848x480 @ 15 fps):
  genuine regions : >= 1.33 s at confidence ~1.000
  noise bursts    : 0.33 s at confidence 0.605-0.645
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import temporal_classification as tc
import train_structure as ts
from global_train_state import SegmentClass, _MasterClassification

FPS = 15.0
E, W, B, U = (SegmentClass.ENGINE, SegmentClass.WAGON,
              SegmentClass.BRAKE_VAN, SegmentClass.UNKNOWN)


def obs(labels_durations_confs, fps=FPS):
    """Build SegmentObservations from (label, duration_s, confidence) triples."""
    out, frame = [], 0
    for i, (label, dur, conf) in enumerate(labels_durations_confs):
        n = max(1, int(round(dur * fps)))
        out.append(tc.SegmentObservation(
            segment_index=i, start_frame=frame, end_frame=frame + n - 1,
            fps=fps, raw_label=label, raw_confidence=conf))
        frame += n
    return out


def smooth(triples, cfg=None):
    return tc.smooth_class_sequence(obs(triples), camera_id="TEST",
                                    cfg=cfg or tc.TemporalClassificationConfig(),
                                    verbose=False)


def sample(semantic, conf):
    return tc.ClassSample(frame=0, time=0.0, raw_label=semantic.lower(),
                          semantic=semantic, confidence=conf)


# ===========================================================================
# 1-2. a single bad observation must not flip a stable class
# ===========================================================================

class TestNoiseRejection(unittest.TestCase):
    def test_stable_wagon_survives_one_brakevan_burst(self):
        """W W B W W  ->  all WAGON."""
        r = smooth([(W, 4.0, 1.0), (W, 4.0, 1.0), (B, 0.33, 0.62),
                    (W, 4.0, 1.0), (W, 4.0, 1.0)])
        self.assertEqual(r.stable_labels, [W, W, W, W, W])
        self.assertEqual(r.n_changed, 1)
        rej = [t for t in r.transitions if not t.accepted]
        self.assertTrue(rej)
        self.assertEqual(rej[0].to_class, B)

    def test_stable_brakevan_survives_one_wagon_burst(self):
        """Symmetry: the same protection in the other direction."""
        r = smooth([(B, 4.0, 1.0), (B, 4.0, 1.0), (W, 0.33, 0.62),
                    (B, 4.0, 1.0), (B, 4.0, 1.0)])
        self.assertEqual(r.stable_labels, [B, B, B, B, B])

    def test_engine_burst_inside_wagon_region_is_rejected(self):
        """The measured RIGHT_UP case: ENGINE 0.33 s @ 0.605."""
        r = smooth([(W, 4.0, 1.0), (E, 0.33, 0.605), (W, 4.0, 1.0)])
        self.assertEqual(r.stable_labels, [W, W, W])

    def test_measured_top_camera_brakevan_burst_is_rejected(self):
        """The measured RIGHT_UP_TOP case: BRAKE_VAN 0.33 s @ 0.645 mid-train."""
        r = smooth([(W, 63.0, 1.0), (B, 0.33, 0.645), (W, 188.0, 1.0)])
        self.assertEqual(r.stable_labels, [W, W, W])
        self.assertTrue(r.observations[1].held_by_hysteresis)

    def test_held_observation_keeps_its_raw_label_for_audit(self):
        r = smooth([(W, 4.0, 1.0), (B, 0.33, 0.62), (W, 4.0, 1.0)])
        self.assertEqual(r.observations[1].raw_label, B, "raw label is preserved")
        self.assertEqual(r.observations[1].stable_label, W)
        self.assertTrue(r.observations[1].to_dict()["changed_by_smoothing"])


# ===========================================================================
# 3-4. genuine transitions must still be accepted, in both directions
# ===========================================================================

class TestGenuineTransitions(unittest.TestCase):
    def test_genuine_brakevan_to_wagon_is_accepted(self):
        r = smooth([(B, 4.0, 1.0), (W, 4.0, 1.0), (W, 4.0, 1.0)])
        self.assertEqual(r.stable_labels, [B, W, W])
        acc = [t for t in r.transitions if t.accepted]
        self.assertEqual([(t.from_class, t.to_class) for t in acc], [(B, W)])

    def test_genuine_wagon_to_brakevan_is_accepted(self):
        r = smooth([(W, 4.0, 1.0), (W, 4.0, 1.0), (B, 4.0, 1.0)])
        self.assertEqual(r.stable_labels, [W, W, B])

    def test_genuine_engine_to_wagon_is_accepted(self):
        r = smooth([(E, 8.0, 1.0), (W, 4.0, 1.0), (W, 4.0, 1.0)])
        self.assertEqual(r.stable_labels, [E, W, W])

    def test_genuine_wagon_to_engine_is_accepted(self):
        r = smooth([(W, 8.0, 1.0), (E, 4.0, 1.0), (E, 4.0, 1.0)])
        self.assertEqual(r.stable_labels, [W, E, E])

    def test_the_real_measured_structure_is_preserved(self):
        """ENGINE -> BRAKE_VAN(3.67s) -> WAGON, exactly as measured on RIGHT_UP.

        The single-segment brake van MUST survive: it is a real vehicle, and a
        segment-count-only rule would have deleted it.
        """
        r = smooth([(U, 2.0, 1.0), (E, 7.67, 1.0), (B, 3.67, 0.998),
                    (W, 261.0, 1.0), (U, 1.33, 1.0)])
        self.assertEqual(r.stable_labels, [U, E, B, W, U])
        self.assertEqual(r.n_changed, 0, "nothing genuine may be smoothed away")

    def test_one_long_confident_observation_switches_on_its_own(self):
        """Duration is the primary route: no 3-segment run required."""
        cfg = tc.TemporalClassificationConfig(switch_persistence=3)
        r = smooth([(W, 10.0, 1.0), (B, 2.0, 1.0), (W, 10.0, 1.0)], cfg)
        self.assertEqual(r.stable_labels[1], B,
                         "a 2.0 s confident region is a real vehicle")

    def test_short_run_switches_via_persistence_route(self):
        """Three consecutive short observations also qualify."""
        cfg = tc.TemporalClassificationConfig(min_stable_region_s=1.0,
                                              switch_persistence=3)
        r = smooth([(W, 10.0, 1.0), (B, 0.3, 0.95), (B, 0.3, 0.95),
                    (B, 0.3, 0.95), (W, 10.0, 1.0)], cfg)
        self.assertEqual(r.stable_labels[1:4], [B, B, B])

    def test_two_short_observations_do_not_switch(self):
        cfg = tc.TemporalClassificationConfig(min_stable_region_s=1.0,
                                              switch_persistence=3)
        r = smooth([(W, 10.0, 1.0), (B, 0.3, 0.95), (B, 0.3, 0.95),
                    (W, 10.0, 1.0)], cfg)
        self.assertEqual(r.stable_labels, [W, W, W, W])


# ===========================================================================
# 5. confidence-weighted evidence
# ===========================================================================

class TestConfidenceWeighting(unittest.TestCase):
    def test_within_segment_vote_prefers_confident_samples(self):
        """3 weak WAGON (0.62) vs 2 strong BRAKE_VAN (1.00): 1.86 < 2.00."""
        samples = [sample(W, 0.62), sample(W, 0.62), sample(W, 0.62),
                   sample(B, 1.0), sample(B, 1.0)]
        label, conf, scores = tc.aggregate_samples(samples)
        self.assertEqual(label, B)
        self.assertAlmostEqual(scores[W], 1.86, places=2)
        self.assertAlmostEqual(scores[B], 2.00, places=2)

    def test_plain_majority_would_have_chosen_differently(self):
        cfg = tc.TemporalClassificationConfig(use_confidence_weighted_vote=False)
        samples = [sample(W, 0.62), sample(W, 0.62), sample(W, 0.62),
                   sample(B, 1.0), sample(B, 1.0)]
        label, _c, _s = tc.aggregate_samples(samples, cfg)
        self.assertEqual(label, W, "unweighted majority favours the 3 weak votes")

    def test_low_confidence_cannot_challenge(self):
        cfg = tc.TemporalClassificationConfig(min_confidence_to_challenge=0.5)
        r = smooth([(W, 10.0, 1.0), (B, 4.0, 0.40), (W, 10.0, 1.0)], cfg)
        self.assertEqual(r.stable_labels, [W, W, W])
        rej = [t for t in r.transitions if not t.accepted]
        self.assertIn("min_confidence_to_challenge", rej[0].reason)

    def test_evidence_score_is_confidence_times_duration(self):
        o = obs([(B, 2.0, 0.5)])
        self.assertAlmostEqual(tc._evidence_score(o, B), 1.0, places=2)
        self.assertAlmostEqual(tc._evidence_score(o, W), 0.0, places=2)

    def test_empty_samples_are_unknown_not_wagon(self):
        label, conf, _ = tc.aggregate_samples([])
        self.assertEqual(label, U)
        self.assertNotEqual(label, W)


# ===========================================================================
# 6-7. boundary stability
# ===========================================================================

def _first_last_wagon(labels_durations_confs, cfg=None):
    """Return (first, last) stable WAGON segment indices."""
    r = smooth(labels_durations_confs, cfg)
    idx = [i for i, lb in enumerate(r.stable_labels) if lb == W]
    return (idx[0] if idx else None, idx[-1] if idx else None), r


class TestBoundaryStability(unittest.TestCase):
    def test_burst_before_the_region_does_not_move_first_wagon(self):
        """A stray WAGON inside the engine region must not pull FIRST forward."""
        (first, last), r = _first_last_wagon(
            [(E, 4.0, 1.0), (W, 0.33, 0.62), (E, 4.0, 1.0),
             (W, 4.0, 1.0), (W, 4.0, 1.0)])
        self.assertEqual(first, 3, "FIRST_VALID_WAGON must be the real region")
        self.assertEqual(last, 4)

    def test_burst_after_the_region_does_not_move_last_wagon(self):
        """A stray WAGON inside the trailing empty track must not push LAST back."""
        (first, last), r = _first_last_wagon(
            [(W, 4.0, 1.0), (W, 4.0, 1.0), (U, 4.0, 1.0),
             (W, 0.33, 0.62), (U, 4.0, 1.0)])
        self.assertEqual(first, 0)
        self.assertEqual(last, 1, "LAST_VALID_WAGON must not follow the burst")

    def test_brakevan_burst_after_the_region_does_not_extend_non_wagon(self):
        """The reported failure: a late BRAKE_VAN burst must not eat wagons."""
        (first, last), r = _first_last_wagon(
            [(E, 8.0, 1.0), (W, 4.0, 1.0), (W, 4.0, 1.0), (B, 0.33, 0.64),
             (W, 4.0, 1.0), (W, 4.0, 1.0)])
        self.assertEqual((first, last), (1, 5),
                         "the wagon region must stay contiguous through the burst")
        self.assertEqual(r.stable_labels[3], W)

    def test_measured_trailing_empty_track_is_preserved(self):
        """The 1.33 s trailing empty-track region must NOT absorb into WAGON,
        or LAST_VALID_WAGON would extend into empty track."""
        (first, last), r = _first_last_wagon(
            [(E, 7.67, 1.0), (B, 3.67, 1.0), (W, 261.0, 1.0), (U, 1.33, 1.0)])
        self.assertEqual(r.stable_labels[3], U)
        self.assertEqual(last, 2)

    def test_flicker_does_not_create_multiple_regions(self):
        r = smooth([(W, 4.0, 1.0), (B, 0.33, 0.62), (W, 4.0, 1.0),
                    (B, 0.33, 0.62), (W, 4.0, 1.0), (B, 0.33, 0.62),
                    (W, 4.0, 1.0)])
        self.assertEqual(set(r.stable_labels), {W})
        wagon_intervals = [iv for iv in r.stable_intervals
                           if iv.classification == W]
        self.assertEqual(len(wagon_intervals), 1, "one contiguous wagon region")

    def test_stable_intervals_are_contiguous_and_complete(self):
        r = smooth([(U, 2.0, 1.0), (E, 8.0, 1.0), (B, 4.0, 1.0),
                    (W, 60.0, 1.0), (U, 2.0, 1.0)])
        self.assertEqual([iv.classification for iv in r.stable_intervals],
                         [U, E, B, W, U])
        total = sum(iv.n_segments for iv in r.stable_intervals)
        self.assertEqual(total, len(r.observations))


# ===========================================================================
# 10-11. UNKNOWN handling
# ===========================================================================

class TestUnknownHandling(unittest.TestCase):
    def test_unknown_never_becomes_wagon_automatically(self):
        r = smooth([(U, 4.0, 1.0), (U, 4.0, 1.0), (U, 4.0, 1.0)])
        self.assertEqual(set(r.stable_labels), {U})
        self.assertNotIn(W, r.stable_labels)

    def test_unknown_burst_between_wagons_is_smoothed_to_wagon(self):
        """A 0.33 s UNKNOWN blip inside a wagon run is noise, not a boundary."""
        r = smooth([(W, 4.0, 1.0), (U, 0.33, 0.60), (W, 4.0, 1.0)])
        self.assertEqual(r.stable_labels, [W, W, W])

    def test_stable_unknown_between_wagons_stays_unknown(self):
        """A genuine long UNKNOWN region is kept; the wagon-window layer then
        decides it is interior and still counts it (existing architecture)."""
        r = smooth([(W, 4.0, 1.0), (U, 4.0, 1.0), (W, 4.0, 1.0)])
        self.assertEqual(r.stable_labels, [W, U, W])

    def test_unmapped_model_class_becomes_unknown_not_wagon(self):
        lm = ts.build_label_mapping({0: "wagon", 1: "mystery_object"})
        self.assertEqual(lm.semantic_for("mystery_object"), U)
        self.assertNotEqual(lm.semantic_for("mystery_object"), W)


# ===========================================================================
# diagnostics, config, bridge
# ===========================================================================

class TestDiagnosticsAndBridge(unittest.TestCase):
    def test_every_transition_records_its_evidence(self):
        r = smooth([(W, 4.0, 1.0), (B, 0.33, 0.62), (W, 4.0, 1.0),
                    (B, 4.0, 1.0)])
        self.assertTrue(r.transitions)
        for t in r.transitions:
            d = t.to_dict()
            for key in ("from_class", "to_class", "frame", "time", "accepted",
                        "reason", "supporting_segments", "supporting_duration_s",
                        "challenger_score", "mean_confidence"):
                self.assertIn(key, d)
            self.assertTrue(d["reason"])
        self.assertTrue(any(t.accepted for t in r.transitions))
        self.assertTrue(any(not t.accepted for t in r.transitions))

    def test_result_reports_raw_and_stable_counts(self):
        r = smooth([(W, 4.0, 1.0), (B, 0.33, 0.62), (W, 4.0, 1.0)])
        d = r.to_dict()
        self.assertEqual(d["raw_class_counts"], {W: 2, B: 1})
        self.assertEqual(d["stable_class_counts"], {W: 3})
        self.assertEqual(d["segments_relabelled_by_smoothing"], 1)

    def test_disabled_passes_raw_labels_through(self):
        cfg = tc.TemporalClassificationConfig(enabled=False)
        r = smooth([(W, 4.0, 1.0), (B, 0.33, 0.62), (W, 4.0, 1.0)], cfg)
        self.assertEqual(r.stable_labels, [W, B, W])

    def test_bridge_returns_smoothed_master_classifications(self):
        cls = [_MasterClassification(0, 0, 59, W, 1.0),
               _MasterClassification(1, 60, 64, B, 0.62),
               _MasterClassification(2, 65, 124, W, 1.0)]
        out, r = tc.apply_temporal_classification(cls, FPS, "RIGHT_UP",
                                                 verbose=False)
        self.assertEqual([c.label for c in out], [W, W, W])
        self.assertEqual([c.segment_index for c in out], [0, 1, 2])
        self.assertEqual([c.start_frame for c in out], [0, 60, 65],
                         "frames and timestamps are never altered")

    def test_bridge_applies_layer1_when_samples_are_available(self):
        cls = [_MasterClassification(0, 0, 74, W, 0.62)]
        history = {0: [sample(W, 0.62), sample(W, 0.62),
                       sample(B, 1.0), sample(B, 1.0), sample(B, 1.0)]}
        out, r = tc.apply_temporal_classification(
            cls, FPS, "RIGHT_UP", sample_history=history, verbose=False)
        self.assertEqual(out[0].label, B, "confidence-weighted re-vote applies")
        self.assertTrue(r.observations[0].weighted_scores)

    def test_single_and_empty_sequences(self):
        self.assertEqual(smooth([(W, 4.0, 1.0)]).stable_labels, [W])
        empty = tc.smooth_class_sequence([], camera_id="X", verbose=False)
        self.assertEqual(empty.stable_labels, [])
        self.assertEqual(empty.stable_intervals, [])

    def test_config_is_serializable_and_documented(self):
        d = tc.TemporalClassificationConfig().describe()
        for k in ("min_stable_region_s", "switch_persistence",
                  "min_confidence_to_challenge", "min_switch_confidence",
                  "use_confidence_weighted_vote"):
            self.assertIn(k, d)
        self.assertEqual(d["switch_persistence"], 3)
        self.assertEqual(d["min_stable_region_s"], 1.0)

    def test_mixed_class_challenge_cannot_switch(self):
        """B then E then W after a stable W: no single successor, so hold."""
        r = smooth([(W, 10.0, 1.0), (B, 0.3, 0.9), (E, 0.3, 0.9),
                    (W, 10.0, 1.0)])
        self.assertEqual(r.stable_labels, [W, W, W, W])


if __name__ == "__main__":
    unittest.main(verbosity=2)
