"""A door snapshot must never appear to contradict the verdict beside it.

From production, train 20260724_063206, GW_45: the combined report showed

    LEFT_UP     Left Door: CLOSED        <- text and JSON agreed
    (picture)   DAMAGED 0.90             <- the same page

Both halves were internally correct, which is what made it hard to read. The
mechanism, confirmed against that wagon's own state file
(`left_door: CLOSED, conf 0.0, tracks 1`):

  1. a raw LEFT_UP detection of DAMAGED at 0.90 filled the evidence bucket;
  2. the tracker confirmed NO left-side track, so `_pick_side_state` returned
     NO_DATA and `run`'s conservative default set the side to CLOSED at
     confidence 0.0 -- a default, not a measurement;
  3. `_resolve_evidence` had no CLOSED frame to show and fell back to the only
     frame captured -- the rejected DAMAGED one;
  4. the label was drawn from THAT FRAME's state, so the picture asserted
     DAMAGED while the row asserted CLOSED.

`conf 0.0` is the proof: it is reachable only through that default, and the
default only fires when there are no decisions at all. Had a DAMAGED *decision*
existed the verdict would have been DAMAGED, because `_pick_side_state` ranks
DAMAGED first ("terminal in the FSM").

The door VERDICT is deliberately unchanged. Confirming a damaged door from a
single unconfirmed frame would raise false positives on every wagon. What
changes is that the evidence stops claiming a state the wagon is not reported
as having.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import constants as C                                   # noqa: E402
from features.door import processor as DP                          # noqa: E402


class _Tracker:
    """Stands in for BestFrameTracker: only `has_data`/`score` are consulted."""

    def __init__(self, score=0.0, data=True):
        self._score = score
        self._data = data

    @property
    def score(self):
        return self._score

    def has_data(self):
        return self._data


# ---------------------------------------------------------------------------
# 1. _resolve_evidence reports whether the frame matches the verdict
# ---------------------------------------------------------------------------

class TestResolveEvidenceReportsMatching(unittest.TestCase):

    def test_a_frame_of_the_reported_state_is_matched(self):
        closed = _Tracker(0.5)
        got, matched = DP._resolve_evidence(
            {C.DOOR_CLOSED: closed, C.DOOR_DAMAGED: _Tracker(0.9)},
            C.DOOR_CLOSED)
        self.assertIs(got, closed)
        self.assertTrue(matched)

    def test_the_production_case_falls_back_and_says_so(self):
        """Only a DAMAGED frame exists; the verdict is the CLOSED default."""
        damaged = _Tracker(0.9)
        got, matched = DP._resolve_evidence({C.DOOR_DAMAGED: damaged},
                                            C.DOOR_CLOSED)
        self.assertIs(got, damaged, "the only captured frame should be used")
        self.assertFalse(matched, "a fallback must not report as a match")

    def test_the_highest_scoring_frame_wins_the_fallback(self):
        best = _Tracker(0.9)
        got, matched = DP._resolve_evidence(
            {C.DOOR_DAMAGED: best, C.DOOR_OPEN: _Tracker(0.3)}, C.DOOR_CLOSED)
        self.assertIs(got, best)
        self.assertFalse(matched)

    def test_an_empty_bucket_for_the_reported_state_is_not_a_match(self):
        """Present-but-empty is the same as absent, and must not claim a match."""
        got, matched = DP._resolve_evidence(
            {C.DOOR_CLOSED: _Tracker(0.0, data=False),
             C.DOOR_DAMAGED: _Tracker(0.9)}, C.DOOR_CLOSED)
        self.assertFalse(matched)

    def test_no_candidates_at_all_yields_no_data_and_no_match(self):
        got, matched = DP._resolve_evidence({}, C.DOOR_CLOSED)
        self.assertFalse(got.has_data())
        self.assertFalse(matched)


# ---------------------------------------------------------------------------
# 2. The label
# ---------------------------------------------------------------------------

class TestEvidenceLabel(unittest.TestCase):

    def test_a_matching_frame_is_labelled_plainly(self):
        self.assertEqual(
            DP.evidence_label(C.DOOR_OPEN, 0.83, matched=True,
                              reported_state=C.DOOR_OPEN),
            f"{C.DOOR_OPEN} 0.83")

    def test_the_production_label_is_marked_unconfirmed(self):
        got = DP.evidence_label(C.DOOR_DAMAGED, 0.90, matched=False,
                                reported_state=C.DOOR_CLOSED)
        self.assertIn("UNCONFIRMED", got)
        self.assertIn(C.DOOR_DAMAGED, got, "the frame's own state must show")
        self.assertIn(C.DOOR_CLOSED, got, "the verdict must be named too")

    def test_a_matching_label_never_says_unconfirmed(self):
        for state in (C.DOOR_CLOSED, C.DOOR_OPEN, C.DOOR_PARTIAL,
                      C.DOOR_DAMAGED):
            self.assertNotIn(
                "UNCONFIRMED",
                DP.evidence_label(state, 0.7, matched=True,
                                  reported_state=state))

    def test_missing_values_do_not_raise(self):
        for kwargs in (dict(frame_state=None, confidence=None),
                       dict(frame_state="", confidence=0)):
            got = DP.evidence_label(matched=False, reported_state="", **kwargs)
            self.assertIn("UNCONFIRMED", got)


# ---------------------------------------------------------------------------
# 3. The metadata a consumer reads
# ---------------------------------------------------------------------------

class TestFallbackIsRecordedInMetadata(unittest.TestCase):
    """A downstream reader must be able to tell evidence OF a state from
    evidence merely captured NEAR it."""

    def _src(self):
        return open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "features", "door", "processor.py"), encoding="utf-8").read()

    def test_the_side_metadata_carries_the_fallback_flags(self):
        src = self._src()
        for key in ("reported_state", "matches_reported_state",
                    "evidence_is_fallback"):
            self.assertIn(f'"{key}"', src, f"{key} is not recorded")

    def test_the_label_goes_through_the_shared_helper(self):
        """One definition, so the picture and the metadata cannot disagree."""
        src = self._src()
        self.assertIn("evidence_label(", src)
        self.assertNotIn('label=f"{side_best.meta.get(\'state\',\'?\')} "', src)

    def test_the_door_verdict_logic_is_untouched(self):
        """DAMAGED still ranks first; the conservative CLOSED default remains."""
        src = self._src()
        self.assertIn("1. Any DAMAGED track  -> DAMAGED", src)
        self.assertIn("l_state, l_conf = C.DOOR_CLOSED, 0.0", src)
