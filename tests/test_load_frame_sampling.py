"""Regression tests for the Load frame-sampling defect.

Load declared `every_nth=2` but passed `max_frames=0`, and
`features/_common.iter_wagon_frames` guards its subsample with
`max_frames is not None`.  A 0 was therefore a hard cap of ZERO --
`np.linspace(0, n-1, 0)` is empty -- so Load yielded no frames at all and
wrote NO_FRAMES for every wagon.  `every_nth=2` never took effect.

These tests pin the contract from the Load side.  The shared helper in
`features/_common.py` is intentionally NOT modified, so the sentinel semantics
it already documents (`None` == unbounded) are asserted here rather than changed.
"""

from __future__ import annotations

import inspect
import os
import tempfile
import unittest

import numpy as np

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

import cv2

from core import constants as C
from features._common import iter_wagon_frames, list_wagon_frames
from features.load import processor as load_proc


def _make_cache(root: str, gw: str, camera: str, n: int) -> None:
    d = os.path.join(root, gw, C.CAMERA_FOLDER[camera])
    os.makedirs(d, exist_ok=True)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    for i in range(n):
        cv2.imwrite(os.path.join(d, f"frame_{i:06d}.jpg"), img)


class TestMaxFramesSentinel(unittest.TestCase):
    """`None` means unbounded; `0` means 'cap at zero'. They are NOT the same."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gw, self.cam = "GW_1", C.CAMERA_RIGHT_UP_TOP
        _make_cache(self.tmp.name, self.gw, self.cam, 40)

    def tearDown(self):
        self.tmp.cleanup()

    def _count(self, **kw):
        return len(list(iter_wagon_frames(self.tmp.name, self.gw, self.cam, **kw)))

    def test_none_is_unbounded(self):
        stable = len(list_wagon_frames(self.tmp.name, self.gw, self.cam,
                                       trim_stable=True))
        self.assertEqual(self._count(max_frames=None, trim_stable=True), stable)

    def test_zero_yields_nothing(self):
        """The defect, pinned so nobody reintroduces 0 as an 'unbounded' value."""
        self.assertEqual(self._count(max_frames=0, trim_stable=True), 0)

    def test_every_nth_takes_effect_only_when_unbounded(self):
        stable = len(list_wagon_frames(self.tmp.name, self.gw, self.cam,
                                       trim_stable=True))
        got = self._count(every_nth=2, max_frames=None, trim_stable=True)
        self.assertEqual(got, (stable + 1) // 2)
        self.assertGreater(got, 0)
        # ...and is completely defeated by the 0 sentinel.
        self.assertEqual(self._count(every_nth=2, max_frames=0, trim_stable=True), 0)

    def test_sampled_frames_keep_original_cache_indices(self):
        idxs = [fi for fi, _ in iter_wagon_frames(
            self.tmp.name, self.gw, self.cam, every_nth=2,
            max_frames=None, trim_stable=True)]
        self.assertEqual(idxs, sorted(idxs))
        self.assertEqual(len(set(idxs)), len(idxs))
        # stride 2 over the stable interior -> gaps of 2 in the ORIGINAL numbering
        self.assertTrue(all(b - a == 2 for a, b in zip(idxs, idxs[1:])))


class TestLoadProcessorDefaults(unittest.TestCase):
    def test_load_default_max_frames_is_none_not_zero(self):
        sig = inspect.signature(load_proc.run)
        default = sig.parameters["max_frames"].default
        self.assertIsNone(
            default,
            "Load must pass max_frames=None; 0 is a cap of zero and silently "
            "disables all Load inference")

    def test_load_default_every_nth_unchanged(self):
        sig = inspect.signature(load_proc.run)
        self.assertEqual(sig.parameters["every_nth"].default, 2,
                         "Load's documented every-2nd-frame sampling must persist")

    def test_load_yields_frames_with_its_own_defaults(self):
        """End-to-end on the sampling path Load actually uses."""
        sig = inspect.signature(load_proc.run)
        every_nth = sig.parameters["every_nth"].default
        max_frames = sig.parameters["max_frames"].default
        with tempfile.TemporaryDirectory() as tmp:
            _make_cache(tmp, "GW_1", C.CAMERA_RIGHT_UP_TOP, 40)
            n = len(list(iter_wagon_frames(
                tmp, "GW_1", C.CAMERA_RIGHT_UP_TOP,
                every_nth=every_nth, max_frames=max_frames, trim_stable=True)))
        self.assertGreater(
            n, 0, "Load's own defaults still yield zero frames -- it would "
                  "write NO_FRAMES for every wagon")


if __name__ == "__main__":
    unittest.main()
