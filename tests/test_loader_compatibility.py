"""Every `load_yolo()` call must actually fit `load_yolo()`.

The first real EC2 run died instantly: the new shared collector called
`load_yolo(path, verbose=verbose)` and the loader takes only the path. Nothing
caught it here because every collector test injected stub models through
`models=`, so the real loader branch was never entered.

Two guards, and they are different in kind:

* A STATIC one that binds every `load_yolo(...)` call site in the tree against
  the loader's real signature. It needs no weights, no video and no GPU, so it
  runs everywhere and covers callers this file has never heard of.
* A FUNCTIONAL one that drives the production collector down the real loader
  branch with a signature-enforcing spy, so the wiring is exercised rather than
  merely read.

The spy takes its signature FROM the real function rather than restating it, so
it cannot drift into accepting something production would reject.
"""

from __future__ import annotations

import ast
import inspect
import os
import tempfile
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

import cv2
import numpy as np

from core import constants as C
from core.master_timeline import CameraClock
from features._common import load_yolo

FPS = 15.0
W, H = 64, 48
N_FRAMES = 12
RU = C.CAMERA_RIGHT_UP

#: Live source trees. The two frozen reference copies of wagon_count are
#: deliberately excluded -- they are held byte-identical on purpose and are not
#: import targets.
SCAN_DIRS = ("core", "features", "orchestrator", "wagon_count",
             "reconstruction", "rendering", "reporting", "fusion",
             "benchmarks", "tests")
EXCLUDE = ("wagon_count - Copy_correct_count", "_legacy_wagon_count_removed",
           "__pycache__")


def _python_files():
    for d in SCAN_DIRS:
        root = os.path.join(V4_ROOT, d)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if any(x in dirpath for x in EXCLUDE):
                continue
            dirnames[:] = [x for x in dirnames if x not in EXCLUDE]
            for fn in filenames:
                if fn.endswith(".py"):
                    yield os.path.join(dirpath, fn)


def _load_yolo_calls():
    """(path, lineno, n_positional, keyword_names) for every call site."""
    out = []
    for path in _python_files():
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                tree = ast.parse(f.read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            try:
                name = ast.unparse(node.func)
            except Exception:
                continue
            if name.split(".")[-1] != "load_yolo":
                continue
            out.append((path, node.lineno, node.args, node.keywords))
    return out


def _video(path: str) -> str:
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(N_FRAMES):
        vw.write(np.full((H, W, 3), 180, dtype=np.uint8))
    vw.release()
    return path


class SignatureEnforcingSpy:
    """Accepts exactly what the real loader accepts, and records the call.

    `__signature__` and the binding check both come from the real function, so
    a call this spy tolerates is a call production tolerates.
    """

    def __init__(self, real, returns=None):
        self.signature = inspect.signature(real)
        self.__signature__ = self.signature
        self.calls = []
        self._returns = returns

    def __call__(self, *args, **kwargs):
        # Raises TypeError for an unknown keyword or wrong arity -- the exact
        # failure the EC2 run hit, surfaced here instead of in production.
        bound = self.signature.bind(*args, **kwargs)
        self.calls.append((args, dict(kwargs)))
        return self._returns


class StubModel:
    names = {0: "open_door"}

    def __call__(self, frame, verbose=False):
        class _A:
            def __init__(self, a):
                self._a = np.asarray(a)

            def cpu(self):
                return self

            def numpy(self):
                return self._a

        class _B:
            def __init__(s):
                s.xyxy = _A(np.array([[5.0, 5.0, 30.0, 40.0]]))
                s.conf = _A(np.array([0.95]))
                s.cls = _A(np.array([0], dtype=int))

            def __len__(s):
                return 1

        class _R:
            boxes = _B()

        return [_R()]


class TestLoaderSignature(unittest.TestCase):
    """What the loader actually accepts, pinned."""

    def test_it_takes_the_path_and_nothing_else(self):
        params = inspect.signature(load_yolo).parameters
        self.assertEqual(list(params), ["model_path"])
        self.assertEqual(params["model_path"].kind,
                         inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertIs(params["model_path"].default,
                      inspect.Parameter.empty)

    def test_it_has_no_verbose_parameter(self):
        """The specific kwarg that broke the run."""
        self.assertNotIn("verbose", inspect.signature(load_yolo).parameters)

    def test_it_accepts_no_arbitrary_keywords(self):
        sig = inspect.signature(load_yolo)
        self.assertFalse([p for p in sig.parameters.values()
                          if p.kind in (inspect.Parameter.VAR_KEYWORD,
                                        inspect.Parameter.VAR_POSITIONAL)],
                         "load_yolo absorbs **kwargs -- an unsupported "
                         "keyword would pass silently")


class TestEveryCallSiteBinds(unittest.TestCase):
    """Static: no caller anywhere can pass something the loader rejects."""

    def test_the_scan_finds_the_known_callers(self):
        """Non-vacuity: an empty scan would make the next test meaningless."""
        calls = _load_yolo_calls()
        files = {os.path.basename(p) for p, _l, _a, _k in calls}
        self.assertGreaterEqual(len(calls), 5,
                                f"only found {len(calls)} call sites")
        for expected in ("processor.py", "raw_collect.py"):
            self.assertIn(expected, files)

    def test_every_call_binds_to_the_real_signature(self):
        sig = inspect.signature(load_yolo)
        for path, lineno, args, keywords in _load_yolo_calls():
            rel = os.path.relpath(path, V4_ROOT)
            if any(k.arg is None for k in keywords):
                continue                       # **kwargs splat, cannot bind
            if any(isinstance(a, ast.Starred) for a in args):
                continue                       # *args splat, cannot bind
            with self.subTest(site=f"{rel}:{lineno}"):
                try:
                    sig.bind(*[object() for _ in args],
                             **{k.arg: object() for k in keywords})
                except TypeError as e:
                    self.fail(f"{rel}:{lineno} calls load_yolo("
                              f"{len(args)} positional, "
                              f"{[k.arg for k in keywords]}) -- {e}")

    def test_no_caller_passes_verbose(self):
        for path, lineno, _args, keywords in _load_yolo_calls():
            rel = os.path.relpath(path, V4_ROOT)
            with self.subTest(site=f"{rel}:{lineno}"):
                self.assertNotIn("verbose", [k.arg for k in keywords])

    def test_the_binder_would_reject_the_regression(self):
        """Negative control: this is what the EC2 run did."""
        sig = inspect.signature(load_yolo)
        with self.assertRaises(TypeError):
            sig.bind("model.pt", verbose=True)


class TestProductionCollectorUsesTheRealLoader(unittest.TestCase):
    """Functional: the collector's real loader branch is exercised.

    Every other collector test injects stubs through `models=`, which skips
    `_load` entirely -- which is exactly why nothing caught this. Here no
    models are injected, a weights file is present so the existence check
    passes, and the loader itself is a signature-enforcing spy.
    """

    def _run(self, models_dir):
        from features import raw_collect
        from features._common import load_yolo as real
        spy = SignatureEnforcingSpy(real, returns=StubModel())
        original = raw_collect.__dict__.get("load_yolo")
        import features._common as common
        saved = common.load_yolo
        common.load_yolo = spy
        try:
            with tempfile.TemporaryDirectory() as root:
                r = raw_collect.collect_camera(
                    camera_id=RU, video_path=_video(os.path.join(root, "v.mp4")),
                    feature_models_dir=models_dir, features=("door",),
                    clock=CameraClock(RU, fps=FPS, total_frames=N_FRAMES),
                    strides={"door": 3}, models=None, verbose=False)
            return spy, r
        finally:
            common.load_yolo = saved
            if original is not None:
                raw_collect.load_yolo = original

    def test_the_loader_is_called_with_only_a_path(self):
        with tempfile.TemporaryDirectory() as md:
            from features.raw_collect import MODEL_FILES
            open(os.path.join(md, MODEL_FILES["door"]), "wb").close()
            spy, r = self._run(md)
        self.assertTrue(spy.calls, "the real loader branch was never reached; "
                                   "this test would not catch a bad kwarg")
        for args, kwargs in spy.calls:
            self.assertEqual(len(args), 1, "loader called with wrong arity")
            self.assertEqual(kwargs, {}, f"unsupported loader kwargs: {kwargs}")
            self.assertTrue(str(args[0]).endswith(MODEL_FILES["door"]))

    def test_collection_actually_proceeds_after_loading(self):
        """The spy returns a usable model, so the pass is real, not aborted."""
        with tempfile.TemporaryDirectory() as md:
            from features.raw_collect import MODEL_FILES
            open(os.path.join(md, MODEL_FILES["door"]), "wb").close()
            spy, r = self._run(md)
        self.assertEqual(r.frames_read, N_FRAMES)
        self.assertIn("door", r.detectors_run)
        self.assertGreater(r.frames_scored.get("door", 0), 0)

    def test_a_bad_kwarg_would_fail_this_test(self):
        """Negative control: the spy rejects what the real loader rejects."""
        from features._common import load_yolo as real
        spy = SignatureEnforcingSpy(real)
        with self.assertRaises(TypeError):
            spy("model.pt", verbose=True)
        spy("model.pt")          # the supported form still works

    def test_a_missing_weights_file_short_circuits_without_calling(self):
        """Unchanged behaviour: absent weights mean no loader call at all."""
        with tempfile.TemporaryDirectory() as md:
            spy, r = self._run(md)
        self.assertEqual(spy.calls, [])
        self.assertTrue(r.skipped or not r.detectors_run)


class TestTheWrapperHasNoDeadParameter(unittest.TestCase):
    """The fix removed the kwarg; it did not absorb it."""

    def test_the_loader_gained_no_ignored_parameter(self):
        self.assertEqual(list(inspect.signature(load_yolo).parameters),
                         ["model_path"])

    def test_the_collectors_wrapper_takes_no_verbose(self):
        from features.raw_collect import _load
        self.assertNotIn("verbose", inspect.signature(_load).parameters)

    def test_the_wrapper_forwards_only_the_path(self):
        import features.raw_collect as rc
        src = inspect.getsource(rc._load)
        calls = [n for n in ast.walk(ast.parse(src.lstrip()))
                 if isinstance(n, ast.Call)
                 and ast.unparse(n.func).endswith("load_yolo")]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].keywords, [])
        self.assertEqual(len(calls[0].args), 1)


if __name__ == "__main__":
    unittest.main()
