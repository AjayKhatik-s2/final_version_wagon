"""The camera-local report must come from the EXISTING proven renderer.

The first sequential implementation shipped a from-scratch ReportLab design,
so its PDFs looked nothing like the batch camera reports. The fix is an
adapter: build the dataclasses `reporting/camera_reports.build_camera_report()`
already expects, keyed by CAMERA-LOCAL ids, and call it unchanged.

These tests pin that -- no second renderer, no `GW_n` before assembly, and the
existing report builders untouched.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

import cv2
import numpy as np

from core import constants as C
from core.camera_evidence import (
    LIFECYCLE, CameraEvidenceBundle, LocalSegment, local_segment_id,
)
from orchestrator import camera_report_adapter as adapter

CAM = "RIGHT_UP"


def _bundle(root, cam=CAM, n=3, with_frames=True, with_features=True):
    """A sealed bundle in the exact shape camera_runner writes."""
    b = CameraEvidenceBundle(root, cam)
    os.makedirs(b.dir, exist_ok=True)
    for s in LIFECYCLE[1:]:
        b.advance(s)
    segs = [LocalSegment(local_id=local_segment_id(cam, i), index=i,
                         start_frame=i * 60, end_frame=i * 60 + 59,
                         start_time=float((i - 1) * 4), end_time=float(i * 4),
                         label="WAGON", confidence=0.95)
            for i in range(1, n + 1)]
    b.write_segments(segs)
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    # Feature JSON only for what this camera is AUTHORITATIVE for, mirroring
    # camera_runner._feature_plan(): side -> door, top -> load + damage.
    payloads = ({"door": {"left_door": "CLOSED", "left_door_confidence": 0.91,
                          "right_door": "OPEN", "right_door_confidence": 0.88}}
                if cam in C.SIDE_CAMERAS else
                {"load": {"load_status": "LOADED",
                          "load_confidence": 0.83},
                 "damage": {"top_damage": "DAMAGE",
                            "top_damage_details": ["dent"],
                            "top_damage_confidence": 0.77}})
    if with_features:
        for feat, extra in payloads.items():
            d = os.path.join(b.dir, "features", feat)
            os.makedirs(d, exist_ok=True)
            for s in segs:
                with open(os.path.join(d, f"{s.local_id}.json"), "w",
                          encoding="utf-8") as f:
                    json.dump({"global_id": s.local_id, "feature": feat,
                               "status": "OK",
                               "supporting_cameras": [cam],
                               "frame_count": 12, **extra}, f)
    if with_frames:
        for s in segs:
            cd = os.path.join(b.dir, "camera_cache", s.local_id,
                              C.CAMERA_FOLDER[cam])
            os.makedirs(cd, exist_ok=True)
            cv2.imwrite(os.path.join(cd, f"frame_{s.start_frame:06d}.jpg"), img)
            for feat in payloads:
                ed = os.path.join(b.dir, "evidence", s.local_id, feat)
                os.makedirs(ed, exist_ok=True)
                cv2.imwrite(os.path.join(ed, "right_best.jpg"), img)
    return b, segs


class TestNoSecondRenderer(unittest.TestCase):
    def test_custom_renderer_is_gone(self):
        self.assertFalse(
            os.path.exists(os.path.join(V4_ROOT, "reporting",
                                        "camera_local_report.py")),
            "the from-scratch camera renderer must be deleted")

    def test_adapter_delegates_to_the_proven_builder(self):
        src = inspect.getsource(adapter.build_local_camera_pdf)
        self.assertIn("from reporting import camera_reports", src)
        self.assertIn("camera_reports.build_camera_report(", src)

    def test_adapter_renders_nothing_itself(self):
        """No ReportLab primitive may appear -- the adapter only builds state."""
        src = inspect.getsource(adapter)
        for banned in ("SimpleDocTemplate", "Paragraph", "TableStyle",
                       "getSampleStyleSheet", "PageBreak", "reportlab"):
            with self.subTest(token=banned):
                self.assertNotIn(banned, src)

    def test_existing_report_builders_unmodified(self):
        import subprocess
        r = subprocess.run(["git", "diff", "--name-only", "HEAD",
                            "reporting/camera_reports.py",
                            "reporting/combined_train_report.py",
                            "reporting/_pages.py", "reporting/_brand.py"],
                           cwd=V4_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("git unavailable")
        self.assertEqual([x for x in r.stdout.split() if x], [])


class TestLocalIds(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.b, self.segs = _bundle(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_state_uses_local_ids(self):
        state, _u, _p = adapter.adapt(self.b)
        self.assertEqual([w.global_id for w in state.wagons],
                         [f"L_{CAM}_{i}" for i in (1, 2, 3)])

    def test_no_gw_id_is_ever_created(self):
        state, unified, _p = adapter.adapt(self.b)
        for w in state.wagons:
            self.assertFalse(w.global_id.startswith("GW_"))
        for k in unified:
            self.assertFalse(k.startswith("GW_"))

    def test_every_segment_appears(self):
        state, unified, _p = adapter.adapt(self.b)
        self.assertEqual(len(state.wagons), len(self.segs))
        self.assertEqual(set(unified), {s.local_id for s in self.segs})

    def test_absolute_frame_numbers_preserved(self):
        state, _u, _p = adapter.adapt(self.b)
        for w, s in zip(state.wagons, self.segs):
            self.assertEqual(w.start_frame_master, s.start_frame)
            self.assertEqual(w.end_frame_master, s.end_frame)
            self.assertEqual(w.start_time, s.start_time)

    def test_per_camera_ids_are_namespaced(self):
        with tempfile.TemporaryDirectory() as root:
            for cam in ("LEFT_UP", "RIGHT_UP_TOP", "LEFT_UP_TOP"):
                b, _ = _bundle(root, cam=cam, n=2)
                state, _u, _p = adapter.adapt(b)
                self.assertEqual([w.global_id for w in state.wagons],
                                 [f"L_{cam}_1", f"L_{cam}_2"])


class TestFeatureAssociation(unittest.TestCase):
    def test_feature_json_binds_to_the_right_local_id(self):
        with tempfile.TemporaryDirectory() as root:
            b, segs = _bundle(root)
            _s, unified, _p = adapter.adapt(b)
            for s in segs:
                u = unified[s.local_id]
                self.assertEqual(u.global_id, s.local_id)
                self.assertEqual(u.right_door, "OPEN")
                self.assertEqual(u.left_door, "CLOSED")
                self.assertIn("RIGHT_DOOR_OPEN", u.anomalies)

    def test_reuses_the_existing_fusion_builder(self):
        src = inspect.getsource(adapter.local_unified)
        self.assertIn("from fusion import wagon_state_builder", src)
        self.assertIn("wagon_state_builder.build(", src)

    def test_does_not_write_a_unified_tree_into_the_bundle(self):
        """Rendering must not leave artefacts assembly could misread."""
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            adapter.adapt(b)
            self.assertFalse(os.path.exists(
                os.path.join(b.dir, "features", "unified")))


class TestPathsResolve(unittest.TestCase):
    def test_adapter_points_at_the_bundle_layout(self):
        with tempfile.TemporaryDirectory() as root:
            b, segs = _bundle(root)
            _s, _u, paths = adapter.adapt(b)
            self.assertTrue(os.path.isdir(paths["cache_root"]))
            self.assertTrue(os.path.isdir(paths["wagon_states_root"]))
            self.assertTrue(os.path.isdir(paths["evidence_root"]))

    def test_wagon_covered_sees_the_local_cache(self):
        """The proven renderer's visibility probe must resolve unchanged."""
        from reporting.camera_reports import _wagon_covered
        with tempfile.TemporaryDirectory() as root:
            b, segs = _bundle(root)
            _s, _u, paths = adapter.adapt(b)
            for s in segs:
                self.assertTrue(_wagon_covered(paths["cache_root"],
                                               s.local_id, CAM),
                                f"{s.local_id} should be visible")

    def test_evidence_snapshot_resolves(self):
        from reporting import _evidence_lookup as ev
        with tempfile.TemporaryDirectory() as root:
            b, segs = _bundle(root)
            _s, _u, paths = adapter.adapt(b)
            snap = ev.evidence_snapshot(paths["evidence_root"],
                                        segs[0].local_id, "door", "right_best")
            self.assertTrue(snap and os.path.isfile(snap))


def _pdf_text(path: str) -> str:
    """Text drawn in a ReportLab PDF.

    ReportLab Flate-compresses page content streams, so scanning the raw file
    for an id finds nothing -- and any "does not contain GW_" assertion on raw
    bytes then passes vacuously. Inflate every stream first, then pull the
    parenthesised operands of the text operators so ids read as plain
    characters.
    """
    import base64
    import re
    import zlib

    def _decode(body: bytes) -> bytes:
        """Return the stream's TEXT operators, or b"" if it isn't text.

        A PDF mixes encodings: ReportLab writes ASCII85 (sometimes over
        Flate) for embedded JPEGs and for content streams alike. Guessing by
        trying decoders in order is unreliable -- `a85decode` "succeeds" on
        data that was never ASCII85 and returns garbage, and ASCII85 payload
        can coincidentally contain the bytes `Tj`. So decode every way, then
        keep the candidate that actually looks like a content stream.
        """
        body = body.strip()
        cands = [body]
        for step in (
            lambda d: base64.a85decode(d, adobe=True),
            lambda d: zlib.decompress(d),
            lambda d: zlib.decompress(base64.a85decode(d, adobe=True)),
        ):
            try:
                cands.append(step(body))
            except Exception:
                pass
        for c in cands:
            if b"BT" in c and (b"Tj" in c or b"TJ" in c):
                return c
        return b""

    with open(path, "rb") as f:
        raw = f.read()
    chunks = [_decode(m.group(1))
              for m in re.finditer(br"stream\r?\n(.*?)endstream", raw, re.S)]
    text = b"\n".join(chunks).decode("latin-1")
    # ReportLab splits strings for kerning, so collect every parenthesised
    # run and join them rather than matching whole words. Scanned by hand:
    # PDF string escaping is backslash-based and awkward to express as a
    # regex without getting the escaping wrong.
    out, buf, depth, i, n = [], [], 0, 0, len(text)
    while i < n:
        ch = text[i]
        if depth:
            if ch == "\\" and i + 1 < n:
                buf.append(text[i + 1])
                i += 2
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if not depth:
                    out.append("".join(buf))
                    buf = []
                    i += 1
                    continue
            buf.append(ch)
        elif ch == "(":
            depth = 1
        i += 1
    return " ".join(out)


class TestRenderedPdf(unittest.TestCase):
    """Render with the real builder and inspect the DRAWN TEXT."""

    def test_helper_can_actually_read_drawn_text(self):
        """Guard the guard: if extraction returned nothing, the id assertions
        below would pass vacuously."""
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            out = os.path.join(root, "x.pdf")
            adapter.build_local_camera_pdf(b, output_pdf=out,
                                          batch_key="RIGHT_UP (camera-local)",
                                          verbose=False)
            txt = _pdf_text(out)
            self.assertGreater(len(txt), 200,
                               "no text extracted -- id checks would be vacuous")
            self.assertIn("RIGHT_UP", txt)

    def test_pdf_is_produced_by_the_proven_renderer(self):
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            out = os.path.join(root, f"{CAM}_report.pdf")
            p = adapter.build_local_camera_pdf(
                b, output_pdf=out, batch_key=f"{CAM} (camera-local)",
                fps=15.0, total_frames=3555, verbose=False)
            self.assertTrue(p and os.path.isfile(p))
            # The proven renderer emits a multi-section document; the earlier
            # hand-rolled one was ~3.5 KB.
            self.assertGreater(os.path.getsize(p), 10000,
                               "PDF too small to be the full proven layout")

    def test_pdf_shows_local_ids_and_no_gw_identifier(self):
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root)
            out = os.path.join(root, f"{CAM}_report.pdf")
            adapter.build_local_camera_pdf(
                b, output_pdf=out, batch_key=f"{CAM} (camera-local)",
                verbose=False)
            txt = _pdf_text(out)
            self.assertIn("L_RIGHT_UP_1", txt,
                          "the camera-local id must be printed in the report")
            self.assertNotIn("GW_", txt,
                             "a camera-local PDF must not show GW_ ids")

    def test_every_local_id_is_printed(self):
        with tempfile.TemporaryDirectory() as root:
            b, segs = _bundle(root)
            out = os.path.join(root, f"{CAM}_report.pdf")
            adapter.build_local_camera_pdf(
                b, output_pdf=out, batch_key=f"{CAM} (camera-local)",
                verbose=False)
            txt = _pdf_text(out)
            for s in segs:
                self.assertIn(s.local_id, txt, f"{s.local_id} missing from PDF")

    def test_report_survives_a_missing_cache(self):
        """A rendering problem must never un-seal a camera."""
        with tempfile.TemporaryDirectory() as root:
            b, _ = _bundle(root, with_frames=False)
            out = os.path.join(root, f"{CAM}_report.pdf")
            p = adapter.build_local_camera_pdf(
                b, output_pdf=out, batch_key="x", verbose=False)
            self.assertTrue(p is None or os.path.isfile(p))


class TestAllFourCameraPdfs(unittest.TestCase):
    """Every camera renders its own PDF, with its own local ids, alone."""

    def test_each_camera_prints_its_own_local_ids(self):
        expected = {"RIGHT_UP": "L_RIGHT_UP_1", "LEFT_UP": "L_LEFT_UP_1",
                    "RIGHT_UP_TOP": "L_RIGHT_UP_TOP_1",
                    "LEFT_UP_TOP": "L_LEFT_UP_TOP_1"}
        self.assertEqual(set(expected), set(C.ALL_CAMERAS))
        for cam, first_id in expected.items():
            with self.subTest(camera=cam), tempfile.TemporaryDirectory() as root:
                b, segs = _bundle(root, cam=cam)
                out = os.path.join(root, f"{cam}_report.pdf")
                p = adapter.build_local_camera_pdf(
                    b, output_pdf=out, batch_key=f"{cam} (camera-local)",
                    fps=15.0, total_frames=3555, verbose=False)
                self.assertTrue(p and os.path.isfile(p), f"{cam}: no PDF")
                txt = _pdf_text(p)
                self.assertIn(first_id, txt, f"{cam}: {first_id} not printed")
                self.assertNotIn("GW_", txt, f"{cam}: PDF shows a GW_ id")
                for s in segs:
                    self.assertIn(s.local_id, txt,
                                  f"{cam}: {s.local_id} not printed")

    def test_a_camera_renders_with_no_other_camera_present(self):
        """Only ONE bundle exists on disk while the PDF is built."""
        for cam in C.ALL_CAMERAS:
            with self.subTest(camera=cam), tempfile.TemporaryDirectory() as root:
                b, _ = _bundle(root, cam=cam)
                self.assertEqual(sorted(os.listdir(root)), [cam])
                p = adapter.build_local_camera_pdf(
                    b, output_pdf=os.path.join(root, "r.pdf"),
                    batch_key=cam, verbose=False)
                self.assertTrue(p and os.path.isfile(p))

    def test_top_cameras_report_load_and_damage(self):
        """Authority is preserved: the top cameras' own features are fused."""
        for cam in C.TOP_CAMERAS:
            with self.subTest(camera=cam), tempfile.TemporaryDirectory() as root:
                b, segs = _bundle(root, cam=cam)
                _s, unified, _p = adapter.adapt(b)
                u = unified[segs[0].local_id]
                self.assertEqual(u.load_status, "LOADED")
                self.assertEqual(u.top_damage, "DAMAGE")


class TestPathsMatchCameraRunner(unittest.TestCase):
    """The adapter's paths must be camera_runner's paths, not a guess.

    Hardcoding the same three strings in both files would look like agreement
    while being an assumption. This reads the literals camera_runner actually
    joins onto bundle.dir and requires the adapter to use exactly those.
    """

    def _bundle_subdirs(self, mod):
        import ast
        tree = ast.parse(inspect.getsource(mod))
        found = set()
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "join"):
                continue
            a = n.args
            if (len(a) >= 2 and isinstance(a[0], ast.Attribute)
                    and a[0].attr == "dir"
                    and isinstance(a[1], ast.Constant)
                    and isinstance(a[1].value, str)):
                found.add(a[1].value)
        return found

    def test_runner_and_adapter_agree_on_the_layout(self):
        from orchestrator import camera_report_adapter, camera_runner
        runner = self._bundle_subdirs(camera_runner)
        adapt = self._bundle_subdirs(camera_report_adapter)
        self.assertTrue(runner, "found no bundle.dir joins in camera_runner")
        self.assertTrue(adapt, "found no bundle.dir joins in the adapter")
        stray = sorted(adapt - runner)
        self.assertEqual(stray, [],
                         f"adapter reads paths camera_runner never writes: {stray}")

    def test_the_three_documented_roots_are_the_runner_s(self):
        from orchestrator import camera_runner
        runner = self._bundle_subdirs(camera_runner)
        for needed in ("camera_cache", "features", "evidence"):
            self.assertIn(needed, runner,
                          f"camera_runner no longer writes {needed}/")


class TestRunnerWiring(unittest.TestCase):
    def test_camera_runner_calls_the_adapter_not_a_custom_renderer(self):
        from orchestrator import camera_runner
        src = inspect.getsource(camera_runner)
        self.assertIn("build_local_camera_pdf", src)
        self.assertNotIn("camera_local_report", src)

    def test_report_is_emitted_before_sealing(self):
        from orchestrator import camera_runner
        src = inspect.getsource(camera_runner.run_camera)
        self.assertLess(src.index("build_local_camera_pdf"),
                        src.index('bundle.advance("SEALED")'),
                        "the camera PDF must exist before the camera seals")


class TestRunnerHasNoUndefinedNames(unittest.TestCase):
    """run_camera() cannot execute without the real weights, so a name that
    an edit leaves dangling would only surface on a 25-minute EC2 run.

    Rewiring the reporting stage did exactly that once: the PDF call was
    replaced but `report_payload` was still referenced one line above. This
    resolves every loaded name in the sequential runner statically.
    """

    def _undefined(self, fn_src_module, fn_name):
        import ast
        import builtins
        tree = ast.parse(inspect.getsource(fn_src_module))
        module_names = {n.asname or n.name.split(".")[0]
                        for node in ast.walk(tree)
                        if isinstance(node, (ast.Import, ast.ImportFrom))
                        for n in node.names}
        module_names |= {"__file__", "__name__", "__doc__", "__package__"}
        module_names |= {t.id for node in tree.body
                         if isinstance(node, ast.Assign)
                         for t in ast.walk(node) if isinstance(t, ast.Name)}
        module_names |= {n.name for n in ast.walk(tree)
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                           ast.ClassDef))}
        module_names |= {n.target.id for n in tree.body
                         if isinstance(n, ast.AnnAssign)
                         and isinstance(n.target, ast.Name)}

        fn = next(n for n in tree.body
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == fn_name)
        bound = set(dir(builtins)) | module_names
        a = fn.args
        for arg in (a.posonlyargs + a.args + a.kwonlyargs +
                    ([a.vararg] if a.vararg else []) +
                    ([a.kwarg] if a.kwarg else [])):
            bound.add(arg.arg)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store,
                                                              ast.Del)):
                bound.add(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for al in n.names:
                    bound.add(al.asname or al.name.split(".")[0])
            elif isinstance(n, ast.ExceptHandler) and n.name:
                bound.add(n.name)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef)):
                bound.add(n.name)
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                bound.update(n.names)
            elif isinstance(n, ast.Lambda):
                la = n.args
                for arg in (la.posonlyargs + la.args + la.kwonlyargs +
                            ([la.vararg] if la.vararg else []) +
                            ([la.kwarg] if la.kwarg else [])):
                    bound.add(arg.arg)
        return sorted({n.id for n in ast.walk(fn)
                       if isinstance(n, ast.Name)
                       and isinstance(n.ctx, ast.Load)
                       and n.id not in bound})

    def test_sequential_modules_resolve(self):
        """Every function in the sequential modules, not a chosen few."""
        import ast
        from orchestrator import camera_report_adapter, camera_runner
        from orchestrator import global_assembler
        for mod in (camera_runner, camera_report_adapter, global_assembler):
            tree = ast.parse(inspect.getsource(mod))
            # Top-level only: a nested closure legitimately reads names from
            # its enclosing function, and its body is already walked as part
            # of that function -- where those names ARE bound.
            fns = [n.name for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            self.assertTrue(fns, f"{mod.__name__}: no functions found")
            for fn in fns:
                with self.subTest(module=mod.__name__, function=fn):
                    self.assertEqual(self._undefined(mod, fn), [])

    def test_the_guard_actually_catches_a_dangling_name(self):
        """Negative control: without it, this test class proves nothing."""
        import textwrap
        mod = types_module = None
        src = textwrap.dedent("""
            def run_camera(bundle):
                bundle.write_json("x", report_payload)
        """)

        class _Fake:
            pass
        fake = _Fake()
        fake_src = src

        def _getsource(_m):
            return fake_src
        real = inspect.getsource
        try:
            inspect.getsource = _getsource
            self.assertEqual(self._undefined(fake, "run_camera"),
                             ["report_payload"])
        finally:
            inspect.getsource = real


if __name__ == "__main__":
    unittest.main()
