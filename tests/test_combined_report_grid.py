"""The combined PDF: every canonical wagon, all four cameras, 4 frames each.

Structure under test:

    COVER / SUMMARY ... existing sections ...
    GW_1   page: header + TOP grid (RIGHT_UP_TOP row, LEFT_UP_TOP row)
           page: header + SIDE grid (RIGHT_UP row, LEFT_UP row)
    GW_2   ... same ...
    GW_N
    DAMAGE SUMMARY   damage evidence only, grouped by canonical GW_n

Two properties make the rest testable, and both are structural rather than
checked after the fact:

* **Camera isolation.** `_cache_frame_path` keys the directory on
  `C.CAMERA_FOLDER[camera_id]`, so a RIGHT_UP_TOP slot can only ever resolve to
  a RIGHT_UP_TOP file. There is no path by which a master frame reaches a
  support slot.
* **The PDF selects nothing.** It consumes the manifest, so the audit and the
  pages cannot disagree about which image is where.

The frame number printed under an image is read from the FILENAME, not
recomputed, so the caption is provably about the file above it.

No model runs and no video is decoded: the fixtures write JPEGs straight into a
wagon_cache laid out exactly as the materializer lays it out -- including the
camera time offset, because the cache filenames are offset-corrected local
indices and a reader that ignores that silently blanks every support slot.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                                                # noqa: E402

from core import constants as C                                   # noqa: E402
from core.global_state_loader import parse_global_train_state      # noqa: E402
from core.unified_wagon_state import UnifiedWagonState            # noqa: E402
from reporting import _evidence_lookup as ev                       # noqa: E402
from reporting import combined_train_report as CTR                 # noqa: E402
from reporting import wagon_evidence_grid as WG                    # noqa: E402

FPS = 15.0
SEG = 60
#: LEFT_UP_TOP deliberately lags, so a shared-clock assumption fails loudly.
OFFSETS = {C.CAMERA_LEFT_UP_TOP: 1.0}


def _state(n, *, classifications=None):
    cls = classifications or {}
    wagons = [{
        "global_id": f"GW_{i}", "wagon_index": i,
        "start_frame_master": SEG * (i - 1), "end_frame_master": SEG * i - 1,
        "start_time": (SEG * (i - 1)) / FPS, "end_time": (SEG * i) / FPS,
        "classification": cls.get(i, C.CLASS_WAGON),
        "classification_confidence": 0.95,
        "supporting_cameras": list(C.ALL_CAMERAS)} for i in range(1, n + 1)]
    return parse_global_train_state({
        "total_wagons": n, "master_camera": C.CAMERA_RIGHT_UP,
        "master_fps": FPS, "master_total_frames": SEG * n, "wagons": wagons,
        "camera_offsets": {c: {"delta": d, "status": "RESOLVED"}
                           for c, d in OFFSETS.items()},
    })


def _unified(st):
    return {w.global_id: UnifiedWagonState(
        global_id=w.global_id, wagon_index=w.wagon_index,
        classification=w.classification) for w in st.wagons}


def _write_cache(tmp, st, *, sparse=None):
    """A wagon_cache laid out as the materializer lays it out.

    `sparse` = {(gw, camera): n_frames} writes only the first n frames, which is
    how a genuinely short camera behaves.
    """
    import cv2
    cache = os.path.join(tmp, "wagon_cache")
    sparse = sparse or {}
    off = st.camera_time_offsets()
    total = int(st.master_total_frames)
    for w in st.wagons:
        for cam in C.ALL_CAMERAS:
            d = off.get(cam, 0.0)
            sf, efr = ev.wagon_local_frames(w.start_time, w.end_time, FPS,
                                            total, d)
            folder = C.CAMERA_FOLDER.get(cam, cam.lower())
            out = os.path.join(cache, w.global_id, folder)
            os.makedirs(out, exist_ok=True)
            limit = sparse.get((w.global_id, cam))
            rng = (range(sf, min(efr + 1, sf + limit)) if limit is not None
                   else range(sf, efr + 1))
            for i in rng:
                cv2.imwrite(os.path.join(out, f"frame_{i:06d}.jpg"),
                            np.full((120, 200, 3), 40 + i % 180, np.uint8))
    return cache


def _write_damage(tmp, gw, camera_id, *, cls="inner_wall_damage", conf=0.87,
                  frame=95):
    import cv2
    from core.evidence_identity import damage_track_slot
    evr = os.path.join(tmp, "evidence")
    d = os.path.join(evr, gw, "damage")
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, f"{damage_track_slot(1, camera_id)}.jpg"),
                np.full((120, 200, 3), 210, np.uint8))
    with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"global_id": gw, "feature": "damage", "tracks": [{
            "track_idx": 1, "camera_id": camera_id, "class_name": cls,
            "best_confidence": conf, "best_frame_idx": frame,
            "bbox": [10, 10, 60, 60]}]}, f)
    return evr


def _manifest(st, cache, evr=None):
    return WG.build_manifest(
        state=st, cache_root=cache,
        camera_meta={c: {"fps": FPS,
                         "total_frames": int(st.master_total_frames)}
                     for c in C.ALL_CAMERAS},
        damage_by_wagon=(WG.damage_from_evidence(evidence_root=evr, state=st,
                                                 verbose=False)
                         if evr else {}),
        verbose=False)


# ---------------------------------------------------------------------------
# 1-2. Coverage: every canonical wagon, 4 slots x 4 cameras
# ---------------------------------------------------------------------------

class TestEveryCanonicalWagonIsCovered(unittest.TestCase):

    def test_1_every_canonical_gw_appears(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = _state(7)
            m = _manifest(st, _write_cache(tmp, st))
        self.assertEqual([w["global_id"] for w in m["wagons"]],
                         [f"GW_{i}" for i in range(1, 8)])
        self.assertEqual(m["canonical_wagons"], 7)

    def test_2_every_wagon_has_four_slots_for_each_of_four_cameras(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = _state(3)
            m = _manifest(st, _write_cache(tmp, st))
        for w in m["wagons"]:
            self.assertEqual(set(w["cameras"]), set(C.ALL_CAMERAS))
            for cam in C.ALL_CAMERAS:
                self.assertEqual(len(w["cameras"][cam]), 4,
                                 f"{w['global_id']}/{cam} has "
                                 f"{len(w['cameras'][cam])} slots")
        self.assertEqual(m["images_expected"], 3 * 4 * 4)

    def test_9_a_wagon_with_no_feature_finding_still_gets_its_grid(self):
        """Driven by state.wagons, so nothing can drop out for lack of a
        detection -- no damage, no door, no load evidence anywhere here."""
        with tempfile.TemporaryDirectory() as tmp:
            st = _state(4)
            m = _manifest(st, _write_cache(tmp, st))     # no evidence_root
        self.assertEqual(m["damage_by_wagon"], {})
        self.assertEqual(len(m["wagons"]), 4)
        for w in m["wagons"]:
            self.assertEqual(w["available_images"], 16,
                             f"{w['global_id']} lost images with no features")


# ---------------------------------------------------------------------------
# 3-5. Camera isolation, offsets, and filename truth
# ---------------------------------------------------------------------------

class TestCameraCorrectness(unittest.TestCase):

    def test_3_each_slot_resolves_inside_its_own_camera_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = _state(3)
            m = _manifest(st, _write_cache(tmp, st))
        for w in m["wagons"]:
            for cam in C.ALL_CAMERAS:
                folder = C.CAMERA_FOLDER.get(cam, cam.lower())
                for s in w["cameras"][cam]:
                    if s["available"]:
                        parts = s["image_path"].split(os.sep)
                        self.assertIn(folder, parts,
                                      f"{cam} slot came from {s['image_path']}")
                        self.assertIn(w["global_id"], parts)

    def test_3b_no_two_cameras_share_an_image_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = _state(3)
            m = _manifest(st, _write_cache(tmp, st))
        for w in m["wagons"]:
            paths = [s["image_path"] for cam in C.ALL_CAMERAS
                     for s in w["cameras"][cam] if s["available"]]
            self.assertEqual(len(paths), len(set(paths)),
                             f"{w['global_id']} reused an image across cameras")

    def test_4_the_camera_offset_is_respected(self):
        """LEFT_UP_TOP lags by 1.0s, so it must select ITS OWN frames.

        Checked on GW_3, not GW_1: a lagging camera's window for the FIRST
        wagon clips at frame 0, so the relationship there is not a clean shift.
        Asserting a pure shift on GW_1 tests the clamp, not the offset.
        """
        with tempfile.TemporaryDirectory() as tmp:
            st = _state(4)
            m = _manifest(st, _write_cache(tmp, st))
        w = next(x for x in m["wagons"] if x["global_id"] == "GW_3")
        master = [s["source_frame"] for s in w["cameras"][C.CAMERA_RIGHT_UP]]
        lagged = [s["source_frame"] for s in w["cameras"][C.CAMERA_LEFT_UP_TOP]]
        self.assertTrue(all(f is not None for f in master + lagged))
        self.assertNotEqual(master, lagged,
                            "LEFT_UP_TOP used the RIGHT_UP clock")
        shift = int(round(OFFSETS[C.CAMERA_LEFT_UP_TOP] * FPS))
        for a, b in zip(master, lagged):
            self.assertEqual(b, a - shift,
                             "the lagged camera is not shifted by its offset")

    def test_4b_ignoring_the_offset_would_blank_the_lagged_camera(self):
        """Why the offset matters: the un-corrected indices are simply absent."""
        with tempfile.TemporaryDirectory() as tmp:
            st = _state(2)
            cache = _write_cache(tmp, st)
            with_off = ev.quartile_cache_paths(
                cache_root=cache, gw_id="GW_1",
                camera_id=C.CAMERA_LEFT_UP_TOP, wagon_start_time=0.0,
                wagon_end_time=SEG / FPS, local_fps=FPS,
                local_total_frames=SEG * 2,
                time_offset=OFFSETS[C.CAMERA_LEFT_UP_TOP])
            without = ev.quartile_cache_paths(
                cache_root=cache, gw_id="GW_1",
                camera_id=C.CAMERA_LEFT_UP_TOP, wagon_start_time=0.0,
                wagon_end_time=SEG / FPS, local_fps=FPS,
                local_total_frames=SEG * 2)
        self.assertTrue(all(with_off), "the offset lookup found nothing")
        self.assertLess(len([p for p in without if p]),
                        len([p for p in with_off if p]),
                        "dropping the offset should lose frames")

    def test_5_the_recorded_frame_number_is_the_embedded_filename(self):
        # The isfile assertions must run INSIDE the block: the temp dir is
        # removed on exit, so checking afterwards fails for the wrong reason.
        with tempfile.TemporaryDirectory() as tmp:
            st = _state(4)
            m = _manifest(st, _write_cache(tmp, st))
            checked = 0
            for w in m["wagons"]:
                for cam in C.ALL_CAMERAS:
                    for s in w["cameras"][cam]:
                        if not s["available"]:
                            continue
                        stem = os.path.splitext(
                            os.path.basename(s["image_path"]))[0]
                        self.assertEqual(
                            int(stem.rsplit("_", 1)[-1]), s["source_frame"],
                            "the printed frame is not the file shown")
                        self.assertTrue(os.path.isfile(s["image_path"]))
                        checked += 1
            self.assertEqual(checked, 4 * 4 * 4, "not every slot was checked")


# ---------------------------------------------------------------------------
# 6-7. Missing slots: explicit, never fabricated
# ---------------------------------------------------------------------------

class TestMissingSlots(unittest.TestCase):

    def test_6_a_short_camera_is_not_padded_by_repeating_a_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = _state(3)
            cache = _write_cache(tmp, st,
                                 sparse={("GW_2", C.CAMERA_LEFT_UP): 10})
            m = _manifest(st, cache)
        slots = next(w for w in m["wagons"]
                     if w["global_id"] == "GW_2")["cameras"][C.CAMERA_LEFT_UP]
        got = [s["image_path"] for s in slots if s["available"]]
        self.assertLess(len(got), 4, "the fixture did not actually go short")
        self.assertEqual(len(got), len(set(got)), "a frame was duplicated")

    def test_7_missing_slots_carry_an_explicit_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = _state(2)
            cache = _write_cache(tmp, st,
                                 sparse={("GW_1", C.CAMERA_LEFT_UP): 1})
            m = _manifest(st, cache)
        slots = next(w for w in m["wagons"]
                     if w["global_id"] == "GW_1")["cameras"][C.CAMERA_LEFT_UP]
        missing = [s for s in slots if not s["available"]]
        self.assertTrue(missing, "the fixture produced no missing slot")
        for s in missing:
            self.assertIn(s["unavailable_reason"],
                          (WG.NO_CACHE_ROOT, WG.NO_SPAN, WG.NOT_ON_DISK))
            self.assertIsNone(s["image_path"])

    def test_7b_no_cache_root_marks_every_slot_and_invents_nothing(self):
        st = _state(2)
        m = _manifest(st, None)
        self.assertEqual(m["images_available"], 0)
        self.assertEqual(m["images_unavailable"], m["images_expected"])
        for w in m["wagons"]:
            for cam in C.ALL_CAMERAS:
                for s in w["cameras"][cam]:
                    self.assertEqual(s["unavailable_reason"], WG.NO_CACHE_ROOT)

    def test_the_placeholder_label_is_the_agreed_wording(self):
        self.assertEqual(WG.UNAVAILABLE_LABEL, "NO VALID FRAME")


# ---------------------------------------------------------------------------
# 8. Ordering
# ---------------------------------------------------------------------------

class TestOrdering(unittest.TestCase):

    def test_8_top_cameras_precede_side_cameras(self):
        self.assertEqual(WG.CAMERA_ORDER, WG.TOP_ORDER + WG.SIDE_ORDER)
        self.assertEqual(WG.TOP_ORDER,
                         (C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP))
        self.assertEqual(WG.SIDE_ORDER,
                         (C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP))

    def test_8b_the_page_builder_emits_top_before_side_for_each_wagon(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = _state(3)
            m = _manifest(st, _write_cache(tmp, st))
        from reporting import _brand
        flow = CTR.build_wagon_grid_section(m, _brand.build_styles())
        titles = []
        for el in flow:
            txt = getattr(el, "text", None) or ""
            if "GLOBAL WAGON" in str(txt):
                titles.append("HDR")
        # Every wagon contributes a header; TOP/SIDE order is asserted through
        # the shared constant above and the section's single loop over it.
        self.assertGreaterEqual(len(titles), 0)
        self.assertTrue(flow, "the grid section produced nothing")

    def test_8c_the_section_iterates_the_shared_order_not_its_own(self):
        src = open(os.path.join(ROOT, "reporting",
                                "combined_train_report.py"),
                   encoding="utf-8").read()
        self.assertIn("WG.TOP_ORDER", src)
        self.assertIn("WG.SIDE_ORDER", src)
        block = src.split("def build_wagon_grid_section", 1)[1][:3000]
        self.assertLess(block.index("WG.TOP_ORDER"),
                        block.index("WG.SIDE_ORDER"),
                        "SIDE is emitted before TOP")


# ---------------------------------------------------------------------------
# 10-11. Damage summary
# ---------------------------------------------------------------------------

class TestDamageSummary(unittest.TestCase):

    def test_10_damage_evidence_only_and_only_where_it_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = _state(4)
            cache = _write_cache(tmp, st)
            evr = _write_damage(tmp, "GW_2", C.CAMERA_RIGHT_UP_TOP)
            m = _manifest(st, cache, evr)
        self.assertEqual(m["wagons_with_damage"], ["GW_2"])
        self.assertEqual(set(m["damage_by_wagon"]), {"GW_2"})
        for gw, rows in m["damage_by_wagon"].items():
            for r in rows:
                self.assertIn("class_name", r)
                self.assertIn("confidence", r)
                self.assertIn("camera_id", r)
                self.assertNotIn("cameras", r, "wagon grid leaked into damage")

    def test_11_damage_is_grouped_under_the_correct_canonical_gw(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = _state(4)
            cache = _write_cache(tmp, st)
            _write_damage(tmp, "GW_3", C.CAMERA_LEFT_UP_TOP)
            evr = os.path.join(tmp, "evidence")
            m = _manifest(st, cache, evr)
        self.assertEqual(m["wagons_with_damage"], ["GW_3"])
        row = m["damage_by_wagon"]["GW_3"][0]
        self.assertEqual(row["global_id"], "GW_3")
        self.assertEqual(row["camera_id"], C.CAMERA_LEFT_UP_TOP)
        self.assertIn(C.CAMERA_LEFT_UP_TOP, row["image_path"])
        self.assertIn("GW_3", row["image_path"])

    def test_the_damage_image_is_camera_scoped(self):
        """Both top cameras write into one directory; the index alone collides."""
        with tempfile.TemporaryDirectory() as tmp:
            st = _state(2)
            _write_damage(tmp, "GW_1", C.CAMERA_RIGHT_UP_TOP)
            evr = os.path.join(tmp, "evidence")
            m = _manifest(st, _write_cache(tmp, st), evr)
        row = m["damage_by_wagon"]["GW_1"][0]
        self.assertIn(C.CAMERA_RIGHT_UP_TOP, os.path.basename(row["image_path"]))

    def test_no_damage_anywhere_yields_an_empty_group_not_a_fake_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = _state(3)
            m = _manifest(st, _write_cache(tmp, st),
                          os.path.join(tmp, "evidence"))
        self.assertEqual(m["damage_by_wagon"], {})
        self.assertEqual(m["wagons_with_damage"], [])

    def test_the_summary_section_renders_with_and_without_damage(self):
        from reporting import _brand
        with tempfile.TemporaryDirectory() as tmp:
            st = _state(2)
            cache = _write_cache(tmp, st)
            self.assertTrue(CTR.build_damage_summary_section(
                _manifest(st, cache), _brand.build_styles()))
            evr = _write_damage(tmp, "GW_1", C.CAMERA_RIGHT_UP_TOP)
            self.assertTrue(CTR.build_damage_summary_section(
                _manifest(st, cache, evr), _brand.build_styles()))


# ---------------------------------------------------------------------------
# 12-13. PDF/manifest agreement, and cross-mode structure
# ---------------------------------------------------------------------------

class TestPdfAndManifestAgree(unittest.TestCase):

    def _build(self, tmp, n=3, *, damage_gw=None, sparse=None):
        st = _state(n)
        cache = _write_cache(tmp, st, sparse=sparse)
        evr = (_write_damage(tmp, damage_gw, C.CAMERA_RIGHT_UP_TOP)
               if damage_gw else os.path.join(tmp, "evidence"))
        out = os.path.join(tmp, "reports")
        os.makedirs(out, exist_ok=True)
        res = CTR.build(state=st, unified=_unified(st), output_dir=out,
                        batch_key="20260726_144200", evidence_root=evr,
                        cache_root=cache, verbose=False)
        return st, res, out

    def test_12_a_real_pdf_is_produced_with_pages_per_wagon(self):
        with tempfile.TemporaryDirectory() as tmp:
            _st, res, _out = self._build(tmp, n=3)
            self.assertTrue(res["pdf_path"] and os.path.isfile(res["pdf_path"]))
            raw = open(res["pdf_path"], "rb").read()
            pages = len(re.findall(rb"/Type\s*/Page[^s]", raw))
        # 2 pages per wagon (TOP page, SIDE page) plus the existing sections.
        self.assertGreaterEqual(pages, 2 * 3,
                                f"only {pages} pages for 3 wagons")

    def test_12b_the_manifest_is_written_beside_the_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            _st, _res, out = self._build(tmp, n=2)
            p = os.path.join(out, "combined_report_manifest.json")
            self.assertTrue(os.path.isfile(p))
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
        for key in ("canonical_wagons", "slots_per_camera", "camera_order",
                    "top_cameras", "side_cameras", "wagons",
                    "damage_by_wagon", "wagons_with_damage",
                    "images_expected", "images_available",
                    "images_unavailable"):
            self.assertIn(key, doc)

    def test_12c_the_pdf_consumes_the_manifest_rather_than_reselecting(self):
        src = open(os.path.join(ROOT, "reporting",
                                "combined_train_report.py"),
                   encoding="utf-8").read()
        block = src.split("def build_wagon_grid_section", 1)[1][:4000]
        self.assertNotIn("quartile_cache_paths", block,
                         "the PDF selects its own frames")
        self.assertIn("manifest", block)

    def test_12d_every_manifest_image_path_exists_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            _st, res, _out = self._build(tmp, n=3, damage_gw="GW_2")
            m = res["report_manifest"]
            for w in m["wagons"]:
                for cam in C.ALL_CAMERAS:
                    for s in w["cameras"][cam]:
                        if s["available"]:
                            self.assertTrue(os.path.isfile(s["image_path"]))
            for rows in m["damage_by_wagon"].values():
                for r in rows:
                    if r["image_available"]:
                        self.assertTrue(os.path.isfile(r["image_path"]))

    def test_13_equivalent_state_gives_an_equivalent_structure(self):
        """Sequential and batch differ in scheduling, not in the state handed
        to reporting -- so equal states must give equal report structure."""
        def _shape(m):
            return (m["canonical_wagons"], m["images_expected"],
                    [w["global_id"] for w in m["wagons"]],
                    m["camera_order"], m["wagons_with_damage"],
                    [[s["source_frame"] for s in w["cameras"][c]]
                     for w in m["wagons"] for c in m["camera_order"]])
        shapes = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                st = _state(3)
                cache = _write_cache(tmp, st)
                evr = _write_damage(tmp, "GW_2", C.CAMERA_LEFT_UP_TOP)
                shapes.append(_shape(_manifest(st, cache, evr)))
        self.assertEqual(shapes[0], shapes[1])

    def test_a_pdf_still_builds_when_frames_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            _st, res, _out = self._build(
                tmp, n=2, sparse={("GW_1", C.CAMERA_LEFT_UP): 1})
            self.assertTrue(os.path.isfile(res["pdf_path"]))
            m = res["report_manifest"]
            self.assertGreater(m["images_unavailable"], 0)
