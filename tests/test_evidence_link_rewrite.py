"""The report's evidence links come from the uploads, not from local config.

The ordering this pins down. Stage 5 writes the report; Stage 6 uploads the
evidence. In `s3` mode the key is `train_batch/<key>/evidence/<rel>`, which
Stage 5 can compute, so the report could be written -- and uploaded -- before the
files it links to existed. Through the Artifact Upload API the BACKEND chooses
the key: there is no base to prefix and no path to predict, so the only correct
URL is the one that file's own upload returned.

Hence: evidence uploads first, then the report is rewritten from the result,
then the report is uploaded. And in `api` mode `evidence_base_url` is removed,
because a consumer joining it to a relative path would build a plausible URL
pointing at nothing.
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from delivery import artifact_uploader as AU
from delivery import evidence_links as EL

REL_DAMAGE = "GW_25/damage/track_1__RIGHT_UP_TOP.jpg"
REL_DOOR = "GW_25/door/left_best.jpg"
BASE = "https://local-bucket.s3.ap-south-1.amazonaws.com/train_batch/K/evidence"
API_DAMAGE = "https://backend-bucket.s3.ap-south-1.amazonaws.com/chosen/a1b2.jpg"
API_DOOR = "https://backend-bucket.s3.ap-south-1.amazonaws.com/chosen/c3d4.jpg"


def a_report(tmp, *, with_base=True):
    doc = {
        "schema": "wagon_eye.combined_train_report.v1",
        "batch_key": "K",
        "wagons": [{
            "global_id": "GW_25", "wagon_index": 25,
            "top_damage": "DAMAGE",
            "top_damage_details": [{
                "track_idx": 1, "camera_id": "RIGHT_UP_TOP",
                "evidence_path": REL_DAMAGE,
                "evidence_url": f"{BASE}/{REL_DAMAGE}"}]}],
        "evidence_pages": {"GW_25": {"damage_track_1__RIGHT_UP_TOP": REL_DAMAGE,
                                     "door_left_best": REL_DOOR}},
        "evidence_page_urls": {"GW_25": {
            "damage_track_1__RIGHT_UP_TOP": f"{BASE}/{REL_DAMAGE}",
            "door_left_best": f"{BASE}/{REL_DOOR}"}},
    }
    if with_base:
        doc["evidence_base_url"] = BASE
    p = os.path.join(tmp, "combined_train_report.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return p


def read(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


API_MAP = {REL_DAMAGE: API_DAMAGE, REL_DOOR: API_DOOR}


class TestApiModeUsesTheReturnedUrls(unittest.TestCase):

    def test_each_link_becomes_the_url_its_own_upload_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = a_report(tmp)
            EL.rewrite(report_json_path=p, url_map=API_MAP,
                       mode=AU.MODE_API, verbose=False)
            doc = read(p)
            self.assertEqual(
                doc["evidence_page_urls"]["GW_25"]
                   ["damage_track_1__RIGHT_UP_TOP"], API_DAMAGE)
            self.assertEqual(
                doc["evidence_page_urls"]["GW_25"]["door_left_best"], API_DOOR)
            self.assertEqual(
                doc["wagons"][0]["top_damage_details"][0]["evidence_url"],
                API_DAMAGE)

    def test_the_computed_base_is_removed(self):
        """Keeping it would be worse than useless: a consumer joining it to a
        relative path builds a plausible URL pointing nowhere."""
        with tempfile.TemporaryDirectory() as tmp:
            p = a_report(tmp)
            res = EL.rewrite(report_json_path=p, url_map=API_MAP,
                             mode=AU.MODE_API, verbose=False)
            self.assertTrue(res.base_url_removed)
            self.assertNotIn("evidence_base_url", read(p))

    def test_the_document_says_where_its_urls_came_from(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = a_report(tmp)
            EL.rewrite(report_json_path=p, url_map=API_MAP,
                       mode=AU.MODE_API, verbose=False)
            self.assertEqual(read(p)["evidence_url_source"], EL.SOURCE_API)

    def test_no_stale_computed_url_survives(self):
        """The pre-rewrite document carried computed links. If any survived, a
        reader could not tell which links to trust."""
        with tempfile.TemporaryDirectory() as tmp:
            p = a_report(tmp)
            EL.rewrite(report_json_path=p, url_map=API_MAP,
                       mode=AU.MODE_API, verbose=False)
            self.assertNotIn("local-bucket", json.dumps(read(p)))


class TestAFileWithNoUploadedUrl(unittest.TestCase):
    """A missing key is a fact a consumer can handle; a URL that 404s is not."""

    def test_an_unresolved_page_link_is_dropped_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = a_report(tmp)
            res = EL.rewrite(report_json_path=p,
                             url_map={REL_DAMAGE: API_DAMAGE},
                             mode=AU.MODE_API, verbose=False)
            doc = read(p)
            self.assertNotIn("door_left_best",
                             doc["evidence_page_urls"]["GW_25"])
            self.assertEqual(res.page_urls_set, 1)
            self.assertTrue(any(REL_DOOR in u for u in res.unresolved))

    def test_an_unresolved_damage_link_is_removed_not_left_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = a_report(tmp)
            EL.rewrite(report_json_path=p, url_map={}, mode=AU.MODE_API,
                       verbose=False)
            row = read(p)["wagons"][0]["top_damage_details"][0]
            self.assertNotIn("evidence_url", row)
            self.assertEqual(row["evidence_path"], REL_DAMAGE,
                             "the path is evidence and must survive")

    def test_an_empty_map_yields_no_links_rather_than_wrong_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = a_report(tmp)
            res = EL.rewrite(report_json_path=p, url_map={},
                             mode=AU.MODE_API, verbose=False)
            self.assertEqual(read(p)["evidence_page_urls"], {})
            self.assertEqual(res.page_urls_set, 0)
            self.assertEqual(len(res.unresolved), 3)


class TestS3ModeIsUnchanged(unittest.TestCase):
    """Keeping S3 the default means the delivered contract must not move."""

    def test_the_base_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = a_report(tmp)
            res = EL.rewrite(report_json_path=p, url_map={},
                             mode=AU.MODE_S3, verbose=False)
            self.assertFalse(res.base_url_removed)
            self.assertEqual(read(p)["evidence_base_url"], BASE)

    def test_the_source_is_recorded_as_computed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = a_report(tmp)
            EL.rewrite(report_json_path=p, url_map={}, mode=AU.MODE_S3,
                       verbose=False)
            self.assertEqual(read(p)["evidence_url_source"], EL.SOURCE_COMPUTED)

    def test_a_real_s3_map_reproduces_the_computed_urls(self):
        """In S3 mode the returned URL and the computed one are the same string,
        so the rewrite is a confirmation rather than a change."""
        s3_map = {REL_DAMAGE: f"{BASE}/{REL_DAMAGE}",
                  REL_DOOR: f"{BASE}/{REL_DOOR}"}
        with tempfile.TemporaryDirectory() as tmp:
            p = a_report(tmp)
            before = read(p)
            EL.rewrite(report_json_path=p, url_map=s3_map, mode=AU.MODE_S3,
                       verbose=False)
            after = read(p)
            self.assertEqual(before["evidence_page_urls"],
                             after["evidence_page_urls"])
            self.assertEqual(
                before["wagons"][0]["top_damage_details"][0]["evidence_url"],
                after["wagons"][0]["top_damage_details"][0]["evidence_url"])


class TestItNeverFailsADelivery(unittest.TestCase):

    def test_a_missing_report_is_reported_not_raised(self):
        res = EL.rewrite(report_json_path="/nope.json", url_map={},
                         mode=AU.MODE_API, verbose=False)
        self.assertFalse(res.rewritten)
        self.assertIn("no report", res.error)

    def test_a_corrupt_report_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "combined_train_report.json")
            with open(p, "w", encoding="utf-8") as f:
                f.write("{not json")
            res = EL.rewrite(report_json_path=p, url_map={},
                             mode=AU.MODE_API, verbose=False)
            self.assertFalse(res.rewritten)
            self.assertTrue(res.error)

    def test_a_report_with_no_evidence_at_all_is_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "combined_train_report.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"batch_key": "K", "wagons": []}, f)
            res = EL.rewrite(report_json_path=p, url_map={},
                             mode=AU.MODE_API, verbose=False)
            self.assertTrue(res.rewritten)
            self.assertEqual(read(p)["evidence_page_urls"], {})


class TestDeliveryUploadsEvidenceBeforeTheReport(unittest.TestCase):
    """The reorder is the whole point: a report uploaded before its evidence
    cannot carry the URLs the uploads returned."""

    @staticmethod
    def _lines():
        return open(os.path.join(V4_ROOT, "delivery/finalize.py"),
                    encoding="utf-8").read().splitlines()

    def _line_of(self, needle):
        for i, l in enumerate(self._lines()):
            if needle in l and not l.strip().startswith("#"):
                return i
        return None

    def test_evidence_goes_up_before_the_report(self):
        ev = self._line_of("sub_prefix=\"evidence\"")
        rep = self._line_of("s3_upload.upload_json(")
        self.assertIsNotNone(ev)
        self.assertIsNotNone(rep)
        self.assertLess(ev, rep)

    def test_the_rewrite_happens_between_them(self):
        ev = self._line_of("sub_prefix=\"evidence\"")
        rw = self._line_of("evidence_links.rewrite(")
        rep = self._line_of("s3_upload.upload_json(")
        self.assertLess(ev, rw)
        self.assertLess(rw, rep)

    def test_evidence_is_not_in_the_generic_tree_loop(self):
        """It was a `("evidence", ...)` entry in the archive loop. Leaving it
        there while also uploading it first would send every frame twice.

        Checked on the LOOP'S TUPLE LIST, not by counting the string
        "evidence" -- that appears six times for unrelated reasons and a count
        would pass or fail for none of them.
        """
        src = open(os.path.join(V4_ROOT, "delivery/finalize.py"),
                   encoding="utf-8").read()
        # Identified by its unpack signature `label, path, extra`. There is a
        # SECOND tuple loop in `find_artifacts` that lists the same directory
        # names for DISCOVERY, and it legitimately still contains "evidence" --
        # matching every tuple loop would fail on that one and say nothing about
        # uploading.
        labels = []
        for n in ast.walk(ast.parse(src)):
            if not isinstance(n, ast.For):
                continue
            tgt = n.target
            names = ([e.id for e in tgt.elts if isinstance(e, ast.Name)]
                     if isinstance(tgt, ast.Tuple) else [])
            if names != ["label", "path", "extra"]:
                continue
            it = n.iter
            if not isinstance(it, (ast.Tuple, ast.List)):
                continue
            for elt in it.elts:
                if (isinstance(elt, ast.Tuple) and elt.elts
                        and isinstance(elt.elts[0], ast.Constant)):
                    labels.append(elt.elts[0].value)
        self.assertTrue(labels, "the archive loop was not found")
        self.assertNotIn("evidence", labels,
                         f"evidence is still in the tree loop: {labels}")
        # The subtrees that legitimately remain.
        for expected in ("global_state", "wagon_states", "reports",
                         "processed_videos"):
            self.assertIn(expected, labels)

    def test_only_one_call_uploads_the_evidence_subtree(self):
        src = open(os.path.join(V4_ROOT, "delivery/finalize.py"),
                   encoding="utf-8").read()
        n_evidence = 0
        for n in ast.walk(ast.parse(src)):
            if not isinstance(n, ast.Call):
                continue
            name = getattr(n.func, "attr", getattr(n.func, "id", ""))
            if name not in ("upload_tree", "upload_tree_detailed"):
                continue
            for kw in n.keywords:
                if (kw.arg == "sub_prefix"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value == "evidence"):
                    n_evidence += 1
        self.assertEqual(n_evidence, 1)

    def test_delivery_records_what_the_rewrite_did(self):
        from delivery.finalize import DeliveryResult
        self.assertIn("evidence_links", DeliveryResult.__dataclass_fields__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
