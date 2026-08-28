"""Every picture in the global train report is reachable from the report.

The gap this closes: the document could say "GW_25 has floor damage" and give a
consumer no way to show the photograph of it. `top_damage_details` carried the
bbox and the frame index but no reference to the file, and `evidence_pages`
carried paths RELATIVE to the evidence root -- so a backend had to know the
bucket, the region and the `train_batch/<key>/evidence/` key layout, and
assemble the URL itself. Every place that has to be assembled by hand is a
place it can be assembled wrongly, and the failure mode is a broken image that
looks like the pipeline's fault.

Both are now absolute, and the relative form is kept alongside so an existing
consumer is not broken.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from _engine_harness import V4_ROOT  # noqa: F401  (path bootstrap)

from core import constants as C
from core.evidence_identity import damage_track_slot, legacy_damage_track_slot
from core.unified_wagon_state import UnifiedWagonState
from reporting import combined_train_report as CTR

RUT, LUT = C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP
BATCH = "20260724_081227"


def wagon(gw="GW_25", idx=25, details=None):
    u = UnifiedWagonState(global_id=gw, wagon_index=idx,
                          classification=C.CLASS_WAGON)
    if details is not None:
        u.top_damage = C.DAMAGE_PRESENT
        u.top_damage_details = details
    return u


def tree(tmp, files):
    for rel in files:
        p = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"\xff\xd8\xff\xd9")
    return tmp


class TestEvidenceBaseUrl(unittest.TestCase):

    def test_the_base_matches_the_stage6_upload_layout(self):
        """Stage 6 mirrors `evidence/` verbatim under
        `train_batch/<key>/evidence/`, so the base has to name exactly that --
        otherwise every link in the document is off by a prefix."""
        base = CTR.evidence_base_url(BATCH)
        self.assertIn(C.S3_OUTPUT_BUCKET, base)
        self.assertIn(C.S3_REGION, base)
        self.assertTrue(base.endswith(
            f"{C.S3_TRAIN_BATCH_PREFIX}/{BATCH}/evidence"))
        self.assertTrue(base.startswith("https://"))

    def test_it_agrees_with_the_delivery_url_builder(self):
        """Two functions building the same URL is one too many, so this pins
        them together: if either layout moves, this fails."""
        from delivery.dashboard_ingest import evidence_url
        rel = "GW_25/damage/track_1__RIGHT_UP_TOP.jpg"
        theirs = evidence_url(C.S3_OUTPUT_BUCKET, C.S3_REGION, BATCH, rel)
        mine = f"{CTR.evidence_base_url(BATCH)}/{rel}"
        self.assertEqual(mine, theirs)

    def test_the_base_changes_with_the_batch(self):
        self.assertNotEqual(CTR.evidence_base_url("A"),
                            CTR.evidence_base_url("B"))


class TestDamageRowsLinkToTheirOwnPhoto(unittest.TestCase):

    def test_a_damage_row_gets_a_path_and_an_absolute_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree(tmp, [f"GW_25/damage/{damage_track_slot(1, RUT)}.jpg"])
            u = wagon(details=[{"track_idx": 1, "camera_id": RUT}])
            base = CTR.evidence_base_url(BATCH)
            CTR._attach_damage_photos([u], tmp, base)
            row = u.top_damage_details[0]
            self.assertEqual(row["evidence_path"],
                             f"GW_25/damage/track_1__{RUT}.jpg")
            self.assertEqual(row["evidence_url"],
                             f"{base}/GW_25/damage/track_1__{RUT}.jpg")

    def test_the_name_is_camera_scoped(self):
        """The two top cameras write into ONE directory and photograph the same
        roof from opposite sides, so the index alone collides and a mix-up
        renders as a plausible picture of the wrong camera."""
        with tempfile.TemporaryDirectory() as tmp:
            tree(tmp, [f"GW_25/damage/{damage_track_slot(1, RUT)}.jpg",
                       f"GW_25/damage/{damage_track_slot(1, LUT)}.jpg"])
            u = wagon(details=[{"track_idx": 1, "camera_id": RUT},
                               {"track_idx": 1, "camera_id": LUT}])
            CTR._attach_damage_photos([u], tmp, CTR.evidence_base_url(BATCH))
            got = [r["evidence_path"] for r in u.top_damage_details]
            self.assertEqual(len(set(got)), 2, got)
            self.assertIn(RUT, got[0])
            self.assertIn(LUT, got[1])

    def test_a_missing_file_gets_no_link_rather_than_a_broken_one(self):
        """A missing key is something a consumer can handle. A URL that 404s
        is not."""
        with tempfile.TemporaryDirectory() as tmp:
            u = wagon(details=[{"track_idx": 9, "camera_id": RUT}])
            CTR._attach_damage_photos([u], tmp, CTR.evidence_base_url(BATCH))
            row = u.top_damage_details[0]
            self.assertNotIn("evidence_path", row)
            self.assertNotIn("evidence_url", row)

    def test_the_legacy_unscoped_name_still_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree(tmp, [f"GW_25/damage/{legacy_damage_track_slot(1)}.jpg"])
            u = wagon(details=[{"track_idx": 1, "camera_id": RUT}])
            CTR._attach_damage_photos([u], tmp, CTR.evidence_base_url(BATCH))
            self.assertEqual(u.top_damage_details[0]["evidence_path"],
                             "GW_25/damage/track_1.jpg")

    def test_a_row_without_an_index_or_camera_is_skipped(self):
        """`track_idx` and `camera_id` together identify the picture; neither
        alone does, so a row missing either gets no link."""
        with tempfile.TemporaryDirectory() as tmp:
            tree(tmp, [f"GW_25/damage/{damage_track_slot(1, RUT)}.jpg"])
            u = wagon(details=[{"camera_id": RUT},
                               {"track_idx": 1},
                               {"track_idx": None, "camera_id": RUT}])
            CTR._attach_damage_photos([u], tmp, CTR.evidence_base_url(BATCH))
            for row in u.top_damage_details:
                self.assertNotIn("evidence_url", row)

    def test_no_evidence_root_is_survivable(self):
        u = wagon(details=[{"track_idx": 1, "camera_id": RUT}])
        CTR._attach_damage_photos([u], None, CTR.evidence_base_url(BATCH))
        self.assertNotIn("evidence_url", u.top_damage_details[0])

    def test_a_wagon_with_no_damage_is_untouched(self):
        u = wagon()
        CTR._attach_damage_photos([u], None, CTR.evidence_base_url(BATCH))
        self.assertEqual(u.top_damage_details, [])


class TestTheDocumentCarriesBothForms(unittest.TestCase):

    def _doc(self, tmp):
        from core.global_state_loader import GlobalTrainState, GlobalWagon
        gw = GlobalWagon(global_id="GW_25", wagon_index=25,
                         start_frame_master=0, end_frame_master=59,
                         start_time=0.0, end_time=4.0,
                         classification=C.CLASS_WAGON)
        state = GlobalTrainState(total_wagons=1, wagons=(gw,),
                                 master_camera=C.CAMERA_RIGHT_UP,
                                 master_fps=15.0, master_total_frames=60)
        u = wagon(details=[{"track_idx": 1, "camera_id": RUT}])
        return CTR._build_json(state=state, unified={"GW_25": u},
                               batch_key=BATCH, evidence_root=tmp)

    def test_the_document_states_the_base_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree(tmp, ["GW_25/door/left_best.jpg"])
            doc = self._doc(tmp)
            self.assertEqual(doc["evidence_base_url"],
                             CTR.evidence_base_url(BATCH))

    def test_relative_paths_are_kept_so_existing_consumers_still_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree(tmp, ["GW_25/door/left_best.jpg"])
            doc = self._doc(tmp)
            self.assertEqual(doc["evidence_pages"]["GW_25"]["door_left_best"],
                             "GW_25/door/left_best.jpg")

    def test_absolute_urls_are_provided_alongside(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree(tmp, ["GW_25/door/left_best.jpg"])
            doc = self._doc(tmp)
            url = doc["evidence_page_urls"]["GW_25"]["door_left_best"]
            self.assertTrue(url.startswith("https://"))
            self.assertTrue(url.endswith("GW_25/door/left_best.jpg"))

    def test_the_two_forms_describe_the_same_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree(tmp, ["GW_25/door/left_best.jpg", "GW_25/ocr/best_frame.jpg"])
            doc = self._doc(tmp)
            self.assertEqual(set(doc["evidence_pages"]["GW_25"]),
                             set(doc["evidence_page_urls"]["GW_25"]))
            base = doc["evidence_base_url"]
            for k, rel in doc["evidence_pages"]["GW_25"].items():
                self.assertEqual(doc["evidence_page_urls"]["GW_25"][k],
                                 f"{base}/{rel}")

    def test_the_damage_row_in_the_document_carries_its_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree(tmp, [f"GW_25/damage/{damage_track_slot(1, RUT)}.jpg"])
            doc = self._doc(tmp)
            row = doc["wagons"][0]["top_damage_details"][0]
            self.assertIn("evidence_url", row)
            self.assertTrue(row["evidence_url"].startswith(
                doc["evidence_base_url"]))

    def test_a_document_with_no_evidence_still_builds(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = self._doc(tmp)
            self.assertEqual(doc["evidence_pages"], {})
            self.assertEqual(doc["evidence_page_urls"], {})
            self.assertIn("evidence_base_url", doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
