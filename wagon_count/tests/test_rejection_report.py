"""The rejection table must report what the pipeline decided, not re-decide it.

This diagnostic is how an under-count gets diagnosed on EC2, where the train's
video exists and this dev machine's does not. So the table is only trustworthy if
it (a) reads the pipeline's own record rather than recomputing anything, (b) keys
train state off the real wagon window, and (c) never silently drops a candidate.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gap_validation as gval
import rejection_report as rr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def feat(track_id, frame_start, frame_end, **kw):
    f = {
        "track_id": track_id, "camera_id": "RIGHT_UP",
        "frame_start": frame_start, "frame_end": frame_end,
        "time_start": frame_start / 15.0, "time_end": frame_end / 15.0,
        "duration_s": (frame_end - frame_start) / 15.0,
        "track_frames": frame_end - frame_start + 1,
        "hits": frame_end - frame_start + 1, "coverage": 1.0,
        "max_detection_gap": 1, "center_start": 700.0, "center_end": 250.0,
        "displacement_px": -450.0, "abs_displacement_px": 450.0,
        "velocity_px_per_sec": 500.0, "direction": -1,
        "monotonic_fraction": 1.0, "n_steps": 12, "step_velocity_median": -37.0,
        "mean_confidence": 0.9, "min_confidence": 0.8,
        "bbox_height_median": 200.0, "bbox_width_median": 40.0,
        "path_efficiency": 1.0, "motion_reference_speed": 520.0,
        "motion_reference_kind": "local", "motion_paused": False,
    }
    f.update(kw)
    return f


def rejection(track_id, frame_start, frame_end, reason, **kw):
    return {
        "reason": reason, "detail": f"{reason} detail",
        "hard": reason in gval.HARD_REJECTION_REASONS,
        "soft": reason in gval.SOFT_REJECTION_REASONS,
        "features": feat(track_id, frame_start, frame_end, **kw),
    }


def state(rejections=None, *, window=(200, 900), recovery=None, gaps=8):
    return {
        "schema": "wagon_eye.global_train_state.v1",
        "master_camera": "RIGHT_UP",
        "wagon_window": {"found": True, "wagon_start_frame": window[0],
                         "wagon_end_frame": window[1], "master_wagon_count": gaps + 1},
        "master_wagon_count": gaps + 1,
        "global_gaps": [{"camera_id": "RIGHT_UP", "time_start": 20.0 + 4.0 * i}
                        for i in range(gaps)],
        "gap_rejection_details": {"RIGHT_UP": list(rejections or [])},
        "wagon_active_recovery": recovery or {},
    }


# ===========================================================================
# every rejected candidate appears -- nothing is silently dropped
# ===========================================================================

class TestCompleteness(unittest.TestCase):
    def test_every_rejection_becomes_exactly_one_row(self):
        rejs = [rejection(i, 300 + i * 40, 312 + i * 40, gval.REJECTED_STATIC)
                for i in range(7)]
        rows = rr.rejection_rows(state(rejs))
        self.assertEqual(len(rows), 7)
        self.assertEqual(sorted(r["track_id"] for r in rows), list(range(7)))

    def test_rows_from_all_cameras_are_reported(self):
        s = state([rejection(1, 300, 312, gval.REJECTED_STATIC)])
        s["gap_rejection_details"]["LEFT_UP"] = [
            rejection(2, 320, 332, gval.REJECTED_LOW_MOTION)]
        rows = rr.rejection_rows(s)
        self.assertEqual({r["camera"] for r in rows}, {"RIGHT_UP", "LEFT_UP"})

    def test_no_rejections_renders_without_crashing(self):
        buf = io.StringIO()
        rows = rr.rejection_rows(state([]))
        rr.render_table(rows, buf)
        rr.render_summary(state([]), rows, buf)
        self.assertIn("No rejected candidates", buf.getvalue())

    def test_every_required_column_is_present(self):
        """The user enumerated the columns; each must be in the row."""
        rows = rr.rejection_rows(state([rejection(1, 300, 340,
                                                  gval.REJECTED_IMPLAUSIBLE_SPEED)]))
        for col in ("track_id", "frames", "duration_s", "confidence",
                    "displacement_px", "speed_px_s", "ref_speed_px_s",
                    "direction", "monotonic", "reason", "train_state",
                    "nearest_prev_s", "nearest_next_s"):
            self.assertIn(col, rows[0], f"missing required column {col}")


# ===========================================================================
# train state comes from the real window
# ===========================================================================

class TestTrainState(unittest.TestCase):
    def test_states_split_on_the_window_boundaries(self):
        rejs = [rejection(1, 50, 62, gval.REJECTED_STATIC),      # before
                rejection(2, 500, 512, gval.REJECTED_STATIC),    # inside
                rejection(3, 950, 962, gval.REJECTED_STATIC)]    # after
        rows = {r["track_id"]: r["train_state"]
                for r in rr.rejection_rows(state(rejs, window=(200, 900)))}
        self.assertEqual(rows[1], rr.PRE_WAGON)
        self.assertEqual(rows[2], rr.WAGON_ACTIVE)
        self.assertEqual(rows[3], rr.POST_WAGON)

    def test_boundary_frames_are_inside_the_window(self):
        self.assertEqual(rr.train_state_at(200, 200, 900), rr.WAGON_ACTIVE)
        self.assertEqual(rr.train_state_at(900, 200, 900), rr.WAGON_ACTIVE)
        self.assertEqual(rr.train_state_at(199, 200, 900), rr.PRE_WAGON)
        self.assertEqual(rr.train_state_at(901, 200, 900), rr.POST_WAGON)

    def test_missing_window_is_unknown_not_wagon_active(self):
        """Absent a window we must not claim candidates were in the count."""
        s = state([rejection(1, 500, 512, gval.REJECTED_STATIC)])
        s["wagon_window"] = {"found": False}
        rows = rr.rejection_rows(s)
        self.assertEqual(rows[0]["train_state"], rr.UNKNOWN_STATE)

    def test_window_is_read_from_the_real_key_names(self):
        """Guards against the reader drifting from WagonWindow.summary()."""
        import train_structure as ts
        keys = set(ts.WagonWindow().summary())
        self.assertIn("wagon_start_frame", keys)
        self.assertIn("wagon_end_frame", keys)
        self.assertEqual(
            rr.wagon_window_frames({"wagon_window": {"wagon_start_frame": 5,
                                                     "wagon_end_frame": 9}}),
            (5, 9))


# ===========================================================================
# HARD / SOFT classification agrees with the validator, not a local copy
# ===========================================================================

class TestClassification(unittest.TestCase):
    def test_hard_and_soft_are_labelled_from_the_record(self):
        rejs = [rejection(1, 300, 312, gval.REJECTED_STATIC),
                rejection(2, 400, 412, gval.REJECTED_IMPLAUSIBLE_SPEED)]
        rows = {r["track_id"]: r["class"] for r in rr.rejection_rows(state(rejs))}
        self.assertEqual(rows[1], "HARD")
        self.assertEqual(rows[2], "SOFT")

    def test_older_json_without_flags_is_classified_by_reason(self):
        r = rejection(1, 300, 312, gval.REJECTED_LOW_CONFIDENCE)
        del r["hard"], r["soft"]
        rows = rr.rejection_rows(state([r]))
        self.assertEqual(rows[0]["class"], "SOFT")

    def test_every_validator_reason_classifies_as_hard_or_soft(self):
        """An unclassified reason would render '?' and hide a real failure."""
        for reason in (gval.HARD_REJECTION_REASONS | gval.SOFT_REJECTION_REASONS):
            r = rejection(1, 300, 312, reason)
            del r["hard"], r["soft"]
            rows = rr.rejection_rows(state([r]))
            self.assertIn(rows[0]["class"], ("HARD", "SOFT"), reason)


# ===========================================================================
# recovery outcome and the "lost wagon" call-out
# ===========================================================================

class TestRecoveryColumn(unittest.TestCase):
    def _with_recovery(self, outcome, note=""):
        rec = {"RIGHT_UP": {"camera_id": "RIGHT_UP", "details": [
            {"track_id": 1, "outcome": outcome, "note": note}]}}
        return state([rejection(1, 500, 540, gval.REJECTED_IMPLAUSIBLE_SPEED)],
                     recovery=rec)

    def test_recovered_candidate_is_marked_recovered(self):
        rows = rr.rejection_rows(self._with_recovery("recovered"))
        self.assertEqual(rows[0]["recovery"], "recovered")

    def test_blocked_candidate_reports_what_blocked_it(self):
        rows = rr.rejection_rows(
            self._with_recovery("blocked", "path efficiency 0.13 < floor 0.4"))
        self.assertIn("blocked", rows[0]["recovery"])
        self.assertIn("path efficiency", rows[0]["recovery"])

    def test_no_recovery_record_shows_a_dash(self):
        rows = rr.rejection_rows(
            state([rejection(1, 500, 540, gval.REJECTED_IMPLAUSIBLE_SPEED)]))
        self.assertEqual(rows[0]["recovery"], "-")

    def test_unrecovered_soft_inside_window_is_flagged_loudly(self):
        """This is the under-count signature; it must not be buried."""
        buf = io.StringIO()
        s = self._with_recovery("blocked", "monotonic 0.20 < floor 0.45")
        rr.render_summary(s, rr.rejection_rows(s), buf)
        self.assertIn("NOT", buf.getvalue())
        self.assertIn("track 1", buf.getvalue())

    def test_recovered_soft_is_not_flagged_as_lost(self):
        buf = io.StringIO()
        s = self._with_recovery("recovered")
        rr.render_summary(s, rr.rejection_rows(s), buf)
        self.assertIn("No unrecovered SOFT rejections", buf.getvalue())

    def test_hard_rejection_inside_window_is_not_flagged_as_lost(self):
        """Hard artefacts inside the window are correct behaviour, not a loss."""
        buf = io.StringIO()
        s = state([rejection(1, 500, 540, gval.REJECTED_STATIC)])
        rr.render_summary(s, rr.rejection_rows(s), buf)
        self.assertIn("No unrecovered SOFT rejections", buf.getvalue())


# ===========================================================================
# nearest accepted neighbours -- the columns that reveal a doubled interval
# ===========================================================================

class TestNearestNeighbours(unittest.TestCase):
    def test_neighbours_bracket_the_rejected_candidate(self):
        # accepted at 20, 24, 28 ... ; candidate at 26.0s (frame 390)
        rows = rr.rejection_rows(
            state([rejection(1, 390, 402, gval.REJECTED_IMPLAUSIBLE_SPEED)]))
        self.assertEqual(rows[0]["nearest_prev_s"], 24.0)
        self.assertEqual(rows[0]["nearest_next_s"], 28.0)

    def test_candidate_before_all_accepts_has_no_previous(self):
        rows = rr.rejection_rows(
            state([rejection(1, 30, 42, gval.REJECTED_STATIC)]))
        self.assertIsNone(rows[0]["nearest_prev_s"])
        self.assertIsNotNone(rows[0]["nearest_next_s"])

    def test_candidate_after_all_accepts_has_no_next(self):
        rows = rr.rejection_rows(
            state([rejection(1, 3000, 3012, gval.REJECTED_STATIC)]))
        self.assertIsNotNone(rows[0]["nearest_prev_s"])
        self.assertIsNone(rows[0]["nearest_next_s"])


# ===========================================================================
# the script itself: runs, is deterministic, and reads only the JSON
# ===========================================================================

class TestCli(unittest.TestCase):
    def _write(self, s):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(s, fh)
        self.addCleanup(os.unlink, path)
        return path

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, os.path.join(ROOT, "rejection_report.py"), *args],
            capture_output=True, text=True, cwd=ROOT)

    def test_runs_on_a_realistic_state(self):
        path = self._write(state([
            rejection(1, 500, 540, gval.REJECTED_IMPLAUSIBLE_SPEED),
            rejection(2, 100, 130, gval.REJECTED_STATIC)]))
        p = self._run(path)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("CAMERA RIGHT_UP", p.stdout)
        self.assertIn("SUMMARY", p.stdout)

    def test_missing_file_fails_cleanly(self):
        p = self._run(os.path.join(ROOT, "definitely-not-here.json"))
        self.assertEqual(p.returncode, 2)
        self.assertIn("no such run JSON", p.stderr)

    def test_malformed_json_fails_cleanly(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self.addCleanup(os.unlink, path)
        p = self._run(path)
        self.assertEqual(p.returncode, 2)
        self.assertIn("not valid JSON", p.stderr)

    def test_filters_narrow_the_rows(self):
        path = self._write(state([
            rejection(1, 500, 540, gval.REJECTED_IMPLAUSIBLE_SPEED),
            rejection(2, 100, 130, gval.REJECTED_STATIC)]))
        soft = self._run(path, "--soft-only").stdout
        self.assertIn("IMPLAUSIBLE", soft)
        self.assertNotIn("REJECTED_STATIC", soft.split("SUMMARY")[0])
        active = self._run(path, "--wagon-active-only").stdout
        self.assertNotIn("REJECTED_STATIC", active.split("SUMMARY")[0])

    def test_csv_round_trips_every_row(self):
        import csv as _csv
        path = self._write(state([
            rejection(1, 500, 540, gval.REJECTED_IMPLAUSIBLE_SPEED),
            rejection(2, 100, 130, gval.REJECTED_STATIC)]))
        out = os.path.join(tempfile.mkdtemp(), "rej.csv")
        p = self._run(path, "--csv", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(out, encoding="utf-8") as fh:
            got = list(_csv.DictReader(fh))
        self.assertEqual(len(got), 2)
        self.assertEqual({r["track_id"] for r in got}, {"1", "2"})

    def test_output_is_deterministic(self):
        path = self._write(state([
            rejection(3, 700, 740, gval.REJECTED_LOW_MOTION),
            rejection(1, 500, 540, gval.REJECTED_IMPLAUSIBLE_SPEED),
            rejection(2, 100, 130, gval.REJECTED_STATIC)]))
        self.assertEqual(self._run(path).stdout, self._run(path).stdout)

    def test_rows_are_ordered_by_camera_then_frame(self):
        rows = rr.rejection_rows(state([
            rejection(3, 700, 740, gval.REJECTED_LOW_MOTION),
            rejection(1, 500, 540, gval.REJECTED_IMPLAUSIBLE_SPEED),
            rejection(2, 100, 130, gval.REJECTED_STATIC)]))
        self.assertEqual([r["frame_start"] for r in rows], [100, 500, 700])

    def test_no_train_specific_constants_in_the_script(self):
        """This diagnostic runs on ANY train; it must carry no local tuning."""
        with open(os.path.join(ROOT, "rejection_report.py"), encoding="utf-8") as fh:
            body = "\n".join(l for l in fh if not l.strip().startswith("#"))
        for token in ("848", "15.0", "= 52", "= 53", "frame 198"):
            self.assertNotIn(token, body.split('"""')[-1],
                             f"train/camera-specific constant {token!r} leaked in")


if __name__ == "__main__":
    unittest.main(verbosity=2)
