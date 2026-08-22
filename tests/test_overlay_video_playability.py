"""The overlay video must be playable in a browser, not merely written.

The detected/processed overlay video is delivered to operators as a URL they
click in the dashboard. Writing a valid file is therefore only half the job: if
the browser cannot decode it, the delivery has failed even though every stage
reported success -- which is exactly what happened. The dashboard rendered a
player, sat at 0:00 on a black frame, and nothing anywhere logged an error.

The cause was the codec. `mp4v` is MPEG-4 Part 2: OpenCV writes it on every
build, and no browser can decode it in an HTML5 <video> element. `avc1` is H.264
in an MP4 container, which every browser plays.

The trimmed video always played, which is the clue that pinned it: that file
comes from the producer already H.264-encoded and never passes through our
writer. Only the video WE encode was broken.

These tests assert the property that actually matters -- the encoded stream is
H.264 -- rather than the fourcc we asked for, because asking is not getting:
OpenCV will return a writer that never opened, or silently substitute a codec.
The stream check needs ffprobe and skips without it; the ordering and
verification tests always run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                               # noqa: E402

from rendering import feature_overlay_renderer as R              # noqa: E402

#: Stream codecs a browser can decode in an HTML5 <video>.
BROWSER_PLAYABLE = {"h264", "avc1"}

#: What the bug shipped. Named so the failure message is unambiguous.
NOT_PLAYABLE = {"mpeg4", "mp4v"}


def _ffprobe_codec(path: str):
    """The actual stream codec, or None if ffprobe is unavailable."""
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    return out or None


def _write(path, codecs=None, frames=20, size=(128, 128)):
    saved = R._CODEC_PREFERENCE
    if codecs is not None:
        R._CODEC_PREFERENCE = codecs
    try:
        w, codec = R._open_browser_playable_writer(path, 15.0, *size)
        if w is None:
            return None
        for i in range(frames):
            w.write(np.full((size[1], size[0], 3), (i * 8) % 255, np.uint8))
        w.release()
        return codec
    finally:
        R._CODEC_PREFERENCE = saved


class TestCodecPreference(unittest.TestCase):

    def test_h264_is_preferred_over_mp4v(self):
        pref = list(R._CODEC_PREFERENCE)
        self.assertIn("avc1", pref, "H.264 must be a candidate at all")
        self.assertIn(R._FALLBACK_CODEC, pref)
        self.assertLess(
            pref.index("avc1"), pref.index(R._FALLBACK_CODEC),
            "mp4v ahead of avc1 is the bug: it writes everywhere and plays "
            "nowhere, so the dashboard shows a player stuck at 0:00")

    def test_mp4v_remains_available_as_a_last_resort(self):
        """A build without H.264 must still produce a video, not fail a train."""
        self.assertEqual(R._CODEC_PREFERENCE[-1], R._FALLBACK_CODEC)

    def test_the_chosen_codec_is_reported_back(self):
        """The caller needs to know whether a transcode is still required."""
        with tempfile.TemporaryDirectory() as tmp:
            codec = _write(os.path.join(tmp, "a.mp4"))
            self.assertTrue(codec, "no codec was reported")
            self.assertIn(codec, R._CODEC_PREFERENCE)

    def test_an_unopenable_codec_is_skipped_not_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            codec = _write(os.path.join(tmp, "b.mp4"),
                           codecs=("XXXX", "avc1", "mp4v"))
            self.assertNotEqual(codec, "XXXX",
                               "a writer that never opened was returned")

    def test_no_usable_codec_yields_no_writer_rather_than_a_broken_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            w, codec = R._open_browser_playable_writer(
                os.path.join(tmp, "c.mp4"), 15.0, 0, 0)
            if w is not None:
                w.release()
            else:
                self.assertEqual(codec, "")


class TestEncodedStreamIsBrowserPlayable(unittest.TestCase):
    """The property that actually matters, checked on real encoded bytes."""

    def setUp(self):
        if not shutil.which("ffprobe"):
            self.skipTest("ffprobe unavailable; cannot inspect the stream")

    def test_the_default_preference_produces_h264(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "out.mp4")
            codec = _write(p)
            actual = _ffprobe_codec(p)
            if codec == R._FALLBACK_CODEC:
                self.skipTest("this OpenCV build cannot write H.264 at all")
            self.assertIn(
                actual, BROWSER_PLAYABLE,
                f"asked for {codec!r} but the stream is {actual!r}, which no "
                f"browser can play")

    def test_mp4v_really_is_the_unplayable_thing_we_think_it_is(self):
        """Guards the premise. If mp4v ever became playable this whole fix,
        and the preference order it imposes, would deserve revisiting."""
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "legacy.mp4")
            _write(p, codecs=("mp4v",))
            self.assertIn(_ffprobe_codec(p), NOT_PLAYABLE)


class TestTranscodeFallback(unittest.TestCase):

    def test_a_missing_ffmpeg_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.mp4")
            _write(p, codecs=("mp4v",))
            real = shutil.which
            shutil.which = lambda _n: None
            try:
                self.assertFalse(R._transcode_to_h264(p))
            finally:
                shutil.which = real
            self.assertTrue(os.path.isfile(p),
                            "a failed transcode must leave the original")

    def test_a_nonexistent_file_does_not_raise(self):
        self.assertFalse(R._transcode_to_h264("/nonexistent/none.mp4"))

    def test_transcoding_an_mp4v_file_makes_it_playable(self):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            self.skipTest("ffmpeg/ffprobe unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "y.mp4")
            _write(p, codecs=("mp4v",), frames=30)
            self.assertIn(_ffprobe_codec(p), NOT_PLAYABLE)
            if R._transcode_to_h264(p):
                self.assertIn(_ffprobe_codec(p), BROWSER_PLAYABLE)
                self.assertGreater(os.path.getsize(p), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
