"""ffmpeg-based video trim/merge with GPU (NVENC) → CPU fallback."""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional, Sequence


def _run_ffmpeg(cmd: Sequence[str], timeout: int, logger: logging.Logger) -> int:
    """Run an ffmpeg command, returning its exit code. Stderr is captured."""
    try:
        result = subprocess.run(
            list(cmd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out after %ds", timeout)
        return 124
    except FileNotFoundError:
        logger.error("ffmpeg not found in PATH")
        raise


def compress_video(
    input_path: str,
    output_path: str,
    logger: logging.Logger,
    duration_sec: Optional[float] = None,
    max_size_mb: float = 50.0,
    crf: int = 26,
) -> str:
    """Re-encode ``input_path`` (e.g. a bulky mp4v-codec overlay video) to H.264,
    GPU (NVENC) first then CPU (libx264) fallback.

    Ported from the V4 Train-Inspection-Engine (`core/video_io.py`).

    Uses a constant-quality encode (``crf``) so visual quality tracks the source.
    If the result is still larger than ``max_size_mb``, re-encodes a second time
    FROM THE ORIGINAL ``input_path`` (never from the first output, which would
    compound generational loss) at a bitrate cap sized to fit under the limit.
    The overlay video has no audio track, so the whole budget goes to video.
    """
    gpu_cmd = [
        "ffmpeg", "-i", input_path,
        "-c:v", "h264_nvenc", "-preset", "p1", "-cq", str(crf),
        "-an", "-y", output_path,
    ]
    cpu_cmd = [
        "ffmpeg", "-i", input_path,
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-an", "-y", output_path,
    ]

    try:
        rc = _run_ffmpeg(gpu_cmd, timeout=600, logger=logger)
        if rc != 0:
            logger.warning("GPU compress failed (rc=%s), falling back to CPU", rc)
            rc = _run_ffmpeg(cpu_cmd, timeout=600, logger=logger)
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.warning("GPU compress raised %s, falling back to CPU", e)
        rc = _run_ffmpeg(cpu_cmd, timeout=600, logger=logger)

    if rc != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"ffmpeg compress failed with code {rc}")

    size_mb = os.path.getsize(output_path) / 1e6
    logger.info("Compressed video: %.1f MB (crf=%d)", size_mb, crf)

    if size_mb <= max_size_mb or not duration_sec or duration_sec <= 0:
        return output_path

    # Still too large -- re-encode from the ORIGINAL input at a bitrate cap sized
    # to fit under max_size_mb (5% headroom for container overhead).
    target_kbps = max(int((max_size_mb * 8192) / duration_sec * 0.95), 200)
    logger.warning(
        "Compressed video still %.1f MB > %.1f MB cap -- re-encoding at ~%dkbps",
        size_mb, max_size_mb, target_kbps,
    )
    rate_args = [
        "-b:v", f"{target_kbps}k", "-maxrate", f"{target_kbps}k",
        "-bufsize", f"{target_kbps * 2}k",
    ]
    capped_gpu_cmd = (["ffmpeg", "-i", input_path,
                       "-c:v", "h264_nvenc", "-preset", "p1"]
                      + rate_args + ["-an", "-y", output_path])
    capped_cpu_cmd = (["ffmpeg", "-i", input_path,
                       "-c:v", "libx264", "-preset", "medium"]
                      + rate_args + ["-an", "-y", output_path])
    try:
        rc = _run_ffmpeg(capped_gpu_cmd, timeout=600, logger=logger)
        if rc != 0:
            rc = _run_ffmpeg(capped_cpu_cmd, timeout=600, logger=logger)
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.warning("GPU capped compress raised %s, falling back to CPU", e)
        rc = _run_ffmpeg(capped_cpu_cmd, timeout=600, logger=logger)

    if rc != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"ffmpeg bitrate-capped compress failed with code {rc}")
    logger.info("Bitrate-capped video: %.1f MB",
                os.path.getsize(output_path) / 1e6)
    return output_path


def trim_video(
    input_path: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    output_path: str,
    logger: logging.Logger,
) -> str:
    """Trim ``input_path`` to the requested frame range.

    Attempts NVENC (GPU) first, falls back to libx264 (CPU).
    """
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps

    gpu_cmd = [
        "ffmpeg",
        "-hwaccel", "cuda",
        "-hwaccel_device", "0",
        "-i", input_path,
        "-ss", str(start_time),
        "-t", str(duration),
        "-c:v", "h264_nvenc",
        "-preset", "p1",
        "-cq", "28",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path,
        "-y",
    ]
    cpu_cmd = [
        "ffmpeg",
        "-i", input_path,
        "-ss", str(start_time),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "32",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path,
        "-y",
    ]

    try:
        rc = _run_ffmpeg(gpu_cmd, timeout=300, logger=logger)
        if rc == 0:
            logger.info("GPU trim succeeded (%.1fs)", duration)
            return output_path
        logger.warning("GPU trim failed (rc=%s), falling back to CPU", rc)
    except FileNotFoundError:
        raise
    except Exception as e:  # noqa: BLE001 — fallback is intentional
        logger.warning("GPU trim raised %s, falling back to CPU", e)

    rc = _run_ffmpeg(cpu_cmd, timeout=300, logger=logger)
    if rc != 0:
        raise RuntimeError(f"CPU ffmpeg trim failed with code {rc}")
    logger.info("CPU trim succeeded (%.1fs)", duration)
    return output_path


def merge_videos(
    video_paths: Sequence[str],
    output_path: str,
    logger: logging.Logger,
    tmp_dir: str | None = None,
) -> str:
    """Concatenate videos via the ffmpeg concat demuxer, GPU first then CPU."""
    if not video_paths:
        raise ValueError("merge_videos requires at least one input path")

    concat_dir = tmp_dir or os.path.dirname(output_path) or "."
    os.makedirs(concat_dir, exist_ok=True)
    concat_file = os.path.join(concat_dir, "concat.txt")
    with open(concat_file, "w", encoding="utf-8") as f:
        for p in video_paths:
            # ffmpeg concat demuxer needs forward slashes & escaped quotes
            f.write(f"file '{p.replace(chr(92), '/')}'\n")

    gpu_cmd = [
        "ffmpeg",
        "-hwaccel", "cuda",
        "-hwaccel_device", "0",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c:v", "h264_nvenc",
        "-preset", "p1",
        "-cq", "28",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path,
        "-y",
    ]
    cpu_cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "28",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path,
        "-y",
    ]

    try:
        rc = _run_ffmpeg(gpu_cmd, timeout=600, logger=logger)
        if rc == 0:
            logger.info("GPU merge of %d clips succeeded", len(video_paths))
            return output_path
        logger.warning("GPU merge failed (rc=%s), falling back to CPU", rc)
    except FileNotFoundError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.warning("GPU merge raised %s, falling back to CPU", e)

    rc = _run_ffmpeg(cpu_cmd, timeout=600, logger=logger)
    if rc != 0:
        raise RuntimeError(f"CPU ffmpeg merge failed with code {rc}")
    logger.info("CPU merge of %d clips succeeded", len(video_paths))

    try:
        os.remove(concat_file)
    except OSError:
        pass

    return output_path
