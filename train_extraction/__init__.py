"""Train-extraction producer package (vendored from the V4 Train-Inspection-Engine).

Reads RAW CCTV video from S3, cuts out the train pass, and uploads trimmed
per-camera clips to a "trimmed" bucket -- the bucket `wagon_eye_v4_new`'s
`--auto` orchestrator polls.  It performs NO inspection.

Core algorithm modules (classifier, direction, extractor, s3, segment_finder,
state, model_store, time_utils, url_utils, video_io) are copied VERBATIM from
the V1 `v4-pipeline` train_extraction package; only `driver.py` (generalized to
all four cameras) and `run_extraction_service.py` (the continuous runner) are
new.  See README.md for how it wires to the inspection pipeline.
"""

LAZY_EXPORTS = {
    "extract_trains": ".driver",
    "get_extractor": ".driver",
    "ALL_CAMERAS": ".driver",
}

__all__ = list(LAZY_EXPORTS)


def __getattr__(name):
    """Resolve the driver exports ON FIRST USE (PEP 562).

    `driver` pulls in boto3 + ultralytics, which only the PRODUCER path needs.
    Importing them eagerly here meant that any consumer of a dependency-free
    sibling -- notably `train_extraction.video_io.compress_video`, used by the
    delivery stage to re-encode overlay videos -- would fail on a box without the
    extraction dependencies.  Deferring keeps `from train_extraction import
    extract_trains` working while letting `import train_extraction.video_io`
    stand alone.
    """
    module_path = LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(module_path, __name__), name)
