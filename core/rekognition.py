"""AWS Rekognition client wrapper used for wagon-number plate OCR.

Ported from the V4 Train-Inspection-Engine (`core/rekognition.py`) so the
global_train pipeline reads wagon numbers through the SAME managed OCR service
the V4 per-camera pipelines use, instead of a local EasyOCR model.

Credential conventions mirror the rest of this package (and `delivery/s3_upload`):
explicit keys when supplied, otherwise boto3's default chain -- so an EC2 IAM
instance role is picked up automatically and no keys live in the repo.

The client is intentionally thin: one `detect_text` call, no retry policy of its
own beyond botocore's, and no image handling.  Preprocessing, sheet assembly and
digit extraction live in `features/inference_lib/{ocr_preprocessor,
three_frame_sheet, rekognition_reader}.py`.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

from core import constants as C
from core.logging_setup import get_logger

log = get_logger("rekognition")


class RekognitionUnavailable(RuntimeError):
    """Raised when a Rekognition client cannot be constructed (no boto3, no
    credentials, bad region).  Callers degrade to NO_DATA / EasyOCR."""


class RekognitionClient:
    """Thin wrapper over ``boto3.client('rekognition')``.

    Parameters
    ----------
    region:
        AWS region.  Defaults to ``WAGONEYE_REKOGNITION_REGION`` then
        ``constants.S3_REGION`` (``ap-south-1``).
    aws_access_key / aws_secret_key:
        Explicit credentials.  When either is missing, boto3's default chain
        (env vars, shared config, EC2 instance role) is used.
    """

    def __init__(
        self,
        region: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        logger=None,
    ) -> None:
        self.logger = logger or log
        self.region = region or REKOGNITION_REGION
        try:
            import boto3
        except ImportError as e:                     # pragma: no cover
            raise RekognitionUnavailable(
                "boto3 is required for Rekognition OCR (pip install boto3)") from e

        try:
            if aws_access_key and aws_secret_key:
                self.client = boto3.client(
                    "rekognition",
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=self.region,
                )
                self.logger.info(
                    "Rekognition client initialised with provided credentials "
                    "(region=%s).", self.region)
            else:
                self.client = boto3.client("rekognition", region_name=self.region)
                self.logger.info(
                    "Rekognition client initialised with default/IAM-role "
                    "credentials (region=%s).", self.region)
        except Exception as e:                       # pragma: no cover - network/config
            raise RekognitionUnavailable(
                f"could not construct a Rekognition client: {e}") from e

    def detect_text(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """Run ``DetectText`` on JPEG/PNG bytes; return the raw TextDetections."""
        response = self.client.detect_text(Image={"Bytes": image_bytes})
        return response.get("TextDetections", [])


# -----------------------------------------------------------------------------
# Configuration + process-wide singleton
# -----------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val else default


#: Region used for Rekognition.  Separate from the S3 region so OCR can be
#: pinned to a region where DetectText is available without moving the buckets.
REKOGNITION_REGION = _env("WAGONEYE_REKOGNITION_REGION", C.S3_REGION)

#: Maximum Rekognition calls per wagon.  Each call is one DetectText request, so
#: this is the per-wagon cost ceiling.  The V4 engine tries at most 2 sheets per
#: band (primary + fallback); 4 leaves room for a second band without unbounded
#: spend on a noisy wagon.
MAX_CALLS_PER_WAGON = int(_env("WAGONEYE_REKOGNITION_MAX_CALLS_PER_WAGON", "4"))


_CLIENT: Optional[RekognitionClient] = None
_CLIENT_FAILED = False
_CLIENT_LOCK = threading.Lock()


def get_client(*, region: Optional[str] = None) -> Optional[RekognitionClient]:
    """Return the process-wide Rekognition client, or ``None`` if unavailable.

    Construction is attempted ONCE per process: a failure is remembered so every
    subsequent wagon short-circuits instead of re-raising per frame.  Thread-safe
    (the wagon-wise scheduler may call features from more than one thread).
    """
    global _CLIENT, _CLIENT_FAILED
    if _CLIENT is not None:
        return _CLIENT
    if _CLIENT_FAILED:
        return None
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT
        if _CLIENT_FAILED:
            return None
        try:
            _CLIENT = RekognitionClient(region=region)
        except RekognitionUnavailable as e:
            _CLIENT_FAILED = True
            log.warning("[REKOGNITION] unavailable: %s", e)
            return None
        return _CLIENT


def reset_client() -> None:
    """Drop the cached client (tests only)."""
    global _CLIENT, _CLIENT_FAILED
    with _CLIENT_LOCK:
        _CLIENT = None
        _CLIENT_FAILED = False
