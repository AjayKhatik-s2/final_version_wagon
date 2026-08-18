"""Pipeline source abstraction -- WHAT the pipeline consumes, not HOW.

The orchestrator's job is to turn *trimmed per-camera train clips* into reports.
Those clips reach it in one of two ways, and that -- the SOURCE of the input --
is the only thing that varies between deployments:

    TRIMMED  the input prefixes already hold trimmed train clips.  An upstream
             producer put them there: the standalone extraction service, a
             different pipeline, or a manual upload.  The orchestrator is a pure
             consumer.  (Default -- preserves the original two-service topology.)

    RAW      only raw CCTV exists.  The orchestrator owns an ExtractionManager
             that discovers raw video, detects train completion, runs the train
             extractor, and produces the trimmed clips itself before consuming
             them.

This names the source.  It deliberately says nothing about threads, services, or
"inline vs standalone" -- those are implementation details owned by the
orchestrator and the ExtractionManager, not configuration the operator reasons
about.  An operator picks what data they have (raw or trimmed); the pipeline
decides how to satisfy it.
"""

from __future__ import annotations

import os
from enum import Enum


class PipelineSource(str, Enum):
    """The kind of input the pipeline consumes."""

    TRIMMED = "trimmed"   # already-cut clips in the input prefixes (default)
    RAW = "raw"           # raw CCTV -> ExtractionManager cuts trains first

    @classmethod
    def resolve(cls, value: "str | PipelineSource | None" = None) -> "PipelineSource":
        """Resolve from an explicit value, else WAGONEYE_PIPELINE_SOURCE, else
        TRIMMED.  Unknown values fall back to TRIMMED (the safe pure-consumer
        default) rather than raising, so a typo never takes the service down."""
        if isinstance(value, cls):
            return value
        raw = (value or os.getenv("WAGONEYE_PIPELINE_SOURCE") or "trimmed").strip().lower()
        if raw in ("raw", "raw_cctv", "cctv"):
            return cls.RAW
        if raw in ("trimmed", "clips", "preprocessed", "pre-processed"):
            return cls.TRIMMED
        return cls.TRIMMED

    @property
    def requires_extraction(self) -> bool:
        """True when the orchestrator must run extraction to obtain its input."""
        return self is PipelineSource.RAW
