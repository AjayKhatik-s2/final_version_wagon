"""Put the REAL evidence URLs into the combined report after uploading.

The ordering problem this exists to solve
-----------------------------------------
Stage 5 writes `combined_train_report.json`. Stage 6 uploads the evidence tree.
In that order, the report cannot contain a URL the upload has not produced yet.

While uploads went straight to S3 that did not matter: the key was
`train_batch/<batch_key>/evidence/<relative path>`, which Stage 5 could compute,
and `reporting.combined_train_report.evidence_base_url` did exactly that.

Through the Artifact Upload API it matters completely. The BACKEND chooses the
bucket and the key, so there is no base to prefix and no path to predict. A
computed link is a link to a file that is not there -- and the failure is silent:
the report validates, the dashboard renders, the image 404s.

So Stage 6 now uploads the evidence FIRST and hands the resulting
`{relative path -> URL}` map here, and this rewrites the report in place before
it is itself uploaded.

What changes in the document
----------------------------
`s3` mode: the computed links were already correct, so they are confirmed
against the map and left alone. Nothing about the contract changes.

`api` mode: `evidence_base_url` is REMOVED. Keeping it would be worse than
useless -- a consumer that joined it to a relative path would build a plausible
URL pointing nowhere. `evidence_page_urls` and each damage row's `evidence_url`
are replaced with the URL that file's own upload returned, and
`evidence_url_source` records which transport produced them so a reader can tell
a computed link from a returned one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from core.logging_setup import get_logger

log = get_logger("delivery.evidence_links")

_TAG = "[EVIDENCE-LINKS]"

SOURCE_COMPUTED = "computed_from_bucket_and_key"
SOURCE_API = "artifact_upload_api"


@dataclass
class RewriteResult:
    """What the rewrite changed, and what it could not resolve."""

    rewritten: bool = False
    mode: str = ""
    page_urls_set: int = 0
    damage_urls_set: int = 0
    unresolved: List[str] = field(default_factory=list)
    base_url_removed: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "wagon_eye.evidence_links.v1",
            "rewritten": self.rewritten,
            "mode": self.mode,
            "page_urls_set": self.page_urls_set,
            "damage_urls_set": self.damage_urls_set,
            "unresolved_count": len(self.unresolved),
            "unresolved_sample": self.unresolved[:20],
            "base_url_removed": self.base_url_removed,
            "error": self.error,
        }

    def render(self) -> str:
        if self.error:
            return f"{_TAG} FAILED: {self.error}"
        if not self.rewritten:
            return f"{_TAG} nothing to rewrite"
        return (f"{_TAG} mode={self.mode} page_urls={self.page_urls_set} "
                f"damage_urls={self.damage_urls_set} "
                f"unresolved={len(self.unresolved)}"
                + ("  evidence_base_url REMOVED (api mode: no predictable base)"
                   if self.base_url_removed else ""))


def rewrite(
    *,
    report_json_path: str,
    url_map: Mapping[str, str],
    mode: str,
    verbose: bool = True,
) -> RewriteResult:
    """Replace every evidence link in the report with the uploaded URL.

    `url_map` is keyed by the path relative to the evidence root, which is the
    same key the report's own `evidence_pages` uses -- so the two join without
    either side needing to know the other's layout.
    """
    from delivery.artifact_uploader import MODE_API

    res = RewriteResult(mode=mode)
    if not report_json_path or not os.path.isfile(report_json_path):
        res.error = f"no report at {report_json_path!r}"
        if verbose:
            log.warning("%s", res.render())
        return res
    try:
        with open(report_json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        res.error = f"{type(e).__name__}: {e}"
        if verbose:
            log.warning("%s", res.render())
        return res

    api_mode = (mode or "").strip().lower() == MODE_API

    # ---- per-wagon evidence pages -------------------------------------
    pages = doc.get("evidence_pages") or {}
    page_urls: Dict[str, Dict[str, str]] = {}
    for gw, snaps in pages.items():
        if not isinstance(snaps, Mapping):
            continue
        out: Dict[str, str] = {}
        for k, rel in snaps.items():
            url = url_map.get(str(rel))
            if url:
                out[k] = url
                res.page_urls_set += 1
            else:
                # No link at all rather than a computed one: a missing key is a
                # fact a consumer can handle, a URL that 404s is not.
                res.unresolved.append(f"{gw}/{k}={rel}")
        if out:
            page_urls[gw] = out
    doc["evidence_page_urls"] = page_urls

    # ---- per-damage-row links -----------------------------------------
    for w in (doc.get("wagons") or []):
        if not isinstance(w, Mapping):
            continue
        for row in (w.get("top_damage_details") or []):
            if not isinstance(row, dict):
                continue
            rel = row.get("evidence_path")
            if not rel:
                continue
            url = url_map.get(str(rel))
            if url:
                row["evidence_url"] = url
                res.damage_urls_set += 1
            else:
                row.pop("evidence_url", None)
                res.unresolved.append(
                    f"{w.get('global_id')}/damage={rel}")

    # ---- the base, and where these URLs came from ----------------------
    if api_mode:
        if "evidence_base_url" in doc:
            del doc["evidence_base_url"]
            res.base_url_removed = True
        doc["evidence_url_source"] = SOURCE_API
    else:
        doc["evidence_url_source"] = SOURCE_COMPUTED

    try:
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, default=str)
    except OSError as e:
        res.error = f"could not write the report back: {e}"
        if verbose:
            log.warning("%s", res.render())
        return res

    res.rewritten = True
    if verbose:
        log.info("%s", res.render())
        if res.unresolved:
            log.warning("%s %d evidence file(s) had no uploaded URL and carry "
                        "no link: %s", _TAG, len(res.unresolved),
                        ", ".join(res.unresolved[:5]))
    return res
