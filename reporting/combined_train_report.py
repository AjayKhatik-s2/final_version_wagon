"""Stage 5 -- emit `combined_train_report.json` + `combined_train_report.pdf`.

Visual identity ported from the legacy WagonEye CombinedReportGenerator
(old_system/RIGHT_UP/combined_report_generator.py).  The PDF reproduces:

    1. Navy title banner ("COMBINED WAGON EYE REPORT" + IST date/time)
    2. VIDEO EVIDENCE table (5 cols: label + 4 cameras; RAW + PROCESSED)
    3. PARTIAL REPORT amber banner when any camera feed is missing
    4. DETAILED CAMERA REPORTS table (links to the 4 camera-wise PDFs)
    5. INSPECTION SUMMARY 10-column KPI table
    6. Wagon Inspection table (7 cols: SR.NO, WAGON#, LEFT DOORS,
       RIGHT DOORS, R-TOP, L-TOP, WAGON TYPE) with legacy issue-row
       highlighting rules
    7. Damaged Wagon Report -- per-anomaly-wagon evidence sections,
       grouped by wagon number, sorted by camera priority (Left, Right,
       Left-Side, Right-Side, Left-Top, Right-Top) with the legacy
       4.2 x 2.8 inch image grid

Data sources (read-only):
    * GlobalTrainState
    * UnifiedWagonState
    * wagon_states/<feature>/<gw>.json
    * evidence/<gw>/<feature>/{*.jpg, metadata.json}
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence

from core import constants as C
from core.global_state_loader import GlobalTrainState
from core.unified_wagon_state import UnifiedWagonState, summarize_wagons

from . import _brand
from . import _adapter
from . import _evidence_lookup as ev
from . import wagon_evidence_grid as WG


_REPORT_SCHEMA = "wagon_eye.combined_report.v4"


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------

def _now_ist():
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))


def _now_ist_iso() -> str:
    return _now_ist().isoformat(timespec="seconds")


def _date_str() -> str:
    return _now_ist().strftime("%d-%m-%Y")


def _time_str() -> str:
    return _now_ist().strftime("%H:%M IST")


def _datetime_str() -> str:
    return _now_ist().strftime("%d-%m-%Y - %H:%M:%S")


# -----------------------------------------------------------------------------
# JSON
# -----------------------------------------------------------------------------

def _evidence_pages(evidence_root: Optional[str], wagons) -> Dict[str, Dict[str, str]]:
    if not evidence_root or not os.path.isdir(evidence_root):
        return {}
    candidates = {
        "door":   ["left_best.jpg", "left_crop.jpg",
                    "right_best.jpg", "right_crop.jpg"],
        "ocr":    ["best_frame.jpg", "number_crop.jpg"],
        "damage": ["track_1.jpg", "track_2.jpg", "track_3.jpg"],
        "load":   ["best_frame.jpg"],
    }
    pages: Dict[str, Dict[str, str]] = {}
    for u in wagons:
        snaps: Dict[str, str] = {}
        for feat, files in candidates.items():
            for fn in files:
                p = os.path.join(evidence_root, u.global_id, feat, fn)
                if os.path.isfile(p):
                    key = f"{feat}_{os.path.splitext(fn)[0]}"
                    snaps[key] = os.path.relpath(p, start=evidence_root).replace(os.sep, "/")
        if snaps:
            pages[u.global_id] = snaps
    return pages


#: Prefix for the completeness audit. Grepping this in a run log gives the wagon
#: count at every stage the report touches, plus the exact ids at any mismatch.
_AUDIT = "[REPORT-AUDIT]"


def _strict_integrity() -> bool:
    """Whether a report-integrity mismatch should RAISE.

    Off by default: a train that got this far should still deliver what it has,
    and the diagnostic names every offending id either way. Set
    WAGONEYE_REPORT_STRICT=1 to make a mismatch fail the train, which is what
    you want in CI or while chasing a regression.
    """
    return (os.getenv("WAGONEYE_REPORT_STRICT") or "").strip().lower() in (
        "1", "true", "yes", "on")


def canonical_wagons(
    state: GlobalTrainState, unified: Dict[str, UnifiedWagonState],
) -> "tuple":
    """`(wagons_in_order, synthesized_ids)` -- EVERY canonical wagon, in order.

    The canonical Global Wagon timeline is the iteration source, never the set of
    wagons that happen to have feature results. `state.wagons` is that timeline:
    RIGHT_UP's authoritative master ordering, which this function reads and does
    not recompute -- there is no second wagon-counting system here.

    What this replaces mattered:

        [unified[w.global_id] for w in state.wagons if w.global_id in unified]

    Right source, right order, and a silent `if`. A wagon absent from `unified`
    vanished from `doc["wagons"]`, from `summary`, and from `evidence_pages`,
    with nothing logged -- so an incomplete report was indistinguishable from a
    short train.

    A wagon with no feature result is still a real wagon. Rather than drop it, a
    state is synthesized from the GlobalWagon alone by the materializer's OWN
    `_fuse_one` with every feature None -- which is exactly its "no observations"
    path, so the placeholder is built by the same code as every other wagon and
    cannot drift from it. Its ids are returned so the caller can report them
    instead of hiding them.

    Absence of a supporting camera, an all-OK feature state and a missing
    snapshot are NOT reasons to omit a wagon; they are things to render.
    """
    synthesized: List[str] = []
    out: List[UnifiedWagonState] = []
    for w in state.wagons:
        u = unified.get(w.global_id)
        if u is None:
            from fusion.wagon_state_builder import _fuse_one
            u = _fuse_one(w, door=None, ocr=None, load=None, damage=None)
            synthesized.append(w.global_id)
        out.append(u)
    return out, synthesized


def audit_report_integrity(
    *, state: GlobalTrainState, wagons_in_order: Sequence[UnifiedWagonState],
    unified: Optional[Dict[str, UnifiedWagonState]] = None,
    synthesized: Optional[Sequence[str]] = None,
    strict: bool = False, verbose: bool = True,
) -> Dict[str, Any]:
    """Verify the report covers the canonical timeline exactly, and say so.

    Checks the SET, the ORDER and the MULTIPLICITY, because the three fail
    differently: a filtered iteration drops ids, a reordered one renumbers later
    wagons, and a merge bug duplicates. Counting alone catches none of them
    reliably -- two drops and one duplicate still total N.

    Returns the audit. On mismatch it logs every offending id (not just counts)
    at high severity, and raises when `strict`. It never silently passes.
    """
    canonical = [w.global_id for w in state.wagons]
    reported = [u.global_id for u in wagons_in_order]
    missing = [g for g in canonical if g not in set(reported)]
    extra = [g for g in reported if g not in set(canonical)]
    seen: Dict[str, int] = {}
    for g in reported:
        seen[g] = seen.get(g, 0) + 1
    duplicated = sorted(g for g, n in seen.items() if n > 1)
    ordered = reported == canonical

    audit = {
        "canonical_wagons": len(canonical),
        "report_wagons": len(reported),
        "missing_from_report": missing,
        "extra_in_report": extra,
        "duplicated_in_report": duplicated,
        "order_matches_master_timeline": ordered,
        "synthesized_no_evidence": list(synthesized or []),
        "fused_wagons": (len(unified) if unified is not None else None),
        "ok": not missing and not extra and not duplicated and ordered,
    }

    if verbose:
        print(f"{_AUDIT} canonical (master timeline) wagons="
              f"{audit['canonical_wagons']}")
        if unified is not None:
            print(f"{_AUDIT} fused/materialized wagons={len(unified)}")
        print(f"{_AUDIT} combined-report input wagons={audit['report_wagons']}")
        if audit["synthesized_no_evidence"]:
            print(f"{_AUDIT} rendered with no feature evidence "
                  f"({len(audit['synthesized_no_evidence'])}): "
                  f"{audit['synthesized_no_evidence']}")

    if not audit["ok"]:
        detail = (f"missing_from_report={missing} extra_in_report={extra} "
                  f"duplicated_in_report={duplicated} "
                  f"order_matches_master_timeline={ordered}")
        print(f"{_AUDIT} SEVERE: report does not match the canonical Global "
              f"Wagon Timeline -- {detail}")
        if strict:
            raise RuntimeError(
                f"combined report integrity check failed: {detail}")
    elif verbose:
        print(f"{_AUDIT} OK: every canonical wagon present exactly once, in "
              f"master order")
    return audit


def _camera_meta_for_manifest(state, wagon_states_root):
    """`{camera: {fps, total_frames}}` for frame selection.

    Read from the per-camera tracking JSON the pipeline already wrote; falls
    back to the master's fps/frames, which is what the historical shared-t=0
    assumption implies. Nothing is derived from video here.
    """
    out = {}
    master_fps = float(getattr(state, "master_fps", 0.0) or 0.0)
    master_total = int(getattr(state, "master_total_frames", 0) or 0)
    counts = dict(getattr(state, "per_camera_local_counts", None) or {})
    for cam in C.ALL_CAMERAS:
        meta = counts.get(cam) if isinstance(counts.get(cam), dict) else {}
        out[cam] = {
            "fps": float((meta or {}).get("fps") or master_fps),
            "total_frames": int((meta or {}).get("total_frames")
                                or master_total),
        }
    return out


def _build_json(
    *,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    batch_key: str,
    source_video_urls: Optional[Dict[str, str]] = None,
    processed_video_urls: Optional[Dict[str, str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    evidence_root: Optional[str] = None,
    legacy_view_model: Optional[_adapter.LegacyViewModel] = None,
) -> Dict[str, Any]:
    wagons_in_order, _synth = canonical_wagons(state, unified)
    summary = summarize_wagons(wagons_in_order)
    doc: Dict[str, Any] = {
        "schema":      _REPORT_SCHEMA,
        "batch_key":   batch_key,
        "generated_at": _now_ist_iso(),
        "train_metadata": {
            "master_camera":       state.master_camera,
            "master_fps":          state.master_fps,
            "master_total_frames": state.master_total_frames,
            "source_video_urls":   dict(source_video_urls or {}),
            "processed_video_urls":dict(processed_video_urls or {}),
        },
        "summary": summary,
        # Machine-readable completeness record, so a consumer can tell a short
        # train from a short report without re-deriving it.
        "report_integrity": audit_report_integrity(
            state=state, wagons_in_order=wagons_in_order, unified=unified,
            synthesized=_synth, strict=False, verbose=False),
        "stage0_fallback_used":    state.fallback_used,
        "stage0_fallback_reason":  state.fallback_reason,
        "stage0_corrections_applied": list(state.corrections_applied),
        "per_camera_local_counts": dict(state.per_camera_local_counts),
        "wagons": [u.to_dict() for u in wagons_in_order],
        "evidence_pages": _evidence_pages(evidence_root, wagons_in_order),
    }
    if legacy_view_model is not None:
        doc["legacy_view_model"] = {
            "summary_kpis": legacy_view_model.summary_kpis,
            "state_counts": legacy_view_model.state_counts,
            "merged_wagons": legacy_view_model.merged_wagons,
        }
    if extra_metadata:
        doc["train_metadata"].update(extra_metadata)
    return doc


# -----------------------------------------------------------------------------
# PDF -- legacy-identity reportlab body
# -----------------------------------------------------------------------------

def _build_pdf(
    *,
    report_manifest: Optional[Dict[str, Any]] = None,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    vm: _adapter.LegacyViewModel,
    batch_key: str,
    output_pdf: str,
    evidence_root: Optional[str],
    source_video_urls: Dict[str, str],
    processed_video_urls: Dict[str, str],
    camera_pdf_urls: Dict[str, str],
    logo_path: Optional[str],
    missing_cameras: Sequence[str],
    cache_root: Optional[str] = None,
) -> str:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        BaseDocTemplate, Frame, PageTemplate,
        Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)) or ".", exist_ok=True)

    page_w, page_h = landscape(A4)
    L = 0.5 * inch
    doc = BaseDocTemplate(
        output_pdf,
        pagesize=landscape(A4),
        leftMargin=L, rightMargin=L,
        topMargin=L,  bottomMargin=L,
        title=f"WagonEye Combined Report -- {batch_key}",
        author="WagonEye",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="content")
    on_page = _brand.make_logo_callback(logo_path)
    doc.addPageTemplates([
        PageTemplate(id="main", frames=[frame], onPage=on_page)
    ])

    styles = _brand.build_styles()
    elements: List[Any] = []

    # ----- 1. TITLE BANNER -----
    elements.append(Spacer(1, 0.25 * inch))
    banner_data = [[
        Paragraph("COMBINED WAGON EYE REPORT", styles["BannerTitle"])
    ], [
        Paragraph(f"{_date_str()}  |  {_time_str()}", styles["BannerDate"])
    ]]
    banner = Table(banner_data, colWidths=[_brand.PAGE_BODY_WIDTH])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _brand.NAVY_DARK),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (0, 0),   14),
        ("BOTTOMPADDING", (0, 0), (0, 0), 2),
        ("TOPPADDING",    (0, 1), (0, 1), 0),
        ("BOTTOMPADDING", (0, 1), (0, 1), 12),
        ("LEFTPADDING",  (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("BOX", (0, 0), (-1, -1), 1.5, _brand.NAVY_DARK),
    ]))
    elements.append(banner)
    elements.append(Spacer(1, 0.18 * inch))

    # ----- 2. VIDEO EVIDENCE -----
    cams = list(C.ALL_CAMERAS)
    label_w = 1.4 * inch
    cam_w   = 2.15 * inch

    def _cam_link(url: Optional[str], cam: str):
        return _brand.make_camera_link(
            url, "Click to View", cam, missing_cameras, styles,
        )

    raw_cells = [_cam_link(source_video_urls.get(cam), cam) for cam in cams]
    proc_cells = [_cam_link(processed_video_urls.get(cam), cam) for cam in cams]

    video_data = [
        [Paragraph("<b>VIDEO EVIDENCE</b>", styles["SectionTitleWhite"]),
         "", "", "", ""],
        [Paragraph("", styles["CameraLabel"])] + [
            Paragraph(f"<b>{cam}</b>", styles["CameraLabel"]) for cam in cams
        ],
        [Paragraph("<b>Raw Video</b>", styles["CameraLabel"])] + raw_cells,
        [Paragraph("<b>Processed Video</b>", styles["CameraLabel"])] + proc_cells,
    ]
    video_t = Table(video_data, colWidths=[label_w, cam_w, cam_w, cam_w, cam_w])
    video_style = [
        ("SPAN", (0, 0), (-1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), _brand.NAVY_MID),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, 1), _brand.SLATE_LIGHT),
        ("TOPPADDING", (0, 1), (-1, 1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 5),
        ("BACKGROUND", (0, 2), (0, 2), _brand.SLATE_LIGHT),
        ("BACKGROUND", (1, 2), (-1, 2), _brand.WHITE),
        ("TOPPADDING", (0, 2), (-1, 2), 8),
        ("BOTTOMPADDING", (0, 2), (-1, 2), 8),
        ("BACKGROUND", (0, 3), (0, 3), _brand.SLATE_LIGHT),
        ("BACKGROUND", (1, 3), (-1, 3), _brand.SLATE_BG),
        ("TOPPADDING", (0, 3), (-1, 3), 8),
        ("BOTTOMPADDING", (0, 3), (-1, 3), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 1, _brand.SLATE_BORDER),
        ("INNERGRID", (0, 1), (-1, -1), 0.5, _brand.SLATE_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for i, cam in enumerate(cams):
        if cam in missing_cameras:
            video_style.append(("BACKGROUND", (i + 1, 1), (i + 1, -1), _brand.NA_BG))
    video_t.setStyle(TableStyle(video_style))
    elements.append(video_t)
    elements.append(Spacer(1, 0.14 * inch))

    # ----- 3. PARTIAL REPORT WARNING -----
    warn = _brand.make_warning_banner(missing_cameras, styles)
    if warn is not None:
        elements.append(warn)
        elements.append(Spacer(1, 0.12 * inch))

    # ----- 4. DETAILED CAMERA REPORTS (legacy per-camera report links) -----
    camera_links_order = [
        (C.CAMERA_LEFT_UP,      "LEFT Detail Report"),
        (C.CAMERA_RIGHT_UP,     "RIGHT Detail Report"),
        (C.CAMERA_RIGHT_UP_TOP, "R-TOP Detail Report"),
        (C.CAMERA_LEFT_UP_TOP,  "L-TOP Detail Report"),
    ]
    feat_cells = []
    for cam, label in camera_links_order:
        url = (camera_pdf_urls or {}).get(cam)
        if cam in missing_cameras:
            feat_cells.append(Paragraph(
                '<font color="#C62828"><i>NO FEED</i></font>',
                styles["NoFeedCell"],
            ))
        elif url:
            feat_cells.append(Paragraph(
                f'<a href="{url}" color="#1565C0"><b><u>{label}</u></b></a>',
                styles["LinkCellPro"],
            ))
        else:
            feat_cells.append(Paragraph(
                f'<font color="#78909C">{label}</font>',
                styles["LinkCell"],
            ))

    report_data = [
        [Paragraph("<b>DETAILED CAMERA REPORTS</b>", styles["SectionTitleWhite"]),
         "", "", ""],
        [Paragraph(f"<b>{C.CAMERA_LEFT_UP}</b>",      styles["CameraLabel"]),
         Paragraph(f"<b>{C.CAMERA_RIGHT_UP}</b>",     styles["CameraLabel"]),
         Paragraph(f"<b>{C.CAMERA_RIGHT_UP_TOP}</b>", styles["CameraLabel"]),
         Paragraph(f"<b>{C.CAMERA_LEFT_UP_TOP}</b>",  styles["CameraLabel"])],
        feat_cells,
    ]
    report_t = Table(report_data, colWidths=[2.5 * inch] * 4)
    report_t.setStyle(TableStyle([
        ("SPAN", (0, 0), (-1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), _brand.TEAL_ACCENT),
        ("ALIGN",  (0, 0), (-1, 0), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, 0), 7),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
        ("BACKGROUND", (0, 1), (-1, 1), _brand.SLATE_LIGHT),
        ("TOPPADDING", (0, 1), (-1, 1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 5),
        ("BACKGROUND", (0, 2), (-1, 2), _brand.WHITE),
        ("TOPPADDING", (0, 2), (-1, 2), 9),
        ("BOTTOMPADDING", (0, 2), (-1, 2), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 1, _brand.SLATE_BORDER),
        ("INNERGRID", (0, 1), (-1, -1), 0.5, _brand.SLATE_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(report_t)
    elements.append(Spacer(1, 0.18 * inch))

    # ----- 5. INSPECTION SUMMARY -----
    kpi = vm.summary_kpis
    rake = kpi["rake_type"]
    status = kpi["status"]
    loco = " / ".join(kpi["loco_numbers"]) if kpi["loco_numbers"] else "Not Detected"

    def _na_if_missing(camera_id, val):
        return "N/A" if camera_id in missing_cameras else str(val)

    status_color = "#C62828" if status == "NOT OK" else "#2E7D32"
    rake_color = (
        "#1565C0" if rake == "LOADED RAKE"
        else ("#E65100" if rake == "EMPTY RAKE" else "#1A1A2E")
    )

    from reportlab.lib.styles import ParagraphStyle
    header_p = ParagraphStyle(
        "SummaryHeader", fontSize=8, alignment=1, textColor=_brand.WHITE,
        fontName="Helvetica-Bold", leading=11,
    )
    data_p = ParagraphStyle(
        "SummaryData", fontSize=9, alignment=1, textColor=_brand.TEXT_DARK,
        fontName="Helvetica", leading=12,
    )
    data_b = ParagraphStyle(
        "SummaryDataBold", fontSize=9, alignment=1, textColor=_brand.TEXT_DARK,
        fontName="Helvetica-Bold", leading=12,
    )

    title_row = [Paragraph("<b>INSPECTION SUMMARY</b>", styles["SectionTitleWhite"]),
                 "", "", "", "", "", "", "", "", ""]
    header_row = [
        Paragraph("DATE-TIME",       header_p),
        Paragraph("LOCO NUMBER",     header_p),
        Paragraph("TOTAL<br/>WAGONS", header_p),
        Paragraph("LEFT OPEN<br/>DOORS",  header_p),
        Paragraph("RIGHT OPEN<br/>DOORS", header_p),
        Paragraph("R-TOP<br/>DAMAGES",    header_p),
        Paragraph("L-TOP<br/>DAMAGES",    header_p),
        Paragraph("PARTIAL<br/>CLOSED",   header_p),
        Paragraph("RAKE<br/>TYPE",        header_p),
        Paragraph("STATUS",          header_p),
    ]
    partial_text = (
        f"L {_na_if_missing(C.CAMERA_LEFT_UP,  kpi['left_partial'])} / "
        f"R {_na_if_missing(C.CAMERA_RIGHT_UP, kpi['right_partial'])}"
    )
    data_row = [
        Paragraph(_datetime_str(), data_p),
        Paragraph(f"<b>{loco}</b>", data_b),
        Paragraph(f"<b>{kpi['total_wagons']}</b>", data_b),
        Paragraph(f"<b>{_na_if_missing(C.CAMERA_LEFT_UP,  kpi['left_open'])}</b>",  data_b),
        Paragraph(f"<b>{_na_if_missing(C.CAMERA_RIGHT_UP, kpi['right_open'])}</b>", data_b),
        Paragraph(f"<b>{_na_if_missing(C.CAMERA_RIGHT_UP_TOP, kpi['top_damages'])}</b>", data_b),
        Paragraph(f"<b>{_na_if_missing(C.CAMERA_LEFT_UP_TOP,  kpi['left_top_damages'])}</b>", data_b),
        Paragraph(partial_text, data_p),
        Paragraph(f'<b><font color="{rake_color}">{rake}</font></b>', data_b),
        Paragraph(f'<b><font color="{status_color}">{status}</font></b>', data_b),
    ]
    summary_data = [title_row, header_row, data_row]
    col_w = [1.2*inch, 1.1*inch, 0.7*inch, 0.8*inch, 0.8*inch,
             0.8*inch, 0.8*inch, 0.9*inch, 1.0*inch, 1.0*inch]
    summary_t = Table(summary_data, colWidths=col_w)

    status_bg = colors.HexColor("#FFEBEE") if status == "NOT OK" else colors.HexColor("#E8F5E9")
    rake_bg = (
        _brand.LOADED_BG if rake == "LOADED RAKE"
        else (_brand.EMPTY_BG if rake == "EMPTY RAKE" else _brand.WHITE)
    )
    summary_style = [
        ("SPAN", (0, 0), (-1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), _brand.NAVY_MID),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, 0), 7),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
        ("BACKGROUND", (0, 1), (-1, 1), _brand.NAVY_DARK),
        ("TOPPADDING", (0, 1), (-1, 1), 8),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("BACKGROUND", (0, 2), (-1, 2), _brand.WHITE),
        ("TOPPADDING", (0, 2), (-1, 2), 10),
        ("BOTTOMPADDING", (0, 2), (-1, 2), 10),
        ("BACKGROUND", (-1, 2), (-1, 2), status_bg),
        ("BACKGROUND", (8, 2), (8, 2), rake_bg),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 1, _brand.SLATE_BORDER),
        ("INNERGRID", (0, 1), (-1, -1), 0.5, _brand.SLATE_BORDER),
        ("LINEBELOW", (0, 0), (-1, 0), 1, _brand.SLATE_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    # Grey out missing-camera KPI cells
    cam_to_col = {
        C.CAMERA_LEFT_UP: 3, C.CAMERA_RIGHT_UP: 4,
        C.CAMERA_RIGHT_UP_TOP: 5, C.CAMERA_LEFT_UP_TOP: 6,
    }
    for cam, col in cam_to_col.items():
        if cam in missing_cameras:
            summary_style.append(("BACKGROUND", (col, 2), (col, 2), _brand.NA_BG))
    summary_t.setStyle(TableStyle(summary_style))
    elements.append(summary_t)
    elements.append(Spacer(1, 0.18 * inch))

    # ----- 6. WAGON INSPECTION TABLE -----
    _wagon_table = _build_wagon_table(vm, styles, missing_cameras)
    elements.append(_wagon_table)

    # ----- 6b. MULTI-ANGLE WAGON EVIDENCE (wagon-centric, all 4 cameras) -----
    multi_angle = _build_multi_angle_section(
        state=state, unified=unified,
        evidence_root=evidence_root, cache_root=cache_root,
        styles=styles, missing_cameras=missing_cameras,
    )
    if multi_angle:
        elements.extend(multi_angle)

    # ----- 7. DAMAGED WAGON REPORT (evidence pages) -----
    evidence_blocks = _build_evidence_section(
        vm=vm, styles=styles, evidence_root=evidence_root,
    )
    if evidence_blocks:
        elements.append(PageBreak())
        elements.extend(evidence_blocks)

    # ----- 8. WAGON EVIDENCE GRID: every canonical GW, all four cameras -----
    # Driven by the MANIFEST, which is driven by `state.wagons`, so a wagon with
    # no feature finding still gets its sixteen slots and cannot drop out. The
    # PDF and the audit read the same selection, so they cannot disagree about
    # which image is on which page.
    if report_manifest:
        try:
            elements.extend(build_wagon_grid_section(report_manifest, styles))
        except Exception as e:  # noqa: BLE001 -- a section must not kill the PDF
            print(f"[REPORT] wagon evidence grid failed: "
                  f"{type(e).__name__}: {e}")

        # ----- 9. FINAL: damage only, grouped by canonical GW -----
        try:
            elements.extend(build_damage_summary_section(report_manifest,
                                                        styles))
        except Exception as e:  # noqa: BLE001
            print(f"[REPORT] damage summary failed: {type(e).__name__}: {e}")

    doc.build(elements)
    return output_pdf


# -----------------------------------------------------------------------------
# Wagon table -- legacy 7-column with issue-row highlighting (legacy 797-1015)
# -----------------------------------------------------------------------------

def _build_wagon_table(vm: _adapter.LegacyViewModel, styles, missing_cameras):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Table, TableStyle

    NO_FEED_TEXT = "⚠ NO FEED"
    # Column layout: SR(0) WAGON#(1) CLASS(2) LEFT(3) RIGHT(4) R-TOP(5) L-TOP(6) TYPE(7)
    cam_col = {
        C.CAMERA_LEFT_UP: 3, C.CAMERA_RIGHT_UP: 4,
        C.CAMERA_RIGHT_UP_TOP: 5, C.CAMERA_LEFT_UP_TOP: 6,
    }
    missing_cols = {cam_col[c] for c in missing_cameras if c in cam_col}

    col_header_p = ParagraphStyle(
        "WagonColHeader", fontSize=8, alignment=1, textColor=_brand.WHITE,
        fontName="Helvetica-Bold", leading=11,
    )
    cell = ParagraphStyle(
        "WagonCell", fontSize=8, alignment=1, textColor=_brand.TEXT_BODY,
        fontName="Helvetica", leading=11,
    )
    cell_b = ParagraphStyle(
        "WagonCellBold", fontSize=8, alignment=1, textColor=_brand.TEXT_DARK,
        fontName="Helvetica-Bold", leading=11,
    )
    issue = ParagraphStyle(
        "WagonIssue", fontSize=8, alignment=1, textColor=_brand.COLOR_NOT_OK,
        fontName="Helvetica-Bold", leading=11,
    )
    nofeed = ParagraphStyle(
        "WagonNoFeed", fontSize=7, alignment=1, textColor=_brand.TEXT_LIGHT,
        fontName="Helvetica-Oblique", leading=10,
    )

    title_row = [Paragraph("<b>WAGON INSPECTION DETAILS</b>",
                            styles["SectionTitleWhite"]),
                 "", "", "", "", "", "", ""]
    header_row = [
        Paragraph("SR.NO",               col_header_p),
        Paragraph("WAGON NUMBER",        col_header_p),
        Paragraph("CLASS",               col_header_p),
        Paragraph("LEFT CAMERA<br/>DOORS",  col_header_p),
        Paragraph("RIGHT CAMERA<br/>DOORS", col_header_p),
        Paragraph("R-TOP<br/>DAMAGES",   col_header_p),
        Paragraph("L-TOP<br/>DAMAGES",   col_header_p),
        Paragraph("WAGON<br/>TYPE",      col_header_p),
    ]
    rows = [title_row, header_row]
    highlight_info = []

    for wagon in vm.merged_wagons:
        row_idx = len(rows)
        sr = str(wagon["wagon_sr_no"])
        wn = wagon.get("ocr_wagon_number") or "-"
        wn_disp = wn if wn != "-" else "-"

        has_l = wagon.get("has_open_left")     and 2 not in missing_cols
        has_r = wagon.get("has_open_right")    and 3 not in missing_cols
        has_t = wagon.get("has_open_top")      and 4 not in missing_cols
        has_lt = wagon.get("has_open_left_top") and 5 not in missing_cols

        l_text = NO_FEED_TEXT if 2 in missing_cols else wagon.get("left_doors_text",  "NO DATA")
        r_text = NO_FEED_TEXT if 3 in missing_cols else wagon.get("right_doors_text", "NO DATA")
        t_text = NO_FEED_TEXT if 4 in missing_cols else wagon.get("top_doors_text",   "NO DATA")
        lt_text = NO_FEED_TEXT if 5 in missing_cols else wagon.get("left_top_doors_text", "NO DATA")

        wt_text = wagon.get("wagon_type", "-")
        if wt_text == "LOADED":
            wt_style = ParagraphStyle("WagonLoaded", parent=cell,
                                       textColor=_brand.COLOR_LOADED,
                                       fontName="Helvetica-Bold")
        elif wt_text == "EMPTY":
            wt_style = ParagraphStyle("WagonEmpty", parent=cell,
                                       textColor=_brand.COLOR_EMPTY,
                                       fontName="Helvetica-Bold")
        else:
            wt_style = cell

        # Global wagon class (authoritative from GlobalTrainState).  Never
        # defaults to "WAGON": an unclassified wagon shows UNKNOWN.
        cls_raw = str(wagon.get("classification") or C.CLASS_UNKNOWN)
        cls_disp = cls_raw.replace("_", " ")
        if cls_raw == C.CLASS_ENGINE:
            cls_style = ParagraphStyle("WagonClsEng", parent=cell_b,
                                       textColor=_brand.COLOR_EMPTY)
        elif cls_raw == C.CLASS_BRAKE_VAN:
            cls_style = ParagraphStyle("WagonClsBv", parent=cell_b,
                                       textColor=_brand.COLOR_LOADED)
        else:
            cls_style = cell_b

        l_s = issue if has_l else (nofeed if 3 in missing_cols else cell)
        r_s = issue if has_r else (nofeed if 4 in missing_cols else cell)
        t_s = issue if has_t else (nofeed if 5 in missing_cols else cell)
        lt_s = issue if has_lt else (nofeed if 6 in missing_cols else cell)

        rows.append([
            Paragraph(f"<b>{sr}</b>", cell_b),
            Paragraph(f"<b>{wn_disp}</b>", cell_b) if wn_disp != "-" else Paragraph(wn_disp, cell),
            Paragraph(cls_disp, cls_style),
            Paragraph(l_text, l_s),
            Paragraph(r_text, r_s),
            Paragraph(t_text, t_s),
            Paragraph(lt_text, lt_s),
            Paragraph(wt_text, wt_style),
        ])

        issue_cols = []
        if has_l:  issue_cols.append(3)
        if has_r:  issue_cols.append(4)
        if has_t:  issue_cols.append(5)
        if has_lt: issue_cols.append(6)
        if issue_cols:
            highlight_info.append((row_idx, issue_cols))

    t = Table(
        rows,
        colWidths=[0.5*inch, 1.3*inch, 1.0*inch, 1.9*inch, 1.9*inch,
                   0.9*inch, 0.9*inch, 0.9*inch],
        repeatRows=2,
    )
    style = [
        ("SPAN", (0, 0), (-1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), _brand.NAVY_MID),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, 0), 7),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
        ("BACKGROUND", (0, 1), (-1, 1), _brand.NAVY_DARK),
        ("TOPPADDING", (0, 1), (-1, 1), 8),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 1, _brand.SLATE_BORDER),
        ("INNERGRID", (0, 1), (-1, -1), 0.5, _brand.SLATE_BORDER),
        ("LINEBELOW", (0, 0), (-1, 0), 1, _brand.SLATE_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 2), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 2), (-1, -1), 8),
    ]
    # Alternating row backgrounds
    n_data = len(vm.merged_wagons)
    for i in range(n_data):
        row_idx = i + 2
        bg = _brand.WHITE if i % 2 == 0 else _brand.SLATE_BG
        style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), bg))
    # Grey columns for missing cameras
    for col in missing_cols:
        if n_data > 0:
            style.append(("BACKGROUND", (col, 2), (col, n_data + 1), _brand.NA_BG))
    # Issue-row highlighting (>=2 cameras whole row; 1 camera SR+WN+col)
    for row_idx, issue_cols in highlight_info:
        if len(issue_cols) >= 2:
            style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), _brand.ISSUE_BG))
        else:
            style.append(("BACKGROUND", (0, row_idx), (2, row_idx), _brand.ISSUE_BG))
            for col in issue_cols:
                style.append(("BACKGROUND", (col, row_idx), (col, row_idx), _brand.ISSUE_BG))
    t.setStyle(TableStyle(style))
    return t


# -----------------------------------------------------------------------------
# Multi-Angle Wagon Evidence  --  one page per anomalous wagon showing all four
# camera perspectives of the SAME global wagon (presentation enhancement).
# -----------------------------------------------------------------------------

# Grid order: top row RIGHT_UP | LEFT_UP ; bottom row RIGHT_UP_TOP | LEFT_UP_TOP
_MULTI_ANGLE_GRID = (
    (C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP),
    (C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP),
)


def _cache_mid_frame(cache_root: Optional[str], gw_id: str, camera_id: str) -> Optional[str]:
    """A representative wagon_cache frame (the temporal middle) for one camera,
    so we can still show the wagon from that angle even with no detection."""
    if not cache_root:
        return None
    folder = C.CAMERA_FOLDER.get(camera_id, camera_id.lower())
    d = os.path.join(cache_root, gw_id, folder)
    if not os.path.isdir(d):
        return None
    try:
        jpgs = sorted(fn for fn in os.listdir(d) if fn.endswith(".jpg"))
    except OSError:
        return None
    if not jpgs:
        return None
    return os.path.join(d, jpgs[len(jpgs) // 2])


def _damage_track_cams(evidence_root: Optional[str], gw_id: str) -> set:
    """Set of top-camera ids that actually produced a damage track for a wagon."""
    md = ev.evidence_metadata(evidence_root, gw_id, "damage")
    cams = set()
    for tr in (md.get("tracks") or []):
        if isinstance(tr, dict) and tr.get("camera_id"):
            cams.add(tr["camera_id"])
    return cams


def _top_damage_snapshot(evidence_root: Optional[str], gw_id: str, camera_id: str) -> Optional[str]:
    """Best damage-track snapshot for ONE top camera (highest best_confidence)."""
    md = ev.evidence_metadata(evidence_root, gw_id, "damage")
    best = None
    best_conf = -1.0
    for tr in (md.get("tracks") or []):
        if not isinstance(tr, dict) or tr.get("camera_id") != camera_id:
            continue
        idx = tr.get("track_idx")
        if idx is None:
            continue
        conf = float(tr.get("best_confidence") or 0.0)
        if conf > best_conf:
            # Camera-scoped slot (`track_2__RIGHT_UP_TOP`), legacy name as
            # fallback -- safe because the loop above already confirmed this
            # track belongs to `camera_id`.
            snap = ev.damage_track_snapshot(evidence_root, gw_id,
                                            camera_id, int(idx))
            if snap:
                best, best_conf = snap, conf
    return best


def _wagon_is_anomalous(u: UnifiedWagonState) -> bool:
    """True when a wagon has any reportable anomaly: open/damaged door, top or
    side damage, OCR missing on a WAGON, or a load NO_DATA on a WAGON."""
    if u is None:
        return False
    if _brand.is_side_anomaly(u.left_door) or _brand.is_side_anomaly(u.right_door):
        return True
    if u.top_damage == C.DAMAGE_PRESENT or u.side_damage == C.DAMAGE_PRESENT:
        return True
    if u.classification == C.CLASS_WAGON:
        if u.wagon_identifier in (None, "", C.NO_DATA):
            return True
        if u.load_status == C.NO_DATA:
            return True
    return False


def _authoritative_cams(u: UnifiedWagonState, damage_cams: set) -> set:
    """Cameras whose authority actually detected one of the wagon's anomalies."""
    cams = set()
    if _brand.is_side_anomaly(u.right_door):
        cams.add(C.CAMERA_RIGHT_UP)
    if _brand.is_side_anomaly(u.left_door):
        cams.add(C.CAMERA_LEFT_UP)
    if u.classification == C.CLASS_WAGON and u.wagon_identifier in (None, "", C.NO_DATA):
        cams.add(C.CAMERA_RIGHT_UP)   # OCR authority
    if u.top_damage == C.DAMAGE_PRESENT:
        cams |= (damage_cams or {C.CAMERA_RIGHT_UP_TOP})
    if u.classification == C.CLASS_WAGON and u.load_status == C.NO_DATA:
        cams.add(C.CAMERA_RIGHT_UP_TOP)  # load authority
    return cams


def _panel_state_text(u: UnifiedWagonState, camera_id: str, damage_cams: set,
                      has_frames: bool) -> str:
    """The per-camera 'detected state' label shown under each panel."""
    if camera_id == C.CAMERA_RIGHT_UP:
        door = (_brand.format_door_status(u.right_door)
                if u.right_door not in (None, "", C.NO_DATA) else "NO DATA")
        s = f"Right Door: {door}"
        if u.classification == C.CLASS_WAGON and u.wagon_identifier in (None, "", C.NO_DATA):
            s += "  |  OCR: MISSING"
        elif u.wagon_identifier not in (None, "", C.NO_DATA):
            s += f"  |  OCR: {u.wagon_identifier}"
        return s
    if camera_id == C.CAMERA_LEFT_UP:
        door = (_brand.format_door_status(u.left_door)
                if u.left_door not in (None, "", C.NO_DATA) else "NO DATA")
        return f"Left Door: {door}"
    if camera_id == C.CAMERA_RIGHT_UP_TOP:
        dmg = "DAMAGE" if C.CAMERA_RIGHT_UP_TOP in damage_cams else ("OK" if has_frames else "NO DATA")
        load = u.load_status if u.load_status not in (None, "") else C.NO_DATA
        return (f"Top Damage: {dmg}{_elsewhere(u, camera_id, damage_cams)}"
                f"  |  Load: {load}")
    # LEFT_UP_TOP
    dmg = "DAMAGE" if C.CAMERA_LEFT_UP_TOP in damage_cams else ("OK" if has_frames else "NO DATA")
    return (f"Top Damage (support): {dmg}"
            f"{_elsewhere(u, camera_id, damage_cams)}")


def _elsewhere(u: UnifiedWagonState, camera_id: str, damage_cams: set) -> str:
    """Attribute damage this camera did NOT see to the camera that did.

    Replaces what the removed `_best_damage_snapshot_any` fallback used to
    achieve by substituting the other camera's IMAGE: the wagon is known
    damaged, this camera has no track of its own, so instead of borrowing a
    frame the panel says who saw it.  The reader still learns the wagon is
    damaged, and the snapshot above it remains this camera's own view.
    """
    if u is None or u.top_damage != C.DAMAGE_PRESENT:
        return ""
    if camera_id in (damage_cams or set()):
        return ""                      # this camera saw it; nothing to add
    others = sorted(c for c in (damage_cams or set()) if c != camera_id)
    if not others:
        return "  (damage fused from top cameras)"
    return f"  (detected by {', '.join(others)})"


def _panel_snapshot(u: UnifiedWagonState, camera_id: str,
                    evidence_root: Optional[str], cache_root: Optional[str],
                    gw_id: str) -> Optional[str]:
    """Resolve a snapshot for one camera: feature evidence first, else a
    representative wagon_cache frame (so all four angles show the wagon).

    CAMERA IDENTITY IS A HARD INVARIANT HERE.  Every branch may only return an
    image produced by `camera_id` itself.  A camera with no usable evidence
    shows its own wagon_cache frame; it NEVER borrows another camera's image,
    not even to illustrate a real anomaly.

    Two camera-blind fallbacks used to live in this function and both are gone:

    * `_best_damage_snapshot_any(...)`, reached by BOTH top panels when the
      wagon was damaged but this camera had no track of its own.  It returned
      the best track across both top cameras, so RIGHT_UP_TOP could render
      LEFT_UP_TOP's frame and vice versa.  Its intent ("ITEM 7": never show a
      clean frame for a damaged wagon) is now served without substitution --
      `_panel_state_text` names the camera that actually saw the damage, so the
      anomaly is still reported on the page, correctly attributed.
    * an unscoped `load/best_frame` lookup, applied to RIGHT_UP_TOP only.
      `load/best_frame.jpg` is a single file per wagon whose owning camera is
      recorded only in `metadata.json`, so when the load processor sourced it
      from LEFT_UP_TOP the RIGHT_UP_TOP panel showed a LEFT_UP_TOP frame --
      and since LEFT_UP_TOP fell through to its own cache frame, BOTH top
      panels ended up showing a LEFT_UP_TOP view.  That is why the duplication
      appeared specifically when there was NO damage: a damage track, when one
      existed, was resolved per-camera first and masked the defect.

    Both top cameras now take the SAME symmetric path, so neither can drift
    from the other again.
    """
    snap = None
    if camera_id == C.CAMERA_RIGHT_UP:
        # Slot name carries the camera: `right_best` is written from RIGHT_UP's
        # stream and `left_best` from LEFT_UP's (features/door/processor.py),
        # so these two are camera-scoped by construction.
        snap = ev.evidence_snapshot(evidence_root, gw_id, "door", "right_best")
    elif camera_id == C.CAMERA_LEFT_UP:
        snap = ev.evidence_snapshot(evidence_root, gw_id, "door", "left_best")
    elif camera_id in C.TOP_CAMERAS:
        # Own damage track first (filtered on camera_id via metadata), then this
        # camera's own load frame -- proven by `source_camera`, never assumed.
        snap = _top_damage_snapshot(evidence_root, gw_id, camera_id)
        if snap is None:
            snap = ev.evidence_snapshot_for_camera(
                evidence_root, gw_id, "load", "best_frame", camera_id)
    if snap and os.path.isfile(snap):
        return snap
    return _cache_mid_frame(cache_root, gw_id, camera_id)


def _make_camera_panel(*, camera_id, state_text, snap_path, authoritative,
                       missing, has_frames):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Table, TableStyle, Image

    border = colors.HexColor("#C62828") if authoritative else colors.HexColor("#9E9E9E")
    hdr_bg = colors.HexColor("#C62828") if authoritative else _brand.NAVY_MID
    cam_p = ParagraphStyle("PanelCam", fontSize=9, alignment=1, textColor=colors.white,
                           fontName="Helvetica-Bold", leading=12)
    st_p = ParagraphStyle("PanelState", fontSize=8, alignment=1,
                          textColor=_brand.TEXT_DARK, fontName="Helvetica-Bold", leading=11)
    tag = "  ● DETECTED HERE" if authoritative else ""
    header = Paragraph(f"{camera_id}{tag}", cam_p)

    def _placeholder(text, color):
        p = ParagraphStyle("PH", fontSize=13, alignment=1, textColor=color,
                           fontName="Helvetica-Bold")
        t = Table([[Paragraph(text, p)]], colWidths=[4.3 * inch], rowHeights=[2.4 * inch])
        t.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.Color(0.95, 0.95, 0.95)),
            ("BOX", (0, 0), (-1, -1), 1, colors.gray),
        ]))
        return t

    body = None
    if missing:
        body = _placeholder("NO FEED", colors.HexColor("#C62828"))
    elif snap_path and os.path.isfile(snap_path):
        try:
            img = Image(snap_path, width=4.3 * inch, height=2.4 * inch)
            img.hAlign = "CENTER"
            body = img
        except Exception:
            body = None
    if body is None and not missing:
        body = _placeholder("NOT VISIBLE", colors.gray)

    inner = Table([[header], [Paragraph(state_text, st_p)], [body]],
                  colWidths=[4.6 * inch])
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), hdr_bg),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 2, border),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (0, 0), 5),
        ("BOTTOMPADDING", (0, 0), (0, 0), 5),
    ]))
    return inner


def _wagon_anomaly_list(u: UnifiedWagonState) -> List[str]:
    """Concise list of the ACTUAL anomalies on a wagon (no normal states).

    Mirrors the anomaly definition in _wagon_is_anomalous + the per-camera
    severity rules in camera_reports so all surfaces agree.  Returns [] for a
    clean wagon.
    """
    out: List[str] = []
    if _brand.is_side_anomaly(u.right_door):
        out.append(f"RIGHT DOOR {_brand.format_door_status(u.right_door)}")
    elif u.right_door == C.DOOR_PARTIAL:
        out.append("RIGHT DOOR PARTIAL CLOSED")
    if _brand.is_side_anomaly(u.left_door):
        out.append(f"LEFT DOOR {_brand.format_door_status(u.left_door)}")
    elif u.left_door == C.DOOR_PARTIAL:
        out.append("LEFT DOOR PARTIAL CLOSED")
    if u.top_damage == C.DAMAGE_PRESENT:
        out.append("TOP DAMAGE")
    if u.side_damage == C.DAMAGE_PRESENT:
        out.append("SIDE DAMAGE")
    if u.classification == C.CLASS_WAGON:
        if u.wagon_identifier in (None, "", C.NO_DATA):
            out.append("OCR MISSING")
        if u.load_status == C.NO_DATA:
            out.append("LOAD NO DATA")
    return out


def _issue_summary_table(u: UnifiedWagonState, styles):
    """Concise issue strip: a single 'NO ISSUES' chip for a clean wagon, or one
    red chip listing only the ACTUAL anomalies.  Normal CLOSED/OK/EMPTY states
    are never printed."""
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle

    hdr_p = ParagraphStyle("IssHdr", fontSize=9, alignment=1, textColor=_brand.WHITE,
                           fontName="Helvetica-Bold", leading=12)
    ok_p = ParagraphStyle("IssOk", fontSize=10, alignment=1,
                          textColor=_brand.COLOR_OK_GREEN,
                          fontName="Helvetica-Bold", leading=13)
    bad_p = ParagraphStyle("IssBad", fontSize=10, alignment=1,
                           textColor=_brand.COLOR_NOT_OK,
                           fontName="Helvetica-Bold", leading=13)

    issues = _wagon_anomaly_list(u)
    if not issues:
        value_row = [Paragraph("NO ISSUES", ok_p)]
        anom = False
    else:
        value_row = [Paragraph("  |  ".join(issues), bad_p)]
        anom = True

    t = Table([[Paragraph("DETECTED ISSUES", hdr_p)], value_row],
              colWidths=[_brand.PAGE_BODY_WIDTH])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _brand.NAVY_DARK),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.5, _brand.SLATE_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, _brand.SLATE_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("BACKGROUND", (0, 1), (-1, 1),
         _brand.ISSUE_BG if anom else _brand.OK_BG),
    ]))
    return t


def _build_multi_angle_section(*, state, unified, evidence_root, cache_root,
                               styles, missing_cameras):
    """One dedicated page per anomalous wagon: header + issue summary + a 2x2
    grid of all four camera perspectives of that SAME global wagon.  Always
    shows all four views (representative wagon_cache frame when a camera has no
    detection); panels whose camera authoritatively detected the anomaly are
    emphasised with a red header + border."""
    from reportlab.lib.units import inch
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (
        Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak,
    )

    anomalous = []
    for idx, gw in enumerate(state.wagons, start=1):
        u = unified.get(gw.global_id)
        if _wagon_is_anomalous(u):
            anomalous.append((idx, gw, u))
    if not anomalous:
        return []

    missing_set = set(missing_cameras or [])
    elements = [PageBreak()]
    elements.append(Paragraph(
        "<b>Multi-Angle Wagon Evidence</b>", styles["ReportTitle"]))
    elements.append(Spacer(1, 0.05 * inch))
    elements.append(Paragraph(
        f"<b>Anomalous Wagons: {len(anomalous)} &mdash; all four camera "
        f"perspectives per wagon</b>", styles["ReportSubtitle"]))
    elements.append(Spacer(1, 0.15 * inch))

    hdr_style = ParagraphStyle("MAHeader", fontSize=15, alignment=TA_CENTER,
                               textColor=_brand.WHITE, fontName="Helvetica-Bold",
                               leading=19)

    for sr, gw, u in anomalous:
        damage_cams = _damage_track_cams(evidence_root, gw.global_id)
        auth = _authoritative_cams(u, damage_cams)

        # 1. Wagon header banner
        banner = Table([[Paragraph(
            f"Wagon No: {sr}  |  {gw.global_id}  |  {u.classification}",
            hdr_style)]], colWidths=[_brand.PAGE_BODY_WIDTH])
        banner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), _brand.NAVY_DARK),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ("BOX", (0, 0), (-1, -1), 1.5, _brand.NAVY_DARK),
        ]))

        # 2. Issue summary
        issue_t = _issue_summary_table(u, styles)

        # 3. 2x2 multi-angle grid
        grid_rows = []
        for cam_row in _MULTI_ANGLE_GRID:
            cells = []
            for cam in cam_row:
                has_frames = _cache_mid_frame(cache_root, gw.global_id, cam) is not None
                cells.append(_make_camera_panel(
                    camera_id=cam,
                    state_text=_panel_state_text(u, cam, damage_cams, has_frames),
                    snap_path=_panel_snapshot(u, cam, evidence_root, cache_root, gw.global_id),
                    authoritative=(cam in auth),
                    missing=(cam in missing_set),
                    has_frames=has_frames,
                ))
            grid_rows.append(cells)
        grid = Table(grid_rows, colWidths=[4.9 * inch, 4.9 * inch])
        grid.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ]))

        elements.append(KeepTogether([
            banner, Spacer(1, 0.08 * inch),
            issue_t, Spacer(1, 0.12 * inch),
            grid, PageBreak(),
        ]))
    return elements


# -----------------------------------------------------------------------------
# Damaged Wagon Report -- per-anomaly evidence pages (legacy 1017-1365)
# -----------------------------------------------------------------------------

def _build_evidence_section(*, vm, styles, evidence_root):
    """Return a list of reportlab elements implementing the per-anomaly
    "Damaged Wagon Report" section.  Skips entirely when no snapshots
    are available."""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph, Spacer, Table, TableStyle, Image, KeepTogether,
    )

    # Gather all anomalous images
    images: List[Dict[str, Any]] = []
    for d in vm.left_doors:
        if d.get("local_snapshot_path"):
            images.append({
                "path": d["local_snapshot_path"],
                "wagon_number": d["wagon_number"],
                "camera": "Left",
                "issue_type": "Damage" if "damage" in str(d.get("state","")).lower() else "Open Door",
                "label": d.get("state", ""),
            })
    for d in vm.right_doors:
        if d.get("local_snapshot_path"):
            images.append({
                "path": d["local_snapshot_path"],
                "wagon_number": d["wagon_number"],
                "camera": "Right",
                "issue_type": "Damage" if "damage" in str(d.get("state","")).lower() else "Open Door",
                "label": d.get("state", ""),
            })
    for d in vm.top_doors:
        if d.get("local_snapshot_path"):
            images.append({
                "path": d["local_snapshot_path"],
                "wagon_number": d["wagon_number"],
                "camera": "Right-Top",
                "issue_type": "Damage",
                "label": d.get("state", ""),
            })
    for d in vm.left_top_doors:
        if d.get("local_snapshot_path"):
            images.append({
                "path": d["local_snapshot_path"],
                "wagon_number": d["wagon_number"],
                "camera": "Left-Top",
                "issue_type": "Damage",
                "label": d.get("state", ""),
            })

    if not images:
        return []

    images.sort(key=lambda x: (
        int(x.get("wagon_number") or 999999),
        _brand.CAMERA_PRIORITY.get(x.get("camera", ""), 99),
    ))

    # Group by wagon_number preserving order
    by_wagon: "OrderedDict[int, List[Dict[str, Any]]]" = OrderedDict()
    for img in images:
        by_wagon.setdefault(int(img["wagon_number"]), []).append(img)

    elements: List[Any] = []
    elements.append(Paragraph(
        "<b>Damaged Wagon Report</b>",
        styles["ReportTitle"],
    ))
    elements.append(Spacer(1, 0.05 * inch))

    total_damaged = len(by_wagon)
    elements.append(Paragraph(
        f"<b>Total Damaged Wagons: {total_damaged}</b>",
        styles["ReportSubtitle"],
    ))
    elements.append(Spacer(1, 0.2 * inch))

    # Wagon# -> ocr lookup for the info table
    wagon_lookup = {w["wagon_sr_no"]: w for w in vm.merged_wagons}

    label_style = ParagraphStyle(
        "SnapLabel", fontSize=8, leading=10, alignment=1,
        textColor=_brand.TEXT_DARK, fontName="Helvetica-Bold",
    )

    timestamp = _now_ist().strftime("%d-%m-%Y %H:%M:%S IST")
    sn = 0
    for wagon_num, imgs in by_wagon.items():
        sn += 1
        wagon_info = wagon_lookup.get(wagon_num, {})
        ocr = wagon_info.get("ocr_wagon_number", "-") or "-"
        camera_angles = ", ".join(sorted(set(i["camera"] for i in imgs)))

        # Info table
        header_row = [
            Paragraph("<b>SN</b>",           styles["TableHeader"]),
            Paragraph("<b>Wagon ID</b>",     styles["TableHeader"]),
            Paragraph("<b>Wagon No.</b>",    styles["TableHeader"]),
            Paragraph("<b>Camera Angles</b>", styles["TableHeader"]),
            Paragraph("<b>Issues</b>",       styles["TableHeader"]),
            Paragraph("<b>Date &amp; Time</b>", styles["TableHeader"]),
        ]
        data_row = [
            Paragraph(f"{sn}.",                  styles["TableCell"]),
            Paragraph(str(wagon_num),            styles["TableCell"]),
            Paragraph(str(ocr),                  styles["TableCell"]),
            Paragraph(f"<b>{camera_angles}</b>", styles["TableCell"]),
            Paragraph(f"<b>{len(imgs)}</b>",     styles["TableCell"]),
            Paragraph(timestamp,                 styles["TableCell"]),
        ]
        info_t = Table([header_row, data_row],
                       colWidths=[0.5*inch, 0.8*inch, 2.2*inch,
                                  1.8*inch, 0.7*inch, 2.6*inch])
        info_t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), _brand.HEADER_GRAY),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))

        # Image cells
        cells: List[List[Any]] = []
        for img_info in imgs:
            try:
                display_label = _brand.CAMERA_LABELS.get(
                    img_info["camera"],
                    f"{img_info['camera']} Camera – {img_info['issue_type']}",
                )
                img = Image(img_info["path"])
                w, h = img.drawWidth, img.drawHeight
                if w > _brand.EVIDENCE_IMG_MAX_W:
                    s = _brand.EVIDENCE_IMG_MAX_W / w
                    w *= s; h *= s
                if h > _brand.EVIDENCE_IMG_MAX_H:
                    s = _brand.EVIDENCE_IMG_MAX_H / h
                    w *= s; h *= s
                img.drawWidth = w
                img.drawHeight = h
                cells.append([
                    Paragraph(f"<b>{display_label}</b>", label_style),
                    Spacer(1, 0.05 * inch),
                    img,
                ])
            except Exception as e:
                print(f"  [combined report] image load failed {img_info.get('path')}: {e}")

        if not cells:
            continue

        full_grid_w = 9.6 * inch
        col_w = 4.8 * inch
        if len(cells) == 1:
            grid = Table([cells], colWidths=[full_grid_w])
        else:
            grid_rows: List[List[Any]] = []
            paired = len(cells) - (len(cells) % 2)
            for i in range(0, paired, 2):
                grid_rows.append([cells[i], cells[i + 1]])
            if len(cells) % 2 == 1:
                grid_rows.append([cells[-1]])
            grid = Table(grid_rows, colWidths=[col_w, col_w])
            if len(cells) % 2 == 1:
                last = len(grid_rows) - 1
                grid.setStyle(TableStyle([("SPAN", (0, last), (1, last))]))

        grid.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E0E0E0")),
        ]))

        block = KeepTogether([
            info_t,
            Spacer(1, 0.15 * inch),
            grid,
            Spacer(1, 0.3 * inch),
        ])
        elements.append(block)

    return elements


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def build(
    *,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    output_dir: str,
    batch_key: str,
    source_video_urls: Optional[Dict[str, str]] = None,
    processed_video_urls: Optional[Dict[str, str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    evidence_root: Optional[str] = None,
    wagon_states_root: Optional[str] = None,
    cache_root: Optional[str] = None,
    missing_cameras: Optional[Sequence[str]] = None,
    camera_pdf_urls: Optional[Dict[str, str]] = None,
    logo_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Optional[str]]:
    """Stage 5 public entry.  Writes JSON always; PDF if reportlab is OK.

    Returns:
        {"json_path": "...", "pdf_path": "..." | None,
         "report_integrity": {...}}   # completeness vs the canonical timeline
    """
    os.makedirs(output_dir, exist_ok=True)
    missing_cameras = list(missing_cameras or [])

    # Completeness audit BEFORE rendering, so an incomplete report is reported
    # as such rather than produced silently. `strict` is opt-in: a train that
    # reached Stage 5 should still deliver whatever it has, and a loud
    # high-severity diagnostic naming the ids beats refusing to write anything.
    _wagons_in_order, _synth = canonical_wagons(state, unified)
    integrity = audit_report_integrity(
        state=state, wagons_in_order=_wagons_in_order, unified=unified,
        synthesized=_synth, strict=_strict_integrity(), verbose=verbose)

    # Always build the view-model.  Even if the PDF fails, the JSON has it.
    vm = _adapter.build_legacy_view_model(
        state=state, unified=unified,
        wagon_states_root=wagon_states_root,
        evidence_root=evidence_root,
        missing_cameras=missing_cameras,
    )

    t0 = time.time()
    json_doc = _build_json(
        state=state, unified=unified, batch_key=batch_key,
        source_video_urls=source_video_urls,
        processed_video_urls=processed_video_urls,
        extra_metadata=extra_metadata,
        evidence_root=evidence_root,
        legacy_view_model=vm,
    )
    json_path = os.path.join(output_dir, "combined_train_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_doc, f, indent=2, default=str)
    if verbose:
        print(f"[STAGE5] wrote {json_path}")

    # ---- report manifest: the AUTHORITATIVE record of PDF evidence ----
    # Built once, consumed by the PDF, and written to disk. The PDF selects
    # nothing on its own, so the audit cannot disagree with the pages.
    report_manifest: Dict[str, Any] = {}
    try:
        _dmg = WG.damage_from_evidence(evidence_root=evidence_root,
                                       state=state, verbose=verbose)
        report_manifest = WG.build_manifest(
            state=state, cache_root=cache_root,
            camera_meta=_camera_meta_for_manifest(state, wagon_states_root),
            damage_by_wagon=_dmg, verbose=verbose)
        _mp = os.path.join(output_dir, "combined_report_manifest.json")
        with open(_mp, "w", encoding="utf-8") as f:
            json.dump(report_manifest, f, indent=2, default=str)
        if verbose:
            print(f"[STAGE5] wrote {_mp}")
    except Exception as e:  # noqa: BLE001 -- the PDF must still be attempted
        print(f"[STAGE5] report manifest FAILED: {type(e).__name__}: {e}")
        report_manifest = {}

    pdf_path: Optional[str] = os.path.join(output_dir, "combined_train_report.pdf")
    try:
        _build_pdf(
            report_manifest=report_manifest,
            state=state, unified=unified, vm=vm,
            batch_key=batch_key, output_pdf=pdf_path,
            evidence_root=evidence_root,
            source_video_urls=dict(source_video_urls or {}),
            processed_video_urls=dict(processed_video_urls or {}),
            camera_pdf_urls=dict(camera_pdf_urls or {}),
            logo_path=logo_path,
            missing_cameras=missing_cameras,
            cache_root=cache_root,
        )
        if verbose:
            print(f"[STAGE5] wrote {pdf_path}")
    except Exception as e:
        print(f"[STAGE5] PDF generation FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        pdf_path = None

    if verbose:
        print(f"[STAGE5] done in {time.time() - t0:.1f}s")
    return {"json_path": json_path, "pdf_path": pdf_path,
            "report_manifest": report_manifest,
            "report_integrity": integrity}


# -----------------------------------------------------------------------------
# Wagon-by-wagon evidence grid: TOP cameras, then SIDE cameras, per canonical GW
# -----------------------------------------------------------------------------

def _grid_image(path, styles, *, cell_w, cell_h):
    """One evidence cell. Never scales an image beyond the cell."""
    from reportlab.platypus import Image, Paragraph
    from reportlab.lib.units import inch
    try:
        img = Image(path, width=cell_w, height=cell_h, kind="proportional")
        img.hAlign = "CENTER"
        return img
    except Exception:                                            # noqa: BLE001
        return Paragraph("<b>IMAGE UNREADABLE</b>", styles["_gridmiss"])


def _grid_placeholder(reason, styles):
    """An explicitly EMPTY slot. Never a repeated image.

    Four copies of one frame reads as four pieces of evidence, so a missing slot
    says so and names the reason the manifest recorded.
    """
    from reportlab.platypus import Paragraph
    return Paragraph(
        f"<b>NO VALID FRAME</b><br/><font size=6>{reason or 'unavailable'}</font>",
        styles["_gridmiss"])


def _camera_row(cam_slots, camera_id, styles, *, cell_w, cell_h):
    """`[label, cell, cell, cell, cell]` for one camera's four slots."""
    from reportlab.platypus import Paragraph
    row = [Paragraph(f"<b>{camera_id}</b>", styles["_gridcam"])]
    for s in cam_slots:
        if s.get("available") and s.get("image_path"):
            row.append(_grid_image(s["image_path"], styles,
                                   cell_w=cell_w, cell_h=cell_h))
        else:
            row.append(_grid_placeholder(s.get("unavailable_reason"), styles))
    while len(row) < 1 + WG.SLOTS_PER_CAMERA:
        row.append(_grid_placeholder("missing slot", styles))
    return row


def _caption_row(cam_slots, styles):
    """Per-image caption: camera, frame, timestamp. The frame number printed
    here is read from the manifest, which read it from the FILENAME -- so the
    number under an image is provably the file above it."""
    from reportlab.platypus import Paragraph
    row = [Paragraph("", styles["_gridcap"])]
    for s in cam_slots:
        if s.get("available"):
            t = s.get("timestamp_sec")
            row.append(Paragraph(
                f"{s['camera_id']}<br/>f{s.get('source_frame')}"
                + (f" @ {float(t):.2f}s" if t is not None else ""),
                styles["_gridcap"]))
        else:
            row.append(Paragraph(f"{s['camera_id']}<br/>-", styles["_gridcap"]))
    while len(row) < 1 + WG.SLOTS_PER_CAMERA:
        row.append(Paragraph("", styles["_gridcap"]))
    return row


def _camera_block(entry, cameras, title, styles, *, cell_w, cell_h):
    """A titled 2x4 grid for one camera pair (TOP or SIDE)."""
    from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    data = [[Paragraph(f"<b>{title}</b>", styles["_gridsec"])]
            + [""] * WG.SLOTS_PER_CAMERA]
    for cam in cameras:
        slots = entry["cameras"].get(cam, [])
        data.append(_camera_row(slots, cam, styles,
                                cell_w=cell_w, cell_h=cell_h))
        data.append(_caption_row(slots, styles))

    t = Table(data, colWidths=[0.95 * inch] + [cell_w + 6] * WG.SLOTS_PER_CAMERA)
    t.setStyle(TableStyle([
        ("SPAN", (0, 0), (-1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12233f")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 1), (-1, -1), 0.4, colors.HexColor("#b9c4d4")),
        ("BOX", (0, 0), (-1, -1), 0.9, colors.HexColor("#12233f")),
        ("TOPPADDING", (0, 1), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
    ]))
    return [t, Spacer(1, 0.10 * inch)]


def _wagon_header(entry, styles):
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    gw = entry["global_id"]
    sub = (f"{entry.get('classification') or '-'}   "
           f"wagon {entry.get('wagon_index')} of {entry.get('total_wagons')}   "
           f"evidence {entry.get('available_images')}/"
           f"{entry.get('expected_images')}")
    t = Table([[Paragraph(f"<b>GLOBAL WAGON {gw}</b>", styles["_gridhdr"])],
               [Paragraph(sub, styles["_gridsub"])]],
              colWidths=[10.6 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f2038")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#e8edf4")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _register_grid_styles(styles):
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib import colors
    for name, kw in (
        ("_gridhdr", dict(fontSize=17, textColor=colors.white,
                          alignment=TA_CENTER, fontName="Helvetica-Bold")),
        ("_gridsub", dict(fontSize=9, textColor=colors.HexColor("#33465f"),
                          alignment=TA_CENTER)),
        ("_gridsec", dict(fontSize=11, textColor=colors.white,
                          alignment=TA_CENTER, fontName="Helvetica-Bold")),
        ("_gridcam", dict(fontSize=8, alignment=TA_CENTER,
                          fontName="Helvetica-Bold")),
        ("_gridcap", dict(fontSize=6.5, alignment=TA_CENTER,
                          textColor=colors.HexColor("#44556b"))),
        ("_gridmiss", dict(fontSize=7, alignment=TA_CENTER,
                           textColor=colors.HexColor("#b03030"))),
    ):
        if name in styles:
            continue
        st = ParagraphStyle(name, **kw)
        # `styles` is a plain dict in this module (not a reportlab
        # StyleSheet1), so assignment is the portable way in. `.add()` is
        # supported too, for a caller that passes a real stylesheet.
        try:
            styles[name] = st
        except TypeError:
            styles.add(st)
    return styles


def build_wagon_grid_section(manifest, styles, *, one_page_per_wagon=False):
    """Wagon-by-wagon pages: header, TOP grid, SIDE grid, next wagon.

    Consumes the MANIFEST rather than selecting frames itself, so the audit and
    the PDF cannot disagree about which image is on which page.

    Logical order is fixed -- `WG.TOP_ORDER` then `WG.SIDE_ORDER`, from the
    selector -- and is preserved whether or not a wagon's sixteen images fit on
    one physical page. `one_page_per_wagon=False` (the default) puts the header
    and the TOP grid on one page and the SIDE grid on the next, so nothing is
    shrunk to the point of being unreadable.
    """
    from reportlab.platypus import PageBreak, Spacer, Paragraph
    from reportlab.lib.units import inch

    _register_grid_styles(styles)
    out = []
    wagons = manifest.get("wagons") or []
    if not wagons:
        return out

    # Sized so four images fit a landscape-A4 row with captions and still be
    # legible; two rows per page, which is why TOP and SIDE take a page each.
    cell_w = 2.35 * inch
    cell_h = 1.55 * inch

    out.append(PageBreak())
    out.append(Paragraph(
        "<b>WAGON EVIDENCE &mdash; ALL FOUR CAMERAS PER CANONICAL WAGON</b>",
        styles["_gridsec"]))
    out.append(Spacer(1, 0.12 * inch))

    for i, entry in enumerate(wagons):
        if i:
            out.append(PageBreak())
        out.append(_wagon_header(entry, styles))
        out.append(Spacer(1, 0.08 * inch))
        out.extend(_camera_block(entry, WG.TOP_ORDER, "TOP CAMERAS",
                                 styles, cell_w=cell_w, cell_h=cell_h))
        if not one_page_per_wagon:
            out.append(PageBreak())
            out.append(_wagon_header(entry, styles))
            out.append(Spacer(1, 0.08 * inch))
        out.extend(_camera_block(entry, WG.SIDE_ORDER, "SIDE CAMERAS",
                                 styles, cell_w=cell_w, cell_h=cell_h))
    return out


def build_damage_summary_section(manifest, styles):
    """FINAL section: damage evidence only, grouped by canonical GW_n.

    Only damage. The wagon pages above already carry the 16-image grid, so
    repeating it here would bury the findings it exists to highlight. Rows come
    from the manifest's `damage_by_wagon`, which came from Stage-3 evidence --
    no damage inference, and the same GW_n identity as the main section and the
    processed videos.
    """
    from reportlab.platypus import (Table, TableStyle, Paragraph, Spacer,
                                    PageBreak)
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    _register_grid_styles(styles)
    by_wagon = manifest.get("damage_by_wagon") or {}
    out = [PageBreak(),
           Paragraph("<b>DAMAGE SUMMARY</b>", styles["_gridsec"]),
           Spacer(1, 0.14 * inch)]
    if not by_wagon:
        out.append(Paragraph(
            "No damage was detected on any canonical wagon.",
            styles["_gridsub"]))
        return out

    for gw in manifest.get("wagons_with_damage") or sorted(by_wagon):
        rows = by_wagon.get(gw) or []
        if not rows:
            continue
        out.append(_wagon_header({"global_id": gw, "classification": "",
                                  "wagon_index": "", "total_wagons": "",
                                  "available_images": len(rows),
                                  "expected_images": len(rows)}, styles))
        out.append(Spacer(1, 0.06 * inch))
        data = [[Paragraph("<b>CAMERA</b>", styles["_gridcam"]),
                 Paragraph("<b>CLASS</b>", styles["_gridcam"]),
                 Paragraph("<b>CONF</b>", styles["_gridcam"]),
                 Paragraph("<b>FRAME</b>", styles["_gridcam"]),
                 Paragraph("<b>EVIDENCE</b>", styles["_gridcam"])]]
        for r in rows:
            conf = r.get("confidence")
            cell = (_grid_image(r["image_path"], styles,
                                cell_w=2.6 * inch, cell_h=1.7 * inch)
                    if r.get("image_available")
                    else _grid_placeholder(r.get("unavailable_reason"), styles))
            data.append([
                Paragraph(str(r.get("camera_id") or "-"), styles["_gridcap"]),
                Paragraph(str(r.get("class_name") or "-"), styles["_gridcap"]),
                Paragraph(f"{float(conf):.2f}" if conf is not None else "-",
                          styles["_gridcap"]),
                Paragraph(str(r.get("frame") if r.get("frame") is not None
                              else "-"), styles["_gridcap"]),
                cell])
        t = Table(data, colWidths=[1.5 * inch, 1.9 * inch, 0.8 * inch,
                                   0.9 * inch, 2.8 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7a1c1c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8b0b0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        out.append(t)
        out.append(Spacer(1, 0.16 * inch))
    return out
