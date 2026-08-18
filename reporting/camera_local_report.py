"""Camera-LOCAL PDF report (sequential mode).

Additive: the existing global builders (`camera_reports.py`,
`combined_train_report.py`) are untouched. Those require a GlobalTrainState;
this one deliberately does not, so a camera can publish its report the moment
it is sealed, before any other camera has arrived.

Wagons are identified by CAMERA-LOCAL ids only -- `L_RIGHT_UP_1`,
`L_RIGHT_UP_2`, ... A `GW_n` is never invented here: global ids do not exist
until assembly, and printing one would be a lie about what this camera knows.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

SCHEMA = "wagon_eye.camera_local_report.v1"

_NAVY = colors.HexColor("#1A237E")
_GREY = colors.HexColor("#ECEFF1")
_AMBER = colors.HexColor("#FFF8E1")


def _ist_now() -> str:
    return datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime(
        "%d-%m-%Y %H:%M:%S IST")


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontSize=18,
                                textColor=colors.white, alignment=1,
                                spaceAfter=0),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=9,
                              textColor=colors.white, alignment=1),
        "h": ParagraphStyle("h", parent=ss["Heading2"], fontSize=12,
                            textColor=_NAVY, spaceBefore=10, spaceAfter=6),
        "b": ParagraphStyle("b", parent=ss["Normal"], fontSize=8.5),
        "cell": ParagraphStyle("c", parent=ss["Normal"], fontSize=7.5),
        "hd": ParagraphStyle("hd", parent=ss["Normal"], fontSize=7.5,
                             textColor=colors.white),
    }


def _banner(camera_id: str, st) -> Table:
    t = Table([[Paragraph(f"{camera_id} — CAMERA-LOCAL REPORT", st["title"])],
               [Paragraph(f"Generated {_ist_now()} · camera-local ids only "
                          f"(L_{camera_id}_n) · no global wagon ids assigned",
                          st["sub"])]],
              colWidths=[7.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def _kv_table(rows: List[List[str]], st) -> Table:
    data = [[Paragraph(str(a), st["b"]), Paragraph(str(b), st["b"])]
            for a, b in rows]
    t = Table(data, colWidths=[2.6 * inch, 4.6 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), _GREY),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _segment_table(report: Dict[str, Any], st) -> Table:
    feat = report.get("feature_summary") or {}
    features = sorted(feat.keys())
    head = ["LOCAL ID", "IDX", "FRAMES", "TIME (s)", "CLASS"] + \
           [f.upper() for f in features]
    data = [[Paragraph(h, st["hd"]) for h in head]]
    for s in report.get("local_segments") or []:
        row = [
            s.get("local_id", ""), str(s.get("index", "")),
            f"{s.get('start_frame')}–{s.get('end_frame')}",
            f"{s.get('start_time', 0):.2f}–{s.get('end_time', 0):.2f}",
            s.get("label", "UNKNOWN"),
        ]
        for f in features:
            row.append(str((feat.get(f) or {}).get(s.get("local_id"), "—")))
        data.append([Paragraph(str(c), st["cell"]) for c in row])
    widths = [1.35 * inch, 0.4 * inch, 1.1 * inch, 1.2 * inch, 0.95 * inch]
    rest = max(0.7 * inch, (7.2 * inch - sum(widths)) / max(1, len(features)))
    t = Table(data, colWidths=widths + [rest] * len(features), repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _GREY]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def build_camera_report(
    *,
    camera_id: str,
    report: Dict[str, Any],
    output_path: str,
    verbose: bool = True,
) -> Optional[str]:
    """Render one camera's own PDF from its bundle report dict.

    `report` is the `camera_report.json` payload written by camera_runner.
    Requires nothing from any other camera and no GlobalTrainState.
    Returns the path, or None if rendering failed (never raises -- a report
    problem must not un-seal a camera whose inference succeeded).
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".",
                    exist_ok=True)
        st = _styles()
        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            leftMargin=0.5 * inch, rightMargin=0.5 * inch,
            topMargin=0.5 * inch, bottomMargin=0.5 * inch,
            title=f"{camera_id} camera-local report")

        segs = report.get("local_segments") or []
        calls = report.get("feature_yolo_calls") or {}
        flow: List[Any] = [_banner(camera_id, st), Spacer(1, 12)]

        flow.append(Paragraph("CAMERA SUMMARY", st["h"]))
        flow.append(_kv_table([
            ["Camera", camera_id +
             (" (MASTER)" if report.get("is_master") else " (support)")],
            ["Source fps / frames",
             f"{report.get('fps')} / {report.get('total_frames')}"],
            ["Raw gap detections", report.get("raw_detections", "—")],
            ["Accepted gaps", report.get("accepted_gaps", 0)],
            ["Rejected gaps", report.get("rejected_gaps", 0)],
            ["Recovered gaps", report.get("recovered_gaps", 0)],
            ["Reclassified after recovery",
             report.get("reclassified_after_recovery", False)],
            ["Local wagon segments", len(segs)],
            ["Frames materialized", report.get("frames_materialized", 0)],
            ["Feature YOLO calls",
             ", ".join(f"{k}={v}" for k, v in sorted(calls.items())) or "—"],
        ], st))
        flow.append(Spacer(1, 10))

        note = Table([[Paragraph(
            "<b>Camera-local report.</b> Segments are identified by this "
            f"camera's own ids (L_{camera_id}_n) in its own absolute frame "
            "numbering. Global wagon ids (GW_n) are assigned only during "
            "global assembly, once every required camera has been sealed.",
            st["b"])]], colWidths=[7.2 * inch])
        note.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), _AMBER),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.orange),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        flow.append(note)

        flow.append(PageBreak())
        flow.append(Paragraph("CAMERA-LOCAL WAGON SEGMENTS", st["h"]))
        if segs:
            flow.append(_segment_table(report, st))
        else:
            flow.append(Paragraph("No local segments were produced.", st["b"]))

        notes = report.get("notes") or []
        if notes:
            flow.append(Spacer(1, 10))
            flow.append(Paragraph("PROCESSING NOTES", st["h"]))
            for n in notes:
                flow.append(Paragraph(f"• {n}", st["b"]))

        doc.build(flow)
        if verbose:
            print(f"[SEQ/{camera_id}] camera PDF -> {output_path}")
        return output_path
    except Exception as e:              # a report must never un-seal a camera
        print(f"[SEQ/{camera_id}] camera PDF FAILED: {type(e).__name__}: {e}")
        return None
