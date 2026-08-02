import json
import time
from typing import Any

from ai.models.pdf_summary_models import (
    HormonalPanelSummary,
    HormoneAiInsights,
    HormoneBiomarker,
    HormoneBiomarkers,
    HormoneNextSteps,
    PdfSummaryRequest,
    PdfSummaryResponse,
)
from ai.utils.db import get_lab_reports
from ai.utils.llm_call import llm_call


RETRYABLE_LLM_STATUS_CODES = {429, 500, 502, 503, 529}
LLM_RETRY_DELAYS_SECONDS = (1.0, 2.0)
MAX_PDF_TEXT_CHARS = 18000
MAX_CHAT_PDF_TEXT_CHARS = 12000
NEEDS_ATTENTION_BIOMARKERS = ("Estradiol (E2)", "Cortisol AM")
NORMAL_RESULT_BIOMARKERS = ("Progesterone", "FSH", "LH")


HORMONE_PANEL_SYSTEM_PROMPT = """You are a careful lab-report analysis assistant for a hormone panel UI.

Rules:
- Analyze only the lab report text provided.
- Do not diagnose, prescribe, or claim medical certainty.
- Use the report's actual values and reference ranges when present.
- If a value is missing, write "Not found" and explain that it was not available in the report text.
- Return valid JSON only. Do not add markdown, code fences, or extra commentary.
"""


def get_lab_reports_for_user(user_id: int) -> dict[str, Any]:
    """Get all lab reports for a user from database."""
    reports = get_lab_reports(user_id)
    return {
        "user_id": user_id,
        "total": len(reports),
        "reports": [
            {
                "id": r["id"],
                "lab_report": r["lab_report"],
                "panel": r["panel"],
                "analysis_status": r["analysis_status"],
                "created_at": str(r["created_at"]) if r.get("created_at") else None,
            }
            for r in reports
        ],
    }


def summarize_pdf(user_id: int, report_id: int | None = None) -> PdfSummaryResponse:
    """Summarize lab report for a user using only data already in MySQL.

    This never fetches the PDF file over HTTP. If the report's biomarkers
    have already been analyzed and stored in the `lab_reports` table, that
    stored analysis is returned. Otherwise a fallback summary is returned
    with `analysis_status` reflecting the database state (pending/processing/failed).
    """
    reports = get_lab_reports(user_id)

    if not reports:
        raise ValueError(f"No lab reports found for user {user_id}")

    # Find specific report or use latest
    if report_id:
        report = next((r for r in reports if r["id"] == report_id), None)
        if not report:
            raise ValueError(f"Lab report {report_id} not found for user {user_id}")
    else:
        report = reports[0]  # Latest report

    if report.get("biomarkers"):
        return _build_response_from_db(report)

    return _build_pending_response(report)


def _build_response_from_db(report: dict[str, Any]) -> PdfSummaryResponse:
    """Build response from existing database analysis."""
    biomarkers_data = report.get("biomarkers") or {}
    ai_insights_data = report.get("ai_insights") or {}
    next_steps_data = report.get("next_steps") or {}
    
    # Parse JSON if stored as string
    if isinstance(biomarkers_data, str):
        biomarkers_data = json.loads(biomarkers_data)
    if isinstance(ai_insights_data, str):
        ai_insights_data = json.loads(ai_insights_data)
    if isinstance(next_steps_data, str):
        next_steps_data = json.loads(next_steps_data)
    
    default_note = "Not available for this lab report."
    summary = HormonalPanelSummary(
        panel=report.get("panel") or "Hormonal Panel",
        biomarkers=HormoneBiomarkers(
            needs_attention=[
                HormoneBiomarker(**b) for b in biomarkers_data.get("needs_attention", [])
            ],
            normal_results=[
                HormoneBiomarker(**b) for b in biomarkers_data.get("normal_results", [])
            ],
        ),
        ai_insights=HormoneAiInsights(
            cross_data_context=ai_insights_data.get("cross_data_context", []),
            estradiol_elevation=ai_insights_data.get("estradiol_elevation", default_note),
            cortisol_near_ceiling=ai_insights_data.get("cortisol_near_ceiling", default_note),
            hormonal_balance=ai_insights_data.get("hormonal_balance", default_note),
        ),
        next_steps=HormoneNextSteps(
            recommendations=next_steps_data.get("recommendations", []),
            medical_disclaimer=next_steps_data.get(
                "medical_disclaimer",
                "This AI summary is for informational purposes only and is not medical advice. "
                "A qualified healthcare professional should interpret abnormal or concerning lab results.",
            ),
        ),
    )
    
    return PdfSummaryResponse(
        report_id=report["id"],
        source_path=report.get("lab_report") or "",
        content_type="application/pdf",
        file_size_bytes=0,
        text_extracted=True,
        summary=summary,
    )


def _build_pending_response(report: dict[str, Any]) -> PdfSummaryResponse:
    """Build a placeholder response when biomarkers haven't been analyzed yet.

    Uses only the `analysis_status` column already in MySQL - no HTTP calls.
    """
    status = report.get("analysis_status") or "pending"
    status_messages = {
        "pending": "This lab report has not been analyzed yet.",
        "processing": "This lab report is currently being analyzed.",
        "failed": "Analysis of this lab report failed. Please try uploading it again.",
    }
    message = status_messages.get(status, "No analysis is available for this lab report yet.")

    summary = HormonalPanelSummary(
        panel=report.get("panel") or "Hormonal Panel",
        biomarkers=HormoneBiomarkers(needs_attention=[], normal_results=[]),
        ai_insights=HormoneAiInsights(
            cross_data_context=[],
            estradiol_elevation=message,
            cortisol_near_ceiling=message,
            hormonal_balance=message,
        ),
        next_steps=HormoneNextSteps(
            recommendations=[],
            medical_disclaimer=(
                "This AI summary is for informational purposes only and is not medical advice. "
                "A qualified healthcare professional should interpret abnormal or concerning lab results."
            ),
        ),
    )

    return PdfSummaryResponse(
        report_id=report["id"],
        source_path=report.get("lab_report") or "",
        content_type="application/pdf",
        file_size_bytes=0,
        text_extracted=False,
        summary=summary,
    )


def fetch_chat_lab_report_context(user_id: int, report_id: int | None = None) -> dict[str, Any]:
    """Fetch lab report context for chat using only data already in MySQL.

    Never fetches the PDF file over HTTP. Returns the stored biomarkers/
    ai_insights/next_steps JSON columns when available, or the analysis
    status otherwise.
    """
    reports = get_lab_reports(user_id)

    if not reports:
        return {
            "report_id": None,
            "error": f"No lab reports found for user {user_id}",
        }

    # Find specific report or use latest
    if report_id:
        report = next((r for r in reports if r["id"] == report_id), None)
        if not report:
            return {
                "report_id": report_id,
                "error": f"Lab report {report_id} not found for user {user_id}",
            }
    else:
        report = reports[0]

    if not report.get("biomarkers"):
        return {
            "report_id": report["id"],
            "analysis_status": report.get("analysis_status") or "pending",
            "text_extracted": False,
            "note": "No AI analysis has been stored yet for this lab report.",
        }

    biomarkers_data = report.get("biomarkers") or {}
    ai_insights_data = report.get("ai_insights") or {}
    next_steps_data = report.get("next_steps") or {}
    if isinstance(biomarkers_data, str):
        biomarkers_data = json.loads(biomarkers_data)
    if isinstance(ai_insights_data, str):
        ai_insights_data = json.loads(ai_insights_data)
    if isinstance(next_steps_data, str):
        next_steps_data = json.loads(next_steps_data)

    return {
        "report_id": report["id"],
        "analysis_status": report.get("analysis_status") or "completed",
        "text_extracted": True,
        "panel": report.get("panel") or "Hormonal Panel",
        "biomarkers": biomarkers_data,
        "ai_insights": ai_insights_data,
        "next_steps": next_steps_data,
    }


def _generate_hormonal_panel_summary(report_text: str) -> HormonalPanelSummary:
    if not report_text.strip():
        return _fallback_panel_summary()

    prompt = _build_hormone_panel_prompt(report_text)
    response_text = _call_summary_llm(prompt)
    parsed = _parse_summary_response(response_text)
    if parsed:
        return parsed

    return _fallback_panel_summary(report_text)


def _build_hormone_panel_prompt(report_text: str) -> str:
    clipped_text = report_text[:MAX_PDF_TEXT_CHARS]
    return f"""Analyze this lab report text and generate a Hormonal Panel response.

Lab report text:
{clipped_text}

Return JSON with exactly this structure:
{{
  "panel": "Hormonal Panel",
  "biomarkers": {{
    "needs_attention": [
      {{"name": "Estradiol (E2)", "value": "...", "reference_range": "...", "status": "High", "interpretation": "..."}},
      {{"name": "Cortisol AM", "value": "...", "reference_range": "...", "status": "Borderline", "interpretation": "..."}}
    ],
    "normal_results": [
      {{"name": "Progesterone", "value": "...", "reference_range": "...", "status": "Normal", "interpretation": "..."}},
      {{"name": "FSH", "value": "...", "reference_range": "...", "status": "Normal", "interpretation": "..."}},
      {{"name": "LH", "value": "...", "reference_range": "...", "status": "Normal", "interpretation": "..."}}
    ]
  }},
  "ai_insights": {{
    "cross_data_context": ["...", "...", "..."],
    "estradiol_elevation": "...",
    "cortisol_near_ceiling": "...",
    "hormonal_balance": "..."
  }},
  "next_steps": {{
    "recommendations": ["...", "...", "...", "..."],
    "medical_disclaimer": "..."
  }}
}}

Requirements:
- Needs attention must include exactly Estradiol (E2) and Cortisol AM.
- Normal results must include exactly Progesterone, FSH, and LH.
- Cross data context should connect the hormone values to broader report context when available.
- Next steps should be practical, cautious recommendations.
- Include a medical disclaimer that this is not medical advice and a clinician should interpret abnormal results.
"""


def _call_summary_llm(prompt: str) -> str:
    attempts = len(LLM_RETRY_DELAYS_SECONDS) + 1

    for attempt in range(attempts):
        try:
            return llm_call(
                prompt=prompt,
                system=HORMONE_PANEL_SYSTEM_PROMPT,
                max_tokens=2400,
            )
        except Exception as exc:
            is_last_attempt = attempt == attempts - 1
            if not _is_retryable_llm_error(exc):
                raise
            if is_last_attempt:
                return ""
            time.sleep(LLM_RETRY_DELAYS_SECONDS[attempt])

    return ""


def _parse_summary_response(text: str) -> HormonalPanelSummary | None:
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(line for line in lines if not line.startswith("```")).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None

    try:
        payload = _coerce_summary_payload(payload)
        return _normalize_panel_summary(HormonalPanelSummary.model_validate(payload))
    except Exception:
        return None


def _coerce_summary_payload(payload: Any) -> dict:
    if not isinstance(payload, dict):
        return {}

    coerced = dict(payload)
    coerced["panel"] = "Hormonal Panel"
    coerced["biomarkers"] = _coerce_biomarkers(coerced.get("biomarkers"))
    coerced["ai_insights"] = _coerce_ai_insights(coerced.get("ai_insights"))
    coerced["next_steps"] = _coerce_next_steps(coerced.get("next_steps"))
    return coerced


def _coerce_biomarkers(value: Any) -> dict:
    if isinstance(value, dict):
        needs_attention = value.get("needs_attention") or []
        normal_results = value.get("normal_results") or []
    elif isinstance(value, list):
        needs_attention = [
            item for item in value
            if _biomarker_key(str(item.get("name", ""))) in {"estradiole2", "estradiol", "cortisolam", "cortisol"}
        ]
        normal_results = [
            item for item in value
            if _biomarker_key(str(item.get("name", ""))) in {"progesterone", "fsh", "lh"}
        ]
    else:
        needs_attention = []
        normal_results = []

    return {
        "needs_attention": [_coerce_biomarker(item) for item in needs_attention],
        "normal_results": [_coerce_biomarker(item) for item in normal_results],
    }


def _coerce_biomarker(value: Any) -> dict:
    if not isinstance(value, dict):
        return {
            "name": "Not found",
            "value": "Not found",
            "reference_range": "Not found",
            "status": "Not found",
            "interpretation": "Not found in the extracted lab report text.",
        }

    raw_value = value.get("value", "Not found")
    unit = value.get("unit")
    if raw_value is None:
        display_value = "Not found"
    elif unit:
        display_value = f"{raw_value} {unit}"
    else:
        display_value = str(raw_value)

    interpretation = value.get("interpretation") or value.get("note") or value.get("summary")
    if not interpretation:
        interpretation = "Review this result with the lab reference range and clinical context."

    return {
        "name": str(value.get("name") or "Not found"),
        "value": display_value,
        "reference_range": str(value.get("reference_range") or "Not found"),
        "status": str(value.get("status") or "Not found"),
        "interpretation": str(interpretation),
    }


def _coerce_ai_insights(value: Any) -> dict:
    if isinstance(value, dict):
        cross_data_context = value.get("cross_data_context") or []
        if isinstance(cross_data_context, str):
            cross_data_context = [cross_data_context]
        return {
            "cross_data_context": [str(item) for item in cross_data_context],
            "estradiol_elevation": str(value.get("estradiol_elevation") or "Estradiol should be interpreted against cycle timing and the lab reference range."),
            "cortisol_near_ceiling": str(value.get("cortisol_near_ceiling") or "Cortisol AM was not clearly available or should be interpreted with symptoms and timing."),
            "hormonal_balance": str(value.get("hormonal_balance") or "Review estradiol, progesterone, FSH, LH, and cortisol together for overall context."),
        }

    context = [str(value)] if value else []
    return {
        "cross_data_context": context,
        "estradiol_elevation": "Estradiol should be interpreted against cycle timing and the lab reference range.",
        "cortisol_near_ceiling": "Cortisol AM was not clearly available or should be interpreted with symptoms and timing.",
        "hormonal_balance": "Review estradiol, progesterone, FSH, LH, and cortisol together for overall context.",
    }


def _coerce_next_steps(value: Any) -> dict:
    disclaimer = (
        "This AI summary is for informational purposes only and is not medical advice. "
        "A qualified healthcare professional should interpret abnormal or concerning lab results."
    )
    if isinstance(value, dict):
        recommendations = value.get("recommendations") or []
        if isinstance(recommendations, str):
            recommendations = [recommendations]
        return {
            "recommendations": [str(item) for item in recommendations],
            "medical_disclaimer": str(value.get("medical_disclaimer") or disclaimer),
        }
    if isinstance(value, list):
        return {
            "recommendations": [str(item) for item in value],
            "medical_disclaimer": disclaimer,
        }
    return {
        "recommendations": [],
        "medical_disclaimer": disclaimer,
    }

def _normalize_panel_summary(summary: HormonalPanelSummary) -> HormonalPanelSummary:
    fallback = _fallback_panel_summary()
    summary.biomarkers.needs_attention = _ordered_biomarkers(
        summary.biomarkers.needs_attention,
        NEEDS_ATTENTION_BIOMARKERS,
        fallback.biomarkers.needs_attention,
    )
    summary.biomarkers.normal_results = _ordered_biomarkers(
        summary.biomarkers.normal_results,
        NORMAL_RESULT_BIOMARKERS,
        fallback.biomarkers.normal_results,
    )
    return summary


def _ordered_biomarkers(
    biomarkers: list[HormoneBiomarker],
    expected_names: tuple[str, ...],
    fallback_biomarkers: list[HormoneBiomarker],
) -> list[HormoneBiomarker]:
    ordered = []
    for index, expected_name in enumerate(expected_names):
        match = _find_biomarker(biomarkers, expected_name)
        ordered.append(match or fallback_biomarkers[index])
    return ordered


def _find_biomarker(
    biomarkers: list[HormoneBiomarker],
    expected_name: str,
) -> HormoneBiomarker | None:
    expected_key = _biomarker_key(expected_name)
    for biomarker in biomarkers:
        candidate_key = _biomarker_key(biomarker.name)
        if expected_key in candidate_key or candidate_key in expected_key:
            biomarker.name = expected_name
            return biomarker
    return None


def _biomarker_key(name: str) -> str:
    return "".join(char for char in name.lower() if char.isalnum())


def _is_retryable_llm_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in RETRYABLE_LLM_STATUS_CODES:
        return True

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "overloaded",
            "rate_limit",
            "rate limit",
            "temporarily unavailable",
            "timeout",
        )
    )


def _fallback_panel_summary(report_text: str = "") -> HormonalPanelSummary:
    missing_note = "Not found in the extracted lab report text."
    source_note = "The PDF was fetched, but AI summary generation could not complete."
    if report_text:
        source_note = "Review the extracted report values with your clinician for interpretation."

    return HormonalPanelSummary(
        biomarkers=HormoneBiomarkers(
            needs_attention=[
                HormoneBiomarker(
                    name="Estradiol (E2)",
                    value="Not found",
                    reference_range="Not found",
                    status="Needs attention",
                    interpretation=missing_note,
                ),
                HormoneBiomarker(
                    name="Cortisol AM",
                    value="Not found",
                    reference_range="Not found",
                    status="Needs attention",
                    interpretation=missing_note,
                ),
            ],
            normal_results=[
                HormoneBiomarker(
                    name="Progesterone",
                    value="Not found",
                    reference_range="Not found",
                    status="Normal",
                    interpretation=missing_note,
                ),
                HormoneBiomarker(
                    name="FSH",
                    value="Not found",
                    reference_range="Not found",
                    status="Normal",
                    interpretation=missing_note,
                ),
                HormoneBiomarker(
                    name="LH",
                    value="Not found",
                    reference_range="Not found",
                    status="Normal",
                    interpretation=missing_note,
                ),
            ],
        ),
        ai_insights=HormoneAiInsights(
            cross_data_context=[source_note],
            estradiol_elevation="Estradiol needs clinician review when elevated or out of range.",
            cortisol_near_ceiling="Morning cortisol should be interpreted against the lab reference range and symptoms.",
            hormonal_balance="Progesterone, FSH, LH, estradiol, and cortisol should be reviewed together rather than in isolation.",
        ),
        next_steps=HormoneNextSteps(
            recommendations=[
                "Discuss the hormone panel with your healthcare provider.",
                "Confirm whether results align with your cycle day and symptoms.",
                "Ask whether repeat testing or follow-up labs are appropriate.",
            ],
            medical_disclaimer=(
                "This AI summary is for informational purposes only and is not medical advice. "
                "A qualified healthcare professional should interpret abnormal or concerning lab results."
            ),
        ),
    )
