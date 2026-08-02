from fastapi import APIRouter, HTTPException, Query

from ai.models.pdf_summary_models import PdfSummaryResponse
from ai.services.pdf_summary_service import summarize_pdf, get_lab_reports_for_user


router = APIRouter()


@router.get("/summarize-pdf", response_model=PdfSummaryResponse)
async def summarize_pdf_get(user_id: int = Query(..., description="User ID")):
    try:
        return summarize_pdf(user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF summary failed: {exc}")


@router.post("/summarize-pdf", response_model=PdfSummaryResponse)
async def summarize_pdf_post(user_id: int = Query(..., description="User ID")):
    try:
        return summarize_pdf(user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF summary failed: {exc}")


@router.get("/lab-reports")
async def list_lab_reports(user_id: int = Query(..., description="User ID")):
    """List all lab reports for a user."""
    try:
        return get_lab_reports_for_user(user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch lab reports: {exc}")
