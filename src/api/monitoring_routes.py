"""Monitoring endpoint: live prediction-log summary + drift status.

Exposes aggregates only — the logged review texts never leave the server.
"""

from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends

from src.api import schemas
from src.api.dependencies import get_drift_reference, get_prediction_records
from src.monitoring.summary import build_summary

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

RecordsDep = Annotated[list, Depends(get_prediction_records)]
ReferenceDep = Annotated[pd.DataFrame | None, Depends(get_drift_reference)]


@router.get(
    "/summary",
    response_model=schemas.MonitoringSummaryResponse,
    summary="Ringkasan prediksi live + status drift",
)
def monitoring_summary(
    records: RecordsDep, reference: ReferenceDep
) -> schemas.MonitoringSummaryResponse:
    """Summarize logged /classify predictions and compare them with training data."""
    return schemas.MonitoringSummaryResponse(**build_summary(records, reference))
