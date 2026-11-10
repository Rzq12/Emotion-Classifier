"""Review lookup endpoint: turn cited review IDs back into readable text.

``/insight``, ``/chat`` and agent drafts cite review IDs as grounding. This
endpoint lets the UI show the actual reviews behind those IDs, so a reader can
verify the grounding instead of trusting an opaque identifier.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from src.api import schemas
from src.api.dependencies import get_review_catalog

router = APIRouter(prefix="/reviews", tags=["reviews"])

CatalogDep = Annotated[dict, Depends(get_review_catalog)]


@router.post(
    "/lookup",
    response_model=schemas.ReviewLookupResponse,
    summary="Ambil teks review berdasarkan ID",
)
def reviews_lookup(
    body: schemas.ReviewLookupRequest, catalog: CatalogDep
) -> schemas.ReviewLookupResponse:
    """Return the reviews for the given IDs (input order), listing unknown IDs."""
    reviews: list[schemas.ReviewItem] = []
    missing: list[str] = []
    for review_id in dict.fromkeys(body.ids):  # de-duplicate, keep order
        entry = catalog.get(review_id)
        if entry is None:
            missing.append(review_id)
        else:
            reviews.append(schemas.ReviewItem(review_id=review_id, **entry))
    return schemas.ReviewLookupResponse(reviews=reviews, missing=missing)
