"""``POST /search`` — the retrieval pipeline without generation (I-7).

Parses the query into hard filters, runs hybrid retrieval + rerank, and returns
the hits alongside the parsed query. Echoing :class:`ParsedQuery` makes the
filtering observable: a caller can confirm "red Bordeaux under £50" became
``color=red`` / ``region=Bordeaux`` / ``price<=50`` rather than free text.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from hedonism_assistant.models.search import SearchHit, SearchRequest, SearchResponse
from hedonism_assistant.retrieval.query_parser import QueryParser, get_query_parser
from hedonism_assistant.retrieval.retriever import Retriever, get_retriever

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    parser: QueryParser = Depends(get_query_parser),
    retriever: Retriever = Depends(get_retriever),
) -> SearchResponse:
    parsed = await parser.parse(request.query)
    retrieved = await retriever.retrieve(parsed)
    if request.limit is not None:
        retrieved = retrieved[: request.limit]
    hits = [SearchHit(wine=r.wine, score=r.score, rerank_score=r.rerank_score) for r in retrieved]
    return SearchResponse(parsed=parsed, hits=hits)
