"""
router.py — API 路由

POST /search       — 混合检索
GET  /stats        — 库内统计
POST /admin/trigger-fetch — 手动触发采集
"""
from fastapi import APIRouter, HTTPException, Request

import db, retriever as retriever_mod
from models import (
    SearchRequest,
    SearchResponse,
    StatsResponse,
    TriggerFetchResponse,
    Paper,
)


router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(request: Request, body: SearchRequest) -> SearchResponse:
    if not body.keyword and not body.semantic_query:
        raise HTTPException(
            status_code=422,
            detail="At least one of keyword or semantic_query is required",
        )

    conn = request.app.state.conn
    qdrant = request.app.state.qdrant
    embed = request.app.state.embed

    result = retriever_mod.search(
        conn=conn,
        qdrant=qdrant,
        embed=embed,
        keyword=body.keyword,
        semantic_query=body.semantic_query,
        categories=body.categories,
        date_from=body.date_from,
        date_to=body.date_to,
        sort_by=body.sort_by,
        sort_order=body.sort_order,
        limit=body.limit,
        offset=body.offset,
    )

    papers = [Paper(**p) for p in result["papers"]]
    return SearchResponse(total=result["total"], papers=papers)


@router.get("/stats", response_model=StatsResponse)
async def stats(request: Request) -> StatsResponse:
    stats_data = db.get_stats(request.app.state.conn)
    return StatsResponse(**stats_data)


@router.post("/admin/trigger-fetch", response_model=TriggerFetchResponse)
async def trigger_fetch(request: Request) -> TriggerFetchResponse:
    crawler = request.app.state.crawler
    result = await crawler.run()
    return TriggerFetchResponse(
        status="completed",
        message=f"Fetched {result.get('count', 0)} papers",
        papers_fetched=result.get("count", 0),
    )