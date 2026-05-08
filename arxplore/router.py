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
from logging_config import get_logger
from errors import APIError, ErrorCode, ArxploreError

logger = get_logger("router")

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(request: Request, body: SearchRequest) -> SearchResponse:
    logger.info(f"搜索请求: keyword={body.keyword}, semantic={body.semantic_query}, limit={body.limit}")

    if not body.keyword and not body.semantic_query:
        logger.warning("搜索请求缺少 keyword 和 semantic_query")
        raise HTTPException(
            status_code=422,
            detail="At least one of keyword or semantic_query is required",
        )

    conn = request.app.state.conn
    qdrant = request.app.state.qdrant
    embed = request.app.state.embed

    try:
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
        logger.info(f"搜索返回: total={result['total']}, papers={len(papers)}")
        return SearchResponse(total=result["total"], papers=papers)

    except ArxploreError as e:
        logger.error(f"搜索失败 [{e.code.value}]: {e.message}")
        raise HTTPException(status_code=500, detail=f"[{e.code.value}] {e.message}")
    except Exception as e:
        logger.error(f"搜索失败 [未预期错误]: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/stats", response_model=StatsResponse)
async def stats(request: Request) -> StatsResponse:
    logger.debug("获取统计信息请求")

    try:
        stats_data = db.get_stats(request.app.state.conn)
        logger.debug(f"统计信息: total={stats_data['total']}")
        return StatsResponse(**stats_data)
    except ArxploreError as e:
        logger.error(f"获取统计失败 [{e.code.value}]: {e.message}")
        raise HTTPException(status_code=500, detail=f"[{e.code.value}] {e.message}")
    except Exception as e:
        logger.error(f"获取统计失败 [未预期错误]: {e}")
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")


@router.post("/admin/trigger-fetch", response_model=TriggerFetchResponse)
async def trigger_fetch(request: Request) -> TriggerFetchResponse:
    logger.info("触发采集请求")

    crawler = request.app.state.crawler

    try:
        result = await crawler.run()
        logger.info(f"采集完成: {result}")
        return TriggerFetchResponse(
            status="completed",
            message=f"Fetched {result.get('count', 0)} papers",
            papers_fetched=result.get("count", 0),
        )
    except ArxploreError as e:
        logger.error(f"采集失败 [{e.code.value}]: {e.message}")
        raise HTTPException(
            status_code=500,
            detail=f"[{e.code.value}] {e.message}: {e.details}",
        )
    except Exception as e:
        logger.error(f"采集失败 [未预期错误]: {e}")
        raise HTTPException(status_code=500, detail=f"Fetch failed: {str(e)}")
