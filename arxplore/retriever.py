"""
retriever.py — 混合检索编排

提供 search() 函数，协调 FTS5 全文检索 + Qdrant 语义检索 + RRF 融合。
"""
from collections import defaultdict
from typing import Any, Optional

import db, qdrant_store, embed_client


# ---------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------
def rrf_fusion(
    keyword_results: list[dict[str, Any]],
    semantic_results: list[dict[str, Any]],
    k: int = 60,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    RRF 融合：双路检索结果按排名融合评分。
    score(d) = Σ 1 / (k + rank_r(d))
    其中 rank_r(d) 从 1 开始。
    """
    scores: dict[str, float] = defaultdict(float)

    for rank, doc in enumerate(keyword_results, start=1):
        aid = doc.get("arxiv_id")
        if aid:
            scores[aid] += 1.0 / (k + rank)

    for rank, doc in enumerate(semantic_results, start=1):
        aid = doc.get("arxiv_id")
        if aid:
            scores[aid] += 1.0 / (k + rank)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {"arxiv_id": aid, "rrf_score": score}
        for aid, score in sorted_docs[offset : offset + limit]
    ]


def apply_sort_by_date(
    results: list[dict[str, Any]],
    all_papers: dict[str, dict[str, Any]],
    sort_order: str = "desc",
) -> list[dict[str, Any]]:
    def get_date(item: dict[str, Any]) -> str:
        aid = item.get("arxiv_id", "")
        paper = all_papers.get(aid, {})
        return paper.get("published_date", "")

    reverse = sort_order == "desc"
    return sorted(results, key=get_date, reverse=reverse)


# ---------------------------------------------------------------------------
# Main search function
# ---------------------------------------------------------------------------

def search(
    conn: Any,  # sqlite3.Connection
    qdrant: "qdrant_store.QdrantStore",
    embed: "embed_client.EmbedClient",
    keyword: Optional[str] = None,
    semantic_query: Optional[str] = None,
    categories: Optional[list[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort_by: str = "relevance",
    sort_order: str = "desc",
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """
    混合检索入口。
    - 仅有 keyword → FTS5 检索
    - 仅有 semantic_query → Qdrant 语义检索
    - 两者都有 → 双路并行 + RRF 融合
    最后回填 SQLite 完整元数据。
    """
    keyword_results: list[dict[str, Any]] = []
    semantic_results: list[dict[str, Any]] = []

    # ---- FTS5 检索 ----
    if keyword:
        keyword_results = db.keyword_search(
            conn,
            query=keyword,
            categories=categories,
            date_from=date_from,
            date_to=date_to,
            sort_by="bm25",
            limit=limit * 2,  # 多取一些避免截断
            offset=0,
        )

    # ---- 语义检索 ----
    if semantic_query:
        import asyncio
        loop = asyncio.new_event_loop()
        query_vector = loop.run_until_complete(
            embed.encode_query(semantic_query)
        )
        loop.close()

        semantic_results = qdrant.search_semantic(
            query_vector,
            categories=categories,
            date_from=date_from,
            date_to=date_to,
            limit=limit * 2,
            offset=0,
        )

    # ---- 融合 ----
    if keyword and semantic_query:
        fused = rrf_fusion(keyword_results, semantic_results, k=60, limit=limit * 2, offset=0)
    elif keyword:
        fused = [{"arxiv_id": r["arxiv_id"], "rrf_score": 0.0} for r in keyword_results]
    else:
        fused = [{"arxiv_id": r["arxiv_id"], "rrf_score": r.get("score", 0.0)} for r in semantic_results]

    # ---- 排序 ----
    if sort_by == "date":
        all_papers = {r["arxiv_id"]: r for r in db.get_by_arxiv_ids(conn, [f["arxiv_id"] for f in fused])}
        fused_sorted = apply_sort_by_date(fused, all_papers, sort_order=sort_order)
    else:
        fused_sorted = sorted(fused, key=lambda x: x["rrf_score"], reverse=True)

    # ---- 回填元数据 ----
    all_ids = [r["arxiv_id"] for r in fused_sorted]
    papers_map = {r["arxiv_id"]: r for r in db.get_by_arxiv_ids(conn, all_ids)}

    backfilled = []
    for item in fused_sorted:
        aid = item["arxiv_id"]
        paper = papers_map.get(aid, {})
        if paper:
            merged = {
                "arxiv_id": aid,
                "title": paper.get("title", ""),
                "authors": paper.get("authors", ""),
                "abstract": paper.get("abstract", ""),
                "categories": paper.get("categories", ""),
                "published_date": paper.get("published_date", ""),
                "updated_date": paper.get("updated_date"),
                "pdf_url": paper.get("pdf_url", ""),
                "abs_url": paper.get("abs_url", ""),
                "source": paper.get("source", ""),
                "created_at": paper.get("created_at"),
                "updated_at": paper.get("updated_at"),
                "rrf_score": item.get("rrf_score"),
            }
            backfilled.append(merged)

    total = len(backfilled)
    paginated = backfilled[offset : offset + limit]

    return {"total": total, "papers": paginated}