"""
retriever.py — 混合检索编排

两阶段检索流程：
  第一阶段（检索，两个独立支路）：
    - 支路1: keyword → FTS5 → 结果A (list[arxiv_id])
    - 支路2: keyword + semantic_query → embed → Qdrant → 结果B (list[arxiv_id])
    - 合并: A ∪ B
  第二阶段（排序）：
    - 送入 rerank API → 按 score 降序返回
"""
import asyncio
import concurrent.futures
from typing import Any, Optional

import db, qdrant_store, embed_client


# ---------------------------------------------------------------------------
# 第一阶段：合并两路候选结果
# ---------------------------------------------------------------------------
def merge_candidates(
    fts_results: list[dict[str, Any]],
    qdrant_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    两路检索结果合并取并集。
    fts_results: FTS5 返回，每项含 arxiv_id
    qdrant_results: Qdrant 返回，每项含 arxiv_id
    返回: list[dict]，每项含 arxiv_id（去重）
    """
    seen = set()
    merged = []
    for doc in fts_results:
        aid = doc.get("arxiv_id")
        if aid and aid not in seen:
            seen.add(aid)
            merged.append({"arxiv_id": aid})
    for doc in qdrant_results:
        aid = doc.get("arxiv_id")
        if aid and aid not in seen:
            seen.add(aid)
            merged.append({"arxiv_id": aid})
    return merged


# ---------------------------------------------------------------------------
# 第二阶段：rerank 排序
# ---------------------------------------------------------------------------
def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    conn: Any,
    embed: "embed_client.EmbedClient",
    default_prompt: str,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    调用 rerank API 对候选结果进行重排序。
    candidates: list[dict]，每项含 arxiv_id
    返回: list[dict]，含 arxiv_id 和 rerank_score，按 score 降序
    """
    if not candidates:
        return []

    # 获取 paper 元数据用于 rerank
    all_ids = [c["arxiv_id"] for c in candidates]
    papers_map = {p["arxiv_id"]: p for p in db.get_by_arxiv_ids(conn, all_ids)}

    # 构造 documents 列表（title + abstract）
    documents = []
    for c in candidates:
        paper = papers_map.get(c["arxiv_id"], {})
        doc_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        documents.append(doc_text)

    # 调用 rerank API（异步）
    def _run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(embed.rerank(query, documents))
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        rerank_results = executor.submit(_run_async).result()

    # rerank_results: [{"index": int, "document": str, "score": float}, ...]
    # 按 score 降序排列
    rerank_results_sorted = sorted(rerank_results, key=lambda x: x.get("score", 0), reverse=True)

    # 构建返回结果
    scored = []
    for entry in rerank_results_sorted:
        idx = entry.get("index", -1)
        if 0 <= idx < len(candidates):
                scored.append({
                    "arxiv_id": candidates[idx]["arxiv_id"],
                    "rerank_score": entry.get("score", 0.0),
                })

    return scored[offset : offset + limit]


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
    - 仅有 keyword → 仅 FTS5 检索
    - 仅有 semantic_query → 仅 Qdrant 语义检索（keyword 为空时跳过 FTS5）
    - 两者都有 → 双路并行 + 并集合并 + rerank 排序

    semantic_query 作为 embedding 的 prompt，用于生成 query 向量。
    """
    fts_results: list[dict[str, Any]] = []
    qdrant_results: list[dict[str, Any]] = []

    # ---- 支路1: FTS5 检索 ----
    if keyword:
        fts_results = db.keyword_search(
            conn,
            query=keyword,
            categories=categories,
            date_from=date_from,
            date_to=date_to,
            sort_by="bm25",
            limit=limit * 2,
            offset=0,
        )

    # ---- 支路2: Qdrant 语义检索 ----
    if semantic_query:
        def _run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(embed.encode_query(semantic_query))
            finally:
                loop.close()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            query_vector = executor.submit(_run_async).result()

        qdrant_results = qdrant.search_semantic(
            query_vector,
            categories=categories,
            date_from=date_from,
            date_to=date_to,
            limit=limit * 2,
            offset=0,
        )

    # ---- 合并两路候选结果 ----
    candidates = merge_candidates(fts_results, qdrant_results)

    # ---- 排序 ----
    if sort_by == "date":
        all_papers = {r["arxiv_id"]: r for r in db.get_by_arxiv_ids(conn, [c["arxiv_id"] for c in candidates])}
        sorted_results = apply_sort_by_date(candidates, all_papers, sort_order=sort_order)
        # 移除 rerank_score 字段（date 排序不需要）
        for r in sorted_results:
            r.pop("rerank_score", None)
    else:
        # rerank 排序
        query_for_rerank = keyword or semantic_query or ""
        reranked = rerank_candidates(
            query=query_for_rerank,
            candidates=candidates,
            conn=conn,
            embed=embed,
            default_prompt=embed.default_prompt,
            limit=limit * 2,
            offset=0,
        )
        sorted_results = reranked

    # ---- 回填元数据 ----
    all_ids = [r["arxiv_id"] for r in sorted_results]
    papers_map = {r["arxiv_id"]: r for r in db.get_by_arxiv_ids(conn, all_ids)}

    backfilled = []
    for item in sorted_results:
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
            }
            if "rerank_score" in item:
                merged["rerank_score"] = item["rerank_score"]
            backfilled.append(merged)

    total = len(backfilled)
    paginated = backfilled[offset : offset + limit]

    return {"total": total, "papers": paginated}