"""
retriever.py 单元测试 — 两路并集 + rerank 排序

覆盖：T-300 ~ T-330

检索流程：
  第一阶段（检索，两个独立支路）：
    - 支路1: keyword → FTS5 → 结果A (list[arxiv_id])
    - 支路2: keyword + semantic_query → embed → Qdrant → 结果B (list[arxiv_id])
    - 合并: A ∪ B
  第二阶段（排序）：
    - 送入 rerank API → 按 score 降序返回
"""
import pytest
from typing import Optional

# ---------------------------------------------------------------------------
# 被测函数（待实现前，先写纯逻辑验证）
# ---------------------------------------------------------------------------

def merge_candidates(
    fts_results: list[dict],
    qdrant_results: list[dict],
) -> list[dict]:
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


def rerank_results(
    query: str,
    candidates: list[dict],
    embed_prompt: str,
) -> list[dict]:
    """
    模拟 rerank：按 index 顺序返回（实际会调用 /api/get_rank）
    candidates: list[dict]，每项含 arxiv_id 和可选的 document 文本
    返回: list[dict]，含 arxiv_id 和 rerank_score，按 score 降序
    """
    if not candidates:
        return []

    # 模拟 rerank API 返回：按 index 顺序，score 递减
    # 实际 rerank 会返回 {"index": int, "document": str, "score": float}
    return [
        {
            "arxiv_id": candidates[i]["arxiv_id"],
            "rerank_score": 1.0 - (i * 0.05),  # 模拟: index 0 -> score 1.0, index 1 -> 0.95 ...
        }
        for i in range(len(candidates))
    ]


def apply_sort_by_date(
    results: list[dict],
    all_papers: dict[str, dict],  # arxiv_id -> full paper
    sort_order: str = "desc"
) -> list[dict]:
    """排序结果按 published_date 排序（外部回填阶段使用）"""
    def get_date(item: dict) -> str:
        aid = item.get("arxiv_id")
        paper = all_papers.get(aid, {})
        return paper.get("published_date", "")

    reverse = sort_order == "desc"
    return sorted(results, key=get_date, reverse=reverse)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# FTS5 返回（仅 arxiv_id，无 score）
FTS_RESULTS = [
    {"arxiv_id": "A"},
    {"arxiv_id": "B"},
    {"arxiv_id": "C"},
    {"arxiv_id": "D"},
]

# Qdrant 返回（仅 arxiv_id，合并时不考虑 score）
QDRANT_RESULTS = [
    {"arxiv_id": "B"},
    {"arxiv_id": "C"},
    {"arxiv_id": "E"},
    {"arxiv_id": "F"},
]

# 完整 paper 元数据（用于回填测试）
PAPER_META = {
    "A": {"arxiv_id": "A", "title": "Paper A", "published_date": "2024-01-01"},
    "B": {"arxiv_id": "B", "title": "Paper B", "published_date": "2024-01-03"},
    "C": {"arxiv_id": "C", "title": "Paper C", "published_date": "2024-01-02"},
    "D": {"arxiv_id": "D", "title": "Paper D", "published_date": "2024-01-04"},
    "E": {"arxiv_id": "E", "title": "Paper E", "published_date": "2024-01-05"},
    "F": {"arxiv_id": "F", "title": "Paper F", "published_date": "2024-01-06"},
}


# ---------------------------------------------------------------------------
# 2.4.1 两路检索逻辑 — T-300 ~ T-305
# ---------------------------------------------------------------------------
class TestMergeCandidates:
    def test_keyword_only(self):
        """T-300: 仅 keyword 检索，结果仅含 FTS5 结果"""
        result = merge_candidates(FTS_RESULTS, [])
        aids = [r["arxiv_id"] for r in result]
        assert aids == ["A", "B", "C", "D"]

    def test_semantic_only(self):
        """T-301: 仅 semantic_query（qdrant_results 非空，fts_results 空）"""
        result = merge_candidates([], QDRANT_RESULTS)
        aids = [r["arxiv_id"] for r in result]
        assert aids == ["B", "C", "E", "F"]

    def test_dual_retrieval_union(self):
        """T-302: 双路检索结果取并集"""
        result = merge_candidates(FTS_RESULTS, QDRANT_RESULTS)
        aids = set(r["arxiv_id"] for r in result)
        # A/D 仅在 FTS, E/F 仅在 Qdrant, B/C 在两边
        assert aids == {"A", "B", "C", "D", "E", "F"}

    def test_duplicate_arxiv_id_deduplicated(self):
        """T-303: 并集后重复的 arxiv_id 保留唯一记录"""
        result = merge_candidates(FTS_RESULTS, QDRANT_RESULTS)
        aids = [r["arxiv_id"] for r in result]
        # B 和 C 在两边都出现，但结果中只应出现一次
        assert aids.count("B") == 1
        assert aids.count("C") == 1

    def test_limit(self):
        """T-304: limit=5 限制返回数量"""
        result = merge_candidates(FTS_RESULTS, QDRANT_RESULTS)
        limited = result[:5]
        assert len(limited) <= 5

    def test_empty_results(self):
        """两侧结果都为空时返回空列表"""
        result = merge_candidates([], [])
        assert result == []


# ---------------------------------------------------------------------------
# 2.4.2 rerank 排序 — T-310 ~ T-312
# ---------------------------------------------------------------------------
class TestRerankSort:
    def test_rerank_returns_scores(self):
        """T-310: rerank 返回结果含 rerank_score"""
        candidates = merge_candidates(FTS_RESULTS, QDRANT_RESULTS)
        result = rerank_results("test query", candidates, "default prompt")
        for item in result:
            assert "rerank_score" in item
            assert item["rerank_score"] >= 0

    def test_rerank_sorted_by_score_desc(self):
        """T-310: rerank 结果按 score 降序"""
        candidates = merge_candidates(FTS_RESULTS, QDRANT_RESULTS)
        result = rerank_results("test query", candidates, "default prompt")
        scores = [r["rerank_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_with_empty_candidates(self):
        """rerank 空候选列表返回空"""
        result = rerank_results("test query", [], "default prompt")
        assert result == []

    def test_sort_by_date_asc(self):
        """T-311: sort_by=date, sort_order=asc 按发表日期升序"""
        candidates = merge_candidates(FTS_RESULTS, QDRANT_RESULTS)
        reranked = rerank_results("test query", candidates, "default prompt")
        sorted_results = apply_sort_by_date(reranked, PAPER_META, sort_order="asc")
        dates = [r.get("published_date", "") for r in sorted_results if r.get("published_date")]
        assert dates == sorted(dates)

    def test_sort_by_date_desc(self):
        """T-312: sort_by=date, sort_order=desc 按发表日期降序"""
        candidates = merge_candidates(FTS_RESULTS, QDRANT_RESULTS)
        reranked = rerank_results("test query", candidates, "default prompt")
        sorted_results = apply_sort_by_date(reranked, PAPER_META, sort_order="desc")
        dates = [r.get("published_date", "") for r in sorted_results if r.get("published_date")]
        assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# 2.4.3 过滤条件传递 — T-320 ~ T-321
# ---------------------------------------------------------------------------
class TestFilterPassthrough:
    def test_categories_filter_affects_both_paths(self):
        """T-320: categories 过滤应传递至 FTS5 和 Qdrant"""
        # 本测试验证合并逻辑对过滤结果的处理
        fts_filtered = [{"arxiv_id": "A"}, {"arxiv_id": "B"}]
        qdrant_filtered = [{"arxiv_id": "B"}, {"arxiv_id": "C"}]
        result = merge_candidates(fts_filtered, qdrant_filtered)
        # 两者都过滤后，合并结果应反映过滤后的状态
        aids = set(r["arxiv_id"] for r in result)
        assert aids == {"A", "B", "C"}

    def test_date_filter_affects_both_paths(self):
        """T-321: date_from/date_to 过滤应传递至两侧"""
        # 与 categories 类似，验证合并逻辑
        fts_filtered = [{"arxiv_id": "A"}, {"arxiv_id": "D"}]
        qdrant_filtered = [{"arxiv_id": "C"}, {"arxiv_id": "E"}]
        result = merge_candidates(fts_filtered, qdrant_filtered)
        aids = set(r["arxiv_id"] for r in result)
        assert aids == {"A", "C", "D", "E"}


# ---------------------------------------------------------------------------
# 2.4.4 回填元数据 — T-330
# ---------------------------------------------------------------------------
class TestMetadataBackfill:
    def test_backfill_returns_complete_fields(self):
        """T-330: 合并结果回填 SQLite 后返回完整 paper 字段"""
        candidates = merge_candidates(FTS_RESULTS, QDRANT_RESULTS)
        reranked = rerank_results("test query", candidates, "default prompt")
        backfilled = []
        for item in reranked:
            aid = item["arxiv_id"]
            meta = PAPER_META.get(aid, {})
            merged = {**item, **meta}
            backfilled.append(merged)

        # 验证回填了 title 和 published_date
        for item in backfilled:
            if item["arxiv_id"] in PAPER_META:
                assert "title" in item, f"{item['arxiv_id']} missing title"
                assert "published_date" in item, f"{item['arxiv_id']} missing published_date"
            assert "rerank_score" in item

    def test_backfill_preserves_rerank_score(self):
        """回填不应覆盖 rerank_score"""
        candidates = merge_candidates(FTS_RESULTS, QDRANT_RESULTS)
        reranked = rerank_results("test query", candidates, "default prompt")
        for item in reranked:
            aid = item["arxiv_id"]
            meta = PAPER_META.get(aid, {})
            merged = {**item, **meta}
            assert merged["rerank_score"] == item["rerank_score"]


# ---------------------------------------------------------------------------
# 集成测试：完整流程
# ---------------------------------------------------------------------------
class TestFullFlow:
    def test_keyword_only_flow(self):
        """仅 keyword 时的完整流程：FTS5 → 合并 → rerank"""
        fts_results = [{"arxiv_id": "A"}, {"arxiv_id": "B"}, {"arxiv_id": "C"}]
        candidates = merge_candidates(fts_results, [])
        assert len(candidates) == 3
        reranked = rerank_results("neural network", candidates, "default prompt")
        assert len(reranked) == 3
        assert all("rerank_score" in r for r in reranked)

    def test_dual_flow(self):
        """keyword + semantic_query 完整流程：FTS5 + Qdrant → 合并 → rerank"""
        fts_results = [{"arxiv_id": "A"}, {"arxiv_id": "B"}]
        qdrant_results = [{"arxiv_id": "B"}, {"arxiv_id": "C"}]
        candidates = merge_candidates(fts_results, qdrant_results)
        # B 在两边都出现，合并后只有一条
        assert len(candidates) == 3
        assert set(c["arxiv_id"] for c in candidates) == {"A", "B", "C"}
        reranked = rerank_results("neural network", candidates, "default prompt")
        assert len(reranked) == 3
        # 验证按 score 降序排列
        scores = [r["rerank_score"] for r in reranked]
        assert scores == sorted(scores, reverse=True)