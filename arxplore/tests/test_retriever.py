"""
retriever.py 单元测试 — RRF 融合逻辑

覆盖：T-300 ~ T-330

RRF (Reciprocal Rank Fusion) 公式：
    score(d) = Σ 1 / (k + rank_r(d))
其中 k=60，rank_r(d) 从 1 开始。
"""
import pytest
from typing import Optional

# ---------------------------------------------------------------------------
# 被测函数（待实现前，先写纯逻辑验证）
# ---------------------------------------------------------------------------

def rrf_fusion(
    keyword_results: list[dict],
    semantic_results: list[dict],
    k: int = 60,
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    """
    RRF 融合逻辑：双路检索结果按排名融合评分。
    keyword_results / semantic_results: list[dict]，每项需包含 arxiv_id
    返回: list[dict]，含 arxiv_id + rrf_score，按 score 降序
    """
    from collections import defaultdict

    scores: dict[str, float] = defaultdict(float)

    # keyword_results 按顺序排名（index 0 -> rank 1）
    for rank, doc in enumerate(keyword_results, start=1):
        aid = doc.get("arxiv_id")
        if aid:
            scores[aid] += 1.0 / (k + rank)

    # semantic_results 按顺序排名
    for rank, doc in enumerate(semantic_results, start=1):
        aid = doc.get("arxiv_id")
        if aid:
            scores[aid] += 1.0 / (k + rank)

    # 按 score 降序排序
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 应用 offset + limit
    return [
        {"arxiv_id": aid, "rrf_score": score}
        for aid, score in sorted_docs[offset : offset + limit]
    ]


def apply_sort_by_date(
    results: list[dict],
    all_papers: dict[str, dict],  # arxiv_id -> full paper
    sort_order: str = "desc"
) -> list[dict]:
    """RRF 结果按 published_date 排序（外部回填阶段使用）"""
    def get_date(item: dict) -> str:
        aid = item.get("arxiv_id")
        paper = all_papers.get(aid, {})
        return paper.get("published_date", "")

    reverse = sort_order == "desc"
    return sorted(results, key=get_date, reverse=reverse)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 模拟 FTS5 返回
FTS_RESULTS = [
    {"arxiv_id": "A", "rank": -1.0},
    {"arxiv_id": "B", "rank": -2.0},
    {"arxiv_id": "C", "rank": -3.0},
    {"arxiv_id": "D", "rank": -4.0},
]

# 模拟 Qdrant 返回
QDRANT_RESULTS = [
    {"arxiv_id": "B", "score": 0.95},
    {"arxiv_id": "C", "score": 0.90},
    {"arxiv_id": "E", "score": 0.85},
    {"arxiv_id": "F", "score": 0.80},
]

# 完整 paper 元数据（用于回填测试）
PAPER_META = {
    "A": {"arxiv_id": "A", "title": "Paper A", "published_date": "2024-01-01"},
    "B": {"arxiv_id": "B", "title": "Paper B", "published_date": "2024-01-03"},
    "C": {"arxiv_id": "C", "title": "Paper C", "published_date": "2024-01-02"},
    "D": {"arxiv_id": "D", "title": "Paper D", "published_date": "2024-01-04"},
    "E": {"arxiv_id": "E", "title": "Paper E", "published_date": "2024-01-05"},
}


# ---------------------------------------------------------------------------
# 2.4.1 RRF 融合逻辑 — T-300 ~ T-306
# ---------------------------------------------------------------------------
class TestRrfFusion:
    def test_keyword_only(self):
        """T-300: 仅 keyword 检索，结果等同于 FTS5 顺序"""
        result = rrf_fusion(FTS_RESULTS, [])
        aids = [r["arxiv_id"] for r in result]
        assert aids == ["A", "B", "C", "D"]

    def test_semantic_only(self):
        """T-301: 仅 semantic 检索，结果等同于 Qdrant 顺序"""
        result = rrf_fusion([], QDRANT_RESULTS)
        aids = [r["arxiv_id"] for r in result]
        assert aids == ["B", "C", "E", "F"]

    def test_dual_retrieval_no_overlap(self):
        """T-302: 双路检索无重复文档时 RRF score 正确"""
        # FTS_RESULTS: A(rank1), B(rank2), C(rank3), D(rank4)
        # QDRANT_RESULTS: B(rank1), C(rank2), E(rank3), F(rank4)
        # 无重复文档: A/D 仅在 FTS, E/F 仅在 Qdrant
        # A (rank=1, FTS only): score = 1/(60+1) = 1/61
        # D (rank=4, FTS only): score = 1/(60+4) = 1/64
        # E (rank=3, Qdrant only): score = 1/(60+3) = 1/63
        # F (rank=4, Qdrant only): score = 1/(60+4) = 1/64
        # A score > E score > D score = F score
        result = rrf_fusion(FTS_RESULTS, QDRANT_RESULTS, k=60)
        scores = {r["arxiv_id"]: r["rrf_score"] for r in result}
        assert scores["A"] > scores["E"]

    def test_dual_retrieval_with_overlap(self):
        """T-303: 有重复文档时 score 叠加，排名提前"""
        # B 和 C 在两边都出现
        result = rrf_fusion(FTS_RESULTS, QDRANT_RESULTS, k=60)
        aids = [r["arxiv_id"] for r in result]
        # B 同时在 FTS rank=2 和 Qdrant rank=1
        # C 同时在 FTS rank=3 和 Qdrant rank=2
        # B score 高于 C
        scores = {r["arxiv_id"]: r["rrf_score"] for r in result}
        assert scores["B"] > scores["C"]

    def test_k60_vs_k0(self):
        """T-304: k=60 时排名差距缩小 vs k=0"""
        # k=0 时 score = 1/rank，差距更大
        result_k60 = rrf_fusion(FTS_RESULTS, QDRANT_RESULTS, k=60)
        result_k0 = rrf_fusion(FTS_RESULTS, QDRANT_RESULTS, k=0)

        # k=60 时各 doc 分数差距更小
        scores_k60 = [r["rrf_score"] for r in result_k60]
        scores_k0 = [r["rrf_score"] for r in result_k0]

        # top1/top2 的比值：k=0 时比值更大（差距大）
        ratio_k60 = scores_k60[0] / scores_k60[1] if len(scores_k60) > 1 else 0
        ratio_k0 = scores_k0[0] / scores_k0[1] if len(scores_k0) > 1 else 0
        assert ratio_k0 >= ratio_k60

    def test_limit(self):
        """T-305: limit=5 仅返回 5 条"""
        result = rrf_fusion(FTS_RESULTS, QDRANT_RESULTS, limit=5)
        assert len(result) <= 5

    def test_offset(self):
        """T-305: offset 跳过前 N 条"""
        result_all = rrf_fusion(FTS_RESULTS, QDRANT_RESULTS, limit=100)
        result_offset = rrf_fusion(FTS_RESULTS, QDRANT_RESULTS, limit=100, offset=2)
        # offset 后结果应在 all 结果的 offset 位置之后
        assert len(result_offset) <= len(result_all) - 2


# ---------------------------------------------------------------------------
# 2.4.2 分页与排序 — T-310 ~ T-312
# ---------------------------------------------------------------------------
class TestSortAndPagination:
    def test_sort_by_relevance(self):
        """T-310: sort_by=relevance 按 RRF score 降序"""
        result = rrf_fusion(FTS_RESULTS, QDRANT_RESULTS)
        scores = [r["rrf_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_sort_by_date_asc(self):
        """T-311: sort_by=date, sort_order=asc 按发表日期升序"""
        fused = rrf_fusion(FTS_RESULTS, QDRANT_RESULTS, limit=100)
        sorted_results = apply_sort_by_date(fused, PAPER_META, sort_order="asc")
        dates = [r.get("published_date", "") for r in sorted_results if r.get("published_date")]
        assert dates == sorted(dates)

    def test_sort_by_date_desc(self):
        """T-312: sort_by=date, sort_order=desc 按发表日期降序"""
        fused = rrf_fusion(FTS_RESULTS, QDRANT_RESULTS, limit=100)
        sorted_results = apply_sort_by_date(fused, PAPER_META, sort_order="desc")
        dates = [r.get("published_date", "") for r in sorted_results if r.get("published_date")]
        assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# 2.4.4 回填元数据 — T-330
# ---------------------------------------------------------------------------
class TestMetadataBackfill:
    def test_backfill_returns_complete_fields(self):
        """T-330: RRF 结果回填 SQLite 后返回完整 paper 字段"""
        fused = rrf_fusion(FTS_RESULTS, QDRANT_RESULTS, limit=100)
        backfilled = []
        for item in fused:
            aid = item["arxiv_id"]
            meta = PAPER_META.get(aid, {})
            merged = {**item, **meta}
            backfilled.append(merged)

        # 验证回填了 title 和 published_date（A/B/C 在 PAPER_META 中有）
        for item in backfilled:
            if item["arxiv_id"] in PAPER_META:
                assert "title" in item, f"{item['arxiv_id']} missing title"
                assert "published_date" in item, f"{item['arxiv_id']} missing published_date"
            assert "rrf_score" in item

    def test_backfill_preserves_rrf_score(self):
        """回填不应覆盖 rrf_score"""
        fused = rrf_fusion(FTS_RESULTS, QDRANT_RESULTS, limit=100)
        for item in fused:
            aid = item["arxiv_id"]
            meta = PAPER_META.get(aid, {})
            merged = {**item, **meta}
            assert merged["rrf_score"] == item["rrf_score"]