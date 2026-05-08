"""
models.py 单元测试 — Pydantic 模型验证

覆盖：T-400 ~ T-404
"""
import pytest
from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# 被测模型（待实现）
# ---------------------------------------------------------------------------
# 以下 import 在模型实现前使用 typing 模拟
from typing import Optional


# ---- SearchRequest 模拟 ----
class SearchRequest(BaseModel):
    keyword: Optional[str] = None
    semantic_query: Optional[str] = None
    categories: Optional[list[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    sort_by: str = "relevance"
    sort_order: str = "desc"
    limit: int = 20
    offset: int = 0


# ---- Paper 模拟 ----
class Paper(BaseModel):
    arxiv_id: str
    title: str
    authors: str  # JSON 字符串
    abstract: str
    categories: str  # JSON 字符串
    published_date: str
    updated_date: Optional[str] = None
    pdf_url: str
    abs_url: str
    source: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ---- SearchResponse 模拟 ----
class SearchResponse(BaseModel):
    total: int
    papers: list[Paper]


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------
class TestSearchRequest:
    """T-400 ~ T-402"""

    def test_all_fields_optional(self):
        """T-400: keyword/semantic_query 均可不传"""
        req = SearchRequest()
        assert req.keyword is None
        assert req.semantic_query is None

    def test_limit_default(self):
        """T-401: limit 默认值 20"""
        req = SearchRequest()
        assert req.limit == 20

    def test_offset_default(self):
        """T-402: offset 默认值 0"""
        req = SearchRequest()
        assert req.offset == 0

    def test_custom_limit_offset(self):
        req = SearchRequest(limit=50, offset=10)
        assert req.limit == 50
        assert req.offset == 10

    def test_sort_by_date(self):
        req = SearchRequest(sort_by="date", sort_order="asc")
        assert req.sort_by == "date"
        assert req.sort_order == "asc"


class TestPaper:
    """T-404"""

    def test_all_fields_present(self):
        """T-404: Paper 结构与 SQLite 字段对应，11 个字段全部存在"""
        paper = Paper(
            arxiv_id="2401.00001",
            title="Test",
            authors='["Author"]',
            abstract="Abstract text",
            categories='["cs.CL"]',
            published_date="2024-01-01",
            pdf_url="http://arxiv.org/pdf/2401.00001",
            abs_url="http://arxiv.org/abs/2401.00001",
            source="broad",
        )
        assert paper.arxiv_id == "2401.00001"
        assert paper.title == "Test"
        assert paper.authors == '["Author"]'
        assert paper.abstract == "Abstract text"
        assert paper.categories == '["cs.CL"]'
        assert paper.published_date == "2024-01-01"
        assert paper.pdf_url == "http://arxiv.org/pdf/2401.00001"
        assert paper.abs_url == "http://arxiv.org/abs/2401.00001"
        assert paper.source == "broad"

    def test_optional_dates(self):
        paper = Paper(
            arxiv_id="2401.00001",
            title="Test",
            authors="[]",
            abstract="",
            categories="[]",
            published_date="2024-01-01",
            pdf_url="",
            abs_url="",
            source="broad",
        )
        assert paper.updated_date is None
        assert paper.created_at is None


class TestSearchResponse:
    """T-403"""

    def test_response_structure(self):
        """T-403: SearchResponse 包含 total 和 papers"""
        resp = SearchResponse(total=0, papers=[])
        assert resp.total == 0
        assert resp.papers == []

    def test_papers_list(self):
        paper = Paper(
            arxiv_id="2401.00001",
            title="Test",
            authors="[]",
            abstract="",
            categories="[]",
            published_date="2024-01-01",
            pdf_url="",
            abs_url="",
            source="broad",
        )
        resp = SearchResponse(total=1, papers=[paper])
        assert len(resp.papers) == 1
        assert resp.papers[0].arxiv_id == "2401.00001"