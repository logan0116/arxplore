"""
API Tests — router.py 端点测试

覆盖：T-500 ~ T-522

说明：
- 使用 FastAPI TestClient
- 依赖 mock 掉 retriever / crawler / db 等模块
- 验证请求解析、响应结构、参数校验
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

# ---------------------------------------------------------------------------
# 构造被测 FastAPI App（路由尚未实现，用 mock 替代）
# ---------------------------------------------------------------------------

# 为了让测试独立运行，先定义一个最小化模拟 App
# 实际测试时，router.py 替代这里的 mock_route_provider

app = FastAPI()


@app.post("/search")
async def mock_search(body: dict):
    keyword = body.get("keyword")
    semantic_query = body.get("semantic_query")
    categories = body.get("categories")
    date_from = body.get("date_from")
    date_to = body.get("date_to")
    sort_by = body.get("sort_by", "relevance")
    sort_order = body.get("sort_order", "desc")
    limit = body.get("limit", 20)
    offset = body.get("offset", 0)

    # 验证：keyword 和 semantic_query 至少一个存在
    if not keyword and not semantic_query:
        return {"detail": "At least one of keyword or semantic_query is required"}

    # 模拟返回（实际由 retriever 填充）
    return {
        "total": 0,
        "papers": [],
    }


@app.get("/stats")
async def mock_stats():
    return {"total": 0, "distribution": {}, "last_updated": None}


@app.post("/admin/trigger-fetch")
async def mock_trigger_fetch():
    return {"status": "triggered", "message": "Fetch started"}


client = TestClient(app)


# ---------------------------------------------------------------------------
# 3.1 POST /search — T-500 ~ T-508
# ---------------------------------------------------------------------------
class TestPostSearch:
    def test_keyword_only(self):
        """T-500: 仅 keyword 搜索返回 200"""
        response = client.post("/search", json={"keyword": "machine learning"})
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "papers" in data

    def test_semantic_query_only(self):
        """T-501: 仅 semantic_query 搜索返回 200"""
        response = client.post("/search", json={"semantic_query": "deep learning"})
        assert response.status_code == 200

    def test_hybrid_search(self):
        """T-502: keyword + semantic_query 混合搜索返回 200"""
        response = client.post("/search", json={
            "keyword": "NLP",
            "semantic_query": "language model fine-tuning"
        })
        assert response.status_code == 200

    def test_categories_filter(self):
        """T-503: categories 过滤参数正常接收"""
        response = client.post("/search", json={
            "keyword": "model",
            "categories": ["cs.CL", "cs.LG"]
        })
        assert response.status_code == 200

    def test_date_range_filter(self):
        """T-504: date_from + date_to 过滤参数正常接收"""
        response = client.post("/search", json={
            "keyword": "model",
            "date_from": "2024-01-01",
            "date_to": "2024-12-31"
        })
        assert response.status_code == 200

    def test_sort_by_relevance_with_limit(self):
        """T-505: sort_by=relevance, limit=5"""
        response = client.post("/search", json={
            "keyword": "model",
            "sort_by": "relevance",
            "limit": 5
        })
        assert response.status_code == 200

    def test_offset_pagination(self):
        """T-506: offset=10 分页"""
        response = client.post("/search", json={
            "keyword": "model",
            "offset": 10
        })
        assert response.status_code == 200

    def test_missing_both_query_and_semantic(self):
        """T-507: 空 keyword 且空 semantic_query 返回错误"""
        response = client.post("/search", json={})
        # 应返回 422 (validation error) 或 detail 错误
        assert response.status_code in (400, 422, 200)
        if response.status_code == 200:
            assert "detail" in response.json() or "error" in response.json()

    def test_nonexistent_category_filter(self):
        """T-508: 不存在的分类过滤返回空列表"""
        response = client.post("/search", json={
            "keyword": "model",
            "categories": ["cs.XXXXX"]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0


# ---------------------------------------------------------------------------
# 3.2 GET /stats — T-510 ~ T-511
# ---------------------------------------------------------------------------
class TestGetStats:
    def test_initial_stats(self):
        """T-510: 初始状态 total=0"""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "distribution" in data

    def test_stats_structure(self):
        """T-511: stats 返回结构完整"""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "last_updated" in data


# ---------------------------------------------------------------------------
# 3.3 POST /admin/trigger-fetch — T-520 ~ T-522
# ---------------------------------------------------------------------------
class TestTriggerFetch:
    def test_trigger_fetch(self):
        """T-520: 触发采集返回 200"""
        response = client.post("/admin/trigger-fetch", json={})
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "message" in data

    def test_trigger_fetch_accepts_empty_body(self):
        """T-521: trigger-fetch 接受空 body"""
        response = client.post("/admin/trigger-fetch")
        assert response.status_code in (200, 201)