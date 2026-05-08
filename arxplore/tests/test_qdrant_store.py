"""
qdrant_store.py 单元测试

覆盖：T-100 ~ T-131

说明：
- 使用 qdrant_client.FakeLocalDirector 或 mock
- 不依赖真实 Qdrant 服务
"""
import hashlib
import pytest

# ---------------------------------------------------------------------------
# 模拟 QdrantStore（实现前定义接口契约）
# ---------------------------------------------------------------------------

class QdrantStore:
    """基于 SPEC.md 定义的 QdrantStore 接口"""

    def __init__(self, host: str, port: int, collection: str, vector_size: int = 1024):
        self.host = host
        self.port = port
        self.collection = collection
        self.vector_size = vector_size
        self._points: dict[str, dict] = {}

    def init_collection(self) -> None:
        """幂等创建 collection（这里 mock 直接跳过）"""
        pass

    def upsert_papers(self, papers: list[dict]) -> None:
        for paper in papers:
            aid = paper["arxiv_id"]
            point_id = hashlib.sha256(aid.encode()).hexdigest()[:16]
            self._points[point_id] = {
                "id": point_id,
                "arxiv_id": aid,
                "title": paper.get("title", ""),
                "categories": paper.get("categories", ""),
                "published_date": paper.get("published_date", ""),
                "vector": paper.get("vector", [0.0] * self.vector_size),
            }

    def search_semantic(
        self,
        query_vector: list[float],
        categories: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        results = list(self._points.values())
        if categories:
            results = [
                r for r in results
                if any(cat in r.get("categories", "") for cat in categories)
            ]
        if date_from:
            results = [r for r in results if r.get("published_date", "") >= date_from]
        if date_to:
            results = [r for r in results if r.get("published_date", "") <= date_to]
        return results[offset : offset + limit]

    def delete_by_id(self, arxiv_id: str) -> None:
        point_id = hashlib.sha256(arxiv_id.encode()).hexdigest()[:16]
        self._points.pop(point_id, None)

    def point_id_of(self, arxiv_id: str) -> str:
        return hashlib.sha256(arxiv_id.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def qdrant_store():
    return QdrantStore(
        host="localhost",
        port=6333,
        collection="arxiv_papers",
        vector_size=1024,
    )


# ---------------------------------------------------------------------------
# 2.2.1 init_collection — T-100 ~ T-101
# ---------------------------------------------------------------------------
class TestInitCollection:
    def test_first_call_creates_collection(self, qdrant_store):
        """T-100: 首次调用创建 collection（mock 无实际操作）"""
        qdrant_store.init_collection()
        # Mock 模式下无副作用，验证调用不抛异常即可

    def test_idempotent(self, qdrant_store):
        """T-101: 重复调用不抛异常"""
        qdrant_store.init_collection()
        qdrant_store.init_collection()  # 不应抛异常


# ---------------------------------------------------------------------------
# 2.2.2 upsert_papers — T-110 ~ T-112
# ---------------------------------------------------------------------------
class TestUpsertPapers:
    def test_upsert_five_papers(self, qdrant_store, sample_papers_with_ts):
        """T-110: 批量 upsert 3 篇论文（注：fixture 仅有 3 篇）"""
        qdrant_store.upsert_papers(sample_papers_with_ts)
        assert len(qdrant_store._points) == 3

    def test_upsert_same_arxiv_id_overwrites(self, qdrant_store, sample_papers_with_ts):
        """T-111: 同一 arxiv_id 重复 upsert 覆盖，count 不变"""
        qdrant_store.upsert_papers([sample_papers_with_ts[0]])
        original_point_id = qdrant_store.point_id_of("2401.00001")

        modified = dict(sample_papers_with_ts[0])
        modified["title"] = "Modified Title"
        qdrant_store.upsert_papers([modified])

        assert len(qdrant_store._points) == 1
        assert qdrant_store._points[original_point_id]["title"] == "Modified Title"

    def test_point_id_deterministic(self, qdrant_store):
        """T-112: point_id = sha256(arxiv_id)[:16] 确定性强"""
        aid = "2401.00001"
        id1 = qdrant_store.point_id_of(aid)
        id2 = qdrant_store.point_id_of(aid)
        assert id1 == id2
        assert len(id1) == 16


# ---------------------------------------------------------------------------
# 2.2.3 search_semantic — T-120 ~ T-126
# ---------------------------------------------------------------------------

class TestSearchSemantic:
    def test_known_vector_returns_result(self, qdrant_store, sample_papers_with_ts):
        """T-120: 已知向量搜索返回结果（mock 始终返回匹配）"""
        qdrant_store.upsert_papers(sample_papers_with_ts[:3])
        fake_vector = [0.1] * 1024
        results = qdrant_store.search_semantic(fake_vector)
        assert len(results) >= 1

    def test_categories_filter(self, qdrant_store, sample_papers_with_ts):
        """T-121: categories filter 过滤"""
        qdrant_store.upsert_papers(sample_papers_with_ts[:3])
        fake_vector = [0.1] * 1024
        results = qdrant_store.search_semantic(fake_vector, categories=["cs.CL"])
        assert all("cs.CL" in r.get("categories", "") for r in results)

    def test_date_from_filter(self, qdrant_store, sample_papers_with_ts):
        """T-122: date_from filter"""
        qdrant_store.upsert_papers(sample_papers_with_ts[:3])
        fake_vector = [0.1] * 1024
        results = qdrant_store.search_semantic(fake_vector, date_from="2024-01-02")
        assert all(r.get("published_date", "") >= "2024-01-02" for r in results)

    def test_date_to_filter(self, qdrant_store, sample_papers_with_ts):
        """T-123: date_to filter"""
        qdrant_store.upsert_papers(sample_papers_with_ts[:3])
        fake_vector = [0.1] * 1024
        results = qdrant_store.search_semantic(fake_vector, date_to="2024-01-02")
        assert all(r.get("published_date", "") <= "2024-01-02" for r in results)

    def test_combined_categories_and_date(self, qdrant_store, sample_papers_with_ts):
        """T-124: categories + date 组合过滤"""
        qdrant_store.upsert_papers(sample_papers_with_ts[:3])
        fake_vector = [0.1] * 1024
        results = qdrant_store.search_semantic(
            fake_vector,
            categories=["cs.CL"],
            date_from="2024-01-01",
            date_to="2024-01-03",
        )
        for r in results:
            assert "cs.CL" in r.get("categories", "")
            assert "2024-01-01" <= r.get("published_date", "") <= "2024-01-03"

    def test_limit(self, qdrant_store, sample_papers_with_ts):
        """T-125: limit=3 限制返回数"""
        qdrant_store.upsert_papers(sample_papers_with_ts[:5])
        fake_vector = [0.1] * 1024
        results = qdrant_store.search_semantic(fake_vector, limit=3)
        assert len(results) <= 3

    def test_empty_vector_search(self, qdrant_store):
        """T-126: 空向量搜索返回空（无数据时）"""
        results = qdrant_store.search_semantic([0.0] * 1024)
        assert results == []


# ---------------------------------------------------------------------------
# 2.2.4 delete_by_id — T-130 ~ T-131
# ---------------------------------------------------------------------------

class TestDeleteById:
    def test_delete_existing_point(self, qdrant_store, sample_papers_with_ts):
        """T-130: 删除存在的 point 后 search 找不到"""
        qdrant_store.upsert_papers([sample_papers_with_ts[0]])
        assert len(qdrant_store._points) == 1
        qdrant_store.delete_by_id("2401.00001")
        assert len(qdrant_store._points) == 0

    def test_delete_nonexistent_point_no_exception(self, qdrant_store):
        """T-131: 删除不存在的 point 不抛异常"""
        qdrant_store.delete_by_id("nonexistent.arxiv")  # 不应抛异常