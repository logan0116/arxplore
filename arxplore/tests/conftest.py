"""
共享测试夹具（Fixtures）
"""
import os
import sqlite3
import tempfile
from datetime import datetime
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# 测试用论文数据
# ---------------------------------------------------------------------------
SAMPLE_PAPERS = [
    {
        "arxiv_id": "2401.00001",
        "title": "Attention Is All You Need",
        "authors": '["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"]',
        "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
        "categories": '["cs.CL", "cs.LG"]',
        "published_date": "2024-01-01",
        "updated_date": "2024-01-15",
        "pdf_url": "https://arxiv.org/pdf/2401.00001",
        "abs_url": "https://arxiv.org/abs/2401.00001",
        "source": "broad",
    },
    {
        "arxiv_id": "2401.00002",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": '["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"]',
        "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations.",
        "categories": '["cs.CL"]',
        "published_date": "2024-01-02",
        "updated_date": "2024-01-16",
        "pdf_url": "https://arxiv.org/pdf/2401.00002",
        "abs_url": "https://arxiv.org/abs/2401.00002",
        "source": "keyword",
    },
    {
        "arxiv_id": "2401.00003",
        "title": "Generative Adversarial Networks",
        "authors": '["Ian Goodfellow", "Jean Pouget-Abadie"]',
        "abstract": "We propose a new framework for estimating generative models via an adversarial process.",
        "categories": '["cs.LG", "cs.NE"]',
        "published_date": "2024-01-03",
        "updated_date": "2024-01-17",
        "pdf_url": "https://arxiv.org/pdf/2401.00003",
        "abs_url": "https://arxiv.org/abs/2401.00003",
        "source": "broad",
    },
]


def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def paper_to_insert(paper: dict[str, Any], **overrides) -> dict[str, Any]:
    """生成插入用的 paper 字典，填充 created_at/updated_at"""
    base = dict(paper)
    base.setdefault("created_at", now_iso())
    base.setdefault("updated_at", now_iso())
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# SQLite Temp DB Fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_db_path():
    """创建临时 SQLite 数据库文件，测试结束后自动删除"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def temp_conn(temp_db_path):
    """创建临时 SQLite 连接，自动 close"""
    conn = sqlite3.connect(temp_db_path)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Mock Embed Client
# ---------------------------------------------------------------------------
class MockEmbedClient:
    """mock embed_client，用于不需要真实 HTTP 调用的测试"""

    def __init__(self, vector_dim: int = 1024):
        self._vector_dim = vector_dim

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        # 返回假的 1024 维零向量（确定性强，便于测试）
        return [[0.0] * self._vector_dim for _ in texts]

    def encode_query(self, query: str, prompt: str | None = None) -> list[float]:
        return [0.0] * self._vector_dim

    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        # 模拟返回：按原顺序，score 随机
        return [
            {"index": i, "document": doc, "score": 1.0 - i * 0.1}
            for i, doc in enumerate(documents)
        ]

    def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Mock Qdrant Store
# ---------------------------------------------------------------------------
class MockQdrantStore:
    """mock qdrant_store，用于不需要真实 Qdrant 的测试"""

    def __init__(self):
        self._points: dict[str, dict] = {}

    def init_collection(self) -> None:
        pass  # 幂等

    def upsert_papers(self, papers: list[dict]) -> None:
        import hashlib
        for paper in papers:
            aid = paper["arxiv_id"]
            hid = hashlib.sha256(aid.encode()).hexdigest()[:16]
            self._points[hid] = {
                "id": hid,
                "arxiv_id": aid,
                "title": paper.get("title", ""),
                "categories": paper.get("categories", ""),
                "published_date": paper.get("published_date", ""),
                "vector": paper.get("vector", [0.0] * 1024),
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
                p for p in results if any(cat in p.get("categories", "") for cat in categories)
            ]
        return results[offset : offset + limit]

    def delete_by_id(self, arxiv_id: str) -> None:
        import hashlib
        hid = hashlib.sha256(arxiv_id.encode()).hexdigest()[:16]
        self._points.pop(hid, None)


# ---------------------------------------------------------------------------
# Sample Papers Fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_papers():
    return [dict(p) for p in SAMPLE_PAPERS]


@pytest.fixture
def sample_papers_with_ts():
    """带时间戳的 sample papers"""
    return [paper_to_insert(p) for p in SAMPLE_PAPERS]