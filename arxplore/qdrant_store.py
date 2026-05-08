"""
qdrant_store.py — Qdrant 向量数据库封装

提供语义检索功能。
"""
import hashlib
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchAny, Range

from logging_config import get_logger
from errors import QdrantError, ErrorCode

logger = get_logger("qdrant")


class QdrantStore:
    def __init__(self, config: dict[str, Any]):
        qdrant_cfg = config["qdrant"]
        self.host = qdrant_cfg["host"]
        self.port = qdrant_cfg["port"]
        self.collection_name = qdrant_cfg["collection"]
        self.vector_size = qdrant_cfg["vector_size"]
        logger.info(f"QdrantStore 初始化: {self.host}:{self.port}, collection={self.collection_name}")
        self._client = QdrantClient(host=self.host, port=self.port)

    def init_collection(self) -> None:
        logger.info(f"初始化 Qdrant collection: {self.collection_name}")
        try:
            collections = self._client.get_collections().collections
            names = [c.name for c in collections]
            if self.collection_name not in names:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Collection '{self.collection_name}' 创建成功")
            else:
                logger.info(f"Collection '{self.collection_name}' 已存在")
        except Exception as e:
            logger.error(f"初始化 collection 失败: {e}")
            raise QdrantError(ErrorCode.QDRANT_COLLECTION_FAILED, "初始化 collection 失败", str(e))

    def upsert_papers(self, papers: list[dict[str, Any]]) -> None:
        from qdrant_client.models import PointStruct, Payload

        if not papers:
            logger.debug("upsert_papers: 空列表，跳过")
            return

        logger.info(f"开始 upsert {len(papers)} 篇论文到 Qdrant")
        try:
            points = []
            for idx, paper in enumerate(papers):
                aid = paper["arxiv_id"]
                point_id = idx + 1  # 使用整数作为 point ID
                vector = paper.get("vector", [0.0] * self.vector_size)
                payload = {
                    "arxiv_id": aid,
                    "title": paper.get("title", ""),
                    "categories": paper.get("categories", ""),
                    "published_date": paper.get("published_date", ""),
                }
                points.append(PointStruct(id=point_id, vector=vector, payload=payload))

            self._client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Qdrant upsert 成功: {len(points)} 条")
        except Exception as e:
            logger.error(f"Qdrant upsert 失败: {e}")
            raise QdrantError(ErrorCode.QDRANT_UPSERT_FAILED, "向量存储失败", str(e))

    def search_semantic(
        self,
        query_vector: list[float],
        categories: Optional[list[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        logger.debug(f"语义搜索: vector_dim={len(query_vector)}, limit={limit}")

        must_clauses = []
        if categories:
            must_clauses.append(
                FieldCondition(
                    key="categories",
                    match=MatchAny(any=categories),
                )
            )
        if date_from or date_to:
            range_cond = {}
            if date_from:
                range_cond["gte"] = date_from
            if date_to:
                range_cond["lte"] = date_to
            if range_cond:
                must_clauses.append(
                    FieldCondition(key="published_date", range=Range(**range_cond))
                )

        search_filter = Filter(must=must_clauses) if must_clauses else None

        try:
            results = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=search_filter,
                limit=limit + offset,
            )
            parsed = []
            for hit in results.points[offset : offset + limit]:
                parsed.append({
                    "arxiv_id": hit.payload["arxiv_id"],
                    "title": hit.payload.get("title", ""),
                    "categories": hit.payload.get("categories", ""),
                    "published_date": hit.payload.get("published_date", ""),
                    "score": hit.score,
                })
            logger.debug(f"语义搜索返回 {len(parsed)} 条结果")
            return parsed
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            raise QdrantError(ErrorCode.QDRANT_SEARCH_FAILED, "语义搜索失败", str(e))

    def delete_by_id(self, arxiv_id: str) -> None:
        try:
            point_id = hashlib.sha256(arxiv_id.encode()).hexdigest()[:16]
            self._client.delete(collection_name=self.collection_name, points=[point_id])
            logger.debug(f"删除向量: {arxiv_id} -> {point_id}")
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            raise QdrantError(ErrorCode.QDRANT_DELETE_FAILED, "删除向量失败", str(e))
