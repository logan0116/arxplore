import hashlib
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchAny, Range, DatetimeRange

from config import load_config


class QdrantStore:
    def __init__(self, config: dict[str, Any]):
        qdrant_cfg = config["qdrant"]
        self.host = qdrant_cfg["host"]
        self.port = qdrant_cfg["port"]
        self.collection_name = qdrant_cfg["collection"]
        self.vector_size = qdrant_cfg["vector_size"]
        self._client = QdrantClient(host=self.host, port=self.port)

    def init_collection(self) -> None:
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

    def upsert_papers(self, papers: list[dict[str, Any]]) -> None:
        from qdrant_client.models import PointStruct, Vector, Payload

        points = []
        for paper in papers:
            aid = paper["arxiv_id"]
            point_id = hashlib.sha256(aid.encode()).hexdigest()[:16]
            vector = paper.get("vector", [0.0] * self.vector_size)
            payload = {
                "arxiv_id": aid,
                "title": paper.get("title", ""),
                "categories": paper.get("categories", ""),
                "published_date": paper.get("published_date", ""),
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        if points:
            self._client.upsert(collection_name=self.collection_name, points=points)

    def search_semantic(
        self,
        query_vector: list[float],
        categories: Optional[list[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
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

        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit + offset,
        )

        return [
            {
                "arxiv_id": hit.payload["arxiv_id"],
                "title": hit.payload.get("title", ""),
                "categories": hit.payload.get("categories", ""),
                "published_date": hit.payload.get("published_date", ""),
                "score": hit.score,
            }
            for hit in results[offset : offset + limit]
        ]

    def delete_by_id(self, arxiv_id: str) -> None:
        point_id = hashlib.sha256(arxiv_id.encode()).hexdigest()[:16]
        self._client.delete(collection_name=self.collection_name, points=[point_id])