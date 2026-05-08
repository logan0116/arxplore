"""
crawler.py — ArXiv 数据采集

- 直接从 cat:cs 拉取最新 CS 论文
- 速率控制：请求间隔 10s + 指数退避重试
- 流程：fetch → embed → insert_to_sqlite + upsert_to_qdrant
"""
import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx

import db, embed_client


ARXIV_API_BASE = "https://export.arxiv.org/api/query"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Crawler:
    def __init__(self, config: dict[str, Any], conn: Any, qdrant: Any):
        self.config = config
        self.arxiv_config = config["arxiv"]
        self.rate_limit = self.arxiv_config.get("rate_limit_seconds", 10)
        self.max_results = self.arxiv_config.get("max_results", 200)
        self.conn = conn
        self.qdrant = qdrant
        self.embed = embed_client.EmbedClient.from_config(config)

    async def run(self) -> dict[str, Any]:
        papers = await self._fetch_arxiv("cat:cs", self.max_results)
        count = await self._embed_and_store(papers)
        return {"count": count, "total_fetched": len(papers)}

    async def _fetch_arxiv(self, query: str, max_results: int) -> list[dict[str, Any]]:
        url = f"{ARXIV_API_BASE}?search_query={query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
        for attempt in range(5):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    return self._parse_atom(response.text)
            except Exception as e:
                wait = self.rate_limit * (2 ** attempt)
                if wait > 120:
                    wait = 120
                await asyncio.sleep(wait)
        return []

    def _parse_atom(self, xml: str) -> list[dict[str, Any]]:
        import re
        papers = []
        entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
        for entry in entries:
            def get(tag):
                m = re.search(f"<{tag}[^>]*>(.*?)</{tag}>", entry, re.DOTALL)
                return m.group(1).strip() if m else ""

            arxiv_id = get("id").split("/")[-1]
            title = get("title").replace("\n", " ")
            authors_xml = re.findall(r"<author>.*?<name>(.*?)</name>", entry, re.DOTALL)
            authors = ",".join(authors_xml)
            abstract = get("summary").replace("\n", " ")
            published = get("published")[:10]
            updated = get("updated")[:10] if get("updated") else None

            pdf_url = ""
            if m := re.search(r'<link title="pdf"[^>]+href="([^"]+)"', entry):
                pdf_url = m.group(1)
            elif m := re.search(r"<link>(.*?pdf.*?)</link>", entry):
                pdf_url = m.group(1)

            abs_url = get("id")
            categories = re.findall(r"<category[^>]+term=\"([^\"]+)\"", entry)
            categories_str = ",".join(categories)

            papers.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "categories": categories_str,
                "published_date": published,
                "updated_date": updated,
                "pdf_url": pdf_url,
                "abs_url": abs_url,
                "source": "cs",
            })
        return papers

    async def _embed_and_store(self, papers: list[dict[str, Any]]) -> int:
        if not papers:
            return 0

        texts = [f"{p['title']} {p['abstract']}" for p in papers]
        vectors = await self.embed.encode_documents(texts)

        to_store = []
        for paper, vector in zip(papers, vectors):
            ts = now_iso()
            paper["vector"] = vector
            paper["created_at"] = ts
            paper["updated_at"] = ts
            to_store.append(paper)

        for paper in to_store:
            db.insert_paper(self.conn, paper)

        self.qdrant.upsert_papers(to_store)

        return len(to_store)
