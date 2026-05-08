"""
crawler.py — ArXiv 数据采集

- 直接从 https://arxiv.org/list/cs/new 拉取 HTML，解析获取完整元数据
- 流程：fetch_html() → parse_articles() → embed() → insert_to_sqlite + upsert_to_qdrant()
"""
import asyncio
import re
from datetime import datetime, timezone
from typing import Any

import httpx
from bs4 import BeautifulSoup

import db, embed_client


ARXIV_LIST_URL = "https://arxiv.org/list/cs/new"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def arxiv_id_to_date(arxiv_id: str) -> str:
    """从 arxiv_id 提取日期，格式如 2605.05208 -> 2026-05-01"""
    try:
        parts = arxiv_id.split(".")
        if len(parts) >= 2:
            yymm = parts[0]
            yy = int(yymm[:2])
            mm = int(yymm[2:4])
            year = 2000 + yy if yy <= 25 else 2026 + (yy - 26)
            return f"{year}-{mm:02d}-01"
    except (ValueError, IndexError):
        pass
    return ""


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
        html = await self._fetch_page()
        papers = self._parse_articles(html)
        papers = papers[: self.max_results]
        count = await self._embed_and_store(papers)
        return {"count": count, "total_fetched": len(papers)}

    async def _fetch_page(self) -> str:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(ARXIV_LIST_URL)
            response.raise_for_status()
            return response.text

    def _parse_articles(self, html: str) -> list[dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        papers = []

        for dt in soup.find_all("dt"):
            # 提取 arxiv_id
            link = dt.find("a", href=re.compile(r"^/abs/"))
            if not link:
                continue
            arxiv_id = link.get_text(strip=True).replace("arXiv:", "")
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

            dd = dt.find_next_sibling("dd")
            if not dd:
                continue

            # 提取标题（移除 "Title:" 前缀）
            title_elem = dd.find("div", class_="list-title")
            title = ""
            if title_elem:
                title = title_elem.get_text(strip=True)
                if title.startswith("Title:"):
                    title = title[6:].strip()
            title = re.sub(r"\s+", " ", title)

            # 提取作者
            authors_elem = dd.find("div", class_="list-authors")
            authors = ""
            if authors_elem:
                author_links = authors_elem.find_all("a", href=re.compile(r"searchtype=author"))
                authors = ", ".join(a.get_text(strip=True) for a in author_links)

            # 提取分类（移除 "Subjects:" 前缀）
            subjects_elem = dd.find("div", class_="list-subjects")
            categories = ""
            if subjects_elem:
                spans = subjects_elem.find_all("span")
                cat_texts = [s.get_text(strip=True) for s in spans]
                if cat_texts and cat_texts[0] == "Subjects:":
                    cat_texts = cat_texts[1:]
                categories = ", ".join(cat_texts)

            # 提取摘要
            abstract_elem = dd.find("p", class_="mathjax")
            abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""
            abstract = re.sub(r"\s+", " ", abstract)

            # 日期：从 arxiv_id 推断
            published_date = arxiv_id_to_date(arxiv_id)

            papers.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "categories": categories,
                "published_date": published_date,
                "updated_date": None,
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

        try:
            self.qdrant.upsert_papers(to_store)
        except Exception as e:
            print(f"Qdrant upsert failed: {e}")

        return len(to_store)
