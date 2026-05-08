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

import db, embed_client


ARXIV_LIST_URL = "https://arxiv.org/list/cs/new"


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
        papers = []
        # 匹配 <dt>...<dd> 结构
        # 先找到所有 (dt, dd) 对
        article_blocks = re.findall(r'<dt>(.*?)</dt>\s*<dd>(.*?)</dd>', html, re.DOTALL)

        for dt_content, dd_content in article_blocks:
            # 从 <dt> 提取 arxiv_id 和 URL
            id_match = re.search(r'href\s*=\s*["\']/abs/([^"\']+)["\']', dt_content)
            if not id_match:
                continue
            arxiv_id = id_match.group(1)
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

            # 从 <dd> 提取标题
            title_match = re.search(r'<div class=\'list-title[^\']*\'>.*?<span class=\'descriptor\'>:</span>\s*(.*?)</div>', dd_content, re.DOTALL)
            title = title_match.group(1).strip() if title_match else ""
            title = re.sub(r'\s+', ' ', title)

            # 提取作者
            authors_match = re.search(r'<div class=\'list-authors\'>(.*?)</div>', dd_content, re.DOTALL)
            authors = ""
            if authors_match:
                author_names = re.findall(r'query=([^"&]+)', authors_match.group(1))
                authors = ", ".join(name.replace("+", " ") for name in author_names)

            # 提取分类
            subjects_match = re.search(r'<div class=\'list-subjects\'>(.*?)</div>', dd_content, re.DOTALL)
            categories = ""
            if subjects_match:
                cats = re.findall(r'>\s*([^<]+?)\s*\(([^)]+)\)', subjects_match.group(1))
                categories = ", ".join(f"{name.strip()} ({abbr})" for name, abbr in cats)
                primary_cat = re.search(r'class="primary-subject">([^<]+)', subjects_match.group(1))
                primary = primary_cat.group(1).strip() if primary_cat else ""

            # 提取摘要
            abstract_match = re.search(r'<p class=\'mathjax\'>(.*?)</p>', dd_content, re.DOTALL)
            abstract = abstract_match.group(1).strip() if abstract_match else ""
            abstract = re.sub(r'\s+', ' ', abstract)
            abstract = re.sub(r'&#39;', "'", abstract)
            abstract = re.sub(r'&amp;', '&', abstract)
            abstract = re.sub(r'&lt;', '<', abstract)
            abstract = re.sub(r'&gt;', '>', abstract)

            # 提取日期（来自 dt 中的日期信息）
            date_match = re.search(r'Submitted on ([^;]+)', dd_content)
            published_date = ""
            if date_match:
                date_str = date_match.group(1).strip()
                # 格式: "6 May 2025"
                try:
                    dt = datetime.strptime(date_str, "%d %b %Y")
                    published_date = dt.strftime("%Y-%m-%d")
                except ValueError:
                    published_date = ""

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

        self.qdrant.upsert_papers(to_store)

        return len(to_store)
