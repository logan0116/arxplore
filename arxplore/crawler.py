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
from logging_config import get_logger
from errors import CrawlerError, ErrorCode

logger = get_logger("crawler")

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
        self.batch_size = self.arxiv_config.get("batch_size", 64)
        self.conn = conn
        self.qdrant = qdrant
        self.embed = embed_client.EmbedClient.from_config(config)
        logger.info(f"Crawler 初始化: max_results={self.max_results}, batch_size={self.batch_size}")

    async def run(self) -> dict[str, Any]:
        logger.info("=" * 50)
        logger.info("开始采集 ArXiv CS 论文...")
        logger.info("=" * 50)

        try:
            html = await self._fetch_page()
            logger.info(f"获取 HTML 成功，大小: {len(html)} bytes")
        except Exception as e:
            logger.error(f"获取 HTML 失败: {e}")
            raise CrawlerError(ErrorCode.CRAWLER_FETCH_FAILED, "获取 HTML 失败", str(e))

        try:
            papers = self._parse_articles(html)
            logger.info(f"解析 HTML 成功，获取 {len(papers)} 篇论文")
        except Exception as e:
            logger.error(f"解析 HTML 失败: {e}")
            raise CrawlerError(ErrorCode.CRAWLER_PARSE_FAILED, "解析 HTML 失败", str(e))

        papers = papers[: self.max_results]
        logger.info(f"限制采集数量: {len(papers)} 篇")

        try:
            count = await self._embed_and_store(papers, batch_size=self.batch_size)
            logger.info(f"采集完成，入库 {count} 篇论文")
        except Exception as e:
            logger.error(f"存储失败: {e}")
            raise CrawlerError(ErrorCode.CRAWLER_STORE_FAILED, "存储失败", str(e))

        return {"count": count, "total_fetched": len(papers)}

    async def _fetch_page(self) -> str:
        logger.debug(f"从 {ARXIV_LIST_URL} 获取 HTML")
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(ARXIV_LIST_URL)
            response.raise_for_status()
            return response.text

    def _parse_articles(self, html: str) -> list[dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        papers = []

        for dt in soup.find_all("dt"):
            link = dt.find("a", href=re.compile(r"^/abs/"))
            if not link:
                continue
            arxiv_id = link.get_text(strip=True).replace("arXiv:", "")
            abs_url = f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

            dd = dt.find_next_sibling("dd")
            if not dd:
                continue

            # 提取标题
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

            # 提取分类
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

            # 日期
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

    async def _embed_and_store(self, papers: list[dict[str, Any]], batch_size: int = 64) -> int:
        if not papers:
            logger.warning("待存储论文列表为空")
            return 0

        total = len(papers)
        logger.info(f"开始向量化 {total} 篇论文（batch_size={batch_size}）...")

        all_vectors = []
        for i in range(0, total, batch_size):
            batch = papers[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            logger.info(f"向量化第 {batch_num}/{total_batches} 批，数量: {len(batch)}")
            try:
                texts = [f"{p['title']} {p['abstract']}" for p in batch]
                vectors = await self.embed.encode_documents(texts)
                all_vectors.extend(vectors)
            except Exception as e:
                logger.error(f"批量向量化失败 [{batch_num}/{total_batches}]: {e}")
                raise CrawlerError(ErrorCode.CRAWLER_EMBED_FAILED, f"批量向量化失败: {e}", str(e))

        logger.info(f"向量化完成，获取 {len(all_vectors)} 个向量")

        to_store = []
        for paper, vector in zip(papers, all_vectors):
            ts = now_iso()
            paper["vector"] = vector
            paper["created_at"] = ts
            paper["updated_at"] = ts
            to_store.append(paper)

        # 写入 SQLite
        logger.info(f"写入 SQLite {len(to_store)} 篇论文...")
        for paper in to_store:
            try:
                db.insert_paper(self.conn, paper)
            except Exception as e:
                logger.error(f"SQLite 插入失败 [{paper.get('arxiv_id')}]: {e}")
                raise CrawlerError(ErrorCode.CRAWLER_STORE_FAILED, f"SQLite 插入失败: {paper.get('arxiv_id')}", str(e))

        # 写入 Qdrant
        logger.info(f"写入 Qdrant {len(to_store)} 篇论文...")
        try:
            self.qdrant.upsert_papers(to_store)
        except Exception as e:
            logger.error(f"Qdrant upsert 失败: {e}")
            raise CrawlerError(ErrorCode.CRAWLER_STORE_FAILED, "Qdrant 存储失败", str(e))

        return len(to_store)
