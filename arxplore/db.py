import json
import sqlite3
from pathlib import Path
from typing import Any, Optional


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS arxiv_papers (
        arxiv_id       TEXT PRIMARY KEY,
        title          TEXT NOT NULL,
        authors        TEXT NOT NULL,
        abstract       TEXT NOT NULL,
        categories    TEXT NOT NULL,
        published_date TEXT NOT NULL,
        updated_date   TEXT,
        pdf_url        TEXT NOT NULL,
        abs_url        TEXT NOT NULL,
        source         TEXT NOT NULL,
        created_at     TEXT NOT NULL,
        updated_at     TEXT NOT NULL
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS arxiv_papers_fts USING fts5(
        title, abstract, authors,
        content='arxiv_papers',
        content_rowid='rowid'
    );

    CREATE TRIGGER IF NOT EXISTS arxiv_papers_ai AFTER INSERT ON arxiv_papers BEGIN
        INSERT INTO arxiv_papers_fts(rowid, title, abstract, authors)
        VALUES (NEW.rowid, NEW.title, NEW.abstract, NEW.authors);
    END;

    CREATE TRIGGER IF NOT EXISTS arxiv_papers_ad AFTER DELETE ON arxiv_papers BEGIN
        INSERT INTO arxiv_papers_fts(arxiv_papers_fts, rowid, title, abstract, authors)
        VALUES ('delete', OLD.rowid, OLD.title, OLD.abstract, OLD.authors);
    END;

    CREATE TRIGGER IF NOT EXISTS arxiv_papers_au AFTER UPDATE ON arxiv_papers BEGIN
        INSERT INTO arxiv_papers_fts(arxiv_papers_fts, rowid, title, abstract, authors)
        VALUES ('delete', OLD.rowid, OLD.title, OLD.abstract, OLD.authors);
        INSERT INTO arxiv_papers_fts(rowid, title, abstract, authors)
        VALUES (NEW.rowid, NEW.title, NEW.abstract, NEW.authors);
    END;
    """)


def insert_paper(conn: sqlite3.Connection, paper: dict[str, Any]) -> None:
    conn.execute("""
        INSERT OR REPLACE INTO arxiv_papers
        (arxiv_id, title, authors, abstract, categories, published_date,
         updated_date, pdf_url, abs_url, source, created_at, updated_at)
        VALUES
        (:arxiv_id, :title, :authors, :abstract, :categories, :published_date,
         :updated_date, :pdf_url, :abs_url, :source, :created_at, :updated_at)
    """, paper)
    conn.commit()


def keyword_search(
    conn: sqlite3.Connection,
    query: str,
    categories: Optional[list[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort_by: str = "bm25",
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    if not query:
        return []

    fts_query = f'"{query}"' if " " in query else query

    sql = """
        SELECT arxiv_papers.*, bm25(arxiv_papers_fts) as rank
        FROM arxiv_papers_fts
        JOIN arxiv_papers ON arxiv_papers.rowid = arxiv_papers_fts.rowid
        WHERE arxiv_papers_fts MATCH ?
    """
    params: list[Any] = [fts_query]

    if categories:
        cat_conditions = " OR ".join(["categories LIKE ?"] * len(categories))
        sql += f" AND ({cat_conditions})"
        params.extend([f"%{cat}%" for cat in categories])

    if date_from:
        sql += " AND published_date >= ?"
        params.append(date_from)
    if date_to:
        sql += " AND published_date <= ?"
        params.append(date_to)

    if sort_by == "bm25":
        sql += " ORDER BY rank"
    elif sort_by == "date":
        sql += " ORDER BY published_date DESC"

    sql += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cur = conn.execute(sql, params)
    return [dict(row) for row in cur.fetchall()]


def get_by_arxiv_ids(conn: sqlite3.Connection, ids: list[str]) -> list[dict[str, Any]]:
    if not ids:
        return []
    placeholders = ",".join(["?"] * len(ids))
    cur = conn.execute(
        f"SELECT * FROM arxiv_papers WHERE arxiv_id IN ({placeholders})",
        ids,
    )
    return [dict(row) for row in cur.fetchall()]


def get_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    cur = conn.execute("SELECT COUNT(*) as total FROM arxiv_papers")
    total = cur.fetchone()["total"]

    cur = conn.execute(
        "SELECT categories, COUNT(*) as cnt FROM arxiv_papers GROUP BY categories"
    )
    distribution = {row["categories"]: row["cnt"] for row in cur.fetchall()}

    cur = conn.execute("SELECT MAX(updated_at) as last_updated FROM arxiv_papers")
    last_updated = cur.fetchone()["last_updated"]

    return {"total": total, "distribution": distribution, "last_updated": last_updated}


def ensure_db(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn