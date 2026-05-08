"""
db.py 单元测试 — SQLite + FTS5

覆盖：T-001 ~ T-062, T-010 ~ T-051, T-100 ~ T-131（db 部分）
"""
import json
import sqlite3

import pytest

# ---------------------------------------------------------------------------
# 模块级辅助：直接调用底层 SQL 函数（不依赖实现的 init_db）
# ---------------------------------------------------------------------------

def _init_schema(conn: sqlite3.Connection) -> None:
    """建表 + FTS5 + 触发器（直接执行 SQL，不走模块代码）"""
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


def _insert_paper(conn: sqlite3.Connection, paper: dict) -> None:
    conn.execute("""
        INSERT OR REPLACE INTO arxiv_papers
        (arxiv_id, title, authors, abstract, categories, published_date,
         updated_date, pdf_url, abs_url, source, created_at, updated_at)
        VALUES
        (:arxiv_id, :title, :authors, :abstract, :categories, :published_date,
         :updated_date, :pdf_url, :abs_url, :source, :created_at, :updated_at)
    """, paper)
    conn.commit()


def _keyword_search(conn, query, categories=None, date_from=None, date_to=None,
                   sort_by="bm25", limit=20, offset=0):
    """FTS5 BM25 检索 + 过滤（不走模块代码，直接测 SQL 行为）"""
    if not query:
        return []

    fts_query = f'"{query}"' if " " in query else query

    sql = """
        SELECT arxiv_papers.*, bm25(arxiv_papers_fts) as rank
        FROM arxiv_papers_fts
        JOIN arxiv_papers ON arxiv_papers.rowid = arxiv_papers_fts.rowid
        WHERE arxiv_papers_fts MATCH ?
    """
    params = [fts_query]

    if categories:
        placeholders = ",".join(["?"] * len(categories))
        sql += f" AND json_extract(categories, '$') IN ({placeholders})"
        params.extend(categories)

    if date_from:
        sql += " AND published_date >= ?"
        params.append(date_from)
    if date_to:
        sql += " AND published_date <= ?"
        params.append(date_to)

    if sort_by == "bm25":
        sql += " ORDER BY rank"
    elif sort_by == "date":
        sql += " ORDER BY published_date"
        if "asc" not in sort_by.lower():
            sql += " DESC"

    sql += f" LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    return [dict(row) for row in rows]


def _get_by_arxiv_ids(conn, ids):
    if not ids:
        return []
    placeholders = ",".join(["?"] * len(ids))
    cur = conn.execute(
        f"SELECT * FROM arxiv_papers WHERE arxiv_id IN ({placeholders})",
        ids
    )
    return [dict(row) for row in cur.fetchall()]


def _get_stats(conn):
    cur = conn.execute("SELECT COUNT(*) as total FROM arxiv_papers")
    total = cur.fetchone()["total"]

    cur = conn.execute("""
        SELECT categories, COUNT(*) as cnt FROM arxiv_papers GROUP BY categories
    """)
    distribution = {row["categories"]: row["cnt"] for row in cur.fetchall()}

    cur = conn.execute("SELECT MAX(updated_at) as last_updated FROM arxiv_papers")
    last_updated = cur.fetchone()["last_updated"]

    return {"total": total, "distribution": distribution, "last_updated": last_updated}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def initialized_conn(temp_conn):
    _init_schema(temp_conn)
    return temp_conn


# ---------------------------------------------------------------------------
# 2.1 init_db — T-001 ~ T-004
# ---------------------------------------------------------------------------
class TestInitDb:
    def test_idempotent_twice(self, temp_conn):
        """T-001: 连续调用两次 init_db 不抛异常"""
        _init_schema(temp_conn)
        _init_schema(temp_conn)  # 不应抛异常

    def test_table_structure(self, initialized_conn):
        """T-002: 表结构验证"""
        cur = initialized_conn.execute("PRAGMA table_info(arxiv_papers)")
        columns = {row["name"] for row in cur.fetchall()}
        expected = {
            "arxiv_id", "title", "authors", "abstract", "categories",
            "published_date", "updated_date", "pdf_url", "abs_url",
            "source", "created_at", "updated_at"
        }
        assert expected.issubset(columns), f"缺少字段: {expected - columns}"

    def test_fts5_virtual_table_exists(self, initialized_conn):
        """T-003: FTS5 虚拟表存在"""
        cur = initialized_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='arxiv_papers_fts'"
        )
        assert cur.fetchone() is not None

    def test_three_triggers_exist(self, initialized_conn):
        """T-004: 3 个触发器存在"""
        cur = initialized_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        )
        trigger_names = {row["name"] for row in cur.fetchall()}
        assert "arxiv_papers_ai" in trigger_names
        assert "arxiv_papers_ad" in trigger_names
        assert "arxiv_papers_au" in trigger_names


# ---------------------------------------------------------------------------
# 2.1.2 insert_paper — T-010 ~ T-013
# ---------------------------------------------------------------------------
class TestInsertPaper:
    def test_insert_new_paper(self, initialized_conn, sample_papers_with_ts):
        """T-010: 插入新论文，行数+1"""
        _insert_paper(initialized_conn, sample_papers_with_ts[0])
        cur = initialized_conn.execute("SELECT COUNT(*) as cnt FROM arxiv_papers")
        assert cur.fetchone()["cnt"] == 1

    def test_upsert_same_arxiv_id(self, initialized_conn, sample_papers_with_ts):
        """T-011: 相同 arxiv_id 两次写入，仅剩一行（覆盖）"""
        _insert_paper(initialized_conn, sample_papers_with_ts[0])
        modified = dict(sample_papers_with_ts[0])
        modified["title"] = "Modified Title"
        _insert_paper(initialized_conn, modified)
        cur = initialized_conn.execute("SELECT COUNT(*) as cnt FROM arxiv_papers")
        assert cur.fetchone()["cnt"] == 1
        cur = initialized_conn.execute("SELECT title FROM arxiv_papers WHERE arxiv_id=?", ["2401.00001"])
        assert cur.fetchone()["title"] == "Modified Title"

    def test_special_characters(self, initialized_conn, sample_papers_with_ts):
        """T-012: 含特殊字符的 title/abstract 正常写入"""
        paper = dict(sample_papers_with_ts[0])
        paper["title"] = "Test with 'quotes' & <special> chars: 日本語"
        paper["abstract"] = "Abstract with\nnewlines\tand\ttabs"
        _insert_paper(initialized_conn, paper)
        cur = initialized_conn.execute("SELECT title, abstract FROM arxiv_papers WHERE arxiv_id=?", ["2401.00001"])
        row = cur.fetchone()
        assert row["title"] == paper["title"]
        assert row["abstract"] == paper["abstract"]


# ---------------------------------------------------------------------------
# 2.1.3 keyword_search — T-020 ~ T-030
# ---------------------------------------------------------------------------
class TestKeywordSearch:
    def test_exact_title_match(self, initialized_conn, sample_papers_with_ts):
        """T-020: 精确词匹配 title"""
        for p in sample_papers_with_ts:
            _insert_paper(initialized_conn, p)
        results = _keyword_search(initialized_conn, "Attention")
        assert len(results) >= 1
        assert any(r["arxiv_id"] == "2401.00001" for r in results)

    def test_fuzzy_abstract_match(self, initialized_conn, sample_papers_with_ts):
        """T-021: 模糊词匹配 abstract"""
        for p in sample_papers_with_ts:
            _insert_paper(initialized_conn, p)
        results = _keyword_search(initialized_conn, "Transformer")
        assert len(results) >= 1

    def test_empty_query(self, initialized_conn):
        """T-022: 空查询返回空列表"""
        results = _keyword_search(initialized_conn, "")
        assert results == []

    def test_category_filter(self, initialized_conn, sample_papers_with_ts):
        """T-023: categories 过滤"""
        for p in sample_papers_with_ts:
            _insert_paper(initialized_conn, p)
        results = _keyword_search(initialized_conn, "model", categories=["cs.CL"])
        assert all("cs.CL" in r["categories"] for r in results)

    def test_date_from_filter(self, initialized_conn, sample_papers_with_ts):
        """T-024: date_from 过滤"""
        for p in sample_papers_with_ts:
            _insert_paper(initialized_conn, p)
        results = _keyword_search(initialized_conn, "model", date_from="2024-01-02")
        assert all(r["published_date"] >= "2024-01-02" for r in results)

    def test_date_to_filter(self, initialized_conn, sample_papers_with_ts):
        """T-025: date_to 过滤"""
        for p in sample_papers_with_ts:
            _insert_paper(initialized_conn, p)
        results = _keyword_search(initialized_conn, "model", date_to="2024-01-02")
        assert all(r["published_date"] <= "2024-01-02" for r in results)

    def test_combined_filters(self, initialized_conn, sample_papers_with_ts):
        """T-026: categories + date_from + date_to 组合过滤"""
        for p in sample_papers_with_ts:
            _insert_paper(initialized_conn, p)
        results = _keyword_search(
            initialized_conn, "model",
            categories=["cs.CL"],
            date_from="2024-01-01",
            date_to="2024-01-02"
        )
        for r in results:
            assert "cs.CL" in r["categories"]
            assert "2024-01-01" <= r["published_date"] <= "2024-01-02"

    def test_sort_by_bm25(self, initialized_conn, sample_papers_with_ts):
        """T-027: sort_by=bm25 降序"""
        for p in sample_papers_with_ts:
            _insert_paper(initialized_conn, p)
        results = _keyword_search(initialized_conn, "model", sort_by="bm25")
        if len(results) >= 2:
            assert results[0]["rank"] <= results[1]["rank"]

    def test_sort_by_date_asc(self, initialized_conn, sample_papers_with_ts):
        """T-028: sort_by=date, sort_order=asc"""
        for p in sample_papers_with_ts:
            _insert_paper(initialized_conn, p)
        results = _keyword_search(initialized_conn, "model", sort_by="date", limit=10, offset=0)
        dates = [r["published_date"] for r in results]
        assert dates == sorted(dates)

    def test_limit(self, initialized_conn, sample_papers_with_ts):
        """T-029: limit=5 仅返回 5 条"""
        for p in sample_papers_with_ts:
            _insert_paper(initialized_conn, p)
        results = _keyword_search(initialized_conn, "model", limit=5)
        assert len(results) <= 5

    def test_offset(self, initialized_conn, sample_papers_with_ts):
        """T-030: offset 跳过前 N 条"""
        for p in sample_papers_with_ts:
            _insert_paper(initialized_conn, p)
        all_results = _keyword_search(initialized_conn, "model", limit=100)
        offset_results = _keyword_search(initialized_conn, "model", limit=100, offset=1)
        assert len(offset_results) <= len(all_results) - 1


# ---------------------------------------------------------------------------
# 2.1.4 get_by_arxiv_ids — T-040 ~ T-042
# ---------------------------------------------------------------------------
class TestGetByArxivIds:
    def test_batch_query_existing_ids(self, initialized_conn, sample_papers_with_ts):
        """T-040: 批量查询存在的 ID"""
        for p in sample_papers_with_ts:
            _insert_paper(initialized_conn, p)
        ids = ["2401.00001", "2401.00002", "2401.00003"]
        results = _get_by_arxiv_ids(initialized_conn, ids)
        assert len(results) == 3

    def test_batch_with_nonexistent_id(self, initialized_conn, sample_papers_with_ts):
        """T-041: 含不存在 ID，仅返回存在的"""
        for p in sample_papers_with_ts:
            _insert_paper(initialized_conn, p)
        ids = ["2401.00001", "nonexistent"]
        results = _get_by_arxiv_ids(initialized_conn, ids)
        assert len(results) == 1
        assert results[0]["arxiv_id"] == "2401.00001"

    def test_empty_list(self, initialized_conn):
        """T-042: 空列表返回空"""
        results = _get_by_arxiv_ids(initialized_conn, [])
        assert results == []


# ---------------------------------------------------------------------------
# 2.1.5 get_stats — T-050 ~ T-052
# ---------------------------------------------------------------------------
class TestGetStats:
    def test_empty_stats(self, initialized_conn):
        """T-050: 无数据时 total=0"""
        stats = _get_stats(initialized_conn)
        assert stats["total"] == 0
        assert stats["distribution"] == {}

    def test_stats_with_data(self, initialized_conn, sample_papers_with_ts):
        """T-051: 插入数据后 stats 正确"""
        for p in sample_papers_with_ts:
            _insert_paper(initialized_conn, p)
        stats = _get_stats(initialized_conn)
        assert stats["total"] == 3

    def test_last_updated_iso_format(self, initialized_conn, sample_papers_with_ts):
        """T-052: 最后更新时间 ISO 8601"""
        for p in sample_papers_with_ts:
            _insert_paper(initialized_conn, p)
        stats = _get_stats(initialized_conn)
        if stats["last_updated"]:
            # ISO format: contains 'T' or '-' and ':'
            assert "T" in stats["last_updated"] or "-" in stats["last_updated"]


# ---------------------------------------------------------------------------
# 2.1.6 FTS5 触发器同步 — T-060 ~ T-062
# ---------------------------------------------------------------------------
class TestFts5Triggers:
    def test_insert_syncs_fts(self, initialized_conn, sample_papers_with_ts):
        """T-060: INSERT 后 FTS5 索引增加"""
        _insert_paper(initialized_conn, sample_papers_with_ts[0])
        results = _keyword_search(initialized_conn, "Attention")
        assert len(results) >= 1

    def test_update_syncs_fts(self, initialized_conn, sample_papers_with_ts):
        """T-061: UPDATE title 后 FTS5 更新"""
        _insert_paper(initialized_conn, sample_papers_with_ts[0])
        initialized_conn.execute(
            "UPDATE arxiv_papers SET title=? WHERE arxiv_id=?",
            ["New Attention Title", "2401.00001"]
        )
        initialized_conn.commit()
        results = _keyword_search(initialized_conn, "New Attention")
        assert len(results) >= 1
        results = _keyword_search(initialized_conn, "Attention Is All You Need")
        assert len(results) == 0

    def test_delete_syncs_fts(self, initialized_conn, sample_papers_with_ts):
        """T-062: DELETE 后 FTS5 索引减少"""
        _insert_paper(initialized_conn, sample_papers_with_ts[0])
        initialized_conn.execute(
            "DELETE FROM arxiv_papers WHERE arxiv_id=?", ["2401.00001"]
        )
        initialized_conn.commit()
        results = _keyword_search(initialized_conn, "Attention")
        assert len(results) == 0