# ArXplore 技术方案设计文档

> 基于 `todo.md` 需求规格说明，并参考 `doc/outer/README_embedding_server.md` 中的 embedding_server 接口文档细化而来，作为开发实施的基准文档。

---

## 1. 项目概述

**项目名称**：ArXplore — ArXiv 文献检索服务

**核心功能**：每日自动拉取 arXiv 新论文，支持关键词检索、语义相似度检索及混合检索（RRF 融合），并通过 FastAPI 提供 RESTful API 供前端调用。

**部署目标**：带 NVIDIA GPU 的 Linux 服务器。实施由 Claude Code 按此文档执行。

---

## 2. 技术架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI App                           │
├─────────────────────────────────────────────────────────────┤
│  router.py (API endpoints)                                   │
├─────────────────────────────────────────────────────────────┤
│  retriever.py (混合检索编排)                                  │
│    ├── keyword_search (FTS5)                                 │
│    ├── semantic_search (Qdrant)                             │
│    ├── RRF 融合 (k=60)                                       │
│    └── 可选：rerank 精排                                      │
├──────────┬──────────────────────┬────────────────────────────┤
│  db.py   │   qdrant_store.py    │  embed_client.py            │
│ SQLite    │   Qdrant (向量库)   │ embedding_server            │
│ + FTS5   │   1024-dim Cosine    │ (外部 HTTP 服务 192.168.1.116)│
└──────────┴──────────────────────┴────────────────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │     crawler.py      │
                │  (定时/手动触发)      │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │      arXiv API        │
                │   (data.arxiv.org)   │
                └──────────────────────┘
```

---

## 3. 配置管理

### 3.1 config.yaml 结构

```yaml
database:
  path: "./data/arxiv.db"       # SQLite 数据库路径

qdrant:
  host: "localhost"
  port: 6333
  collection: "arxiv_papers"
  vector_size: 1024             # Qwen3-Embedding-0.6B 输出维度

embedding:
  base_url: "http://192.168.1.116:9000"
  timeout: 30
  default_prompt: "Given a web search query, retrieve relevant passages that answer the query"

scheduler:
  enabled: true
  hour: 3                       # 每日凌晨 3:00 执行

arxiv:
  max_results: 200             # 单次采集上限
  rate_limit_seconds: 10

app:
  host: "0.0.0.0"
  port: 8000
```

---

## 4. 数据模型

### 4.1 SQLite 表结构

```sql
CREATE TABLE arxiv_papers (
    arxiv_id       TEXT PRIMARY KEY,      -- 格式: "2401.12345"
    title          TEXT NOT NULL,
    authors        TEXT NOT NULL,         -- JSON 数组序列化
    abstract       TEXT NOT NULL,
    categories    TEXT NOT NULL,          -- JSON 数组序列化
    published_date TEXT NOT NULL,
    updated_date   TEXT,
    pdf_url        TEXT NOT NULL,
    abs_url        TEXT NOT NULL,
    source         TEXT NOT NULL,         -- "cs"
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE VIRTUAL TABLE arxiv_papers_fts USING fts5(
    title, abstract, authors,
    content='arxiv_papers',
    content_rowid='rowid'
);

-- FTS5 与主表自动同步触发器
CREATE TRIGGER arxiv_papers_ai AFTER INSERT ON arxiv_papers BEGIN
    INSERT INTO arxiv_papers_fts(rowid, title, abstract, authors)
    VALUES (NEW.rowid, NEW.title, NEW.abstract, NEW.authors);
END;

CREATE TRIGGER arxiv_papers_ad AFTER DELETE ON arxiv_papers BEGIN
    INSERT INTO arxiv_papers_fts(arxiv_papers_fts, rowid, title, abstract, authors)
    VALUES ('delete', OLD.rowid, OLD.title, OLD.abstract, OLD.authors);
END;

CREATE TRIGGER arxiv_papers_au AFTER UPDATE ON arxiv_papers BEGIN
    INSERT INTO arxiv_papers_fts(arxiv_papers_fts, rowid, title, abstract, authors)
    VALUES ('delete', OLD.rowid, OLD.title, OLD.abstract, OLD.authors);
    INSERT INTO arxiv_papers_fts(rowid, title, abstract, authors)
    VALUES (NEW.rowid, NEW.title, NEW.abstract, NEW.authors);
END;
```

### 4.2 Qdrant Collection

- **Collection name**: `arxiv_papers`（通过 config 配置）
- **Vector dimension**: 1024（Cosine 距离，匹配 Qwen3-Embedding-0.6B 输出）
- **Index**: HNSW
- **Payload fields**:
  - `arxiv_id`: keyword 过滤
  - `title`: 展示
  - `categories`: keyword 过滤（数组）
  - `published_date`: datetime 范围过滤

---

## 5. 模块详细设计

### 5.1 embed_client.py

**职责**：封装对 embedding_server（`192.168.1.116:9000`）的 HTTP 调用。

**接口**：

```python
class EmbedClient:
    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """调用 POST /api/get_embedding，embed_type="document"，不加 prompt"""

    def encode_query(self, query: str, prompt: str | None = None) -> list[float]:
        """调用 POST /api/get_embedding，embed_type="query"，使用 default_prompt"""

    def rerank(self, query: str, documents: list[str]) -> list[dict]:
        """调用 POST /api/get_rank，返回 [{"index": int, "document": str, "score": float}]"""
        # 按 score 降序排列

    def health_check(self) -> bool:
        """调用 GET /health"""
```

**请求/响应映射**：

- `encode_documents`:
  - 请求体：`{"embed_type": "document", "items": [...]}`
  - 响应：`{"code": 200, "data": [[...], [...]]}`

- `encode_query`:
  - 请求体：`{"embed_type": "query", "prompt": <default_prompt>, "items": [query]}`
  - 响应：`{"code": 200, "data": [[...]]}`

- `rerank`:
  - 请求体：`{"query": str, "documents": [...], "query_prompt": <default_prompt>}`
  - 响应：`{"code": 200, "data": [{"index": 0, "document": "...", "score": 0.95}, ...]}`

**实现要点**：
- 使用 `httpx.AsyncClient`，连接池复用
- 单例模式（`__new__` 或模块级 singleton）
- 超时控制（config 中的 timeout）
- 响应检查 `code == 200`，非 200 抛异常

### 5.2 db.py

**职责**：SQLite + FTS5 封装，提供全文检索和元数据存储。

**接口**：

```python
def init_db(conn: sqlite3.Connection) -> None:
    """建表 + 建触发器，幂等"""

def insert_paper(conn: sqlite3.Connection, paper: dict) -> None:
    """UPSERT：INSERT OR REPLACE，避免重复"""

def keyword_search(
    conn: sqlite3.Connection,
    query: str,
    categories: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sort_by: str = "bm25",
    limit: int = 20,
    offset: int = 0
) -> list[dict]:
    """FTS5 BM25 检索 + 过滤"""

def get_by_arxiv_ids(conn: sqlite3.Connection, ids: list[str]) -> list[dict]:
    """批量回填元数据"""

def get_stats(conn: sqlite3.Connection) -> dict:
    """返回总数、分类分布、最后更新时间"""
```

### 5.3 qdrant_store.py

**职责**：Qdrant 封装，提供语义检索。

**接口**：

```python
class QdrantStore:
    def init_collection(self) -> None:
        """创建 collection（已存在则跳过）"""

    def upsert_papers(self, papers: list[dict]) -> None:
        """批量 upsert，point_id = sha256(arxiv_id)[:16]"""

    def search_semantic(
        self,
        query_vector: list[float],
        categories: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 20,
        offset: int = 0
    ) -> list[dict]:
        """带 payload filter 的语义检索"""

    def delete_by_id(self, arxiv_id: str) -> None:
        """删除指定论文"""
```

### 5.4 retriever.py

**职责**：混合检索编排，实现 RRF（Reciprocal Rank Fusion）融合，可选 rerank 精排。

**检索模式**：
| 场景 | 行为 |
|------|------|
| 仅有 keyword | 仅 FTS5 检索 |
| 仅有 semantic_query | 仅 Qdrant 语义检索 |
| 两者都有 | 双路并行检索 + RRF 融合（k=60） |

**RRF 公式**：
```
score(d) = Σ 1 / (k + rank_r(d))
```
其中 k=60，`rank_r(d)` 为文档 d 在检索结果列表 r 中的排名（从 1 开始）。

**可选精排**：RRF 融合后取 top N 调用 rerank 服务进行 CrossEncoder 精排。

**接口**：

```python
def search(
    keyword: str | None,
    semantic_query: str | None,
    categories: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sort_by: str = "relevance",   # "relevance" | "date"
    sort_order: str = "desc",
    limit: int = 20,
    offset: int = 0
) -> dict:
    """返回 {total, papers: [...]}"""
```

### 5.5 crawler.py

**职责**：从 arXiv 获取数据，经过 embedding 后入库。

**采集策略**：
- 直接从 `https://arxiv.org/list/cs/new` 解析 HTML 页面，一次请求获取全部元数据（标题、作者、摘要、分类）

**流程**：
```
fetch_html() → parse_articles() → embed() → insert_to_sqlite() + upsert_to_qdrant()
```

### 5.6 router.py

**API 端点**：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/search` | 混合检索 |
| GET | `/stats` | 库内统计 |
| POST | `/admin/trigger-fetch` | 手动触发采集 |

**POST /search 请求体**：
```json
{
  "keyword": "machine learning",
  "semantic_query": "deep learning for NLP",
  "categories": ["cs.LG", "cs.CL"],
  "date_from": "2024-01-01",
  "date_to": "2024-12-31",
  "sort_by": "relevance",
  "sort_order": "desc",
  "limit": 20,
  "offset": 0
}
```

### 5.7 main.py

**Lifespan 管理**：

```python
async def lifespan(app: FastAPI):
    # 启动
    config = load_config("config.yaml")
    conn = init_db(config)
    qdrant = QdrantStore(config)
    embed = EmbedClient(config)
    scheduler = AsyncIOScheduler()

    # 每日调度
    scheduler.add_job(daily_fetch, "cron", hour=3, args=[...])

    yield

    # 关闭
    scheduler.shutdown()
    conn.close()
```

---

## 6. 测试策略

| 测试类型 | 覆盖范围 | 说明 |
|----------|----------|------|
| 单元测试 | RRF 融合逻辑 | 纯函数，可 mock |
| 单元测试 | db.py FTS5 查询 | 需临时 SQLite 文件 |
| 集成测试 | 采集 → 入库 → 检索 | 端到端，需 arXiv mock 或用真实数据 |
| API 测试 | embedding_server 接口 | 验证 encode_documents/encode_query/rerank 行为（参考 embedding_server 测试套件） |

---

## 7. 文件结构

```
arxplore/
├── main.py              # FastAPI 入口 + lifespan
├── db.py                # SQLite + FTS5
├── qdrant_store.py      # Qdrant collection 管理
├── embed_client.py     # embedding_server HTTP 客户端
├── retriever.py         # 混合检索编排
├── crawler.py           # arXiv 数据采集
├── router.py            # API 路由
├── models.py            # Pydantic 模型
├── config.yaml          # 配置文件
├── tests/               # 测试
│   ├── test_retriever.py
│   ├── test_db.py
│   └── test_integration.py
└── requirements.txt
```

---

## 8. 实施顺序

| # | 模块 | 前置依赖 | 产出文件 |
|---|------|----------|----------|
| 1 | 骨架 + 配置 | 无 | config.yaml, models.py |
| 2 | db.py | config | db.py |
| 3 | qdrant_store.py | config | qdrant_store.py |
| 4 | embed_client.py | 无 | embed_client.py |
| 5 | retriever.py | db, qdrant, embed | retriever.py |
| 6 | router.py + models | retriever | router.py |
| 7 | crawler.py | db, qdrant, embed, config | crawler.py |
| 8 | main.py | 上述所有 | main.py |
| 9 | tests/ | 全部 | tests/ |
| 10 | requirements.txt | - | requirements.txt |

---

## 9. 验收标准

- [ ] `POST /search` 对 keyword、semantic_query、两者的组合均返回正确结果
- [ ] `GET /stats` 返回准确的总论文数、分类分布
- [ ] `POST /admin/trigger-fetch` 成功触发一次完整采集流程
- [ ] FTS5 触发器正常工作（插入/更新/删除后索引自动同步）
- [ ] Qdrant HNSW 索引正确创建，语义检索延迟 < 100ms（1000 条数据）
- [ ] RRF 融合在双路检索场景下正确工作
- [ ] embed_client 正确调用 embedding_server（query/doc 区分、prompt 使用、rerank 解析）
- [ ] 单元测试覆盖 RRF 融合逻辑和 FTS5 查询

---

## 10. 参考资料

- embedding_server 接口文档：`doc/outer/README_embedding_server.md`
  - 模型：Qwen3-Embedding-0.6B（1024 维）、Qwen3-Reranker-0.6B
  - 端点：`POST /api/get_embedding`、`POST /api/get_rank`、`GET /health`