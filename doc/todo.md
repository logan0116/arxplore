# ArXplore — ArXiv 文献检索服务

> 每日自动拉取 arXiv 新论文，支持关键词检索、语义相似度检索及混合检索。
> 部署目标：带 NVIDIA GPU 的 Linux 服务器。实施由 Claude Code 按此需求文档执行。

## 现有环境

- FastAPI conda 环境已就绪，可直接使用
- Qdrant Docker 镜像 `qdrant/qdrant:latest` 已拉取
- Embedding 由外部服务 **embedding_server** (`192.168.1.116:9000`) 提供，支持 embedding 和 reranker 两种能力

## 技术选型（已确定）

| 组件 | 选型 | 说明 |
|------|------|------|
| 元数据存储 | SQLite + FTS5 | 全文检索 |
| 向量存储 | Qdrant | 语义检索，512 维 Cosine |
| Embedding | embedding_server (HTTP) | `/api/get_embedding` + `/api/get_rank` |
| Web 框架 | FastAPI + uvicorn | |
| 定时任务 | APScheduler 或 cron | 每日自动采集 |

## 项目结构

```
arxplore/
├── main.py              # FastAPI 入口 + 生命周期
├── db.py                # SQLite + FTS5
├── qdrant_store.py      # Qdrant collection 管理 + 语义检索
├── embed_client.py      # embedding_server HTTP 客户端
├── retriever.py         # 混合检索编排（RRF 融合）
├── crawler.py           # arXiv 数据采集
├── router.py            # API 路由
├── models.py            # Pydantic 请求/响应模型
├── config.yaml          # 配置文件
├── tests/               # 测试
└── requirements.txt
```

## 模块需求

### config.yaml
- arXiv 数据源：监控的分类列表、关键词列表、每类/每词拉取上限、速率限制
- SQLite 数据库文件路径
- Qdrant 连接信息（host, port, collection_name, vector_size=512）
- embedding_server 连接信息（base_url, timeout, default_prompt）
- 调度时间（默认凌晨 3:00）
- 服务端口

### db.py — SQLite + FTS5
- `arxiv_papers` 表字段：arxiv_id(唯一), title, authors, abstract, categories(JSON数组), published_date, updated_date, pdf_url, abs_url, source(broad/keyword), created_at, updated_at
- FTS5 全文索引覆盖 title + abstract + authors
- 触发器保持 FTS5 索引与主表自动同步
- 接口：init_db, insert_paper(upsert), keyword_search(FTS5 bm25, 支持分类/时间过滤和排序), get_by_arxiv_ids(批量回填元数据), get_stats

### qdrant_store.py — Qdrant 向量存储
- Collection：512 维 Cosine 距离，HNSW 索引
- Payload 最小化：仅存 arxiv_id, title, categories, published_date（过滤和展示必要字段）
- point_id 用 arxiv_id 的确定性哈希
- 接口：init_collection(幂等), upsert_papers(批量), search_semantic(支持分类/时间 payload filter), delete_by_id

### embed_client.py — embedding_server 客户端
- 通过 HTTP 调用 embedding_server，不持有本地模型
- 使用 httpx（连接池、超时）
- encode_documents(texts) → 512 维向量列表（不加 instruct prompt）
- encode_query(query, prompt?) → 512 维向量（加 instruct prompt）
- rerank(query, documents) → 精排分数（CrossEncoder）
- health_check() → 可用性检查
- 单例模式

### retriever.py — 混合检索
- 三种检索模式：
  - keyword only → FTS5 全文检索
  - semantic_query only → Qdrant 语义检索
  - 两者都有 → 双路检索 + RRF 融合（k=60）
- 检索结果按 arxiv_id 回填 SQLite 完整元数据
- 支持分类过滤、时间范围、排序（relevance/date）、分页

### crawler.py — arXiv 数据采集
- 参考 `~/.hermes/scripts/arxiv_daily.py` 的采集策略
- Broad sweep：按分类拉取最新论文
- Keyword search：按关键词精确检索
- 去重：同一 arxiv_id 保留 keyword 来源
- 速率控制：请求间隔 10s + 指数退避重试
- 完整流程：fetch → dedup → 调 embedding_server 生成向量 → 写入 SQLite + Qdrant

### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/search` | 混合检索。keyword 和 semantic_query 至少提供一个 |
| GET | `/stats` | 库内统计（总数、分类分布、最后更新时间） |
| POST | `/admin/trigger-fetch` | 手动触发一次数据采集 |

`/search` 请求参数：keyword?, semantic_query?, categories?, date_from?, date_to?, sort_by(relevance|date), sort_order, limit, offset
返回：`{total, papers: [{arxiv_id, title, authors, abstract, categories, published_date, updated_date, pdf_url, abs_url, source}]}`

### main.py
- FastAPI lifespan 管理：启动时加载配置 → 初始化 SQLite → 连接 Qdrant → 初始化 embed_client → 启动调度器；关闭时清理资源
- 通过 `app.state` 传递共享资源（config, sqlite_conn, qdrant_client, embed_client）

## 测试

- SQLite 建表 + FTS5 查询正确性
- RRF 融合逻辑（纯函数，不依赖外部服务）
- 集成测试：采集 → embedding → 入库 → 检索端到端

## 实施顺序

按依赖关系，低级模块先行：

| # | 模块 | 依赖 |
|---|------|------|
| 1 | config.yaml + main.py 骨架 | 无 |
| 2 | db.py | config |
| 3 | qdrant_store.py | config |
| 4 | embed_client.py | 无 |
| 5 | retriever.py | db, qdrant, embed_client |
| 6 | models.py + router.py | retriever |
| 7 | crawler.py | db, qdrant, embed_client, config |
| 8 | main.py 完整版 | router, crawler |
| 9 | tests/ | 全部 |
