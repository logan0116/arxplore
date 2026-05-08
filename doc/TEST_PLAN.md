# ArXplore 测试大纲

> 本文档基于 `doc/SPEC.md` 细化测试用例，作为测试实施的基准。

---

## 1. 测试分层概览

```
┌─────────────────────────────────────────────────┐
│              E2E / Integration Tests             │
│         (端到端: crawler → 入库 → API)           │
├─────────────────────────────────────────────────┤
│               API Tests (router.py)              │
│         POST /search, GET /stats,               │
│         POST /admin/trigger-fetch               │
├─────────────────────────────────────────────────┤
│              Service Tests                       │
│     retriever.py  |  crawler.py                 │
├──────────────┬──────────────┬────────────────────┤
│  Unit Tests  │  Unit Tests  │   Unit Tests      │
│   db.py     │ qdrant_store │  embed_client      │
├──────────────┴──────────────┴────────────────────┤
│              Mock / Fixture Layer                │
│    SQLite temp DB | Qdrant mock | HTTP mock     │
└─────────────────────────────────────────────────┘
```

---

## 2. Unit Tests

### 2.1 db.py — SQLite + FTS5

**测试文件**: `tests/test_db.py`

#### 2.1.1 init_db

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-001 | 连续调用两次 init_db（幂等） | 第二次不抛异常，表和触发器已存在 |
| T-002 | 建表后检查 arxiv_papers 表结构 | 11 个字段，arxiv_id PRIMARY KEY |
| T-003 | 建表后检查 FTS5 虚拟表存在 | FTS5 表存在且列名匹配 |
| T-004 | 建表后检查 3 个触发器存在 | INSERT/UPDATE/DELETE 触发器均在册 |

#### 2.1.2 insert_paper

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-010 | 插入新论文 | 成功写入，行数+1 |
| T-011 | 插入相同 arxiv_id 两次（UPSERT） | 第二条覆盖第一条，只剩一行 |
| T-012 | 插入包含特殊字符的 title/abstract | 正常写入，FTS5 触发器同步 |
| T-013 | 插入空 authors 或空 abstract | 写入成功（字段 NOT NULL 约束） |

#### 2.1.3 keyword_search

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-020 | 精确词匹配 title | 返回匹配文档，BM25 > 0 |
| T-021 | 模糊词匹配 abstract | 返回相关文档 |
| T-022 | 空查询字符串 | 返回空列表 |
| T-023 | 按 categories 过滤 | 仅返回指定分类的文档 |
| T-024 | 按 date_from 过滤 | 仅返回 >= 起始日期的文档 |
| T-025 | 按 date_to 过滤 | 仅返回 <= 结束日期的文档 |
| T-026 | categories + date_from + date_to 组合过滤 | 正确应用所有过滤条件 |
| T-027 | sort_by=bm25（默认） | 按 BM25 降序 |
| T-028 | sort_by=date, sort_order=asc | 按发表日期升序 |
| T-029 | limit=5 时仅返回 5 条 | 不超过指定 limit |
| T-030 | offset 跳过前 N 条 | 正确分页，不重复 |

#### 2.1.4 get_by_arxiv_ids

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-040 | 批量查询 3 个存在的 ID | 返回 3 条完整记录 |
| T-041 | 批量查询含不存在 ID | 仅返回存在的记录 |
| T-042 | 批量查询空列表 | 返回空列表 |

#### 2.1.5 get_stats

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-050 | 无数据时 stats | 返回 total=0，分类分布为空 |
| T-051 | 插入数据后 stats | total 正确，分类分布包含对应分类 |
| T-052 | 最后更新时间格式 | ISO 8601 字符串 |

#### 2.1.6 FTS5 触发器同步

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-060 | INSERT 后 FTS5 索引增加 | 新论文可通过 FTS5 查询到 |
| T-061 | UPDATE title 后 FTS5 更新 | 用新 title 可搜到，旧 title 搜不到 |
| T-062 | DELETE 后 FTS5 索引减少 | 被删论文无法通过 FTS5 查到 |

---

### 2.2 qdrant_store.py

**测试文件**: `tests/test_qdrant_store.py`

> 需启动 Qdrant 容器或使用 qdrant_client.FakeLocalDirector（mock）

#### 2.2.1 init_collection

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-100 | 首次调用创建 collection | collection 存在，vector_size=1024 |
| T-101 | 重复调用不抛异常 | 幂等 |

#### 2.2.2 upsert_papers

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-110 | 批量 upsert 5 篇论文 | 5 个 point 均成功写入 |
| T-111 | 同一 arxiv_id 重复 upsert | 覆盖更新，count 不变 |
| T-112 | point_id = sha256(arxiv_id)[:16] | ID 确定，同一篇论文多次 upsert 结果一致 |

#### 2.2.3 search_semantic

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-120 | 已知向量搜索 | 返回结果，score > 0 |
| T-121 | categories filter 过滤 | 仅返回指定分类的 point |
| T-122 | date_from filter | 仅返回 >= 起始日期 |
| T-123 | date_to filter | 仅返回 <= 结束日期 |
| T-124 | categories + date 组合过滤 | 同时满足所有条件 |
| T-125 | limit=3 限制返回数 | 最多 3 个 point |
| T-126 | 空向量搜索 | 返回空列表 |

#### 2.2.4 delete_by_id

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-130 | 删除存在的 point | 删除后 search 找不到 |
| T-131 | 删除不存在的 point | 不抛异常 |

---

### 2.3 embed_client.py

**测试文件**: `tests/test_embed_client.py`

> 需 mock httpx.AsyncClient，或启动 embedding_server（见 4.2）

#### 2.3.1 encode_documents

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-200 | 传入 ["doc1", "doc2"] | 返回 2 个向量，每个 1024 维 |
| T-201 | embed_type="document" | 请求体不含 prompt |
| T-202 | 空列表 | 返回空列表 |

#### 2.3.2 encode_query

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-210 | 传入 "query text" | 返回 1 个向量，1024 维 |
| T-211 | embed_type="query" | 请求体含 prompt |
| T-212 | 自定义 prompt 覆盖 default | 请求体使用自定义 prompt |

#### 2.3.3 rerank

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-220 | 正常调用 | 返回 list[dict]，含 index/document/score |
| T-221 | 返回结果按 score 降序 | 第一条 score >= 最后一条 |
| T-222 | 传入 5 个 docs | 返回 5 条记录，index 为原始位置 |

#### 2.3.4 health_check

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-230 | embedding_server 在线 | 返回 True |
| T-231 | embedding_server 离线 | 返回 False 或抛异常 |

#### 2.3.5 错误处理

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-240 | 响应 code != 200 | 抛出异常（含错误码信息） |
| T-241 | 网络超时 | 抛出超时异常 |

---

### 2.4 retriever.py — RRF 融合

**测试文件**: `tests/test_retriever.py`

> 纯函数单元测试，不依赖外部服务

#### 2.4.1 RRF 融合逻辑

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-300 | 单路检索（仅 keyword） | 结果等同于 FTS5 直接结果 |
| T-301 | 单路检索（仅 semantic） | 结果等同于 Qdrant 直接结果 |
| T-302 | 双路检索融合（无重复文档） | RRF score 正确 |
| T-303 | 双路检索融合（有重复文档） | 重复文档 score 叠加，排名提前 |
| T-304 | k=60 与 k=0 对比 | k=60 时排名差距缩小 |
| T-305 | limit=5 时仅返回 5 条 | 不超过 limit |
| T-306 | offset 分页 | 正确跳过前 N 条 |

#### 2.4.2 分页与排序

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-310 | sort_by=relevance（默认） | 按 RRF score 降序 |
| T-311 | sort_by=date, sort_order=asc | 按 published_date 升序 |
| T-312 | sort_by=date, sort_order=desc | 按 published_date 降序 |

#### 2.4.3 过滤条件传递

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-320 | categories 过滤 | 传递至 db.keyword_search 和 qdrant.search_semantic |
| T-321 | date_from/date_to 过滤 | 传递至两侧检索 |

#### 2.4.4 回填元数据

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-330 | RRF 融合结果回填 SQLite | 返回完整 paper 字段（title/authors/abstract...） |

---

### 2.5 models.py — Pydantic 模型

**测试文件**: `tests/test_models.py`

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-400 | SearchRequest 所有字段可空 | keyword/semantic_query 可不传 |
| T-401 | SearchRequest limit 默认值 | 默认 20 |
| T-402 | SearchRequest offset 默认值 | 默认 0 |
| T-403 | SearchResponse 结构正确 | 包含 total 和 papers 字段 |
| T-404 | Paper 结构与 SQLite 字段对应 | 11 个字段全部存在 |

---

## 3. API Tests

**测试文件**: `tests/test_api.py`

> 使用 TestClient 或 httpx.AsyncClient 测试 FastAPI

### 3.1 POST /search

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-500 | 仅 keyword 搜索 | 返回 FTS5 结果 |
| T-501 | 仅 semantic_query 搜索 | 返回 Qdrant 结果 |
| T-502 | keyword + semantic_query 混合 | 返回 RRF 融合结果 |
| T-503 | categories 过滤 | 仅返回指定分类 |
| T-504 | date_from + date_to 过滤 | 时间范围正确 |
| T-505 | sort_by=relevance, limit=5 | 按相关度排序，最多 5 条 |
| T-506 | offset=10 分页 | 跳过前 10 条 |
| T-507 | 空 keyword 且空 semantic_query | 返回 422 或错误提示 |
| T-508 | 不存在的分类过滤 | 返回空列表 |

### 3.2 GET /stats

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-510 | 初始状态 stats | total=0 |
| T-511 | 插入数据后 stats | total > 0，含分类分布 |

### 3.3 POST /admin/trigger-fetch

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-520 | 触发采集（无数据） | 返回进行中状态，arxiv 数据入库 |
| T-521 | 触发时已有数据 | 去重后增量入库 |
| T-522 | 触发后 stats 更新 | total 增加 |

---

## 4. Integration Tests

**测试文件**: `tests/test_integration.py`

> 需真实依赖：SQLite、Qdrant（可用 Docker）、embedding_server（如可访问）

### 4.1 完整采集流程

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-600 | fetch_from_arxiv（cat:cs，limit=3） | 获取 3 篇论文元数据 |
| T-601 | embedding 生成 | 调用 embed_client.encode_documents |
| T-602 | 写入 SQLite 后 FTS5 触发器同步 | 可通过 keyword_search 查到 |
| T-603 | 写入 Qdrant 后语义检索正常 | search_semantic 返回结果 |
| T-604 | trigger-fetch 端到端 | 数据成功入库，API 可检索 |

### 4.2 向量一致性

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-610 | 同一文本 encode 两次 | 向量完全一致（deterministic） |
| T-611 | encode_query 与 encode_documents 维度一致 | 均为 1024 维 |

### 4.3 RRF 与排序

| ID | 测试项 | 预期结果 |
|----|--------|---------|
| T-620 | 混合检索结果 relevance 排序 | RRF score 最高的在最前 |
| T-621 | 混合检索结果 date 排序 | published_date 最高的在最前 |
| T-622 | 检索结果包含完整的 paper 字段 | title/authors/abstract/pdf_url/abs_url 均存在 |

---

## 5. 测试数据准备

### 5.1 测试用论文数据

```python
SAMPLE_PAPERS = [
    {
        "arxiv_id": "2401.00001",
        "title": "Attention Is All You Need",
        "authors": '["Ashish Vaswani", "Noam Shazeer"]',
        "abstract": "We propose a new simple network architecture...",
        "categories": '["cs.CL", "cs.LG"]',
        "published_date": "2024-01-01",
        "pdf_url": "https://arxiv.org/pdf/2401.00001",
        "abs_url": "https://arxiv.org/abs/2401.00001",
        "source": "broad",
    },
    {
        "arxiv_id": "2401.00002",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": '["Jacob Devlin", "Ming-Wei Chang"]',
        "abstract": "We introduce a new language representation model...",
        "categories": '["cs.CL"]',
        "published_date": "2024-01-02",
        "pdf_url": "https://arxiv.org/pdf/2401.00002",
        "abs_url": "https://arxiv.org/abs/2401.00002",
        "source": "keyword",
    },
]
```

### 5.2 测试夹具（Fixtures）

| Fixture | 用途 |
|---------|------|
| `temp_db` | 临时 SQLite 文件，测试结束后删除 |
| `qdrant_collection` | 临时 collection，测试后清理 |
| `mock_embed_client` | Mock EmbedClient，避免真实 HTTP 调用 |
| `sample_papers` | 预置论文数据 |

---

## 6. 测试执行

### 6.1 本地执行

```bash
# 单元测试（无需外部依赖）
pytest tests/test_retriever.py tests/test_db.py tests/test_models.py -v

# API 测试（需 FastAPI 运行）
pytest tests/test_api.py -v

# 集成测试（需 Qdrant + embedding_server）
pytest tests/test_integration.py -v
```

### 6.2 CI 执行

- 单元测试：每次 PR 必须通过
- API 测试：需 docker-compose 启动依赖
- 集成测试：可标记 `pytest.mark.integration`，CI 按需跳过

---

## 7. 测试覆盖目标

| 模块 | 覆盖要求 |
|------|---------|
| retriever.py | RRF 融合逻辑 100% 覆盖 |
| db.py | FTS5 触发器 + 查询覆盖 90%+ |
| embed_client.py | 接口映射覆盖 100% |
| API endpoints | 所有端点 + 参数组合覆盖 |
| E2E | 采集→入库→检索完整路径 |

---

## 8. 验收与测试的对应关系

| 验收标准 | 对应测试 |
|---------|---------|
| POST /search 三种模式返回正确结果 | T-500, T-501, T-502 |
| GET /stats 返回准确统计 | T-510, T-511 |
| POST /admin/trigger-fetch 触发采集 | T-520, T-521, T-522 |
| FTS5 触发器正常工作 | T-060, T-061, T-062 |
| Qdrant 语义检索延迟 < 100ms | T-120（需性能测试） |
| RRF 融合正确工作 | T-300, T-301, T-302, T-303 |
| embed_client 正确调用 | T-200, T-201, T-210, T-220 |