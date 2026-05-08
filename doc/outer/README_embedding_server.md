# Embedding Server

基于 SentenceTransformer 和 Qwen-Reranker 的语义 embedding 与重排序服务。

## 技术栈

- **[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)** - Embedding 模型
- **[Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)** - 重排序模型

## 模型信息

### Qwen3-Embedding-0.6B

| 属性 | 值 |
|------|-----|
| 模型类型 | Text Embedding |
| 支持语言 | 100+ 语言 |
| 参数量 | 0.6B |
| 上下文长度 | 32k |
| Embedding 维度 | Up to 1024 |

### Qwen3-Reranker-0.6B

| 属性 | 值 |
|------|-----|
| 模型类型 | Text Reranking |
| 支持语言 | 100+ 语言 |
| 参数量 | 0.6B |
| 上下文长度 | 32k |

## 快速启动

```bash
python main.py --embed_model_path /path/to/embedding/model --reranker_model_path /path/to/reranker/model
```

默认端口：`9000`

## 接口文档

### 1. get_embedding

获取句子的语义 embedding 向量。

**请求**

```
POST /api/get_embedding
Content-Type: application/json
```

**请求体字段说明**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `embed_type` | string | 否 | 编码类型，`"query"` 或 `"document"`（默认 `"document"`） |
| `prompt` | string | 否 | Instruct prompt，query 编码时使用（默认使用 `--default_prompt`） |
| `items` | array[string] | 是 | 待编码的句子列表 |

**输入示例**

```json
{
    "embed_type": "query",
    "prompt": "Given a web search query, retrieve relevant passages that answer the query",
    "items": ["What is the capital of China?", "Explain gravity"]
}
```

**输出示例**

```json
{
    "code": 200,
    "msg": "Success",
    "data": [
        [0.123, -0.456, 0.789, ...],
        [0.234, -0.567, 0.890, ...]
    ]
}
```

---

### 2. get_rank

两阶段重排序：Embedding 余弦相似度初筛 + CrossEncoder 精排。

**请求**

```
POST /api/get_rank
Content-Type: application/json
```

**请求体字段说明**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 查询字符串 |
| `documents` | array[string] | 是 | 待排序的文档列表（取 top 20 进入精排） |
| `query_prompt` | string | 否 | Instruct prompt（默认使用 `--default_prompt`） |

**输入示例**

```json
{
    "query": "find settings page",
    "documents": ["go to settings", "open home page", "about us"],
    "query_prompt": "Given a web search query, retrieve relevant passages that answer the query"
}
```

**响应字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | int | 状态码，200 表示成功 |
| `msg` | string | 状态信息 |
| `data` | array[object] | 排序结果数组，按 score 降序排列 |

**data 数组中每个对象的字段**

| 字段 | 类型 | 说明 |
|------|------|------|
| `index` | int | 原始文档索引 |
| `document` | string | 文档内容 |
| `score` | float | CrossEncoder 评分，范围 [0, 1]，越高表示越相关 |

**输出示例**

```json
{
    "code": 200,
    "msg": "Success",
    "data": [
        {"index": 0, "document": "go to settings", "score": 0.95},
        {"index": 1, "document": "open home page", "score": 0.23},
        {"index": 2, "document": "about us", "score": 0.08}
    ]
}
```

---

### 3. health

健康检查端点。

**请求**

```
GET /health
```

**输出示例**

```json
{"status": "ok", "code": 200}
```

---

## 错误码

| 错误码 | 说明 |
|--------|------|
| `200` | 成功 |
| `400` | 请求格式错误 |
| `422` | 请求参数校验失败（缺少必填字段、类型错误、取值非法） |
| `500` | 服务器内部错误（模型推理异常） |
| `503` | 模型服务不可用（模型加载失败或损坏） |

---

## 日志规范

服务启动和请求处理均输出结构化日志，格式如下：

```
2026-05-08 10:30:15 | INFO | __main__ | ==================================================
2026-05-08 10:30:15 | INFO | __main__ | Starting Embedding Server
2026-05-08 10:30:15 | INFO | __main__ | Embed model: /app/models/qwen3_embedding_0.6b
2026-05-08 10:30:15 | INFO | __main__ | Reranker model: /app/models/qwen3_reranker_0.6b
2026-05-08 10:30:15 | INFO | __main__ | Host: 0.0.0.0:9000
2026-05-08 10:30:15 | INFO | __main__ | Default prompt: Given a web search query...
2026-05-08 10:30:15 | INFO | __main__ | ==================================================
2026-05-08 10:30:15 | INFO | __main__ | Loading embedding model...
2026-05-08 10:30:18 | INFO | __main__ | Embedding model loaded successfully.
2026-05-08 10:30:18 | INFO | __main__ | Loading reranker model...
2026-05-08 10:30:20 | INFO | __main__ | Reranker model loaded successfully.
2026-05-08 10:30:20 | INFO | __main__ | All models loaded. Server is ready.
2026-05-08 10:31:05 | INFO | __main__ | [get_embedding] type=document, items_count=3, prompt=Given a web search query...
2026-05-08 10:31:05 | INFO | __main__ | [get_embedding] success, items=3, elapsed=0.215s
2026-05-08 10:31:12 | ERROR | __main__ | [get_rank] failed after 1.052s: cannot reshape tensor...
```

**日志级别说明**

| 级别 | 使用场景 |
|------|----------|
| `INFO` | 服务启动/关闭、请求开始/成功、模型加载状态 |
| `WARNING` | APIError 异常返回（如参数校验失败、bad request） |
| `ERROR` | 请求处理失败（异常堆栈输出，使用 `exc_info=True`） |

---

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--embed_model_path` | `/app/models/qwen3_embedding_0.6b` | Embedding 模型路径 |
| `--reranker_model_path` | `/app/models/qwen3_reranker_0.6b` | Reranker 模型路径 |
| `--host` | `0.0.0.0` | 服务监听 IP |
| `--port` | `9000` | 服务监听端口 |
| `--default_prompt` | `Given a web search query, retrieve relevant passages that answer the query` | 默认 instruct prompt |

---

## 测试

```bash
EMBEDDING_SERVER_URL=http://192.168.1.116:9000 pytest tests/test_api.py -v
```

| 测试类 | 测试项数 | 内容 |
|--------|----------|------|
| `TestGetEmbedding` | 22 | query/doc 编码、极限短文本、Unicode、特殊字符、类型校验、一致性、归一化验证 |
| `TestGetRank` | 18 | 正常排序、边界输入、Unicode、空值、重复文档、类型校验 |
| `TestEmbeddingQuality` | 3 | 相似度、反义词、语义聚类 |
| `TestRerankerQuality` | 4 | 相关性打分、完全匹配、不相关文档、排序顺序 |
| `TestConcurrency` | 3 | 并发 embedding、并发 rank、混合并发 |
| `TestErrorHandling` | 4 | 无效 JSON、空请求体、错误 Content-Type、方法不允许 |
| `TestHealthEndpoint` | 3 | 健康检查、code 字段、404 |

总计 **67** 个测试用例，完整覆盖接口功能、边界条件、质量验证、并发安全与错误处理。

---

## 性能基准

### Embedding 模型评测 (MTEB)

| Model | Mean (Task) | Retri. |
|-------|-------------|--------|
| Qwen3-Embedding-0.6B | 64.33 | 64.64 |
| Qwen3-Embedding-4B | 69.45 | 69.60 |
| Qwen3-Embedding-8B | **70.58** | **70.88** |

### Reranker 模型评测

| Model | MTEB-R | CMTEB-R | MMTEB-R |
|-------|--------|---------|---------|
| Qwen3-Reranker-0.6B | 65.80 | 71.31 | 66.36 |
| Qwen3-Reranker-4B | **69.76** | 75.94 | 72.74 |
| Qwen3-Reranker-8B | 69.02 | **77.45** | **72.94** |

---

## 注意事项

- **模型来源**：务必使用 HuggingFace 原版模型，ModelScope 下载的模型文件不包含 `sentence_transformers` 所需配置，会导致 reranker 无法正常工作
- **query vs document**：query 编码需要传入 `prompt`（instruct），document 编码不需要
- **重排序限制**：reranker 只对 embedding 初筛后的 top 20 文档进行精排