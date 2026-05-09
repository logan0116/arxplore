# ArXplore

ArXiv 文献检索服务，支持关键词全文检索和语义相似度检索，提供 FastAPI RESTful API。

## 功能特性

- **混合检索**：关键词 FTS5 全文检索 + Qdrant 语义向量检索 + rerank 重排序
- **定时采集**：每日自动从 arXiv 拉取 CS 分类最新论文
- **RESTful API**：提供搜索、统计、触发采集等端点

## 技术栈

- **FastAPI** — Web 框架
- **SQLite + FTS5** — 全文检索
- **Qdrant** — 向量数据库
- **Qwen3-Embedding-0.6B** — 文本向量化

## 项目结构

```
arxplore/
├── main.py              # FastAPI 入口 + lifespan
├── router.py            # API 路由
├── retriever.py         # 混合检索编排
├── crawler.py           # arXiv 数据采集
├── db.py                # SQLite + FTS5
├── qdrant_store.py      # Qdrant 封装
├── embed_client.py      # embedding_server HTTP 客户端
├── models.py            # Pydantic 模型
├── config.yaml          # 配置文件
├── requirements.txt     # Python 依赖
└── tests/               # 测试套件
doc/
├── SPEC.md              # 技术方案设计文档
└── TEST_PLAN.md         # 测试大纲
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r arxplore/requirements.txt
```

### 2. 启动 Qdrant

```bash
docker run \
  -d --restart=always --name qdrant_server \
  -p 6333:6333 \
  -v ${PWD}/arxplore/data/qdrant:/qdrant/storage \
  qdrant/qdrant:v1.17
```

### 3. 启动服务

```bash
cd arxplore
python main.py
```

服务启动后访问：
- API 文档：http://localhost:4215/docs
- ReDoc：http://localhost:4215/redoc

### 4. 使用 Docker 部署

```bash
./start.sh
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/arxplore/search` | 混合检索 |
| GET | `/arxplore/stats` | 库内统计 |
| POST | `/arxplore/admin/trigger-fetch` | 手动触发采集 |

### 搜索示例

```bash
# 关键词搜索
curl -X POST http://localhost:4215/arxplore/search \
  -H "Content-Type: application/json" \
  -d '{"keyword": "machine learning"}'

# 语义搜索
curl -X POST http://localhost:4215/arxplore/search \
  -H "Content-Type: application/json" \
  -d '{"semantic_query": "deep learning for NLP"}'

# 混合搜索 + 过滤
curl -X POST http://localhost:4215/arxplore/search \
  -H "Content-Type: application/json" \
  -d '{"keyword": "neural", "categories": ["cs.LG"], "limit": 10}'
```

## 配置

配置文件：`arxplore/config.yaml`

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `database.path` | SQLite 数据库路径 | `./arxplore/data/arxiv.db` |
| `qdrant.host` | Qdrant 服务地址 | `localhost` |
| `qdrant.port` | Qdrant 端口 | `6333` |
| `embedding.base_url` | embedding_server 地址 | `http://192.168.1.116:9000` |
| `scheduler.hour` | 每日采集时间（小时） | `3` |
| `arxiv.max_results` | 单次采集上限 | `2000` |
| `arxiv.batch_size` | 向量化分批大小 | `64` |

## 测试

```bash
# 单元测试
pytest arxplore/tests/test_db.py arxplore/tests/test_retriever.py arxplore/tests/test_models.py -v

# API 测试
pytest arxplore/tests/test_api.py -v

# 全部测试
pytest arxplore/tests/ -v
```
