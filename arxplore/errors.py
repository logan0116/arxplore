"""
errors.py — 错误码定义

错误码格式：XXX-YYY
  XXX: 模块代码
  YYY: 具体错误编号

模块代码：
  100: 数据库错误
  200: Embedding 服务错误
  300: Qdrant 服务错误
  400: 检索器错误
  500: 爬虫错误
  600: API 路由错误
"""
from enum import Enum


class ErrorCode(str, Enum):
    # 100: 数据库错误
    DB_CONNECTION_FAILED = "100-001"  # 数据库连接失败
    DB_INIT_FAILED = "100-002"  # 数据库初始化失败
    DB_INSERT_FAILED = "100-003"  # 插入论文失败
    DB_QUERY_FAILED = "100-004"  # 查询失败
    DB_TRANSACTION_FAILED = "100-005"  # 事务执行失败

    # 200: Embedding 服务错误
    EMBED_CONNECTION_FAILED = "200-001"  # 无法连接 embedding 服务
    EMBED_ENCODE_DOCS_FAILED = "200-002"  # 文档向量化失败
    EMBED_ENCODE_QUERY_FAILED = "200-003"  # 查询向量化失败
    EMBED_RERANK_FAILED = "200-004"  # 重排序失败
    EMBED_TIMEOUT = "200-005"  # embedding 服务超时
    EMBED_INVALID_RESPONSE = "200-006"  # embedding 服务响应格式错误

    # 300: Qdrant 服务错误
    QDRANT_CONNECTION_FAILED = "300-001"  # 无法连接 Qdrant
    QDRANT_COLLECTION_FAILED = "300-002"  # 创建 collection 失败
    QDRANT_UPSERT_FAILED = "300-003"  # 向量 upsert 失败
    QDRANT_SEARCH_FAILED = "300-004"  # 向量搜索失败
    QDRANT_DELETE_FAILED = "300-005"  # 删除向量失败
    QDRANT_TIMEOUT = "300-006"  # Qdrant 服务超时

    # 400: 检索器错误
    RETRIEVER_KEYWORD_FAILED = "400-001"  # 关键词检索失败
    RETRIEVER_SEMANTIC_FAILED = "400-002"  # 语义检索失败
    RETRIEVER_FUSION_FAILED = "400-003"  # 结果融合失败
    RETRIEVER_BACKFILL_FAILED = "400-004"  # 元数据回填失败

    # 500: 爬虫错误
    CRAWLER_FETCH_FAILED = "500-001"  # 获取 HTML 失败
    CRAWLER_PARSE_FAILED = "500-002"  # 解析 HTML 失败
    CRAWLER_EMBED_FAILED = "500-003"  # 批量向量化失败
    CRAWLER_STORE_FAILED = "500-004"  # 存储失败
    CRAWLER_TIMEOUT = "500-005"  # 爬虫超时

    # 600: API 路由错误
    API_INVALID_REQUEST = "600-001"  # 请求参数无效
    API_SEARCH_FAILED = "600-002"  # 搜索失败
    API_STATS_FAILED = "600-003"  # 统计失败
    API_FETCH_FAILED = "600-004"  # 触发采集失败


class ArxploreError(Exception):
    """ArXplore 基础异常类"""

    def __init__(self, code: ErrorCode, message: str, details: str = None):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(f"[{code.value}] {message}")


class DatabaseError(ArxploreError):
    pass


class EmbedError(ArxploreError):
    pass


class QdrantError(ArxploreError):
    pass


class RetrieverError(ArxploreError):
    pass


class CrawlerError(ArxploreError):
    pass


class APIError(ArxploreError):
    pass
