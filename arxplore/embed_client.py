"""
embed_client.py — embedding_server HTTP 客户端

封装对 embedding_server 的调用。
"""
from typing import Any, Optional

import httpx

from logging_config import get_logger
from errors import EmbedError, ErrorCode

logger = get_logger("embed")


class EmbedClient:
    _instance: Optional["EmbedClient"] = None

    def __new__(
        cls,
        base_url: str,
        timeout: int = 30,
        default_prompt: str = "",
    ):
        if cls._instance is not None:
            return cls._instance
        instance = super().__new__(cls)
        cls._instance = instance
        return instance

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        default_prompt: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_prompt = default_prompt
        self._client: httpx.AsyncClient | None = None
        logger.info(f"EmbedClient 初始化: {base_url}, timeout={timeout}s")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def encode_documents(self, texts: list[str]) -> list[list[float]]:
        logger.info(f"开始向量化文档，数量: {len(texts)}")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "embed_type": "document",
                    "items": texts,
                }
                response = await client.post(
                    f"{self.base_url}/api/get_embedding",
                    json=payload,
                )
                data = response.json()
                if data.get("code") != 200:
                    msg = data.get("msg", "Unknown error")
                    logger.error(f"文档向量化失败: [{data.get('code')}] {msg}")
                    raise EmbedError(
                        ErrorCode.EMBED_ENCODE_DOCS_FAILED,
                        f"文档向量化失败: {msg}",
                        f"code={data.get('code')}",
                    )
                result = data.get("data", [])
                logger.info(f"文档向量化完成，返回 {len(result)} 个向量")
                return result
        except httpx.TimeoutException as e:
            logger.error(f"文档向量化超时: {e}")
            raise EmbedError(ErrorCode.EMBED_TIMEOUT, "文档向量化超时", str(e))
        except httpx.ConnectError as e:
            logger.error(f"无法连接 embedding 服务: {e}")
            raise EmbedError(ErrorCode.EMBED_CONNECTION_FAILED, "无法连接 embedding 服务", str(e))
        except Exception as e:
            logger.error(f"文档向量化失败: {e}")
            raise EmbedError(ErrorCode.EMBED_ENCODE_DOCS_FAILED, "文档向量化失败", str(e))

    async def encode_query(
        self, query: str, prompt: Optional[str] = None
    ) -> list[float]:
        logger.debug(f"开始向量化查询: '{query[:50]}...'")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                use_prompt = prompt or self.default_prompt
                payload = {
                    "embed_type": "query",
                    "prompt": use_prompt,
                    "items": [query],
                }
                response = await client.post(
                    f"{self.base_url}/api/get_embedding",
                    json=payload,
                )
                data = response.json()
                if data.get("code") != 200:
                    msg = data.get("msg", "Unknown error")
                    logger.error(f"查询向量化失败: [{data.get('code')}] {msg}")
                    raise EmbedError(
                        ErrorCode.EMBED_ENCODE_QUERY_FAILED,
                        f"查询向量化失败: {msg}",
                        f"code={data.get('code')}",
                    )
                vectors = data.get("data", [])
                if not vectors:
                    logger.warning("查询向量化返回空向量")
                    return []
                logger.debug(f"查询向量化完成，向量维度: {len(vectors[0])}")
                return vectors[0]
        except httpx.TimeoutException as e:
            logger.error(f"查询向量化超时: {e}")
            raise EmbedError(ErrorCode.EMBED_TIMEOUT, "查询向量化超时", str(e))
        except httpx.ConnectError as e:
            logger.error(f"无法连接 embedding 服务: {e}")
            raise EmbedError(ErrorCode.EMBED_CONNECTION_FAILED, "无法连接 embedding 服务", str(e))
        except Exception as e:
            logger.error(f"查询向量化失败: {e}")
            raise EmbedError(ErrorCode.EMBED_ENCODE_QUERY_FAILED, "查询向量化失败", str(e))

    async def rerank(
        self, query: str, documents: list[str]
    ) -> list[dict[str, Any]]:
        logger.info(f"开始重排序: query='{query[:30]}...', documents={len(documents)}")
        try:
            client = await self._get_client()
            payload = {
                "query": query,
                "documents": documents,
                "query_prompt": self.default_prompt,
            }
            response = await client.post(
                f"{self.base_url}/api/get_rank",
                json=payload,
            )
            data = response.json()
            if data.get("code") != 200:
                msg = data.get("msg", "Unknown error")
                logger.error(f"重排序失败: [{data.get('code')}] {msg}")
                raise EmbedError(
                    ErrorCode.EMBED_RERANK_FAILED,
                    f"重排序失败: {msg}",
                    f"code={data.get('code')}",
                )
            result = data.get("data", [])
            logger.info(f"重排序完成，返回 {len(result)} 条结果")
            return result
        except httpx.TimeoutException as e:
            logger.error(f"重排序超时: {e}")
            raise EmbedError(ErrorCode.EMBED_TIMEOUT, "重排序超时", str(e))
        except httpx.ConnectError as e:
            logger.error(f"无法连接 embedding 服务: {e}")
            raise EmbedError(ErrorCode.EMBED_CONNECTION_FAILED, "无法连接 embedding 服务", str(e))
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            raise EmbedError(ErrorCode.EMBED_RERANK_FAILED, "重排序失败", str(e))

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Embedding 服务健康检查失败: {e}")
            return False

    async def close(self):
        if self._client:
            logger.info("关闭 EmbedClient HTTP 连接")
            await self._client.aclose()
            self._client = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EmbedClient":
        embed_cfg = config["embedding"]
        return cls(
            base_url=embed_cfg["base_url"],
            timeout=embed_cfg["timeout"],
            default_prompt=embed_cfg["default_prompt"],
        )
