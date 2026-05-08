from typing import Any, Optional

import httpx


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

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def encode_documents(self, texts: list[str]) -> list[list[float]]:
        client = await self._get_client()
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
            raise RuntimeError(f"Embedding server error: {data.get('msg')}")
        return data.get("data", [])

    async def encode_query(
        self, query: str, prompt: Optional[str] = None
    ) -> list[float]:
        client = await self._get_client()
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
            raise RuntimeError(f"Embedding server error: {data.get('msg')}")
        vectors = data.get("data", [])
        return vectors[0] if vectors else []

    async def rerank(
        self, query: str, documents: list[str]
    ) -> list[dict[str, Any]]:
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
            raise RuntimeError(f"Rerank server error: {data.get('msg')}")
        return data.get("data", [])

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        if self._client:
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