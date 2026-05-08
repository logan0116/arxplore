"""
embed_client.py 单元测试 — embedding_server HTTP 客户端

覆盖：T-200 ~ T-241

说明：
- 使用 unittest.mock.patch 模拟 httpx.AsyncClient
- 不依赖真实 embedding_server
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx


# ---------------------------------------------------------------------------
# 模拟被测类（在实现前先定义接口契约）
# ---------------------------------------------------------------------------

class EmbedClient:
    """基于 embedding_server API 定义的 embed_client 接口"""

    def __init__(self, base_url: str, timeout: int = 30, default_prompt: str = ""):
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

    async def encode_query(self, query: str, prompt: str | None = None) -> list[float]:
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

    async def rerank(self, query: str, documents: list[str]) -> list[dict]:
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def embed_client():
    return EmbedClient(
        base_url="http://192.168.1.116:9000",
        timeout=30,
        default_prompt="Given a web search query, retrieve relevant passages that answer the query",
    )


# ---------------------------------------------------------------------------
# 2.3.1 encode_documents — T-200 ~ T-202
# ---------------------------------------------------------------------------

class TestEncodeDocuments:
    @pytest.mark.asyncio
    async def test_returns_two_vectors(self, embed_client):
        """T-200: 传入 ["doc1", "doc2"] 返回 2 个向量，每个 1024 维（模拟）"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": 200,
            "data": [
                [0.1] * 1024,
                [0.2] * 1024,
            ],
        }
        mock_response.status_code = 200

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await embed_client.encode_documents(["doc1", "doc2"])

        assert len(result) == 2
        assert len(result[0]) == 1024
        assert len(result[1]) == 1024

    @pytest.mark.asyncio
    async def test_embed_type_document_no_prompt(self, embed_client):
        """T-201: embed_type="document" 请求体不含 prompt"""
        captured_payload = {}

        async def capture_post(url, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"code": 200, "data": [[0.0] * 1024]}
            return mock_resp

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = capture_post
            await embed_client.encode_documents(["doc"])

        assert captured_payload.get("embed_type") == "document"
        assert "prompt" not in captured_payload

    @pytest.mark.asyncio
    async def test_empty_list(self, embed_client):
        """T-202: 空列表返回空列表"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 200, "data": []}

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await embed_client.encode_documents([])

        assert result == []


# ---------------------------------------------------------------------------
# 2.3.2 encode_query — T-210 ~ T-212
# ---------------------------------------------------------------------------

class TestEncodeQuery:
    @pytest.mark.asyncio
    async def test_returns_one_vector_1024_dim(self, embed_client):
        """T-210: 传入 query 返回 1 个向量，1024 维"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": 200,
            "data": [[0.5] * 1024],
        }

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await embed_client.encode_query("deep learning")

        assert len(result) == 1024

    @pytest.mark.asyncio
    async def test_embed_type_query_with_prompt(self, embed_client):
        """T-211: embed_type="query" 请求体含 prompt"""
        captured_payload = {}

        async def capture_post(url, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"code": 200, "data": [[0.0] * 1024]}
            return mock_resp

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = capture_post
            await embed_client.encode_query("test query")

        assert captured_payload.get("embed_type") == "query"
        assert "prompt" in captured_payload
        assert captured_payload["prompt"] == embed_client.default_prompt

    @pytest.mark.asyncio
    async def test_custom_prompt_overrides_default(self, embed_client):
        """T-212: 自定义 prompt 覆盖 default"""
        captured_payload = {}

        async def capture_post(url, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"code": 200, "data": [[0.0] * 1024]}
            return mock_resp

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = capture_post
            await embed_client.encode_query("test query", prompt="custom prompt")

        assert captured_payload["prompt"] == "custom prompt"


# ---------------------------------------------------------------------------
# 2.3.3 rerank — T-220 ~ T-222
# ---------------------------------------------------------------------------

class TestRerank:
    @pytest.mark.asyncio
    async def test_returns_list_of_dicts(self, embed_client):
        """T-220: 返回 list[dict]，含 index/document/score"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": 200,
            "data": [
                {"index": 0, "document": "doc a", "score": 0.95},
                {"index": 1, "document": "doc b", "score": 0.23},
            ],
        }

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await embed_client.rerank("query", ["doc a", "doc b"])

        assert len(result) == 2
        assert result[0]["index"] == 0
        assert result[0]["document"] == "doc a"
        assert result[0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_results_sorted_by_score_desc(self, embed_client):
        """T-221: 返回结果按 score 降序"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": 200,
            "data": [
                {"index": 0, "document": "doc a", "score": 0.95},
                {"index": 1, "document": "doc b", "score": 0.10},
            ],
        }

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await embed_client.rerank("query", ["doc a", "doc b"])

        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_five_docs_returns_five_records(self, embed_client):
        """T-222: 传入 5 个 docs 返回 5 条记录"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "code": 200,
            "data": [
                {"index": i, "document": f"doc{i}", "score": 1.0 - i * 0.1}
                for i in range(5)
            ],
        }

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            docs = [f"doc{i}" for i in range(5)]
            result = await embed_client.rerank("query", docs)

        assert len(result) == 5
        indices = [r["index"] for r in result]
        assert indices == list(range(5))


# ---------------------------------------------------------------------------
# 2.3.4 health_check — T-230 ~ T-231
# ---------------------------------------------------------------------------

class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_online(self, embed_client):
        """T-230: embedding_server 在线返回 True"""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await embed_client.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_offline(self, embed_client):
        """T-231: embedding_server 离线返回 False 或抛异常"""
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("connection refused")
            result = await embed_client.health_check()

        assert result is False


# ---------------------------------------------------------------------------
# 2.3.5 错误处理 — T-240 ~ T-241
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_nonzero_code_raises_exception(self, embed_client):
        """T-240: 响应 code != 200 抛出异常"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 500, "msg": "Internal error"}

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(RuntimeError, match="Embedding server error"):
                await embed_client.encode_documents(["doc"])

    @pytest.mark.asyncio
    async def test_network_timeout(self, embed_client):
        """T-241: 网络超时抛超时异常"""
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectTimeout("timed out")
            with pytest.raises(httpx.ConnectTimeout):
                await embed_client.encode_documents(["doc"])