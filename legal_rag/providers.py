"""Provider adapters for LLMs, OCR, embeddings, and reranking.

Adapters hide external service details behind small methods and always expose a
local fallback path where practical. This keeps the pipeline runnable without
API keys while still allowing stronger providers to be configured explicitly.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import math
import mimetypes
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .text_utils import tokenize


class ProviderError(RuntimeError):
    """Raised when an external or local model provider cannot complete a call."""

    pass


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int = 90) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(f"HTTP {exc.code} from {url}: {body[:800]}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Could not reach {url}: {exc}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise ProviderError(f"Timed out calling {url} after {timeout}s") from exc


def _env_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


class OpenAIClient:
    """Minimal OpenAI text and JSON generation client with model fallbacks."""

    def __init__(self, api_key: str | None = None, draft_model: str | None = None, mini_model: str | None = None):
        """Create a client from explicit values or environment variables."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.draft_model = draft_model or os.getenv("OPENAI_DRAFT_MODEL", "gpt-5.5")
        self.mini_model = mini_model or os.getenv("OPENAI_MINI_MODEL", "gpt-5.4-mini")
        self.text_model_fallbacks = _env_list(
            "OPENAI_TEXT_MODEL_FALLBACKS",
            f"{self.draft_model},gpt-5.4,gpt-5.4-mini",
        )
        self.json_model_fallbacks = _env_list(
            "OPENAI_JSON_MODEL_FALLBACKS",
            f"{self.mini_model},gpt-5.4,gpt-5.5",
        )

    @property
    def available(self) -> bool:
        """Return whether an API key is configured."""
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ProviderError("OPENAI_API_KEY is not set")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_text(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_output_tokens: int = 1800,
    ) -> str:
        """Generate plain text, trying the Responses API before chat fallback."""
        models = [model] if model else self.text_model_fallbacks
        errors: list[str] = []
        for candidate in models:
            try:
                return self._generate_text_responses(system, user, candidate, max_output_tokens)
            except ProviderError as exc:
                errors.append(f"{candidate}/responses: {exc}")
            try:
                return self._generate_text_chat(system, user, candidate)
            except ProviderError as exc:
                errors.append(f"{candidate}/chat: {exc}")
        raise ProviderError("OpenAI text generation failed: " + " | ".join(errors[-4:]))

    def _generate_text_responses(self, system: str, user: str, model: str, max_output_tokens: int) -> str:
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_output_tokens": max_output_tokens,
        }
        response = _post_json("https://api.openai.com/v1/responses", self._headers(), payload)
        text = _extract_response_text(response)
        if not text:
            raise ProviderError("Responses API returned no text output")
        return text

    def _generate_text_chat(self, system: str, user: str, model: str) -> str:
        chat_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        response = _post_json("https://api.openai.com/v1/chat/completions", self._headers(), chat_payload)
        return response["choices"][0]["message"]["content"]

    def generate_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        model: str | None = None,
        max_output_tokens: int = 1600,
    ) -> dict[str, Any]:
        """Generate JSON constrained by the supplied JSON Schema."""
        models = [model] if model else self.json_model_fallbacks
        errors: list[str] = []
        for candidate in models:
            try:
                return self._generate_json_responses(system, user, schema, candidate, max_output_tokens)
            except (ProviderError, json.JSONDecodeError) as exc:
                errors.append(f"{candidate}/responses: {exc}")
            try:
                return self._generate_json_chat(system, user, schema, candidate)
            except (ProviderError, json.JSONDecodeError) as exc:
                errors.append(f"{candidate}/chat: {exc}")
        raise ProviderError("OpenAI structured generation failed: " + " | ".join(errors[-4:]))

    def _generate_json_responses(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "structured_result",
                    "schema": schema,
                    "strict": True,
                }
            },
            "max_output_tokens": max_output_tokens,
        }
        response = _post_json("https://api.openai.com/v1/responses", self._headers(), payload)
        text = _extract_response_text(response)
        return json.loads(text)

    def _generate_json_chat(self, system: str, user: str, schema: dict[str, Any], model: str) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_result",
                    "schema": schema,
                    "strict": True,
                },
            },
        }
        response = _post_json("https://api.openai.com/v1/chat/completions", self._headers(), payload)
        return json.loads(response["choices"][0]["message"]["content"])


class OllamaClient:
    """Client for local Ollama chat/generate endpoints."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        """Create a local model client from explicit values or environment variables."""
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
        self.is_local = True
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", "900"))
        self.stream = os.getenv("OLLAMA_STREAM", "true").lower() not in {"0", "false", "no"}
        self.think = os.getenv("OLLAMA_THINK", "false").lower() in {"1", "true", "yes"}

    @property
    def available(self) -> bool:
        """Return whether the configured Ollama server responds to ``/api/tags``."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            return False

    def generate_text(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_output_tokens: int = 1800,
    ) -> str:
        """Generate text via Ollama chat, falling back to the generate endpoint."""
        payload = {
            "model": model or self.model,
            "stream": self.stream,
            "think": self.think,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": self._options(max_output_tokens, float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))),
        }
        if self.stream:
            try:
                return self._generate_text_stream(payload)
            except ProviderError as exc:
                if "streamed no message content" not in str(exc):
                    raise
                return self._generate_text_generate(system, user, max_output_tokens=max_output_tokens, model=model)
        response = _post_json(f"{self.base_url}/api/chat", {"Content-Type": "application/json"}, payload, timeout=self.timeout)
        if response.get("error"):
            raise ProviderError(f"Ollama error: {response['error']}")
        content = response.get("message", {}).get("content", "")
        if not content:
            return self._generate_text_generate(system, user, max_output_tokens=max_output_tokens, model=model)
        return str(content)

    def _generate_text_stream(self, payload: dict[str, Any]) -> str:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        chunks: list[str] = []
        thinking_chars = 0
        errors: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if item.get("error"):
                        errors.append(str(item["error"]))
                    if item.get("message", {}).get("content"):
                        chunks.append(str(item["message"]["content"]))
                    if item.get("message", {}).get("thinking"):
                        thinking_chars += len(str(item["message"]["thinking"]))
                    if item.get("done"):
                        break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"HTTP {exc.code} from Ollama: {body[:800]}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"Could not reach Ollama: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ProviderError(f"Timed out streaming from Ollama after {self.timeout}s") from exc
        text = "".join(chunks).strip()
        if not text:
            detail = "; ".join(errors) if errors else f"thinking_chars={thinking_chars}"
            raise ProviderError(f"Ollama streamed no message content ({detail})")
        return text

    def _generate_text_generate(
        self,
        system: str,
        user: str,
        *,
        max_output_tokens: int,
        model: str | None = None,
    ) -> str:
        payload = {
            "model": model or self.model,
            "stream": self.stream,
            "think": self.think,
            "system": system,
            "prompt": user,
            "options": self._options(max_output_tokens, float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))),
        }
        if not self.stream:
            response = _post_json(
                f"{self.base_url}/api/generate",
                {"Content-Type": "application/json"},
                payload,
                timeout=self.timeout,
            )
            if response.get("error"):
                raise ProviderError(f"Ollama error: {response['error']}")
            text = str(response.get("response", "")).strip()
            if not text:
                raise ProviderError("Ollama generate endpoint returned no text")
            return text
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        chunks: list[str] = []
        errors: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if item.get("error"):
                        errors.append(str(item["error"]))
                    if item.get("response"):
                        chunks.append(str(item["response"]))
                    if item.get("done"):
                        break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"HTTP {exc.code} from Ollama generate: {body[:800]}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"Could not reach Ollama generate: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ProviderError(f"Timed out streaming from Ollama generate after {self.timeout}s") from exc
        text = "".join(chunks).strip()
        if not text:
            detail = "; ".join(errors) if errors else "no response chunks"
            raise ProviderError(f"Ollama generate endpoint returned no text ({detail})")
        return text

    def generate_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        model: str | None = None,
        max_output_tokens: int = 1600,
    ) -> dict[str, Any]:
        """Generate JSON through Ollama's JSON response format."""
        prompt = (
            f"{user}\n\nReturn only valid JSON matching this JSON Schema:\n"
            f"{json.dumps(schema, indent=2)}"
        )
        payload = {
            "model": model or self.model,
            "stream": False,
            "format": "json",
            "think": self.think,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": self._options(max_output_tokens, 0.0),
        }
        response = _post_json(f"{self.base_url}/api/chat", {"Content-Type": "application/json"}, payload, timeout=self.timeout)
        content = response.get("message", {}).get("content", "{}")
        return json.loads(content)

    def _options(self, max_output_tokens: int, temperature: float) -> dict[str, Any]:
        options: dict[str, Any] = {
            "num_predict": max_output_tokens,
            "temperature": temperature,
        }
        num_thread = _env_int("OLLAMA_NUM_THREAD")
        if num_thread:
            options["num_thread"] = num_thread
        num_ctx = _env_int("OLLAMA_NUM_CTX")
        if num_ctx:
            options["num_ctx"] = num_ctx
        return options


def _extract_response_text(response: dict[str, Any]) -> str:
    if "output_text" in response:
        return str(response["output_text"])
    chunks: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if "text" in content:
                chunks.append(str(content["text"]))
    return "\n".join(chunks).strip()


class MistralOCRClient:
    """Mistral OCR adapter for PDFs and image files."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """Create an OCR client from explicit values or environment variables."""
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model = model or os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")

    @property
    def available(self) -> bool:
        """Return whether an OCR API key is configured."""
        return bool(self.api_key)

    def process_file(self, path: str | Path) -> dict[str, Any]:
        """Send a PDF or image to Mistral OCR and return the raw response."""
        if not self.api_key:
            raise ProviderError("MISTRAL_API_KEY is not set")
        p = Path(path)
        mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(p.read_bytes()).decode("ascii")
        if mime.startswith("image/"):
            document = {"type": "image_url", "image_url": f"data:{mime};base64,{encoded}"}
        else:
            document = {"type": "document_url", "document_url": f"data:{mime};base64,{encoded}"}
        payload = {
            "model": self.model,
            "document": document,
            "include_image_base64": False,
            "table_format": "markdown",
            "extract_header": True,
            "extract_footer": True,
            "confidence_scores_granularity": "page",
        }
        return _post_json(
            "https://api.mistral.ai/v1/ocr",
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload,
            timeout=180,
        )


class EmbeddingProvider:
    """Interface implemented by all embedding backends."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts as dense vectors."""
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        """Return a serializable provider fingerprint for cache invalidation."""
        return {"backend": "unknown"}


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding backend with hashing fallback when no key is present."""

    def __init__(self, api_key: str | None = None, model: str | None = None, fallback: EmbeddingProvider | None = None):
        """Create an embedding provider from explicit values or environment variables."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        self.fallback = fallback or HashingEmbeddingProvider()

    @property
    def available(self) -> bool:
        """Return whether OpenAI embeddings can be called directly."""
        return bool(self.api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches, falling back per batch after retry failure."""
        if not self.api_key:
            return self.fallback.embed_texts(texts)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            payload = {"model": self.model, "input": batch}
            for attempt in range(3):
                try:
                    response = _post_json("https://api.openai.com/v1/embeddings", headers, payload, timeout=120)
                    ordered = sorted(response["data"], key=lambda item: item["index"])
                    embeddings.extend([item["embedding"] for item in ordered])
                    break
                except ProviderError:
                    if attempt == 2:
                        embeddings.extend(self.fallback.embed_texts(batch))
                    time.sleep(1.0 + attempt)
        return embeddings

    def config(self) -> dict[str, Any]:
        """Return the active embedding backend configuration."""
        if not self.api_key:
            return self.fallback.config()
        return {"backend": "openai", "model": self.model}


class LocalSentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Sentence Transformers embedding backend for local/private runs."""

    def __init__(
        self,
        model: str | None = None,
        *,
        device: str | None = None,
        batch_size: int | None = None,
        fallback: EmbeddingProvider | None = None,
    ):
        """Configure the local embedding model, device, batch size, and fallback."""
        self.model_name = model or os.getenv("LOCAL_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
        self.device = device or os.getenv("LOCAL_EMBEDDING_DEVICE", "cpu")
        self.batch_size = batch_size or int(os.getenv("LOCAL_EMBEDDING_BATCH_SIZE", "4"))
        self.fallback = fallback or HashingEmbeddingProvider()
        self._model: Any | None = None

    @property
    def available(self) -> bool:
        """Return whether Sentence Transformers is installed."""
        return importlib.util.find_spec("sentence_transformers") is not None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts locally, optionally falling back to hashing on failure."""
        if not texts:
            return []
        if not self.available:
            if os.getenv("LOCAL_EMBEDDING_FALLBACK", "false").lower() not in {"1", "true", "yes"}:
                raise ProviderError("sentence_transformers is not installed for local embeddings")
            return self.fallback.embed_texts(texts)
        try:
            model = self._load_model()
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [list(map(float, row)) for row in embeddings]
        except Exception as exc:
            if os.getenv("LOCAL_EMBEDDING_FALLBACK", "false").lower() not in {"1", "true", "yes"}:
                raise ProviderError(f"local embedding model failed: {exc}") from exc
            print(f"Warning: local embedding model failed; using hashing fallback: {exc}", file=sys.stderr)
            return self.fallback.embed_texts(texts)

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
        return self._model

    def config(self) -> dict[str, Any]:
        """Return a cache key for the local model/device settings."""
        return {
            "backend": "local_sentence_transformer",
            "model": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "provider_version": 2,
        }


class HashingEmbeddingProvider(EmbeddingProvider):
    """Small deterministic embedding fallback for offline demos and tests."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with deterministic feature hashing."""
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def config(self) -> dict[str, Any]:
        """Return the hashing backend configuration."""
        return {"backend": "hashing", "dimensions": self.dimensions}


def cosine(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity, tolerating empty or mismatched vectors."""
    if not a or not b:
        return 0.0
    limit = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(limit))
    an = math.sqrt(sum(a[i] * a[i] for i in range(limit))) or 1.0
    bn = math.sqrt(sum(b[i] * b[i] for i in range(limit))) or 1.0
    return dot / (an * bn)


class LocalCrossEncoderReranker:
    """Local Sentence Transformers cross-encoder reranker."""

    def __init__(self, model: str | None = None, *, device: str | None = None, batch_size: int | None = None):
        """Configure the reranker model, device, batch size, and text limit."""
        self.model_name = model or os.getenv("LOCAL_RERANK_MODEL", "BAAI/bge-reranker-base")
        self.device = device or os.getenv("LOCAL_RERANK_DEVICE", "cpu")
        self.batch_size = batch_size or int(os.getenv("LOCAL_RERANK_BATCH_SIZE", "8"))
        self.max_text_chars = int(os.getenv("LOCAL_RERANK_TEXT_CHARS", "1200"))
        self._model: Any | None = None

    @property
    def available(self) -> bool:
        """Return whether Sentence Transformers is installed."""
        return importlib.util.find_spec("sentence_transformers") is not None

    def rerank(self, query: str, chunks: list[Any], *, top_k: int) -> list[Any]:
        """Return chunks sorted by local cross-encoder relevance scores."""
        if not chunks or not self.available:
            return chunks[:top_k]
        try:
            model = self._load_model()
            pairs = [(query, chunk.text[: self.max_text_chars]) for chunk in chunks]
            scores = model.predict(pairs, batch_size=self.batch_size)
            scored = sorted(zip(chunks, scores), key=lambda item: float(item[1]), reverse=True)
            reranked: list[Any] = []
            for rank, (chunk, score) in enumerate(scored[:top_k], start=1):
                chunk.score = float(score)
                chunk.vector_rank = rank
                reranked.append(chunk)
            return reranked
        except Exception as exc:
            print(f"Warning: local reranker failed; using fused retrieval order: {exc}", file=sys.stderr)
            return chunks[:top_k]

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model


def make_embedding_provider(backend: str = "auto") -> EmbeddingProvider:
    """Factory for CLI-selected embedding providers."""
    if backend == "local":
        return LocalSentenceTransformerEmbeddingProvider()
    if backend == "hashing":
        return HashingEmbeddingProvider()
    if backend == "openai":
        return OpenAIEmbeddingProvider()
    return OpenAIEmbeddingProvider()


def embedding_provider_from_config(config: dict[str, Any] | None) -> EmbeddingProvider:
    """Restore an embedding provider from an index configuration file."""
    if not config:
        return OpenAIEmbeddingProvider()
    backend = config.get("backend")
    if backend == "local_sentence_transformer":
        return LocalSentenceTransformerEmbeddingProvider(model=config.get("model"), device=config.get("device"))
    if backend == "hashing":
        return HashingEmbeddingProvider(dimensions=int(config.get("dimensions", 384)))
    if backend == "openai":
        return OpenAIEmbeddingProvider(model=config.get("model"))
    return OpenAIEmbeddingProvider()


def make_reranker(backend: str = "none") -> LocalCrossEncoderReranker | None:
    """Factory for CLI-selected reranker providers."""
    if backend == "local":
        return LocalCrossEncoderReranker()
    return None
