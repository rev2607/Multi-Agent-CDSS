"""LLM and embedding clients: Gemini primary, OpenRouter optional fallback."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

from app.core.config import get_settings, settings as _boot_settings

logger = logging.getLogger(__name__)

# Fixed size for local fallback dense vectors (matches config default)
FALLBACK_DIM = 768


def _settings():
    """Always read current settings (supports tests / env reloads)."""
    try:
        return get_settings()
    except Exception:
        return _boot_settings


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _hash_embed(text: str, dim: int = FALLBACK_DIM) -> List[float]:
    """Deterministic bag-of-tokens hash embedding for offline / no-key demos."""
    vec = [0.0] * dim
    tokens = _tokenize(text) or ["empty"]
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class LLMAuthError(RuntimeError):
    """Raised when provider API keys are missing/invalid (HTTP 401/403)."""


def _is_auth_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "error code: 401",
        "error code: 403",
        "status code: 401",
        "status code: 403",
        "unauthorized",
        "user not found",
        "invalid api key",
        "incorrect api key",
        "api key not valid",
        "authentication failed",
        "permission_denied",
    )
    return any(m in text for m in markers)


def _auth_help_message(provider: str, exc: BaseException) -> str:
    return (
        f"LLM authentication failed ({provider}): {exc}\n\n"
        "Fix in backend/.env then restart the API server:\n"
        "  • Gemini: set GEMINI_API_KEY from https://aistudio.google.com/apikey\n"
        "  • OpenRouter: set OPENROUTER_API_KEY from https://openrouter.ai/keys "
        "(or leave blank if unused)\n"
        "  • Set LLM_PROVIDER=gemini to use only Gemini\n"
        "  • Invalid OpenRouter keys cause 'User not found' (401) — remove or replace them"
    )


class LLMClient:
    """Unified chat + embedding interface with provider failover."""

    def __init__(self) -> None:
        self._gemini = None
        self._openai = None
        self._openrouter_disabled = False
        self._openrouter_disable_reason = ""
        self._provider = self._resolve_provider()
        self._init_clients()
        s = _settings()
        logger.info(
            "LLM client ready | provider=%s | gemini_key=%s | openrouter_key=%s",
            self._provider,
            "yes" if s.gemini_api_key else "no",
            "yes" if s.openrouter_api_key else "no",
        )

    def _resolve_provider(self) -> str:
        s = _settings()
        pref = (s.llm_provider or "auto").lower().strip()
        has_gemini = bool(s.effective_gemini_key)
        has_or = bool(s.effective_openrouter_key)

        if pref == "gemini":
            if has_gemini:
                return "gemini"
            logger.warning("LLM_PROVIDER=gemini but GEMINI_API_KEY is missing/placeholder")
        if pref == "openrouter":
            if has_or:
                return "openrouter"
            logger.warning(
                "LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is missing/placeholder"
            )
        # auto (default): Gemini first when available
        if has_gemini:
            return "gemini"
        if has_or:
            return "openrouter"
        logger.warning(
            "No valid LLM API keys configured — chat will use stub responses; "
            "embeddings use local hash vectors. Edit backend/.env"
        )
        return "stub"

    def _init_clients(self) -> None:
        s = _settings()
        gemini_key = s.effective_gemini_key
        or_key = s.effective_openrouter_key

        if gemini_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=gemini_key)
                self._gemini = genai
            except Exception as e:  # pragma: no cover
                logger.error("Failed to init Gemini: %s", e)

        # OpenRouter only when explicitly enabled AND not gemini-only
        pref = (s.llm_provider or "auto").lower().strip()
        allow_or = bool(or_key) and s.openrouter_enabled and pref != "gemini"
        if allow_or:
            try:
                from openai import OpenAI

                self._openai = OpenAI(
                    api_key=or_key,
                    base_url=s.openrouter_base_url,
                    default_headers={
                        "HTTP-Referer": "http://localhost:8000",
                        "X-Title": "Medical Multi-Agent CDSS",
                    },
                )
            except Exception as e:  # pragma: no cover
                logger.error("Failed to init OpenRouter client: %s", e)
        else:
            self._openai = None
            if s.openrouter_api_key:
                reason = (
                    f"ignored (LLM_PROVIDER={pref}, "
                    f"OPENROUTER_ENABLED={s.openrouter_enabled})"
                )
                self._openrouter_disabled = True
                self._openrouter_disable_reason = reason
                logger.info("OpenRouter not active: %s", reason)

    def _disable_openrouter(self, reason: str) -> None:
        if self._openrouter_disabled:
            return
        self._openrouter_disabled = True
        self._openrouter_disable_reason = reason
        self._openai = None
        logger.warning(
            "OpenRouter disabled for this process (will not retry): %s", reason
        )

    def _openrouter_available(self) -> bool:
        return self._openai is not None and not self._openrouter_disabled

    @property
    def provider(self) -> str:
        return self._provider

    def status(self) -> Dict[str, Any]:
        s = _settings()
        return {
            "provider": self._provider,
            "gemini_configured": bool(s.effective_gemini_key),
            "openrouter_configured": bool(s.effective_openrouter_key)
            and not self._openrouter_disabled
            and self._openai is not None,
            "openrouter_disabled": self._openrouter_disabled or self._openai is None,
            "openrouter_disable_reason": self._openrouter_disable_reason or None,
            "openrouter_model": s.openrouter_model,
            "gemini_model": s.gemini_model,
            "llm_provider_setting": s.llm_provider,
        }

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """Synchronous chat completion. messages: [{role, content}, ...]."""
        # Gemini 2.5 "thinking" models consume output budget; keep a usable floor
        if max_tokens < 256:
            max_tokens = 256

        # Gemini path — never fall back to OpenRouter by default
        # (invalid OpenRouter keys surface as 401 "User not found")
        if self._gemini is not None and self._provider != "openrouter":
            try:
                return self._chat_gemini(messages, temperature, max_tokens, json_mode)
            except Exception as e:
                logger.warning("Gemini chat failed: %s", e)
                s = _settings()
                # Explicit opt-in only
                if (
                    s.openrouter_enabled
                    and self._openrouter_available()
                    and (s.llm_provider or "").lower().strip() == "auto"
                ):
                    try:
                        return self._chat_openrouter(
                            messages, temperature, max_tokens, json_mode
                        )
                    except Exception as e2:
                        if _is_auth_error(e2):
                            self._disable_openrouter(str(e2))
                        raise e from e2
                if _is_auth_error(e):
                    raise LLMAuthError(_auth_help_message("gemini", e)) from e
                raise

        # Primary: OpenRouter only when explicitly selected
        if self._provider == "openrouter" and self._openrouter_available():
            try:
                return self._chat_openrouter(messages, temperature, max_tokens, json_mode)
            except Exception as e:
                if _is_auth_error(e):
                    self._disable_openrouter(str(e))
                    if self._gemini is not None:
                        logger.warning(
                            "OpenRouter auth failed; switching to Gemini for this process"
                        )
                        self._provider = "gemini"
                        return self._chat_gemini(
                            messages, temperature, max_tokens, json_mode
                        )
                    raise LLMAuthError(_auth_help_message("openrouter", e)) from e
                logger.warning("OpenRouter chat failed (%s); trying Gemini", e)
                if self._gemini is not None:
                    return self._chat_gemini(
                        messages, temperature, max_tokens, json_mode
                    )
                raise

        if self._gemini is not None:
            return self._chat_gemini(messages, temperature, max_tokens, json_mode)

        return self._chat_stub(messages)

    def _chat_gemini(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        assert self._gemini is not None
        s = _settings()
        system_bits = [m["content"] for m in messages if m["role"] == "system"]
        system = "\n\n".join(system_bits) if system_bits else None
        history: List[Dict[str, Any]] = []
        last_user = ""
        for m in messages:
            if m["role"] == "system":
                continue
            role = "user" if m["role"] == "user" else "model"
            if role == "user":
                last_user = m["content"]
            history.append({"role": role, "parts": [m["content"]]})

        # Use only completed turns for history; send last user as generate content
        if history and history[-1]["role"] == "user":
            history = history[:-1]

        # Prefer configured model; rotate on quota / not-found / empty
        model_candidates = [
            s.gemini_model,
            "gemini-2.5-flash",
            "gemini-flash-latest",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
        ]
        seen: set[str] = set()
        last_err: Optional[Exception] = None
        for model_name in model_candidates:
            if not model_name or model_name in seen:
                continue
            seen.add(model_name)
            try:
                model = self._gemini.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        **(
                            {"response_mime_type": "application/json"} if json_mode else {}
                        ),
                    },
                )
                chat = model.start_chat(history=history)
                resp = chat.send_message(last_user or "Continue.")
                text = self._safe_gemini_text(resp)
                if text:
                    return text
                last_err = RuntimeError(
                    f"Gemini model {model_name} returned empty text "
                    f"(finish_reason={self._gemini_finish_reason(resp)})"
                )
                logger.warning("%s; trying next model", last_err)
                continue
            except Exception as e:
                last_err = e
                err = str(e).lower()
                if any(
                    m in err
                    for m in (
                        "429",
                        "quota",
                        "404",
                        "not found",
                        "response.text",
                        "finish_reason",
                        "valid `part`",
                    )
                ):
                    logger.warning(
                        "Gemini model %s failed (%s); trying next", model_name, e
                    )
                    continue
                raise
        if last_err:
            raise last_err
        raise RuntimeError("Gemini chat failed with no models attempted")

    @staticmethod
    def _gemini_finish_reason(resp: Any) -> str:
        try:
            cands = getattr(resp, "candidates", None) or []
            if cands:
                return str(getattr(cands[0], "finish_reason", "?"))
        except Exception:
            pass
        return "?"

    @staticmethod
    def _safe_gemini_text(resp: Any) -> str:
        try:
            t = getattr(resp, "text", None)
            if t:
                return str(t).strip()
        except Exception:
            pass
        try:
            for cand in getattr(resp, "candidates", None) or []:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) or []
                chunks = []
                for p in parts:
                    if getattr(p, "text", None):
                        chunks.append(p.text)
                if chunks:
                    return "".join(chunks).strip()
        except Exception:
            pass
        return ""

    def _chat_openrouter(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        if not self._openrouter_available():
            raise RuntimeError(
                self._openrouter_disable_reason
                or "OpenRouter is not available"
            )
        assert self._openai is not None
        s = _settings()
        models_to_try = [
            s.openrouter_model,
            "openai/gpt-4o-mini",
            "google/gemini-2.5-flash",
            "google/gemini-flash-1.5",
        ]
        seen: set[str] = set()
        last_err: Optional[Exception] = None
        for model in models_to_try:
            if not model or model in seen:
                continue
            seen.add(model)
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = self._openai.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                if _is_auth_error(e):
                    self._disable_openrouter(str(e))
                    raise
                logger.warning("OpenRouter model %s failed: %s", model, e)
                continue
        if last_err:
            raise last_err
        raise RuntimeError("OpenRouter chat failed with no models attempted")

    def _chat_stub(self, messages: Sequence[Dict[str, str]]) -> str:
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return (
            "{\n"
            '  "summary": "Stub LLM response (configure GEMINI_API_KEY or OPENROUTER_API_KEY).",\n'
            f'  "note": "Last user message length: {len(last)} chars.",\n'
            '  "assessment": "Unable to generate clinical content without an API key.",\n'
            '  "recommendations": ["Set API keys in backend/.env and restart the server."]\n'
            "}"
        )

    # ------------------------------------------------------------- embeddings
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        s = _settings()
        if self._provider == "gemini" and self._gemini is not None:
            try:
                return self._embed_gemini(texts)
            except Exception as e:
                logger.warning("Gemini embed failed (%s); fallback", e)
        if self._openrouter_available() and s.openrouter_api_key:
            try:
                return self._embed_openrouter(texts)
            except Exception as e:
                if _is_auth_error(e):
                    self._disable_openrouter(str(e))
                logger.warning("OpenRouter embed failed (%s); hash fallback", e)
        return [_hash_embed(t, s.dense_vector_size) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def _embed_gemini(self, texts: Sequence[str]) -> List[List[float]]:
        assert self._gemini is not None
        s = _settings()
        out: List[List[float]] = []
        # Prefer modern embedding model names; older text-embedding-004 may 404
        models = [
            s.gemini_embedding_model,
            "models/text-embedding-004",
            "models/embedding-001",
            "models/gemini-embedding-001",
        ]
        last_err: Optional[Exception] = None
        for model in models:
            if not model:
                continue
            try:
                out = []
                for t in texts:
                    result = self._gemini.embed_content(
                        model=model,
                        content=t,
                        task_type="retrieval_document",
                    )
                    values = (
                        result["embedding"]
                        if isinstance(result, dict)
                        else result.embedding
                    )
                    out.append(list(values))
                if out and len(out[0]) != s.dense_vector_size:
                    object.__setattr__(s, "dense_vector_size", len(out[0]))
                return out
            except Exception as e:
                last_err = e
                logger.warning("Gemini embed model %s failed: %s", model, e)
                continue
        if last_err:
            raise last_err
        raise RuntimeError("Gemini embed failed")

    def _embed_openrouter(self, texts: Sequence[str]) -> List[List[float]]:
        assert self._openai is not None
        s = _settings()
        resp = self._openai.embeddings.create(
            model=s.openrouter_embedding_model,
            input=list(texts),
        )
        return [list(d.embedding) for d in resp.data]

    # --------------------------------------------------------------- vision
    def describe_image(self, image_bytes: bytes, mime_type: str, prompt: str) -> str:
        """OCR / clinical description of an image (handwritten notes, lesions, etc.)."""
        s = _settings()
        if self._gemini is not None and s.gemini_api_key:
            try:
                import io

                import PIL.Image

                model = self._gemini.GenerativeModel(s.gemini_vision_model)
                img = PIL.Image.open(io.BytesIO(image_bytes))
                resp = model.generate_content([prompt, img])
                text = self._safe_gemini_text(resp)
                if text:
                    return text
            except Exception as e:
                logger.warning("Gemini vision failed: %s", e)
        if self._openrouter_available():
            try:
                import base64

                b64 = base64.b64encode(image_bytes).decode("ascii")
                data_url = f"data:{mime_type};base64,{b64}"
                resp = self._openai.chat.completions.create(
                    model=s.openrouter_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                    max_tokens=2048,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                if _is_auth_error(e):
                    self._disable_openrouter(str(e))
                logger.warning("OpenRouter vision failed: %s", e)
        return (
            f"[Vision unavailable] Image ({mime_type}, {len(image_bytes)} bytes). "
            f"{prompt[:200]}"
        )

    # ---------------------------------------------------------------- utils
    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        return self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()
