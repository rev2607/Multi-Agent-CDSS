"""Application configuration loaded from environment / .env."""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]

# Treat common template / dummy values as unset
_PLACEHOLDER_KEY_MARKERS = (
    "your_",
    "changeme",
    "replace_me",
    "xxx",
    "todo",
    "example",
    "paste",
    "insert",
    "api_key_here",
    "sk-or-v1-xxxx",
)


def _clean_secret(value: object) -> str:
    """Strip quotes/whitespace; drop placeholder keys from .env.example copies."""
    if value is None:
        return ""
    s = str(value).strip().strip('"').strip("'")
    if not s:
        return ""
    low = s.lower()
    if any(m in low for m in _PLACEHOLDER_KEY_MARKERS):
        return ""
    # Real Gemini keys are typically longer than short placeholders
    if low.startswith("your"):
        return ""
    return s


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "models/text-embedding-004"
    gemini_vision_model: str = "gemini-2.5-flash"

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_embedding_model: str = "openai/text-embedding-3-large"

    llm_provider: str = "gemini"  # gemini | openrouter | auto
    # OpenRouter OFF by default — invalid system keys caused 401 "User not found"
    # Set OPENROUTER_ENABLED=true and a valid key only if you want OpenRouter.
    openrouter_enabled: bool = False

    @field_validator("gemini_api_key", "openrouter_api_key", mode="before")
    @classmethod
    def clean_api_keys(cls, v: object) -> str:
        return _clean_secret(v)

    @property
    def effective_openrouter_key(self) -> str:
        """OpenRouter key only when enabled and not forced to Gemini-only."""
        pref = (self.llm_provider or "auto").lower().strip()
        if pref == "gemini" or not self.openrouter_enabled:
            return ""
        return self.openrouter_api_key or ""

    @property
    def effective_gemini_key(self) -> str:
        pref = (self.llm_provider or "auto").lower().strip()
        if pref == "openrouter":
            # Still allow Gemini as fallback unless user has no key
            return self.gemini_api_key or ""
        return self.gemini_api_key or ""

    # Paths
    data_dir: Path = Field(default=BACKEND_ROOT / "data")
    upload_dir: Path = Field(default=BACKEND_ROOT / "data" / "uploads")
    qdrant_path: Path = Field(default=BACKEND_ROOT / "data" / "qdrant")
    sqlite_path: Path = Field(default=BACKEND_ROOT / "data" / "cdss.db")
    sample_data_dir: Path = Field(default=BACKEND_ROOT / "data" / "sample")

    # Retrieval
    qdrant_collection: str = "medical_kb"
    hybrid_top_k: int = 12
    rrf_k: int = 60
    enable_rerank: bool = False
    dense_vector_size: int = 768
    # Evidence quality (post RRF): top 4–6 specialty-matched, case-isolated chunks
    evidence_top_k: int = 5
    retrieval_min_relevance: float = 0.22
    agentic_max_steps: int = 3
    agentic_wall_clock_sec: int = 45
    agentic_weak_hit_threshold: float = 0.25

    # Whisper
    whisper_model: str = "tiny"
    whisper_enabled: bool = True

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    log_level: str = "INFO"

    @field_validator(
        "data_dir",
        "upload_dir",
        "qdrant_path",
        "sqlite_path",
        "sample_data_dir",
        mode="before",
    )
    @classmethod
    def resolve_paths(cls, v: object) -> Path:
        p = Path(str(v))
        if not p.is_absolute():
            p = (BACKEND_ROOT / p).resolve()
        return p

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.sample_data_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
