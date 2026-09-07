"""Process-wide configuration: paths, env, and the handful of knobs that are
genuinely global. Everything strategy- or run-specific lives in YAML configs
instead, so that a run is reproducible from (commit, config, data_version).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=False)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


@dataclass(frozen=True)
class Paths:
    """Every on-disk location the lab uses. Resolved once, absolute always."""

    root: Path
    data: Path
    parquet: Path
    raw: Path
    runs: Path
    cache: Path
    cfg: Path
    strategies: Path
    registry_db: Path
    journal_db: Path

    def ensure(self) -> "Paths":
        for p in (self.data, self.parquet, self.raw, self.runs, self.cache):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class Settings:
    paths: Paths
    govgreed_api_key: str | None
    govgreed_base_url: str
    alpaca_key_id: str | None
    alpaca_secret_key: str | None
    alpaca_paper: bool
    anthropic_api_key: str | None
    agent_model: str
    #: Which LLM backend the agentic layer talks to: ``anthropic`` (API key),
    #: ``claude_code`` (the `claude -p` CLI, billed to a Claude subscription
    #: rather than an API key), ``openai`` (any OpenAI-compatible endpoint --
    #: OpenRouter, LM Studio, Ollama, vLLM, Together, OpenAI itself), or
    #: ``auto`` to pick the first one that can actually run.
    agent_provider: str
    agent_base_url: str | None
    agent_api_key: str | None
    claude_bin: str
    openrouter_api_key: str | None
    openai_api_key: str | None
    #: Sent by OpenRouter for leaderboard attribution; harmless elsewhere.
    openrouter_referer: str | None
    openrouter_title: str | None
    alert_webhook: str | None
    ui_token: str | None
    #: Whether the console may START a research session. Off by default and
    #: deliberately separate from every other control: research cannot touch a
    #: broker, but it does spend model budget and execute model-authored Python,
    #: so a compromised browser tab should not be able to begin one unasked.
    ui_allow_research: bool
    kill_switch_env: bool
    kill_file: Path
    extra: dict[str, Any] = field(default_factory=dict)

    # --- kill switch -------------------------------------------------------
    def kill_switch_engaged(self) -> tuple[bool, str | None]:
        """Env flag *or* a sentinel file, so a panic-stop needs no deploy."""
        if self.kill_switch_env:
            return True, "LAB_KILL_SWITCH env flag is set"
        if self.kill_file.exists():
            return True, f"kill file present at {self.kill_file}"
        return False, None


def _project_root() -> Path:
    # lab/config.py -> lab/ -> project root
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = _project_root()
    data = _env_path("LAB_DATA_DIR", str(root / "data"))
    runs = _env_path("LAB_RUNS_DIR", str(root / "runs"))
    paths = Paths(
        root=root,
        data=data,
        parquet=data / "parquet",
        raw=data / "raw",
        runs=runs,
        cache=data / "cache",
        cfg=root / "cfg",
        strategies=_env_path("LAB_STRATEGIES_DIR", str(root / "strategies")),
        registry_db=data / "registry.sqlite",
        journal_db=data / "journal.sqlite",
    ).ensure()

    return Settings(
        paths=paths,
        govgreed_api_key=os.getenv("GOVGREED_API_KEY") or None,
        govgreed_base_url=os.getenv(
            "GOVGREED_BASE_URL", "https://www.govgreed.com/api/v1"
        ).rstrip("/"),
        alpaca_key_id=os.getenv("ALPACA_API_KEY_ID") or None,
        alpaca_secret_key=os.getenv("ALPACA_API_SECRET_KEY") or None,
        alpaca_paper=_env_bool("ALPACA_PAPER", True),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        agent_model=os.getenv("LAB_AGENT_MODEL", "claude-sonnet-5"),
        agent_provider=(os.getenv("LAB_AGENT_PROVIDER") or "auto").strip().lower(),
        agent_base_url=(os.getenv("LAB_AGENT_BASE_URL") or "").rstrip("/") or None,
        agent_api_key=os.getenv("LAB_AGENT_API_KEY") or None,
        claude_bin=os.getenv("LAB_CLAUDE_BIN", "claude"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openrouter_referer=os.getenv("OPENROUTER_REFERER") or None,
        openrouter_title=os.getenv("OPENROUTER_TITLE") or None,
        alert_webhook=os.getenv("LAB_ALERT_WEBHOOK") or None,
        ui_token=os.getenv("LAB_UI_TOKEN") or None,
        ui_allow_research=_env_bool("LAB_UI_ALLOW_RESEARCH", False),
        kill_switch_env=_env_bool("LAB_KILL_SWITCH", False),
        kill_file=Path(os.getenv("LAB_KILL_FILE", str(data / "KILL"))).resolve(),
    )


def reset_settings_cache() -> None:
    """Tests mutate the environment; let them re-read it."""
    get_settings.cache_clear()
