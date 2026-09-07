"""Promote the strategy that produced a run into the operator's library.

Promotion keys off a **run_id**, never a file path, and that is the whole point
of the module. A research session writes into a scratch workspace and rewrites
the same filename on later turns, so the file sitting at the path a run recorded
is simply the last thing written there -- frequently a later, worse variant. On a
real session the agent's own recommended run scored 0.636 while the file left on
disk under that name scored 0.510, and nothing in the filesystem said so.

A run is immutable. Since every run now archives ``strategy.py`` alongside its
metrics, the code that produced a given result can be recovered exactly, checked
against the hash recorded at execution time, and copied into ``strategies/``
with a config naming the parameters it actually ran with.

Runs made before source archiving have only the hash. Those cannot be promoted,
and this says so rather than falling back to the path -- silently promoting the
wrong code is the failure this exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lab.config import get_settings


class PromotionError(RuntimeError):
    """Refused. The message says what to do about it."""


@dataclass
class Promotion:
    """What promoting a run did, or would do under ``dry_run``."""

    run_id: str
    strategy: str
    strategy_path: Path
    config_path: Path | None
    params: dict[str, Any] = field(default_factory=dict)
    overwrote: bool = False
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "strategy": self.strategy,
            "strategy_path": str(self.strategy_path),
            "config_path": None if self.config_path is None else str(self.config_path),
            "params": self.params,
            "overwrote": self.overwrote,
            "dry_run": self.dry_run,
        }


def _artifact_dir(run_id: str) -> Path:
    from lab.backtest.runner import _artifact_dir as d

    return d(run_id)


def _source_hash(text: str) -> str:
    """Must match ``load_strategy``'s, or verification is theatre."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_run_source(run_id: str) -> tuple[str, dict[str, Any]]:
    """The archived source for ``run_id`` and the config it ran under.

    Raises ``PromotionError`` if the run is unknown, predates source archiving,
    or carries source that does not hash to what was recorded -- an artifact
    edited after the fact is not evidence of anything.
    """
    directory = _artifact_dir(run_id)
    cfg_path = directory / "config.json"
    if not cfg_path.exists():
        raise PromotionError(
            f"no run artifacts for {run_id}. `lab runs list` shows what exists."
        )
    config = json.loads(cfg_path.read_text(encoding="utf-8"))

    src_path = directory / "strategy.py"
    if not src_path.exists():
        raise PromotionError(
            f"{run_id} has no archived source -- it predates source archiving, so the "
            f"exact code it ran cannot be recovered. Re-run the strategy to get a "
            f"promotable run. (Recorded hash was {config.get('strategy_hash') or 'unknown'}.)"
        )

    source = src_path.read_text(encoding="utf-8")
    recorded = str(config.get("strategy_hash") or "")
    actual = _source_hash(source)
    if recorded and actual != recorded:
        raise PromotionError(
            f"{run_id}'s archived source does not match the hash recorded when it ran "
            f"(recorded {recorded}, archive hashes to {actual}). The artifact has been "
            f"modified; refusing to promote it."
        )
    return source, config


def promote_run(
    run_id: str,
    *,
    name: str | None = None,
    strategies_dir: Path | None = None,
    config_dir: Path | None = None,
    write_config: bool = True,
    force: bool = False,
    dry_run: bool = False,
) -> Promotion:
    """Copy the code behind ``run_id`` into the library, with a matching config."""
    source, config = read_run_source(run_id)

    stem = (name or Path(str(config.get("strategy") or run_id)).stem).removesuffix(".py")
    if not stem or stem.startswith("_") or "/" in stem or "\\" in stem:
        raise PromotionError(f"{stem!r} is not a usable strategy name")

    settings = get_settings()
    lib = Path(strategies_dir) if strategies_dir else Path(settings.paths.strategies)
    target = lib / f"{stem}.py"

    overwrote = target.exists()
    if overwrote and not force:
        existing = target.read_text(encoding="utf-8")
        if _source_hash(existing) == _source_hash(source):
            # Already the same code. Nothing to do, and no reason to complain.
            overwrote = False
        else:
            raise PromotionError(
                f"{target} already exists with different code. Pass --force to replace "
                f"it, or --name to promote under another name."
            )

    params = dict(config.get("resolved_params") or {})
    cfg_out: Path | None = None
    if write_config:
        cfg_dir = Path(config_dir) if config_dir else Path("cfg")
        cfg_out = cfg_dir / f"{stem}.yaml"
        if cfg_out.exists() and not force:
            raise PromotionError(
                f"{cfg_out} already exists. Pass --force to replace it, or "
                f"--no-config to promote the strategy without one."
            )

    if not dry_run:
        lib.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
        if cfg_out is not None:
            cfg_out.parent.mkdir(parents=True, exist_ok=True)
            cfg_out.write_text(_render_config(stem, target, config, run_id), encoding="utf-8")

    return Promotion(
        run_id=run_id,
        strategy=f"{stem}.py",
        strategy_path=target,
        config_path=cfg_out,
        params=params,
        overwrote=overwrote,
        dry_run=dry_run,
    )


#: Copied from the run so the promoted config reproduces it. Anything absent from
#: the run's config is left out rather than defaulted -- a config that quietly
#: substitutes a different universe or date range does not reproduce anything.
_CARRIED = (
    "tickers", "timeframe", "source", "start", "end", "cash", "benchmark",
    "warmup", "regular_hours", "sources", "fills", "limits",
)


def _relative(path: Path) -> str:
    """Repo-relative where possible: a config with an absolute path is not portable."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _render_config(stem: str, target: Path, config: dict[str, Any], run_id: str) -> str:
    body: dict[str, Any] = {"strategy": _relative(target)}
    for key in _CARRIED:
        if config.get(key) is not None:
            body[key] = config[key]
    body["params"] = dict(config.get("resolved_params") or {})
    body["origin"] = "promoted"
    body["notes"] = f"Promoted from {run_id}. Params are the ones that run resolved."

    header = (
        f"# Promoted from run {run_id}.\n"
        f"#\n"
        f"# The params below are what that run actually resolved, not the strategy\n"
        f"# file's defaults -- an agent varies params per experiment, so the file\n"
        f"# alone does not reproduce the result.\n"
    )
    return header + yaml.safe_dump(body, sort_keys=False, default_flow_style=False)
