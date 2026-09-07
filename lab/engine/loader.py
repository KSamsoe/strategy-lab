"""Loading a strategy from a plain ``.py`` file.

A strategy is just a module. No base class, no registration decorator, no
plugin manifest -- because the fastest possible edit-test loop is "save file,
run command", and because an agent authoring a strategy should be writing
Python, not filling in a framework's blanks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from lab.config import get_settings


@dataclass
class LoadedStrategy:
    name: str
    instance: Any
    module: ModuleType
    path: Path
    params: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    source_hash: str = ""
    doc: str = ""

    def hook(self, name: str):
        fn = getattr(self.instance, name, None)
        return fn if callable(fn) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "params": self.params,
            "source_hash": self.source_hash,
            "doc": self.doc,
        }


def _import_module(path: Path) -> ModuleType:
    mod_name = f"lab_strategy_{path.stem}_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import a strategy from {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses and pickling inside the module resolve.
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


def _find_strategy_object(module: ModuleType, params: Mapping[str, Any]) -> Any:
    """Resolution order, most explicit first."""
    obj = getattr(module, "STRATEGY", None)
    if obj is not None:
        return obj

    build = getattr(module, "build", None)
    if callable(build):
        try:
            return build(dict(params))
        except TypeError:
            return build()

    cls = getattr(module, "Strategy", None)
    if isinstance(cls, type):
        return _instantiate(cls, params)

    candidates = [
        obj
        for _, obj in vars(module).items()
        if isinstance(obj, type)
        and callable(getattr(obj, "on_bar", None))
        and obj.__module__ == module.__name__
    ]
    if len(candidates) == 1:
        return _instantiate(candidates[0], params)
    if len(candidates) > 1:
        names = ", ".join(sorted(c.__name__ for c in candidates))
        raise ValueError(
            f"{module.__file__} defines several strategy classes ({names}); "
            f"set STRATEGY = <the one you mean> to disambiguate"
        )
    raise ValueError(
        f"{module.__file__} exposes no strategy: expected STRATEGY, build(params), "
        f"a class named Strategy, or exactly one class with an on_bar method"
    )


def _instantiate(cls: type, params: Mapping[str, Any]) -> Any:
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return cls()
    if not sig.parameters:
        return cls()
    if "params" in sig.parameters:
        return cls(params=dict(params))
    try:
        return cls(dict(params))
    except TypeError:
        return cls()


def resolve_path(path: str | Path) -> Path:
    """Accept a full path, a bare filename, or a strategy name."""
    p = Path(path)
    if p.exists():
        return p.resolve()
    strategies = get_settings().paths.strategies
    for candidate in (strategies / p.name, strategies / f"{p.stem}.py"):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"no strategy file at {path!r} or in {strategies}")


def load_strategy(path: str | Path, params: Mapping[str, Any] | None = None) -> LoadedStrategy:
    """Import ``path`` and return the strategy it exposes.

    Module-level ``PARAMS`` are defaults; the caller's ``params`` win, which is
    what makes sweeps and agent-authored variation a config change rather than
    an edit.
    """
    resolved = resolve_path(path)
    module = _import_module(resolved)

    defaults = dict(getattr(module, "PARAMS", {}) or {})
    merged = defaults | dict(params or {})

    instance = _find_strategy_object(module, merged)
    if not callable(getattr(instance, "on_bar", None)):
        raise ValueError(f"{resolved} strategy object has no callable on_bar")

    # Let an instance carry its own params, so a strategy can read self.params
    # as well as ctx.params.
    if not hasattr(instance, "params"):
        try:
            instance.params = merged
        except AttributeError:
            pass

    source = resolved.read_text(encoding="utf-8")
    return LoadedStrategy(
        name=getattr(module, "NAME", None) or resolved.stem,
        instance=instance,
        module=module,
        path=resolved,
        params=merged,
        source=source,
        source_hash=hashlib.sha256(source.encode("utf-8")).hexdigest()[:16],
        doc=(module.__doc__ or "").strip(),
    )


def discover(directory: Path | None = None) -> list[dict[str, Any]]:
    """Every loadable strategy in ``directory``, for `lab strategies` and the UI."""
    root = Path(directory) if directory else get_settings().paths.strategies
    out: list[dict[str, Any]] = []
    if not root.exists():
        return out
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        entry: dict[str, Any] = {"name": path.stem, "path": str(path)}
        try:
            loaded = load_strategy(path)
            entry |= {
                "params": loaded.params,
                "doc": loaded.doc.splitlines()[0] if loaded.doc else "",
                "source_hash": loaded.source_hash,
                "loadable": True,
            }
        except Exception as exc:
            entry |= {"loadable": False, "error": f"{type(exc).__name__}: {exc}"}
        out.append(entry)
    return out
