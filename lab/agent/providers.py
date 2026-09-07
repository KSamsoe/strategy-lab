"""Where the agentic layer's tokens actually come from.

Three backends, one shape:

``anthropic``    the Messages API, billed to ``ANTHROPIC_API_KEY``.
``claude_code``  the ``claude -p`` CLI, billed to a **Claude subscription**. No
                 API key at all -- the CLI carries its own auth.
``openai``       any OpenAI-compatible ``/chat/completions`` endpoint:
                 OpenRouter, LM Studio, Ollama, vLLM, Together, Groq, OpenAI.

Every provider is adapted to the **Anthropic Messages request/response shape**
rather than the layer above being taught three dialects. That is the whole trick
here: ``schemas.call_tool`` -- the single audited chokepoint that writes the
``agent_calls`` ledger, extracts the tool block and prices the exchange -- keeps
working unchanged, and so does every test written against it. A provider's job
is translation, nothing else.

A note on structured output, because the guarantee genuinely differs. The
Anthropic and OpenAI paths can *force* a tool call, so a well-formed payload is
close to guaranteed. ``claude -p`` cannot: it takes a prompt and returns prose,
so the schema is injected into the prompt and the reply is parsed out. That is
acceptable here for one specific reason -- ``schemas.validate_targets`` already
treats model output as untrusted input and the risk gate re-validates
everything downstream -- but it is a weaker guarantee and the provider reports
``forces_tools = False`` so callers can see it rather than assume it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lab.config import get_settings

_log = logging.getLogger(__name__)

#: Providers that can guarantee a schema-shaped reply, and the one that cannot.
FORCES_TOOLS = {"anthropic": True, "openai": True, "claude_code": False}

#: Where an OpenAI-compatible provider points when nothing says otherwise.
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENAI_URL = "https://api.openai.com/v1"


class ProviderUnavailable(RuntimeError):
    """The chosen backend cannot run, with a reason a human can act on."""


# --- the Anthropic-shaped response every provider synthesizes -------------------


@dataclass(slots=True)
class Block:
    """One content block. ``tool_use`` blocks carry the parsed arguments."""

    type: str
    text: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(slots=True)
class Response:
    """Shaped exactly like an ``anthropic.types.Message`` as far as the reader
    cares, plus two fields the reader treats as optional.

    ``cost_usd`` is set when the backend reports real money (OpenRouter's usage
    accounting, Claude Code's own tally) so the ledger can prefer a measured
    figure over the price-table estimate. ``billing`` says what that number
    means, which matters: on a subscription the dollars are notional.
    """

    model: str
    content: list[Block] = field(default_factory=list)
    stop_reason: str = ""
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    billing: str = "api"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderInfo:
    name: str
    model: str
    available: bool
    reason: str = ""
    billing: str = "api"
    base_url: str | None = None
    forces_tools: bool = True
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "available": self.available,
            "reason": self.reason,
            "billing": self.billing,
            "base_url": self.base_url,
            "forces_tools": self.forces_tools,
            "detail": self.detail,
        }


class _Messages:
    """Gives a provider the ``client.messages.create(**kw)`` surface that
    ``call_tool`` already speaks."""

    __slots__ = ("_create",)

    def __init__(self, create) -> None:
        self._create = create

    def create(self, **request: Any) -> Response:
        return self._create(**request)


class BaseProvider:
    name = ""
    billing = "api"

    def __init__(self, *, model: str | None = None, timeout: float = 120.0) -> None:
        self.settings = get_settings()
        self.model = model or self.settings.agent_model
        self.timeout = timeout

    @property
    def messages(self) -> _Messages:
        return _Messages(self.create)

    @property
    def forces_tools(self) -> bool:
        return FORCES_TOOLS.get(self.name, True)

    def available(self) -> tuple[bool, str]:
        return True, ""

    def info(self) -> ProviderInfo:
        ok, reason = self.available()
        return ProviderInfo(
            name=self.name,
            model=self.model,
            available=ok,
            reason=reason,
            billing=self.billing,
            base_url=getattr(self, "base_url", None),
            forces_tools=self.forces_tools,
        )

    def create(self, **request: Any) -> Response:  # pragma: no cover - abstract
        raise NotImplementedError

    def require(self) -> "BaseProvider":
        ok, reason = self.available()
        if not ok:
            raise ProviderUnavailable(f"{self.name}: {reason}")
        return self


# --- helpers shared by the non-Anthropic providers ------------------------------


def flatten_content(content: Any) -> str:
    """Anthropic accepts a string or a list of blocks; the others want a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        return str(content.get("text", ""))
    if isinstance(content, Iterable):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p)
    return str(content)


def schema_instructions(tools: Sequence[Mapping[str, Any]], force_tool: str | None) -> str:
    """Render a tool schema as prompt text, for a backend that cannot force one.

    Deliberately blunt about the output contract. The parser downstream is
    forgiving, but every ambiguity here becomes a dropped decision later.
    """
    chosen = None
    for tool in tools:
        if force_tool is None or tool.get("name") == force_tool:
            chosen = tool
            break
    if chosen is None and tools:
        chosen = tools[0]
    if chosen is None:
        return ""

    schema = chosen.get("input_schema") or chosen.get("parameters") or {}
    return (
        "\n\n---\n"
        "RESPONSE FORMAT — this is not optional.\n"
        f"Reply with a single JSON object matching the schema for the tool "
        f"`{chosen.get('name', 'respond')}`. No prose before it, no prose after "
        "it, no markdown fences, no explanation. If you have nothing to do, "
        "return the object with an empty list rather than a sentence.\n\n"
        f"Tool: {chosen.get('name', 'respond')}\n"
        f"Purpose: {chosen.get('description', '')}\n"
        f"JSON schema:\n{json.dumps(schema, indent=2)}\n"
    )


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort recovery of one JSON object from a model's prose.

    Tries the whole string, then a fenced block, then the outermost balanced
    braces. A model that was told not to wrap its answer still sometimes does,
    and losing a decision to a stray ``` would be a silly way to fail.
    """
    if not text:
        return None
    candidates: list[str] = [text.strip()]
    for match in _FENCE.finditer(text):
        candidates.append(match.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def tool_name_of(tools: Sequence[Mapping[str, Any]], force_tool: str | None) -> str:
    if force_tool:
        return force_tool
    return str(tools[0].get("name", "")) if tools else ""


# --- anthropic ------------------------------------------------------------------


class AnthropicProvider(BaseProvider):
    """The Messages API. Passes requests through untouched -- this is the shape
    everything else is translated into, so there is nothing to translate."""

    name = "anthropic"
    billing = "api"

    def __init__(self, *, model: str | None = None, timeout: float = 120.0,
                 api_key: str | None = None) -> None:
        super().__init__(model=model, timeout=timeout)
        self.api_key = api_key or self.settings.anthropic_api_key
        self.base_url = None
        self._client: Any = None

    def available(self) -> tuple[bool, str]:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "the `anthropic` package is not installed (pip install -e .[agent])"
        if not self.api_key:
            return False, "no ANTHROPIC_API_KEY in the environment or .env"
        return True, ""

    def _ensure(self) -> Any:
        if self._client is None:
            self.require()
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)
        return self._client

    def create(self, **request: Any) -> Any:
        # Returned verbatim: the SDK object already satisfies the reader.
        return self._ensure().messages.create(**request)


# --- claude code (subscription) --------------------------------------------------


class ClaudeCodeProvider(BaseProvider):
    """``claude -p`` as a model backend, billed to a Claude subscription.

    Two things make this different from an API call and both are load-bearing.

    **It is an agentic CLI, not a completions endpoint.** Left at its defaults
    it can read files, run commands and reach the network. Inside a loop that
    decides trades that is unacceptable, so tools are disabled outright unless a
    caller explicitly opts in, and ``--permission-mode`` is pinned. A trading
    decision must be a function of the context bundle it was handed, not of
    whatever the model felt like reading off the disk.

    **It cannot force a tool schema.** The schema is injected into the prompt and
    the JSON is parsed back out. The gate treats the result as untrusted either
    way, which is what makes the weaker guarantee tolerable rather than reckless.

    Cost: the CLI reports ``total_cost_usd``, but on a subscription that is a
    notional API-equivalent figure, not money leaving an account. It is passed
    through with ``billing="subscription"`` so a budget meter can say so instead
    of implying a bill. Note also that each fresh invocation pays for Claude
    Code's own system prompt (tens of thousands of cache-creation tokens), so
    the per-call figure looks large next to a bare API call.
    """

    name = "claude_code"
    billing = "subscription"

    #: Why not 1. The CLI counts an assistant turn even when it is only thinking,
    #: so a single-turn cap makes the *prompt* the failure mode: a long context or
    #: a reply that needs a moment to compose ends as `error_max_turns` with no
    #: result at all. One real session lost 6 of 20 calls that way and the agent
    #: concluded the tooling was broken. Tools are disabled on this invocation, so
    #: extra turns can only buy room to finish -- they cannot do anything.

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: float = 300.0,
        binary: str | None = None,
        allowed_tools: str = "",
        permission_mode: str = "default",
        max_turns: int = 8,
        cwd: str | None = None,
    ) -> None:
        super().__init__(model=model, timeout=timeout)
        self.binary = binary or self.settings.claude_bin
        self.allowed_tools = allowed_tools
        self.permission_mode = permission_mode
        self.max_turns = max_turns
        self.base_url = None
        # Run somewhere inert by default: with tools disabled this is belt and
        # braces, but a CLI that cannot see the repo cannot leak it either.
        self.cwd = cwd or str(self.settings.paths.data)

    def resolve_binary(self) -> str | None:
        return shutil.which(self.binary) or (
            self.binary if os.path.isfile(self.binary) else None
        )

    def available(self) -> tuple[bool, str]:
        if self.resolve_binary() is None:
            return False, (
                f"the `{self.binary}` CLI is not on PATH; install Claude Code "
                f"(npm i -g @anthropic-ai/claude-code) or set LAB_CLAUDE_BIN"
            )
        return True, ""

    def build_argv(self, system_file: str | Path | None = None) -> list[str]:
        """Flags only. Both prompts are delivered out of band, deliberately.

        The user prompt goes in on **stdin**: passing it as an argv element loses
        it, because on Windows ``subprocess`` joins the list with
        ``list2cmdline``, whose quoting rules mangle a payload full of JSON
        double-quotes, and the CLI receives an empty request while still exiting
        0 -- a silent wrong answer, the worst possible failure.

        The system prompt goes in as a **file path**. It used to be inlined via
        ``--append-system-prompt``, on the assumption that the ceiling was the
        32,767-character ``CreateProcess`` limit and nothing realistic would
        approach it. That was wrong twice over: ``claude`` resolves to an npm
        shim (``claude.CMD``), so the command runs through ``cmd.exe``, whose
        limit is **8,191** characters; and the system prompt grows every time
        guidance is added to it. It crossed 8,191 mid-development and every call
        began failing with "The command line is too long" -- a whole session lost
        to a limit nobody was watching. A file has no such ceiling, so the size
        of the prompt and the health of the backend are no longer coupled.
        """
        argv = [
            self.resolve_binary() or self.binary,
            "-p",
            "--output-format",
            "json",
            # An empty allowlist is the point: no file reads, no shell, no fetch.
            "--allowedTools",
            self.allowed_tools,
            "--max-turns",
            str(int(self.max_turns)),
            "--permission-mode",
            self.permission_mode,
        ]
        if self.model:
            argv += ["--model", self.model]
        if system_file:
            argv += ["--append-system-prompt-file", str(system_file)]
        return argv

    def create(self, **request: Any) -> Response:
        self.require()
        model = str(request.get("model") or self.model)
        # flatten, not str(): with caching on, `system` is a list of blocks, and
        # str() would send the model a Python repr of its own prompt.
        system = flatten_content(request.get("system"))
        tools = list(request.get("tools") or [])
        force_tool = _force_tool_name(request.get("tool_choice"))

        user = "\n\n".join(
            flatten_content(m.get("content"))
            for m in request.get("messages") or []
            if m.get("role") == "user"
        )
        prompt = user + schema_instructions(tools, force_tool)

        # Written per call rather than cached on disk: the system prompt is built
        # from live session state on some paths, and a stale file would be a
        # silent wrong answer of exactly the kind this provider already had once.
        handle = None
        if system:
            handle = tempfile.NamedTemporaryFile(
                "w", suffix=".txt", delete=False, encoding="utf-8"
            )
            try:
                handle.write(system)
            finally:
                handle.close()

        argv = self.build_argv(handle.name if handle else None)
        try:
            proc = subprocess.run(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.cwd,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderUnavailable(
                f"`{self.binary} -p` timed out after {self.timeout:.0f}s"
            ) from exc
        except OSError as exc:
            raise ProviderUnavailable(f"could not run `{self.binary}`: {exc}") from exc
        finally:
            if handle is not None:
                Path(handle.name).unlink(missing_ok=True)

        return self._parse(proc.stdout, proc.stderr, proc.returncode, model, tools, force_tool)

    def _parse(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
        model: str,
        tools: Sequence[Mapping[str, Any]],
        force_tool: str | None,
    ) -> Response:
        envelope = extract_json_object(stdout) or {}
        if not envelope:
            detail = (stderr or stdout or "").strip()[:400]
            raise ProviderUnavailable(
                f"`{self.binary} -p` returned no JSON (exit {returncode}): {detail}"
            )
        if envelope.get("is_error") or envelope.get("subtype") not in (None, "success"):
            raise ProviderUnavailable(
                f"claude -p failed: {envelope.get('subtype') or 'error'} "
                f"{envelope.get('api_error_status') or ''} "
                f"{str(envelope.get('result') or '')[:300]}".strip()
            )

        text = str(envelope.get("result") or "")
        usage_raw = envelope.get("usage") or {}
        usage = Usage(
            input_tokens=int(usage_raw.get("input_tokens") or 0),
            output_tokens=int(usage_raw.get("output_tokens") or 0),
            cache_read_input_tokens=int(usage_raw.get("cache_read_input_tokens") or 0),
            cache_creation_input_tokens=int(usage_raw.get("cache_creation_input_tokens") or 0),
        )

        content: list[Block] = []
        payload = extract_json_object(text)
        if payload is not None and tools:
            content.append(
                Block(type="tool_use", name=tool_name_of(tools, force_tool), input=payload)
            )
        if text:
            content.append(Block(type="text", text=text))

        return Response(
            model=_canonical_model(envelope, model),
            content=content,
            stop_reason=str(envelope.get("stop_reason") or "end_turn"),
            usage=usage,
            cost_usd=_maybe_float(envelope.get("total_cost_usd")),
            billing=self.billing,
            raw={
                "session_id": envelope.get("session_id"),
                "num_turns": envelope.get("num_turns"),
                "duration_ms": envelope.get("duration_ms"),
                "permission_denials": envelope.get("permission_denials") or [],
            },
        )


def _canonical_model(envelope: Mapping[str, Any], fallback: str) -> str:
    """Prefer what the CLI says it actually used over what we asked for."""
    usage = envelope.get("modelUsage")
    if isinstance(usage, Mapping) and usage:
        # The heaviest entry is the assistant model; the others are side calls.
        best = max(
            usage.items(),
            key=lambda kv: float((kv[1] or {}).get("outputTokens") or 0)
            if isinstance(kv[1], Mapping)
            else 0.0,
        )
        detail = best[1] if isinstance(best[1], Mapping) else {}
        return str(detail.get("canonicalModel") or best[0] or fallback)
    return fallback


def _maybe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _force_tool_name(tool_choice: Any) -> str | None:
    if isinstance(tool_choice, Mapping):
        return tool_choice.get("name") or None
    return None


# --- openai-compatible (openrouter, local, anything) ----------------------------


class OpenAICompatProvider(BaseProvider):
    """Any ``/chat/completions`` endpoint that speaks OpenAI function calling.

    One class covers OpenRouter, OpenAI, LM Studio, Ollama, vLLM, Together and
    Groq, because the only things that differ are the base URL, the key, and a
    couple of attribution headers.
    """

    name = "openai"
    billing = "api"

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: float = 120.0,
        base_url: str | None = None,
        api_key: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
        transport: Any = None,
    ) -> None:
        super().__init__(model=model, timeout=timeout)
        s = self.settings
        self.base_url = (base_url or s.agent_base_url or _default_base_url(s)).rstrip("/")
        self.api_key = api_key or _default_api_key(s, self.base_url)
        self.extra_headers = dict(extra_headers or {})
        self.transport = transport

    @property
    def is_openrouter(self) -> bool:
        return "openrouter.ai" in (self.base_url or "")

    @property
    def is_local(self) -> bool:
        return any(h in (self.base_url or "") for h in ("localhost", "127.0.0.1", "0.0.0.0"))

    def available(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "no LAB_AGENT_BASE_URL and no default for this provider"
        # A local runtime (Ollama, LM Studio, vLLM) usually needs no key at all,
        # so demanding one would make the most convenient setup look broken.
        if not self.api_key and not self.is_local:
            return False, (
                "no API key: set OPENROUTER_API_KEY, OPENAI_API_KEY or LAB_AGENT_API_KEY"
            )
        return True, ""

    def headers(self) -> dict[str, str]:
        out = {"Content-Type": "application/json"}
        if self.api_key:
            out["Authorization"] = f"Bearer {self.api_key}"
        if self.is_openrouter:
            s = self.settings
            out["HTTP-Referer"] = s.openrouter_referer or "https://github.com/strategy-lab"
            out["X-Title"] = s.openrouter_title or "Strategy Lab"
        out.update(self.extra_headers)
        return out

    def build_payload(self, request: Mapping[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        system = request.get("system")
        if system:
            messages.append({"role": "system", "content": flatten_content(system)})
        for m in request.get("messages") or []:
            messages.append(
                {"role": m.get("role", "user"), "content": flatten_content(m.get("content"))}
            )

        payload: dict[str, Any] = {
            "model": str(request.get("model") or self.model),
            "messages": messages,
            "max_tokens": int(request.get("max_tokens") or 2048),
        }
        if request.get("temperature") is not None:
            payload["temperature"] = request["temperature"]

        tools = list(request.get("tools") or [])
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name"),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema") or t.get("parameters") or {},
                    },
                }
                for t in tools
            ]
            forced = _force_tool_name(request.get("tool_choice"))
            payload["tool_choice"] = (
                {"type": "function", "function": {"name": forced}} if forced else "auto"
            )
        if self.is_openrouter:
            # Ask OpenRouter to bill-report the call so the ledger can record
            # what it actually cost instead of guessing from a price table.
            payload["usage"] = {"include": True}
        return payload

    def create(self, **request: Any) -> Response:
        self.require()
        import httpx

        payload = self.build_payload(request)
        client_kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self.transport is not None:
            client_kwargs["transport"] = self.transport
        with httpx.Client(**client_kwargs) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=self.headers()
            )
        if resp.status_code >= 400:
            raise ProviderUnavailable(
                f"{self.base_url} returned HTTP {resp.status_code}: {resp.text[:300]}"
            )
        try:
            body = resp.json()
        except ValueError as exc:
            raise ProviderUnavailable(
                f"{self.base_url} returned a non-JSON body: {resp.text[:200]}"
            ) from exc
        return self.parse(body, payload["model"])

    def parse(self, body: Mapping[str, Any], fallback_model: str) -> Response:
        if body.get("error"):
            err = body["error"]
            message = err.get("message") if isinstance(err, Mapping) else str(err)
            raise ProviderUnavailable(f"provider error: {message}")

        choices = body.get("choices") or []
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}

        content: list[Block] = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments")
            parsed: dict[str, Any] = {}
            if isinstance(args, Mapping):
                parsed = dict(args)
            elif isinstance(args, str):
                # Arguments arrive as a JSON *string* on this API; a model that
                # emits slightly malformed JSON should cost one decision, not
                # the run.
                parsed = extract_json_object(args) or {}
            content.append(Block(type="tool_use", name=str(fn.get("name") or ""), input=parsed))

        text = message.get("content")
        if isinstance(text, str) and text.strip():
            content.append(Block(type="text", text=text))

        usage_raw = body.get("usage") or {}
        usage = Usage(
            input_tokens=int(usage_raw.get("prompt_tokens") or 0),
            output_tokens=int(usage_raw.get("completion_tokens") or 0),
            cache_read_input_tokens=int(
                (usage_raw.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            ),
        )
        return Response(
            model=str(body.get("model") or fallback_model),
            content=content,
            stop_reason=str(choice.get("finish_reason") or ""),
            usage=usage,
            cost_usd=_maybe_float(usage_raw.get("cost")),
            billing=self.billing,
            raw={"id": body.get("id"), "provider": body.get("provider")},
        )


def _default_base_url(settings: Any) -> str:
    if settings.openrouter_api_key:
        return OPENROUTER_URL
    if settings.openai_api_key:
        return OPENAI_URL
    return OPENROUTER_URL


def _default_api_key(settings: Any, base_url: str) -> str | None:
    """Pick a key for ``base_url`` -- and only a key that belongs to it.

    A vendor key is scoped to its vendor host. Falling back to "whatever key is
    in the environment" for an unrecognized endpoint would quietly POST an
    OpenRouter secret to whatever ``LAB_AGENT_BASE_URL`` happens to point at,
    which is a credential leak wearing the costume of a convenience. Anything
    that is not a known vendor must be given a key explicitly through
    ``LAB_AGENT_API_KEY``, and a local runtime needs none at all.
    """
    if settings.agent_api_key:
        return settings.agent_api_key
    if "openrouter.ai" in base_url:
        return settings.openrouter_api_key
    if "api.openai.com" in base_url:
        return settings.openai_api_key
    return None


# --- registry --------------------------------------------------------------------

REGISTRY: dict[str, type[BaseProvider]] = {
    "anthropic": AnthropicProvider,
    "claude_code": ClaudeCodeProvider,
    "openai": OpenAICompatProvider,
}

#: Aliases people will reasonably type.
ALIASES = {
    "claude": "anthropic",
    "api": "anthropic",
    "cli": "claude_code",
    "claude-code": "claude_code",
    "claudecode": "claude_code",
    "subscription": "claude_code",
    "openrouter": "openai",
    "openai_compatible": "openai",
    "oai": "openai",
    "local": "openai",
    "ollama": "openai",
    "lmstudio": "openai",
}

#: Order ``auto`` tries. API first because it is the only one that can force a
#: tool schema *and* run unattended without a logged-in CLI; the subscription
#: path second because it needs no key; a remote OpenAI-compatible endpoint last
#: because it is the one most likely to be pointed somewhere unintended.
AUTO_ORDER = ("anthropic", "claude_code", "openai")


def normalize(name: str | None) -> str:
    key = (name or "auto").strip().lower().replace("-", "_")
    return ALIASES.get(key, key)


def build(name: str | None = None, **kwargs: Any) -> BaseProvider:
    key = normalize(name)
    if key not in REGISTRY:
        known = ", ".join(sorted(REGISTRY) + sorted(ALIASES))
        raise ValueError(f"unknown agent provider {name!r}; known: {known}")
    return REGISTRY[key](**kwargs)


def resolve(name: str | None = None, **kwargs: Any) -> BaseProvider:
    """Pick a provider, honouring ``auto``.

    ``auto`` returns the first backend that can actually run. When none can, the
    error lists every reason at once -- being told only about the last thing
    tried is how a five-minute setup problem becomes an hour.
    """
    key = normalize(name or get_settings().agent_provider)
    if key != "auto":
        return build(key, **kwargs).require()

    reasons: list[str] = []
    for candidate in AUTO_ORDER:
        provider = build(candidate, **kwargs)
        ok, reason = provider.available()
        if ok:
            if reasons:
                # Never fall through silently: "auto" quietly spending a Claude
                # subscription because an API key was missing is exactly the
                # surprise a billing mode should not spring on anyone.
                _log.info(
                    "agent provider auto-selected %r (billing: %s); skipped %s",
                    provider.name,
                    provider.billing,
                    "; ".join(reasons).strip(),
                )
            return provider
        reasons.append(f"  {candidate}: {reason}")
    raise ProviderUnavailable(
        "no agent provider is usable. Set LAB_AGENT_PROVIDER explicitly, or fix "
        "one of these:\n" + "\n".join(reasons)
    )


def list_providers(**kwargs: Any) -> list[ProviderInfo]:
    out: list[ProviderInfo] = []
    for key in REGISTRY:
        try:
            out.append(build(key, **kwargs).info())
        except Exception as exc:  # one broken provider must not hide the others
            out.append(
                ProviderInfo(
                    name=key,
                    model="",
                    available=False,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
    return out


def probe(name: str | None = None, *, prompt: str = "Reply with the single word: ok") -> dict[str, Any]:
    """Make one real, tiny call. What ``lab agent providers --probe`` runs.

    Worth having as a first-class action: "is my key/CLI/endpoint actually
    wired up" is a different question from "is it configured", and only one of
    them can be answered without spending a token.
    """
    provider = resolve(name)
    started = time.perf_counter()
    try:
        # No `tools` key at all: the Messages API rejects an empty tools array,
        # and a liveness check has nothing to call anyway.
        response = provider.messages.create(
            model=provider.model,
            max_tokens=64,
            system="Answer in as few tokens as possible.",
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        return {
            "provider": provider.name,
            "model": provider.model,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    text = ""
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", "") == "text":
            text = getattr(block, "text", "")
            break
    usage = getattr(response, "usage", None)
    return {
        "provider": provider.name,
        "model": getattr(response, "model", provider.model),
        "ok": True,
        "billing": getattr(response, "billing", provider.billing),
        "text": text.strip()[:200],
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cost_usd": getattr(response, "cost_usd", None),
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }
