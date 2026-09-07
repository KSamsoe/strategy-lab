"""Provider-layer tests. Nothing here touches a network or spends a token.

The three backends exist so that the layer above them -- ``schemas.call_tool``,
the single audited chokepoint -- never learns there is more than one dialect.
That makes *translation* the whole contract, and translation is exactly what a
test can pin down offline: the CLI gets a mocked ``subprocess.run``, the
OpenAI-compatible path gets an ``httpx.MockTransport``, and the Anthropic path
is a pass-through with nothing to translate.

Two properties get disproportionate attention because both are silent when they
break. The ``claude -p`` prompt must travel on **stdin**: as an argv element
Windows' ``list2cmdline`` mangles a JSON-heavy payload and the CLI answers an
empty prompt while exiting 0 -- a confident wrong answer, which already bit once.
And the CLI's tools must stay disabled: a trading decision has to be a function
of the context bundle it was handed, not of whatever the model felt like reading
off the disk.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Callable, Mapping

import httpx
import pathlib

import pytest

from lab.agent import providers as P
from lab.agent.providers import (
    ClaudeCodeProvider,
    OpenAICompatProvider,
    ProviderUnavailable,
    Response,
)
from lab.agent.schemas import TARGETS_TOOL, call_tool, estimate_cost, list_calls
from lab.config import get_settings, reset_settings_cache
from lab.registry import db as registry_db

#: Every environment variable that can move a provider decision. Cleared for
#: each test so a developer's real .env cannot make the suite pass locally and
#: fail in CI (or, worse, quietly aim a test at a live endpoint).
ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "LAB_AGENT_API_KEY",
    "LAB_AGENT_BASE_URL",
    "LAB_AGENT_PROVIDER",
    "LAB_AGENT_MODEL",
    "LAB_CLAUDE_BIN",
    "OPENROUTER_REFERER",
    "OPENROUTER_TITLE",
)

#: A key with a shape a leak-scan can find. It must never reach a stored row.
SECRET = "sk-or-v1-NEVERPERSISTTHIS0000"

USER = "PROMPT-CANARY context_bundle: {\"universe\": [\"AAPL\"], \"equity\": 100000}"


@pytest.fixture(autouse=True)
def clean_env(isolated_paths, monkeypatch):
    """Depends on the autouse path fixture so it runs *after* it and its
    ``reset_settings_cache`` is the last word."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    registry_db.close_all()
    reset_settings_cache()
    yield
    registry_db.close_all()
    reset_settings_cache()


def setenv(monkeypatch, **values: str | None) -> None:
    for key, value in values.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    reset_settings_cache()


# --- registry / selection ------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("openrouter", "openai"),
        ("ollama", "openai"),
        ("lmstudio", "openai"),
        ("local", "openai"),
        ("oai", "openai"),
        ("openai_compatible", "openai"),
        ("openai", "openai"),
        ("cli", "claude_code"),
        ("claude-code", "claude_code"),
        ("claude_code", "claude_code"),
        ("claudecode", "claude_code"),
        ("subscription", "claude_code"),
        ("claude", "anthropic"),
        ("api", "anthropic"),
        ("anthropic", "anthropic"),
    ],
)
def test_normalize_resolves_every_alias(alias, expected):
    assert P.normalize(alias) == expected
    assert P.normalize(alias.upper()) == expected
    assert P.normalize(f"  {alias}  ") == expected


def test_normalize_defaults_to_auto():
    assert P.normalize(None) == "auto"
    assert P.normalize("") == "auto"


def test_every_alias_points_at_a_real_provider():
    assert set(P.ALIASES.values()) <= set(P.REGISTRY)


def test_build_rejects_an_unknown_name():
    with pytest.raises(ValueError) as exc:
        P.build("gpt-please")
    text = str(exc.value)
    assert "gpt-please" in text
    # The error has to be a menu, not just a complaint.
    for known in ("anthropic", "claude_code", "openai", "openrouter"):
        assert known in text


def test_auto_takes_the_first_available_in_auto_order(monkeypatch):
    monkeypatch.setattr(P.AnthropicProvider, "available", lambda self: (True, ""))
    monkeypatch.setattr(P.ClaudeCodeProvider, "available", lambda self: (True, ""))
    assert P.resolve("auto").name == "anthropic"

    monkeypatch.setattr(P.AnthropicProvider, "available", lambda self: (False, "no key"))
    assert P.resolve("auto").name == "claude_code"

    monkeypatch.setattr(P.ClaudeCodeProvider, "available", lambda self: (False, "no CLI"))
    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET)
    chosen = P.resolve("auto")
    assert chosen.name == "openai" and P.AUTO_ORDER == ("anthropic", "claude_code", "openai")


def test_auto_with_nothing_available_names_all_three_with_a_reason_each(monkeypatch):
    """Being told only about the last thing tried is how a five-minute setup
    problem becomes an hour."""
    monkeypatch.setattr(P.AnthropicProvider, "available", lambda self: (False, "no ANTHROPIC_API_KEY"))
    monkeypatch.setattr(P.ClaudeCodeProvider, "available", lambda self: (False, "the claude CLI is not on PATH"))

    with pytest.raises(ProviderUnavailable) as exc:
        P.resolve("auto")

    text = str(exc.value)
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith(("anthropic:", "claude_code:", "openai:"))
    ]
    assert {line.split(":", 1)[0] for line in lines} == {"anthropic", "claude_code", "openai"}
    for line in lines:
        assert len(line.split(":", 1)[1].strip()) > 8, f"reasonless line: {line!r}"


def test_resolve_honours_an_explicit_provider_over_the_auto_order(monkeypatch):
    setenv(monkeypatch, LAB_AGENT_BASE_URL="http://localhost:11434/v1")
    monkeypatch.setattr(P.AnthropicProvider, "available", lambda self: (True, ""))
    assert P.resolve("ollama").name == "openai"


def test_resolve_reads_the_provider_from_settings_when_unnamed(monkeypatch):
    setenv(monkeypatch, LAB_AGENT_PROVIDER="lmstudio", LAB_AGENT_BASE_URL="http://127.0.0.1:1234/v1")
    assert P.resolve().name == "openai"


def test_list_providers_survives_one_provider_exploding(monkeypatch):
    class Exploding(P.BaseProvider):
        name = "anthropic"

        def __init__(self, **_: Any) -> None:
            raise RuntimeError("constructor blew up")

    monkeypatch.setitem(P.REGISTRY, "anthropic", Exploding)
    infos = P.list_providers()

    assert [i.name for i in infos] == list(P.REGISTRY)
    broken = next(i for i in infos if i.name == "anthropic")
    assert broken.available is False and "constructor blew up" in broken.reason
    # The point of the guard: the other two are still reported.
    assert {i.name for i in infos} >= {"claude_code", "openai"}
    assert all(isinstance(i.to_dict(), dict) for i in infos)


def test_only_the_cli_admits_it_cannot_force_a_schema():
    assert P.build("anthropic").forces_tools is True
    assert P.build("openai").forces_tools is True
    assert P.build("claude_code").forces_tools is False
    assert P.build("claude_code").info().billing == "subscription"


# --- claude code ---------------------------------------------------------------


class FakeRun:
    """Stands in for ``subprocess.run``, recording exactly how it was called."""

    def __init__(self, stdout: str = "", *, stderr: str = "", returncode: int = 0,
                 raises: BaseException | None = None) -> None:
        self.stdout, self.stderr, self.returncode, self.raises = stdout, stderr, returncode, raises
        self.argv: list[str] = []
        self.kwargs: dict[str, Any] = {}
        self.calls = 0

    def __call__(self, argv, **kwargs: Any):
        self.calls += 1
        self.argv, self.kwargs = list(argv), dict(kwargs)
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(
            args=argv, returncode=self.returncode, stdout=self.stdout, stderr=self.stderr
        )


def cli_stdout(result: str, **over: Any) -> str:
    """A realistic ``claude -p --output-format json`` envelope."""
    envelope: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 8_123,
        "num_turns": 1,
        "result": result,
        "stop_reason": "end_turn",
        "session_id": "0f2c9a41-7c3e-4b0a-9c11-3d5f9a2b6e77",
        "total_cost_usd": 0.0731,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 340,
            # A fresh invocation re-pays for Claude Code's own system prompt, so
            # cache-creation dwarfs the real input every single time.
            "cache_creation_input_tokens": 21_456,
            "cache_read_input_tokens": 9_012,
        },
        "modelUsage": {
            "claude-sonnet-5-20260201": {
                "inputTokens": 12,
                "outputTokens": 340,
                "costUSD": 0.07,
                "canonicalModel": "claude-sonnet-5",
            },
            "claude-haiku-4-5-20251001": {
                "inputTokens": 900,
                "outputTokens": 7,
                "costUSD": 0.0031,
                "canonicalModel": "claude-haiku-4-5",
            },
        },
        "permission_denials": [],
    }
    envelope.update(over)
    return json.dumps(envelope)


PAYLOAD = {"targets": [{"ticker": "AAPL", "target_pct": 0.1, "rationale": "momentum"}]}


@pytest.fixture()
def cli(monkeypatch) -> Callable[..., tuple[ClaudeCodeProvider, FakeRun]]:
    """A ClaudeCodeProvider with the binary resolved and ``subprocess`` mocked."""
    monkeypatch.setattr(ClaudeCodeProvider, "resolve_binary", lambda self: "/usr/local/bin/claude")

    def make(stdout: str = "", **kw: Any) -> tuple[ClaudeCodeProvider, FakeRun]:
        provider_kw = {k: kw.pop(k) for k in list(kw) if k in {"model", "timeout", "allowed_tools"}}
        run = FakeRun(stdout, **kw)
        monkeypatch.setattr(P.subprocess, "run", run)
        return ClaudeCodeProvider(**provider_kw), run

    return make


def cli_create(provider: ClaudeCodeProvider, *, tools=(TARGETS_TOOL,), system: str = "SYS") -> Response:
    return provider.create(
        model="claude-sonnet-5",
        max_tokens=2_048,
        system=system,
        messages=[{"role": "user", "content": USER}],
        tools=list(tools),
        tool_choice={"type": "tool", "name": "set_targets"},
    )


def test_the_prompt_goes_on_stdin_and_never_into_argv(cli):
    """The regression that already bit: argv delivery silently produced an
    answer to an empty prompt while exiting 0."""
    provider, run = cli(cli_stdout(json.dumps(PAYLOAD)))
    cli_create(provider)

    assert run.calls == 1
    assert run.kwargs["input"].startswith(USER)
    # The schema rides along on stdin too -- this backend cannot be handed one.
    assert "set_targets" in run.kwargs["input"] and "target_pct" in run.kwargs["input"]

    for part in run.argv:
        assert "PROMPT-CANARY" not in part, f"prompt leaked into argv: {part!r}"
        assert "context_bundle" not in part
    assert not any("PROMPT-CANARY" in part for part in provider.build_argv("SYS"))
    # Stdin only works if the child is actually reading text.
    assert run.kwargs["text"] is True and run.kwargs["capture_output"] is True


def test_tools_are_disabled_by_default(cli):
    """A trading decision must be a function of its context bundle, not of
    whatever the CLI felt like reading off the disk."""
    provider, run = cli(cli_stdout(json.dumps(PAYLOAD)))
    cli_create(provider)

    argv = run.argv
    assert "--allowedTools" in argv
    assert argv[argv.index("--allowedTools") + 1] == "", argv
    assert provider.allowed_tools == ""
    assert argv[argv.index("--permission-mode") + 1] == "default"
    # Not 1: the CLI counts a thinking turn, so a single-turn cap makes a long
    # prompt fail as `error_max_turns` with no result. Tools are disabled on this
    # invocation, so extra turns only buy room to finish.
    assert int(argv[argv.index("--max-turns") + 1]) >= 4
    assert argv[:4] == ["/usr/local/bin/claude", "-p", "--output-format", "json"]
    # By path now, not inline: the inline flag put the whole system prompt on the
    # command line, and cmd.exe (claude is an npm .CMD shim) caps that at 8,191.
    sys_file = pathlib.Path(argv[argv.index("--append-system-prompt-file") + 1])
    assert sys_file.name.endswith(".txt")
    # Nowhere useful to read from even if the allowlist ever slipped.
    assert run.kwargs["cwd"] == str(get_settings().paths.data)


def test_the_envelope_becomes_an_anthropic_shaped_response(cli):
    provider, _ = cli(cli_stdout(json.dumps(PAYLOAD)))
    out = cli_create(provider)

    assert isinstance(out, Response)
    tool_blocks = [b for b in out.content if b.type == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0].name == "set_targets"
    assert tool_blocks[0].input == PAYLOAD
    assert [b.type for b in out.content] == ["tool_use", "text"]

    assert out.usage.input_tokens == 12 and out.usage.output_tokens == 340
    assert out.usage.cache_creation_input_tokens == 21_456
    assert out.usage.cache_read_input_tokens == 9_012
    assert out.cost_usd == pytest.approx(0.0731)
    assert out.billing == "subscription"
    assert out.stop_reason == "end_turn"
    assert out.raw["session_id"] == "0f2c9a41-7c3e-4b0a-9c11-3d5f9a2b6e77"


def test_canonical_model_prefers_the_heaviest_model_usage_entry(cli):
    """What the CLI says it used beats what we asked for -- a haiku side call
    must not be mistaken for the assistant."""
    provider, _ = cli(cli_stdout(json.dumps(PAYLOAD)))
    assert cli_create(provider).model == "claude-sonnet-5"

    provider, _ = cli(cli_stdout(json.dumps(PAYLOAD), modelUsage={}))
    assert cli_create(provider).model == "claude-sonnet-5"  # falls back to the request

    heavy = {
        "claude-opus-5-20260101": {"outputTokens": 9_000, "canonicalModel": "claude-opus-5"},
        "claude-sonnet-5-20260201": {"outputTokens": 12, "canonicalModel": "claude-sonnet-5"},
    }
    provider, _ = cli(cli_stdout(json.dumps(PAYLOAD), modelUsage=heavy))
    assert cli_create(provider).model == "claude-opus-5"


def test_a_model_usage_entry_without_a_canonical_name_falls_back_to_its_key(cli):
    provider, _ = cli(
        cli_stdout(json.dumps(PAYLOAD), modelUsage={"some-local-build": {"outputTokens": 5}})
    )
    assert cli_create(provider).model == "some-local-build"


@pytest.mark.parametrize(
    "wrapped",
    [
        "```json\n{payload}\n```",
        "```\n{payload}\n```",
        "Sure -- here is the allocation:\n{payload}\nLet me know if you want changes.",
        "{payload}",
        "  \n{payload}\n  ",
    ],
    ids=["fenced-json", "fenced-bare", "prose-both-sides", "clean", "whitespace"],
)
def test_json_is_recovered_however_the_model_wrapped_it(cli, wrapped):
    """Losing a decision to a stray ``` would be a silly way to fail."""
    provider, _ = cli(cli_stdout(wrapped.format(payload=json.dumps(PAYLOAD))))
    out = cli_create(provider)
    tool_blocks = [b for b in out.content if b.type == "tool_use"]
    assert tool_blocks and tool_blocks[0].input == PAYLOAD


def test_a_reply_with_no_json_yields_no_tool_block_and_does_not_raise(cli):
    provider, _ = cli(cli_stdout("I would rather not answer that."))
    out = cli_create(provider)

    assert [b.type for b in out.content] == ["text"]
    assert out.content[0].text == "I would rather not answer that."
    # call_tool turns the missing block into a recorded failure; the provider's
    # job is to report honestly, not to invent a payload.
    assert not any(b.type == "tool_use" for b in out.content)


def test_a_json_reply_with_no_tools_requested_stays_text(cli):
    provider, _ = cli(cli_stdout(json.dumps(PAYLOAD)))
    out = cli_create(provider, tools=())
    assert [b.type for b in out.content] == ["text"]


def test_is_error_becomes_provider_unavailable(cli):
    provider, _ = cli(
        cli_stdout("Credit balance is too low", is_error=True, subtype="error_during_execution",
                   api_error_status=402)
    )
    with pytest.raises(ProviderUnavailable) as exc:
        cli_create(provider)
    text = str(exc.value)
    assert "error_during_execution" in text and "402" in text and "Credit balance" in text


@pytest.mark.parametrize(
    ("stdout", "stderr", "code"),
    [
        ("", "", 0),
        ("   \n", "", 0),
        ("Error: not logged in. Run `claude login`.", "", 1),
        ("", "command not found: claude", 127),
    ],
    ids=["empty", "whitespace", "non-json-prose", "non-zero-exit"],
)
def test_a_broken_invocation_raises_with_something_actionable(cli, stdout, stderr, code):
    provider, _ = cli(stdout, stderr=stderr, returncode=code)
    with pytest.raises(ProviderUnavailable) as exc:
        cli_create(provider)
    text = str(exc.value)
    assert "returned no JSON" in text and f"exit {code}" in text
    if stderr or stdout.strip():
        assert (stderr or stdout).strip()[:20] in text


def test_a_timeout_raises_rather_than_hanging_the_loop(cli):
    provider, _ = cli(raises=subprocess.TimeoutExpired(cmd="claude", timeout=300.0))
    with pytest.raises(ProviderUnavailable) as exc:
        cli_create(provider)
    assert "timed out" in str(exc.value) and "300" in str(exc.value)


def test_an_oserror_raises_rather_than_escaping_as_itself(cli):
    provider, _ = cli(raises=OSError("Exec format error"))
    with pytest.raises(ProviderUnavailable) as exc:
        cli_create(provider)
    assert "could not run" in str(exc.value) and "Exec format error" in str(exc.value)


def test_available_is_false_with_a_fix_when_the_binary_is_missing(monkeypatch):
    setenv(monkeypatch, LAB_CLAUDE_BIN="claude-that-does-not-exist-xyz")
    provider = ClaudeCodeProvider()
    ok, reason = provider.available()

    assert ok is False
    assert "claude-that-does-not-exist-xyz" in reason
    assert "PATH" in reason and "LAB_CLAUDE_BIN" in reason
    assert provider.info().available is False
    with pytest.raises(ProviderUnavailable) as exc:
        provider.require()
    assert exc.value.args[0].startswith("claude_code:")


# --- openai-compatible ----------------------------------------------------------


class Endpoint:
    """A recording ``httpx.MockTransport``. Nothing leaves the process."""

    def __init__(self, *, body: Mapping[str, Any] | None = None, status: int = 200,
                 text: str = "") -> None:
        self.body, self.status, self.text = body, status, text
        self.requests: list[httpx.Request] = []
        self.payloads: list[dict[str, Any]] = []
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.payloads.append(json.loads(request.content.decode() or "{}"))
        if self.body is not None:
            return httpx.Response(self.status, json=self.body)
        return httpx.Response(self.status, text=self.text)

    @property
    def payload(self) -> dict[str, Any]:
        return self.payloads[-1]

    @property
    def headers(self) -> httpx.Headers:
        return self.requests[-1].headers


def completion(
    *,
    arguments: Any = '{"targets": [{"ticker": "AAPL", "target_pct": 0.1, "rationale": "momentum"}]}',
    name: str = "set_targets",
    content: str | None = None,
    prompt_tokens: int = 1_000,
    completion_tokens: int = 200,
    cached: int = 0,
    cost: float | None = None,
    tool_calls: bool = True,
    model: str = "anthropic/claude-sonnet-5",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}
        ]
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    if cost is not None:
        usage["cost"] = cost
    return {
        "id": "gen-abc123",
        "provider": "Anthropic",
        "model": model,
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
        "usage": usage,
    }


def oai_create(provider: OpenAICompatProvider, *, tools=(TARGETS_TOOL,), force: bool = True) -> Response:
    request: dict[str, Any] = {
        "model": "anthropic/claude-sonnet-5",
        "max_tokens": 2_048,
        "system": "SYSTEM RULES",
        "messages": [{"role": "user", "content": USER}],
        "tools": list(tools),
    }
    if force:
        request["tool_choice"] = {"type": "tool", "name": "set_targets"}
    return provider.create(**request)


def test_request_is_translated_into_openai_function_calling(monkeypatch):
    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET)
    endpoint = Endpoint(body=completion())
    provider = OpenAICompatProvider(transport=endpoint.transport)
    oai_create(provider)

    assert str(endpoint.requests[-1].url) == "https://openrouter.ai/api/v1/chat/completions"
    payload = endpoint.payload
    # `system` is a top-level field on Anthropic and a message here.
    assert payload["messages"][0] == {"role": "system", "content": "SYSTEM RULES"}
    assert payload["messages"][1] == {"role": "user", "content": USER}
    assert payload["max_tokens"] == 2_048

    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "set_targets",
                "description": TARGETS_TOOL["description"],
                "parameters": TARGETS_TOOL["input_schema"],
            },
        }
    ]
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "set_targets"}}


def test_tool_choice_falls_back_to_auto_and_vanishes_without_tools(monkeypatch):
    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET)
    endpoint = Endpoint(body=completion())
    provider = OpenAICompatProvider(transport=endpoint.transport)

    oai_create(provider, force=False)
    assert endpoint.payload["tool_choice"] == "auto"

    oai_create(provider, tools=(), force=False)
    assert "tools" not in endpoint.payload and "tool_choice" not in endpoint.payload


def test_usage_accounting_and_attribution_headers_are_openrouter_only(monkeypatch):
    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET, OPENROUTER_TITLE="Strategy Lab", OPENROUTER_REFERER="https://example.test")
    endpoint = Endpoint(body=completion())
    OpenAICompatProvider(transport=endpoint.transport).create(
        model="m", max_tokens=16, messages=[{"role": "user", "content": "hi"}], tools=[]
    )
    assert endpoint.payload["usage"] == {"include": True}
    assert endpoint.headers["Authorization"] == f"Bearer {SECRET}"
    assert endpoint.headers["HTTP-Referer"] == "https://example.test"
    assert endpoint.headers["X-Title"] == "Strategy Lab"

    local = Endpoint(body=completion())
    OpenAICompatProvider(
        base_url="http://localhost:1234/v1", transport=local.transport
    ).create(model="m", max_tokens=16, messages=[{"role": "user", "content": "hi"}], tools=[])
    # Asking a local runtime to bill-report a call is at best noise.
    assert "usage" not in local.payload
    assert "HTTP-Referer" not in local.headers and "X-Title" not in local.headers


def test_a_keyless_endpoint_sends_no_authorization_header(monkeypatch):
    endpoint = Endpoint(body=completion())
    OpenAICompatProvider(
        base_url="http://localhost:1234/v1", transport=endpoint.transport
    ).create(model="m", max_tokens=16, messages=[{"role": "user", "content": "hi"}], tools=[])
    assert "Authorization" not in endpoint.headers


def test_openrouter_attribution_headers_have_defaults(monkeypatch):
    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET)
    headers = OpenAICompatProvider().headers()
    assert headers["HTTP-Referer"] and headers["X-Title"]
    assert headers["Content-Type"] == "application/json"


def test_extra_headers_win_over_the_defaults(monkeypatch):
    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET)
    headers = OpenAICompatProvider(extra_headers={"X-Title": "custom"}).headers()
    assert headers["X-Title"] == "custom"


def test_response_is_translated_back_into_anthropic_blocks(monkeypatch):
    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET)
    endpoint = Endpoint(body=completion(content="Rebalancing into AAPL.", cached=640))
    out = oai_create(OpenAICompatProvider(transport=endpoint.transport))

    assert [b.type for b in out.content] == ["tool_use", "text"]
    tool = out.content[0]
    assert tool.name == "set_targets"
    # `arguments` is a JSON *string* on this API; it must arrive parsed.
    assert tool.input == {"targets": [{"ticker": "AAPL", "target_pct": 0.1, "rationale": "momentum"}]}
    assert out.content[1].text == "Rebalancing into AAPL."

    assert out.usage.input_tokens == 1_000 and out.usage.output_tokens == 200
    assert out.usage.cache_read_input_tokens == 640
    assert out.model == "anthropic/claude-sonnet-5"
    assert out.stop_reason == "tool_calls"
    assert out.billing == "api"
    assert out.cost_usd is None  # no reported cost -> the ledger estimates
    assert out.raw == {"id": "gen-abc123", "provider": "Anthropic"}


@pytest.mark.parametrize(
    "arguments",
    ['{"targets": [oops', "", "null", "not json at all", '["not", "an", "object"]'],
    ids=["truncated", "empty", "null", "prose", "array"],
)
def test_malformed_tool_arguments_degrade_to_an_empty_object(monkeypatch, arguments):
    """A model that emits slightly bent JSON should cost one decision, not the
    run -- and the validator downstream refuses an empty payload anyway."""
    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET)
    endpoint = Endpoint(body=completion(arguments=arguments))
    out = oai_create(OpenAICompatProvider(transport=endpoint.transport))
    assert out.content[0].type == "tool_use" and out.content[0].input == {}


def test_arguments_already_decoded_are_accepted(monkeypatch):
    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET)
    endpoint = Endpoint(body=completion(arguments={"targets": []}))
    out = oai_create(OpenAICompatProvider(transport=endpoint.transport))
    assert out.content[0].input == {"targets": []}


def test_blank_content_does_not_become_an_empty_text_block(monkeypatch):
    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET)
    endpoint = Endpoint(body=completion(content="   "))
    out = oai_create(OpenAICompatProvider(transport=endpoint.transport))
    assert [b.type for b in out.content] == ["tool_use"]


@pytest.mark.parametrize("status", [400, 401, 402, 429, 500, 503])
def test_an_http_error_raises_provider_unavailable(monkeypatch, status):
    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET)
    endpoint = Endpoint(status=status, text="upstream said no")
    with pytest.raises(ProviderUnavailable) as exc:
        oai_create(OpenAICompatProvider(transport=endpoint.transport))
    text = str(exc.value)
    assert f"HTTP {status}" in text and "upstream said no" in text


def test_a_200_with_an_error_object_also_raises(monkeypatch):
    """OpenRouter reports upstream failures inside a 200 body; a provider that
    only checks the status code would hand back an empty decision instead."""
    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET)
    endpoint = Endpoint(body={"error": {"code": 429, "message": "rate limited upstream"}})
    with pytest.raises(ProviderUnavailable) as exc:
        oai_create(OpenAICompatProvider(transport=endpoint.transport))
    assert "rate limited upstream" in str(exc.value)


def test_a_non_json_body_raises(monkeypatch):
    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET)
    endpoint = Endpoint(status=200, text="<html>proxy login page</html>")
    with pytest.raises(ProviderUnavailable) as exc:
        oai_create(OpenAICompatProvider(transport=endpoint.transport))
    assert "non-JSON" in str(exc.value) and "proxy login" in str(exc.value)


@pytest.mark.parametrize(
    ("base_url", "needs_key"),
    [
        ("http://localhost:11434/v1", False),
        ("http://127.0.0.1:1234/v1", False),
        ("http://0.0.0.0:8000/v1", False),
        ("https://api.together.xyz/v1", True),
        ("https://openrouter.ai/api/v1", True),
    ],
)
def test_only_a_remote_endpoint_demands_a_key(monkeypatch, base_url, needs_key):
    """Demanding a key from Ollama would make the most convenient setup look
    broken."""
    provider = OpenAICompatProvider(base_url=base_url)
    ok, reason = provider.available()
    assert ok is not needs_key
    if needs_key:
        assert "OPENROUTER_API_KEY" in reason and "OPENAI_API_KEY" in reason
    setenv(monkeypatch, LAB_AGENT_API_KEY=SECRET)
    assert OpenAICompatProvider(base_url=base_url).available()[0] is True


def test_base_url_and_key_default_from_whichever_key_is_present(monkeypatch):
    setenv(monkeypatch, OPENROUTER_API_KEY="or-key")
    provider = OpenAICompatProvider()
    assert provider.base_url == P.OPENROUTER_URL and provider.api_key == "or-key"
    assert provider.is_openrouter and not provider.is_local

    setenv(monkeypatch, OPENROUTER_API_KEY=None, OPENAI_API_KEY="oa-key")
    provider = OpenAICompatProvider()
    assert provider.base_url == P.OPENAI_URL and provider.api_key == "oa-key"
    assert not provider.is_openrouter

    # An explicit base URL wins over both, and the generic key travels with it.
    setenv(monkeypatch, OPENROUTER_API_KEY="or-key", OPENAI_API_KEY="oa-key",
           LAB_AGENT_BASE_URL="http://localhost:1234/v1", LAB_AGENT_API_KEY="lab-key")
    provider = OpenAICompatProvider()
    assert provider.base_url == "http://localhost:1234/v1" and provider.api_key == "lab-key"
    assert provider.is_local

    # Trailing slashes must not become a doubled path segment.
    setenv(monkeypatch, LAB_AGENT_BASE_URL="https://api.together.xyz/v1/")
    assert OpenAICompatProvider().base_url == "https://api.together.xyz/v1"

    # With no key at all we still point somewhere sane rather than nowhere.
    setenv(monkeypatch, OPENROUTER_API_KEY=None, OPENAI_API_KEY=None,
           LAB_AGENT_API_KEY=None, LAB_AGENT_BASE_URL=None)
    assert OpenAICompatProvider().base_url == P.OPENROUTER_URL


def test_a_vendor_key_is_never_sent_to_an_unrecognized_endpoint(monkeypatch):
    """A vendor key is scoped to its vendor host.

    Falling back to "whatever key is in the environment" would quietly POST an
    OpenRouter secret to whatever LAB_AGENT_BASE_URL points at -- a credential
    leak wearing the costume of a convenience. An unknown host must be given a
    key explicitly, and a local runtime needs none.
    """
    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET, LAB_AGENT_BASE_URL="http://localhost:1234/v1")
    provider = OpenAICompatProvider()
    assert provider.api_key is None
    ok, _ = provider.available()
    assert ok, "a local runtime is usable with no key at all"

    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET, LAB_AGENT_BASE_URL="https://evil.example/v1")
    stranger = OpenAICompatProvider()
    assert stranger.api_key is None
    ok, reason = stranger.available()
    assert not ok and "LAB_AGENT_API_KEY" in reason


def test_an_explicit_key_still_reaches_a_custom_endpoint(monkeypatch):
    setenv(
        monkeypatch,
        LAB_AGENT_API_KEY="sk-explicit",
        LAB_AGENT_BASE_URL="http://localhost:8000/v1",
    )
    provider = OpenAICompatProvider()
    assert provider.api_key == "sk-explicit"
    assert provider.headers()["Authorization"] == "Bearer sk-explicit"


def test_an_explicit_constructor_argument_beats_the_environment(monkeypatch):
    setenv(monkeypatch, LAB_AGENT_BASE_URL="https://openrouter.ai/api/v1", OPENROUTER_API_KEY="or-key")
    provider = OpenAICompatProvider(base_url="http://localhost:9/v1", api_key="explicit")
    assert provider.base_url == "http://localhost:9/v1" and provider.api_key == "explicit"


# --- through the real chokepoint -------------------------------------------------


def test_call_tool_through_the_cli_produces_a_priced_audited_row(cli):
    provider, run = cli(cli_stdout(json.dumps(PAYLOAD)))
    out = call_tool(
        provider,
        model="claude-sonnet-5",
        system="SYSTEM RULES",
        user=USER,
        tools=[TARGETS_TOOL],
        force_tool="set_targets",
        run_id="r_cli",
        strategy="agent_daily",
    )

    assert out.ok is True and out.error == ""
    assert out.tool == "set_targets" and out.input == PAYLOAD
    assert out.model == "claude-sonnet-5"
    assert out.input_tokens == 12 and out.output_tokens == 340
    # The CLI's own tally beats the price table -- and says the dollars are notional.
    assert out.cost_usd == pytest.approx(0.0731)
    assert out.billing == "subscription"
    assert run.kwargs["input"].startswith(USER)

    rows = list_calls(run_id="r_cli")
    assert len(rows) == 1
    row = rows[0]
    assert row["ok"] is True and row["tool"] == "set_targets"
    assert row["model"] == "claude-sonnet-5" and row["strategy"] == "agent_daily"
    assert row["cost_usd"] == pytest.approx(0.0731)
    assert "AAPL" in row["response"] and "PROMPT-CANARY" in row["prompt"]


def test_call_tool_through_openai_prefers_the_reported_cost(monkeypatch):
    """OpenRouter bills per request; a price table keyed on a model id it has
    never heard of would over-charge the budget meter by an order of magnitude."""
    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET)
    endpoint = Endpoint(body=completion(cost=0.0123, cached=640))
    provider = OpenAICompatProvider(transport=endpoint.transport)

    out = call_tool(
        provider,
        model="anthropic/claude-sonnet-5",
        system="SYSTEM RULES",
        user=USER,
        tools=[TARGETS_TOOL],
        force_tool="set_targets",
        run_id="r_oai",
        strategy="agent_daily",
    )

    assert out.ok is True and out.tool == "set_targets"
    assert out.input == PAYLOAD
    assert out.input_tokens == 1_000 and out.output_tokens == 200
    assert out.billing == "api"
    estimate = estimate_cost("anthropic/claude-sonnet-5", 1_000, 200, cache_read_tokens=640)
    assert out.cost_usd == pytest.approx(0.0123)
    assert out.cost_usd != pytest.approx(estimate), "the reported figure must win"

    rows = list_calls(run_id="r_oai")
    assert len(rows) == 1 and rows[0]["cost_usd"] == pytest.approx(0.0123)


def test_a_provider_failure_is_one_recorded_row_and_no_exception(monkeypatch):
    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET)
    endpoint = Endpoint(status=503, text="upstream unavailable")
    out = call_tool(
        OpenAICompatProvider(transport=endpoint.transport),
        model="anthropic/claude-sonnet-5",
        system="SYSTEM RULES",
        user=USER,
        tools=[TARGETS_TOOL],
        force_tool="set_targets",
        run_id="r_fail",
    )

    assert out.ok is False and "ProviderUnavailable" in out.error
    rows = list_calls(run_id="r_fail")
    assert len(rows) == 1 and rows[0]["ok"] is False


def test_the_api_key_never_reaches_a_persisted_row(monkeypatch, cli):
    """The ledger is the replay debugger; it gets read, exported and pasted into
    issues. A key in it is a key in a screenshot."""
    setenv(monkeypatch, OPENROUTER_API_KEY=SECRET, OPENAI_API_KEY=SECRET, LAB_AGENT_API_KEY=SECRET)
    endpoint = Endpoint(body=completion(cost=0.0123))
    for provider, run_id in (
        (OpenAICompatProvider(transport=endpoint.transport), "r_leak_oai"),
        (cli(cli_stdout(json.dumps(PAYLOAD)))[0], "r_leak_cli"),
    ):
        call_tool(
            provider,
            model="anthropic/claude-sonnet-5",
            system="SYSTEM RULES",
            user=USER,
            tools=[TARGETS_TOOL],
            force_tool="set_targets",
            run_id=run_id,
        )

    rows = list_calls(run_id="r_leak_oai") + list_calls(run_id="r_leak_cli")
    assert len(rows) == 2
    for row in rows:
        blob = json.dumps({k: str(v) for k, v in row.items()})
        assert SECRET not in blob, f"key leaked into the ledger: {row['id']}"
        assert "sk-or-v1" not in blob


def test_extra_turns_cannot_do_anything_because_tools_are_off(monkeypatch, tmp_path):
    """Raising max_turns is safe only while the invocation is tool-less.

    The turn budget exists so a reply has room to finish, not so the CLI can go
    and do things. If tools were ever enabled by default, this cap would stop
    being a latency knob and start being an authority grant.
    """
    from lab.agent.providers import ClaudeCodeProvider

    provider = ClaudeCodeProvider(binary="claude")
    monkeypatch.setattr(provider, "resolve_binary", lambda: "claude")
    argv = provider.build_argv("system")

    assert provider.allowed_tools == "", "the default must stay empty"
    assert argv[argv.index("--allowedTools") + 1] == ""
    assert int(argv[argv.index("--max-turns") + 1]) >= 4


# --- the command-line ceiling -------------------------------------------------
#
# `claude` resolves to an npm shim (claude.CMD) on Windows, so the command runs
# through cmd.exe, whose limit is 8,191 characters -- not the 32,767 that
# CreateProcess allows and that this provider originally assumed. The system
# prompt grew past 8,191 during development and every call started failing with
# "The command line is too long". A file path has no such ceiling.


def test_the_system_prompt_never_reaches_the_command_line(monkeypatch):
    """The size of the prompt must not be coupled to whether the backend works.

    Asserted through ``create`` rather than ``build_argv``: build_argv now takes a
    path and would pass this trivially. The guarantee that matters is that the
    provider never puts prompt text on the command line at all.
    """
    import json as _json
    import subprocess
    from types import SimpleNamespace

    from lab.agent.providers import ClaudeCodeProvider

    seen: dict = {}

    def fake_run(argv, **kw):
        seen["line"] = subprocess.list2cmdline(argv)
        seen["argv"] = list(argv)
        return SimpleNamespace(
            stdout=_json.dumps({
                "subtype": "success", "is_error": False, "result": "{}",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }),
            stderr="", returncode=0,
        )

    monkeypatch.setattr("lab.agent.providers.subprocess.run", fake_run)
    provider = ClaudeCodeProvider(binary="claude")
    monkeypatch.setattr(provider, "resolve_binary", lambda: "claude")
    monkeypatch.setattr(provider, "require", lambda: None)

    provider.create(
        model="m",
        system="GUIDANCE LINE\n" * 5_000,  # ~70k chars, past every limit
        messages=[{"role": "user", "content": "CANARY-USER-PROMPT"}],
        tools=[],
    )

    # cmd.exe caps at 8,191 -- claude resolves to an npm .CMD shim, so that, not
    # CreateProcess's 32,767, is the ceiling that actually applies.
    assert len(seen["line"]) < 2_000, (
        f"command line is {len(seen['line'])} chars; it must not scale with the prompt"
    )
    assert "GUIDANCE LINE" not in seen["line"], "prompt text leaked into argv"
    assert "CANARY-USER-PROMPT" not in seen["line"]
    assert "--append-system-prompt-file" in seen["argv"]
    assert "--append-system-prompt" not in seen["argv"], (
        "the inline flag is what created the ceiling"
    )


def test_a_huge_system_prompt_still_runs(monkeypatch, tmp_path):
    """End to end: the provider writes a file, passes the path, and cleans up."""
    import json as _json
    from pathlib import Path
    from types import SimpleNamespace

    from lab.agent.providers import ClaudeCodeProvider

    seen: dict = {}

    def fake_run(argv, **kw):
        seen["argv"] = list(argv)
        seen["stdin"] = kw.get("input") or ""
        idx = argv.index("--append-system-prompt-file")
        path = Path(argv[idx + 1])
        seen["system_on_disk"] = path.read_text(encoding="utf-8")
        seen["path"] = path
        return SimpleNamespace(
            stdout=_json.dumps({
                "subtype": "success", "is_error": False, "result": "{}",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }),
            stderr="", returncode=0,
        )

    monkeypatch.setattr("lab.agent.providers.subprocess.run", fake_run)
    provider = ClaudeCodeProvider(binary="claude")
    monkeypatch.setattr(provider, "resolve_binary", lambda: "claude")
    monkeypatch.setattr(provider, "require", lambda: None)

    system = "RULE\n" * 4_000
    provider.create(
        model="m", system=system,
        messages=[{"role": "user", "content": "do the thing"}], tools=[],
    )

    assert seen["system_on_disk"] == system, "the file must carry the prompt verbatim"
    assert "do the thing" in seen["stdin"], "the user prompt still rides stdin"
    assert not seen["path"].exists(), "the temp file must not be left behind"
