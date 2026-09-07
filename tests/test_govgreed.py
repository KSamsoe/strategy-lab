"""Offline coverage for the GovGreed client and adapter.

Everything here runs through ``httpx.MockTransport``: no network, no key, no
sleeping. The vendor numbers used below (137/42, tier "founders") are
deliberately not any documented quota, because the point of several of these
tests is that the code reports what the wire says rather than what a doc page
claimed last month.
"""

from __future__ import annotations

import gzip
import logging
from datetime import timedelta

import httpx
import pytest

from lab import config
from lab.adapters import govgreed as gg
from lab.adapters.base import AdapterUnavailable, QuotaExceeded
from lab.timeutil import to_utc, utcnow

API_KEY = "gg_live_5Ecr3t_key_never_log_me"
BASE_URL = "https://api.test.invalid/api/v1"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LAB_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("GOVGREED_API_KEY", API_KEY)
    monkeypatch.setenv("GOVGREED_BASE_URL", BASE_URL)
    config.reset_settings_cache()
    yield
    config.reset_settings_cache()


class Sleeps(list):
    """Stand-in for time.sleep that records instead of waiting."""

    def __call__(self, seconds: float) -> None:
        self.append(round(float(seconds), 6))


def envelope(data, *, request_id="req-001", meta=None):
    return {"data": data, "meta": {"request_id": request_id, **(meta or {})}}


def problem(status, code, *, title="problem", detail="", request_id="req-err"):
    return {"type": f"https://www.govgreed.com/errors/{code.lower()}", "title": title,
            "status": status, "detail": detail, "code": code, "request_id": request_id}


def make_client(handler, **kwargs):
    sleeps = Sleeps()
    client = gg.GovGreedClient(
        api_key=API_KEY,
        base_url=BASE_URL,
        session=httpx.MockTransport(handler),
        min_interval=0.0,
        sleep=sleeps,
        **kwargs,
    )
    return client, sleeps


def make_adapter(handler, **kwargs):
    sleeps = Sleeps()
    adapter = gg.GovGreedAdapter(
        api_key=API_KEY,
        base_url=BASE_URL,
        session=httpx.MockTransport(handler),
        min_interval=0.0,
        sleep=sleeps,
        **kwargs,
    )
    return adapter, sleeps


def snapshots():
    root = config.get_settings().paths.raw / "govgreed"
    return sorted(root.rglob("*.jsonl.gz")) if root.exists() else []


def snapshot_text():
    return "".join(gzip.open(p, "rt", encoding="utf-8").read() for p in snapshots())


# --- envelope, auth header, request_id ---------------------------------------


def test_envelope_unwrapped_and_request_id_captured():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["path"] = request.url.path
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json=envelope([{"ticker": "NVDA"}], request_id="req-abc"))

    client, _ = make_client(handler)
    data, meta = client.get("/signals/top", tier="A", fresh=True, limit=25)

    assert data == [{"ticker": "NVDA"}]
    assert meta["request_id"] == "req-abc"
    assert client.last_request_id == "req-abc"
    assert seen["auth"] == f"Bearer {API_KEY}"
    assert seen["path"] == "/api/v1/signals/top"
    assert seen["query"] == {"tier": "A", "fresh": "true", "limit": "25"}
    assert client.calls_made == 1


def test_request_id_falls_back_to_header_when_meta_omits_it():
    def handler(request):
        return httpx.Response(200, json={"data": [], "meta": {}},
                              headers={"X-Request-Id": "hdr-77"})

    client, _ = make_client(handler)
    _, meta = client.get("/status")
    assert client.last_request_id == "hdr-77"
    assert meta == {}


def test_missing_envelope_is_tolerated_as_bare_data():
    def handler(request):
        return httpx.Response(200, json=[{"ticker": "AAPL"}])  # schema drift

    client, _ = make_client(handler)
    data, meta = client.get("/signals/top")
    assert data == [{"ticker": "AAPL"}]
    assert meta == {}


# --- quota, from both sources, none of it hardcoded --------------------------


def test_quota_read_from_rate_limit_headers_only():
    reset = int(utcnow().timestamp()) + 3600

    def handler(request):
        return httpx.Response(200, json={"data": []}, headers={
            "X-RateLimit-Limit": "137",
            "X-RateLimit-Remaining": "42",
            "X-RateLimit-Tier": "founders",
            "X-RateLimit-Reset": str(reset),
        })

    client, _ = make_client(handler)
    client.get("/status")
    quota = client.quota
    assert (quota.limit, quota.remaining) == (137, 42)
    assert quota.used == 95  # derived arithmetically, not assumed
    assert quota.tier == "founders"
    assert quota.reset_at == to_utc(reset)


def test_quota_read_from_meta_envelope_only():
    def handler(request):
        return httpx.Response(200, json=envelope([], meta={
            "quota": {"used": 8, "limit": 251, "tier": "free_week_one",
                      "resets_at": "2026-08-22T00:00:00Z"}
        }))

    client, _ = make_client(handler)
    client.get("/status")
    quota = client.quota
    assert (quota.used, quota.limit, quota.remaining) == (8, 251, 243)
    assert quota.tier == "free_week_one"
    assert quota.reset_at == to_utc("2026-08-22T00:00:00Z")


def test_quota_prefers_envelope_over_headers_and_is_never_defaulted():
    def handler(request):
        return httpx.Response(200, json=envelope([], meta={"quota": {"limit": 999}}),
                              headers={"X-RateLimit-Limit": "137"})

    client, _ = make_client(handler)
    client.get("/status")
    assert client.quota.limit == 999

    def silent(request):
        return httpx.Response(200, json=envelope([]))

    bare, _ = make_client(silent)
    bare.get("/status")
    q = bare.quota
    assert (q.used, q.limit, q.remaining, q.tier) == (None, None, None, None)


# --- error paths --------------------------------------------------------------


def test_daily_quota_error_aborts_without_burning_retries():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(429, json=problem(429, "DAILY_QUOTA_EXCEEDED",
                                                title="Daily quota exceeded",
                                                request_id="req-quota"),
                              headers={"X-RateLimit-Limit": "20", "X-RateLimit-Remaining": "0"})

    client, sleeps = make_client(handler)
    with pytest.raises(gg.GovGreedQuotaError) as excinfo:
        client.signals_top()

    err = excinfo.value
    assert isinstance(err, QuotaExceeded)  # generic callers must catch it
    assert err.request_id == "req-quota"
    assert err.quota.remaining == 0
    assert len(calls) == 1  # no retries: the buffer is not spent on a lost cause
    assert sleeps == []
    assert API_KEY not in str(err)


def test_burst_limit_honours_retry_after_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json=problem(429, "BURST_LIMIT_EXCEEDED"),
                                  headers={"Retry-After": "2"})
        return httpx.Response(200, json=envelope([{"ticker": "MSFT"}]))

    client, sleeps = make_client(handler)
    assert client.signals_top() == [{"ticker": "MSFT"}]
    assert len(calls) == 2
    assert sleeps == [2.0]


def test_burst_limit_without_retry_after_uses_one_second_default():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json=problem(429, "BURST_LIMIT_EXCEEDED"))
        return httpx.Response(200, json=envelope([]))

    client, sleeps = make_client(handler)
    client.signals_top()
    assert sleeps == [1.0]


@pytest.mark.parametrize("status", [401, 403])
def test_auth_error_is_immediate_and_never_retried(status):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(status, json=problem(status, "INVALID_API_KEY",
                                                   detail="key revoked",
                                                   request_id="req-auth"))

    client, sleeps = make_client(handler)
    with pytest.raises(gg.GovGreedAuthError) as excinfo:
        client.me()

    assert len(calls) == 1
    assert sleeps == []
    assert excinfo.value.status == status
    assert excinfo.value.request_id == "req-auth"
    assert API_KEY not in str(excinfo.value)


def test_5xx_backs_off_exponentially_and_gives_up_after_three_attempts():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, json=problem(503, "INTERNAL_ERROR",
                                                title="upstream unavailable",
                                                request_id="req-5xx"))

    client, sleeps = make_client(handler)
    with pytest.raises(gg.GovGreedError) as excinfo:
        client.herd_signals()

    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]
    assert "req-5xx" in str(excinfo.value)  # diagnosable without the logs
    assert not isinstance(excinfo.value, gg.GovGreedQuotaError)


def test_5xx_that_recovers_within_the_budget_returns_data():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(500, json=problem(500, "INTERNAL_ERROR"))
        return httpx.Response(200, json=envelope([{"ticker": "TSLA"}]))

    client, sleeps = make_client(handler)
    assert client.signals_top() == [{"ticker": "TSLA"}]
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]


def test_burst_that_never_clears_raises_after_the_retry_budget():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json=problem(429, "BURST_LIMIT_EXCEEDED"),
                              headers={"Retry-After": "1"})

    client, sleeps = make_client(handler)
    with pytest.raises(gg.GovGreedBurstError) as excinfo:
        client.signals_top()
    assert len(calls) == 3
    assert sleeps == [1.0, 1.0]
    assert excinfo.value.retry_after == 1.0
    assert not isinstance(excinfo.value, gg.GovGreedQuotaError)


def test_unlabelled_429_with_a_long_retry_after_is_treated_as_the_daily_wall():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, json={"title": "Too Many Requests"},
                              headers={"Retry-After": "3600"})

    client, _ = make_client(handler)
    with pytest.raises(gg.GovGreedQuotaError):
        client.signals_top()
    assert len(calls) == 1


def test_transport_failure_retries_then_reports_without_a_snapshot():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("dns is having a day", request=request)

    client, sleeps = make_client(handler)
    with pytest.raises(gg.GovGreedError) as excinfo:
        client.signals_top()
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]
    assert excinfo.value.code == "TRANSPORT_ERROR"
    assert snapshots() == []  # nothing arrived, so there is nothing to persist


def test_client_self_throttles_between_calls():
    def handler(request):
        return httpx.Response(200, json=envelope([]))

    sleeps = Sleeps()
    client = gg.GovGreedClient(api_key=API_KEY, base_url=BASE_URL,
                               session=httpx.MockTransport(handler),
                               min_interval=0.5, sleep=sleeps)
    client.status()
    client.status()
    client.status()
    # First call is free; the next two wait out the ~2 req/sec self-limit.
    assert len(sleeps) == 2
    assert all(0 < s <= 0.5 for s in sleeps)


def test_client_4xx_other_than_auth_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404, json=problem(404, "NOT_FOUND"))

    client, _ = make_client(handler)
    with pytest.raises(gg.GovGreedError):
        client.insider_signal("nvda")
    assert len(calls) == 1


# --- defensive parsing --------------------------------------------------------


def test_unknown_fields_ride_along_and_bad_records_are_skipped(caplog):
    payload = [
        {"ticker": "nvda", "trade_date": "2026-06-01", "tier": "A+", "score": 91.5,
         "direction": "BULLISH", "sector": "Technology", "fresh": True,
         "kalshi_overlay": {"unknown": "field"}, "brand_new_column": 7},   # keep
        {"trade_date": "2026-06-02", "tier": "A"},                          # no ticker
        {"ticker": "MSFT", "tier": "A"},                                    # no date
    ]

    def handler(request):
        return httpx.Response(200, json=envelope(payload))

    adapter, _ = make_adapter(handler)
    with caplog.at_level(logging.WARNING, logger="lab.adapters.govgreed"):
        events = list(adapter.fetch_signals(kinds="signal"))

    assert len(events) == 1
    ev = events[0]
    assert ev.ticker == "NVDA" and ev.kind == "signal" and ev.source == "govgreed"
    assert ev.tier == "A+" and ev.score == 91.5 and ev.direction == "BUY"
    assert ev.sector == "Technology" and ev.fresh is True
    assert ev.payload["kalshi_overlay"] == {"unknown": "field"}
    assert ev.payload["brand_new_column"] == 7
    assert ev.as_row()["ticker"] == "NVDA"  # survives the store's own validation

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert any("no ticker" in w for w in warnings)
    assert any("no usable date" in w for w in warnings)


def test_non_dict_records_and_garbage_payloads_do_not_crash():
    def handler(request):
        return httpx.Response(200, json=envelope(["not-a-record", None, 42]))

    adapter, _ = make_adapter(handler)
    assert list(adapter.fetch_signals(kinds="signal")) == []


# --- the two-timestamp mapping ------------------------------------------------


def test_forty_five_day_disclosure_lag_keeps_the_two_times_apart():
    def handler(request):
        return httpx.Response(200, json=envelope([{
            "ticker": "LMT",
            "trade_date": "2026-06-01",
            "disclosure_date": "2026-07-16",   # 45 days later, per the STOCK Act
            "politician": "Rep. Example",
            "tier": "A",
        }]))

    adapter, _ = make_adapter(handler)
    before = utcnow()
    ev = list(adapter.fetch_signals(kinds="signal"))[0]

    assert ev.event_time == to_utc("2026-06-01")
    assert (to_utc("2026-07-16") - ev.event_time).days == 45
    assert ev.knowledge_time >= ev.event_time
    assert ev.knowledge_time >= before          # we knew it when we fetched it
    assert (ev.knowledge_time - ev.event_time).days >= 45
    assert ev.as_row()["knowledge_time"] >= ev.as_row()["event_time"]


def test_disclosure_later_than_our_fetch_wins_as_knowledge_time():
    future = utcnow() + timedelta(days=2)

    def handler(request):
        return httpx.Response(200, json=envelope([{
            "ticker": "BA", "transaction_date": "2026-05-01",
            "disclosed_at": future.isoformat(),
        }]))

    adapter, _ = make_adapter(handler)
    ev = list(adapter.fetch_signals(kinds="signal"))[0]
    assert ev.knowledge_time == to_utc(future)


def test_knowledge_time_is_never_before_event_time_even_on_a_future_dated_trade():
    ahead = utcnow() + timedelta(days=10)

    def handler(request):
        return httpx.Response(200, json=envelope([{"ticker": "GE",
                                                   "trade_date": ahead.isoformat()}]))

    adapter, _ = make_adapter(handler)
    ev = list(adapter.fetch_signals(kinds="signal"))[0]
    assert ev.knowledge_time == ev.event_time
    assert ev.as_row()  # Event.as_row() rejects kt < et; the clamp is what saves it


# --- uid stability ------------------------------------------------------------


def test_uid_is_stable_across_identical_pulls_and_tolerates_drift():
    call = {"n": 0}

    def handler(request):
        call["n"] += 1
        record = {"ticker": "NVDA", "trade_date": "2026-06-01",
                  "politician": "Rep. Example", "score": 90.0}
        if call["n"] > 1:  # second pull: score moved, vendor added a column
            record = {**record, "score": 93.5, "new_beta_field": "whatever"}
        return httpx.Response(200, json=envelope([record]))

    adapter, _ = make_adapter(handler)
    first = list(adapter.fetch_signals(kinds="signal"))
    second = list(adapter.fetch_signals(kinds="signal"))

    assert first[0].uid == second[0].uid
    assert first[0].uid.startswith("signal:NVDA:2026-06-01:")
    assert first[0].score != second[0].score  # the restatement, same identity


def test_distinct_politicians_on_the_same_day_get_distinct_uids():
    def handler(request):
        return httpx.Response(200, json=envelope([
            {"ticker": "NVDA", "trade_date": "2026-06-01", "politician": "A"},
            {"ticker": "NVDA", "trade_date": "2026-06-01", "politician": "B"},
        ]))

    adapter, _ = make_adapter(handler)
    events = list(adapter.fetch_signals(kinds="signal"))
    assert len({e.uid for e in events}) == 2


# --- raw persistence ----------------------------------------------------------


def test_raw_snapshot_is_written_before_parsing_even_when_parsing_explodes(monkeypatch):
    def handler(request):
        return httpx.Response(200, json=envelope([{"ticker": "NVDA",
                                                   "trade_date": "2026-06-01"}]))

    adapter, _ = make_adapter(handler)

    def boom(*args, **kwargs):
        raise RuntimeError("normalizer hit a renamed field")

    monkeypatch.setattr(gg, "normalize", boom)
    with pytest.raises(RuntimeError):
        list(adapter.fetch_signals(kinds="signal"))

    files = snapshots()
    assert [p.name for p in files] == ["signals_top.jsonl.gz"]
    assert '"ticker": "NVDA"' in snapshot_text()


def test_error_responses_are_snapshotted_too():
    def handler(request):
        return httpx.Response(500, json=problem(500, "INTERNAL_ERROR"))

    client, _ = make_client(handler, max_retries=1)
    with pytest.raises(gg.GovGreedError):
        client.signals_top()
    assert "INTERNAL_ERROR" in snapshot_text()


def test_api_key_never_reaches_a_raw_snapshot():
    def handler(request):
        # A server echoing our credential back in a header must not get it
        # written to disk: persisted headers are whitelisted, not scrubbed.
        return httpx.Response(200, json=envelope([{"ticker": "NVDA",
                                                   "trade_date": "2026-06-01"}]),
                              headers={"Authorization": f"Bearer {API_KEY}",
                                       "X-RateLimit-Limit": "137"})

    adapter, _ = make_adapter(handler)
    list(adapter.fetch_signals(kinds="signal"))

    text = snapshot_text()
    assert API_KEY not in text
    assert "authorization" not in text.lower()
    assert "137" in text  # the rate-limit header is kept


# --- daily pull ---------------------------------------------------------------


def _pull_handler(insider_quota_after: int | None = None, counts=None):
    counts = counts if counts is not None else {}

    def handler(request):
        path = request.url.path
        counts[path] = counts.get(path, 0) + 1
        if path.endswith("/signals/top"):
            return httpx.Response(200, json=envelope([
                {"ticker": "NVDA", "trade_date": "2026-06-01", "tier": "A+"},
                {"ticker": "MSFT", "trade_date": "2026-06-02", "tier": "A"},
                {"ticker": "LMT", "trade_date": "2026-06-03", "tier": "A"},
            ], meta={"quota": {"used": 1, "limit": 137, "tier": "founders"}}))
        if path.endswith("/herd-signals"):
            return httpx.Response(200, json=envelope([
                {"ticker": "NVDA", "window_end": "2026-06-10", "herd_tier": "B",
                 "net_direction": "LONG"},
            ]))
        if path.endswith("/predictions/top"):
            return httpx.Response(200, json=envelope([
                {"ticker": "MSFT", "as_of": "2026-06-11", "status": "ACTIVE",
                 "probability": 0.61},
            ]))
        if path.endswith("/status"):
            return httpx.Response(200, json=envelope({"ok": True}))
        if path.endswith("/insider-signal"):
            n = sum(v for k, v in counts.items() if k.endswith("/insider-signal"))
            if insider_quota_after is not None and n > insider_quota_after:
                return httpx.Response(429, json=problem(429, "DAILY_QUOTA_EXCEEDED",
                                                        request_id="req-dead"),
                                      headers={"X-RateLimit-Remaining": "0"})
            ticker = path.split("/")[-2]
            return httpx.Response(200, json=envelope({"insider_score": 0.8,
                                                      "as_of": "2026-06-12",
                                                      "sector": "Technology"},
                                                     request_id=f"req-{ticker}"))
        return httpx.Response(404, json=problem(404, "NOT_FOUND"))

    return handler, counts


def test_daily_pull_follows_the_call_budget():
    handler, counts = _pull_handler()
    adapter, _ = make_adapter(handler)
    events = adapter.daily_pull(top_n_enrich=5)

    assert adapter.client().calls_made == 6  # 3 list calls + 3 candidates
    assert sum(1 for k in counts if k.endswith("/insider-signal")) == 3
    kinds = {e.kind for e in events}
    assert kinds == {"signal", "herd", "prediction", "insider"}
    assert adapter.last_pull["ok"] is True
    assert adapter.last_pull["aborted"] is False
    assert adapter.last_pull["quota"]["limit"] == 137
    assert adapter.last_pull["quota"]["tier"] == "founders"

    insiders = [e for e in events if e.kind == "insider"]
    assert {e.ticker for e in insiders} == {"NVDA", "MSFT", "LMT"}
    assert all(e.request_id for e in events)


def test_daily_pull_keeps_partial_results_when_quota_dies_mid_run():
    handler, _ = _pull_handler(insider_quota_after=1)
    adapter, _ = make_adapter(handler)

    events = adapter.daily_pull(top_n_enrich=5)  # must not raise

    assert [e.kind for e in events].count("insider") == 1
    assert {e.kind for e in events} == {"signal", "herd", "prediction", "insider"}
    pull = adapter.last_pull
    assert pull["aborted"] is True
    assert "quota" in pull["reason"].lower()
    assert pull["quota"]["remaining"] == 0
    assert pull["steps"][-1] == {"step": "insider:MSFT", "ok": False, "quota_exhausted": True}
    assert API_KEY not in pull["reason"]

    # Progress is durable, not just in memory.
    assert "_pull_progress.jsonl.gz" in {p.name for p in snapshots()}


def test_daily_pull_records_a_failed_step_and_keeps_going():
    def handler(request):
        path = request.url.path
        if path.endswith("/herd-signals"):
            return httpx.Response(500, json=problem(500, "INTERNAL_ERROR",
                                                    request_id="req-herd"))
        return _pull_handler()[0](request)

    adapter, _ = make_adapter(handler, max_retries=1)
    events = adapter.daily_pull(top_n_enrich=1)

    assert {e.kind for e in events} == {"signal", "prediction", "insider"}
    assert adapter.last_pull["ok"] is False
    assert adapter.last_pull["aborted"] is False
    assert adapter.last_pull["errors"][0]["request_id"] == "req-herd"


def test_daily_pull_propagates_auth_failure_instead_of_degrading():
    def handler(request):
        return httpx.Response(401, json=problem(401, "INVALID_API_KEY"))

    adapter, _ = make_adapter(handler)
    with pytest.raises(gg.GovGreedAuthError):
        adapter.daily_pull()


# --- replay -------------------------------------------------------------------


def test_replay_rebuilds_events_from_snapshots_without_key_or_network(monkeypatch):
    handler, _ = _pull_handler()
    adapter, _ = make_adapter(handler)
    live = adapter.daily_pull(top_n_enrich=5)

    monkeypatch.delenv("GOVGREED_API_KEY", raising=False)
    config.reset_settings_cache()

    offline = gg.GovGreedAdapter()  # no key, no transport: replay must still work
    assert offline.available() == (False, "no GOVGREED_API_KEY")
    replayed = list(offline.replay())

    assert {e.uid for e in replayed} == {e.uid for e in live}
    assert {(e.ticker, e.kind) for e in replayed} == {(e.ticker, e.kind) for e in live}
    by_uid = {e.uid: e for e in replayed}
    for event in live:
        twin = by_uid[event.uid]
        assert twin.event_time == event.event_time
        assert twin.knowledge_time == event.knowledge_time  # fetch time, preserved
        assert twin.request_id == event.request_id
        assert twin.source == "govgreed"

    # The insider snapshot carries no ticker in its body; the recorded params do.
    insider = [e for e in replayed if e.kind == "insider"]
    assert {e.ticker for e in insider} == {"NVDA", "MSFT", "LMT"}
    assert all(e.as_row() for e in replayed)


def test_replay_windows_on_event_time_and_ignores_non_signal_snapshots():
    handler, _ = _pull_handler()
    adapter, _ = make_adapter(handler)
    adapter.client().status()  # a snapshot with nothing to normalize
    adapter.daily_pull(top_n_enrich=0)

    windowed = sorted(adapter.replay(start="2026-06-02", end="2026-06-10"),
                      key=lambda e: e.event_time)
    assert [(e.ticker, e.kind, e.event_time.date().isoformat()) for e in windowed] == [
        ("MSFT", "signal", "2026-06-02"),
        ("LMT", "signal", "2026-06-03"),
        ("NVDA", "herd", "2026-06-10"),
    ]  # NVDA's 06-01 signal and MSFT's 06-11 prediction fall outside the window


# --- availability -------------------------------------------------------------


def test_adapter_without_key_is_constructible_and_honest(monkeypatch):
    monkeypatch.delenv("GOVGREED_API_KEY", raising=False)
    config.reset_settings_cache()

    adapter = gg.GovGreedAdapter()  # must not raise at construction
    assert adapter.available() == (False, "no GOVGREED_API_KEY")

    info = adapter.info()
    assert info.available is False
    assert info.reason == "no GOVGREED_API_KEY"
    assert info.provides == frozenset({"signals"})
    assert info.quota_limit is None

    with pytest.raises(AdapterUnavailable):
        list(adapter.fetch_signals())


def test_registry_resolves_the_adapter():
    from lab.adapters.base import get_adapter

    adapter = get_adapter("govgreed")
    assert isinstance(adapter, gg.GovGreedAdapter)
    assert adapter.name == "govgreed"


def test_bad_input_raises_value_error():
    def handler(request):  # pragma: no cover - never reached
        return httpx.Response(200, json=envelope([]))

    client, _ = make_client(handler)
    with pytest.raises(ValueError):
        client.insider_signal("   ")
    with pytest.raises(ValueError):
        gg.GovGreedClient(api_key=API_KEY, max_retries=0)

    adapter, _ = make_adapter(handler)
    with pytest.raises(ValueError):
        list(adapter.fetch_signals(kinds="not-a-kind"))
    with pytest.raises(ValueError):
        adapter.daily_pull(top_n_enrich=-1)
