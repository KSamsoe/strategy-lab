"""GovGreed alt-data: the official REST client and the signal adapter over it.

This talks to the documented API at ``https://www.govgreed.com/api/v1`` only.
The earlier incarnation of this bot drove the web dashboard with a headless
browser; that approach is dead and is not modelled anywhere below -- no HTML,
no DOM, no page objects.

Three properties of the vendor shape everything here:

*Quota is the scarce resource.* The free tier is documented as 100 calls/day for
week one and then 20/day, except one page of their docs says 250, and the
founders tier is 750. Those numbers are in flux, so none of them appear in this
file. The client reads the ``X-RateLimit-*`` headers and the ``meta.quota``
envelope field and reports whatever the server claims; callers budget against
that, not against a constant.

*The schema is a moving target.* Closed beta, fields being renamed, tables
quarantined. Parsing is therefore defensive on both sides: unknown fields ride
along in ``Event.payload`` instead of raising, and a record missing something
required is logged and dropped so one bad row cannot cost a whole run.

*Backfill is not for sale on our tier.* Every response is written verbatim to
``lab.store.raw`` **before** it is parsed, which makes the accumulated daily
pulls our backtest dataset -- see :meth:`GovGreedAdapter.replay`. That ordering
is load-bearing: a normalizer that crashes must still leave the bytes on disk.

The two-timestamp mapping is the reason this adapter exists at all. A
congressional trade happens up to 45 days before the STOCK Act disclosure that
tells us about it, so ``event_time`` is the trade and ``knowledge_time`` is the
disclosure (or our fetch, whichever is later). Collapsing the two produces a
backtest that looks like fraud.

The API key is shown once at issuance and lives only in the environment. It is
never logged, never put in an exception message, and never persisted -- the raw
snapshots keep response bodies plus a whitelist of rate-limit headers, nothing
that could carry credentials back to disk.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.parse import quote

import httpx

from lab.adapters.base import (
    AdapterError,
    AdapterInfo,
    BaseAdapter,
    QuotaExceeded,
)
from lab.config import get_settings
from lab.store import raw
from lab.store.schema import Event
from lab.timeutil import to_utc, utcnow

_LOG = logging.getLogger(__name__)

#: ``source`` column value for everything this module emits.
SOURCE = "govgreed"

#: ``Event.kind`` values this adapter produces, one per signal-bearing endpoint.
KINDS: tuple[str, ...] = ("signal", "herd", "prediction", "insider")

#: Stable endpoint label -> kind. The label is what :func:`lab.store.raw.save`
#: records, so :meth:`GovGreedAdapter.replay` can tell snapshots apart years
#: later. Keys are normalized (``/`` and ``-`` folded to ``_``) because the raw
#: store sanitizes endpoint names into filenames.
ENDPOINT_KINDS: dict[str, str] = {
    "signals_top": "signal",
    "herd_signals": "herd",
    "predictions_top": "prediction",
    "companies_insider_signal": "insider",
}

#: Only these response headers reach disk. Whitelisting rather than blacklisting
#: means a future header cannot leak anything by default.
_KEPT_HEADERS = ("x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-used",
                 "x-ratelimit-reset", "x-ratelimit-tier", "x-request-id", "retry-after")

_BACKOFF_BASE = 0.5
_DEFAULT_RETRY_AFTER = 1.0
#: An unlabelled 429 asking us back within this many seconds is read as a burst
#: trip, not the daily wall; anything longer is treated as the wall, because
#: guessing "burst" wrongly burns the retry buffer on a lost cause.
_BURST_RETRY_CEILING = 60.0


def _norm_endpoint(endpoint: str) -> str:
    return endpoint.strip("/").replace("/", "_").replace("-", "_").lower()


# --- quota -------------------------------------------------------------------


@dataclass(slots=True)
class Quota:
    """Whatever the server currently claims about our budget.

    Every field is optional because every field is optional at the vendor: the
    headers and the envelope disagree about which are present, and both change.
    Nothing here is defaulted to a documented number.
    """

    used: int | None = None
    limit: int | None = None
    remaining: int | None = None
    tier: str | None = None
    reset_at: datetime | None = None

    @property
    def exhausted(self) -> bool:
        return self.remaining is not None and self.remaining <= 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "used": self.used,
            "limit": self.limit,
            "remaining": self.remaining,
            "tier": self.tier,
            "reset_at": self.reset_at.isoformat() if self.reset_at else None,
        }


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _parse_time(value: Any) -> datetime | None:
    """Best-effort timestamp coercion; unparseable input is *absent*, not fatal."""
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        if isinstance(value, (int, float)):
            return to_utc(float(value))
        text = str(value).strip()
        if text.isdigit() and len(text) >= 9:  # epoch seconds, not a YYYYMMDD
            return to_utc(float(text))
        return to_utc(text)
    except Exception:  # noqa: BLE001 - any parse failure means "no timestamp"
        return None


def _quota_from(headers: Mapping[str, str], meta_quota: Mapping[str, Any] | None) -> Quota:
    """Merge the two sources the vendor publishes, envelope winning on conflict.

    The envelope is the explicit, versioned surface; headers are the cheap one.
    Whichever arrives, the numbers come from the wire.
    """
    h = {k.lower(): v for k, v in dict(headers).items()}
    q = Quota(
        used=_int_or_none(h.get("x-ratelimit-used")),
        limit=_int_or_none(h.get("x-ratelimit-limit")),
        remaining=_int_or_none(h.get("x-ratelimit-remaining")),
        tier=(h.get("x-ratelimit-tier") or None),
        reset_at=_parse_time(h.get("x-ratelimit-reset")),
    )
    if isinstance(meta_quota, Mapping):
        for attr, keys in (
            ("used", ("used", "calls_used", "count")),
            ("limit", ("limit", "daily_limit", "quota", "max")),
            ("remaining", ("remaining", "left")),
        ):
            for key in keys:
                got = _int_or_none(meta_quota.get(key))
                if got is not None:
                    setattr(q, attr, got)
                    break
        for key in ("tier", "plan"):
            if meta_quota.get(key):
                q.tier = str(meta_quota[key])
                break
        for key in ("reset_at", "resets_at", "reset", "window_reset"):
            got = _parse_time(meta_quota.get(key))
            if got is not None:
                q.reset_at = got
                break
    # Derive only the arithmetic identity, never a magnitude.
    if q.used is None and q.limit is not None and q.remaining is not None:
        q.used = q.limit - q.remaining
    if q.remaining is None and q.limit is not None and q.used is not None:
        q.remaining = q.limit - q.used
    return q


def _merge_quota(old: Quota, new: Quota) -> Quota:
    """Keep the last known value for anything this response did not mention."""
    return Quota(
        used=new.used if new.used is not None else old.used,
        limit=new.limit if new.limit is not None else old.limit,
        remaining=new.remaining if new.remaining is not None else old.remaining,
        tier=new.tier or old.tier,
        reset_at=new.reset_at or old.reset_at,
    )


# --- errors ------------------------------------------------------------------


class GovGreedError(AdapterError):
    """An RFC 7807 problem response, carried with its ``request_id``.

    The id is in the message because it is the only handle support has on a
    closed-beta failure; the message is assembled from server-supplied fields
    only, so a credential can never end up in a traceback.
    """

    def __init__(
        self,
        status: int,
        code: str | None = None,
        title: str = "",
        detail: str = "",
        request_id: str | None = None,
    ) -> None:
        self.status = status
        self.code = code
        self.title = title
        self.detail = detail
        self.request_id = request_id
        label = code or title or f"HTTP {status}"
        message = f"GovGreed {status} {label}"
        if detail:
            message += f": {detail}"
        super().__init__(f"{message} [request_id={request_id or 'unknown'}]")


class GovGreedAuthError(GovGreedError):
    """401/403. Halt; a retry cannot mint a valid key and the old one may be gone."""


class GovGreedQuotaError(QuotaExceeded):
    """429 ``DAILY_QUOTA_EXCEEDED``. Abort the run cleanly, record how far it got.

    Subclasses :class:`lab.adapters.base.QuotaExceeded` rather than
    :class:`GovGreedError` because callers handle "out of budget" generically,
    across vendors, and must not catch it as a transport failure to be retried.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        request_id: str | None = None,
        quota: Quota | None = None,
        code: str = "DAILY_QUOTA_EXCEEDED",
    ) -> None:
        super().__init__(f"{message} [request_id={request_id or 'unknown'}]", retry_after=retry_after)
        self.status = 429
        self.code = code
        self.request_id = request_id
        self.quota = quota or Quota()


class GovGreedBurstError(GovGreedError):
    """429 ``BURST_LIMIT_EXCEEDED``, surviving ``Retry-After`` and the retry budget.

    The client self-throttles below the documented burst cap, so seeing this
    means the cap moved or something else is sharing the key.
    """

    def __init__(self, *args: Any, retry_after: float | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.retry_after = retry_after


# --- client ------------------------------------------------------------------


def _qparam(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(v) for v in value)
    return value


def _problem_code(body: Mapping[str, Any]) -> str | None:
    for key in ("code", "error_code", "error"):
        value = body.get(key)
        if isinstance(value, str) and value:
            return value.strip().upper()
    kind = body.get("type")
    if isinstance(kind, str) and kind:
        tail = kind.rstrip("/").rsplit("/", 1)[-1]
        if tail:
            return tail.strip().upper().replace("-", "_")
    return None


def _retry_after(headers: Mapping[str, str]) -> float | None:
    value = {k.lower(): v for k, v in dict(headers).items()}.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(str(value).strip()))
    except (TypeError, ValueError):
        pass
    try:  # HTTP-date form
        return max(0.0, (parsedate_to_datetime(str(value)) - utcnow()).total_seconds())
    except Exception:  # noqa: BLE001
        return None


class GovGreedClient:
    """Thin, honest wrapper: auth, throttle, envelope, quota, raw persistence.

    ``session`` accepts an ``httpx.Client`` or a bare ``httpx.BaseTransport``
    (including ``httpx.MockTransport``) so the whole client is exercisable with
    no network. ``sleep`` is injectable for the same reason -- backoff must be
    testable without spending the wall clock on it.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        min_interval: float = 0.5,
        max_retries: int = 3,
        timeout: float = 20.0,
        session: httpx.Client | httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.govgreed_api_key
        self.base_url = (base_url or settings.govgreed_base_url).rstrip("/")
        if min_interval < 0:
            raise ValueError("min_interval must be >= 0")
        if max_retries < 1:
            raise ValueError("max_retries is the total attempt budget; must be >= 1")
        self.min_interval = float(min_interval)
        self.max_retries = int(max_retries)
        self.timeout = float(timeout)
        self._sleep = time.sleep if sleep is None else sleep

        if isinstance(session, httpx.Client):
            self._http, self._owns_http = session, False
        elif isinstance(session, httpx.BaseTransport):
            self._http, self._owns_http = httpx.Client(transport=session, timeout=timeout), True
        elif session is None:
            self._http, self._owns_http = httpx.Client(timeout=timeout), True
        else:
            raise ValueError("session must be an httpx.Client, an httpx transport, or None")

        self._quota = Quota()
        self._calls = 0
        self._last_sent = 0.0
        self._last_call: datetime | None = None
        #: Stamped on every response so the adapter can attach provenance to the
        #: events a typed helper returns without threading meta through each one.
        self.last_request_id: str | None = None
        self.last_fetched_at: datetime | None = None

    # -- plumbing ------------------------------------------------------------

    @property
    def quota(self) -> Quota:
        return self._quota

    @property
    def calls_made(self) -> int:
        """HTTP requests actually sent, retries included -- the vendor counts those."""
        return self._calls

    @property
    def last_call(self) -> datetime | None:
        return self._last_call

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "GovGreedClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - never render the key
        return f"<GovGreedClient base_url={self.base_url!r} calls={self._calls}>"

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "strategy-lab/0.1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        gap = time.monotonic() - self._last_sent
        if self._last_sent and gap < self.min_interval:
            self._sleep(self.min_interval - gap)
        self._last_sent = time.monotonic()

    # -- the one request path ------------------------------------------------

    def get(self, path: str, **params: Any) -> tuple[Any, dict[str, Any]]:
        """``GET path`` and return ``(data, meta)`` from the ``{data, meta}`` envelope."""
        return self._request(path, params=params)

    def _request(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        endpoint: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        query = {k: _qparam(v) for k, v in dict(params or {}).items() if v is not None}
        label = endpoint or path.strip("/") or "root"
        url = f"{self.base_url}/{path.strip('/')}"

        attempt = 0
        while True:
            attempt += 1
            self._throttle()
            self._calls += 1
            try:
                response = self._http.get(url, params=query, headers=self._headers())
            except httpx.HTTPError as exc:
                # Connection-level failure: no body to snapshot, same shape of
                # remedy as a 5xx.
                if attempt >= self.max_retries:
                    raise GovGreedError(0, "TRANSPORT_ERROR", "request failed",
                                        f"{type(exc).__name__} after {attempt} attempts",
                                        None) from exc
                self._sleep(_BACKOFF_BASE * (2 ** (attempt - 1)))
                continue

            fetched_at = utcnow()
            self._last_call = fetched_at
            self.last_fetched_at = fetched_at

            body = self._decode(response)
            meta = body.get("meta") if isinstance(body, dict) else None
            meta = dict(meta) if isinstance(meta, Mapping) else {}
            request_id = self._request_id(body, meta, response.headers)
            self.last_request_id = request_id

            # Persist first, parse second. If normalization later explodes on a
            # renamed field, the bytes are still on disk and re-parseable.
            kept = {k: v for k, v in response.headers.items() if k.lower() in _KEPT_HEADERS}
            raw.save(
                SOURCE,
                label,
                body,
                request_id=request_id,
                fetched_at=fetched_at,
                params=query,
                meta={"status": response.status_code, "headers": kept},
            )

            self._quota = _merge_quota(self._quota, _quota_from(response.headers, meta.get("quota")))

            if response.status_code < 400:
                data = body.get("data") if isinstance(body, dict) and "data" in body else body
                return data, meta

            problem = body if isinstance(body, dict) else {}
            code = _problem_code(problem)
            title = str(problem.get("title") or response.reason_phrase or f"HTTP {response.status_code}")
            detail = str(problem.get("detail") or "")
            status = response.status_code

            if status in (401, 403):
                raise GovGreedAuthError(status, code or "UNAUTHORIZED", title, detail, request_id)

            if status == 429:
                after = _retry_after(response.headers)
                burst = code == "BURST_LIMIT_EXCEEDED" or (
                    code not in {"DAILY_QUOTA_EXCEEDED", "QUOTA_EXCEEDED"}
                    and after is not None
                    and after <= _BURST_RETRY_CEILING
                )
                if not burst:
                    raise GovGreedQuotaError(
                        f"GovGreed daily quota exhausted ({title})",
                        retry_after=after,
                        request_id=request_id,
                        quota=self._quota,
                        code=code or "DAILY_QUOTA_EXCEEDED",
                    )
                if attempt >= self.max_retries:
                    raise GovGreedBurstError(status, code or "BURST_LIMIT_EXCEEDED", title,
                                             f"{detail} (still throttled after {attempt} attempts)",
                                             request_id, retry_after=after)
                self._sleep(after if after is not None else _DEFAULT_RETRY_AFTER)
                continue

            if status >= 500:
                if attempt >= self.max_retries:
                    raise GovGreedError(status, code or "INTERNAL_ERROR", title,
                                        f"{detail} (gave up after {attempt} attempts)".strip(),
                                        request_id)
                self._sleep(_BACKOFF_BASE * (2 ** (attempt - 1)))
                continue

            raise GovGreedError(status, code, title, detail, request_id)

    @staticmethod
    def _decode(response: httpx.Response) -> Any:
        try:
            return response.json()
        except Exception:  # noqa: BLE001 - HTML error page, empty body, truncated JSON
            return {"_non_json_body": response.text[:4000]}

    @staticmethod
    def _request_id(body: Any, meta: Mapping[str, Any], headers: Mapping[str, str]) -> str | None:
        for candidate in (meta.get("request_id"), meta.get("requestId")):
            if candidate:
                return str(candidate)
        if isinstance(body, Mapping):
            for key in ("request_id", "requestId", "trace_id"):
                if body.get(key):
                    return str(body[key])
        lowered = {k.lower(): v for k, v in dict(headers).items()}
        for key in ("x-request-id", "x-correlation-id"):
            if lowered.get(key):
                return str(lowered[key])
        return None

    # -- typed endpoints -----------------------------------------------------

    def status(self) -> dict[str, Any]:
        return _as_dict(self._request("/status")[0])

    def me(self) -> dict[str, Any]:
        return _as_dict(self._request("/me")[0])

    def usage(self) -> dict[str, Any]:
        return _as_dict(self._request("/me/usage")[0])

    def atlas(self) -> dict[str, Any]:
        return _as_dict(self._request("/atlas")[0])

    def signals_top(self, *, tier: str = "A", fresh: bool = True, limit: int = 25) -> list[dict[str, Any]]:
        data, _ = self._request(
            "/signals/top", params={"tier": tier, "fresh": fresh, "limit": limit},
            endpoint="signals/top",
        )
        return _as_records(data)

    def herd_signals(self, *, days: int = 30) -> list[dict[str, Any]]:
        data, _ = self._request("/herd-signals", params={"days": days}, endpoint="herd-signals")
        return _as_records(data)

    def predictions_top(self, *, tier: str = "A", status: str = "ACTIVE") -> list[dict[str, Any]]:
        data, _ = self._request(
            "/predictions/top", params={"tier": tier, "status": status},
            endpoint="predictions/top",
        )
        return _as_records(data)

    def insider_signal(self, ticker: str) -> dict[str, Any]:
        symbol = _require_ticker(ticker)
        # The ticker is a path segment, but it is also recorded as a param so a
        # replay years later can tell whose insider score this snapshot is.
        data, _ = self._request(
            f"/companies/{quote(symbol)}/insider-signal",
            params={"ticker": symbol},
            endpoint="companies/insider-signal",
        )
        return _as_dict(data)

    def bill_timeline(self, bill: str) -> dict[str, Any]:
        if not str(bill).strip():
            raise ValueError("bill identifier is required")
        data, _ = self._request(f"/bills/{quote(str(bill).strip())}/timeline",
                                params={"bill": str(bill).strip()}, endpoint="bills/timeline")
        return _as_dict(data)

    def sector_positioning(self, sector: str) -> dict[str, Any]:
        if not str(sector).strip():
            raise ValueError("sector is required")
        data, _ = self._request(f"/sectors/{quote(str(sector).strip())}/positioning",
                                params={"sector": str(sector).strip()},
                                endpoint="sectors/positioning")
        return _as_dict(data)


# --- defensive normalization -------------------------------------------------

_TICKER_FIELDS = ("ticker", "symbol", "ticker_symbol", "company_ticker")
#: Priority order matters: the *trade* is the event, and a filing date is only a
#: fallback for payloads that never expose the underlying transaction.
_EVENT_TIME_FIELDS = (
    "trade_date", "transaction_date", "traded_at", "transacted_at", "trade_time",
    "filed_at", "filing_date", "event_date", "event_time",
    "window_end", "period_end", "as_of", "as_of_date", "date",
)
_DISCLOSURE_FIELDS = (
    "disclosure_date", "disclosed_at", "published_at", "reported_at",
    "filed_at", "filing_date", "updated_at", "last_updated",
)
#: Whatever makes two rows for the same ticker and date genuinely different.
#: Deliberately excludes scores and timestamps: those move between pulls, and a
#: uid that moves with them would double-count instead of restating.
_IDENTITY_FIELDS = (
    "id", "signal_id", "prediction_id", "filing_id", "disclosure_id", "transaction_id",
    "politician", "politician_name", "member", "member_name", "representative",
    "senator", "chamber", "fund", "cluster_id", "source",
)
_DIRECTION_FIELDS = ("direction", "net_direction", "side", "bias", "action")
_TIER_FIELDS = ("tier", "composite_tier", "herd_tier", "signal_tier", "grade")
_SCORE_FIELDS = ("score", "composite_score", "insider_score", "herd_score",
                 "conviction", "confidence", "probability", "strength")
_SECTOR_FIELDS = ("sector", "gics_sector", "industry_sector")
_FRESH_FIELDS = ("fresh", "is_fresh", "freshness")

_DIRECTION_MAP = {
    "BUY": "BUY", "LONG": "BUY", "BULLISH": "BUY", "ACCUMULATE": "BUY", "ACCUMULATING": "BUY",
    "SELL": "SELL", "SHORT": "SELL", "BEARISH": "SELL", "DISTRIBUTE": "SELL", "DISTRIBUTING": "SELL",
}

#: Kinds whose payload is one record, not a collection.
_SINGLETON_KINDS = frozenset({"insider"})


def _require_ticker(ticker: str) -> str:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        raise ValueError("ticker is required")
    return symbol


def _as_dict(data: Any) -> dict[str, Any]:
    return dict(data) if isinstance(data, Mapping) else {}


def _as_records(data: Any) -> list[dict[str, Any]]:
    """Coerce whatever ``data`` turned out to be into a list of records."""
    if data is None:
        return []
    if isinstance(data, list):
        return [dict(r) for r in data if isinstance(r, Mapping)]
    if isinstance(data, Mapping):
        for key in ("items", "results", "records", "signals", "rows", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return [dict(r) for r in value if isinstance(r, Mapping)]
        return [dict(data)]
    return []


def _first(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1", "fresh"}:
            return True
        if text in {"false", "no", "n", "0", "stale"}:
            return False
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _uid(kind: str, ticker: str, event_time: datetime, record: Mapping[str, Any]) -> str:
    """Stable across pulls: same underlying fact, same uid, so re-pulls restate.

    Identity comes from vendor ids and the person or fund behind the filing. When
    a payload offers none, the natural key (kind, ticker, date) stands alone --
    which is exactly the dedupe we want for one composite score per day.
    """
    bits = [f"{key}={record[key]}" for key in _IDENTITY_FIELDS
            if key in record and record[key] not in (None, "")]
    digest = hashlib.blake2s("|".join(bits).encode("utf-8"), digest_size=6).hexdigest()
    return f"{kind}:{ticker}:{event_time.date().isoformat()}:{digest}"


def normalize_record(
    record: Mapping[str, Any],
    kind: str,
    *,
    fetched_at: datetime,
    request_id: str | None = None,
    default_ticker: str | None = None,
    endpoint: str | None = None,
) -> Event | None:
    """One vendor record -> one :class:`Event`, or ``None`` if it is unusable.

    Returning ``None`` instead of raising is the schema-drift policy: a renamed
    field costs one row and a warning, not the run.
    """
    if not isinstance(record, Mapping):
        _LOG.warning("govgreed %s: skipping non-object record %r", kind, type(record).__name__)
        return None

    fetched_at = to_utc(fetched_at)  # public entry point: coerce at the boundary
    ticker = _first(record, _TICKER_FIELDS) or default_ticker
    ticker = str(ticker).strip().upper() if ticker else ""
    if not ticker:
        _LOG.warning("govgreed %s: skipping record with no ticker (keys=%s)",
                     kind, sorted(record)[:12])
        return None

    event_time = None
    for key in _EVENT_TIME_FIELDS:
        event_time = _parse_time(record.get(key))
        if event_time is not None:
            break
    if event_time is None:
        _LOG.warning("govgreed %s/%s: skipping record with no usable date (keys=%s)",
                     kind, ticker, sorted(record)[:12])
        return None

    disclosed = None
    for key in _DISCLOSURE_FIELDS:
        disclosed = _parse_time(record.get(key))
        if disclosed is not None:
            break

    # We knew it when we fetched it -- unless the vendor stamps a disclosure
    # later than our fetch, in which case believe the vendor. The final clamp
    # defends against a payload dating a trade in the future: knowing something
    # before it happens is the look-ahead bug this whole schema exists to stop.
    knowledge_time = fetched_at
    if disclosed is not None and disclosed > knowledge_time:
        knowledge_time = disclosed
    if knowledge_time < event_time:
        knowledge_time = event_time

    direction = _first(record, _DIRECTION_FIELDS)
    if direction is not None:
        text = str(direction).strip().upper()
        direction = _DIRECTION_MAP.get(text, text) or None

    tier = _first(record, _TIER_FIELDS)
    sector = _first(record, _SECTOR_FIELDS)
    payload = dict(record)
    if endpoint:
        payload.setdefault("_endpoint", endpoint)

    return Event(
        event_time=event_time,
        knowledge_time=knowledge_time,
        source=SOURCE,
        ticker=ticker,
        kind=kind,
        uid=_uid(kind, ticker, event_time, record),
        direction=direction,
        tier=str(tier).strip().upper() if tier is not None else None,
        score=_coerce_float(_first(record, _SCORE_FIELDS)),
        sector=str(sector).strip() if sector is not None else None,
        fresh=_coerce_bool(_first(record, _FRESH_FIELDS)),
        request_id=request_id,
        payload=payload,
    )


def normalize(
    data: Any,
    kind: str,
    *,
    fetched_at: datetime,
    request_id: str | None = None,
    default_ticker: str | None = None,
    endpoint: str | None = None,
) -> list[Event]:
    if kind not in KINDS:
        raise ValueError(f"unknown govgreed kind {kind!r}; known: {', '.join(KINDS)}")
    records = [_as_dict(data)] if kind in _SINGLETON_KINDS and isinstance(data, Mapping) \
        else _as_records(data)
    out: list[Event] = []
    for record in records:
        if not record:
            continue
        event = normalize_record(record, kind, fetched_at=fetched_at, request_id=request_id,
                                 default_ticker=default_ticker, endpoint=endpoint)
        if event is not None:
            out.append(event)
    return out


# --- adapter -----------------------------------------------------------------


class GovGreedAdapter(BaseAdapter):
    """Signals only. Bars come from a market-data adapter; this is alt-data."""

    name = SOURCE
    provides = frozenset({"signals"})

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        client: GovGreedClient | None = None,
        session: httpx.Client | httpx.BaseTransport | None = None,
        min_interval: float = 0.5,
        max_retries: int = 3,
        timeout: float = 20.0,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._client = client
        self._client_kwargs: dict[str, Any] = {
            "session": session, "min_interval": min_interval,
            "max_retries": max_retries, "timeout": timeout, "sleep": sleep,
        }
        #: Outcome of the last :meth:`daily_pull`, including partial runs.
        self.last_pull: dict[str, Any] = {}

    # -- availability --------------------------------------------------------

    def available(self) -> tuple[bool, str]:
        if self._client is not None and self._client.api_key:
            return True, ""
        if self._api_key or get_settings().govgreed_api_key:
            return True, ""
        return False, "no GOVGREED_API_KEY"

    def client(self) -> GovGreedClient:
        """Constructed lazily, so an adapter with no key is still constructible."""
        if self._client is None:
            self.require_available()
            self._client = GovGreedClient(
                api_key=self._api_key, base_url=self._base_url, **self._client_kwargs
            )
        return self._client

    def info(self) -> AdapterInfo:
        ok, reason = self.available()
        quota = self._client.quota if self._client is not None else Quota()
        return AdapterInfo(
            name=self.name,
            provides=self.provides,
            available=ok,
            reason=reason,
            quota_used=quota.used,
            quota_limit=quota.limit,
            quota_tier=quota.tier,
            last_call=self._client.last_call if self._client is not None else None,
            detail={
                "base_url": self._base_url or get_settings().govgreed_base_url,
                "calls_made": self._client.calls_made if self._client is not None else 0,
                "quota": quota.as_dict(),
                "last_pull": {k: v for k, v in self.last_pull.items() if k != "events"},
            },
        )

    # -- live fetch ----------------------------------------------------------

    def fetch_signals(
        self, start: datetime | None = None, end: datetime | None = None, **query: Any
    ) -> Iterator[Event]:
        """Pull the signal-bearing endpoints and yield normalized events.

        ``start``/``end`` filter on ``event_time`` (the trade), matching the
        store's convention; ``knowledge_time`` filtering is the backtester's job.
        """
        self.require_available()
        kinds = query.pop("kinds", None) or ("signal", "herd", "prediction")
        if isinstance(kinds, str):
            kinds = tuple(k.strip() for k in kinds.split(",") if k.strip())
        unknown = sorted(set(kinds) - set(KINDS))
        if unknown:
            raise ValueError(f"unknown govgreed kinds {unknown}; known: {', '.join(KINDS)}")

        client = self.client()
        events: list[Event] = []
        if "signal" in kinds:
            records = client.signals_top(
                tier=query.get("tier", "A"),
                fresh=bool(query.get("fresh", True)),
                limit=int(query.get("limit", 25)),
            )
            events += self._normalize_last(client, records, "signal", "signals/top")
        if "herd" in kinds:
            records = client.herd_signals(days=int(query.get("days", 30)))
            events += self._normalize_last(client, records, "herd", "herd-signals")
        if "prediction" in kinds:
            records = client.predictions_top(
                tier=query.get("tier", "A"), status=query.get("status", "ACTIVE")
            )
            events += self._normalize_last(client, records, "prediction", "predictions/top")
        if "insider" in kinds:
            for ticker in _ticker_list(query.get("tickers")):
                record = client.insider_signal(ticker)
                events += self._normalize_last(client, record, "insider",
                                               "companies/insider-signal", ticker)

        return iter(_window(events, start, end))

    def _normalize_last(
        self,
        client: GovGreedClient,
        data: Any,
        kind: str,
        endpoint: str,
        default_ticker: str | None = None,
    ) -> list[Event]:
        return normalize(
            data, kind,
            fetched_at=client.last_fetched_at or utcnow(),
            request_id=client.last_request_id,
            default_ticker=default_ticker,
            endpoint=endpoint,
        )

    # -- the daily job -------------------------------------------------------

    def daily_pull(self, *, top_n_enrich: int = 5) -> list[Event]:
        """The §4 call budget: 3 list calls plus up to ``top_n_enrich`` lookups.

        Quota exhaustion mid-run is an expected outcome, not a failure: the run
        stops calling, keeps everything already normalized, and records where it
        got to in :attr:`last_pull` and in the raw store. Throwing away a partial
        pull would waste calls we cannot get back until the window resets.
        """
        if top_n_enrich < 0:
            raise ValueError("top_n_enrich must be >= 0")
        self.require_available()
        client = self.client()
        started = utcnow()
        events: list[Event] = []
        steps: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        aborted_reason: str | None = None
        candidates: list[dict[str, Any]] = []

        def run(step: str, fn: Callable[[], list[Event]]) -> bool:
            """Returns False when the run must stop entirely."""
            nonlocal aborted_reason
            try:
                got = fn()
            except GovGreedQuotaError as exc:
                aborted_reason = str(exc)
                steps.append({"step": step, "ok": False, "quota_exhausted": True})
                _LOG.warning("govgreed daily_pull: quota exhausted at %s; keeping %d events",
                             step, len(events))
                return False
            except GovGreedAuthError:
                # Auth cannot be worked around and may mean a rotated key; the
                # caller needs the exception, not a quietly short pull.
                raise
            except GovGreedError as exc:
                errors.append({"step": step, "error": str(exc), "request_id": exc.request_id})
                steps.append({"step": step, "ok": False, "events": 0})
                _LOG.warning("govgreed daily_pull: %s failed (%s); continuing", step, exc)
                return True
            events.extend(got)
            steps.append({"step": step, "ok": True, "events": len(got)})
            return True

        def signals() -> list[Event]:
            nonlocal candidates
            candidates = client.signals_top()
            return self._normalize_last(client, candidates, "signal", "signals/top")

        proceed = run("signals/top", signals)
        if proceed:
            proceed = run("herd-signals",
                          lambda: self._normalize_last(client, client.herd_signals(),
                                                       "herd", "herd-signals"))
        if proceed:
            proceed = run("predictions/top",
                          lambda: self._normalize_last(client, client.predictions_top(),
                                                       "prediction", "predictions/top"))
        if proceed:
            for ticker in _top_tickers(candidates, top_n_enrich):
                if not run(f"insider:{ticker}",
                           lambda t=ticker: self._normalize_last(
                               client, client.insider_signal(t), "insider",
                               "companies/insider-signal", t)):
                    break

        self.last_pull = {
            "started_at": started.isoformat(),
            "finished_at": utcnow().isoformat(),
            "ok": aborted_reason is None and not errors,
            "aborted": aborted_reason is not None,
            "reason": aborted_reason,
            "steps": steps,
            "errors": errors,
            "events": len(events),
            "calls_made": client.calls_made,
            "quota": client.quota.as_dict(),
        }
        # Progress belongs on disk too: a crashed cron job should still be able
        # to answer "how much of today's budget did we already spend?".
        raw.save(SOURCE, "_pull_progress", self.last_pull, request_id=client.last_request_id)
        return events

    # -- replay --------------------------------------------------------------

    def replay(
        self, start: datetime | None = None, end: datetime | None = None
    ) -> Iterator[Event]:
        """Re-normalize stored snapshots into events. No key, no network.

        This is what turns daily pulls into a backtestable history: the vendor
        sells backfill at institutional tier, we accumulate it for free, and a
        parser fix reaches all of it retroactively.
        """
        for record in raw.iter_raw(SOURCE):
            endpoint = str(record.get("endpoint") or "")
            kind = ENDPOINT_KINDS.get(_norm_endpoint(endpoint))
            if kind is None:
                continue  # /status, /me, _pull_progress: nothing to normalize
            body = record.get("payload")
            data = body.get("data") if isinstance(body, Mapping) and "data" in body else body
            meta = body.get("meta") if isinstance(body, Mapping) else None
            meta = meta if isinstance(meta, Mapping) else {}
            params = record.get("params") if isinstance(record.get("params"), Mapping) else {}
            request_id = record.get("request_id") or meta.get("request_id")
            fetched_at = record.get("_fetched_at") or _parse_time(record.get("fetched_at"))
            if fetched_at is None:
                continue
            events = normalize(
                data, kind,
                fetched_at=to_utc(fetched_at),
                request_id=str(request_id) if request_id else None,
                default_ticker=params.get("ticker"),
                endpoint=endpoint,
            )
            yield from _window(events, start, end)


def _window(events: Iterable[Event], start: datetime | None, end: datetime | None) -> list[Event]:
    lo = to_utc(start) if start is not None else None
    hi = to_utc(end) if end is not None else None
    out = [e for e in events
           if (lo is None or e.event_time >= lo) and (hi is None or e.event_time <= hi)]
    out.sort(key=lambda e: (e.event_time, e.uid))
    return out


def _ticker_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [t.strip().upper() for t in value.split(",") if t.strip()]
    return [str(t).strip().upper() for t in value if str(t).strip()]


def _top_tickers(records: Sequence[Mapping[str, Any]], n: int) -> list[str]:
    """First ``n`` distinct tickers in the vendor's own ranking order."""
    seen: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        ticker = _first(record, _TICKER_FIELDS)
        if not ticker:
            continue
        symbol = str(ticker).strip().upper()
        if symbol and symbol not in seen:
            seen.append(symbol)
        if len(seen) >= n:
            break
    return seen[:n]
