"""Report tests.

The property most likely to regress silently is self-containment: a report that
quietly grows an external stylesheet, webfont, or image reference still looks
fine on the machine that made it and is broken everywhere else. So every
document rendered here is scanned for external references, and the two standing
caveats -- optimistic fills and survivorship -- are asserted as *content*.

Runs are real event-driven backtests. The suite's autouse path isolation points
the store at a tmpdir, so the tape is the conftest's deterministic bar builder
rather than the developer's parquet store; the benchmark test writes its own
bars into the isolated store, which is also what exercises the report's
store-read path.
"""

from __future__ import annotations

import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest

from lab.backtest import report as R
from lab.backtest.report import (
    CONTAMINATION_WARNING,
    OPTIMISTIC_WARNING,
    render_report,
    report_path,
)
from lab.backtest.runner import BacktestConfig, BacktestResult, run_backtest

from tests.conftest import make_bars, make_strategy, make_view

TICKERS = ("AAA", "BBB")
BARS = 140
#: Wide enough that the gate never clips the probe's targets -- this file is
#: testing the report, and a clipped run has fewer round trips to render.
LIMITS = {
    "max_position_pct": 0.5,
    "max_positions": 4,
    "max_gross_exposure": 1.0,
    "max_daily_loss_pct": 0.9,
    "max_orders_per_day": 50,
    "min_order_notional": 10.0,
}

#: Void elements never close, so a nesting check must not expect an end tag.
_VOID = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
     "meta", "param", "source", "track", "wbr"}
)
_EXTERNAL = re.compile(r"https?://")


# --- helpers -------------------------------------------------------------------


class _Nesting(HTMLParser):
    """Strict enough to catch a malformed fragment, lax enough not to
    reimplement an HTML5 parser: tags must nest and every open tag must close."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.seen: set[str] = set()
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        self.seen.add(tag)
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        if not self.stack:
            self.errors.append(f"</{tag}> with nothing open")
        elif self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes <{self.stack[-1]}>")
            self.stack.pop()
        else:
            self.stack.pop()


def parse_html(text: str) -> _Nesting:
    p = _Nesting()
    p.feed(text)
    p.close()
    assert not p.errors, p.errors[:5]
    assert not p.stack, f"unclosed tags: {p.stack}"
    return p


def assert_self_contained(text: str) -> None:
    hits = _EXTERNAL.findall(text)
    assert not hits, f"report references {len(hits)} external URL(s); it must render offline"
    for token in ('src="', "@import", "<script", "<iframe"):
        assert token not in text, f"report grew a {token!r} reference"


def swing_strategy(tickers: Sequence[str] = TICKERS, hold: int = 7) -> Any:
    """Alternate between two names every ``hold`` bars.

    Deterministic churn, not a strategy: the report needs a ledger with winners
    and losers in it, and a rule with no free parameters keeps the assertions
    stable.
    """
    state = {"i": -1}

    def on_bar(ctx: Any) -> None:
        state["i"] += 1
        long_first = (state["i"] // hold) % 2 == 0
        ctx.order_target_pct(tickers[0], 0.45 if long_first else 0.0, reason="swing")
        ctx.order_target_pct(tickers[1], 0.0 if long_first else 0.45, reason="swing")

    return make_strategy(on_bar, name="swing")


def run(**overrides: Any) -> BacktestResult:
    #: Merged, not splatted alongside: an override of a defaulted field (``fills``
    #: is the one every caller reaches for) must replace it, not collide with it.
    fields: dict[str, Any] = {
        "strategy": "<generated:swing>",
        "tickers": list(TICKERS),
        "cash": 100_000.0,
        "limits": dict(LIMITS),
        "fills": {"mode": "next_open", "slippage_bps": 5.0},
    }
    fields.update(overrides)
    cfg = BacktestConfig(**fields)
    view = make_view(TICKERS, BARS)
    return run_backtest(
        cfg, data=view, strategy=swing_strategy(), register=False, journal=False
    )


def render(result: BacktestResult, out: Path | None = None) -> str:
    return render_report(result, out=out).read_text(encoding="utf-8")


@pytest.fixture()
def result() -> BacktestResult:
    return run()


@pytest.fixture()
def rendered(result: BacktestResult) -> str:
    return render(result)


# --- the file itself --------------------------------------------------------------


def test_renders_to_the_default_artifact_path(result: BacktestResult, paths: Any) -> None:
    path = render_report(result)
    assert path == report_path(result.run_id) == paths.runs / result.run_id / "report.html"
    assert path.stat().st_size > 4_000


def test_parses_as_html(rendered: str) -> None:
    parsed = parse_html(rendered)
    assert rendered.lstrip().lower().startswith("<!doctype html>")
    for tag in ("html", "head", "title", "style", "body", "table", "svg", "footer"):
        assert tag in parsed.seen, f"missing <{tag}>"


def test_is_self_contained(rendered: str) -> None:
    """No CDN, no webfont, no remote image -- and specifically no ``xmlns`` on
    the inline SVG, the one place a namespace URL sneaks back in."""
    assert_self_contained(rendered)
    assert "xmlns" not in rendered
    assert "<style>" in rendered  # CSS inline, never linked
    assert "IBM Plex Sans" in rendered and "IBM Plex Mono" in rendered
    assert "@font-face" not in rendered


def test_out_overrides_the_destination(result: BacktestResult, tmp_path: Path) -> None:
    target = tmp_path / "nested" / "custom.html"
    assert render_report(result, out=target) == target
    assert_self_contained(target.read_text(encoding="utf-8"))


def test_out_directory_gets_the_default_filename(result: BacktestResult, tmp_path: Path) -> None:
    assert render_report(result, out=tmp_path) == tmp_path / "report.html"


# --- content -----------------------------------------------------------------------


def test_header_carries_provenance(rendered: str, result: BacktestResult) -> None:
    for token in (result.run_id, "swing", "config hash", "data version", "attempt", "range"):
        assert token in rendered
    assert result.data_version in rendered


def test_key_metrics_appear(rendered: str, result: BacktestResult) -> None:
    m = result.metrics
    for label in ("total return", "cagr", "sharpe", "sortino", "max drawdown", "hit rate"):
        assert label in rendered
    assert f"{m['total_return'] * 100:,.2f}%" in rendered
    assert f"{m['max_drawdown'] * 100:,.2f}%" in rendered
    assert f"{m['sharpe']:,.3f}" in rendered
    assert f"{m['final_equity']:,.2f}" in rendered


def test_every_metric_is_rendered_somewhere(rendered: str, result: BacktestResult) -> None:
    """The report shows the whole of metrics.json, not a curated subset."""
    for key in result.metrics:
        if key == "warnings":  # surfaced as banners, not as a table row
            continue
        assert key.replace("_", " ") in rendered, f"metric {key} dropped from the report"


def test_metric_numbers_are_right_aligned_tabular(rendered: str) -> None:
    assert "tabular-nums" in rendered
    assert 'class="num' in rendered
    assert ".num{text-align:right" in rendered


def test_parameters_are_shown_at_full_precision(tmp_path: Path) -> None:
    result = run(params={"tiny": 0.00025, "flag": True, "n": 12})
    text = render(result, out=tmp_path / "p.html")
    # Scope the truncation check to the parameters panel. Matching "0.000<"
    # across the whole document also catches unrelated metrics that legitimately
    # round to three places.
    panel = text.split("parameters", 1)[1].split("</section>", 1)[0]
    assert "0.00025" in panel and "0.000<" not in panel
    assert "parameters" in text


def test_trade_ledger_lists_trades(rendered: str, result: BacktestResult) -> None:
    assert "trade ledger" in rendered
    assert len(result.trades) >= 5, "the probe stopped producing round trips"
    assert f"{len(result.trades):,d} round trips" in rendered
    for trade in result.trades[: R.MAX_LEDGER_ROWS]:
        assert trade.ticker in rendered
    assert "gain" in rendered and "loss" in rendered  # P&L sign is coloured


def test_ledger_truncation_is_declared(
    result: BacktestResult, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    total = len(result.trades)
    monkeypatch.setattr(R, "MAX_LEDGER_ROWS", 2)
    text = render(result, out=tmp_path / "capped.html")
    assert f"showing 2 of {total:,d} trades" in text
    assert "trades.csv" in text


def test_empty_ledger_says_so(tmp_path: Path) -> None:
    result = run_backtest(
        BacktestConfig(strategy="<generated:flat>", tickers=list(TICKERS), limits=dict(LIMITS)),
        data=make_view(TICKERS, 30),
        strategy=make_strategy(lambda ctx: None, name="flat"),
        register=False,
        journal=False,
    )
    text = render(result, out=tmp_path / "empty.html")
    assert "no closed round trips" in text
    parse_html(text)


def test_charts_share_an_x_axis(rendered: str) -> None:
    """Two stacked SVGs on one viewBox width, so a date lands on the same pixel
    in both -- the reason drawdown is a second chart and not a second y axis."""
    boxes = re.findall(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', rendered)
    assert len(boxes) == 2, boxes
    assert boxes[0][0] == boxes[1][0]
    assert float(boxes[0][1]) > float(boxes[1][1])  # equity is the taller panel
    assert "line-equity" in rendered and "line-dd" in rendered


# --- the benchmark overlay ------------------------------------------------------------


def _seed_benchmark(paths: Any, timestamps: Sequence[Any]) -> None:
    from lab.store.parquet_io import write_bars
    from lab.store.schema import Bar

    frame = make_bars("SPY", len(timestamps), timestamps=list(timestamps), base=400.0, phase=4)
    write_bars(
        Bar(
            event_time=ts,
            knowledge_time=ts,
            source="test",
            ticker="SPY",
            timeframe="1d",
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for ts, row in frame.iterrows()
    )


def test_benchmark_is_overlaid_when_metrics_carry_it(paths: Any, tmp_path: Path) -> None:
    view = make_view(TICKERS, BARS)
    _seed_benchmark(paths, view.timestamps())
    cfg = BacktestConfig(
        strategy="<generated:swing>",
        tickers=list(TICKERS),
        limits=dict(LIMITS),
        benchmark="SPY",
    )
    result = run_backtest(
        cfg, data=view, strategy=swing_strategy(), register=False, journal=False
    )
    assert "benchmark_total_return" in result.metrics

    text = render(result, out=tmp_path / "bench.html")
    assert "benchmark total return" in text
    assert "SPY (rebased)" in text
    assert 'class="line-bench"' in text
    assert_self_contained(text)


def test_missing_benchmark_bars_degrade_quietly(rendered: str) -> None:
    """No benchmark in the config: no overlay, no crash, no empty legend entry.

    Asserted on rendered *markup*, not on the raw text: the inline stylesheet is a
    constant that always carries the ``.line-bench`` and ``.sw-bench`` rules, and a
    rule with no element to style is invisible. What must be absent is the
    benchmark polyline, its legend swatch, and the rebase note beside it.
    """
    assert 'class="line-bench"' not in rendered
    assert 'class="sw sw-bench"' not in rendered
    assert "(rebased)" not in rendered


# --- the caveats ------------------------------------------------------------------------


def test_survivorship_note_always_present(rendered: str) -> None:
    assert "Survivorship bias" in rendered
    assert "point-in-time universe" in rendered


def test_optimistic_fills_warning_absent_for_next_open(rendered: str, result: BacktestResult) -> None:
    assert result.metrics["fill_mode"] == "next_open"
    assert result.metrics["optimistic_fills"] is False
    assert "OPTIMISTIC FILLS" not in rendered


def test_optimistic_fills_warning_loud_for_same_close(tmp_path: Path) -> None:
    result = run(fills={"mode": "same_close", "slippage_bps": 5.0})
    assert result.metrics["optimistic_fills"] is True

    text = render(result, out=tmp_path / "optimistic.html")
    assert "OPTIMISTIC FILLS" in text
    assert OPTIMISTIC_WARNING[:60] in text
    assert text.count("OPTIMISTIC FILLS") >= 2  # banner up top *and* provenance footer
    assert 'class="banner warn"' in text
    assert "same_close" in text
    assert_self_contained(text)
    parse_html(text)


def test_contamination_warning_when_flagged(tmp_path: Path) -> None:
    result = run(contaminated=True)
    assert result.metrics["contaminated"] is True

    text = render(result, out=tmp_path / "contaminated.html")
    assert "TRAINING-DATA CONTAMINATION" in text
    assert CONTAMINATION_WARNING[:60] in text
    parse_html(text)


def test_contamination_absent_by_default(rendered: str) -> None:
    assert "TRAINING-DATA CONTAMINATION" not in rendered


# --- walk-forward shading ------------------------------------------------------------------


def _windows(timestamps: Sequence[Any]) -> list[dict[str, Any]]:
    """Two OOS spans carved out of the middle and the tail of the tape."""
    n = len(timestamps)
    return [
        {
            "index": 0,
            "is_start": timestamps[0].isoformat(),
            "is_end": timestamps[n // 4].isoformat(),
            "oos_start": timestamps[n // 4 + 1].isoformat(),
            "oos_end": timestamps[n // 2].isoformat(),
        },
        {
            "index": 1,
            "is_start": timestamps[n // 2 + 1].isoformat(),
            "is_end": timestamps[3 * n // 4].isoformat(),
            "oos_start": timestamps[3 * n // 4 + 1].isoformat(),
            "oos_end": timestamps[-1].isoformat(),
        },
    ]


def test_oos_windows_are_shaded(tmp_path: Path) -> None:
    view = make_view(TICKERS, BARS)
    result = run(windows=_windows(view.timestamps()))
    text = render(result, out=tmp_path / "wf.html")

    # Two spans, shaded on the equity chart *and* the drawdown chart.
    assert text.count('class="oos"') == 4
    assert "OOS 1" in text and "OOS 2" in text
    assert "out-of-sample (2 windows)" in text
    parse_html(text)


def test_no_windows_says_so_rather_than_shading_nothing(rendered: str) -> None:
    assert 'class="oos"' not in rendered
    assert "no walk-forward windows in this config" in rendered


def test_naive_window_bounds_do_not_crash_a_tz_aware_index(tmp_path: Path) -> None:
    """Walk-forward windows round-trip through JSON as naive strings; comparing
    those against the tz-aware equity index must normalize, not raise."""
    view = make_view(TICKERS, BARS)
    stamps = view.timestamps()
    naive = [
        {
            "oos_start": stamps[10].replace(tzinfo=None).isoformat(),
            "oos_end": stamps[40].replace(tzinfo=None).isoformat(),
        }
    ]
    text = render(run(windows=naive), out=tmp_path / "naive.html")
    assert 'class="oos"' in text


# --- the artifacts-only path -------------------------------------------------------------------


def test_render_from_run_id_string(result: BacktestResult) -> None:
    path = render_report(result.run_id)
    assert path == report_path(result.run_id)
    text = path.read_text(encoding="utf-8")
    parse_html(text)
    assert_self_contained(text)
    assert result.run_id in text


def test_render_from_artifacts_alone(result: BacktestResult, paths: Any) -> None:
    """A run directory the registry has never heard of still renders: the
    artifacts are the archive, the registry is only an index over them."""
    orphan = "archived-20200101T000000-abcdef"
    target = paths.runs / orphan
    target.mkdir(parents=True)
    for name in ("metrics.json", "config.json", "equity.parquet", "trades.csv", "provenance.json"):
        shutil.copy(paths.runs / result.run_id / name, target / name)

    path = render_report(orphan)
    assert path == target / "report.html"
    text = path.read_text(encoding="utf-8")
    parse_html(text)
    assert_self_contained(text)
    assert orphan in text
    assert f"{result.metrics['total_return'] * 100:,.2f}%" in text
    assert f"{len(result.trades):,d} round trips" in text


def test_artifact_and_live_renders_agree(result: BacktestResult, tmp_path: Path) -> None:
    """Same run, two loading paths, one document -- otherwise the archived copy
    quietly says something different from what the runner emitted."""
    live = render(result, out=tmp_path / "live.html")
    archived = render_report(result.run_id, out=tmp_path / "archived.html").read_text(
        encoding="utf-8"
    )
    assert live == archived


def test_unknown_run_id_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="no run artifacts"):
        render_report("nope-20200101T000000-000000")


def test_unsafe_run_id_is_refused() -> None:
    with pytest.raises(ValueError, match="unsafe run_id"):
        render_report("../../etc/passwd")


def test_bad_argument_type_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="BacktestResult or a run_id"):
        render_report(object())  # type: ignore[arg-type]


# --- chart internals worth pinning ----------------------------------------------------------------


def test_downsampling_keeps_the_extremes() -> None:
    values = [0.0] * 5000
    values[1234] = 9.0
    values[4321] = -9.0
    keep = R._downsample(values, 400)
    assert len(keep) <= 400
    assert 1234 in keep and 4321 in keep
    assert keep[0] == 0 and keep[-1] == len(values) - 1
    assert keep == sorted(keep)


def test_downsampling_is_a_no_op_below_the_cap() -> None:
    assert R._downsample([1.0, 2.0, 3.0], 400) == [0, 1, 2]


@pytest.mark.parametrize(
    "lo,hi",
    [(95_000.0, 215_000.0), (-0.4, 0.0), (0.999, 1.001), (0.0, 0.0)],
)
def test_ticks_stay_inside_their_range(lo: float, hi: float) -> None:
    ticks = R._nice_ticks(lo, hi)
    assert ticks
    assert all(lo - 1e-9 <= t <= max(hi, lo) + abs(hi - lo) + 1e-9 for t in ticks)
    assert ticks == sorted(ticks)


def test_open_browser_uses_a_file_uri(
    result: BacktestResult, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import webbrowser

    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url) or True)
    path = render_report(result, out=tmp_path / "opened.html", open_browser=True)
    assert opened == [path.resolve().as_uri()]
    assert opened[0].startswith("file:")
