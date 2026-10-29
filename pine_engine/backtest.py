"""Lønnsomhetstest av et Pine-skript på valgte intervaller, marked og periode.

Strategier (``strategy()``) handler med sine egne ``strategy.*``-ordre. Indikatorer handler på
signalene de viser: long ved long-signal, short (eller ut, ved «kun long») ved short-signal. Alt
fylles ved neste åpning med samme regnskap på alle intervaller, så resultatene kan sammenlignes.

Dette er en historisk test valgt av brukeren selv. Velger man intervall, marked eller periode etter
resultatet, er det kurvetilpasning; godt resultat her er ingen garanti for framtidig gevinst.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pine_engine import data as D
from pine_engine import runner
from pine_engine.broker import Broker
from pine_engine.compiler import compile_source
from pine_engine.parser import PineError
from pine_engine.runtime import PineTimeout

TIMEFRAMES = {"1 min": 1, "3 min": 3, "5 min": 5, "15 min": 15, "30 min": 30, "45 min": 45, "1 time": 60,
              "2 timer": 120, "4 timer": 240, "1 dag": 1440}
LABELS = {v: k for k, v in TIMEFRAMES.items()}
AUTO = "auto"


@dataclass
class Settings:
    source: str = "binance"
    symbol: str = "BTCUSDT"
    start: str = ""
    end: str = ""
    timeframes: list = field(default_factory=lambda: [1, 5, 30, 45, 60])
    long_only: bool = False
    cost: float = 0.001
    funding: bool = True
    overrides: dict = field(default_factory=dict)
    long_key: str = AUTO
    short_key: str = AUTO
    use_orders: bool = True
    warmup: int = 1500
    seconds: float = 240.0


def analyze(source):
    """Oversett skriptet uten data: type, versjon, input og statiske signaltitler (for nettsiden)."""
    return compile_source(source)


def _daily_sharpe(times, equity):
    series = pd.Series(equity, index=pd.to_datetime(times, unit="ms", utc=True)).resample("1D").last().dropna()
    returns = series.pct_change().dropna()
    if len(returns) < 5 or returns.std() == 0:
        return None
    return float(returns.mean() / returns.std() * math.sqrt(365))


def metrics(broker, bars, first):
    equity = np.asarray(broker.equity[first:], dtype=float)
    equity = np.where(np.isnan(equity), 1.0, equity)
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    trades = broker.trades
    pnls = np.array([t.pnl for t in trades], dtype=float)
    gains, losses = pnls[pnls > 0].sum(), -pnls[pnls < 0].sum()
    bh = float(bars.C[-1] / bars.O[first] - 1)
    span = len(bars) - first
    return {
        "net": float(equity[-1] - 1), "buy_hold": bh, "trades": len(trades),
        "long_trades": int(sum(1 for t in trades if t.side > 0)), "short_trades": int(sum(1 for t in trades if t.side < 0)),
        "win_rate": float((pnls > 0).mean()) if len(pnls) else None,
        "profit_factor": float(gains / losses) if losses > 0 else (None if gains == 0 else math.inf),
        "max_drawdown": float(np.min(equity / peak - 1)), "fees": float(broker.fees),
        "funding": float(broker.funding_paid), "exposure": broker.exposed_bars / span if span else 0.0,
        "sharpe": _daily_sharpe(bars.TC[first:], equity) if equity[-1] > 0.01 else None,
        "bankrupt": bool(broker.bankrupt),
        "avg_trade": float(pnls.mean()) if len(pnls) else None,
    }


def curve(bars, first, equity, points=600):
    equity = np.asarray(equity[first:], dtype=float)
    equity = np.where(np.isnan(equity), 1.0, equity)
    step = max(1, len(equity) // points)
    idx = np.r_[np.arange(0, len(equity), step), len(equity) - 1]
    times = pd.to_datetime(bars.TC[first:][idx], unit="ms", utc=True)
    bh = bars.C[first:][idx] / bars.O[first]
    return {"time": [str(t) for t in times], "equity": equity[idx].tolist(), "buy_hold": bh.tolist()}


def trade_rows(bars, trades, limit=2000):
    """Handler, nyeste først. Resultat i prosent av posisjonens verdi ved inngang (etter gebyr og funding)."""
    rows = []
    for t in reversed(trades[-limit:]):
        notional = t.qty * t.entry_price
        rows.append({"Retning": "Long" if t.side > 0 else "Short",
                     "Inngang (UTC)": str(pd.Timestamp(bars.T[t.entry_bar], unit="ms", tz="UTC"))[:16],
                     "Inngangskurs": float(f"{t.entry_price:.8g}"),
                     "Utgang (UTC)": str(pd.Timestamp(bars.T[t.exit_bar], unit="ms", tz="UTC"))[:16],
                     "Utgangskurs": float(f"{t.exit_price:.8g}"),
                     "Kursendring (%)": round(100 * t.side * (t.exit_price / t.entry_price - 1), 3),
                     "Netto handel (%)": round(100 * t.pnl / notional, 3) if notional > 0 else None,
                     "Gebyr (%)": round(100 * t.fees / notional, 3) if notional > 0 else None,
                     "Funding (%)": round(100 * t.funding / notional, 4) if notional > 0 else None,
                     "Årsak": t.reason, "Ordre-id": t.entry_id})
    return rows


def _provider(settings, base_bars):
    """Data til request.security: høyere tidsramme av samme symbol eller et annet Binance-symbol."""
    def provide(symbol, minutes):
        start = pd.Timestamp(base_bars.T[0], unit="ms", tz="UTC")
        end = pd.Timestamp(base_bars.TC[-1], unit="ms", tz="UTC")
        bars, _, _ = D.load_bars(settings.source, symbol or settings.symbol, minutes, start, end, warmup_bars=200)
        return bars
    return provide


def run_timeframe(compiled, settings, minutes, deadline, progress=None, chosen=None, keep_raw=False):
    """Én tidsramme: hent data, kjør skriptet, lag handler og nøkkeltall (``keep_raw``: også lys og handler)."""
    started = time.monotonic()
    bars, first, base = D.load_bars(settings.source, settings.symbol, minutes, settings.start, settings.end,
                                    settings.warmup, progress)
    funding = None
    funding_note = "ikke trukket"
    if settings.funding and settings.source == "binance":
        found = D.binance_funding(bars.symbol, int(bars.T[first]), int(bars.TC[-1]))
        if found is None:
            funding_note = "paret finnes ikke som Binance-perp"
        else:
            funding = D.funding_per_bar(bars, *found)
            funding_note = f"{len(found[0])} faktiske rater"
    inputs, sources = runner.resolve_inputs(compiled, settings.overrides)
    provider = _provider(settings, bars)
    use_orders = compiled.is_strategy and settings.use_orders
    on_close = bool(compiled.decl.get("args", {}).get("process_orders_on_close"))

    def make_broker():
        return Broker(bars, first, cost=settings.cost, funding=funding, long_only=settings.long_only,
                      on_close=on_close, mintick=bars.mintick)

    result, _ = runner.run_with_security(compiled, bars, inputs=inputs, sources=sources,
                                         broker_factory=make_broker if use_orders else None, provider=provider,
                                         deadline=deadline)
    cands = runner.candidates(result.events)
    used = {}
    if use_orders:
        broker = result.broker
        broker.finish(len(bars) - 1)
        mode = "strategy-ordre"
    else:
        if chosen is not None:
            long_key, short_key = chosen["long"], chosen["short"]
        else:
            long_key = runner.choose(cands, "long") if settings.long_key == AUTO else (settings.long_key or None)
            short_key = runner.choose(cands, "short") if settings.short_key == AUTO else (settings.short_key or None)
        if not long_key and not short_key:
            raise PineError("fant ingen kjøps- eller salgssignaler. Velg signal manuelt, eller bruk plotshape/"
                            "alertcondition med tittel som «Buy»/«Sell» eller strategy.entry().")
        used = {"long": long_key, "short": short_key}
        n = len(bars)
        broker = runner.simulate_signals(bars, first, runner.signal_flags(result.events, long_key, n),
                                         runner.signal_flags(result.events, short_key, n), settings.cost, funding,
                                         settings.long_only, bars.mintick)
        mode = "signaler"
    stats = metrics(broker, bars, first)
    return {"minutes": minutes, "label": LABELS.get(minutes, f"{minutes} min"), "bars": len(bars) - first,
            "warmup": first, "base": base, "start": str(pd.Timestamp(bars.T[first], unit="ms", tz="UTC"))[:16],
            "end": str(pd.Timestamp(bars.TC[-1], unit="ms", tz="UTC"))[:16], "stats": stats,
            "curve": curve(bars, first, broker.equity), "trades": trade_rows(bars, broker.trades), "mode": mode,
            "candidates": cands, "used": used, "funding_note": funding_note, "seconds": time.monotonic() - started,
            "symbol": bars.symbol, **({"raw": {"bars": bars, "first": first, "trades": list(broker.trades)}}
                                      if keep_raw else {})}


def run(source, settings, progress=None):
    """Hele testen: oversett én gang, kjør hver valgt tidsramme. Feil på ett intervall stopper ikke resten."""
    compiled = compile_source(source)
    deadline = time.monotonic() + settings.seconds
    results, errors, chosen = [], [], None
    for k, minutes in enumerate(sorted(set(settings.timeframes))):
        label = LABELS.get(minutes, f"{minutes} min")
        if progress:
            progress(k / max(len(settings.timeframes), 1), desc=f"{label}: henter data og kjører skriptet")
        if time.monotonic() > deadline:
            errors.append((label, "tidsgrensen for testen er nådd"))
            continue
        try:
            item = run_timeframe(compiled, settings, minutes, deadline, chosen=chosen)
        except (PineError, D.DataError) as exc:
            errors.append((label, str(exc)))
            continue
        except PineTimeout as exc:
            errors.append((label, f"{exc} – prøv kortere periode eller færre intervaller"))
            continue
        if chosen is None and item["used"]:
            chosen = item["used"]       # samme signalkilde på alle intervaller
        results.append(item)
    return {"compiled": compiled, "results": results, "errors": errors, "chosen": chosen}
