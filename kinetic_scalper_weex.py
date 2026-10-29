"""WEEX-planer fra den frosne, uavhengige Kinetic-rekonstruksjonen.

Kjører den samme Pine-kilden i sandkassen; ingen alternativ signalport, HTTP,
kontotilgang eller ordre. Basislysene må allerede være avsluttet. Høyere rammer
bygges fra disse samme WEEX-lysene, med originalt volum. En siste, uferdig
HTF-bøtte beholdes som formingsplass: de to kontrollerte security-uttrykkene i
frosset kilde leser kun [1], altså forrige ferdige HTF-lys. Det sikrer stabil
historikk også når en prefiks slutter midt i en HTF-bøtte. Ingen framtidige
basislys brukes. Source-SHA låser denne forutsetningen.

En plan er en signalreferanse, ikke en fyllbekreftelse. Bare fresh=True kan
åpne en ny handel. Børsens faktiske posisjon og ordre avgjør utførelsen.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pine_engine import runner
from pine_engine.compiler import compile_source
from pine_engine.parser import PineError
from pine_engine.runtime import PineTimeout

ROOT = Path(__file__).resolve().parent
PINE_FILE = "tradingview/kinetic_scalper_reconstruction.pine"
PINE_SHA256 = "e757cfc1540c94ee034ea92043e3703b1aef71fd01a89cbaeacb9dd73228a082"
ATTRIBUTION = "Kinetic Scalper · uavhengig rekonstruksjon (ikke originalkode)"
MIN_BARS = 150
DEADLINE_SECONDS = 15.0
_COMPILED: dict[str, Any] = {}
_REQUIRED = ("Long Signal", "Short Signal", "Signal Entry", "Signal SL", "Signal TP1", "Signal TP2",
             "Tracked Direction")


def _prepare(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Valider faktiske, sammenhengende WEEX-lys; fyll aldri hull eller manglende volum."""
    if isinstance(minutes, bool) or not isinstance(minutes, (int, np.integer)) or minutes <= 0:
        raise ValueError("Tidsintervallet må være et positivt helt antall minutter.")
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        raise ValueError("Ingen avsluttede WEEX-lys tilgjengelig.")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("WEEX-lysene må ha UTC-åpningstid som DatetimeIndex.")
    frame = bars.copy()
    index = pd.to_datetime(frame.index, utc=True).as_unit("ms")
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("WEEX-lysene må ha unike, stigende åpningstider.")
    epoch = index.asi8
    step = int(minutes) * 60_000
    if np.any(epoch % step) or np.any(np.diff(epoch) != step):
        raise ValueError("Hull eller feil tidsjustering i WEEX-lysene; ingen ny plan kan brukes.")
    for name in ("open", "high", "low", "close", "volume"):
        if name not in frame:
            raise ValueError(f"WEEX-lysene mangler {name}.")
        frame[name] = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
    values = frame[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("WEEX-lysene inneholder ikke-endelige priser eller volum.")
    o, h, l, c, v = values.T
    if np.any(values[:, :4] <= 0) or np.any(v < 0):
        raise ValueError("WEEX-priser må være positive og volum må være ikke-negativt.")
    if np.any(h < np.maximum(o, c)) or np.any(l > np.minimum(o, c)) or np.any(h < l):
        raise ValueError("Ugyldige OHLC-verdier i WEEX-lysene.")
    for name in ("valid", "available"):
        if name in frame and not frame[name].fillna(False).to_numpy(bool).all():
            raise ValueError("Ugyldige eller utilgjengelige WEEX-lys; ingen ny plan kan brukes.")
    for name in ("synthetic", "filled", "missing"):
        if name in frame and frame[name].fillna(True).to_numpy(bool).any():
            raise ValueError("Syntetiske eller utfylte lys kan ikke brukes til Kinetic-handel.")
    frame.index = pd.DatetimeIndex(index, name="open_time")
    return frame


def _to_bars(frame: pd.DataFrame, minutes: int, symbol: str, mintick: float) -> runner.Bars:
    t = frame.index.as_unit("ms").asi8.astype(np.int64)
    return runner.Bars(symbol, int(minutes), t, t + int(minutes) * 60_000,
                       *(frame[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume")),
                       tickerid=f"WEEX:{symbol}", mintick=mintick, prefix="WEEX",
                       basecurrency=symbol.removesuffix("USDT"))


def _provider(frame: pd.DataFrame, minutes: int, symbol: str, mintick: float):
    """Samme symbol og kilde; siste formingsbøtte har bare allerede avsluttede basislys."""
    cutoff = int(frame.index.as_unit("ms").asi8[-1]) + minutes * 60_000
    cache: dict[int, runner.Bars] = {}

    def provide(requested_symbol, requested_minutes):
        if requested_symbol not in (None, "", "self", symbol, f"WEEX:{symbol}"):
            raise ValueError("Kinetic kan bare bruke samme WEEX-symbol i høyere tidsrammer.")
        target = int(requested_minutes)
        if target < minutes or target % minutes:
            raise ValueError("Høyere tidsramme må være et helt multiplum av WEEX-intervallet.")
        if target == minutes:
            return _to_bars(frame, minutes, symbol, mintick)
        if target in cache:
            return cache[target]
        # Ukelys starter mandag UTC, som TradingViews kryptografer.
        step = target * 60_000
        anchor = 4 * 86_400_000 if target == 7 * 1440 else 0
        epoch = frame.index.as_unit("ms").asi8
        buckets = (epoch - anchor) // step * step + anchor
        group = frame.groupby(buckets, sort=True)
        higher = group.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                           close=("close", "last"), volume=("volume", "sum"))
        counts = group.size()
        starts = higher.index.to_numpy(np.int64)
        complete = (counts.to_numpy() == target // minutes) & (starts + step <= cutoff)
        forming = (starts <= cutoff) & (starts + step > cutoff)
        # Et mangelfullt første, men allerede avsluttet HTF-lys er ingen reell full bar.
        higher = higher.loc[complete | forming].copy()
        if higher.empty:
            raise ValueError("For lite sammenhengende WEEX-historikk til høyere tidsramme.")
        higher.index = pd.DatetimeIndex(pd.to_datetime(higher.index.to_numpy(np.int64), unit="ms", utc=True),
                                       name="open_time")
        cache[target] = _to_bars(higher, target, symbol, mintick)
        return cache[target]
    return provide


def _compiled():
    try:
        source = (ROOT / PINE_FILE).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError("Den frosne Kinetic-kilden er ikke tilgjengelig.") from exc
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if digest != PINE_SHA256:
        raise ValueError("Kinetic-kildens SHA-256 stemmer ikke med den kontrollerte WEEX-profilen.")
    if digest not in _COMPILED:
        compiled = compile_source(source)
        if compiled.is_strategy:
            raise ValueError("WEEX-profilen krever den kontrollerte Kinetic-indikatoren.")
        if len(_COMPILED) > 4:
            _COMPILED.clear()
        _COMPILED[digest] = compiled
    return _COMPILED[digest]


def _plot_frame(result, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    titles = set()
    for (_, title), samples in result.events.series_values.items():
        if title in titles:
            raise ValueError(f"Kinetic-kilden har flere plot-serier med tittelen {title}.")
        titles.add(title)
        values = np.full(len(frame), np.nan)
        for index, value in samples:
            try:
                values[index] = float(value)
            except (ValueError, TypeError, OverflowError):
                values[index] = np.nan
        out[title] = values
    missing = [name for name in _REQUIRED if name not in out]
    if missing:
        raise ValueError("Kinetic-kilden mangler ordreserier: " + ", ".join(missing))
    for title, alias in (("Signal Entry", "entry"), ("Signal SL", "sl"), ("Signal TP1", "tp1"),
                         ("Signal TP2", "tp2"), ("Tracked Direction", "side"),
                         ("Kinetic Pulse", "pulse"), ("Long Score", "long_score"), ("Short Score", "short_score")):
        if title in out:
            out[alias] = out[title]
    out["buy"] = out["Long Signal"].eq(1)
    out["sell"] = out["Short Signal"].eq(1)
    out["long_signal"], out["short_signal"] = out["buy"], out["sell"]
    out["target"] = np.where(out["buy"], 1, np.where(out["sell"], -1, 0))
    out["plan_active"] = out["side"].isin((-1, 1))
    return out


def evaluate(bars: pd.DataFrame, minutes: int = 60, symbol: str = "BTCUSDT", mintick: float = 0.01) -> dict[str, Any]:
    """Beregn en fryst signalplan fra avsluttede WEEX-lys; sender aldri ordre.

    ``fresh`` betyr gyldig signal på akkurat siste lys. ``plan_id``/``signal_close``
    er siste signalets UTC-lukketid i millisekunder. Historiske planer kan vises,
    men gir aldri fresh=True ved oppstart. Datafeil og kildeendringer gir ValueError.
    """
    if not isinstance(symbol, str) or not symbol or ":" in symbol or not symbol.isalnum():
        raise ValueError("WEEX-symbolet må være et enkelt ticker-navn, f.eks. BTCUSDT.")
    symbol = symbol.upper()
    if not np.isfinite(mintick) or mintick <= 0:
        raise ValueError("Minste prissteg må være endelig og positivt.")
    frame = _prepare(bars, minutes)
    empty = {"target": 0, "fresh": False, "signal_close": None, "plan_id": None, "valid": False,
             "entry": None, "sl": None, "tp1": None, "tp2": None, "plan_active": False,
             "frame": frame, "source_sha256": PINE_SHA256}
    if len(frame) < MIN_BARS:
        return {**empty, "reason": f"Kinetic trenger minst {MIN_BARS} sammenhengende WEEX-lys."}
    try:
        compiled = _compiled()
        inputs, sources = runner.resolve_inputs(compiled)
        main = _to_bars(frame, minutes, symbol, float(mintick))
        result, _ = runner.run_with_security(
            compiled, main, inputs=inputs, sources=sources,
            provider=_provider(frame, minutes, symbol, float(mintick)),
            deadline=time.monotonic() + DEADLINE_SECONDS, capture_plots=True)
    except (PineError, PineTimeout) as exc:
        raise ValueError(f"Kinetic kunne ikke beregnes i Pine-sandkassen: {exc}") from None
    frame = _plot_frame(result, frame)
    empty["frame"] = frame
    signals = np.flatnonzero((frame["buy"] ^ frame["sell"]).to_numpy())
    if not len(signals):
        return {**empty, "reason": "Kinetic venter på et nytt, bekreftet signal."}
    k = int(signals[-1])
    row = frame.iloc[k]
    target = 1 if bool(row["buy"]) else -1
    levels = row[["entry", "sl", "tp1", "tp2"]].to_numpy(float)
    valid = bool(np.isfinite(levels).all() and np.all(levels > 0))
    if valid:
        entry, sl, tp1, tp2 = levels
        valid = bool(sl < entry < tp1 < tp2 if target == 1 else tp2 < tp1 < entry < sl)
    plan_id = int(main.TC[k])
    answer = {**empty, "target": target if valid else 0, "fresh": bool(valid and k == len(frame) - 1),
              "signal_close": plan_id, "plan_id": plan_id, "valid": valid,
              "entry": float(levels[0]) if valid else None, "sl": float(levels[1]) if valid else None,
              "tp1": float(levels[2]) if valid else None, "tp2": float(levels[3]) if valid else None,
              "plan_active": bool(valid and frame["side"].iloc[-1] == target),
              "reason": "Kinetic: nytt bekreftet signal." if valid and k == len(frame) - 1 else
                        "Kinetic: historisk signalplan; venter på et nytt signal." if valid else
                        "Kinetic: ugyldige SL/TP-nivåer; ingen ny handel."}
    return answer
