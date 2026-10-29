"""Triple Supertrend Confluence [MarkitTick], portert fra Pine v6 til Python.

Original: «Triple Supertrend Confluence [MarkitTick]» © MarkitTick, lisensiert under
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(https://creativecommons.org/licenses/by-nc-sa/4.0/). Denne porten er et avledet verk med
samme lisens: kreditering beholdes, ingen kommersiell bruk, og endringer deles på samme vilkår.
Originalkoden ligger uendret i ``tradingview/triple_supertrend_confluence.pine``.

Porten følger skriptet linje for linje på avsluttede barer (``barstate.isconfirmed``):

- Hoved-Supertrend på kilden (hl2 eller valgt adaptivt filter), ATR 10 og adaptiv faktor
  1,5–4,5 etter ATR sin persentilrang over 100 barer (fast 3,0 når adaptiv faktor er av).
- Rask (2,0 / 7) og treg (4,0 / 14) Supertrend til flertallskravet («Triple Consensus»).
- Filtre: ADX ≥ 20 (DMI 14), konsensus ≥ 2/3, valgfritt høyere intervall (4 t, forrige
  avsluttede lys), volum og «cooldown».
- **BULL** når hovedtrenden snur opp og alle filtre godkjenner baren; **BEAR** tilsvarende ned.
- Handelsnivåer ved signal: inngang = close, SL = close ∓ 1,5 × ATR, TP1–3 = 1, 2 og 3 R.

Retning følger Pine sin ``ta.supertrend``-konvensjon: −1 = opptrend, +1 = nedtrend.
Rene grafikkvalg (farger, etiketter, «Lock Signal», tabellplassering) påvirker ikke signalene.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from research.clean_wave import pine_atr, pine_rma

ATTRIBUTION = ("Triple Supertrend Confluence [MarkitTick] · © MarkitTick · CC BY-NC-SA 4.0 "
               "(https://creativecommons.org/licenses/by-nc-sa/4.0/)")
PINE_FILE = "tradingview/triple_supertrend_confluence.pine"
FILTER_TYPES = ("None", "SMA", "EMA", "RMA", "Double WMA", "Triple VWMA", "HMA", "LLAMA", "Kalman Filter")
DEFAULTS = {
    "atr_period": 10, "factor": 3.0, "use_adaptive_factor": True, "factor_min": 1.5, "factor_max": 4.5,
    "factor_rank_len": 100, "use_adx_filter": True, "adx_threshold": 20.0, "adx_len": 14,
    "adaptive_filter_type": "None", "adaptive_filter_len": 20, "use_htf": False, "htf_minutes": 240,
    "use_volume_filter": False, "volume_len": 20, "volume_mult": 1.2, "use_cooldown": False, "cooldown_bars": 10,
    "use_consensus": True, "factor_fast": 2.0, "period_fast": 7, "factor_slow": 4.0, "period_slow": 14,
    "sl_atr_mult": 1.5, "tp1_rr": 1.0, "tp2_rr": 2.0, "tp3_rr": 3.0,
}


# ---------------------------------------------------------------- Pine-hjelpere

def nz(value, replacement=0.0):
    return replacement if value is None or (isinstance(value, float) and math.isnan(value)) else value


def sma(values, length):
    return pd.Series(values, dtype=float).rolling(length).mean().to_numpy()


def ema(values, length):
    """``ta.ema``: alpha 2/(n+1), startverdi = første tilgjengelige verdi."""
    values = np.asarray(values, dtype=float)
    out, prev, alpha = np.full(len(values), np.nan), np.nan, 2.0 / (length + 1)
    for t, x in enumerate(values):
        if not np.isnan(x):
            prev = x if np.isnan(prev) else alpha * x + (1 - alpha) * prev
        out[t] = prev
    return out


def wma(values, length):
    """``ta.wma``: vekter 1..n, nyeste verdi får vekt n."""
    weights = np.arange(1, length + 1, dtype=float)
    return pd.Series(values, dtype=float).rolling(length).apply(lambda w: np.dot(w, weights) / weights.sum(),
                                                                 raw=True).to_numpy()


def vwma(values, volume, length):
    values, volume = np.asarray(values, dtype=float), np.asarray(volume, dtype=float)
    return sma(values * volume, length) / sma(volume, length)


def hma(values, length):
    half = wma(values, length // 2)
    full = wma(values, length)
    return wma(2 * half - full, int(math.floor(math.sqrt(length))))


def llama(values, length):
    values = np.asarray(values, dtype=float)
    slope = (values - np.r_[np.full(length, np.nan), values[:-length]]) / length
    return sma(values, length) + slope * (length / 2)


def kalman(values, length):
    est, err, q, r = np.nan, 1.0, 1.0 / length, 1.0
    out = np.full(len(values), np.nan)
    for t, x in enumerate(np.asarray(values, dtype=float)):
        est = x if np.isnan(est) else est
        pred = err + q
        gain = pred / (pred + r)
        est = est + gain * (x - est)
        err = (1 - gain) * pred
        out[t] = est
    return out


def true_range(high, low, close, handle_na):
    previous = np.r_[np.nan, close[:-1]]
    tr = np.maximum.reduce([high - low, np.abs(high - previous), np.abs(low - previous)])
    tr[0] = high[0] - low[0] if handle_na else np.nan
    return tr


def fixnan(values):
    return pd.Series(values, dtype=float).ffill().to_numpy()


def dmi(high, low, close, di_length, adx_smoothing):
    """``ta.dmi``: (+DI, −DI, ADX) med Wilder-utjevning; ``ta.tr`` uten NaN-håndtering."""
    up = np.r_[np.nan, np.diff(high)]
    down = -np.r_[np.nan, np.diff(low)]
    plus_dm = np.where(np.isnan(up), np.nan, np.where((up > down) & (up > 0), up, 0.0))
    minus_dm = np.where(np.isnan(down), np.nan, np.where((down > up) & (down > 0), down, 0.0))
    trur = pine_rma(true_range(high, low, close, handle_na=False), di_length)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus = fixnan(100 * pine_rma(plus_dm, di_length) / trur)
        minus = fixnan(100 * pine_rma(minus_dm, di_length) / trur)
        total = plus + minus
        adx = 100 * pine_rma(np.abs(plus - minus) / np.where(total == 0, 1, total), adx_smoothing)
    return plus, minus, adx


def percentrank(values, length):
    """``ta.percentrank``: andel av de forrige ``length`` verdiene som er ≤ dagens verdi (0–100)."""
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    for t in range(length, len(values)):
        window = values[t - length:t]
        if np.isfinite(window).all() and np.isfinite(values[t]):
            out[t] = (window <= values[t]).sum() / length * 100
    return out


def supertrend(src, high, low, close, factor, period):
    """``f_supertrend`` fra skriptet. Returnerer (linje, retning) der −1 = opp og +1 = ned."""
    n = len(close)
    atr = pine_atr(high, low, close, period)
    factor = np.broadcast_to(np.asarray(factor, dtype=float), (n,))
    raw_upper, raw_lower = src + factor * atr, src - factor * atr
    upper, lower = np.full(n, np.nan), np.full(n, np.nan)
    line, direction = np.full(n, np.nan), np.ones(n, dtype=int)
    for t in range(n):
        prev_lower = nz(float(lower[t - 1])) if t else 0.0
        prev_upper = nz(float(upper[t - 1])) if t else 0.0
        close_prev = close[t - 1] if t else np.nan
        lo, up = raw_lower[t], raw_upper[t]
        # Sammenligninger med NaN er usanne, som i Pine.
        lower[t] = lo if (lo > prev_lower or close_prev < prev_lower) else prev_lower
        upper[t] = up if (up < prev_upper or close_prev > prev_upper) else prev_upper
        if t == 0 or np.isnan(atr[t - 1]):
            d = 1
        elif line[t - 1] == prev_upper:
            d = -1 if close[t] > upper[t] else 1
        else:
            d = 1 if close[t] < lower[t] else -1
        direction[t] = d
        line[t] = lower[t] if d == -1 else upper[t]
    return line, direction


def source(bars, kind, length):
    hl2 = ((bars["high"] + bars["low"]) / 2).to_numpy(dtype=float)
    volume = bars["volume"].to_numpy(dtype=float) if "volume" in bars else np.ones(len(bars))
    return {"None": lambda: hl2, "SMA": lambda: sma(hl2, length), "EMA": lambda: ema(hl2, length),
            "RMA": lambda: pine_rma(hl2, length), "Double WMA": lambda: wma(wma(hl2, length), length),
            "Triple VWMA": lambda: vwma(vwma(vwma(hl2, volume, length), volume, length), volume, length),
            "HMA": lambda: hma(hl2, length), "LLAMA": lambda: llama(hl2, length),
            "Kalman Filter": lambda: kalman(hl2, length)}[kind]()


def htf_direction(bars, settings, minutes=60):
    """``request.security(..., f_htfDirection()[1], lookahead_on)``: forrige avsluttede HTF-lys.

    Er grafens intervall like langt eller lengre enn HTF, gir TradingView grafens egen forrige bar;
    da brukes Supertrend på grafens lys med ett lys forsinkelse.
    """
    htf = int(settings["htf_minutes"])
    if htf <= int(minutes):
        h, l, c = (bars[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
        _, direction = supertrend((h + l) / 2, h, l, c, settings["factor"], settings["atr_period"])
        return np.r_[np.nan, direction[:-1].astype(float)]
    per_group = htf // int(minutes)
    groups = bars.index.floor(f"{htf}min")
    frame = bars.groupby(groups).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                     close=("close", "last"), count=("close", "size"))
    complete = frame[frame["count"] == per_group]
    h, l, c = (complete[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    _, direction = supertrend((h + l) / 2, h, l, c, settings["factor"], settings["atr_period"])
    known = pd.Series(direction.astype(float), index=complete.index + pd.Timedelta(minutes=htf))
    # Lyset som pågår ser forrige avsluttede HTF-lys; manglende HTF-lys gir NaN (filteret avviser).
    return known.reindex(pd.DatetimeIndex(groups)).to_numpy()


def compute(bars, minutes=60, **overrides):
    """Hele indikatoren per avsluttet bar. ``bars`` har open/high/low/close/volume og UTC-indeks."""
    s = {**DEFAULTS, **overrides}
    if s["adaptive_filter_type"] not in FILTER_TYPES:
        raise ValueError("Ukjent adaptivt filter.")
    o, h, l, c = (bars[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    volume = bars["volume"].to_numpy(dtype=float) if "volume" in bars else np.full(len(c), np.nan)
    n = len(c)
    src = source(bars, s["adaptive_filter_type"], s["adaptive_filter_len"])
    _, _, adx = dmi(h, l, c, s["adx_len"], s["adx_len"])
    adx_pass = ~np.full(n, s["use_adx_filter"]) | (adx >= s["adx_threshold"])
    atr = pine_atr(h, l, c, s["atr_period"])
    rank = percentrank(atr, s["factor_rank_len"])
    if s["use_adaptive_factor"]:
        factor = np.where(np.isnan(rank), s["factor"], s["factor_min"] + (s["factor_max"] - s["factor_min"]) * rank / 100)
    else:
        factor = np.full(n, s["factor"])
    line, direction = supertrend(src, h, l, c, factor, s["atr_period"])
    _, fast = supertrend(src, h, l, c, s["factor_fast"], s["period_fast"])
    _, slow = supertrend(src, h, l, c, s["factor_slow"], s["period_slow"])
    agree = (direction == fast).astype(int) + (direction == slow).astype(int) + 1
    consensus_pass = ~np.full(n, s["use_consensus"]) | (agree >= 2)
    htf = htf_direction(bars, s, minutes) if s["use_htf"] else np.full(n, np.nan)
    htf_pass = ~np.full(n, s["use_htf"]) | np.where(direction == -1, htf == -1, htf == 1)
    volume_sma = sma(volume, s["volume_len"])
    has_volume = np.isfinite(volume) & (volume > 0)
    with np.errstate(invalid="ignore"):
        volume_pass = ~np.full(n, s["use_volume_filter"]) | ~has_volume | (volume >= volume_sma * s["volume_mult"])
    flip = np.r_[False, direction[1:] != direction[:-1]]
    long_active, short_active = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    last_flip, long_end, short_end = 0, None, None
    for t in range(n):
        if t >= 2 and direction[t - 1] != direction[t - 2]:
            if (t - 1) - last_flip <= s["cooldown_bars"]:
                if direction[t - 2] == -1:
                    long_end = (t - 1) + s["cooldown_bars"]
                else:
                    short_end = (t - 1) + s["cooldown_bars"]
            last_flip = t - 1
        long_active[t] = s["use_cooldown"] and long_end is not None and t <= long_end
        short_active[t] = s["use_cooldown"] and short_end is not None and t <= short_end
    cooldown_pass = np.where(direction == -1, ~long_active, ~short_active)
    confirmed = adx_pass & consensus_pass & htf_pass & volume_pass & cooldown_pass
    bull = flip & (direction == -1) & confirmed
    bear = flip & (direction == 1) & confirmed
    sl_long, sl_short = c - atr * s["sl_atr_mult"], c + atr * s["sl_atr_mult"]
    frame = pd.DataFrame({
        "open": o, "high": h, "low": l, "close": c, "supertrend": line, "direction": direction,
        "dir_fast": fast, "dir_slow": slow, "agree": agree, "adx": adx, "atr": atr, "factor": factor,
        "htf_direction": htf, "adx_pass": adx_pass, "consensus_pass": consensus_pass, "htf_pass": htf_pass,
        "volume_pass": volume_pass, "cooldown_long": long_active, "cooldown_short": short_active,
        "confirmed": confirmed, "flip": flip, "bull": bull, "bear": bear,
        "sl_long": sl_long, "sl_short": sl_short,
    }, index=bars.index)
    for k, key in enumerate(("tp1_rr", "tp2_rr", "tp3_rr"), 1):
        frame[f"tp{k}_long"] = c + (c - sl_long) * s[key]
        frame[f"tp{k}_short"] = c - (sl_short - c) * s[key]
    return frame


def last_levels(frame):
    """Nivåene fra siste BULL/BEAR og om SL/TP er truffet etterpå (som skriptets låser)."""
    signals = np.flatnonzero(frame["bull"].to_numpy() | frame["bear"].to_numpy())
    if not len(signals):
        return None
    k = int(signals[-1])
    row = frame.iloc[k]
    side = "long" if row["bull"] else "short"
    levels = {"side": side, "bar": frame.index[k], "entry": float(row["close"]),
              "sl": float(row[f"sl_{side}"]), **{f"tp{i}": float(row[f"tp{i}_{side}"]) for i in (1, 2, 3)}}
    after = frame.iloc[k + 1:]
    hit = {}
    for name in ("sl", "tp1", "tp2", "tp3"):
        level = levels[name]
        if side == "long":
            touched = after["low"] <= level if name == "sl" else after["high"] >= level
        else:
            touched = after["high"] >= level if name == "sl" else after["low"] <= level
        hit[name] = bool(touched.any())
    levels["hit"] = hit
    return levels


def state(frame):
    """Målposisjon etter skriptets varsler: long etter BULL (lukk short), short etter BEAR (lukk long)."""
    out, current = np.zeros(len(frame), dtype=int), 0
    for t, (bull, bear) in enumerate(zip(frame["bull"].to_numpy(), frame["bear"].to_numpy())):
        current = 1 if bull else -1 if bear else current
        out[t] = current
    return out


def prepare(bars, gap_bars=200, minutes=60):
    """Sammenhengende akse på intervallet for live-data. Manglende barer får forrige close og null volum.

    Returnerer (barer, om et hull ligger blant de siste ``gap_bars`` barene). Et nylig hull
    stopper nye posisjoner, fordi ATR, ADX og rangeringen da bygger på syntetiske priser.
    """
    grid = pd.date_range(bars.index[0], bars.index[-1], freq=f"{int(minutes)}min", name=bars.index.name)
    full = bars.reindex(grid)
    missing = full["close"].isna().to_numpy()
    close = full["close"].ffill()
    for column in ("open", "high", "low"):
        full[column] = full[column].fillna(close)
    full["close"] = close
    full["volume"] = full["volume"].fillna(0.0) if "volume" in full else 0.0
    return full, bool(missing[-gap_bars:].any())


# ---------------------------------------------------------------- datakilde for signaler

BINANCE_URL = "https://api.binance.com/api/v3/klines"
# TradingView sin vanlige BTC-graf er BINANCE:BTCUSDT. ADX bygger på toppene og bunnene i lysene, så
# signalene kan avvike mellom børser når ADX ligger nær 20. Ordrene går uansett til WEEX.
SIGNAL_SOURCES = {"binance": "Binance spot (som TradingView BINANCE:BTCUSDT)", "weex": "WEEX futures"}


BINANCE_INTERVALS = {"5": "5m", "15": "15m", "30": "30m", "60": "1h", "240": "4h", "1440": "1d"}


def fetch_binance(symbol, now_ms=None, session=None, limit=1000, minutes=60):
    """Avsluttede lys fra Binance spot på valgt intervall (offentlig, ingen nøkler)."""
    import time

    import requests
    client = session or requests
    response = client.get(BINANCE_URL, params={"symbol": symbol, "interval": BINANCE_INTERVALS[str(minutes)],
                                               "limit": limit},
                          headers={"User-Agent": "dat158-btc-prognose/1.0"}, timeout=(3, 10))
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list) or not rows:
        raise ValueError("Binance leverte ingen lys.")
    frame = pd.DataFrame([[int(r[0]), *map(float, r[1:6])] for r in rows],
                         columns=["open_time", "open", "high", "low", "close", "volume"])
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    frame = frame[frame.open_time + int(minutes) * 60_000 <= now].drop_duplicates("open_time").sort_values("open_time")
    frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
    if frame.empty:
        raise ValueError("Ingen avsluttede Binance-lys.")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("open_time"), unit="ms", utc=True), name="open_time")
    return frame
