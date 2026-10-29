"""Avgrensede hypoteser om kompresjon og tidligere utgang rundt TMG Dagsbekreftelse.

Alle beslutninger tas ved close og handles tidligst neste open i forskningen.
En beregnet gevinstlinje gjelder fra neste bar, aldri retroaktivt intrabar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROFILES = {
    "range_floor": "Unngå nederste kvartil av historisk prisbredde",
    "range_rising": "Krev økende prisbredde over fire timer",
    "range_volume": "Prisbredde og døgnvolum over tidligere referanse",
    "efficiency24": "Krev retningsbevegelse i siste døgn",
    "daily_rsi_bounds": "Unngå kjøp over dags-RSI70 og short under dags-RSI30",
    "daily_exit": "Lukk når bekreftet dagskontekst går imot",
    "ema_exit_gain": "Etter 2 ATR gevinst: lukk ved brudd på EMA50",
    "vwap_exit_gain": "Etter 2 ATR gevinst: lukk ved brudd på rullerende VWAP",
    "entry_guard": "Etter 2 ATR gevinst: close-linje ved inngang + kostbuffer",
    "trail4": "Etter 2 ATR gevinst: 4 ATR fra beste close",
    "trail6": "Etter 3 ATR gevinst: 6 ATR fra beste close",
}


def features(bars, frame, context):
    if len(frame) != len(bars) or not bars.index.equals(context.index):
        raise ValueError("Pris, grunnsignaler og dagsdata må ha samme tidsakse")
    c = bars.close.astype(float)
    import tmg_trades_weex as tmg
    # Kontekstverdien er gårsdagens close allerede ved midnatt, ikke dagens siste time.
    daily_close = context.daily_close.resample("1D").first()
    daily_rsi = pd.Series(tmg._rsi(daily_close.to_numpy(float), 14), index=daily_close.index)
    daily_rsi = daily_rsi.where(daily_close.notna().rolling(15).sum() == 15)
    width = 4 * c.rolling(20).std(ddof=0) / c.rolling(20).mean()
    result = pd.DataFrame({"width": width, "width4": width.shift(4),
                           "width_q25": width.shift(1).rolling(720).quantile(.25),
                           "volume24": bars.volume.rolling(24).sum(),
                           "volume_reference": bars.volume.shift(24).rolling(720).sum() / 30,
                           "efficiency24": c.diff(24).abs() / c.diff().abs().rolling(24).sum(),
                           "atr": frame.atr.to_numpy(float), "ema": frame.ema50.to_numpy(float),
                           "vwap": frame.vwap.to_numpy(float),
                           "daily_close": context.daily_close, "daily50": context.daily_sma50,
                           "daily_rsi": daily_rsi.reindex(bars.index, method="ffill")}, index=bars.index)
    return result.replace([np.inf, -np.inf], np.nan)


def allowed(values, profile):
    if profile not in PROFILES:
        raise ValueError("Ukjent TMG-tillegg")
    permit = pd.Series(True, index=values.index)
    if profile in ("range_floor", "range_volume"):
        permit = values.width >= values.width_q25
    elif profile == "range_rising":
        permit = values.width > values.width4
    elif profile == "efficiency24":
        permit = values.efficiency24 >= .25
    if profile == "range_volume":
        permit &= values.volume24 >= values.volume_reference
    return permit.fillna(False)


def apply(frame, values, profile):
    """Ingen økt gearing, gjeninntreden på gamle signaler eller fyll på beskyttelseslinjen."""
    permit = allowed(values, profile).to_numpy(bool)
    if len(frame) != len(values):
        raise ValueError("Ulik lengde på kontekst og signaler")
    n = len(frame)
    target = frame.position.to_numpy(int)
    old_buy, old_sell = frame.buy_signal.to_numpy(bool), frame.sell_signal.to_numpy(bool)
    close = frame.close.to_numpy(float)
    atr, ema, vwap, dc, dm = (values[k].to_numpy(float) for k in ("atr", "ema", "vwap", "daily_close", "daily50"))
    rsi = values.daily_rsi.to_numpy(float)
    positions, buys, sells, exits = np.zeros(n, int), np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    guards, extra, armed_series = np.full(n, np.nan), np.zeros(n, bool), np.zeros(n, bool)
    side, entry, entry_atr, peak, guard, armed = 0, np.nan, np.nan, np.nan, np.nan, False
    for i in range(n):
        if side and target[i] != side:
            side = 0
            exits[i] = True
        if side:
            breached = np.isfinite(guard) and side * (close[i] - guard) <= 0
            if profile == "daily_exit":
                breached = breached or (np.isfinite(dc[i]) and np.isfinite(dm[i]) and side * (dc[i] - dm[i]) <= 0)
            elif armed and profile in ("ema_exit_gain", "vwap_exit_gain"):
                line = ema[i] if profile == "ema_exit_gain" else vwap[i]
                breached = breached or (np.isfinite(line) and side * (close[i] - line) <= 0)
            if breached:
                side = 0
                exits[i] = extra[i] = True
                guard = np.nan
            else:
                peak = max(peak, close[i]) if side == 1 else min(peak, close[i])
                trigger = 3. if profile == "trail6" else 2.
                armed = armed or side * (peak - entry) >= trigger * entry_atr
                proposed = np.nan
                if armed and profile == "entry_guard":
                    proposed = entry * (1 + side * .003)
                elif armed and profile in ("trail4", "trail6") and np.isfinite(atr[i]):
                    proposed = peak - side * (4. if profile == "trail4" else 6.) * atr[i]
                if np.isfinite(proposed):
                    guard = proposed if not np.isfinite(guard) else max(guard, proposed) if side == 1 else min(guard, proposed)
        if not side and not extra[i] and permit[i] and np.isfinite(atr[i]) and atr[i] > 0:
            direction = 1 if old_buy[i] else -1 if old_sell[i] else 0
            if profile == "daily_rsi_bounds" and (not np.isfinite(rsi[i]) or (direction == 1 and rsi[i] >= 70) or (direction == -1 and rsi[i] <= 30)):
                direction = 0
            if direction:
                side, entry, entry_atr, peak, guard, armed = direction, close[i], atr[i], close[i], np.nan, False
                buys[i], sells[i] = side == 1, side == -1
        positions[i], guards[i], armed_series[i] = side, guard if side else np.nan, armed if side else False
    out = frame.copy()
    out["position"], out["buy_signal"], out["sell_signal"], out["exit_signal"] = positions, buys, sells, exits
    out["extra_exit"], out["protection_line"], out["profit_armed"] = extra, guards, armed_series
    out["extension_rejected"] = (old_buy | old_sell) & ~(buys | sells)
    return out
