"""Åtte faste videohypoteser etter OI-bekreftelsen; ingen børsordre.

Pris/OI er observerte sammenhenger, ikke identifiserte likvidasjoner eller
market maker-posisjoner. Alle beslutninger tas ved avsluttet timeslys.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PROFILES = {
    "spot4": "OI + ekte spotpress i samme retning · 4 t",
    "spot24": "OI + ekte spotpress i samme retning · 24 t",
    "participation": "OI + sterk spotdeltakelse alle ukedager",
    "breakout24": "OI + close utenfor forrige 24-timersintervall",
    "unwind_reversal": "Vent på fallende OI, motsatt 4 t-pris og 1 t-vending",
    "sweep_reversal": "Vent på retur innenfor 24 t-intervall med fallende OI",
    "exhaustion_exit": "OI-inngang; lukk ved OI-fall og kort prisvending",
    "flow_exit": "OI-inngang; lukk ved motsatt spotpress på 4 og 24 t",
}
REQUIRED = {
    "spot4": ["oi4", "price4", "spot_buy4"],
    "spot24": ["oi4", "price4", "spot_buy24"],
    "participation": ["oi4", "price4", "participation"],
    "breakout24": ["oi4", "price4", "range_high", "range_low", "close"],
    "unwind_reversal": ["oi4", "price4", "price1"],
    "sweep_reversal": ["oi4", "range_high", "range_low", "close", "high", "low"],
    "exhaustion_exit": ["oi4", "price4", "price1"],
    "flow_exit": ["oi4", "price4", "price1", "spot_buy4", "spot_buy24"],
}


def features(bars, spot, original):
    if not bars.index.equals(spot.index) or not bars.index.equals(original.index):
        raise ValueError("Kildene må ha samme timeakse")
    out = original.copy()
    for key in ("close", "high", "low"):
        out[key] = bars[key]
    out["price1"] = bars.close / bars.close.shift(1) - 1
    # Referansen utelater timen som undersøkes; ingen høyrebekreftede pivoter.
    out["range_high"] = bars.high.shift(1).rolling(24).max()
    out["range_low"] = bars.low.shift(1).rolling(24).min()
    sv, pv = spot.volume.where(spot.volume.gt(0)), bars.volume.where(bars.volume.gt(0))
    now = sv.rolling(24).sum() / pv.rolling(24).sum()
    previous = sv.shift(24).rolling(720).sum() / pv.shift(24).rolling(720).sum()
    out["participation"] = now / previous
    return out.replace([np.inf, -np.inf], np.nan)


def decisions(values, profile, side):
    if profile not in PROFILES or side not in (-1, 1):
        raise ValueError("Ukjent hypotese/retning")
    entry = values.oi4.gt(0) & (side * values.price4).gt(0)
    leave = pd.Series(False, index=values.index)
    if profile in ("spot4", "spot24"):
        entry &= (side * (values["spot_buy" + profile[4:]] - .5)).gt(0)
    elif profile == "participation":
        entry &= values.participation.ge(1)
    elif profile == "breakout24":
        entry &= values.close.gt(values.range_high) if side == 1 else values.close.lt(values.range_low)
    elif profile == "unwind_reversal":
        entry = values.oi4.lt(0) & (side * values.price4).lt(0) & (side * values.price1).gt(0)
    elif profile == "sweep_reversal":
        entry = values.oi4.lt(0) & ((values.low.lt(values.range_low) & values.close.gt(values.range_low))
                    if side == 1 else (values.high.gt(values.range_high) & values.close.lt(values.range_high)))
    elif profile == "exhaustion_exit":
        leave = values.oi4.lt(0) & (side * values.price4).gt(0) & (side * values.price1).lt(0)
    elif profile == "flow_exit":
        leave = ((side * (values.spot_buy4 - .5)).lt(0) &
                 (side * (values.spot_buy24 - .5)).lt(0) & (side * values.price1).lt(0))
    return entry.fillna(False).to_numpy(bool), leave.fillna(False).to_numpy(bool)


def apply(frame, values, profile):
    if len(frame) != len(values) or not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError("Signalene krever samsvarende timeakse")
    if values.index.has_duplicates or not values.index.is_monotonic_increasing:
        raise ValueError("Tidsaksen er ugyldig")
    if len(values) > 1 and not (values.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("Regulariser timeaksen og marker hull før bruk")
    permit = {s: decisions(values, profile, s) for s in (-1, 1)}
    target = frame.position.to_numpy(int)
    fresh = np.where(frame.buy_signal, 1, np.where(frame.sell_signal, -1, 0))
    eligible = values.eligible.fillna(False).to_numpy(bool)
    close_times = values.index + pd.Timedelta(hours=1)
    n = len(frame)
    pos, buy, sell, exits, extra, waiting = [np.zeros(n, dtype=t) for t in (int, bool, bool, bool, bool, int)]
    side, pending, expires = 0, 0, None
    for i, when in enumerate(close_times):
        if side and target[i] != side:
            side, exits[i] = 0, True
        if pending and (target[i] != pending or when > expires or not eligible[i]):
            pending = 0
        if side and permit[side][1][i]:
            side, exits[i], extra[i] = 0, True, True
        if fresh[i] and eligible[i]:
            pending, expires = int(fresh[i]), when + pd.Timedelta(hours=6)
        if not side and pending and eligible[i] and not extra[i] and permit[pending][0][i]:
            side, pending = pending, 0
            buy[i], sell[i] = side == 1, side == -1
        if side:
            pending = 0
        pos[i], waiting[i] = side, pending
    out = frame.copy()
    out["position"], out["buy_signal"], out["sell_signal"], out["exit_signal"] = pos, buy, sell, exits
    out["video_exit"], out["waiting_direction"] = extra, waiting
    return out
