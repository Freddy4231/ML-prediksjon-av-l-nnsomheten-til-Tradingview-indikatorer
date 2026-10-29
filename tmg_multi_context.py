"""Nye TMG-hypoteser: flere spotvinduer og allerede publisert funding.

Ekte taker-volum og prisbasert CLV holdes adskilt. CLV er bare en proxy.
Reglene endrer mål ved close; forskningsmotoren fyller neste ekte åpning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROFILES = {
    "real_multi_entry": "Inngang: ekte spotpress enig på 4 og 24 timer",
    "proxy_multi_entry": "Inngang: CLV-proxy enig på 4 og 24 timer",
    "real_multi_exit": "Utgang: ekte spotpress motsatt på 4 og 24 timer",
    "proxy_multi_exit": "Utgang: CLV-proxy motsatt på 4 og 24 timer",
    "funding_neutral": "Inngang: siste kjente funding-snitt betaler ikke imot",
    "funding_extreme_exit": "Utgang: kjent funding-snitt over 0,03 % mot posisjonen",
    "real_pressure_funding": "Inngang: ekte spotpress og funding-kostnad høyst 0,01 %",
    "proxy_pressure_funding": "Inngang: CLV-proxy og funding-kostnad høyst 0,01 %",
}


def features(spot, flow):
    if not spot.index.equals(flow.index):
        raise ValueError("Spot og kontekst må ha samme tidsakse")
    out = flow[["spot_buy4", "spot_buy24", "known_funding3"]].copy()
    span = spot.high - spot.low
    clv = ((2 * spot.close - spot.high - spot.low) / span.where(span > 0)).where(span != 0, 0.)
    valid = spot[["open", "high", "low", "close", "volume"]].notna().all(axis=1) & spot.volume.gt(0)
    signed = (clv.clip(-1, 1) * spot.volume).where(valid)
    for n in (4, 24):
        out[f"proxy{n}"] = signed.rolling(n).sum() / spot.volume.where(valid).rolling(n).sum()
    return out.replace([np.inf, -np.inf], np.nan)


def apply(frame, values, profile):
    if profile not in PROFILES or len(frame) != len(values):
        raise ValueError("Ukjent hypotese eller ulik signallengde")
    target = frame.position.to_numpy(int)
    old_buy, old_sell = frame.buy_signal.to_numpy(bool), frame.sell_signal.to_numpy(bool)
    a, b = (values[c].to_numpy(float) for c in
            (("proxy4", "proxy24") if profile.startswith("proxy") else ("spot_buy4", "spot_buy24")))
    midpoint = 0. if profile.startswith("proxy") else .5
    rate = values.known_funding3.to_numpy(float)
    positions, buys, sells, exits, extra, rejected = [np.zeros(len(frame), dtype=t) for t in (int, bool, bool, bool, bool, bool)]
    side = 0
    for i in range(len(frame)):
        if side and target[i] != side:
            side = 0
            exits[i] = True
        if side:
            reverse_pressure = np.isfinite(a[i]) and np.isfinite(b[i]) and side * (a[i] - midpoint) < 0 and side * (b[i] - midpoint) < 0
            expensive = np.isfinite(rate[i]) and side * rate[i] > .0003
            if (profile.endswith("multi_exit") and reverse_pressure) or (profile == "funding_extreme_exit" and expensive):
                side = 0
                exits[i] = extra[i] = True
        direction = 1 if old_buy[i] else -1 if old_sell[i] else 0
        if not side and direction and not extra[i]:
            permit = True
            if profile.endswith("multi_entry") or profile.endswith("pressure_funding"):
                permit = np.isfinite(a[i]) and np.isfinite(b[i]) and direction * (a[i] - midpoint) > 0 and direction * (b[i] - midpoint) > 0
            if profile == "funding_neutral" or profile.endswith("pressure_funding"):
                limit = 0. if profile == "funding_neutral" else .0001
                permit = permit and np.isfinite(rate[i]) and direction * rate[i] <= limit
            if permit:
                side = direction
                buys[i], sells[i] = side == 1, side == -1
            else:
                rejected[i] = True
        positions[i] = side
    out = frame.copy()
    out["position"], out["buy_signal"], out["sell_signal"], out["exit_signal"] = positions, buys, sells, exits
    out["context_exit"], out["context_rejected"] = extra, rejected
    return out
