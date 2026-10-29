"""TMG Dagsbekreftelse: et separat, kausalt inngangsfilter rundt TMG V2.

Bare gårsdagens ferdige UTC-døgn brukes, også på siste time før midnatt.
Dette er vår regel fra kanalens tema, ikke TMGs proprietære indikator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROFILES = {
    "daily50": "Dagsclose over/under SMA50",
    "daily200": "Dagsclose over/under SMA200",
    "daily_cross": "SMA50 over/under SMA200",
    "daily_stack": "Dagsclose, SMA50 og SMA200 i trendens rekkefølge",
    "daily50_weekend": "SMA50 + helgens spotdeltakelse",
    "daily200_weekend": "SMA200 + helgens spotdeltakelse",
    "daily_cross_weekend": "50/200-kryss + helgens spotdeltakelse",
    "daily_stack_weekend": "Samstemt dagstrend + helgens spotdeltakelse",
}


def align_daily(index, daily_close):
    """Døgnclose indeksert ved avslutning blir kjent ved neste timebars åpning."""
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Timeserien må ha en sortert, unik UTC-tidsakse")
    daily = daily_close.copy().sort_index()
    if daily.index.tz is None or not daily.index.is_unique or not daily.index.tz_convert("UTC").equals(daily.index.tz_convert("UTC").normalize()):
        raise ValueError("Døgnclose må ha unike UTC-midnattstider for lukking")
    daily.index = daily.index.tz_convert("UTC")
    daily = daily.reindex(pd.date_range(daily.index[0], daily.index[-1], freq="1D"))
    context = pd.DataFrame({"daily_close": daily,
                            "daily_sma50": daily.rolling(50).mean(),
                            "daily_sma200": daily.rolling(200).mean(),
                            "daily_known_at": daily.index}, index=daily.index)
    # reindex med metode fyller bare mellom lukkede døgn. Et ugyldig døgn forblir NaN.
    out = context.reindex(index.tz_convert("UTC"), method="ffill")
    age = index.tz_convert("UTC") - pd.DatetimeIndex(out.daily_known_at)
    out.loc[(age < pd.Timedelta(0)) | (age >= pd.Timedelta(days=1)),
            ["daily_close", "daily_sma50", "daily_sma200"]] = np.nan
    out.index = index
    return out


def daily_context(bars, real=None):
    """Krev alle 24 ekte timer før et UTC-døgn får bidra til et glidende snitt."""
    if len(bars) < 24:
        raise ValueError("For lite døgnhistorikk")
    index = bars.index
    if index.tz is None or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Ugyldig tidsakse")
    grid = pd.date_range(index[0], index[-1], freq="1h")
    if not grid.equals(index) or not index.equals(index.floor("1h")):
        raise ValueError("Dagsfilteret krever en regelmessig timeserie")
    values = bars.close.where(np.isfinite(bars.close) & (bars.close > 0))
    if real is not None:
        if len(real) != len(bars):
            raise ValueError("Ulik lengde på gyldighet og priser")
        values = values.where(np.asarray(real, dtype=bool))
    groups = values.resample("1D", label="right", closed="left")
    close = groups.last().where(groups.count() == 24)
    return align_daily(index, close)


def allowed(context, profile, flow=None):
    if profile not in PROFILES:
        raise ValueError("Ukjent dagsfilter")
    core = profile.removesuffix("_weekend")
    c, ma50, ma200 = (context[k] for k in ("daily_close", "daily_sma50", "daily_sma200"))
    if core == "daily50":
        long, short = c > ma50, c < ma50
    elif core == "daily200":
        long, short = c > ma200, c < ma200
    elif core == "daily_cross":
        long, short = ma50 > ma200, ma50 < ma200
    else:
        long, short = (c > ma50) & (ma50 > ma200), (c < ma50) & (ma50 < ma200)
    if profile.endswith("_weekend"):
        if flow is None or not flow.index.equals(context.index):
            raise ValueError("Helgefilteret krever sammenfallende spot-/futuresdata")
        confidence = flow.spot_share24 >= flow.spot_share720_prior
        weekend_pass = ~flow.weekend | confidence
        long, short = long & weekend_pass, short & weekend_pass
    return long.fillna(False), short.fillna(False)


def apply_filter(frame, long_allowed, short_allowed, direction="both"):
    """Filtrer bare ferske signaler; en avvist vending lukker fortsatt gammel posisjon."""
    if direction not in ("both", "long_only"):
        raise ValueError("Ukjent retning")
    n = len(frame)
    if n != len(long_allowed) or n != len(short_allowed):
        raise ValueError("Filter og signaler har ulik lengde")
    target = frame.position.to_numpy(int)
    base_buy, base_sell = frame.buy_signal.to_numpy(bool), frame.sell_signal.to_numpy(bool)
    allow_l = np.asarray(pd.Series(long_allowed).fillna(False), dtype=bool)
    allow_s = np.asarray(pd.Series(short_allowed).fillna(False), dtype=bool) & (direction == "both")
    position, buy, sell, exit_ = np.zeros(n, int), np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    side = 0
    for i in range(n):
        if side and target[i] != side:
            side = 0
            exit_[i] = True
        if base_buy[i] and allow_l[i]:
            side, buy[i] = 1, True
        elif base_sell[i] and allow_s[i]:
            side, sell[i] = -1, True
        position[i] = side
    out = frame.copy()
    out["position"], out["buy_signal"], out["sell_signal"], out["exit_signal"] = position, buy, sell, exit_
    out["rejected_buy"], out["rejected_sell"] = base_buy & ~allow_l, base_sell & ~allow_s
    return out
