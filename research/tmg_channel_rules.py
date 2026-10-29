"""Testbare hypoteser fra TMG Trades, adskilt fra hans skjønnsmessige markedsanalyse."""
from __future__ import annotations

import numpy as np
import pandas as pd

PROFILES = {
    "v2": "Uendret TMG V2",
    "spot4": "Ekte spottrykk · 4 timer",
    "spot24": "Ekte spottrykk · 24 timer",
    "agree4": "Spot og futures enige · 4 timer",
    "agree24": "Spot og futures enige · 24 timer",
    "confidence24": "Spottrykk + spotdeltakelse over eget 30-dagersnivå",
    "agree_confidence24": "Spot/futures enige + sterk spotdeltakelse",
    "funding": "Unngå ensidig fundingposisjonering",
    "ny_wait": "Vent gjennom første time etter New York-åpningen",
    "weekend_confidence": "Krev sterk spotdeltakelse på helgesignaler",
}


def real_flow_features(spot, perp, funding):
    """Bare avsluttede timer og publiserte fundingrater. Ingen estimerte kjøpsvolumer.

    Deltakelse måles i samme mynts basevolum på én børs, ikke kanalens globale
    USD-aggregat. Ingen utfylling av manglende volum. Funding må være minst én
    time gammel før den kan brukes som forklaringsvariabel.
    """
    if not spot.index.equals(perp.index):
        raise ValueError("Spot og futures må ha nøyaktig samme tidsakse")
    out = pd.DataFrame(index=perp.index)
    for source_name, source in (("spot", spot), ("perp", perp)):
        volume = source.volume.astype(float)
        buy = source.taker_buy_volume.astype(float)
        valid = (np.isfinite(volume) & np.isfinite(buy) & (volume > 0) & (buy >= 0) &
                 (buy - volume <= 8 * np.spacing(volume.to_numpy())))
        volume, buy = volume.where(valid), buy.clip(upper=volume).where(valid)
        for length in (4, 24):
            out[f"{source_name}_buy{length}"] = buy.rolling(length).sum() / volume.rolling(length).sum()
        out[f"{source_name}_vol24"] = volume.rolling(24).sum()
        # Målepunktet sammenlignes med et tidligere, ikke-overlappende 30-dagersvindu.
        out[f"{source_name}_vol720_prior"] = volume.shift(24).rolling(720).sum()
    out["spot_share24"] = out.spot_vol24 / out.perp_vol24
    out["spot_share720_prior"] = out.spot_vol720_prior / out.perp_vol720_prior
    dates = pd.DatetimeIndex(pd.to_datetime(funding.funding_time, utc=True, format="mixed"))
    known = pd.Series(funding.funding_rate.to_numpy(float), index=dates).sort_index()
    known = known[~known.index.duplicated(keep="last")].rolling(3).mean()
    closes = perp.index + pd.Timedelta(hours=1)
    lookup = closes - pd.Timedelta(hours=1)
    out["known_funding3"] = known.reindex(lookup, method="ffill").to_numpy()
    out["funding_known_from"] = known.index.to_series().reindex(lookup, method="ffill").to_numpy()
    ny = closes.tz_convert("America/New_York")
    minute = ny.hour * 60 + ny.minute
    out["ny_first_hour"] = (ny.dayofweek < 5) & (minute >= 9 * 60 + 30) & (minute < 10 * 60 + 30)
    out["weekend"] = closes.dayofweek >= 5
    return out


def allowed(features, profile):
    if profile not in PROFILES:
        raise ValueError("Ukjent TMG-hypotese")
    yes = pd.Series(True, index=features.index)
    confidence = features.spot_share24 >= features.spot_share720_prior
    if profile == "v2":
        return yes, yes
    if profile in ("spot4", "spot24", "agree4", "agree24", "confidence24", "agree_confidence24"):
        length = 4 if profile.endswith("4") and not profile.endswith("24") else 24
        long = features[f"spot_buy{length}"] > .5
        short = features[f"spot_buy{length}"] < .5
        if profile.startswith("agree"):
            long &= features[f"perp_buy{length}"] > .5
            short &= features[f"perp_buy{length}"] < .5
        if "confidence" in profile:
            long &= confidence
            short &= confidence
        return long.fillna(False), short.fillna(False)
    if profile == "funding":
        # Tersklene er våre faste hypoteser, ikke en oppgitt formel fra TMG.
        return features.known_funding3.le(.0001).fillna(False), features.known_funding3.ge(-.0001).fillna(False)
    if profile == "ny_wait":
        return ~features.ny_first_hour, ~features.ny_first_hour
    condition = ~features.weekend | confidence
    return condition.fillna(False), condition.fillna(False)


def apply_filter(frame, long_allowed, short_allowed):
    """En avvist inngang forblir avvist; utgang og vending følger originalens bekreftede mål."""
    if len(long_allowed) != len(frame) or len(short_allowed) != len(frame):
        raise ValueError("Filter og signaler har ulik lengde")
    out = frame.copy()
    pos = np.zeros(len(frame), dtype=int)
    buy, sell = np.zeros(len(frame), dtype=bool), np.zeros(len(frame), dtype=bool)
    side = 0
    for i, target in enumerate(frame.position):
        if side and int(target) != side:
            side = 0
        if frame.buy_signal.iloc[i] and bool(long_allowed.iloc[i]):
            side = 1
            buy[i] = True
        elif frame.sell_signal.iloc[i] and bool(short_allowed.iloc[i]):
            side = -1
            sell[i] = True
        pos[i] = side
    out["position"], out["buy_signal"], out["sell_signal"] = pos, buy, sell
    return out
