"""ML-prediksjon av lønnsomheten til skriptene på nettsiden, per tidsintervall (neste dag, uke og måned).

ML endrer ikke skriptene og velger ikke handler. Skriptet handler som motoren på WEEX på valgt lysintervall:
long etter kjøpssignal, short etter salgssignal, snur ved motsatt signal, ordre ved neste bars open, 1× uten
gearing, 0,10 % kostnad per side og funding. Fra skriptets kapitalkurve regnes **fasiten**: avkastningen fra
et tidspunkt og én dag, én uke eller én måned frem. Hvert skript får sin egen ML-modell per intervall og horisont,
som lærer å forutsi fasiten fra markedet og skriptets tilstand ved tidspunktet.

Features beregnes i barer på valgt intervall (som når man bytter intervall i TradingView): ``ret_24h`` er avkastningen
over 24 barer. Skriptets egen historikk (siste uke og måned) og tid siden signal regnes i kalendertid.
Samme funksjoner brukes i treningen (``research/script_ml_check.py``) og i nettsidens live-prediksjon.
Ikke finansiell rådgivning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

import clean_wave_weex
import kivanc_supertrend
import triple_supertrend
from features import FEATURE_COLUMNS, LOOKBACK_HOURS, make_features
from ut_bot_signals import calculate_ut_bot

COST = 0.001
TIMEFRAMES = {"30": "30 minutter", "60": "1 time", "240": "4 timer", "1440": "1 dag"}
HORIZONS = {"Neste dag": 24, "Neste uke": 168, "Neste måned": 720}
EXTRA_COLUMNS = ["position", "script_ret_7d", "script_ret_30d", "hours_since_flip"]
MODEL_COLUMNS = FEATURE_COLUMNS + EXTRA_COLUMNS
# Feature engineering (utvidet sett): uro, skriptets fall fra topp, volatilitetsregime, trendeffektivitet, avstand.
ENGINEERED_COLUMNS = ["signals_7d", "script_drawdown_30d", "vol_ratio", "efficiency_7d", "abs_dist_sma_168h"]
ALL_COLUMNS = MODEL_COLUMNS + ENGINEERED_COLUMNS
FEATURE_SETS = {"grunnleggende": MODEL_COLUMNS, "utvidet": ALL_COLUMNS}
CLEANING = {"av": "ingen rensing", "på": "fjern eksempler nær utfylte barer + klipp ekstremverdier (1.–99. persentil)"}
SCALERS = {"ingen": None, "standard": StandardScaler, "robust": RobustScaler}
NAMES = {"clean_wave": "Clean Wave Matrix PRO", "kivanc_st": "SuperTrend [KivancOzbilgic]",
         "chandelier": "Chandelier Exit (22, 3)", "ut_bot": "UT Bot Alerts",
         "triple_st": "Triple Supertrend [MarkitTick]"}
LIVE_BARS = 2000


class Columns(BaseEstimator, TransformerMixin):
    """Velger featuresettet (kolonnene i ``ALL_COLUMNS``-rekkefølge)."""

    def __init__(self, names=tuple(MODEL_COLUMNS)):
        self.names = names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        index = [ALL_COLUMNS.index(name) for name in self.names]
        return np.asarray(X, dtype=float)[:, index]


class Clipper(BaseEstimator, TransformerMixin):
    """Rensing: klipper hver feature til 1.–99. persentil i treningsdataene (samme grenser brukes live)."""

    def __init__(self, low=1.0, high=99.0):
        self.low, self.high = low, high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_, self.upper_ = np.nanpercentile(X, self.low, axis=0), np.nanpercentile(X, self.high, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def build(model, features="grunnleggende", cleaning="av", scaler="ingen"):
    """Rørledning: featuresett → (rensing) → (skalering) → modell. Brukes likt i trening og live."""
    steps = [Columns(tuple(FEATURE_SETS[features]))]
    if cleaning == "på":
        steps.append(Clipper())
    if SCALERS.get(scaler):
        steps.append(SCALERS[scaler]())
    steps.append(model)
    return make_pipeline(*steps)


class ConstantClassifier:
    """Fast sannsynlighet når treningsdataene bare har én klasse (f.eks. aldri lønnsom over en måned)."""

    def __init__(self, rate=0.0):
        self.rate = float(rate)

    def fit(self, X, y):
        self.rate = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict_proba(self, X):
        rate = min(max(self.rate, 1e-4), 1 - 1e-4)
        return np.tile([1 - rate, rate], (len(X), 1))


def bars_for(hours, minutes):
    return max(int(round(hours * 60 / int(minutes))), 1)


def regularize(bars, minutes):
    """Nøyaktig én rad per intervall (UTC). Manglende barer får forrige close; volum fylles fremover.

    Kolonnen ``filled`` merker utfylte (syntetiske) barer, så rensingen kan holde dem utenfor."""
    df = bars.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    grid = pd.date_range(df.index[0], df.index[-1], freq=f"{int(minutes)}min")
    filled_before = df["filled"].reindex(grid) if "filled" in df else None
    df = df.drop(columns=["filled"], errors="ignore").reindex(grid)
    missing = df["close"].isna().to_numpy()
    df["close"] = df["close"].ffill()
    for column in ("open", "high", "low"):
        df[column] = df[column].fillna(df["close"])
    for column in df.columns.difference(["open", "high", "low", "close"]):
        df[column] = df[column].ffill()
    df["filled"] = missing | (filled_before.fillna(True).to_numpy(dtype=bool) if filled_before is not None else False)
    df.index.name = "open_time"
    return df


def resample(bars, minutes):
    """Timesbarer til lengre intervall (4 t, 1 dag) med vanlig OHLCV-sammenslåing på UTC-grenser."""
    rule = f"{int(minutes)}min"
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "filled" in bars:
        agg["filled"] = "max"
    out = bars.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["close"])


def _ms_frame(bars):
    frame = bars[["open", "high", "low", "close", "volume"]].copy()
    frame["timestamp"] = frame.index.as_unit("ms").asi8
    return frame.reset_index(drop=True)


def positions(bars, minutes):
    """Skriptets mål (+1 long, −1 short, 0 ingen) ved hver bars close, for alle fem skript."""
    minutes = int(minutes)
    frame = _ms_frame(bars)
    out = {
        "clean_wave": clean_wave_weex.compute(frame, minutes=minutes, mode="combined")["position"].to_numpy(),
        "kivanc_st": kivanc_supertrend.compute(frame, minutes=minutes)["position"].to_numpy(),
        "chandelier": kivanc_supertrend.compute_chandelier(frame, minutes=minutes)["position"].to_numpy(),
    }
    ut = calculate_ut_bot(bars[["open", "high", "low", "close"]])
    side, current = np.zeros(len(ut), dtype=int), 0
    for i, (buy, sell) in enumerate(zip(ut["buy"].to_numpy(), ut["sell"].to_numpy())):
        current = 1 if buy else -1 if sell else current
        side[i] = current
    out["ut_bot"] = side
    out["triple_st"] = triple_supertrend.state(triple_supertrend.compute(bars, minutes=minutes))
    return {key: np.asarray(value, dtype=int) for key, value in out.items()}


def bar_returns(bars, position, funding=None):
    """Skriptets avkastning per bar (open t → open t+1).

    Posisjonen fra close t gjelder fra open t+1. Kostnaden trekkes når posisjonen endres (to sider ved snuing),
    og funding trekkes av long og betales til short. Siste bar har ukjent utgang (NaN)."""
    o = bars["open"].to_numpy(dtype=float)
    held = np.r_[0, position[:-1]].astype(float)
    move = np.r_[o[1:] / o[:-1] - 1, np.nan]
    turnover = np.abs(np.diff(np.r_[0.0, held]))
    rates = np.zeros(len(o)) if funding is None else np.asarray(funding, dtype=float)
    return held * move - COST * turnover - held * rates


def equity(returns):
    """Kapital ved hver bars open (start 1)."""
    clean = np.nan_to_num(returns, nan=0.0)
    return np.r_[1.0, np.cumprod(1 + clean)[:-1]]


def sample_features(features, position, equity_no_funding, rows, minutes, close=None):
    """Features ved close av barene ``rows`` (bare kjent historikk): de 23 markedsfeaturene, skriptets tilstand og
    de konstruerte featurene i ``ENGINEERED_COLUMNS``."""
    rows = np.asarray(rows)
    flips = np.flatnonzero(np.r_[True, position[1:] != position[:-1]])
    last_flip = flips[np.searchsorted(flips, rows, side="right") - 1]
    week, month = bars_for(168, minutes), bars_for(720, minutes)

    def past_return(steps):
        before = rows - steps
        out = np.full(len(rows), np.nan)
        ok = before >= 0
        out[ok] = equity_no_funding[rows[ok]] / equity_no_funding[before[ok]] - 1
        return out

    changes = np.r_[0, np.cumsum(position[1:] != position[:-1])]
    signals = np.full(len(rows), np.nan)
    drawdown = np.full(len(rows), np.nan)
    efficiency = np.full(len(rows), np.nan)
    log_close = np.log(np.asarray(close, dtype=float)) if close is not None else None
    step_abs = np.r_[0, np.cumsum(np.abs(np.diff(log_close)))] if log_close is not None else None
    for i, row in enumerate(rows):
        if row >= week:
            signals[i] = changes[row] - changes[row - week]
            if log_close is not None:
                path = step_abs[row] - step_abs[row - week]
                efficiency[i] = abs(log_close[row] - log_close[row - week]) / path if path > 0 else 0.0
        if row >= month:
            drawdown[i] = equity_no_funding[row] / equity_no_funding[row - month:row + 1].max() - 1

    market = features.iloc[rows]
    extra = pd.DataFrame({"position": position[rows], "script_ret_7d": past_return(week),
                          "script_ret_30d": past_return(month),
                          "hours_since_flip": (rows - last_flip) * int(minutes) / 60,
                          "signals_7d": signals, "script_drawdown_30d": drawdown,
                          "vol_ratio": (market["log_rv_24h"] - market["log_rv_168h"]).to_numpy(),
                          "efficiency_7d": efficiency,
                          "abs_dist_sma_168h": market["dist_sma_168h"].abs().to_numpy()},
                         index=features.index[rows])
    return pd.concat([market[FEATURE_COLUMNS], extra], axis=1)[ALL_COLUMNS]


def clean_rows(filled, rows, minutes):
    """Rensing: True der featurevinduet (LOOKBACK_HOURS) ikke berører utfylte barer."""
    filled = np.asarray(filled, dtype=bool)
    cum = np.r_[0, np.cumsum(filled)]
    window = bars_for(LOOKBACK_HOURS, minutes)
    rows = np.asarray(rows)
    start = np.maximum(rows - window, 0)
    return (cum[rows + 1] - cum[start]) == 0


def fetch_live(minutes, count=LIVE_BARS, session=None):
    """Siste avsluttede Binance-lys på intervallet, hentet bakover i sider à 1000."""
    import requests
    session = session or requests.Session()
    interval = triple_supertrend.BINANCE_INTERVALS[str(int(minutes))]
    rows, end = [], None
    while len(rows) < count:
        params = {"symbol": "BTCUSDT", "interval": interval, "limit": 1000}
        if end is not None:
            params["endTime"] = end
        response = session.get(triple_supertrend.BINANCE_URL, params=params, timeout=(3, 10),
                               headers={"User-Agent": "dat158-btc-prognose/1.0"})
        response.raise_for_status()
        page = response.json()
        if not isinstance(page, list) or not page:
            break
        rows = page + rows
        end = int(page[0][0]) - 1
        if len(page) < 1000:
            break
    frame = pd.DataFrame([[int(r[0]), *map(float, r[1:6])] for r in rows],
                         columns=["open_time", "open", "high", "low", "close", "volume"])
    now = pd.Timestamp.now(tz="UTC").value // 1_000_000
    frame = frame[frame.open_time + int(minutes) * 60_000 <= now].drop_duplicates("open_time")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("open_time"), unit="ms", utc=True), name="open_time")
    return frame.sort_index()


def live_inputs(bars, minutes):
    """Siste avsluttede bar: features per skript, og hvilken posisjon skriptet har nå."""
    bars = regularize(bars[["open", "high", "low", "close", "volume"]], minutes)
    features = make_features(bars)
    last = len(bars) - 1
    clean = bool(clean_rows(bars["filled"], [last], minutes)[0])
    out = {}
    for key, position in positions(bars, minutes).items():
        eq = equity(bar_returns(bars, position))
        out[key] = {"x": sample_features(features, position, eq, [last], minutes, bars["close"].to_numpy()),
                    "position": int(position[last]), "clean": clean}
    return bars.index[last], out


def predict(bundle, minutes, key, x):
    """Sjanse for gevinst og forventet avkastning per horisont med skriptets lagrede modell for intervallet."""
    out = {}
    values = x.to_numpy(dtype=float)
    for horizon in HORIZONS:
        entry = bundle["models"].get(f"{int(minutes)}|{key}|{horizon}")
        if entry is None or not np.isfinite(values).all():
            out[horizon] = None
            continue
        out[horizon] = {"p_profit": float(entry["clf"].predict_proba(values)[0, 1]),
                        "expected": float(entry["reg"].predict(values)[0]),
                        "clf_name": entry["clf_name"], "reg_name": entry["reg_name"]}
    return out
