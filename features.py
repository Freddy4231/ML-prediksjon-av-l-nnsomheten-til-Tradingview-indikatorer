"""Felles databehandling og features for trening (train.py) og app (App.py).

Alt som gjør om prisdata til modell-input ligger her, slik at modellen får
nøyaktig de samme features under trening som når appen spår på live data.
Alle features er beregnet KUN fra fortiden, så de lekker ikke fremtidig info.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.pipeline import Pipeline

# Prognosehorisonter i timer: 4 timer, 1 døgn og 3 døgn
HORIZONS = (4, 24, 72)

# Lengste tilbakeblikk som featurene trenger (8 døgn: vol_of_vol_168h er et 7-døgns
# vindu over 24-timers volatilitet). Appen må hente minst så mange timer pluss litt
# ekstra for at alle features skal kunne beregnes. Coinbase gir bare 300 timer, så
# hold LOOKBACK_HOURS + 24 under det.
LOOKBACK_HOURS = 192

# Liten konstant som hindrer log(0) i perioder uten prisbevegelse
EPS = 1e-6

FEATURE_COLUMNS = [
    # Avkastning (log-endring i pris) over ulike tidsvinduer
    "ret_1h", "ret_4h", "ret_12h", "ret_24h", "ret_72h", "ret_168h",
    # Realisert volatilitet, skalert til «per døgn» og log-transformert
    "log_rv_4h", "log_rv_24h", "log_rv_168h",
    # Gjennomsnittlig spenn mellom høy og lav per time siste døgn
    "log_range_24h",
    # Average True Range (snitt av sann svingning per time, inkl. hopp mellom timer)
    "log_atr_14",
    # Bredden på Bollinger-båndet (20 t, ±2σ) relativt til prisen. Smalt bånd = «squeeze»
    "bb_width_20h",
    # Hvor ustabil volatiliteten selv har vært siste uke (vol-of-vol)
    "vol_of_vol_168h",
    # Største timesbevegelse siste døgn relativt til døgnets volatilitet (fanger hopp)
    "max_jump_24h",
    # Teknisk indikator: Relative Strength Index
    "rsi_14",
    # Avstand fra glidende gjennomsnitt (1 døgn og 1 uke)
    "dist_sma_24h", "dist_sma_168h",
    # Volum relativt til eget snitt (uavhengig av hvilken børs dataene er fra)
    "volume_rel_24h", "volume_rel_168h",
    # Tid på døgnet og ukedag, kodet som sirkel så kl 23 ligger nær kl 00
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def regularize_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Sørger for nøyaktig én rad per time (UTC), sortert og uten duplikater.

    Rullerende vinduer teller rader, så hull i tidsserien ville gitt feil
    vindusstørrelser. Manglende timer (f.eks. vedlikehold på børsen) fylles
    med forrige pris. Volumet fylles også fremover, siden 0 i volum ville gitt
    ekstreme verdier i log-volum-featurene. Utfylte timer er ikke ekte data,
    så train.py tar ikke med rader i nærheten av dem (se gap_mask).
    """
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    full_index = pd.date_range(df.index[0], df.index[-1], freq="h")
    df = df.reindex(full_index)
    df["close"] = df["close"].ffill()
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    # Volum (og ordreflyt-kolonnene hvis de finnes) fylles fremover av samme grunn
    for col in df.columns.difference(["open", "high", "low", "close"]):
        df[col] = df[col].ffill()
    df.index.name = "open_time"
    return df


def gap_mask(filled: pd.Series, horizon: int) -> pd.Series:
    """True for timer der featurene (siste LOOKBACK_HOURS) eller målet (de neste `horizon`
    timene) bygger på utfylte timer. Flat utfylt pris gir null volatilitet og
    ekstreme feature-verdier som ikke finnes i virkeligheten."""
    filled = filled.astype(float)
    past = filled.rolling(LOOKBACK_HOURS, min_periods=1).max()
    future = filled.rolling(horizon + 1, min_periods=1).max().shift(-horizon)  # time t ... t+horizon
    return (past > 0) | (future.fillna(1.0) > 0)


def realized_vol(log_returns: pd.Series, window: int) -> pd.Series:
    """Realisert volatilitet: kvadratroten av summen av kvadrerte avkastninger."""
    return np.sqrt((log_returns**2).rolling(window).sum())


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index (Wilders utjevning), verdi mellom 0 og 100."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + avg_gain / avg_loss)
    # Helt flat pris (0/0) gir nøytral RSI
    return out.where(~(avg_gain.eq(0) & avg_loss.eq(0)), 50.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range (Wilders utjevning). Sann svingning er det største av
    høy–lav, |høy − forrige close| og |lav − forrige close|."""
    prev_close = df["close"].shift()
    true_range = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lager features for hver time. Krever kolonnene open, high, low, close, volume
    og en sammenhengende timeindeks i UTC (se regularize_hourly)."""
    log_close = np.log(df["close"])
    r1 = log_close.diff()
    f = pd.DataFrame(index=df.index)

    for h in (1, 4, 12, 24, 72, 168):
        f[f"ret_{h}h"] = log_close.diff(h)

    for h in (4, 24, 168):
        f[f"log_rv_{h}h"] = np.log(realized_vol(r1, h) * np.sqrt(24 / h) + EPS)

    f["log_range_24h"] = np.log(np.log(df["high"] / df["low"]).rolling(24).mean() + EPS)

    # Volatilitetsindikatorer, alle delt på prisen så de er like på tvers av børser
    f["log_atr_14"] = np.log(atr(df, 14) / df["close"] + EPS)
    sma_20 = df["close"].rolling(20).mean()
    f["bb_width_20h"] = np.log(4 * df["close"].rolling(20).std() / sma_20 + EPS)
    rv_24h = realized_vol(r1, 24)
    f["vol_of_vol_168h"] = np.log(rv_24h + EPS).rolling(168).std()
    f["max_jump_24h"] = np.log(r1.abs().rolling(24).max() / (rv_24h + EPS) + EPS)

    f["rsi_14"] = rsi(df["close"], 14)
    f["dist_sma_24h"] = log_close - log_close.rolling(24).mean()
    f["dist_sma_168h"] = log_close - log_close.rolling(168).mean()

    log_volume = np.log(df["volume"].clip(lower=EPS))
    f["volume_rel_24h"] = log_volume - log_volume.rolling(24).mean()
    f["volume_rel_168h"] = log_volume.rolling(24).mean() - log_volume.rolling(168).mean()

    hour = df.index.hour.to_numpy()
    weekday = df.index.dayofweek.to_numpy()
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["dow_sin"] = np.sin(2 * np.pi * weekday / 7)
    f["dow_cos"] = np.cos(2 * np.pi * weekday / 7)
    return f[FEATURE_COLUMNS]


def make_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Det vi vil spå, sett fra hver time t (bruker fremtidige priser, kun til trening):

    - ret: log-avkastning fra t til t + horizon
    - up:  1 hvis prisen er høyere ved t + horizon, ellers 0
    - rv:  realisert volatilitet i timene t+1 ... t+horizon
    """
    log_close = np.log(df["close"])
    r1 = log_close.diff()
    future_ret = log_close.shift(-horizon) - log_close
    targets = pd.DataFrame(index=df.index)
    targets["ret"] = future_ret
    targets["up"] = (future_ret > 0).astype(float).where(future_ret.notna())
    targets["rv"] = realized_vol(r1, horizon).shift(-horizon)
    return targets


def load_ohlcv(path) -> pd.DataFrame:
    """Leser CSV-filen som download_data.py lager. Ordreflyt-kolonnene (trades og
    taker_buy_volume) tas med hvis de finnes i filen."""
    df = pd.read_csv(path, index_col="open_time")
    # Noen få timer i februar 2018 har tidspunkter med sekunder (f.eks. 09:28:14.789), så
    # formatet er blandet. ISO8601 leser begge, og floor("h") legger dem på hele timer.
    df.index = pd.to_datetime(df.index, utc=True, format="ISO8601").floor("h")
    df.index.name = "open_time"
    columns = ["open", "high", "low", "close", "volume"]
    columns += [c for c in ("trades", "taker_buy_volume") if c in df.columns]
    return df[columns].astype(float)


class AddAbsDistance(TransformerMixin, BaseEstimator):
    """Legger til hvor langt prisen er fra 168-timers snittet, uansett retning, målt i
    ukens volatilitet: |dist_sma_168h| / exp(log_rv_168h).

    Langt fra snittet (i begge retninger) betyr ofte høyere volatilitet fremover. En
    lineær modell som Ridge kan ikke lage absoluttverdier selv, mens trebaserte modeller
    kan. Derfor brukes dette bare i Ridge-pipelinen. Ga lavere CV-feil for volatilitet på
    72 t i alle fem foldene (oktober 2026). Ligger her fordi appen må kunne laste modellen.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["abs_dist_sma_168h_rv"] = X["dist_sma_168h"].abs() / np.exp(X["log_rv_168h"])
        return X


class TimeWeighted(RegressorMixin, BaseEstimator):
    """Gir nyere treningsrader større vekt: vekt = 0,5 ** (alder / halveringstid).

    Markedet endrer seg over tid (regimeskifte), så det siste par års data sier mest om
    fremtiden. Alderen regnes fra siste time i treningsdataene, så ingen fremtidig info brukes.
    Krever at X er en DataFrame med tidsindeks. Ligger her, og ikke i train.py, fordi
    appen må kunne laste den lagrede modellen.
    """

    def __init__(self, estimator, half_life_hours):
        self.estimator = estimator
        self.half_life_hours = half_life_hours

    def fit(self, X, y):
        age = np.asarray((X.index[-1] - X.index) / pd.Timedelta(hours=1))
        weight = 0.5 ** (age / self.half_life_hours)
        model = clone(self.estimator)
        if isinstance(model, Pipeline):
            # I en pipeline må vekten sendes til siste steg (selve modellen)
            model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": weight})
        else:
            model.fit(X, y, sample_weight=weight)
        self.estimator_ = model
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class PriorShrunkClassifier(ClassifierMixin, BaseEstimator):
    """Demper retningsmodellens sannsynlighet mot andelen opp i treningsperioden.

    Svake signaler gir lett for bastante sannsynligheter. Fast shrinkage er enkel
    regularisering: (1 - styrke) * p_modell + styrke * p_trening. Ingen fremtidig
    validering eller testdata brukes i fit. Kandidaten må slå baselinen i tidsserie-CV.
    Klassen ligger her slik at appen kan laste modellen uten å importere train.py.
    """

    def __init__(self, estimator, strength=0.5):
        self.estimator = estimator
        self.strength = strength

    def fit(self, X, y):
        if not 0 <= self.strength <= 1:
            raise ValueError("styrke må ligge mellom 0 og 1")
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError("Retningsmodellen krever begge klassene 0 og 1.")
        self.prior_ = float(np.mean(y))
        self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict_proba(self, X):
        p = self.estimator_.predict_proba(X)[:, 1]
        p = (1 - self.strength) * p + self.strength * self.prior_
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
