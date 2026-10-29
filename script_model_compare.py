"""ML-vurdering på /ml-skript: valgt ML-modell forutsier lønnsomheten til valgt teknisk indikator.

Brukeren velger teknisk indikator, tidsintervall, periode (dag/uke/måned) og ML-modell. Modellen trenes på de samme
daglige eksemplene som siden viser under «Se datasettet» (SHA-kontrollert pakke) og gir to prediksjoner:

- **Sannsynlighet for at indikatoren blir lønnsom** i perioden (klassifisering). Treffsikkerheten gjelder tidsintervallet
  brukeren har valgt: andelen perioder der modellens svar (P ≥ 50 % = lønnsom) stemte, mot grunnlinjen.
- **Meravkastning mot vanlig kjøp og hold** (regresjon): indikatorens nettoavkastning minus BTC-kursendringen fra samme
  åpning til samme slutt. Kvadratisk avvik (MSE) måler hvor mye anslaget bommer, mot historisk snitt som grunnlinje.

Ingen statisk trening: modellen trenes på nytt fremover i tid (walk-forward) hver måned, kvartal, halvår eller år, og
bare eksempler der fasiten var kjent ved treningstidspunktet brukes. Alle mål gjelder eksempler modellen ikke hadde
sett. «Automatisk» velger den hyppigste treningen som holder hele beregningen på omtrent 10–20 sekunder på serveren.
De andre modellene kan tas med til sammenligning. Latency, throughput, treningstid og feature importance
(permutasjon, bare 2019–2024) vises for valgt modell. 2025–2026 er kjent kontrollhistorikk, ikke urørt test.
Den lagrede, tunede modellen fra prosjektets trening vises som statisk referanse. Ikke finansiell rådgivning.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import script_profit_ml as spm
import script_profit_predictor as predictor

LR, RF, HGB, XGB, LGBM = ("Logistisk regresjon / Ridge", "Random Forest", "Gradient boosting (HistGB, LightGBM-type)",
                          "XGBoost", "LightGBM")
BASELINE = "Grunnlinje (ingen modell)"
AUTO = "Automatisk · hele beregningen på 10–20 s"
CADENCES = {AUTO: None, "Hver måned": "MS", "Hvert kvartal": "QS", "Hvert halvår": "6MS", "Hvert år": "YS"}
EXPANDING, ROLLING = "Voksende · all kjent historikk", "Rullerende · siste 2 år"
WINDOWS = {EXPANDING: None, ROLLING: pd.Timedelta(days=730)}
FEATURE_SETS = {"Utvidet · 32 egenskaper": spm.ALL_COLUMNS, "Grunnleggende · 27 egenskaper": spm.MODEL_COLUMNS}
HZ = {"Neste dag": "day", "Neste uke": "week", "Neste måned": "month"}
HORIZON_DAYS = {"Neste dag": 1, "Neste uke": 7, "Neste måned": 30}
DEV, CONTROL = "2019–2024", "2025–2026"
EVAL_START = pd.Timestamp("2019-01-01", tz="UTC")
CONTROL_START = pd.Timestamp("2025-01-01", tz="UTC")
MIN_TRAIN = 200                 # som originaltreningen i research/script_ml_check.py
TARGET_SECONDS = (10, 20)
BUDGET_SECONDS = 15.0           # automatisk modus: fortløpende trening ferdig innen dette; resten til viktighet og figurer
HARD_LIMIT_SECONDS = 30.0       # etter dette brukes forrige modell videre (vises i rapporten)
FOLDER = "reports/script_profit_website_2026-10-27"
FEATURE_LABELS = {
    "ret_1h": "Avkastning 1 lys", "ret_4h": "Avkastning 4 lys", "ret_12h": "Avkastning 12 lys",
    "ret_24h": "Avkastning 24 lys", "ret_72h": "Avkastning 72 lys", "ret_168h": "Avkastning 168 lys",
    "log_rv_4h": "Volatilitet 4 lys", "log_rv_24h": "Volatilitet 24 lys", "log_rv_168h": "Volatilitet 168 lys",
    "log_range_24h": "Høy–lav-spenn 24 lys", "log_atr_14": "ATR 14", "bb_width_20h": "Bollinger-bredde 20",
    "vol_of_vol_168h": "Vol-of-vol 168 lys", "max_jump_24h": "Største bevegelse 24 lys", "rsi_14": "RSI 14",
    "dist_sma_24h": "Avstand SMA 24", "dist_sma_168h": "Avstand SMA 168", "volume_rel_24h": "Volum mot snitt 24",
    "volume_rel_168h": "Volum 24 mot 168", "hour_sin": "Tid på døgnet (sin)", "hour_cos": "Tid på døgnet (cos)",
    "dow_sin": "Ukedag (sin)", "dow_cos": "Ukedag (cos)", "position": "Skriptets posisjon",
    "script_ret_7d": "Skriptets avkastning 7 d", "script_ret_30d": "Skriptets avkastning 30 d",
    "hours_since_flip": "Timer siden signal", "signals_7d": "Signaler siste 7 d",
    "script_drawdown_30d": "Skriptets fall fra 30 d-topp", "vol_ratio": "Kort/lang volatilitet",
    "efficiency_7d": "Trendeffektivitet 7 d", "abs_dist_sma_168h": "|Avstand SMA 168|"}
COLORS = {LR: "#3995e8", RF: "#68bca2", HGB: "#d6a461", XGB: "#a77bd8", LGBM: "#e0707a", BASELINE: "#929baa"}
_FILES, _RESULTS, _LIVE = {}, OrderedDict(), {}
_LOCK = threading.Lock()


def _installed(module):
    try:
        __import__(module)
        return True
    except Exception:
        return False


def available_models():
    """XGBoost og LightGBM vises bare når pakkene er installert der nettsiden kjører."""
    return [LR, RF, HGB] + ([XGB] if _installed("xgboost") else []) + ([LGBM] if _installed("lightgbm") else [])


def make_models(name):
    """(klassifikator, regressor) fra samme familie, med faste, moderate innstillinger (ingen tuning per klikk)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == LR:
        from sklearn.linear_model import LogisticRegression, Ridge
        return (make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
                make_pipeline(StandardScaler(), Ridge(alpha=10.0)))
    if name == RF:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        common = dict(n_estimators=60, max_depth=4, min_samples_leaf=20, n_jobs=2, random_state=42)
        return RandomForestClassifier(**common), RandomForestRegressor(max_features=0.33, **common)
    if name == HGB:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        common = dict(max_depth=3, learning_rate=0.05, max_iter=120, min_samples_leaf=30, random_state=42)
        return HistGradientBoostingClassifier(**common), HistGradientBoostingRegressor(**common)
    if name == XGB:
        from xgboost import XGBClassifier, XGBRegressor
        common = dict(n_estimators=120, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      n_jobs=2, random_state=42)
        return XGBClassifier(eval_metric="logloss", **common), XGBRegressor(**common)
    if name == LGBM:
        from lightgbm import LGBMClassifier, LGBMRegressor
        common = dict(n_estimators=120, num_leaves=7, learning_rate=0.05, min_child_samples=30, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8, n_jobs=2, random_state=42, verbose=-1)
        return LGBMClassifier(**common), LGBMRegressor(**common)
    raise ValueError(f"Ukjent modell: {name}")


# ---------------------------------------------------------------- datasett

def _read(path, manifest, **kwargs):
    """CSV fra pakken, bare når SHA-256 stemmer med manifestet. Kontrolleres én gang per filversjon."""
    file = predictor.ROOT / path
    stat = file.stat()
    key = (str(file), stat.st_size, stat.st_mtime_ns, tuple(kwargs.get("usecols") or ()))
    with _LOCK:
        if key not in _FILES:
            if predictor.digest(file) != manifest["dataset_sha256"].get(path):
                raise ValueError("Datasettet er endret. Sammenligningen er sperret.")
            _FILES[key] = pd.read_csv(file, **kwargs)
        return _FILES[key]


def _ns(values):
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def load_table(indicator, minutes, horizon):
    """Én rad per døgn: egenskaper, skriptets avkastning, kjøp og hold over samme åpninger og når fasiten ble kjent."""
    if indicator not in spm.NAMES or minutes not in spm.TIMEFRAMES or horizon not in spm.HORIZONS:
        raise ValueError("Velg en trent indikator, tidsramme og periode.")
    manifest, _ = predictor.load()
    hz = HZ[horizon]
    examples = _read(f"{FOLDER}/{minutes}_{indicator}_examples.csv", manifest)
    bars = _read(f"{FOLDER}/BTCUSDT_{minutes}_bars.csv", manifest, usecols=["open_time", "open"])
    frame = examples.dropna(subset=[f"return_{hz}"])
    decision = pd.to_datetime(frame["decision_time"], utc=True, format="ISO8601")
    known = pd.to_datetime(frame[f"known_{hz}"], utc=True, format="ISO8601")
    opens = pd.Series(bars["open"].to_numpy(float), index=_ns(pd.to_datetime(bars["open_time"], utc=True,
                                                                              format="ISO8601")))
    # Skriptets fasit går fra åpningen ved beslutningen til åpningen da fasiten ble kjent; kjøp og hold likeså.
    entry, exit_ = opens.reindex(_ns(decision)).to_numpy(), opens.reindex(_ns(known)).to_numpy()
    if not (np.isfinite(entry).all() and np.isfinite(exit_).all()):
        raise ValueError("Kjøp og hold kan ikke beregnes for alle eksemplene.")
    table = frame[spm.ALL_COLUMNS].astype(float).reset_index(drop=True)
    table["decision"], table["known"] = decision.to_numpy(), known.to_numpy()
    table["ret"] = frame[f"return_{hz}"].to_numpy(float)
    table["hold"] = exit_ / entry - 1
    table["excess"] = table["ret"] - table["hold"]
    table["win"] = (table["ret"] > 0).astype(int)
    if (_ns(table["known"]) <= _ns(table["decision"])).any():
        raise ValueError("Fasiten må bli kjent etter beslutningen.")
    return table


# ---------------------------------------------------------------- fortløpende trening

@dataclass
class Block:
    rows: np.ndarray
    train: np.ndarray
    clf: object
    reg: object
    base_rate: float
    stale: bool = False


@dataclass
class Run:
    name: str
    proba: np.ndarray
    excess: np.ndarray
    base_rate: np.ndarray
    base_excess: np.ndarray
    blocks: list = field(default_factory=list)
    fit_ms: list = field(default_factory=list)


def _train_rows(decision, known, t, window):
    """Bare eksempler med kjent fasit ved tidspunktet ``t`` (ns). Rullerende vindu teller fra beslutningstiden."""
    mask = known <= t
    if window is not None:
        mask &= decision >= t - window.value
    return np.flatnonzero(mask)


def _fit_pair(name, X, win, excess):
    clf, reg = make_models(name)
    started = time.perf_counter()
    clf = clf.fit(X, win) if len(np.unique(win)) == 2 else None
    reg = reg.fit(X, excess)
    elapsed = (time.perf_counter() - started) * 1000
    # Én tråd ved prediksjon: små kall blir raskere (lavere latency) uten å endre modellen.
    for model in (clf, reg):
        if model is not None and "n_jobs" in model.get_params(deep=False):
            model.set_params(n_jobs=1)
    return clf, reg, elapsed


def _proba(clf, X, rate):
    return np.full(len(X), rate) if clf is None else clf.predict_proba(X)[:, 1]


def refit_points(table, freq):
    return pd.date_range(EVAL_START, table["decision"].max(), freq=freq)


def walk_forward(table, columns, names, points, window, deadline=None, progress=None):
    """Ny trening ved hvert punkt; modellen predikerer eksemplene frem til neste punkt."""
    decision, known = _ns(table["decision"]), _ns(table["known"])
    X = table[columns].to_numpy(float)
    win, excess = table["win"].to_numpy(int), table["excess"].to_numpy(float)
    n = len(table)
    runs = {name: Run(name, np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
            for name in names}
    stamps = list(_ns(points)) + [np.iinfo(np.int64).max]
    for i in range(len(points)):
        rows = np.flatnonzero((decision >= stamps[i]) & (decision < stamps[i + 1]))
        train = _train_rows(decision, known, stamps[i], window)
        if not len(rows) or len(train) < MIN_TRAIN:
            continue
        rate, mean_excess = float(win[train].mean()), float(excess[train].mean())
        for name, run in runs.items():
            if progress:
                progress((i + 1) / len(points), desc=f"Trener {name} · {pd.Timestamp(points[i]):%Y-%m}")
            late = deadline is not None and time.monotonic() > deadline and run.blocks
            if late:                         # tidsgrensen er nådd: forrige modell brukes videre, merkes i rapporten
                last = run.blocks[-1]
                clf, reg, rate_used = last.clf, last.reg, last.base_rate
            else:
                clf, reg, ms = _fit_pair(name, X[train], win[train], excess[train])
                run.fit_ms.append(ms)
                rate_used = rate
            run.proba[rows] = _proba(clf, X[rows], rate_used)
            run.excess[rows] = reg.predict(X[rows])
            run.base_rate[rows], run.base_excess[rows] = rate, mean_excess
            run.blocks.append(Block(rows, train, clf, reg, rate_used, stale=bool(late)))
    return runs


def choose_cadence(table, full_ms, window, elapsed):
    """Hyppigste trening som får plass i budsjettet. Snittvinduet er mindre enn hele datasettet (faktor 0,8)."""
    remaining = max(BUDGET_SECONDS - elapsed, 1.0)
    per_refit = sum(full_ms) / 1000 * (0.6 if window is not None else 0.8)
    for label in ("Hver måned", "Hvert kvartal", "Hvert halvår"):
        if len(refit_points(table, CADENCES[label])) * per_refit <= remaining:
            return label
    return "Hvert år"


# ---------------------------------------------------------------- mål

def _mse_pct(prediction, actual):
    """Kvadratisk avvik i prosentpoeng² (avkastning i prosent)."""
    return float(np.mean((100 * (np.asarray(prediction) - np.asarray(actual))) ** 2))


def evaluate(table, run, mask):
    rows = np.flatnonzero(mask & ~np.isnan(run.proba))
    if not len(rows):
        return None
    from sklearn.metrics import log_loss, roc_auc_score
    y, p, b = table["win"].to_numpy(int)[rows], run.proba[rows], run.base_rate[rows]
    e, pe, be = table["excess"].to_numpy(float)[rows], run.excess[rows], run.base_excess[rows]
    r = table["ret"].to_numpy(float)[rows]
    said = p >= 0.5
    fp, fn = said & (y == 0), ~said & (y == 1)
    return {"n": len(rows), "base_rate": float(y.mean()),
            "accuracy": float(np.mean(said == y)), "baseline_accuracy": float(np.mean((b >= 0.5) == y)),
            "brier": float(np.mean((p - y) ** 2)), "baseline_brier": float(np.mean((b - y) ** 2)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "mse_excess": _mse_pct(pe, e), "baseline_mse_excess": _mse_pct(be, e),
            "fp": int(fp.sum()), "fn": int(fn.sum()),
            "fp_return": float(r[fp].mean()) if fp.any() else None,
            "fn_return": float(r[fn].mean()) if fn.any() else None,
            "baseline_fp": int(((b >= 0.5) & (y == 0)).sum()), "baseline_fn": int(((b < 0.5) & (y == 1)).sum())}


def period_masks(table):
    decision = table["decision"]
    return {DEV: ((decision >= EVAL_START) & (decision < CONTROL_START)).to_numpy(),
            CONTROL: (decision >= CONTROL_START).to_numpy()}


def speed(final, X, repeats=15, batch=2000):
    """Latency: median svartid for én prediksjon (sannsynlighet + meravkastning). Throughput: rader per sekund."""
    clf, reg, rate = final
    row = X[-1:]
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        _proba(clf, row, rate)
        reg.predict(row)
        times.append(time.perf_counter() - started)
    big = np.resize(X, (batch, X.shape[1]))
    best = np.inf
    for _ in range(3):
        started = time.perf_counter()
        _proba(clf, big, rate)
        reg.predict(big)
        best = min(best, time.perf_counter() - started)
    return float(np.median(times) * 1000), float(batch / best) if best > 0 else None


def permutation_importance(table, run, columns, mask, repeats=2, seed=0):
    """Økning i MSE (Brier) når én egenskap stokkes, på eksempler modellen ikke hadde sett (bare ``mask``)."""
    rows = np.flatnonzero(mask & ~np.isnan(run.proba))
    if not len(rows):
        return pd.Series(dtype=float)
    X = table[columns].to_numpy(float)
    y = table["win"].to_numpy(int)
    rng = np.random.default_rng(seed)
    order = {row: i for i, row in enumerate(rows)}
    shuffled = [[rng.permutation(X[rows, j]) for _ in range(repeats)] for j in range(len(columns))]
    loss = np.zeros((len(columns), repeats))
    base = 0.0
    for block in run.blocks:
        mine = block.rows[np.isin(block.rows, rows)]
        if not len(mine):
            continue
        local = np.array([order[r] for r in mine])
        valid, yb = X[mine], y[mine]
        base += np.sum((_proba(block.clf, valid, block.base_rate) - yb) ** 2)
        copies = []
        for j in range(len(columns)):
            for k in range(repeats):
                copy = valid.copy()
                copy[:, j] = shuffled[j][k][local]
                copies.append(copy)
        p = _proba(block.clf, np.vstack(copies), block.base_rate).reshape(len(columns), repeats, len(mine))
        loss += ((p - yb) ** 2).sum(axis=2)
    return pd.Series(loss.mean(axis=1) / len(rows) - base / len(rows), index=columns).sort_values(ascending=False)


def native_importance(clf, columns):
    if clf is None:
        return pd.Series(dtype=float)
    model = clf.steps[-1][1] if hasattr(clf, "steps") else clf
    if hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, float)
    else:
        return pd.Series(dtype=float)
    return pd.Series(values / values.sum() if values.sum() > 0 else values, index=columns).sort_values(ascending=False)


def calibration(table, run, masks, edges=(0.0, 0.4, 0.5, 0.6, 1.0001)):
    """Når modellen sa en gitt sannsynlighet: hvor ofte var skriptet faktisk lønnsomt?"""
    y = table["win"].to_numpy(int)
    out = []
    for period, mask in masks.items():
        for low, high in zip(edges[:-1], edges[1:]):
            rows = mask & (run.proba >= low) & (run.proba < high)
            out.append({"periode": period, "bin": f"{low:.0%}–{min(high, 1):.0%}".replace("%", " %"),
                        "n": int(rows.sum()), "said": float(run.proba[rows].mean()) if rows.any() else None,
                        "actual": float(y[rows].mean()) if rows.any() else None})
    return pd.DataFrame(out)


def rolling_accuracy(table, run, window=180):
    ok = ~np.isnan(run.proba) & (table["decision"] >= EVAL_START).to_numpy()
    y = table["win"].to_numpy(int)[ok]
    hits = pd.Series(((run.proba[ok] >= 0.5) == y).astype(float))
    base = pd.Series(((run.base_rate[ok] >= 0.5) == y).astype(float))
    times = table["decision"][ok]
    return times, hits.rolling(window, min_periods=window).mean(), base.rolling(window, min_periods=window).mean()


# ---------------------------------------------------------------- tekst

def _no(value, digits=2):
    if value is None or not np.isfinite(value):
        return "–"
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",").replace("-", "−")


def _pct(value, digits=1, sign=True):
    if value is None or not np.isfinite(value):
        return "–"
    return f"{value * 100:{'+' if sign else ''}.{digits}f} %".replace(".", ",").replace("-", "−")


def profit_summary(table, masks):
    """Skriptets snittprofitt mot vanlig kjøp og hold per periode (beskrivende, ingen modell)."""
    lines = []
    for period, mask in masks.items():
        if mask.any():
            part = table.loc[mask]
            lines.append(f"| {period} | {_no(len(part), 0)} | {_pct(part.ret.mean(), 2)} | {_pct(part.hold.mean(), 2)} | "
                         f"{_pct(part.excess.mean(), 2)} | {_pct(float((part.excess > 0).mean()), 0, False)} | "
                         f"{_pct(float(part.win.mean()), 0, False)} |")
    return lines


# ---------------------------------------------------------------- hele analysen

def analyse(indicator, minutes, horizon, names, cadence_label, window_label, feature_label, progress=None,
            primary=None):
    """Fortløpende trening og alle mål. ``primary`` er modellen brukeren valgte (feil-prediksjoner, viktighet og
    kalibrering gjelder den). Resultatet mellomlagres per valg, så et nytt klikk svarer med en gang."""
    started = time.monotonic()
    names = [name for name in available_models() if name in (names or [])]
    if not names:
        raise ValueError("Velg en ML-modell.")
    if primary is not None and primary not in names:
        raise ValueError("Ukjent ML-modell.")
    if cadence_label not in CADENCES or window_label not in WINDOWS or feature_label not in FEATURE_SETS:
        raise ValueError("Ukjent valg for trening, vindu eller egenskaper.")
    manifest, _ = predictor.load()
    key = (indicator, minutes, horizon, tuple(names), primary, cadence_label, window_label, feature_label,
           tuple(sorted(manifest["dataset_sha256"].items())))
    with _LOCK:
        if key in _RESULTS:
            _RESULTS.move_to_end(key)
            return _RESULTS[key] | {"cached": True}
    table = load_table(indicator, minutes, horizon)
    columns, window = FEATURE_SETS[feature_label], WINDOWS[window_label]
    X, win, excess = table[columns].to_numpy(float), table["win"].to_numpy(int), table["excess"].to_numpy(float)
    decision, known = _ns(table["decision"]), _ns(table["known"])
    # Endelige modeller: all fasit som er kjent nå. De brukes til dagens prediksjon og til å anslå treningstiden.
    latest = _train_rows(decision, known, known.max(), window)
    final, full_ms = {}, []
    for name in names:
        if progress:
            progress(0.02, desc=f"Trener {name} på all kjent fasit")
        clf, reg, ms = _fit_pair(name, X[latest], win[latest], excess[latest])
        final[name] = (clf, reg, float(win[latest].mean()))
        full_ms.append(ms)
    chosen = cadence_label if CADENCES[cadence_label] else choose_cadence(table, full_ms, window,
                                                                          time.monotonic() - started)
    points = refit_points(table, CADENCES[chosen])
    runs = walk_forward(table, columns, names, points, window, started + HARD_LIMIT_SECONDS, progress)
    masks = period_masks(table)
    reports = {}
    for name, run in runs.items():
        latency, throughput = speed(final[name], X)
        reports[name] = {period: evaluate(table, run, mask) for period, mask in masks.items()}
        reports[name] |= {"latency_ms": latency, "throughput": throughput, "refits": len(run.fit_ms),
                          "stale": sum(block.stale for block in run.blocks),
                          "fit_ms_median": float(np.median(run.fit_ms)) if run.fit_ms else 0.0,
                          "fit_seconds": float(np.sum(run.fit_ms) / 1000)}
    usable = [name for name in names if reports[name].get(DEV)]
    if not usable:
        raise ValueError("For få eksempler med kjent fasit før 2019 til å starte den fortløpende treningen.")
    best = min(usable, key=lambda k: reports[k][DEV]["brier"])
    primary = primary if primary in usable else best
    if progress:
        progress(0.97, desc="Feature importance")
    importance = permutation_importance(table, runs[primary], columns, masks[DEV])
    result = {"indicator": indicator, "minutes": minutes, "horizon": horizon, "names": names, "columns": columns,
              "cadence": chosen, "auto": CADENCES[cadence_label] is None, "window": window_label,
              "features": feature_label, "points": len(points), "table": table, "runs": runs, "reports": reports,
              "best": best, "primary": primary, "final": final, "trained_rows": len(latest),
              "trained_until": pd.Timestamp(table["known"].iloc[latest].max()), "importance": importance,
              "native": native_importance(final[primary][0], columns), "masks": masks,
              "calibration": calibration(table, runs[primary], masks), "seconds": time.monotonic() - started}
    with _LOCK:
        _RESULTS[key] = result
        while len(_RESULTS) > 12:
            _RESULTS.popitem(last=False)
    return result | {"cached": False}


def current_input(indicator, minutes, source):
    """Samme input som «Beregn med ML» bruker. Ferske data hentes høyst én gang per døgn og intervall."""
    if source == predictor.DEMO:
        snapshot = json.loads((predictor.DIRECTORY / "demo_inputs.json").read_text())
        return snapshot["timeframes"][minutes]["scripts"][indicator]
    if source != predictor.LIVE:
        raise ValueError("Ukjent datakilde.")
    day = pd.Timestamp.now(tz="UTC").normalize()
    key = (minutes, indicator, day)
    with _LOCK:
        if key in _LIVE:
            return _LIVE[key]
    item = predictor.live_input(minutes, indicator)
    with _LOCK:
        if len(_LIVE) > 40:
            _LIVE.clear()
        _LIVE[key] = item
    return item


def predict_now(result, item, capital):
    """Hver modell (trent på all kjent fasit) predikerer for valgt input."""
    if pd.Timestamp(item["decision_time"]) < result["trained_until"]:
        raise ValueError("Input er eldre enn treningsgrunnlaget.")
    values = np.asarray(item["values"], dtype=float)
    x = values[[spm.ALL_COLUMNS.index(c) for c in result["columns"]]].reshape(1, -1)
    if not np.isfinite(x).all():
        raise ValueError("Ugyldig modellinput.")
    rows = []
    for name in result["names"]:
        clf, reg, rate = result["final"][name]
        started = time.perf_counter()
        p, excess = float(_proba(clf, x, rate)[0]), float(reg.predict(x)[0])
        rows.append({"name": name, "p": p, "excess": excess, "nok": capital * excess,
                     "ms": (time.perf_counter() - started) * 1000})
    return rows


# ---------------------------------------------------------------- visning

# MSE-kolonnene er tatt ut etter brukerens ønske; MSE brukes fortsatt internt (viktighet, beste modell).
QUALITY_COLUMNS = ["Periode", "Modell", "Eksempler", "Treffsikkerhet", "AUC", "Falske positive", "Falske negative"]
SPEED_COLUMNS = ["Modell", "Nye treninger", "Treningstid per gang (ms)", "Treningstid totalt (s)",
                 "Latency per prediksjon (ms)", "Throughput (prediksjoner/s)"]
SIDE = {1: "long", -1: "short", 0: "flat"}


def _label(result, name):
    return name + (" ★ valgt" if name == result["primary"] else "")


def quality_table(result):
    rows = []
    for period in (DEV, CONTROL):
        first = next((result["reports"][n][period] for n in result["names"] if result["reports"][n].get(period)), None)
        if first is None:
            continue
        rows.append([period, BASELINE, first["n"], _pct(first["baseline_accuracy"], 1, False), "0,500",
                     first["baseline_fp"], first["baseline_fn"]])
        for name in result["names"]:
            rep = result["reports"][name].get(period)
            if rep:
                rows.append([period, _label(result, name), rep["n"], _pct(rep["accuracy"], 1, False),
                             _no(rep["auc"], 3), rep["fp"], rep["fn"]])
    return pd.DataFrame(rows, columns=QUALITY_COLUMNS)


def speed_table(result):
    rows = []
    for name in result["names"]:
        rep = result["reports"][name]
        rows.append([_label(result, name),
                     str(rep["refits"]) + (f" (+{rep['stale']} uten ny trening)" if rep["stale"] else ""),
                     _no(rep["fit_ms_median"], 0), _no(rep["fit_seconds"], 2), _no(rep["latency_ms"], 2),
                     _no(rep["throughput"], 0)])
    return pd.DataFrame(rows, columns=SPEED_COLUMNS)


def prediction_table(result, predictions, horizon, capital):
    rows = []
    for row in predictions:
        rows.append([_label(result, row["name"]), _pct(row["p"], 1, False), _pct(row["excess"], 2),
                     f"{row['nok']:+,.0f} kr".replace(",", " ").replace("-", "−"), _no(row["ms"], 2)])
    return pd.DataFrame(rows, columns=["Modell", f"P(lønnsom) · {horizon.lower()}", "Meravkastning mot kjøp og hold",
                                       f"Meravkastning på {capital:,.0f} kr".replace(",", " "), "Svartid (ms)"])


def drift_figure(result):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                        subplot_titles=("Treffsikkerhet over tid · 180 døgn rullerende",
                                        f"Kalibrering · {result['primary']}"))
    times = None
    for name in result["names"]:
        times, hits, base = rolling_accuracy(result["table"], result["runs"][name])
        width = 2.4 if name == result["primary"] else 1.2
        fig.add_trace(go.Scatter(x=times, y=100 * hits, name=name, line={"color": COLORS.get(name), "width": width}),
                      row=1, col=1)
    if times is not None:
        fig.add_trace(go.Scatter(x=times, y=100 * base, name=BASELINE,
                                 line={"color": COLORS[BASELINE], "dash": "dot"}), row=1, col=1)
    fig.add_vline(x=CONTROL_START, line_dash="dash", line_color="#888", row=1, col=1)
    cal = result["calibration"]
    for period, color in ((DEV, "#68bca2"), (CONTROL, "#d6a461")):
        part = cal[cal.periode == period]
        fig.add_trace(go.Bar(x=part["bin"], y=100 * part["actual"].astype(float), name=f"Faktisk lønnsom · {period}",
                             marker_color=color, text=[f"n={n}" for n in part["n"]], textposition="outside"),
                      row=1, col=2)
    fig.update_yaxes(title_text="% riktige", row=1, col=1)
    fig.update_yaxes(title_text="% faktisk lønnsomme", range=[0, 105], row=1, col=2)
    fig.update_xaxes(title_text="Modellens P(lønnsom)", row=1, col=2)
    fig.update_layout(template="plotly_white", height=420, barmode="group", legend={"orientation": "h", "y": -0.22},
                      margin={"l": 55, "r": 25, "t": 60, "b": 40})
    return fig


def importance_figure(result):
    import plotly.graph_objects as go
    data = result["importance"].head(15)[::-1]
    fig = go.Figure(go.Bar(x=1000 * data.to_numpy(), y=[FEATURE_LABELS.get(f, f) for f in data.index],
                           orientation="h", marker_color=["#68bca2" if v > 0 else "#cf4651" for v in data.to_numpy()]))
    fig.add_vline(x=0, line_color="#888")
    fig.update_layout(template="plotly_white", height=460, margin={"l": 190, "r": 25, "t": 60, "b": 45},
                      title=f"Feature importance · {result['primary']} · 2019–2024",
                      xaxis_title="Økning i MSE (Brier) × 1000 når egenskapen stokkes · positiv = hjelper")
    return fig


def empty():
    import plotly.graph_objects as go
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), go.Figure(), go.Figure()


def stored_reference(indicator, minutes, horizon, item):
    """Den lagrede, tunede modellen fra prosjektets trening på samme input (statisk referanse)."""
    manifest, _ = predictor.load()
    model = predictor.bundle(manifest)["models"][f"{minutes}|{indicator}|{horizon}"]
    x = np.asarray(item["values"], dtype=float).reshape(1, -1)
    return float(model["clf"].predict_proba(x)[0, 1]), float(model["reg"].predict(x)[0])


def view(indicator="ut_bot", minutes="60", horizon="Neste uke", model=RF, capital=10000, source=predictor.DEMO,
         cadence=AUTO, window=EXPANDING, feature_set="Utvidet · 32 egenskaper", compare_all=True, progress=None):
    """Nettsidens knapp: valgt ML-modell forutsier lønnsomheten til valgt indikator, med alle mål.

    Returnerer tekst, prediksjoner, treffsikkerhet/MSE, latency/throughput og to figurer."""
    started = time.monotonic()
    try:
        amount = float(capital)
        if isinstance(capital, bool) or not np.isfinite(amount) or not 1 <= amount <= 10_000_000:
            raise ValueError("Kapital må være fra 1 til 10 000 000 kroner.")
        if model not in available_models():
            raise ValueError("Velg en ML-modell.")
        names = available_models() if compare_all else [model]
        result = analyse(indicator, minutes, horizon, names, cadence, window, feature_set, progress, primary=model)
        try:
            item = current_input(indicator, minutes, source)
            predictions = predict_now(result, item, amount)
            problem = None
        except Exception as exc:
            item, predictions = None, []
            problem = str(exc) if isinstance(exc, ValueError) else "Ferske data kunne ikke hentes."
        seconds = time.monotonic() - started
        low, high = TARGET_SECONDS
        primary, name = result["primary"], spm.NAMES[indicator]
        speed_rep = result["reports"][primary]
        head = f"### {name} · {spm.TIMEFRAMES[minutes]} · {horizon.lower()} · {primary}\n\n"
        if predictions:
            mine = next(row for row in predictions if row["name"] == primary)
            stamp = pd.Timestamp(item["decision_time"])
            money = f"{mine['nok']:+,.0f} kr på {amount:,.0f} kr".replace(",", " ").replace("-", "−")
            head += (f"**{primary} anslår {_pct(mine['p'], 1, False)} sjanse for at {name} blir lønnsom "
                     f"{horizon.lower()}, og {_pct(mine['excess'], 2)} meravkastning mot vanlig kjøp og hold** "
                     f"({money} ved 1×). Vurdert ved {stamp:%d.%m.%Y %H:%M} UTC med «{source}»; indikatoren var "
                     f"{SIDE.get(int(item['position']), item['position'])}. Dette er et usikkert modellanslag, ikke en "
                     "garanti eller finansiell rådgivning.\n\n")
        else:
            head += f"**Ingen prediksjon for valgt input:** {problem}\n\n"
        timing = (f"**Beregningstid: {_no(seconds, 1)} s** (mål: innen {low}–{high} s)"
                  + (f"; hentet fra mellomlager, første beregning tok {_no(result['seconds'], 1)} s" if result["cached"]
                     else "") + ". ")
        cadence_text = (f"Ny trening **{result['cadence'].lower()}** ({speed_rep['refits']} ganger"
                        f"{', valgt automatisk for å holde tidsmålet' if result['auto'] else ''}), "
                        f"{result['window'].lower()}, {result['features'].lower()}. Modellen som predikerer nå, er trent "
                        f"på {_no(result['trained_rows'], 0)} eksempler med fasit kjent til "
                        f"{result['trained_until']:%d.%m.%Y}.")
        stale = sum(r["stale"] for r in result["reports"].values())
        if stale:
            cadence_text += f" Tidsgrensen ble nådd: {stale} blokker brukte forrige modell uten ny trening."
        reference = ""
        if item is not None:
            try:
                p, expected = stored_reference(indicator, minutes, horizon, item)
                reference = (f"\n\nTil sammenligning anslår den lagrede, tunede modellen fra prosjektets trening "
                             f"(statisk) {_pct(p, 1, False)} sjanse for gevinst og {_pct(expected, 2)} nettoavkastning "
                             "(ikke målt mot kjøp og hold).")
            except Exception:
                reference = ""
        table = ["| Periode | Perioder | Indikatoren snitt | Kjøp og hold snitt | Meravkastning snitt | "
                 "Slo kjøp og hold | Lønnsomme |", "|---|---:|---:|---:|---:|---:|---:|",
                 *profit_summary(result["table"], result["masks"])]
        text = (head + timing + cadence_text + reference + "\n\n**Indikatorens profitt mot vanlig kjøp og hold** "
                f"(samme åpninger, {horizon.lower()} frem, uten modell):\n\n" + "\n".join(table))
        return (text, prediction_table(result, predictions, horizon, amount) if predictions else pd.DataFrame(),
                quality_table(result), speed_table(result), drift_figure(result), importance_figure(result))
    except Exception as exc:
        message = str(exc) if isinstance(exc, ValueError) else "ML-vurderingen kunne ikke beregnes. Prøv igjen senere."
        return (f"**Ingen ML-vurdering:** {message}", *empty())
