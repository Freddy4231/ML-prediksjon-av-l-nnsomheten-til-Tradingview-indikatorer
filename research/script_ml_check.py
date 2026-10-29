"""Trener, tuner og tester én ML-modell per skript og tidsintervall som forutsier skriptets lønnsomhet.

ML endrer ikke skriptene og velger ikke handler; den gir bare en prediksjon. Se ``script_profit_ml.py``.

**Datasett:** Binance BTCUSDT spot 2017-08-17 til 2026-10-10: ``data/btcusdt_latest_1h.csv`` (1 time),
``data/30m/btcusdt_30m.csv`` (30 minutter), og 4 timer og 1 dag slått sammen fra timesdataene. Funding fra Binance
USDT-M fra 2019-09. Ett eksempel per døgn (beslutning ved døgnets siste close, 00:00 UTC) per skript og horisont.
**Fasit:** skriptets avkastning fra neste open og én dag, uke eller måned frem (1×, 0,10 % per side, funding).

**Tuning (bare på utviklingsårene 2019–2024):**
- Rensing: av, eller på (fjern treningseksempler der featurevinduet eller fasiten berører utfylte barer + klipp hver
  feature til 1.–99. persentil i treningsdataene).
- Feature engineering: grunnleggende (23 markedsfeatures + 4 om skriptet) eller utvidet (+ signaler siste 7 dager,
  skriptets fall fra 30-dagers topp, kort/lang volatilitet, trendeffektivitet og |avstand fra SMA168|).
- Skalering: ingen, StandardScaler eller RobustScaler (påvirker bare den lineære modellen).
- Hyperparametre: logistisk C ∈ {0,01; 0,1; 1}; Random Forest (dybde, blad) ∈ {(4, 20), (6, 50)};
  gradient boosting (dybde, læringsrate) ∈ {(2; 0,05), (3; 0,05)}.
Valgt oppsett per skript, intervall og horisont = lavest snitt log loss i gående-fremover-testene 2019–2024.
Utgangspunktet (før tuning) er grunnleggende features, ingen rensing, StandardScaler, C = 0,1, RF (4, 20) og
boosting (3; 0,05). Forventet avkastning bruker samme rensing/features/skalering som valgt sannsynlighetsmodell og
velger modellfamilie på MAE. 2025–2026 er en kjent historisk kontroll, ikke brukt i valget.

    python -m research.script_ml_check
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score

import script_profit_ml as spm
from features import load_ohlcv, make_features
from research import futures_funding

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/btcusdt_latest_1h.csv"
DATA_30M = ROOT / "data/30m/btcusdt_30m.csv"
OUT = ROOT / "reports/script_ml_check"
DEV_YEARS = list(range(2019, 2025))
RETRO = ("2025–2026", pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2027-01-01", tz="UTC"))
MIN_TRAIN = 200
TUNING_TREES, FINAL_TREES = 100, 300
MODEL_NAMES = {"linear": "Logistisk regresjon / Ridge", "random_forest": "Random Forest",
               "boosting": "Gradient boosting", "baseline": "Baseline (historisk snitt)"}
PARAMS = {"linear": "logistisk C ∈ {0,01; 0,1; 1}, Ridge alpha = 10",
          "random_forest": "dybde/blad ∈ {(4, 20), (6, 50)}, 300 trær i ferdig modell",
          "boosting": "HistGradientBoosting, dybde/læringsrate ∈ {(2; 0,05), (3; 0,05)}, 200 runder, minst 30 per blad"}
GRID = {"linear": [{"C": c} for c in (0.01, 0.1, 1.0)],
        "random_forest": [{"max_depth": 4, "min_samples_leaf": 20}, {"max_depth": 6, "min_samples_leaf": 50}],
        "boosting": [{"max_depth": 2, "learning_rate": 0.05}, {"max_depth": 3, "learning_rate": 0.05}]}
DEFAULT = {"features": "grunnleggende", "cleaning": "av", "scaler": "standard",
           "params": {"linear": {"C": 0.1}, "random_forest": {"max_depth": 4, "min_samples_leaf": 20},
                      "boosting": {"max_depth": 3, "learning_rate": 0.05}}}


def classifier(family, params, trees=TUNING_TREES):
    if family == "linear":
        return LogisticRegression(C=params["C"], max_iter=2000)
    if family == "random_forest":
        return RandomForestClassifier(n_estimators=trees, random_state=42, n_jobs=2, **params)
    return HistGradientBoostingClassifier(max_iter=200, min_samples_leaf=30, random_state=42, **params)


def regressor(family, trees=FINAL_TREES):
    if family == "linear":
        return Ridge(alpha=10.0)
    if family == "random_forest":
        return RandomForestRegressor(n_estimators=trees, max_depth=4, min_samples_leaf=20, random_state=42, n_jobs=2)
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=30,
                                         random_state=42)


def configs():
    """Alle tuningkombinasjoner. Trær påvirkes ikke av skalering, så de prøves bare uten skalering."""
    for features, cleaning in itertools.product(spm.FEATURE_SETS, spm.CLEANING):
        for scaler in spm.SCALERS:
            for params in GRID["linear"]:
                yield {"family": "linear", "features": features, "cleaning": cleaning, "scaler": scaler,
                       "params": params}
        for family in ("random_forest", "boosting"):
            for params in GRID[family]:
                yield {"family": family, "features": features, "cleaning": cleaning, "scaler": "ingen",
                       "params": params}


def is_default(config):
    scaler = DEFAULT["scaler"] if config["family"] == "linear" else "ingen"
    return (config["features"] == DEFAULT["features"] and config["cleaning"] == DEFAULT["cleaning"]
            and config["scaler"] == scaler and config["params"] == DEFAULT["params"][config["family"]])


def describe(config):
    params = ", ".join(f"{k}={v}" for k, v in config["params"].items())
    return (f"{MODEL_NAMES[config['family']]} ({params}) · rensing {config['cleaning']} · {config['features']} "
            f"features · skalering {config['scaler']}")


def load_funding():
    try:
        fund = futures_funding.load("BTCUSDT")
    except Exception:
        return None
    return pd.Series(fund["funding_rate"].to_numpy(dtype=float), index=pd.to_datetime(fund["funding_time"], utc=True))


def load(minutes, funding):
    """Barer på intervallet (regularisert, med ``filled``) og funding summert per bar."""
    raw = load_ohlcv(DATA)
    if minutes == "30":
        raw = pd.read_csv(DATA_30M, index_col="open_time")
        raw.index = pd.to_datetime(raw.index, utc=True, format="ISO8601").floor("30min")
        raw = raw[["open", "high", "low", "close", "volume"]].astype(float)
    bars = spm.regularize(raw[["open", "high", "low", "close", "volume"]], 60 if minutes in ("240", "1440") else minutes)
    if minutes in ("240", "1440"):
        bars = spm.regularize(spm.resample(bars, minutes), minutes)
    rates = np.zeros(len(bars))
    if funding is not None:
        where = bars.index.get_indexer(funding.index.floor(f"{int(minutes)}min"))
        np.add.at(rates, where[where >= 0], funding.to_numpy()[where >= 0])
    return raw, bars, rates


def dataset(bars, features, position, rates, hours, minutes):
    """Ett eksempel per døgn: alle features ved døgnets siste close, fasit ``hours`` frem, og rensemerket."""
    eq_label = spm.equity(spm.bar_returns(bars, position, rates))
    eq_feature = spm.equity(spm.bar_returns(bars, position))
    n, steps = len(bars), spm.bars_for(hours, minutes)
    close_time = bars.index + pd.Timedelta(minutes=int(minutes))
    rows = np.flatnonzero((close_time.hour == 0) & (close_time.minute == 0) & (np.arange(n) + 1 + steps < n))
    x = spm.sample_features(features, position, eq_feature, rows, minutes, bars["close"].to_numpy())
    y = eq_label[rows + 1 + steps] / eq_label[rows + 1] - 1
    known = bars.index[rows + 1 + steps]              # fasiten er kjent først ved denne open
    filled = bars["filled"].to_numpy(dtype=bool) if "filled" in bars else np.zeros(n, dtype=bool)
    # Fasiten må heller ikke bygge på utfylte barer for at eksemplet skal regnes som rent.
    cum = np.r_[0, np.cumsum(filled)]
    future_clean = (cum[rows + 2 + steps] - cum[rows + 1]) == 0
    clean = spm.clean_rows(filled, rows, minutes) & future_clean
    ok = np.isfinite(x.to_numpy(dtype=float)).all(axis=1) & np.isfinite(y)
    return x[ok], y[ok], known[ok], clean[ok]


def periods():
    return [(str(year), pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
            for year in DEV_YEARS] + [RETRO]


def folds(x, known, clean, cleaning):
    """(navn, treningsmaske, testmaske) for gående fremover. Rensing fjerner urene eksempler fra treningen."""
    times = x.index
    for label, start, end in periods():
        train = np.asarray(known < start)
        if cleaning == "på":
            train &= clean
        test = np.asarray((times >= start) & (times < end))
        yield label, train, test


def score_classifier(x, win, known, clean, config):
    X = x.to_numpy(dtype=float)
    out = {}
    for label, train, test in folds(x, known, clean, config["cleaning"]):
        if train.sum() < MIN_TRAIN or test.sum() < 20:
            continue
        if len(set(win[train])) < 2:
            model = spm.ConstantClassifier()
        else:
            model = spm.build(classifier(config["family"], config["params"]), config["features"],
                              config["cleaning"], config["scaler"])
        p = np.clip(model.fit(X[train], win[train]).predict_proba(X[test])[:, 1], 1e-4, 1 - 1e-4)
        out[label] = {"log_loss": log_loss(win[test], p, labels=[0, 1]),
                      "auc": roc_auc_score(win[test], p) if len(set(win[test])) == 2 else None,
                      "acc": float(np.mean(win[test] == (p >= 0.5)))}
    return out


def score_baseline(x, y, known, clean):
    win, target = (np.asarray(y) > 0).astype(int), np.asarray(y)
    out = {}
    for label, train, test in folds(x, known, clean, "av"):
        if train.sum() < MIN_TRAIN or test.sum() < 20:
            continue
        rate = float(np.clip(win[train].mean(), 1e-4, 1 - 1e-4))
        out[label] = {"log_loss": log_loss(win[test], np.full(test.sum(), rate), labels=[0, 1]), "auc": 0.5,
                      "acc": float(np.mean(win[test] == int(rate >= 0.5))),
                      "mae": mean_absolute_error(target[test], np.full(test.sum(), target[train].mean())), "corr": None}
    return out


def score_regressor(x, y, known, clean, family, prep):
    X, target = x.to_numpy(dtype=float), np.asarray(y)
    out = {}
    for label, train, test in folds(x, known, clean, prep["cleaning"]):
        if train.sum() < MIN_TRAIN or test.sum() < 20:
            continue
        model = spm.build(regressor(family, TUNING_TREES), prep["features"], prep["cleaning"],
                          prep["scaler"] if family == "linear" else "ingen")
        pred = model.fit(X[train], target[train]).predict(X[test])
        out[label] = {"mae": mean_absolute_error(target[test], pred),
                      "corr": float(np.corrcoef(pred, target[test])[0, 1]) if np.std(pred) > 0 else None}
    return out


def mean_of(per, metric, labels):
    values = [per[y][metric] for y in labels if y in per and per[y].get(metric) is not None]
    return float(np.mean(values)) if values else None


def summary(per, dev):
    retro = per.get(RETRO[0], {})
    return {"dev_log_loss": mean_of(per, "log_loss", dev), "dev_auc": mean_of(per, "auc", dev),
            "dev_acc": mean_of(per, "acc", dev), "retro_log_loss": retro.get("log_loss"),
            "retro_auc": retro.get("auc"), "retro_acc": retro.get("acc")}


def rounded(d):
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}


def tune_group(x, y, known, clean, dev):
    """Prøver alle kombinasjoner; returnerer valgt, utgangspunkt og beste per (rensing, features, skalering)."""
    win = (np.asarray(y) > 0).astype(int)
    tried = []
    for config in configs():
        stats = summary(score_classifier(x, win, known, clean, config), dev)
        if stats["dev_log_loss"] is not None:
            tried.append((config, stats))
    chosen = min(tried, key=lambda item: item[1]["dev_log_loss"])
    default = {c["family"]: (c, st) for c, st in tried if is_default(c)}
    explorer = {}
    for features, cleaning, scaler in itertools.product(spm.FEATURE_SETS, spm.CLEANING, spm.SCALERS):
        # Trær bruker ikke skalering; de er med i alle skaleringsvalg.
        options = [(c, st) for c, st in tried if c["features"] == features and c["cleaning"] == cleaning
                   and (c["scaler"] == scaler or c["family"] != "linear")]
        best = min(options, key=lambda item: item[1]["dev_log_loss"])
        linear = min((item for item in options if item[0]["family"] == "linear" and item[0]["scaler"] == scaler),
                     key=lambda item: item[1]["dev_log_loss"])
        explorer[f"{cleaning}|{features}|{scaler}"] = {"family": best[0]["family"], "params": best[0]["params"],
                                                       **rounded(best[1]),
                                                       "linear": {"params": linear[0]["params"], **rounded(linear[1])}}
    return chosen, default, explorer


def main(timeframes=tuple(spm.TIMEFRAMES)):
    started = time.time()
    funding = load_funding()
    dev = [str(y) for y in DEV_YEARS]
    result = {
        "dataset": {"source": "Binance BTCUSDT spot", "files": {"60": str(DATA.relative_to(ROOT)),
                                                               "30": str(DATA_30M.relative_to(ROOT)),
                                                               "240": "slått sammen fra 1 time",
                                                               "1440": "slått sammen fra 1 time"},
                    "funding": "Binance USDT-M fra 2019-09", "features": spm.ALL_COLUMNS,
                    "sampling": "ett eksempel per døgn (døgnets siste close, 00:00 UTC)"},
        "target": "skriptets avkastning fra neste open og én dag/uke/måned frem (1×, 0,10 % per side, funding)",
        "models": MODEL_NAMES, "params": PARAMS, "horizons": spm.HORIZONS, "dev_years": DEV_YEARS,
        "tuning": {"cleaning": spm.CLEANING, "feature_sets": {k: len(v) for k, v in spm.FEATURE_SETS.items()},
                   "engineered": spm.ENGINEERED_COLUMNS, "scalers": list(spm.SCALERS), "grid": GRID,
                   "default": DEFAULT, "configs": sum(1 for _ in configs())},
        "selection": "per skript, intervall og horisont: lavest snitt log loss 2019–2024 over alle tuningkombinasjoner",
        "timeframes": {}}
    bundle = {"features": spm.ALL_COLUMNS, "models": {}}
    for minutes in timeframes:
        raw, bars, rates = load(minutes, funding)
        features = make_features(bars)
        positions = spm.positions(bars, minutes)
        frame_info = {"label": spm.TIMEFRAMES[minutes], "first": str(bars.index[0]), "last": str(bars.index[-1]),
                      "bars": len(bars), "filled_bars": int(bars["filled"].sum()), "scripts": {}}
        for key, position in positions.items():
            entry = {"name": spm.NAMES[key], "signals": int((np.diff(position) != 0).sum()), "horizons": {}}
            for horizon, hours in spm.HORIZONS.items():
                x, y, known, clean = dataset(bars, features, position, rates, hours, minutes)
                Y = np.asarray(y)
                chosen, default, explorer = tune_group(x, y, known, clean, dev)
                config, stats = chosen
                base = score_baseline(x, y, known, clean)
                regs = {f: score_regressor(x, y, known, clean, f, config) for f in ("linear", "random_forest", "boosting")}
                reg = min(regs, key=lambda f: mean_of(regs[f], "mae", dev) or np.inf)
                X = x.to_numpy(dtype=float)
                train = clean if config["cleaning"] == "på" else np.ones(len(Y), dtype=bool)
                clf_model = (spm.build(classifier(config["family"], config["params"], FINAL_TREES), config["features"],
                                       config["cleaning"], config["scaler"])
                             if len(set(Y[train] > 0)) == 2 else spm.ConstantClassifier())
                reg_model = spm.build(regressor(reg), config["features"], config["cleaning"],
                                      config["scaler"] if reg == "linear" else "ingen")
                bundle["models"][f"{minutes}|{key}|{horizon}"] = {
                    "clf": clf_model.fit(X[train], (Y[train] > 0).astype(int)), "reg": reg_model.fit(X[train], Y[train]),
                    "clf_name": describe(config), "reg_name": MODEL_NAMES[reg]}
                entry["horizons"][horizon] = {
                    "samples": int(len(Y)), "clean_share": float(np.mean(clean)), "profitable_share": float((Y > 0).mean()),
                    "mean_return": float(Y.mean()),
                    "chosen": {**config, "description": describe(config), **rounded(stats)},
                    "default": {f: {"params": c["params"], **rounded(st)} for f, (c, st) in default.items()},
                    "baseline": rounded(summary(base, dev) | {"dev_mae": mean_of(base, "mae", dev),
                                                              "retro_mae": base.get(RETRO[0], {}).get("mae")}),
                    "regression": {f: rounded({"dev_mae": mean_of(r, "mae", dev), "dev_corr": mean_of(r, "corr", dev),
                                               "retro_mae": r.get(RETRO[0], {}).get("mae"),
                                               "retro_corr": r.get(RETRO[0], {}).get("corr")})
                                   for f, r in regs.items()},
                    "chosen_reg": reg, "explorer": explorer}
                d = default[config["family"]][1] if config["family"] in default else {}
                print(f"{spm.TIMEFRAMES[minutes]} · {spm.NAMES[key]} · {horizon}: {describe(config)} · AUC "
                      f"{stats['dev_auc'] or 0:.3f}/{stats['retro_auc'] or 0:.3f} (før tuning "
                      f"{d.get('dev_auc') or 0:.3f}/{d.get('retro_auc') or 0:.3f})", flush=True)
            frame_info["scripts"][key] = entry
        result["timeframes"][minutes] = frame_info
    OUT.mkdir(parents=True, exist_ok=True)
    model_path = OUT / "models.joblib"
    joblib.dump(bundle, model_path, compress=3)
    result["model_file"] = {"path": str(model_path.relative_to(ROOT)),
                            "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
    result["sources"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                         for p in ("research/script_ml_check.py", "script_profit_ml.py", "features.py")}
    result["seconds"] = round(time.time() - started, 1)
    (OUT / "results.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    main()
