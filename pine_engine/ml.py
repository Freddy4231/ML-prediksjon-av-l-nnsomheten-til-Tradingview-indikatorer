"""Maskinlæring på handlene til et Pine-skript: forutsi om neste handel blir lønnsom.

Hver handel skriptet gjør i backtesten blir ett eksempel. Egenskapene (features) beregnes på signallyset, altså
lyset før fyllet ved neste åpning, med bare kjent historikk: avkastning, volatilitet, ATR, RSI, avstand til
glidende snitt, plass i 48-lysers spenn, volum, tid på døgnet/uken, retning og forrige avsluttede handel.
Fasiten er om handelen ga gevinst etter gebyr og funding.

Ingen statisk trening: modellene trenes på nytt fortløpende fremover i tid (walk-forward). Ved hver ny trening
brukes bare handler som var avsluttet før beslutningen for neste handel, i et voksende eller rullerende vindu.
Alle tall under er derfor beregnet på handler modellen ikke hadde sett, slik den ville virket live.

Mål: treffsikkerhet (accuracy) og grunnrate, kvadratisk avvik (MSE/Brier) og log loss, AUC, treningstid,
svartid per prediksjon og gjennomstrømning (prediksjoner per sekund), feil-prediksjoner (falske positive/negative
og hva de kostet) og feature importance (permutasjon på handler utenfor treningen + modellens egen). Valgfritt
feature-utvalg i hvert treningssteg bruker bare treningsvinduet (eldste 70 % trener, nyeste 30 % måler), så
det kan sammenlignes ærlig med modellen uten utvalg. Ikke finansiell rådgivning.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MARKET_FEATURES = ["ret_1", "ret_4", "ret_12", "ret_48", "vol_12", "vol_48", "vol_ratio", "atr_rel", "rsi_14",
                   "dist_sma_20", "dist_sma_100", "range_pos_48", "volume_rel_20", "hour_sin", "hour_cos",
                   "dow_sin", "dow_cos"]
TRADE_FEATURES = ["side", "prev_return", "log_bars_since_exit"]
FEATURES = MARKET_FEATURES + TRADE_FEATURES
FEATURE_LABELS = {
    "ret_1": "Avkastning siste lys", "ret_4": "Avkastning 4 lys", "ret_12": "Avkastning 12 lys",
    "ret_48": "Avkastning 48 lys", "vol_12": "Volatilitet 12 lys", "vol_48": "Volatilitet 48 lys",
    "vol_ratio": "Volatilitet 12/48", "atr_rel": "ATR14 / pris", "rsi_14": "RSI 14",
    "dist_sma_20": "Avstand til SMA 20", "dist_sma_100": "Avstand til SMA 100",
    "range_pos_48": "Plass i 48-lysers spenn", "volume_rel_20": "Volum mot snitt 20",
    "hour_sin": "Tid på døgnet (sin)", "hour_cos": "Tid på døgnet (cos)", "dow_sin": "Ukedag (sin)",
    "dow_cos": "Ukedag (cos)", "side": "Retning (long/short)", "prev_return": "Forrige handels resultat",
    "log_bars_since_exit": "Lys siden forrige utgang (log)"}
LR, RF, HGB, XGB, LGBM = ("Logistisk regresjon", "Random Forest", "Gradient boosting (HistGB, LightGBM-type)",
                          "XGBoost", "LightGBM")
BASELINE = "Grunnrate (ingen modell)"
MAX_REFITS = 150
MIN_TRADES = 40


def _installed(module):
    try:
        __import__(module)
        return True
    except Exception:
        return False


def available_models():
    """Modellene som kan velges. XGBoost og LightGBM vises bare når pakkene er installert på serveren."""
    names = [LR, RF, HGB]
    if _installed("xgboost"):
        names.append(XGB)
    if _installed("lightgbm"):
        names.append(LGBM)
    return names


def make_model(name):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if name == LR:
        from sklearn.linear_model import LogisticRegression
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("model", LogisticRegression(C=0.5, max_iter=2000))])
    if name == RF:
        from sklearn.ensemble import RandomForestClassifier
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt",
                                                          n_jobs=1, random_state=0))])
    if name == HGB:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=10,
                                              l2_regularization=1.0, random_state=0)
    if name == XGB:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             n_jobs=1, eval_metric="logloss", random_state=0)
    if name == LGBM:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05, min_child_samples=10,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=1, random_state=0,
                              verbose=-1)
    raise ValueError(f"ukjent modell: {name}")


# ---------------------------------------------------------------- datasett

def market_features(bars):
    """Features per lys, beregnet med bare lyset selv og tidligere lys (kjent ved lysets lukking)."""
    c = pd.Series(np.asarray(bars.C, float))
    h, l = pd.Series(np.asarray(bars.H, float)), pd.Series(np.asarray(bars.L, float))
    v = pd.Series(np.asarray(bars.V, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(c)
        r1 = logc.diff()
        out = pd.DataFrame({"ret_1": r1, "ret_4": logc.diff(4), "ret_12": logc.diff(12), "ret_48": logc.diff(48)})
        out["vol_12"] = r1.rolling(12).std()
        out["vol_48"] = r1.rolling(48).std()
        out["vol_ratio"] = out.vol_12 / out.vol_48
        prev = c.shift()
        tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
        out["atr_rel"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c
        change = c.diff()
        gain = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        out["rsi_14"] = 100 - 100 / (1 + gain / loss)
        out["dist_sma_20"] = c / c.rolling(20).mean() - 1
        out["dist_sma_100"] = c / c.rolling(100).mean() - 1
        high, low = h.rolling(48).max(), l.rolling(48).min()
        out["range_pos_48"] = (c - low) / (high - low)
        out["volume_rel_20"] = np.log((v + 1e-12) / (v.rolling(20).mean() + 1e-12))
    closes = pd.to_datetime(np.asarray(bars.TC, np.int64), unit="ms", utc=True)
    hours = closes.hour + closes.minute / 60
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)
    out["dow_sin"], out["dow_cos"] = np.sin(2 * np.pi * closes.dayofweek / 7), np.cos(2 * np.pi * closes.dayofweek / 7)
    return out.replace([np.inf, -np.inf], np.nan)


def trade_table(bars, trades, first=0):
    """Én rad per handel: features på signallyset (lyset før fyllet), fasit og tidspunktene for lekkasjekontroll."""
    trades = sorted((t for t in trades if t.entry_bar - 1 >= first), key=lambda t: (t.entry_bar, t.exit_bar))
    if not trades:
        return pd.DataFrame(columns=FEATURES + ["decision", "entry_bar", "exit_bar", "ret", "win"])
    features = market_features(bars)
    rows = []
    exits = []          # (exit_bar, nettoavkastning) for allerede avsluttede handler
    for t in trades:
        decision = t.entry_bar - 1
        notional = t.qty * t.entry_price
        ret = t.pnl / notional if notional > 0 else 0.0
        known = [e for e in exits if e[0] <= decision]
        last = max(known, key=lambda e: e[0]) if known else None
        row = features.iloc[decision].to_dict()
        row.update(side=float(t.side), prev_return=last[1] if last else 0.0,
                   log_bars_since_exit=math.log1p(decision - last[0]) if last else math.log1p(decision - first),
                   decision=decision, entry_bar=t.entry_bar, exit_bar=t.exit_bar, ret=ret, win=int(ret > 0),
                   entry_time=int(bars.T[t.entry_bar]))
        rows.append(row)
        exits.append((t.exit_bar, ret))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- fortløpende trening

def _brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


@dataclass
class Block:
    model: object
    rows: np.ndarray
    features: list
    train_rows: np.ndarray
    base_rate: float
    fit_ms: float
    predict_ms: float
    single_ms: float
    constant: bool = False


@dataclass
class WalkForward:
    name: str
    proba: np.ndarray
    base: np.ndarray
    blocks: list = field(default_factory=list)
    step: int = 1
    selected: bool = False
    requested: int = 1
    adjusted: bool = False


def _fit(name, X, y):
    if len(np.unique(y)) < 2:
        return None
    model = make_model(name)
    model.fit(X, y)
    return model


def _predict(model, X, constant):
    if model is None:
        return np.full(len(X), constant)
    return model.predict_proba(X)[:, 1]


def _stacked_scores(model, valid, y, constant, features, shuffled_columns):
    """Brier-økning per feature: alle stokkede kopier predikeres i ett kall (raskt også for trær)."""
    base = _brier(_predict(model, valid, constant), y)
    copies = []
    for feature in features:
        for column in shuffled_columns[feature]:
            copy = valid.copy()
            copy[feature] = column
            copies.append(copy)
    if not copies:
        return {}
    p = _predict(model, pd.concat(copies, ignore_index=True), constant)
    n, repeats = len(valid), len(next(iter(shuffled_columns.values())))
    scores = {}
    for i, feature in enumerate(features):
        losses = [_brier(p[(i * repeats + k) * n:(i * repeats + k + 1) * n], y) for k in range(repeats)]
        scores[feature] = float(np.mean(losses) - base)
    return scores


def _select_features(name, X, y, features, seed=0, repeats=3):
    """Feature-utvalg bare på treningsvinduet: eldste 70 % trener, nyeste 30 % måler permutasjonsviktighet."""
    cut = int(len(X) * 0.7)
    if cut < 20 or len(X) - cut < 10 or len(np.unique(y[:cut])) < 2 or len(np.unique(y[cut:])) < 2:
        return list(features)
    model = _fit(name, X.iloc[:cut], y[:cut])
    valid = X.iloc[cut:].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    shuffled = {f: [rng.permutation(valid[f].to_numpy()) for _ in range(repeats)] for f in features}
    scores = _stacked_scores(model, valid, y[cut:], float(y[:cut].mean()), features, shuffled)
    keep = [f for f in features if scores[f] > 0]
    if len(keep) < 3:
        keep = sorted(features, key=lambda f: scores[f], reverse=True)[:3]
    return keep


def walk_forward(table, name, mode="expanding", window=200, refit=5, min_train=30, select=False, deadline=None,
                 budget=None):
    """Trener på nytt hver ``refit``-te handel. Bare handler med utgang ≤ beslutningslyset brukes til trening.

    Med ``budget`` (sekunder) økes avstanden mellom treningene underveis hvis resten ellers ikke rekker å bli ferdig
    i tide. Det endrer bare hvor ofte modellen trenes, aldri hvilke handler den får se (merkes ``adjusted``)."""
    n = len(table)
    proba, base = np.full(n, np.nan), np.full(n, np.nan)
    step = max(int(refit), math.ceil(n / MAX_REFITS))
    result = WalkForward(name, proba, base, step=step, selected=select, requested=step)
    started_all, durations = time.perf_counter(), []
    decision, exit_bar = table["decision"].to_numpy(), table["exit_bar"].to_numpy()
    y_all = table["win"].to_numpy(int)
    j = 0
    while j < n:
        if deadline is not None and time.monotonic() > deadline:
            break
        end = min(j + step, n)
        labeled = np.flatnonzero(exit_bar <= decision[j])
        if mode == "rolling":
            labeled = labeled[-int(window):]
        if len(labeled) >= int(min_train):
            block_started = time.perf_counter()
            y = y_all[labeled]
            features = _select_features(name, table.iloc[labeled][FEATURES], y, FEATURES) if select else FEATURES
            X = table.iloc[labeled][features]
            started = time.perf_counter()
            model = _fit(name, X, y)
            fit_ms = (time.perf_counter() - started) * 1000
            rows = np.arange(j, end)
            target = table.iloc[rows][features]
            started = time.perf_counter()
            p = _predict(model, target, float(y.mean()))
            predict_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            _predict(model, target.iloc[:1], float(y.mean()))
            single_ms = (time.perf_counter() - started) * 1000
            proba[rows], base[rows] = p, float(y.mean())
            result.blocks.append(Block(model, rows, list(features), labeled, float(y.mean()), fit_ms, predict_ms,
                                       single_ms, constant=model is None))
            durations.append(time.perf_counter() - block_started)
        j = end
        # Tidsbudsjett: anslaget bruker de siste treningene (første kall har engangskostnader) og venter til tre er gjort.
        if budget is not None and len(durations) >= 3 and j < n:
            spent = time.perf_counter() - started_all
            blocks_left = max(int((budget - spent) / float(np.median(durations[-5:]))), 1)
            needed = math.ceil((n - j) / blocks_left)
            if needed > step:
                step, result.adjusted = needed, True
    result.step = step
    return result


# ---------------------------------------------------------------- mål

def evaluate(table, wf, threshold=0.5):
    """Mål på handlene modellen ikke hadde sett. Grunnraten bruker treningsvinduets andel lønnsomme handler."""
    mask = ~np.isnan(wf.proba)
    y, r = table["win"].to_numpy(int)[mask], table["ret"].to_numpy(float)[mask]
    p, b = wf.proba[mask], wf.base[mask]
    if not mask.any():
        return {"predicted": 0}
    from sklearn.metrics import log_loss, roc_auc_score
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    take = p >= threshold
    fit_ms = [blk.fit_ms for blk in wf.blocks]
    predict_seconds = sum(blk.predict_ms for blk in wf.blocks) / 1000
    out = {
        "predicted": int(mask.sum()), "refits": len(wf.blocks), "step": wf.step,
        "requested_step": wf.requested, "adjusted": wf.adjusted,
        "train_sizes": (min(len(b.train_rows) for b in wf.blocks), max(len(b.train_rows) for b in wf.blocks)),
        "base_rate": float(y.mean()),
        "accuracy": float(np.mean((p >= 0.5) == y)), "baseline_accuracy": float(np.mean((b >= 0.5) == y)),
        "mse": _brier(p, y), "baseline_mse": _brier(b, y),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "baseline_log_loss": float(log_loss(y, np.clip(b, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "fit_ms_median": float(np.median(fit_ms)), "fit_ms_total": float(np.sum(fit_ms)),
        "latency_ms": float(np.median([blk.single_ms for blk in wf.blocks])),
        "throughput": float(mask.sum() / predict_seconds) if predict_seconds > 0 else None,
        "tp": int(np.sum(take & (y == 1))), "fp": int(np.sum(take & (y == 0))),
        "fn": int(np.sum(~take & (y == 1))), "tn": int(np.sum(~take & (y == 0))),
        "fp_cost": float(np.sum(r[take & (y == 0)])), "fn_missed": float(np.sum(r[~take & (y == 1)])),
        "all_trades": float(np.prod(1 + r) - 1), "filtered": float(np.prod(1 + r[take]) - 1) if take.any() else 0.0,
        "taken": int(take.sum()), "constant_blocks": int(sum(blk.constant for blk in wf.blocks)),
    }
    half = len(y) // 2
    if half >= 10:
        out["accuracy_first_half"] = float(np.mean((p[:half] >= 0.5) == y[:half]))
        out["accuracy_second_half"] = float(np.mean((p[half:] >= 0.5) == y[half:]))
    return out


def permutation_importance(table, wf, repeats=5, seed=0):
    """Hvor mye MSE (Brier) øker på ukjente handler når én feature stokkes. Positivt = featuren hjelper.

    Hver feature stokkes på tvers av alle handlene modellen forutså, og hver blokk predikeres med sin egen modell.
    """
    mask = ~np.isnan(wf.proba)
    if not mask.any():
        return pd.Series(dtype=float)
    y = table["win"].to_numpy(int)
    rows = np.flatnonzero(mask)
    base = _brier(wf.proba[rows], y[rows])
    rng = np.random.default_rng(seed)
    used = [f for f in FEATURES if any(f in blk.features for blk in wf.blocks)]
    position = {row: i for i, row in enumerate(rows)}
    permuted = {f: [rng.permutation(table[f].to_numpy()[rows]) for _ in range(repeats)] for f in used}
    total = {f: np.zeros(repeats) for f in used}
    for blk in wf.blocks:
        local = np.array([position[r] for r in blk.rows])
        valid = table.iloc[blk.rows][blk.features].reset_index(drop=True)
        mine = [f for f in used if f in blk.features]
        p0 = _predict(blk.model, valid, blk.base_rate)
        shuffled = {f: [perm[local] for perm in permuted[f]] for f in mine}
        copies = []
        for f in mine:
            for column in shuffled[f]:
                copy = valid.copy()
                copy[f] = column
                copies.append(copy)
        p = _predict(blk.model, pd.concat(copies, ignore_index=True), blk.base_rate) if copies else np.array([])
        yb, n = y[blk.rows], len(blk.rows)
        for i, f in enumerate(mine):
            for k in range(repeats):
                total[f][k] += np.sum((p[(i * repeats + k) * n:(i * repeats + k + 1) * n] - yb) ** 2)
        for f in used:
            if f not in mine:                        # featuren var valgt bort i denne blokken: uendret tap
                total[f] += np.sum((p0 - yb) ** 2)
    scores = {f: float(np.mean(total[f] / len(rows)) - base) for f in used}
    return pd.Series(scores).sort_values(ascending=False)


def native_importance(wf):
    """Modellens egen viktighet i siste trening: |koeffisient| (standardisert) eller trærnes splitt-viktighet."""
    last = next((blk for blk in reversed(wf.blocks) if blk.model is not None), None)
    if last is None:
        return pd.Series(dtype=float)
    model = last.model.named_steps["model"] if hasattr(last.model, "named_steps") else last.model
    if hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, float)
    else:
        return pd.Series(dtype=float)
    if values.sum() > 0:
        values = values / values.sum()
    return pd.Series(values, index=last.features).sort_values(ascending=False)


def rolling_accuracy(table, wf, window=20):
    mask = ~np.isnan(wf.proba)
    y = table["win"].to_numpy(int)[mask]
    hits = pd.Series(((wf.proba[mask] >= 0.5) == y).astype(float))
    base = pd.Series(((wf.base[mask] >= 0.5) == y).astype(float))
    times = pd.to_datetime(table["entry_time"].to_numpy()[mask], unit="ms", utc=True)
    return times, hits.rolling(window, min_periods=window).mean(), base.rolling(window, min_periods=window).mean()


def mispredictions(table, wf, threshold=0.5, limit=15):
    mask = ~np.isnan(wf.proba)
    frame = table.loc[mask].copy()
    frame["p"] = wf.proba[mask]
    wrong = frame[(frame.p >= threshold) != (frame.win == 1)].copy()
    wrong["sikkerhet"] = (wrong.p - 0.5).abs()
    wrong = wrong.sort_values("sikkerhet", ascending=False).head(limit)
    rows = []
    for _, row in wrong.iterrows():
        rows.append({"Inngang (UTC)": str(pd.Timestamp(int(row.entry_time), unit="ms", tz="UTC"))[:16],
                     "Retning": "Long" if row.side > 0 else "Short",
                     "Modellens P(gevinst)": round(float(row.p), 3),
                     "Faktisk handel (%)": round(100 * float(row.ret), 3),
                     "Feiltype": "Falsk positiv (tok tap)" if row.p >= threshold else "Falsk negativ (gikk glipp)",
                     "RSI 14": round(float(row.rsi_14), 1) if np.isfinite(row.rsi_14) else None,
                     "Avkastning 12 lys (%)": round(100 * float(row.ret_12), 2) if np.isfinite(row.ret_12) else None,
                     "Volatilitet 48 lys (%)": round(100 * float(row.vol_48), 3) if np.isfinite(row.vol_48) else None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- forbedringer

def _no(value, digits=2):
    """Norsk tallformat: desimalkomma og ekte minustegn."""
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def _pct(value, digits=1):
    return f"{value * 100:+.{digits}f} %".replace(".", ",").replace("-", "−")


def suggestions(table, reports, importance, threshold=0.5, select=False):
    """Konkrete råd ut fra målingene: hva tallene betyr og hvordan modellene kan forbedres."""
    tips = []
    usable = {name: rep for name, rep in reports.items() if rep.get("predicted")}
    if not usable:
        return ["For få handler med kjent utfall til å trene modellene. Velg lengre periode eller kortere intervall."]
    n = next(iter(usable.values()))["predicted"]
    if n < 60:
        tips.append(f"Bare {n} handler kunne forutsies. Med så få handler er tilfeldige utslag store; velg lengre "
                    "periode for sikrere tall.")
    best_name = min(usable, key=lambda k: usable[k]["mse"])
    best = usable[best_name]
    gain = best["baseline_mse"] - best["mse"]
    if gain <= 0:
        tips.append("Ingen modell har lavere MSE enn grunnraten (andelen lønnsomme handler i treningsvinduet). "
                    "Med disse featurene kan lønnsomheten ikke forutsies bedre enn å gjette; bruk ikke modellen som "
                    "filter. Prøv andre features (f.eks. skriptets egne verdier), lengre periode eller et annet marked.")
    else:
        tips.append(f"Beste modell er **{best_name}**: MSE {_no(best['mse'], 4)} mot grunnratens "
                    f"{_no(best['baseline_mse'], 4)} (−{_no(gain, 4)}). Sjekk at forbedringen holder i en annen "
                    "periode før modellen brukes.")
    if best["accuracy"] <= best["baseline_accuracy"]:
        tips.append("Treffsikkerheten er ikke høyere enn grunnratens. Ved ujevn fordeling (f.eks. 35 % lønnsomme "
                    "handler) kan treffsikkerhet lure; MSE og log loss viser om sannsynlighetene er nyttige.")
    if importance is not None and len(importance):
        weak = [FEATURE_LABELS.get(f, f) for f, v in importance.items() if v <= 0]
        strong = [FEATURE_LABELS.get(f, f) for f, v in importance.items() if v > 0][:3]
        if strong:
            tips.append("Mest nyttige features (MSE øker når de stokkes): " + ", ".join(strong) + ".")
        if weak:
            tips.append(f"{len(weak)} features hjelper ikke på ukjente handler (" + ", ".join(weak[:6])
                        + ("…" if len(weak) > 6 else "") + ")."
                        + ("" if select else " Slå på «feature-utvalg i hvert treningssteg» for å fjerne slike "
                           "automatisk uten å se på testhandlene."))
    selected = {k: v for k, v in usable.items() if k.endswith("+ feature-utvalg")}
    for name, rep in selected.items():
        plain = usable.get(name.replace(" + feature-utvalg", ""))
        if plain:
            better = rep["mse"] < plain["mse"]
            tips.append(f"Feature-utvalg ga {'lavere' if better else 'høyere'} MSE for "
                        f"{name.replace(' + feature-utvalg', '')} ({_no(rep['mse'], 4)} mot {_no(plain['mse'], 4)})"
                        + (": bruk utvalget." if better else ": behold alle features."))
    if best["fp"] and best["fp_cost"] < 0 and abs(best["fp_cost"]) > abs(best["fn_missed"]):
        tips.append(f"Falske positive (modellen sa gevinst, handelen tapte) kostet mer ({_pct(best['fp_cost'])}) "
                    f"enn gevinstene den hoppet over ({_pct(best['fn_missed'])}). En høyere terskel enn "
                    f"{_no(threshold)} gir færre, men sikrere handler.")
    if "accuracy_second_half" in best and best["accuracy_second_half"] < best["accuracy_first_half"] - 0.05:
        tips.append("Treffsikkerheten falt i andre halvdel av perioden (modellen eldes). Prøv rullerende vindu og "
                    "hyppigere trening.")
    fastest = min(usable, key=lambda k: usable[k]["latency_ms"])
    slowest = max(usable, key=lambda k: usable[k]["fit_ms_median"])
    tips.append(f"Raskest per prediksjon: {fastest} ({_no(usable[fastest]['latency_ms'])} ms). Tregest å trene: "
                f"{slowest} ({_no(usable[slowest]['fit_ms_median'], 0)} ms per trening). Alle er raske nok for lys på "
                "1 minutt eller mer; trener du på nytt ved hver handel, betyr treningstiden mest.")
    if best["taken"]:
        tips.append(f"Som filter (ta handelen bare når P ≥ {_no(threshold)}) ville {best_name} tatt {best['taken']} av "
                    f"{best['predicted']} handler: {_pct(best['filtered'])} mot {_pct(best['all_trades'])} med alle "
                    "(handlene etter hverandre, uten gearing). Velg ikke terskel etter dette tallet.")
    return tips


def run(bars, trades, first=0, models=(LR, RF, HGB), mode="expanding", window=200, refit=5, min_train=30,
        threshold=0.5, select=False, deadline=None, budget=None):
    """Hele analysen for valgte modeller; returnerer målinger, viktighet, feil-prediksjoner og råd.

    ``budget`` (sekunder) holder den fortløpende treningen innenfor tidsmålet: hver modell får resten av budsjettet
    delt på modellene som gjenstår, så tid de raske modellene ikke bruker, går til de tregere."""
    table = trade_table(bars, trades, first)
    if len(table) < MIN_TRADES:
        raise ValueError(f"skriptet gjorde bare {len(table)} handler i perioden; ML trenger minst {MIN_TRADES}. "
                         "Velg lengre periode eller kortere intervall.")
    runs, reports = {}, {}
    variants = [(m, False) for m in models] + ([(m, True) for m in models] if select else [])
    started = time.perf_counter()
    for k, (name, chosen) in enumerate(variants):
        label = name + (" + feature-utvalg" if chosen else "")
        share = (max(budget - (time.perf_counter() - started), 0.0) / (len(variants) - k)) if budget else None
        wf = walk_forward(table, name, mode, window, refit, min_train, chosen, deadline, share)
        runs[label], reports[label] = wf, evaluate(table, wf, threshold)
    usable = [k for k, rep in reports.items() if rep.get("predicted")]
    if not usable:
        raise ValueError("ingen handler kunne forutsies; senk «minste antall treningshandler» eller velg lengre periode.")
    best = min(usable, key=lambda k: reports[k]["mse"])
    importance = {k: permutation_importance(table, runs[k]) for k in usable}
    native = {k: native_importance(runs[k]) for k in usable}
    return {"table": table, "runs": runs, "reports": reports, "best": best, "importance": importance,
            "native": native, "errors": mispredictions(table, runs[best], threshold),
            "tips": suggestions(table, reports, importance[best], threshold, select)}
