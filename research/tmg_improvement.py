"""Låst, avgrenset TMG-søk på ekte data; ingen nettverk, kontodata eller ordre.

Valg: 2020–23, deretter godkjenning i 2024. 2025–26 er kjent historisk kontroll,
ikke en urørt test. Samme CLV-proxy som Pine/WEEX brukes også når filen har ekte
taker-volum. Alle handler fylles ved neste ekte åpning, med 1× notional.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import tmg_trades_weex as tmg

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/tmg_improvement_2026-10-26"
DATA = ROOT / "data/tv_search_2026-10-21"
END = "2026-10-11"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
TRANSFER = ("BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT")
TIMEFRAMES = (30, 60, 240)
COST, STRESS = .0015, .003
PROFILES = {
    "v2": {},
    "trend": {"mode": "stonksy_trend"},
    "slow_hybrid": {"htf_anchor_minutes": 240},
    "slow_trend": {"htf_anchor_minutes": 240, "mode": "stonksy_trend"},
    "slow_price_flow": {"htf_anchor_minutes": 240, "mode": "stonksy_trend", "use_cvd_filter": False},
    "medium_price_flow": {"mode": "stonksy_trend", "use_cvd_filter": False},
    "fast_trend": {"htf_anchor_minutes": 0, "mode": "stonksy_trend"},
    "strict_trend": {"htf_anchor_minutes": 240, "mode": "stonksy_trend", "bg_threshold": 1.0},
}
SOURCES = ("research/tmg_improvement.py", "tmg_trades_weex.py", "tradingview/tmg_trades_strategy.pine")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, doc):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def candidates():
    return [{"id": f"{profile}_{minutes}_{direction}", "profile": profile,
             "minutes": minutes, "direction": direction, "parameters": params}
            for profile, params in PROFILES.items() for minutes in TIMEFRAMES
            for direction in ("both", "long_only")]


def lock():
    paths = list(SOURCES)
    for symbol in (*SYMBOLS, *TRANSFER):
        paths.extend([f"data/tv_search_2026-10-21/{symbol}_raw_1h.csv",
                      f"data/tv_search_2026-10-21/{symbol.lower()}_funding.csv",
                      f"data/30m/{symbol.lower()}_30m.csv"])
    document = {
        "version": "tmg_target_next_open_v1", "candidates": candidates(),
        "selection_markets": SYMBOLS, "transfer_markets": TRANSFER,
        "training": ["2020-01-01", "2024-01-01"], "validation": ["2024-01-01", "2025-01-01"],
        "known_control": ["2025-01-01", END], "cost_per_side": COST, "stress_cost_per_side": STRESS,
        "funding": "Faktiske Binance-hendelser, ved første åpning etter hendelsen; stress ×2.",
        "selection": "Høyest median stress-Sharpe i 2020–23 av forhåndsgodkjente kandidater; 2024 bare godkjenning.",
        "gates": {"btc_training_stress_positive": True, "training_median_stress_positive": True,
                  "btc_training_drawdown_above": -.55, "btc_training_trades_min": 20,
                  "btc_positive_training_years_min": 3, "btc_validation_normal_and_stress_positive": True,
                  "validation_median_stress_positive": True,
                  "beat_v2_same_timeframe_direction_on_training_and_validation": True},
        "control_rule": "Kontroll kjøres bare etter låst valg; feilet kontroll gir ingen automatisk erstatning.",
        "limitations": ["Binance spot som prisproxy for futures; ikke faktiske WEEX-fyll.",
                        "1× notional; ikke test av 50× eller likvidasjon.",
                        "Motorens mål handles ved neste åpning, også POC/close-stopp. Ikke native Pine-TP.",
                        "CLV er estimert volumdelta, ikke målt CVD eller likvidasjoner.",
                        "Ingen handel ved syntetiske lys, 800 barers inngangssperre etter hull.",
                        "Historikken har vært brukt i andre forsøk og er ikke urørt.",
                        "30m slutter 11. oktober; felles slutt for sammenlignbarhet."],
        "sha256": {p: sha(ROOT / p) for p in paths},
    }
    target = OUTPUT / "locked_plan.json"
    if target.exists():
        original = json.loads(target.read_text())
        repairs_path = OUTPUT / "implementation_repairs.json"
        repairs = json.loads(repairs_path.read_text()) if repairs_path.exists() else []
        for repair in repairs:
            key = repair["path"]
            if (original["sha256"].get(key) == repair["original_sha256"] and
                    document["sha256"].get(key) == repair["current_sha256"]):
                document["sha256"][key] = repair["original_sha256"]
        if original != json.loads(json.dumps(document)):
            raise ValueError("Den låste planen kan ikke endres.")
    else:
        write(target, document)
    return document


def load_market(symbol, minutes):
    path = (ROOT / f"data/30m/{symbol.lower()}_30m.csv" if minutes == 30
            else DATA / f"{symbol}_raw_1h.csv")
    raw = pd.read_csv(path, usecols=["open_time", "open", "high", "low", "close", "volume"])
    raw.index = pd.DatetimeIndex(pd.to_datetime(raw.pop("open_time"), utc=True, format="mixed"))
    raw = raw.sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]
    raw = raw.loc[(raw.index >= "2019-01-01") & (raw.index < END)]
    step = "30min" if minutes == 30 else "1h"
    grid = pd.date_range(raw.index[0], raw.index[-1], freq=step)
    bars = raw.reindex(grid)
    missing = bars.close.isna()
    bars.close = bars.close.ffill()
    for col in ("open", "high", "low"):
        bars[col] = bars[col].fillna(bars.close)
    bars.volume = bars.volume.fillna(0.)
    bars["synthetic"] = missing
    if minutes == 240:
        groups = bars.resample("4h", origin="epoch")
        result = groups.agg({"open": "first", "high": "max", "low": "min", "close": "last",
                             "volume": "sum", "synthetic": "max"})
        bars = result.loc[groups.size() == 4]
    synthetic = bars.pop("synthetic").to_numpy(bool)
    bars.index = bars.index.as_unit("ms")
    blocked = pd.Series(synthetic.astype(int)).rolling(801, min_periods=1).max().to_numpy(bool)
    funding = pd.read_csv(DATA / f"{symbol.lower()}_funding.csv")
    when = pd.DatetimeIndex(pd.to_datetime(funding.funding_time, utc=True, format="mixed")).as_unit("ms")
    positions = bars.index.searchsorted(when, side="left")
    rates = np.zeros(len(bars))
    inside = (when >= bars.index[0]) & (positions < len(bars))
    np.add.at(rates, positions[inside], funding.funding_rate.to_numpy(float)[inside])
    return {"bars": bars, "synthetic": synthetic, "blocked": blocked, "funding": rates,
            "symbol": symbol, "minutes": minutes}


def metrics(equity, trades, index, cost, fees, funding):
    curve = pd.Series(equity, index=index)
    daily = curve.resample("1D").last()
    daily = pd.concat([pd.Series([1.], index=[daily.index[0] - pd.Timedelta(days=1)]), daily])
    returns = daily.pct_change().dropna()
    std = returns.std()
    peaks = np.maximum.accumulate(np.r_[1., equity])[1:]
    yearly, previous = {}, 1.
    for year, value in curve.groupby(curve.index.year).last().items():
        yearly[str(year)] = float(value / previous - 1.)
        previous = value
    pnls = np.array([t["pnl"] for t in trades])
    return {"return": float(equity[-1] - 1.), "max_drawdown": float(np.min(equity / peaks - 1.)),
            "sharpe": float(returns.mean() / std * math.sqrt(365)) if std > 0 else 0.,
            "trades": len(trades), "win_rate": float((pnls > 0).mean()) if len(pnls) else None,
            "profit_factor": float(pnls[pnls > 0].sum() / -pnls[pnls < 0].sum()) if (pnls < 0).any() else None,
            "yearly": yearly, "fees": float(fees), "funding": float(funding), "cost_per_side": cost}


def simulate(market, frame, start, end, direction="both", cost=COST, funding_scale=1.):
    """Fast mengde per handel, kausale mål ved close og neste-open-fyll. Ingen gjeninntreden på gamle signaler."""
    bars = market["bars"]
    index = bars.index
    first = int(index.searchsorted(pd.Timestamp(start, tz="UTC")))
    last = int(index.searchsorted(pd.Timestamp(end, tz="UTC")))
    if last - first < 2:
        raise ValueError("For kort prisvindu")
    o, c = bars.open.to_numpy(float), bars.close.to_numpy(float)
    target = frame.position.to_numpy(int)
    fresh = (frame.buy_signal | frame.sell_signal).to_numpy(bool)
    side = wanted = 0
    cash, qty, entry, fees, paid = 1., 0., 0., 0., 0.
    entry_bar = 0
    equity = np.empty(last - first)
    trades = []
    before_trade = 1.
    for k in range(first, last):
        if side:
            amount = side * qty * o[k] * market["funding"][k] * funding_scale
            cash -= amount
            paid += amount
        if not market["synthetic"][k] and wanted != side:
            if side:
                fee = qty * o[k] * cost
                cash += side * qty * (o[k] - entry) - fee
                fees += fee
                trades.append({"side": side, "entry_bar": entry_bar, "exit_bar": k,
                               "entry_price": entry, "exit_price": float(o[k]),
                               "pnl": float(cash - before_trade), "terminal": False})
                side, qty = 0, 0.
            if wanted and cash > 0:
                before_trade = cash
                entry_bar, entry, side = k, float(o[k]), wanted
                qty = cash / (entry * (1. + cost))
                fee = qty * entry * cost
                cash -= fee
                fees += fee
        value = cash + side * qty * (c[k] - entry)
        equity[k - first] = max(value, 0.)
        if value <= 0:
            equity[k - first:] = 0.
            side = wanted = 0
            break
        t = int(target[k])
        if direction == "long_only" and t < 0:
            t = 0
        if side and t != side:
            wanted = t if not market["blocked"][k] else 0
        elif not side and t and fresh[k] and not market["blocked"][k]:
            wanted = t
        elif market["blocked"][k]:
            wanted = 0
        else:
            wanted = side
    if side:
        fee = qty * c[last - 1] * cost
        cash += side * qty * (c[last - 1] - entry) - fee
        fees += fee
        trades.append({"side": side, "entry_bar": entry_bar, "exit_bar": last - 1,
                       "entry_price": entry, "exit_price": float(c[last - 1]),
                       "pnl": float(cash - before_trade), "terminal": True})
        equity[-1] = max(cash, 0.)
    result = metrics(equity, trades, index[first:last], cost, fees, paid)
    return result, equity, trades


def evaluate(market, frame, period, direction):
    return {"normal": simulate(market, frame, *period, direction, COST)[0],
            "stress": simulate(market, frame, *period, direction, STRESS, 2.)[0]}


def develop():
    plan = lock()
    cache, frames, results = {}, {}, []
    for candidate in candidates():
        if candidate["direction"] == "both":
            frames.clear()
        datasets = {}
        for symbol in SYMBOLS:
            key = (symbol, candidate["minutes"])
            if key not in cache:
                cache[key] = load_market(*key)
            frame_key = (*key, candidate["profile"])
            if frame_key not in frames:
                frames[frame_key] = tmg.compute(cache[key]["bars"], minutes=key[1], **candidate["parameters"])
            frame = frames[frame_key]
            datasets[symbol] = {
                "training": evaluate(cache[key], frame, plan["training"], candidate["direction"]),
                "validation": evaluate(cache[key], frame, plan["validation"], candidate["direction"])}
        results.append({**candidate, "datasets": datasets})
        btc = datasets["BTCUSDT"]
        print(candidate["id"], "train", round(btc["training"]["normal"]["return"] * 100, 1),
              "validation", round(btc["validation"]["normal"]["return"] * 100, 1), flush=True)
    baselines = {(x["minutes"], x["direction"]): x for x in results if x["profile"] == "v2"}
    for row in results:
        training = [row["datasets"][s]["training"]["stress"] for s in SYMBOLS]
        validation = [row["datasets"][s]["validation"]["stress"] for s in SYMBOLS]
        btc = row["datasets"]["BTCUSDT"]
        base = baselines[row["minutes"], row["direction"]]["datasets"]["BTCUSDT"]
        row["score"] = float(np.median([r["sharpe"] for r in training]))
        gates = {
            "training_stress_btc": btc["training"]["stress"]["return"] > 0,
            "training_stress_median": np.median([r["return"] for r in training]) > 0,
            "training_drawdown": btc["training"]["normal"]["max_drawdown"] > -.55,
            "training_trades": btc["training"]["normal"]["trades"] >= 20,
            "training_years": sum(v > 0 for v in btc["training"]["normal"]["yearly"].values()) >= 3,
            "validation_btc": min(btc["validation"][k]["return"] for k in ("normal", "stress")) > 0,
            "validation_median": np.median([r["return"] for r in validation]) > 0,
            "training_improvement": btc["training"]["stress"]["return"] > base["training"]["stress"]["return"],
            "validation_improvement": btc["validation"]["stress"]["return"] > base["validation"]["stress"]["return"],
        }
        row["gates"] = {k: bool(v) for k, v in gates.items()}
        row["qualified"] = all(gates.values())
    ranked = sorted(results, key=lambda r: (r["qualified"], r["score"]), reverse=True)
    qualified = [r for r in ranked if r["qualified"]]
    write(OUTPUT / "development.json", ranked)
    choice = {"selected": qualified[0] if qualified else None,
              "plan_sha256": sha(OUTPUT / "locked_plan.json"),
              "development_sha256": sha(OUTPUT / "development.json")}
    write(OUTPUT / "frozen_choice.json", choice)
    print("VALGT", choice["selected"]["id"] if choice["selected"] else None, flush=True)


def control():
    plan = json.loads((OUTPUT / "locked_plan.json").read_text())
    choice = json.loads((OUTPUT / "frozen_choice.json").read_text())
    if choice["plan_sha256"] != sha(OUTPUT / "locked_plan.json"):
        raise ValueError("Endret plan")
    target = OUTPUT / "known_control.json"
    if target.exists():
        raise ValueError("Kontrollen finnes allerede; ikke velg på nytt.")
    selected = choice["selected"]
    checks = [{"id": "v2_60_both", "minutes": 60, "direction": "both", "parameters": {}},
              {"id": "v2_30_both", "minutes": 30, "direction": "both", "parameters": {}}]
    if selected:
        checks += [selected, {"id": "matched_v2", "minutes": selected["minutes"],
                              "direction": selected["direction"], "parameters": {}}]
    output = {}
    for candidate in checks:
        result = {}
        for symbol in (*SYMBOLS, *TRANSFER):
            market = load_market(symbol, candidate["minutes"])
            frame = tmg.compute(market["bars"], minutes=candidate["minutes"], **candidate["parameters"])
            result[symbol] = {"known_2025_26": evaluate(market, frame, plan["known_control"], candidate["direction"]),
                              "known_2026": evaluate(market, frame, ["2026-01-01", END], candidate["direction"])}
            for name, args in (("normal", (COST, 1.)), ("stress", (STRESS, 2.))):
                metric, curve, trades = simulate(market, frame, *plan["known_control"], candidate["direction"], *args)
                first = market["bars"].index.searchsorted(pd.Timestamp(plan["known_control"][0], tz="UTC"))
                pd.DataFrame({"open_time": market["bars"].index[first:first + len(curve)], "equity": curve}).to_csv(
                    OUTPUT / f"{candidate['id']}_{symbol}_{name}_equity.csv", index=False)
                pd.DataFrame(trades).to_csv(OUTPUT / f"{candidate['id']}_{symbol}_{name}_trades.csv", index=False)
        output[candidate["id"]] = result
        print(candidate["id"], result["BTCUSDT"]["known_2025_26"], flush=True)
    write(target, {"choice_sha256": sha(OUTPUT / "frozen_choice.json"), "results": output})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--develop", action="store_true")
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    if args.lock:
        lock()
    if args.develop:
        develop()
    if args.control:
        control()
