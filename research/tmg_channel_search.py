"""Låst test av avgrensede, kildefestede TMG-hypoteser på reelle perpetualdata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import tmg_channel_data as source
from research import tmg_channel_rules as rules
from research import tmg_improvement as base

ROOT, OUTPUT, DATA = source.ROOT, source.REPORT, source.DATA
TRAIN, VALID, CONTROL = ("2021-01-01", "2024-01-01"), ("2024-01-01", "2025-01-01"), ("2025-01-01", "2026-10-01")
SHIFTS = (720, 1080, 1440, 1800, 2160, 2520, 2880, 3240)


def candidates():
    return [{"id": f"{profile}_{direction}", "profile": profile, "direction": direction}
            for profile in rules.PROFILES for direction in ("both", "long_only")]


def lock():
    paths = ["research/tmg_channel_data.py", "research/tmg_channel_rules.py", "research/tmg_channel_search.py",
             "research/tmg_improvement.py", "tmg_trades_weex.py", "tradingview/tmg_trades_strategy.pine",
             "reports/tmg_channel_2026-10-26/SOURCE_NOTES.md", "reports/tmg_channel_2026-10-26/data_manifest.json"]
    for s in source.SYMBOLS:
        paths += [f"data/tmg_channel_2026-10-26/{s}_perp_1h.csv",
                  f"data/tv_search_2026-10-21/{s}_raw_1h.csv",
                  f"data/tv_search_2026-10-21/{s.lower()}_funding.csv"]
    doc = {"version": "tmg_channel_real_flow_next_open_v1", "candidates": candidates(),
           "training": TRAIN, "validation": VALID, "known_control": CONTROL,
           "timeframe_minutes": 60, "symbols": source.SYMBOLS, "costs_per_side": [base.COST, base.STRESS],
           "price_source": "Binance USD-M perpetual, ikke spotproxy. Ekte basevolum/quotevolum beholdes.",
           "execution": "Samme close-mål/neste ekte åpning som tidligere TMG-test; fast 1× notional.",
           "gates": "BTC trening normal/stress>0, minst 15 handler og to positive år, DD>-55%; median stress>0. "
                    "2024 BTC og median stress>0. Bedre BTC stress enn identisk V2 i begge perioder.",
           "ranking": "Median stress-Sharpe 2021–23 på BTC/ETH/SOL blant godkjente; ingen kontrollrangering.",
           "placebo": {"past_hour_shifts": SHIFTS, "require": "Valgt BTC trenings-stress over 75-persentil av åtte placeboer, 2024 over median."},
           "release": "Valgt BTC 2025–26 normal/stress>0 og bedre enn V2, median stress>0, minst to av tre positive, "
                      "BTC normal-DD ikke verre enn V2. Placebo må bestås. Ingen automatisk oppstart.",
           "limitations": ["2025–26 og videoene er kjent; ingen urørt holdout.",
                           "Én børs, basevolumforhold, ikke kanalens globale USD-aggregasjon.",
                           "Ikke faktiske OI-/ETF-/likvidasjonsdata. Ingen proxy kalles slik.",
                           "Motorens SL/TP, gearing/likvidasjon og fyllavvik er ikke fullstendig modellert.",
                           "Python-POC lukkes neste åpning; Pine har hvilende limit. Ingen native Pine-test.",
                           "Ekte taker-volum kan ikke erstattes med candle-farge i en påstått identisk Pine-port."],
           "sha256": {p: base.sha(ROOT / p) for p in paths}}
    path = OUTPUT / "locked_plan.json"
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(doc)):
            raise ValueError("Den låste planen kan ikke endres")
    else:
        base.write(path, doc)
    return doc


def read_bars(path):
    data = pd.read_csv(path)
    data.index = pd.DatetimeIndex(pd.to_datetime(data.pop("open_time"), utc=True, format="mixed")).as_unit("ms")
    return data.loc[~data.index.duplicated(keep="last")].sort_index()


def market(symbol):
    raw = read_bars(DATA / f"{symbol}_perp_1h.csv")
    raw = raw.loc[raw.index < pd.Timestamp(CONTROL[1], tz="UTC")]
    grid = pd.date_range(raw.index[0], raw.index[-1], freq="1h").as_unit("ms")
    real = raw.reindex(grid)
    spot = read_bars(base.DATA / f"{symbol}_raw_1h.csv").reindex(grid)
    synthetic = real.close.isna().to_numpy()
    missing = synthetic | spot.close.isna().to_numpy()
    bars = real[["open", "high", "low", "close", "volume"]].copy()
    bars.close = bars.close.ffill()
    for c in ("open", "high", "low"):
        bars[c] = bars[c].fillna(bars.close)
    bars.volume = bars.volume.fillna(0.)
    blocked = pd.Series(missing.astype(int)).rolling(801, min_periods=1).max().to_numpy(bool)
    funding = pd.read_csv(base.DATA / f"{symbol.lower()}_funding.csv")
    dates = pd.DatetimeIndex(pd.to_datetime(funding.funding_time, utc=True, format="mixed")).as_unit("ms")
    positions = grid.searchsorted(dates, side="left")
    rates = np.zeros(len(grid))
    inside = (dates >= grid[0]) & (positions < len(grid))
    np.add.at(rates, positions[inside], funding.funding_rate.to_numpy(float)[inside])
    return {"bars": bars, "funding": rates, "synthetic": synthetic, "blocked": blocked,
            "symbol": symbol, "minutes": 60}, rules.real_flow_features(spot, real, funding)


def contexts():
    data, features, frames = {}, {}, {}
    for s in source.SYMBOLS:
        data[s], features[s] = market(s)
        frames[s] = base.tmg.compute(data[s]["bars"], minutes=60)
    return data, features, frames


def develop():
    lock()
    data, features, frames = contexts()
    rows = []
    for c in candidates():
        result = {}
        for s in source.SYMBOLS:
            long, short = rules.allowed(features[s], c["profile"])
            filtered = rules.apply_filter(frames[s], long, short)
            result[s] = {"training": base.evaluate(data[s], filtered, TRAIN, c["direction"]),
                         "validation": base.evaluate(data[s], filtered, VALID, c["direction"])}
        rows.append({**c, "datasets": result})
        btc = result["BTCUSDT"]
        print(c["id"], "trening", round(100 * btc["training"]["normal"]["return"], 2),
              "2024", round(100 * btc["validation"]["normal"]["return"], 2), flush=True)
    refs = {r["direction"]: r["datasets"] for r in rows if r["profile"] == "v2"}
    for r in rows:
        b, ref = r["datasets"]["BTCUSDT"], refs[r["direction"]]["BTCUSDT"]
        gates = {"btc_training_positive": all(b["training"][k]["return"] > 0 for k in ("normal", "stress")),
                 "btc_training_trades": b["training"]["normal"]["trades"] >= 15,
                 "btc_training_years": sum(v > 0 for v in b["training"]["normal"]["yearly"].values()) >= 2,
                 "btc_training_dd": b["training"]["normal"]["max_drawdown"] > -.55,
                 "training_median": np.median([d["training"]["stress"]["return"] for d in r["datasets"].values()]) > 0,
                 "btc_validation": b["validation"]["stress"]["return"] > 0,
                 "validation_median": np.median([d["validation"]["stress"]["return"] for d in r["datasets"].values()]) > 0,
                 "training_improvement": b["training"]["stress"]["return"] > ref["training"]["stress"]["return"],
                 "validation_improvement": b["validation"]["stress"]["return"] > ref["validation"]["stress"]["return"]}
        r["gates"], r["qualified"] = {k: bool(v) for k, v in gates.items()}, bool(all(gates.values()))
        r["score"] = float(np.median([d["training"]["stress"]["sharpe"] for d in r["datasets"].values()]))
    rows.sort(key=lambda r: (r["qualified"], r["score"]), reverse=True)
    base.write(OUTPUT / "development.json", rows)
    selected = rows[0] if rows[0]["qualified"] else None
    base.write(OUTPUT / "frozen_choice.json", {"selected": selected, "plan_sha256": base.sha(OUTPUT / "locked_plan.json"),
                                             "development_sha256": base.sha(OUTPUT / "development.json")})
    print("VALGT", selected["id"] if selected else None, flush=True)


def control():
    target = OUTPUT / "known_control.json"
    if target.exists():
        raise ValueError("Kontrollen finnes allerede; ingen ny rangering")
    choice = json.loads((OUTPUT / "frozen_choice.json").read_text())
    selected = choice["selected"]
    data, features, frames = contexts()
    checks = [{"id": "v2_both", "profile": "v2", "direction": "both"},
              {"id": "v2_long_only", "profile": "v2", "direction": "long_only"}]
    placebo = None
    if selected:
        checks.append(selected)
        long, short = rules.allowed(features["BTCUSDT"], selected["profile"])
        values = []
        for hours in SHIFTS:
            shifted = rules.apply_filter(frames["BTCUSDT"], long.shift(hours, fill_value=False), short.shift(hours, fill_value=False))
            values.append({"hours": hours, "training": base.evaluate(data["BTCUSDT"], shifted, TRAIN, selected["direction"]),
                           "validation": base.evaluate(data["BTCUSDT"], shifted, VALID, selected["direction"])})
        train = selected["datasets"]["BTCUSDT"]["training"]["stress"]["return"]
        valid = selected["datasets"]["BTCUSDT"]["validation"]["stress"]["return"]
        passed = train > np.quantile([v["training"]["stress"]["return"] for v in values], .75) and valid > np.median([v["validation"]["stress"]["return"] for v in values])
        placebo = {"passed": bool(passed), "shifts": values}
        base.write(OUTPUT / "placebo.json", placebo)
    result = {}
    for c in checks:
        result[c["id"]] = {}
        for s in source.SYMBOLS:
            long, short = rules.allowed(features[s], c["profile"])
            filtered = rules.apply_filter(frames[s], long, short)
            result[c["id"]][s] = {"known_2025_26": base.evaluate(data[s], filtered, CONTROL, c["direction"]),
                                   "known_2026": base.evaluate(data[s], filtered, ("2026-01-01", CONTROL[1]), c["direction"])}
            _, curve, trades = base.simulate(data[s], filtered, *CONTROL, c["direction"])
            first = data[s]["bars"].index.searchsorted(pd.Timestamp(CONTROL[0], tz="UTC"))
            pd.DataFrame({"time": data[s]["bars"].index[first:first + len(curve)] + pd.Timedelta(hours=1),
                          "equity": curve}).to_csv(OUTPUT / f"{c['id']}_{s}_equity.csv", index=False)
            pd.DataFrame(trades).to_csv(OUTPUT / f"{c['id']}_{s}_trades.csv", index=False)
        print(c["id"], result[c["id"]]["BTCUSDT"]["known_2025_26"], flush=True)
    gates = {}
    if selected:
        tested = result[selected["id"]]
        b = tested["BTCUSDT"]["known_2025_26"]
        ref = result[f"v2_{selected['direction']}"]["BTCUSDT"]["known_2025_26"]
        gates = {"placebo": placebo["passed"], "btc_positive": all(b[k]["return"] > 0 for k in ("normal", "stress")),
                 "btc_improved": all(b[k]["return"] > ref[k]["return"] for k in ("normal", "stress")),
                 "median_positive": np.median([d["known_2025_26"]["stress"]["return"] for d in tested.values()]) > 0,
                 "two_positive": sum(d["known_2025_26"]["stress"]["return"] > 0 for d in tested.values()) >= 2,
                 "dd_not_worse": b["normal"]["max_drawdown"] >= ref["normal"]["max_drawdown"]}
    base.write(target, {"choice_sha256": base.sha(OUTPUT / "frozen_choice.json"), "results": result,
                       "gates": {k: bool(v) for k, v in gates.items()}, "released": bool(selected is not None and all(gates.values()))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--develop", action="store_true")
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    if args.develop:
        develop()
    if args.control:
        control()
