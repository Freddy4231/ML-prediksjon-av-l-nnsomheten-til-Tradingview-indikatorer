"""Låst, separat test av dagsbekreftelse; 2025–26 er allerede kjent historikk."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import tmg_daily_confirmation as daily
from research import tmg_channel_search as previous
from research import tmg_channel_rules as flow_rules
from research import tmg_improvement as base

ROOT = base.ROOT
OUTPUT = ROOT / "reports/tmg_daily_2026-10-26"
SYMBOLS, TRAIN, VALID, CONTROL = previous.source.SYMBOLS, previous.TRAIN, previous.VALID, previous.CONTROL


def candidates():
    return [{"id": f"{profile}_{direction}", "profile": profile, "direction": direction}
            for profile in daily.PROFILES for direction in ("both", "long_only")]


def lock():
    paths = ["tmg_daily_confirmation.py", "research/tmg_daily_search.py", "research/tmg_channel_search.py",
             "research/tmg_channel_rules.py", "research/tmg_improvement.py", "tmg_trades_weex.py",
             "reports/tmg_daily_2026-10-26/SOURCE_NOTES.md"]
    paths += list(json.loads((previous.OUTPUT / "locked_plan.json").read_text())["sha256"])
    doc = {"version": "tmg_daily_confirmed_next_open_v1", "candidates": candidates(),
           "training": TRAIN, "validation": VALID, "known_control": CONTROL,
           "timeframe_minutes": 60, "symbols": SYMBOLS,
           "daily_timing": "Bare forrige komplette UTC-døgn kjent ved timebars åpning. 24 ekte timer kreves per døgn.",
           "entry": "Bare et ferskt TMG V2-signal; avviste signaler blir ikke gamle seninnganger. Utganger uendret.",
           "costs": "1× notional, 0,15 % per side + faktisk funding; stress 0,30 % og funding ×2.",
           "selection": "Median trenings-stress-Sharpe på BTC/ETH/SOL blant kvalifiserte; godkjenning kun i 2024.",
           "gates": "BTC trening stress>0, minst15 handler, minst2 positive år, DD>-55%, median stress>0; "
                    "BTC og median stress>0 i2024. Bedre BTC stress enn samme V2-retning i begge perioder.",
           "fallback": "Hvis ingen kvalifiseres: vis høyest treningsscore som eksplorativ indikator, uten godkjenning.",
           "placebo": "Valgt BTC trenings-stress over75-persentil av8 fortidsskift;2024 overmedian. Ingen fremtidsskift.",
           "release": "Kun kvalifisert valg+placebo og BTC normal/stress positiv og bedre enn V2; "
                      "median stress positiv, minst2 positive, BTC DD ikke verre. Ingen automatisk motorendring.",
           "limitations": ["Retrospektiv; de samme kontrollårene er brukt tidligere. Ingen urørt holdout.",
                           "Videoene gir skjønnsmessig analyse, ikke komplette eller verifiserte formler.",
                           "Python neste-open-mål; ikke native Pine-/WEEX-ordre, full børs-SL/TP eller gearing.",
                           "Base-TMG bruker CLV, ikke ekte taker-flagg. Ingen OI/likvidasjons-/ETF-proxy."],
           "sha256": {p: base.sha(ROOT / p) for p in sorted(set(paths))}}
    path = OUTPUT / "locked_plan.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(doc)), "Låst plan er endret"
    else:
        base.write(path, doc)
    return doc


def contexts():
    data, features, frames = previous.contexts()
    context = {s: daily.daily_context(data[s]["bars"], ~data[s]["synthetic"]) for s in SYMBOLS}
    return data, features, frames, context


def filtered(frame, context, flow, candidate):
    long, short = daily.allowed(context, candidate["profile"], flow)
    return daily.apply_filter(frame, long, short, candidate["direction"])


def develop():
    lock()
    data, flow, frames, context = contexts()
    references = {d: {s: {"training": base.evaluate(data[s], frames[s], TRAIN, d),
                           "validation": base.evaluate(data[s], frames[s], VALID, d)} for s in SYMBOLS}
                  for d in ("both", "long_only")}
    rows = []
    for c in candidates():
        result = {}
        for s in SYMBOLS:
            frame = filtered(frames[s], context[s], flow[s], c)
            result[s] = {"training": base.evaluate(data[s], frame, TRAIN, c["direction"]),
                         "validation": base.evaluate(data[s], frame, VALID, c["direction"])}
        b, ref = result["BTCUSDT"], references[c["direction"]]["BTCUSDT"]
        gates = {"train_positive": b["training"]["stress"]["return"] > 0,
                 "train_trades": b["training"]["normal"]["trades"] >= 15,
                 "train_years": sum(v > 0 for v in b["training"]["normal"]["yearly"].values()) >= 2,
                 "train_dd": b["training"]["normal"]["max_drawdown"] > -.55,
                 "train_median": np.median([r["training"]["stress"]["return"] for r in result.values()]) > 0,
                 "valid_btc": b["validation"]["stress"]["return"] > 0,
                 "valid_median": np.median([r["validation"]["stress"]["return"] for r in result.values()]) > 0,
                 "train_improvement": b["training"]["stress"]["return"] > ref["training"]["stress"]["return"],
                 "valid_improvement": b["validation"]["stress"]["return"] > ref["validation"]["stress"]["return"]}
        rows.append({**c, "datasets": result, "gates": {k: bool(v) for k, v in gates.items()},
                     "qualified": bool(all(gates.values())),
                     "score": float(np.median([r["training"]["stress"]["sharpe"] for r in result.values()]))})
        print(c["id"], "trening", round(b["training"]["normal"]["return"] * 100, 2),
              "2024", round(b["validation"]["normal"]["return"] * 100, 2), "godkjent", all(gates.values()), flush=True)
    rows.sort(key=lambda r: (r["qualified"], r["score"]), reverse=True)
    base.write(OUTPUT / "development.json", {"candidates": rows, "references": references})
    base.write(OUTPUT / "frozen_choice.json", {"selected": rows[0], "qualified": rows[0]["qualified"],
               "plan_sha256": base.sha(OUTPUT / "locked_plan.json"),
               "development_sha256": base.sha(OUTPUT / "development.json")})
    print("FRYST VALG", rows[0]["id"], flush=True)


def control():
    target = OUTPUT / "known_control.json"
    if target.exists():
        raise ValueError("Kontrollen finnes allerede; ingen ny rangering")
    plan = lock()
    for p, h in plan["sha256"].items():
        assert base.sha(ROOT / p) == h, p
    choice = json.loads((OUTPUT / "frozen_choice.json").read_text())
    assert choice["plan_sha256"] == base.sha(OUTPUT / "locked_plan.json")
    assert choice["development_sha256"] == base.sha(OUTPUT / "development.json")
    selected = choice["selected"]
    data, flow, frames, context = contexts()
    long, short = daily.allowed(context["BTCUSDT"], selected["profile"], flow["BTCUSDT"])
    placebo = []
    for hours in previous.SHIFTS:
        frame = daily.apply_filter(frames["BTCUSDT"], long.shift(hours, fill_value=False),
                                   short.shift(hours, fill_value=False), selected["direction"])
        placebo.append({"hours": hours, "training": base.evaluate(data["BTCUSDT"], frame, TRAIN, selected["direction"]),
                        "validation": base.evaluate(data["BTCUSDT"], frame, VALID, selected["direction"])})
    b = selected["datasets"]["BTCUSDT"]
    placebo_ok = b["training"]["stress"]["return"] > np.quantile([p["training"]["stress"]["return"] for p in placebo], .75) and b["validation"]["stress"]["return"] > np.median([p["validation"]["stress"]["return"] for p in placebo])
    base.write(OUTPUT / "placebo.json", {"passed": bool(placebo_ok), "shifts": placebo})
    results = {}
    for s in SYMBOLS:
        frames_to_test = {"v2": frames[s], "daily": filtered(frames[s], context[s], flow[s], selected)}
        l, sh = flow_rules.allowed(flow[s], "weekend_confidence")
        frames_to_test["weekend"] = daily.apply_filter(frames[s], l, sh, selected["direction"])
        results[s] = {}
        for key, frame in frames_to_test.items():
            results[s][key] = base.evaluate(data[s], frame, CONTROL, selected["direction"])
            _, curve, trades = base.simulate(data[s], frame, *CONTROL, selected["direction"])
            import pandas as pd
            first = data[s]["bars"].index.searchsorted(pd.Timestamp(CONTROL[0], tz="UTC"))
            pd.DataFrame({"time": data[s]["bars"].index[first:first + len(curve)] + pd.Timedelta(hours=1),
                          "equity": curve}).to_csv(OUTPUT / f"{s}_{key}_equity.csv", index=False)
            pd.DataFrame(trades).to_csv(OUTPUT / f"{s}_{key}_trades.csv", index=False)
        print(s, {k: {x: round(r[x]["return"] * 100, 2) for x in ("normal", "stress")}
                  for k, r in results[s].items()}, flush=True)
    b, ref = results["BTCUSDT"]["daily"], results["BTCUSDT"]["v2"]
    gates = {"development": choice["qualified"], "placebo": bool(placebo_ok),
             "btc_positive": all(b[k]["return"] > 0 for k in ("normal", "stress")),
             "btc_improved": all(b[k]["return"] > ref[k]["return"] for k in ("normal", "stress")),
             "median_positive": np.median([r["daily"]["stress"]["return"] for r in results.values()]) > 0,
             "two_positive": sum(r["daily"]["stress"]["return"] > 0 for r in results.values()) >= 2,
             "drawdown": b["normal"]["max_drawdown"] >= ref["normal"]["max_drawdown"]}
    base.write(target, {"choice_sha256": base.sha(OUTPUT / "frozen_choice.json"), "selected": selected["id"],
                       "results": results, "gates": {k: bool(v) for k, v in gates.items()}, "released": bool(all(gates.values()))})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--develop", action="store_true")
    p.add_argument("--control", action="store_true")
    args = p.parse_args()
    if args.develop:
        develop()
    if args.control:
        control()
