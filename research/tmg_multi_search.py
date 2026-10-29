"""Åtte forhåndslåste videohypoteser; kontrollen brukes ikke til rangering."""
from __future__ import annotations

import argparse
import json
import numpy as np

import tmg_multi_context as rules
from research import tmg_context_search as previous
from research import tmg_daily_search as daily
from research import tmg_channel_search as channel
from research import tmg_improvement as base

ROOT = base.ROOT
OUTPUT = ROOT / "reports/tmg_multi_2026-10-26"
SYMBOLS, TRAIN, VALID, CONTROL = daily.SYMBOLS, daily.TRAIN, daily.VALID, daily.CONTROL


def lock():
    paths = ["tmg_multi_context.py", "research/tmg_multi_search.py", "reports/tmg_multi_2026-10-26/SOURCE_NOTES.md"]
    paths += list(json.loads((daily.OUTPUT / "locked_plan.json").read_text())["sha256"])
    plan = {"version": "tmg_multi_close_next_open_v1", "profiles": list(rules.PROFILES),
            "training": TRAIN, "validation": VALID, "known_control": CONTROL,
            "reference": "Uendret daily50_weekend_both. Fast 1 time, begge retninger, ingen økt gearing.",
            "execution": "1×, signalclose/neste ekte open, 0,15 % per side og faktisk funding; stress dobler begge.",
            "hypotheses": "Ekte spotandel eller CLV-proxy, begge 4/24h enige; motsatt som utgang. Kjent trehendelsers funding-snitt: inngang <=0 eller <=0,01% i posisjonens retning; utgang >0,03%.",
            "funding_timing": "Bare publiserte realiserte rater minst en time gamle, aldri estimert kommende rate.",
            "gates": "BTC trening stress>0,>=15handler,>=2positiveår,DD>-55%,medianstress>0. BTC/median2024stress>0. BTCstress bedre enn dagsfilter i begge perioder.",
            "ranking": "Median stress-Sharpe i trening på tre markeder blant godkjente. Ingen kontrollrangering.",
            "placebo": "Samme åtte tidligere kontekstskift 720–3240h, trening over75persentil,2024overmedian.",
            "release": "Utvikling+placebo, BTC kontroll normal/stress positiv og bedre, medianstress>0,minst2positive,BTC-DD ikke verre. Manuell forskningsvisning kan være aktuell; ingen motoraktivering.",
            "no_choice": "Ingen godkjent utviklingskandidat: ikke kjør ny kontrollrangering, behold signalregler.",
            "limitations": ["Retrospektiv historie brukt mange ganger; ingen urørt test.", "Egne formler, ikke kanalens komplette regler eller verifiserte avkastning.",
                            "Binance alene, ikke global spot/OI. Taker, CLV og funding er ulike størrelser.",
                            "Ekte taker/funding er ikke tilgjengelig som identiske Pine-serier i vanlig OHLCV. Ingen falsk port.",
                            "Ingen native TradingView-/WEEX-fyll, børs-SL/TP eller gearingtest."],
            "sha256": {p: base.sha(ROOT / p) for p in sorted(set(paths))}}
    path = OUTPUT / "locked_plan.json"
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(plan)), "Låst plan er endret"
    else:
        base.write(path, plan)
    return plan


def contexts():
    data, flow, originals, context = daily.contexts()
    frames, values = {}, {}
    for s in SYMBOLS:
        frames[s] = daily.filtered(originals[s], context[s], flow[s], {"profile": "daily50_weekend", "direction": "both"})
        spot = channel.read_bars(base.DATA / f"{s}_raw_1h.csv").reindex(data[s]["bars"].index)
        values[s] = rules.features(spot, flow[s])
    return data, frames, values


def develop():
    lock()
    data, frames, values = contexts()
    refs = {s: {"training": base.evaluate(data[s], frames[s], TRAIN, "both"), "validation": base.evaluate(data[s], frames[s], VALID, "both")} for s in SYMBOLS}
    rows = []
    for profile in rules.PROFILES:
        result = {}
        for s in SYMBOLS:
            frame = rules.apply(frames[s], values[s], profile)
            result[s] = {"training": base.evaluate(data[s], frame, TRAIN, "both"), "validation": base.evaluate(data[s], frame, VALID, "both")}
        b, ref = result["BTCUSDT"], refs["BTCUSDT"]
        gates = {"train_positive": b["training"]["stress"]["return"] > 0,
                 "train_trades": b["training"]["normal"]["trades"] >= 15,
                 "train_years": sum(v > 0 for v in b["training"]["normal"]["yearly"].values()) >= 2,
                 "train_dd": b["training"]["normal"]["max_drawdown"] > -.55,
                 "train_median": np.median([r["training"]["stress"]["return"] for r in result.values()]) > 0,
                 "valid_btc": b["validation"]["stress"]["return"] > 0,
                 "valid_median": np.median([r["validation"]["stress"]["return"] for r in result.values()]) > 0,
                 "train_improved": b["training"]["stress"]["return"] > ref["training"]["stress"]["return"],
                 "valid_improved": b["validation"]["stress"]["return"] > ref["validation"]["stress"]["return"]}
        row = {"id": profile, "datasets": result, "gates": {k: bool(v) for k,v in gates.items()}, "qualified": bool(all(gates.values())),
               "score": float(np.median([r["training"]["stress"]["sharpe"] for r in result.values()]))}
        rows.append(row)
        print(profile, round(b["training"]["normal"]["return"]*100,2), round(b["validation"]["normal"]["return"]*100,2), row["qualified"], flush=True)
    rows.sort(key=lambda r: (r["qualified"], r["score"]), reverse=True)
    base.write(OUTPUT / "development.json", {"candidates": rows, "references": refs})
    base.write(OUTPUT / "frozen_choice.json", {"selected": rows[0] if rows[0]["qualified"] else None,
               "plan_sha256": base.sha(OUTPUT / "locked_plan.json"), "development_sha256": base.sha(OUTPUT / "development.json")})
    print("VALGT", rows[0]["id"] if rows[0]["qualified"] else None, flush=True)


def control():
    path = OUTPUT / "known_control.json"
    if path.exists():
        raise ValueError("Kontrollen er registrert; ingen ny kjøring")
    plan = lock()
    for p,h in plan["sha256"].items():
        assert base.sha(ROOT / p) == h, p
    choice = json.loads((OUTPUT / "frozen_choice.json").read_text())
    assert choice["plan_sha256"] == base.sha(OUTPUT / "locked_plan.json")
    assert choice["development_sha256"] == base.sha(OUTPUT / "development.json")
    selected = choice["selected"]
    if selected is None:
        base.write(path, {"selected": None, "released": False, "reason": "Ingen bestod utvikling/2024; ingen ny kontroll kjørt."})
        return
    data, frames, values = contexts()
    nulls = []
    for h in previous.SHIFTS:
        frame = rules.apply(frames["BTCUSDT"], values["BTCUSDT"].shift(h), selected["id"])
        nulls.append({"hours": h, "training": base.evaluate(data["BTCUSDT"], frame, TRAIN, "both"), "validation": base.evaluate(data["BTCUSDT"], frame, VALID, "both")})
    b = selected["datasets"]["BTCUSDT"]
    placebo = b["training"]["stress"]["return"] > np.quantile([r["training"]["stress"]["return"] for r in nulls], .75) and b["validation"]["stress"]["return"] > np.median([r["validation"]["stress"]["return"] for r in nulls])
    base.write(OUTPUT / "placebo.json", {"passed": bool(placebo), "shifts": nulls})
    results = {s: {"daily": base.evaluate(data[s], frames[s], CONTROL, "both"),
                   "candidate": base.evaluate(data[s], rules.apply(frames[s], values[s], selected["id"]), CONTROL, "both")} for s in SYMBOLS}
    b, ref = results["BTCUSDT"]["candidate"], results["BTCUSDT"]["daily"]
    gates = {"placebo": bool(placebo), "btc_positive": all(b[k]["return"] > 0 for k in ("normal","stress")),
             "btc_improved": all(b[k]["return"] > ref[k]["return"] for k in ("normal","stress")),
             "median_positive": np.median([r["candidate"]["stress"]["return"] for r in results.values()]) > 0,
             "two_positive": sum(r["candidate"]["stress"]["return"] > 0 for r in results.values()) >= 2,
             "drawdown": b["normal"]["max_drawdown"] >= ref["normal"]["max_drawdown"]}
    base.write(path, {"selected": selected["id"], "results": results, "gates": {k: bool(v) for k,v in gates.items()}, "released": bool(all(gates.values()))})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--develop", action="store_true")
    p.add_argument("--control", action="store_true")
    args = p.parse_args()
    if args.develop:
        develop()
    if args.control:
        control()
