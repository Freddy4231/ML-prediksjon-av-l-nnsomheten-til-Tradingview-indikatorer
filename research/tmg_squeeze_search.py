"""Åtte låste videotolkninger, sammenlignet med den fryste OI-kandidaten."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import tmg_position_context as position
import tmg_squeeze_context as rules
from research import tmg_position_search as previous
from research import tmg_channel_search as channel
from research import tmg_improvement as base

ROOT = base.ROOT
OUTPUT = ROOT / "reports/tmg_squeeze_2026-10-26"
SYMBOLS, TRAIN, VALID = previous.SYMBOLS, previous.TRAIN, previous.VALID


def lock():
    paths = list(json.loads((previous.OUTPUT / "locked_plan.json").read_text())["sha256"])
    paths += ["research/tmg_squeeze_search.py", "tmg_squeeze_context.py",
              "reports/tmg_squeeze_2026-10-26/SOURCE_NOTES.md", "reports/tmg_oi_2026-10-26/locked_plan.json",
              "reports/tmg_oi_2026-10-26/frozen_choice.json"]
    plan = {"version": "tmg_squeeze_next_open_v1", "profiles": list(rules.PROFILES),
            "training": TRAIN, "validation": VALID, "references": ["daily50_weekend_both", "pending_oi4"],
            "execution": "1h, begge retninger, 1× fast mengde, neste ekte open. 0,15%/side + funding, stress dobler begge.",
            "entry": "Kun ferskt dagsfiltersignal, maks6h venting, kansellering ved målendring/hullsperre/utløp. Ingen ny inngang etter ekstra utgang før neste ferske signal.",
            "rules": rules.PROFILES,
            "selection": "Høyest median treningsstress-Sharpe blant kvalifiserte; 2024 kun godkjenning.",
            "gates": "BTC treningstress>0,min15handler,min2positiveår,DD>-55%; medianstress>0 i begge perioder; BTC2024stress>0; BTCstress bedre enn pending_oi4 i begge perioder; medianstress minst like god i begge; BTC2024DD minst like god.",
            "placebo": "Kun valgt kandidat. Åtte fortidsskift720–3240h av kontekst, virkelig eligibility beholdes; felles tilgjengelighetsmaske for alle. BTC treningstress over75persentil og2024overmedian. Tester hele regelen, ikke isolert inkrement.",
            "control": "Ingen2025–26-kjøring: OI-publisering og native feed fortsatt uverifisert. Ingen automatisk erstatning.",
            "limitations": ["Kjent, ofte gjenbrukt historikk. Tilleggsreglene bygger på tidligere valg.",
                            "OI-arkivtid+65min er antakelse, ikke historisk publiseringsbevis.",
                            "1× Python-simulering, ikke TradingView-/WEEX-fyll eller likvidasjon.",
                            "Ekte spot-takervolum finnes ikke som identisk serie i dagens Pine."],
            "sha256": {p: base.sha(ROOT / p) for p in sorted(set(paths))}}
    path = OUTPUT / "locked_plan.json"
    if path.exists():
        old = json.loads(path.read_text())
        assert {k:v for k,v in old.items() if k != "locked_at_utc"} == json.loads(json.dumps(plan)), "Låst plan endret"
        return old
    plan["locked_at_utc"] = datetime.now(timezone.utc).isoformat()
    base.write(path, plan)
    return plan


def contexts():
    data, daily, values = previous.contexts()
    oi, new = {}, {}
    for s in SYMBOLS:
        bars = data[s]["bars"]
        spot = channel.read_bars(base.DATA / f"{s}_raw_1h.csv").reindex(bars.index)
        new[s] = rules.features(bars, spot, values[s])
        oi[s] = position.apply(daily[s], values[s], "pending_oi4")
    return data, daily, oi, new


def periods(market, frame):
    return {"training": base.evaluate(market, frame, TRAIN, "both"),
            "validation": base.evaluate(market, frame, VALID, "both")}


def median(results, period):
    return float(np.median([r[period]["stress"]["return"] for r in results.values()]))


def develop():
    if (OUTPUT / "development.json").exists():
        raise ValueError("Dette forsøket er allerede kjørt; ikke overskriv resultater")
    lock()
    data, daily, oi, values = contexts()
    refs = {name: {s: periods(data[s], frames[s]) for s in SYMBOLS}
            for name, frames in (("daily", daily), ("oi", oi))}
    rows = []
    for name in rules.PROFILES:
        results = {s: periods(data[s], rules.apply(daily[s], values[s], name)) for s in SYMBOLS}
        b, ref = results["BTCUSDT"], refs["oi"]["BTCUSDT"]
        gates = {"train_positive": b["training"]["stress"]["return"] > 0,
                 "train_trades": b["training"]["normal"]["trades"] >= 15,
                 "train_years": sum(v > 0 for v in b["training"]["normal"]["yearly"].values()) >= 2,
                 "train_dd": b["training"]["normal"]["max_drawdown"] > -.55,
                 "train_median": median(results, "training") > 0,
                 "valid_btc": b["validation"]["stress"]["return"] > 0,
                 "valid_median": median(results, "validation") > 0,
                 "train_improved": b["training"]["stress"]["return"] > ref["training"]["stress"]["return"],
                 "valid_improved": b["validation"]["stress"]["return"] > ref["validation"]["stress"]["return"],
                 "train_cross": median(results, "training") >= median(refs["oi"], "training"),
                 "valid_cross": median(results, "validation") >= median(refs["oi"], "validation"),
                 "valid_dd": b["validation"]["normal"]["max_drawdown"] >= ref["validation"]["normal"]["max_drawdown"]}
        rows.append({"id": name, "datasets": results, "gates": {k:bool(v) for k,v in gates.items()},
                     "qualified": bool(all(gates.values())),
                     "score": float(np.median([r["training"]["stress"]["sharpe"] for r in results.values()]))})
        print(name, {p:round(b[p]["normal"]["return"]*100,2) for p in ("training","validation")},
              "bestått", all(gates.values()), flush=True)
    rows.sort(key=lambda r:(r["qualified"],r["score"]), reverse=True)
    base.write(OUTPUT / "development.json", {"references": refs, "candidates": rows})
    selected = rows[0]["id"] if rows[0]["qualified"] else None
    base.write(OUTPUT / "frozen_choice.json", {"selected": selected, "released": False,
               "plan_sha256": base.sha(OUTPUT / "locked_plan.json"),
               "development_sha256": base.sha(OUTPUT / "development.json")})
    check = {"selected": selected, "released": False, "control_run": False,
             "reason": "OI-publisering/native feed ubekreftet. Ingen2025–26-kjøring eller motorendring."}
    if selected:
        original = values["BTCUSDT"]
        shifted = [original.shift(h) for h in channel.SHIFTS]
        valid = original[rules.REQUIRED[selected]].notna().all(axis=1)
        for v in shifted:
            valid &= v[rules.REQUIRED[selected]].notna().all(axis=1)
        eligible = original.eligible & valid
        actual = original.copy()
        actual["eligible"] = eligible
        actual_stats = periods(data["BTCUSDT"], rules.apply(daily["BTCUSDT"], actual, selected))
        nulls = []
        for h, v in zip(channel.SHIFTS, shifted):
            v["eligible"] = eligible
            nulls.append({"hours":h, **periods(data["BTCUSDT"], rules.apply(daily["BTCUSDT"],v,selected))})
        passed = (actual_stats["training"]["stress"]["return"] > np.quantile([v["training"]["stress"]["return"] for v in nulls],.75)
                  and actual_stats["validation"]["stress"]["return"] > np.median([v["validation"]["stress"]["return"] for v in nulls]))
        base.write(OUTPUT / "placebo.json", {"passed": bool(passed), "same_causal_coverage": True,
                   "actual": actual_stats, "shifts": nulls, "scope": "Hele regelen, ikke isolert merverdi over OI."})
        check["placebo_passed"] = bool(passed)
    else:
        check["reason"] = "Ingen av åtte bestod alle utviklingskrav. OI-kandidaten beholdes. Ingen2025–26-kjøring."
    base.write(OUTPUT / "conclusion.json", check)
    print("VALGT", selected, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lock", action="store_true")
    p.add_argument("--develop", action="store_true")
    args = p.parse_args()
    if args.lock: print(lock()["locked_at_utc"])
    if args.develop: develop()
