"""Pakk fryste forsøk og kurver eksplisitt, aldri ved visning på nettsiden."""
import json
import numpy as np
import pandas as pd
import tmg_squeeze_context as rules
import tmg_squeeze_dashboard as dashboard
from research import tmg_squeeze_search as search
from research import tmg_improvement as base


def build():
    output=search.OUTPUT
    plan=search.lock()
    for p,h in plan["sha256"].items():assert base.sha(base.ROOT/p)==h,p
    choice=json.loads((output/"frozen_choice.json").read_text())
    assert choice["plan_sha256"]==base.sha(output/"locked_plan.json")
    assert choice["development_sha256"]==base.sha(output/"development.json")
    development=json.loads((output/"development.json").read_text())
    conclusion=json.loads((output/"conclusion.json").read_text())
    package={"development":development["candidates"],"references":development["references"],
             "conclusion":conclusion,"selected":choice["selected"],"markets":{}}
    data,daily,oi,values=search.contexts()
    for symbol in search.SYMBOLS:
        market=data[symbol]; bars=market["bars"]
        mask=(bars.index>=pd.Timestamp(search.VALID[0],tz="UTC"))&(bars.index<pd.Timestamp(search.VALID[1],tz="UTC"))
        points=values[symbol].loc[mask,["close","oi4","spot_buy4","spot_buy24"]].copy()
        points["time"]=(points.index+pd.Timedelta(hours=1)).map(lambda t:t.isoformat())
        variants={}
        for name in ("daily","oi",*rules.PROFILES):
            frame=(daily[symbol] if name=="daily" else oi[symbol] if name=="oi" else rules.apply(daily[symbol],values[symbol],name))
            metrics,equity,trades=base.simulate(market,frame,*search.VALID,"both")
            stored=(development["references"][name][symbol] if name in ("daily","oi") else
                    next(r["datasets"][symbol]for r in development["candidates"]if r["id"]==name))["validation"]["normal"]
            assert metrics==stored, (symbol,name,"Kurven må samsvare med fryste tall")
            signal=np.where(frame.loc[mask,"buy_signal"],1,np.where(frame.loc[mask,"sell_signal"],-1,0))
            variants[name]={"signal":signal.tolist(),"exit":frame.loc[mask,"exit_signal"].to_numpy(bool).tolist(),
                            "equity":equity.tolist(),"trades":trades}
        package["markets"][symbol]={"points":json.loads(points.to_json(orient="records")),"variants":variants}
    base.write(output/"dashboard_data.json",package)
    lines=["# Videre videotest · spotstøtte og squeeze", "", "Låst og evaluert 27. oktober 2026. Arbeidsmappen ble opprettet 26. oktober.",
           "",conclusion["reason"],"", "Alle tall er Python-simulering ved 1×, etter kostnader og faktisk funding.",
           "0,15 % per side, stress 0,30 % og dobbel funding. Ingen verifiserte TradingView-/WEEX-fyll.",
           "", "## BTC · alle forhåndslåste kandidater", "", "| Regel | 2021–23 netto | 2021–23 stress | Handler | 2024 netto | 2024 stress | Bestått |",
           "|---|---:|---:|---:|---:|---:|---|"]
    entries=[("Dagsfilter",development["references"]["daily"]["BTCUSDT"],None),
             ("OI-referanse",development["references"]["oi"]["BTCUSDT"],None)]+[
             (rules.PROFILES[r["id"]],r["datasets"]["BTCUSDT"],r["qualified"])for r in development["candidates"]]
    for name,stats,passed in entries:
        t,v=stats["training"],stats["validation"]
        lines.append(f"| {name} | {t['normal']['return']:+.2%} | {t['stress']['return']:+.2%} | {t['normal']['trades']} | {v['normal']['return']:+.2%} | {v['stress']['return']:+.2%} | {'referanse'if passed is None else 'ja'if passed else 'nei'} |")
    lines += ["", "## Begrensninger", "", *["- "+x for x in plan["limitations"]],
              "- Ingen kandidat kvalifisert: ingen ny placebo eller 2025–26-kjøring i dette forsøket." if not choice["selected"] else "- Se separat placeborapport; hele regelen testes, ikke isolert merverdi.",
              "- Tidligere Pine-kilder og aktiv motor beholdes. Nytt panel er historisk sammenligning.",
              "- Alle åtte forsøk og feilede krav finnes i development.json; data-/kodehash i locked_plan.json.",
              "- Videoenes OI-tolkning beviser ikke bestemte market maker-posisjoner eller likvidasjoner.",
              "", "Kilder og eksakte egne regler: [SOURCE_NOTES.md](SOURCE_NOTES.md).",""]
    (output/"README.md").write_text("\n".join(lines))
    paths=[p for p in dashboard.DEPLOY_FILES if not p.endswith("dashboard_manifest.json")]
    # Dataenes hash bevares i planen; store treningsfiler er ikke nødvendige for statisk serving.
    paths += [p for p in plan["sha256"] if not p.startswith("data/")]
    base.write(output/"dashboard_manifest.json",{"sha256":{p:base.sha(base.ROOT/p)for p in sorted(set(paths))},
               "dataset_sha256":{p:h for p,h in plan["sha256"].items()if p.startswith("data/")}})
    print("Pakket tre markeder, begge referanser og alle åtte regler",flush=True)


if __name__=="__main__":build()
