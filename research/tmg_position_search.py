"""Tolv låste posisjons-/bekreftelsesideer med separat kilde- og dataproveniens."""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import tmg_position_context as rules
from research import tmg_oi_data as oi_data
from research import tmg_daily_search as prior
from research import tmg_channel_search as channel
from research import tmg_improvement as base

ROOT=base.ROOT
OUTPUT=ROOT/"reports/tmg_oi_2026-10-26"
SYMBOLS,TRAIN,VALID,CONTROL=prior.SYMBOLS,prior.TRAIN,prior.VALID,prior.CONTROL


def lock():
    data=json.loads((OUTPUT/"data_manifest.json").read_text())
    paths=["tmg_position_context.py","research/tmg_position_search.py","research/tmg_oi_data.py",
           "reports/tmg_oi_2026-10-26/SOURCE_NOTES.md","reports/tmg_oi_2026-10-26/data_manifest.json"]
    paths+=list(json.loads((prior.OUTPUT/"locked_plan.json").read_text())["sha256"])
    paths+=[d["path"]for d in data["symbols"].values()]
    plan={"version":"tmg_position_close_next_open_v1","profiles":list(rules.PROFILES),
          "training":TRAIN,"validation":VALID,"known_control":CONTROL,
          "reference":"daily50_weekend_both, 1 time, begge retninger, uendret.",
          "execution":"1× fast mengde, signalclose/neste ekte open, 0,15%/side+funding, stress dobler begge.",
          "features":"OI prosent4/24h, krever sammenhengende gyldig OI. Original arkivtid +65min og maks2h framfyll. Pris4/24h; to lukkede UTC4h-blokker kjent ved timebaråpning; ekte/CLV spot168h adskilt.",
          "entry":"Ferske signaler; kun pending-profiler kan vente inntil6timer, kansellering ved grunnmålets endring/utløp/nytt signal/hullsperre. Ferskt signal i hullsperren oppretter aldri plan. Ingen sen inngang etterpå.",
          "gates":"BTC trening stress>0,min15handler,min2positiveår,DD>-55%,medianstress>0; BTC/median2024stress>0; BTCstress bedre enn dagsfilter i begge perioder.",
          "rank":"Median trenings-stress-Sharpe på BTC/ETH/SOL blant kvalifiserte. Ingen kontrollrangering.",
          "placebo":"Åtte bakoverskift720–3240h; valgtBTCtreningstress over75persentil,2024overmedian.",
          "release":"Utvikling+placebo+BTCkontroll normal/stress positiv og forbedret,medianstresspositiv,min2positive,BTCDDikkeverre. OI-profiler feiler dokumentert-tilgjengelighetskrav uansett avkastning.",
          "no_choice":"Ingen kvalifisert: ingen ny2025–26kjøring, behold signaler.",
          "limitations":["Retrospektiv, kjent historie brukt mange ganger, ingen urørt test.",
                         "OI-arkivets historiske tilgjengelighet/versjoner ikke dokumentert; +65min er antakelse, ikke fasit.",
                         "OI er samlet åpen kontraktsinteresse, ikke åpning av én bestemt side. En børs, ikke globalt.",
                         "Ingen native TradingView-/WEEX-paritet, børs-SL/TP eller gearingtest."],
          "sha256":{p:base.sha(ROOT/p)for p in sorted(set(paths))}}
    path=OUTPUT/"locked_plan.json"
    if path.exists():
        assert json.loads(path.read_text())==json.loads(json.dumps(plan)),"Planen kan ikke endres"
    else:base.write(path,plan)
    return plan


def contexts():
    data,flow,original,context=prior.contexts()
    frames,values={},{}
    manifest=json.loads((OUTPUT/"data_manifest.json").read_text())
    for s in SYMBOLS:
        path=ROOT/manifest["symbols"][s]["path"]
        assert base.sha(path)==manifest["symbols"][s]["sha256"]
        snapshots=pd.read_csv(path)
        snapshots.index=pd.DatetimeIndex(pd.to_datetime(snapshots.pop("observed_time"),utc=True,format="mixed"))
        bars=data[s]["bars"]
        aligned=oi_data.align(bars.index,snapshots)
        spot=channel.read_bars(base.DATA/f"{s}_raw_1h.csv").reindex(bars.index)
        frames[s]=prior.filtered(original[s],context[s],flow[s],{"profile":"daily50_weekend","direction":"both"})
        values[s]=rules.features(bars,spot,flow[s],aligned,data[s]["blocked"])
    return data,frames,values


def develop():
    lock()
    data,frames,values=contexts()
    refs={s:{"training":base.evaluate(data[s],frames[s],TRAIN,"both"),"validation":base.evaluate(data[s],frames[s],VALID,"both")}for s in SYMBOLS}
    rows=[]
    for profile in rules.PROFILES:
        results={}
        for s in SYMBOLS:
            frame=rules.apply(frames[s],values[s],profile)
            results[s]={"training":base.evaluate(data[s],frame,TRAIN,"both"),"validation":base.evaluate(data[s],frame,VALID,"both")}
        b,ref=results["BTCUSDT"],refs["BTCUSDT"]
        gates={"train_positive":b["training"]["stress"]["return"]>0,
               "train_trades":b["training"]["normal"]["trades"]>=15,
               "train_years":sum(v>0 for v in b["training"]["normal"]["yearly"].values())>=2,
               "train_dd":b["training"]["normal"]["max_drawdown"]>-.55,
               "train_median":np.median([r["training"]["stress"]["return"]for r in results.values()])>0,
               "valid_btc":b["validation"]["stress"]["return"]>0,
               "valid_median":np.median([r["validation"]["stress"]["return"]for r in results.values()])>0,
               "train_improved":b["training"]["stress"]["return"]>ref["training"]["stress"]["return"],
               "valid_improved":b["validation"]["stress"]["return"]>ref["validation"]["stress"]["return"]}
        row={"id":profile,"datasets":results,"gates":{k:bool(v)for k,v in gates.items()},"qualified":bool(all(gates.values())),
             "score":float(np.median([r["training"]["stress"]["sharpe"]for r in results.values()]))}
        rows.append(row)
        print(profile,round(b["training"]["normal"]["return"]*100,2),round(b["validation"]["normal"]["return"]*100,2),row["qualified"],flush=True)
    rows.sort(key=lambda r:(r["qualified"],r["score"]),reverse=True)
    base.write(OUTPUT/"development.json",{"candidates":rows,"references":refs})
    base.write(OUTPUT/"frozen_choice.json",{"selected":rows[0]if rows[0]["qualified"]else None,
               "plan_sha256":base.sha(OUTPUT/"locked_plan.json"),"development_sha256":base.sha(OUTPUT/"development.json")})
    print("VALGT",rows[0]["id"]if rows[0]["qualified"]else None,flush=True)


def control():
    path=OUTPUT/"known_control.json"
    if path.exists():raise ValueError("Kontroll er allerede registrert")
    plan=lock()
    for p,h in plan["sha256"].items():assert base.sha(ROOT/p)==h,p
    choice=json.loads((OUTPUT/"frozen_choice.json").read_text())
    assert choice["plan_sha256"]==base.sha(OUTPUT/"locked_plan.json")
    assert choice["development_sha256"]==base.sha(OUTPUT/"development.json")
    selected=choice["selected"]
    if selected is None:
        base.write(path,{"selected":None,"released":False,"reason":"Ingen bestod utvikling/2024; ingen ny kontroll."})
        return
    data,frames,values=contexts()
    nulls=[]
    for h in channel.SHIFTS:
        shifted=values["BTCUSDT"].shift(h)
        shifted["eligible"]=values["BTCUSDT"]["eligible"]
        frame=rules.apply(frames["BTCUSDT"],shifted,selected["id"])
        nulls.append({"hours":h,"training":base.evaluate(data["BTCUSDT"],frame,TRAIN,"both"),"validation":base.evaluate(data["BTCUSDT"],frame,VALID,"both")})
    b=selected["datasets"]["BTCUSDT"]
    placebo=b["training"]["stress"]["return"]>np.quantile([r["training"]["stress"]["return"]for r in nulls],.75)and b["validation"]["stress"]["return"]>np.median([r["validation"]["stress"]["return"]for r in nulls])
    base.write(OUTPUT/"placebo.json",{"passed":bool(placebo),"shifts":nulls})
    if "oi" in selected["id"]:
        base.write(path,{"selected":selected["id"],"released":False,"placebo_passed":bool(placebo),
                        "reason":"OI-tilgjengelighet og TradingView-feed ikke verifisert. Ingen ny kontroll på kortere/ufullstendig OI-historikk."})
        return
    results={}
    for s in SYMBOLS:
        results[s]={}
        for key,frame in (("daily",frames[s]),("candidate",rules.apply(frames[s],values[s],selected["id"]))):
            results[s][key]=base.evaluate(data[s],frame,CONTROL,"both")
            _,curve,trades=base.simulate(data[s],frame,*CONTROL,"both")
            first=data[s]["bars"].index.searchsorted(pd.Timestamp(CONTROL[0],tz="UTC"))
            pd.DataFrame({"time":data[s]["bars"].index[first:first+len(curve)]+pd.Timedelta(hours=1),"equity":curve}).to_csv(OUTPUT/f"{s}_{key}_equity.csv",index=False)
            pd.DataFrame(trades).to_csv(OUTPUT/f"{s}_{key}_trades.csv",index=False)
        print(s,{k:round(v["normal"]["return"]*100,2)for k,v in results[s].items()},flush=True)
    b,ref=results["BTCUSDT"]["candidate"],results["BTCUSDT"]["daily"]
    gates={"placebo":bool(placebo),"btc_positive":all(b[k]["return"]>0 for k in ("normal","stress")),
           "btc_improved":all(b[k]["return"]>ref[k]["return"]for k in ("normal","stress")),
           "median_positive":np.median([r["candidate"]["stress"]["return"]for r in results.values()])>0,
           "two_positive":sum(r["candidate"]["stress"]["return"]>0 for r in results.values())>=2,
           "drawdown":b["normal"]["max_drawdown"]>=ref["normal"]["max_drawdown"]}
    base.write(path,{"selected":selected["id"],"results":results,"gates":{k:bool(v)for k,v in gates.items()},"released":bool(all(gates.values()))})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--develop",action="store_true")
    p.add_argument("--control",action="store_true")
    args=p.parse_args()
    if args.develop:develop()
    if args.control:control()
