"""Vis alle låste videoforsøk uten datainnhenting, trening eller ordre."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tmg_squeeze_context as rules

ROOT = Path(__file__).resolve().parent
DIRECTORY = ROOT / "reports/tmg_squeeze_2026-10-26"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
CHOICES = [("Gjeldende OI-kandidat · referanse", "oi"), *[(v,k) for k,v in rules.PROFILES.items()]]
GATES = {"train_positive":"negativ BTC treningsstress", "train_trades":"færre enn 15 BTC treningshandler",
         "train_years":"færre enn to positive BTC treningsår", "train_dd":"for stort BTC treningsfall",
         "train_median":"negativ median treningsstress", "valid_btc":"negativ BTC stress i 2024",
         "valid_median":"negativ median stress i 2024", "train_improved":"slår ikke OI på BTC treningsstress",
         "valid_improved":"slår ikke OI på BTC stress i 2024", "train_cross":"svakere median treningsstress",
         "valid_cross":"svakere median stress i 2024", "valid_dd":"større BTC fall i 2024"}
DEPLOY_FILES = ["tmg_squeeze_context.py", "tmg_squeeze_dashboard.py", "research/tmg_squeeze_search.py",
                "research/tmg_squeeze_report.py", "research/tmg_channel_data.py", "research/tmg_channel_rules.py",
                "research/tmg_channel_search.py", "research/tmg_daily_search.py",
                "reports/tmg_channel_2026-10-26/data_manifest.json"] + [f"reports/tmg_squeeze_2026-10-26/{p}" for p in
                ("SOURCE_NOTES.md","README.md","locked_plan.json","development.json","frozen_choice.json",
                 "conclusion.json","dashboard_data.json","dashboard_manifest.json")]


def load():
    manifest = json.loads((DIRECTORY / "dashboard_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("rapporten er skjult fordi kilde eller resultater er endret")
    return json.loads((DIRECTORY / "dashboard_data.json").read_text())


def view(symbol="BTCUSDT", profile="oi"):
    try:
        if symbol not in SYMBOLS or profile not in dict((v,k) for k,v in CHOICES):
            raise ValueError("Ukjent marked eller regel")
        data = load()
        rows = []
        entries = [("Dagsfilter · eldre referanse", data["references"]["daily"][symbol],None),
                   ("OI-bekreftelse · referanse",data["references"]["oi"][symbol],None)] + [
                   (rules.PROFILES[r["id"]],r["datasets"][symbol],r["gates"]) for r in data["development"]]
        for label, stats, gates in entries:
            rows.append({"Regel":label,"2021–23 netto":f"{stats['training']['normal']['return']:+.2%}",
                         "2021–23 stress":f"{stats['training']['stress']['return']:+.2%}",
                         "Treningshandler":stats["training"]["normal"]["trades"],
                         "2024 netto":f"{stats['validation']['normal']['return']:+.2%}",
                         "2024 stress":f"{stats['validation']['stress']['return']:+.2%}",
                         "2024 største fall":f"{stats['validation']['normal']['max_drawdown']:.2%}",
                         "Samlede krav på tvers av markeder":"Referanse" if gates is None else
                         "; ".join(GATES[k] for k,v in gates.items() if not v) or "Bestått"})
        market=data["markets"][symbol]
        points=pd.DataFrame(market["points"])
        times=pd.to_datetime(points.time,utc=True)
        chosen=market["variants"][profile]
        chosen_label=dict((v,k)for k,v in CHOICES)[profile]
        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.12,
                          subplot_titles=("Historiske signaler · pris ved avsluttet time","Netto kapital · start 100, kostnader og funding inkludert"))
        fig.add_trace(go.Scatter(x=times,y=points.close,name="Futures close",line={"color":"#acbbcf"}),row=1,col=1)
        signal=np_series(chosen["signal"])
        for side,label,color,marker in ((1,"KJØP","#3ad5a0","triangle-up"),(-1,"SELG/SHORT","#ff667c","triangle-down")):
            mask=signal.eq(side).to_numpy()
            fig.add_trace(go.Scatter(x=times[mask],y=points.close[mask],mode="markers",name=label,
                                    marker={"symbol":marker,"size":11,"color":color}),row=1,col=1)
        mask=np_series(chosen["exit"]).eq(True).to_numpy()
        fig.add_trace(go.Scatter(x=times[mask],y=points.close[mask],mode="markers",name="LUKK",
                                marker={"symbol":"x","size":8,"color":"#f2c879"}),row=1,col=1)
        for key,label,color in (("daily","Dagsfilter","#929baa"),("oi","OI-referanse","#6ec3ff")):
            fig.add_trace(go.Scatter(x=times,y=100*np_series(market["variants"][key]["equity"]),name=label,
                                    line={"color":color}),row=2,col=1)
        if profile != "oi":
            fig.add_trace(go.Scatter(x=times,y=100*np_series(chosen["equity"]),name="Valgt videoregel",line={"color":"#d7a1ff"}),row=2,col=1)
        fig.update_layout(template="plotly_dark",height=720,title=f"{symbol} · {chosen_label} · historisk 2024",
                          legend={"orientation":"h","y":-.1},margin={"l":65,"r":20,"t":95,"b":110})
        fig.update_yaxes(title_text="USDT",row=1,col=1)
        fig.update_yaxes(title_text="Start 100",row=2,col=1)
        events=points.loc[signal.ne(0).to_numpy()|mask].copy()
        events["signal"]=signal.to_numpy()[signal.ne(0).to_numpy()|mask]
        events["exit"]=np_series(chosen["exit"]).to_numpy()[signal.ne(0).to_numpy()|mask]
        table=pd.DataFrame({"Signal ved close · UTC":pd.to_datetime(events.time,utc=True).dt.strftime("%Y-%m-%d %H:%M"),
                            "Handling":events.apply(lambda r:("LUKK + " if r.exit and r.signal else "")+
                            ({1:"KJØP",-1:"SELG/SHORT"}.get(r.signal,"LUKK")),axis=1),
                            "Signalpris":events.close.round(4),"OI4t · %":(100*events.oi4).round(3),
                            "Spotkjøp4t · %":(100*events.spot_buy4).round(2),
                            "Spotkjøp24t · %":(100*events.spot_buy24).round(2)})
        conclusion=data["conclusion"]["reason"]
        text=("### Videre videotest · spotstøtte og squeeze\n\n"
              f"**{conclusion}** Testene ble låst og kjørt 27. oktober 2026. "
              "Velg en regel for å se dens historiske KJØP, SELG/SHORT og LUKK, samt kapitalutvikling mot begge referansene. "
              "Valget her viser kun forskning og aktiverer ingen strategi.\n\n"
              "[Videoen om timing ved intervalltopper, 08:16–09:18](https://www.youtube.com/watch?v=BesZsOHBLWA&t=496) "
              "inspirerte vending etter OI-fall. [Analysen av spotstøtte, 13:08–13:48](https://www.youtube.com/watch?v=gY46g0pz7To&t=788) "
              "inspirerte bekreftelse av prisbrudd. De åtte reglene er våre testbare tolkninger; videoene gir ikke full kildekode.\n\n"
              "**Viktig resultat:** En høyere avkastning i 2024 alene er ikke nok. Regelen måtte også slå den fryste "
              "OI-referansen i 2021–23 og bestå krav til antall handler, drawdown og flere markeder. "
              "Alle forsøk og alle feilede krav vises i tabellen.\n\n"
              "1 time, begge retninger, 1×, neste ekte åpning, 0,15 % per side + faktisk funding. Stress: 0,30 % og dobbel funding. "
              "Grafens markører er signaler ved close, mens simuleringen fyller neste åpning. Starten av 2024 er flat; eldre signaler blir ikke nye handler. "
              "OI er antatt tilgjengelig etter 65 minutter; historisk publisering og identisk TradingView-feed er fortsatt ubekreftet. "
              "OI-fall identifiserer ikke hvem som ble likvidert. Ekte spotpress er Binance takervolum, ikke CLV. "
              "Historikken er gjenbrukt. Ingen ny 2025–26-kontroll, gearingtest eller verifiserte TradingView-resultater.")
        if table.empty:
            text += "\n\nDen valgte regelen ga ingen kjøps-/salgssignaler i 2024 for dette markedet."
        return text,pd.DataFrame(rows),fig,table
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return f"Videotesten kan ikke vises: {exc}",pd.DataFrame(),go.Figure(),pd.DataFrame()


def np_series(values):
    """Indeksfri tallkolonne for ferdigpakkede tidsserier."""
    return pd.Series(values)
