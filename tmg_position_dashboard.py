"""Kildefestet OI-/strukturtest og historisk signalkontekst uten ordre."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tmg_position_context as rules
from tmg_multi_dashboard import GATES

ROOT=Path(__file__).resolve().parent
DIRECTORY=ROOT/"reports/tmg_oi_2026-10-26"
SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT")
WINDOWS=("Siste 90 dager i 2024","Hele 2024")
DEPLOY_FILES=["tmg_position_context.py","tmg_position_dashboard.py","research/tmg_oi_data.py","research/tmg_position_search.py",
              "research/tmg_oi_pine_audit.py","research/export_tmg_oi.py","tradingview/tmg_oi_confirmation.pine","tradingview/TMG_OI_README.md"]+[
    f"reports/tmg_oi_2026-10-26/{name}"for name in
    ("dashboard_data.json","dashboard_manifest.json","README.md","SOURCE_NOTES.md","data_manifest.json",
     "locked_plan.json","development.json","frozen_choice.json","known_control.json","placebo.json","pine_logic_audit.json")]


def load():
    manifest=json.loads((DIRECTORY/"dashboard_manifest.json").read_text())
    for name,expected in manifest["sha256"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError("OI-testen er skjult fordi kildekoden eller rapporten er endret")
    return json.loads((DIRECTORY/"dashboard_data.json").read_text())


def view(symbol="BTCUSDT",window=WINDOWS[0]):
    try:
        if symbol not in SYMBOLS or window not in WINDOWS:
            raise ValueError("Ukjent marked eller historisk vindu")
        data=load()
        rows=[]
        entries=[("Dagens dagsfilter · referanse",data["references"][symbol],None)]+[
            (rules.PROFILES[r["id"]],r["datasets"][symbol],r["gates"])for r in data["development"]]
        for label,stats,gates in entries:
            rows.append({"Oppsett":label,"2021–23 netto":f"{stats['training']['normal']['return']:+.2%}",
                         "Treningshandler":stats["training"]["normal"]["trades"],
                         "2024 netto":f"{stats['validation']['normal']['return']:+.2%}",
                         "2024 stress":f"{stats['validation']['stress']['return']:+.2%}",
                         "2024 største fall":f"{stats['validation']['normal']['max_drawdown']:.2%}",
                         "Utviklingskrav · alle markeder":"Referanse"if gates is None else "; ".join(GATES[k]for k,v in gates.items()if not v)or"Bestått"})
        market=data["markets"][symbol]
        points=pd.DataFrame(market["points"])
        points["time"]=pd.to_datetime(points.time,utc=True)
        cutoff=points.time.max()-pd.Timedelta(days=90)if window==WINDOWS[0]else points.time.min()
        points=points.loc[points.time>=cutoff]
        fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.09,row_heights=[.5,.25,.25],
                          subplot_titles=("Pris og dagens kjøp/selg ved close","Kjent OI · arkivserie med 65 min forsinkelse",
                                          "Endring i kjent OI · prosent"))
        fig.add_trace(go.Scatter(x=points.time,y=points.close,name="Futures close",line={"color":"#acbbcf"}),row=1,col=1)
        for side,label,color,marker in ((1,"KJØP · dagens regel","#3ad5a0","triangle-up"),(-1,"SELG/SHORT · dagens regel","#ff667c","triangle-down")):
            hit=points.loc[points.base_signal.eq(side)]
            fig.add_trace(go.Scatter(x=hit.time,y=hit.close,name=label,mode="markers",marker={"symbol":marker,"color":color,"size":10}),row=1,col=1)
        if data["selected"]:
            for side,label,color,marker in ((1,"Kandidat KJØP · forskning","#a0ffe2","circle-open"),(-1,"Kandidat SELG · forskning","#ffb8c4","circle-open")):
                hit=points.loc[points.candidate_signal.eq(side)]
                fig.add_trace(go.Scatter(x=hit.time,y=hit.close,name=label,mode="markers",marker={"symbol":marker,"color":color,"size":14}),row=1,col=1)
        fig.add_trace(go.Scatter(x=points.time,y=points.oi,name="OI · kildens mengdeserie",line={"color":"#c69aff"},connectgaps=False),row=2,col=1)
        for h,color in ((4,"#51b2ff"),(24,"#ffb45f")):
            fig.add_trace(go.Scatter(x=points.time,y=100*points[f"oi{h}"],name=f"OI-endring {h} t",line={"color":color,"width":1.2},connectgaps=False),row=3,col=1)
        fig.add_hline(y=0,line_dash="dot",line_color="#657589",row=3,col=1)
        fig.update_yaxes(title_text="USDT",row=1,col=1)
        fig.update_yaxes(title_text="Kildemengde",row=2,col=1)
        fig.update_yaxes(title_text="%",row=3,col=1)
        fig.update_layout(template="plotly_dark",height=800,title=f"TMG posisjonskontekst · {symbol} · 2024, ikke sanntid",
                          legend={"orientation":"h","y":-.1},margin={"l":70,"r":25,"t":90,"b":125})
        signals=points.loc[points.base_signal.ne(0)|points.candidate_signal.ne(0)].tail(40)
        events=pd.DataFrame({"Tid ved close · UTC":signals.time.dt.strftime("%Y-%m-%d %H:%M"),
                             "Dagens signal":signals.base_signal.map({0:"—",1:"KJØP",-1:"SELG/SHORT"}),
                             "Kandidatsignal · forskning":signals.candidate_signal.map({0:"—",1:"KJØP",-1:"SELG/SHORT"}),
                             "OI-observasjon · UTC":signals.oi_observed_time,
                             "OI 4 t · %":(100*signals.oi4).round(3),"OI 24 t · %":(100*signals.oi24).round(3)})
        picked=data["selected"]
        conclusion=("Ingen av de tolv reglene bestod alle utviklingskravene. Ingen ny 2025–26-kjøring eller signalendring."if picked is None else
                    f"Valgt i utvikling: **{rules.PROFILES[picked]}**. {data['control'].get('reason','Se separat kontrollrapport.')} "
                    f"Godkjent til erstatning: {'ja' if data['control']['released'] else 'nei'}. Ingen automatisk motoraktivering.")
        selected_stats=next((r["datasets"][symbol]for r in data["development"]if r["id"]==picked),None)
        if selected_stats:
            reference=data["references"][symbol]["validation"]
            current=selected_stats["validation"]
            conclusion+=(f" {symbol} i 2024: netto **{current['normal']['return']:+.2%}** mot **{reference['normal']['return']:+.2%}**; "
                         f"stress **{current['stress']['return']:+.2%}** mot **{reference['stress']['return']:+.2%}**. "
                         f"Placebokrav: {'bestått'if data['control'].get('placebo_passed')else'ikke bestått'}.")
        coverage=market["coverage"]
        text=("### Nye videoer · OI, ukespress og bekreftet inngang\n\n"
              "[Timing-videoen ved 00:29](https://www.youtube.com/watch?v=3iSVeAV3Dfk&t=29) venter på spotpress. "
              "[Analysen ved 05:03 og 10:44](https://www.youtube.com/watch?v=2PUWkW400mk&t=303) bruker sju dagers press og pris sammen med OI. "
              "Vi testet **tolv faste regler**: OI-bekreftelse, avvik/utgang, ukespress og to bekreftede 4-timersblokker. "
              "To regler kan vente høyst seks timer etter et nytt signal; planene utløper eller kanselleres når grunnmålet endres. "
              f"{conclusion}\n\n"
              f"Nytt grunnlag: **{data['archive_count']} SHA-kontrollerte daglige arkiver**, "
              f"{data['observation_count']:,} OI-observasjoner på tre markeder. {symbol} har "
              f"{coverage['2024']:.1%} tilgjengelig OI på 2024-timene med vår forsinkelse. "
              "Manglende/for gamle verdier gir hull. Motstridende tidsstempler er utelatt.\n\n"
              "**Historisk OI-tilgjengelighet er ikke verifisert.** Arkivets observasjonstid pluss 65 minutter er en "
              "konservativ antakelse, ikke dokumentert historisk API-publisering. OI-testene er eksplorative. "
              "[Kildens åpne tids-/versjonsspørsmål](https://github.com/binance/binance-public-data/issues/509) er ikke løst av en SHA-kontroll. "
              "OI er samlet åpen kontraktsinteresse på Binance; det måler ikke alene hvor mange som åpner long eller short. "
              "Ekte taker-volum, CLV og OI holdes adskilt. Identisk TradingView-feed/native fyll er ikke bekreftet.\n\n"
              "Grafen viser 2024, ikke sanntid. Åpne sirkler er kandidatens kjøp/selg, fylte trekanter er dagens regel. "
              "Den separate Pine-prototypen bruker TradingViews egen 5-minutters OI-feed med minst 65 min forsinkelse. "
              "Avvik i feed, manglende lys og historikk kan endre signalene. Lokal ventetilstand er kontrollert, ikke native feed eller hele skriptets kjøring i TradingView. "
              "Prototypen er for native TradingView; appens Pine-test/WEEX-adapter støtter ikke denne forespørselen mot lavere tidsramme. "
              "Tabellen gjelder hele 2021–23 og 2024 uansett grafvindu. "
              "1×, neste ekte åpning, 0,15 % per side + funding; stress dobler begge. Historikken er kjent fra tidligere arbeid. "
              "Ingen gearing, kontoendring eller nye ordre.")
        return text,pd.DataFrame(rows),fig,events
    except (OSError,ValueError,KeyError,TypeError)as exc:
        return f"Posisjonstesten kan ikke vises: {exc}",pd.DataFrame(),go.Figure(),pd.DataFrame()
