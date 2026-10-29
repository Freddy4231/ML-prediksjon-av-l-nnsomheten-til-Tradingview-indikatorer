"""Fryst videoforskning og historisk signalkontekst; ingen nettverk eller ordre."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tmg_multi_context as rules

ROOT = Path(__file__).resolve().parent
DIRECTORY = ROOT / "reports/tmg_multi_2026-10-26"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
WINDOWS = ("Siste 90 dager i 2024", "Hele 2024")
DEPLOY_FILES = ["tmg_multi_context.py", "tmg_multi_dashboard.py", "research/tmg_multi_search.py", "research/tmg_improvement.py"] + [
    f"reports/tmg_multi_2026-10-26/{name}" for name in
    ("dashboard_data.json", "dashboard_manifest.json", "README.md", "SOURCE_NOTES.md", "locked_plan.json",
     "development.json", "frozen_choice.json", "known_control.json")]
GATES = {"train_positive": "BTC-trening taper under stress", "train_trades": "Færre enn 15 BTC-treningshandler",
         "train_years": "Færre enn to positive BTC-år", "train_dd": "For stort BTC-fall", "train_median": "Treningsmedian taper under stress",
         "valid_btc": "BTC 2024 taper under stress", "valid_median": "Markedsmedian 2024 taper under stress",
         "train_improved": "Slår ikke BTC-referansen i trening", "valid_improved": "Slår ikke BTC-referansen i 2024"}


def load():
    manifest = json.loads((DIRECTORY / "dashboard_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Videotesten er skjult fordi kildekoden eller rapporten er endret")
    data = json.loads((DIRECTORY / "dashboard_data.json").read_text())
    if data["selected"] is not None or data["released"]:
        raise ValueError("Visningen krever den fryste testen uten godkjent erstatning")
    return data


def view(symbol="BTCUSDT", window=WINDOWS[0]):
    try:
        if symbol not in SYMBOLS or window not in WINDOWS:
            raise ValueError("Ukjent marked eller tidsvindu")
        data = load()
        market = data["markets"][symbol]
        rows = []
        ref = data["references"][symbol]
        entries = [("Dagens dagsfilter · referanse", ref, None)] + [
            (rules.PROFILES[r["id"]], r["datasets"][symbol], r["gates"]) for r in data["development"]]
        for label, stats, gates in entries:
            rows.append({"Testet regel": label,
                         "2021–23 netto": f"{stats['training']['normal']['return']:+.2%}",
                         "Treningshandler": stats["training"]["normal"]["trades"],
                         "2024 netto": f"{stats['validation']['normal']['return']:+.2%}",
                         "2024 stress": f"{stats['validation']['stress']['return']:+.2%}",
                         "2024 største fall": f"{stats['validation']['normal']['max_drawdown']:.2%}",
                         "Hvorfor ikke godkjent · alle markeder": "Referanse" if gates is None else "; ".join(GATES[k] for k,v in gates.items() if not v)})
        points = pd.DataFrame(market["points"])
        points["time"] = pd.to_datetime(points.time, utc=True)
        cut = points.time.max() - pd.Timedelta(days=90) if window == WINDOWS[0] else points.time.min()
        points = points.loc[points.time >= cut]
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.065,
                            row_heights=[.4,.23,.22,.15],
                            subplot_titles=("Pris og uendrede dagsfilter-signaler", "Ekte spot-taker · kjøpsandel 4/24 t",
                                            "Prisbasert CLV-proxy · separat fra ekte volum", "Kjent funding · snitt av tre hendelser"))
        fig.add_trace(go.Scatter(x=points.time,y=points.close,name="Futures close",line={"color":"#aab8cb"}),row=1,col=1)
        for side, label, color, marker in ((1,"KJØP · eksisterende regel","#3ad5a0","triangle-up"),(-1,"SELG/SHORT · eksisterende regel","#ff667c","triangle-down")):
            events = points.loc[points.signal.eq(side)]
            fig.add_trace(go.Scatter(x=events.time,y=events.close,mode="markers",name=label,
                                    marker={"symbol":marker,"color":color,"size":11}),row=1,col=1)
        for n,color in ((4,"#51b2ff"),(24,"#ffb45f")):
            fig.add_trace(go.Scatter(x=points.time,y=100*points[f"spot_buy{n}"],name=f"Ekte kjøpsandel {n} t",
                                    line={"color":color,"width":1.3},connectgaps=False),row=2,col=1)
            fig.add_trace(go.Scatter(x=points.time,y=points[f"proxy{n}"],name=f"CLV-proxy {n} t",
                                    line={"color":color,"width":1.3},connectgaps=False),row=3,col=1)
        fig.add_trace(go.Scatter(x=points.time,y=100*points.known_funding3,name="Publisert funding-snitt · % per hendelse",
                                line={"color":"#b894ff","shape":"hv"},connectgaps=False),row=4,col=1)
        fig.add_hline(y=50,line_dash="dot",line_color="#657589",row=2,col=1)
        fig.add_hline(y=0,line_dash="dot",line_color="#657589",row=3,col=1)
        fig.add_hline(y=0,line_dash="dot",line_color="#657589",row=4,col=1)
        fig.update_yaxes(title_text="Pris · USDT",row=1,col=1)
        fig.update_yaxes(title_text="Kjøp · %",row=2,col=1)
        fig.update_yaxes(title_text="CLV −1 til +1",row=3,col=1)
        fig.update_yaxes(title_text="%",row=4,col=1)
        fig.update_layout(template="plotly_dark",height=900,title=f"TMG signalkontekst · {symbol} · avsluttede 1-timeslys i 2024",
                          legend={"orientation":"h","y":-.08},margin={"l":65,"r":25,"t":90,"b":130})
        signals = points.loc[points.signal.ne(0)].tail(40).copy()
        table = pd.DataFrame({"Tid ved close · UTC":signals.time.dt.strftime("%Y-%m-%d %H:%M"),
                              "Eksisterende signal":signals.signal.map({1:"KJØP",-1:"SELG/SHORT"}),
                              "Ekte kjøpsandel 4 t":signals.spot_buy4.map(lambda x:f"{x:.1%}" if pd.notna(x) else "Mangler"),
                              "Ekte kjøpsandel 24 t":signals.spot_buy24.map(lambda x:f"{x:.1%}" if pd.notna(x) else "Mangler"),
                              "CLV 4 t":signals.proxy4.round(3), "CLV 24 t":signals.proxy24.round(3),
                              "Kjent funding · %":(100*signals.known_funding3).round(4)})
        text = ("### Flere videofunn · spotpress og funding\n\n"
                "[TMG ved 14:17–14:52](https://www.youtube.com/watch?v=1O0XCCfjvkU&t=857) sammenholder spotpress på flere tidsvinduer. "
                "[Videoen ved 02:16 og 09:21](https://www.youtube.com/watch?v=wkgWeJafDV8&t=136) drøfter funding og posisjoner. "
                "**Åtte faste hypoteser ble testet; ingen bestod alle utviklingskravene.** "
                "Ekte spotenighet ga BTC +77,81 % i 2024 (stress +63,80 %), mot referansens +46,30 % (+29,99 %), "
                "men hadde bare 14 BTC-handler i 2021–23, under det låste minimumet 15. "
                "Pine-egnet CLV-enighet ga BTC −20,31 % i 2024. Ingen ny 2025–26-rangering ble kjørt.\n\n"
                "**Grafen er historisk diagnostikk fra 2024, ikke nye handelssignaler.** KJØP/SELG er dagens uendrede dagsfilter. "
                "Ekte kjøpsandel betyr taker-kjøpt volum / samlet spotvolum på Binance, ikke nye longposisjoner. "
                "CLV bruker plasseringen av close i lysintervallet og er en annen størrelse. "
                "Funding er et snitt av tre publiserte hendelser, minst én time gammelt; positiv rate betales normalt av long. "
                "Dette er verken global spot/OI, kommende funding eller identiske Pine-data.\n\n"
                "Tabellen gjelder hele trenings-/2024-perioden uansett grafvindu. 1×, neste ekte åpning, 0,15 % per side + faktisk funding; "
                "stress dobler begge. Historikken er brukt tidligere og er ikke urørt. Ingen dokumentert ny lønnsomhetsforbedring; "
                "standard, TradingView-regler og WEEX-motor beholdes.")
        return text, pd.DataFrame(rows), fig, table
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"Videotesten kan ikke vises: {exc}", pd.DataFrame(), go.Figure(), pd.DataFrame()
