"""Fryst kanalanalyse og signalfilter: ingen trening, innhenting eller ordre ved visning."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
DIRECTORY = ROOT / "reports/tmg_channel_2026-10-26"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
DEPLOY_FILES = ["tmg_channel_dashboard.py", "tradingview/tmg_spot_participation.pine"] + [
    f"reports/tmg_channel_2026-10-26/{name}" for name in
    ("dashboard_data.json", "dashboard_manifest.json", "README.md", "SOURCE_NOTES.md")]


def load():
    manifest = json.loads((DIRECTORY / "dashboard_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Resultatene er skjult fordi TMG-koden eller rapporten er endret.")
    return json.loads((DIRECTORY / "dashboard_data.json").read_text())


def view(symbol="BTCUSDT"):
    try:
        data = load()
        if symbol not in SYMBOLS:
            raise ValueError("Ukjent mynt")
        market = data["markets"][symbol]
        text = (
            "### Nye funn fra tre TMG Trades-videoer\n\n"
            "TMG legger vekt på **ekte kjøps-/salgsvolum, spotdeltakelse mot futures og markedskontekst**. "
            "I [US-åpningsvideoen ved 05:37](https://www.youtube.com/watch?v=6JiCtRBSsGo&t=337) og "
            "[helgevideoen ved 04:00](https://www.youtube.com/watch?v=OCnkVk2KG7o&t=240) sammenligner han "
            "døgnets spotdeltakelse med en lengre referanse. "
            "[OI-videoen ved 04:40](https://www.youtube.com/watch?v=5pBOHaOeu4U&t=280) viser også korte analysevinduer.\n\n"
            "Vi har testet **20 faste oppsett** med reelt taker-volum og ekte futureskurser. "
            "Kandidaten som ble valgt før kontrollen, krever at spot/futures-basevolum siste 24 timer "
            "minst matcher de foregående 30 dagene ved **nye long-innganger i helgen**. "
            "Dette er vår testbare tolkning, ikke en komplett oppgitt strategi fra kanalen.\n\n"
            "**Relativ forbedring, fortsatt ikke godkjent som ny standard:** BTC tapte også med filteret. "
            "Originalens handelsregler er beholdt. Volumpanelet for TradingView nedenfor er en datavisning uten ordre.\n\n"
            "**1 time · kun long · 1. januar 2025–1. oktober 2026 (UTC)**. "
            "Simulering med 1× notional, 0,15 % per side og faktisk Binance-funding. Stress dobler begge. "
            "Priser er nå Binance perpetual, og perioden er kortere enn i rapporten over; sammenlign bare radene her "
            "med hverandre. Ingen native TradingView-/WEEX-test, gearing eller full simulering av børs-SL/TP.\n\n"
            "Røde kryss viser avviste, bekreftede V2-signaler. Grønne trekanter viser signaler filteret ville "
            "tillatt, ikke faktiske handler. OI, ETF og likvidasjonskart er ikke erstattet med prisproxies.")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.10,
                            subplot_titles=("Simulert kapital · identiske kostnader", "Historiske futureskurser og bekreftede inngangssignaler"))
        rows = []
        for key, label, color in (("v2", "TMG V2 · kun long", "#5f9bff"),
                                   ("filtered", "Helgevolumfilter · forskning", "#00dcb9")):
            entry = market[key]
            normal, stress = entry["normal"], entry["stress"]
            fig.add_trace(go.Scatter(x=entry["curve"]["time"], y=[10000 * x for x in entry["curve"]["equity"]],
                                     mode="lines", name=label, line={"color": color}), row=1, col=1)
            rows.append({"Oppsett": label, "Netto": f"{normal['return']:+.2%}", "Stress": f"{stress['return']:+.2%}",
                         "Største fall": f"{normal['max_drawdown']:.2%}", "Handler": normal["trades"]})
        fig.add_trace(go.Scatter(x=market["price"]["time"], y=market["price"]["close"], name="Perpetualkurs",
                                 mode="lines", line={"color": "#aab4c7", "width": 1}), row=2, col=1)
        for key, label, color, marker in (("accepted", "Volumkrav oppfylt", "#00dcb9", "triangle-up"),
                                          ("rejected", "Volumkrav ikke oppfylt", "#eb5564", "x")):
            fig.add_trace(go.Scatter(x=market[key]["time"], y=market[key]["price"], name=label, mode="markers",
                                     marker={"color": color, "symbol": marker, "size": 10}), row=2, col=1)
        fig.update_layout(template="plotly_dark", title=f"TMG kanalanalyse · {symbol} · 1 time", height=640,
                          margin={"l": 50, "r": 20, "t": 65, "b": 40}, legend={"orientation": "h"})
        return text, fig, pd.DataFrame(rows)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"Kanalanalysen kan ikke vises: {exc}", go.Figure(), pd.DataFrame()
