"""Skrivebeskyttet TMG-sammenligning. Leser fryste resultater; trener eller handler aldri."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
DIRECTORY = ROOT / "reports/tmg_improvement_2026-10-26"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT")
DEPLOY_FILES = ["tmg_research_dashboard.py", "reports/tmg_improvement_2026-10-26/dashboard_data.json",
                "reports/tmg_improvement_2026-10-26/dashboard_manifest.json",
                "reports/tmg_improvement_2026-10-26/README.md"]


def load():
    manifest = json.loads((DIRECTORY / "dashboard_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("TMG-koden eller rapporten er endret siden evalueringen; resultatet er skjult.")
    return json.loads((DIRECTORY / "dashboard_data.json").read_text())


def view(symbol="BTCUSDT"):
    try:
        data = load()
        if symbol not in SYMBOLS:
            raise ValueError("Ukjent mynt")
        lines = ["### TMG · test av 120 forbedringsforsøk",
                 "**Ingen ny variant godkjent. TMG V2 er beholdt.**",
                 "Valg og godkjenning på eldre data. 2025–2026 er kjent historisk kontroll, ikke en urørt test.",
                 "Grafen viser simulert kapital etter **0,15 % per side + faktisk Binance-funding**, "
                 "med **1× notional** fra 1. januar 2025 til 11. oktober 2026 (UTC). "
                 "Stress dobler både handelskostnad og funding. Kolonnen «2026 netto» starter en separat simulering flat "
                 "1. januar 2026. Avkastningen er ikke målt på WEEX eller i TradingView. "
                 "WEEX-nødstopp og gearing/likvidasjon er ikke inkludert.",
                 "48 parameteroppsett, 48 varianter med gevinstsikring og 24 ML-filtre er vurdert. "
                 "Den langsomme kandidaten nedenfor ble valgt før kontrollen og forkastet etterpå."]
        fig = go.Figure()
        rows = []
        colors = ("#00DCB9", "#5f9bff", "#aaaaaa", "#EB5564")
        for i, profile in enumerate(data["profiles"]):
            entry = profile["markets"][symbol]
            normal = entry["known_2025_26"]["normal"]
            stress = entry["known_2025_26"]["stress"]
            curve = entry["curve"]
            fig.add_trace(go.Scatter(x=curve["time"], y=[10000 * x for x in curve["equity"]],
                                     name=profile["label"], mode="lines",
                                     line={"color": colors[i], "dash": "dash" if profile["rejected"] else "solid"}))
            rows.append({"Oppsett": profile["label"], "Netto": f"{normal['return']:+.2%}",
                         "Stress": f"{stress['return']:+.2%}", "Største fall": f"{normal['max_drawdown']:.2%}",
                         "Handler": normal["trades"], "2026 netto": f"{entry['known_2026']['normal']['return']:+.2%}",
                         "Status": "Forkastet kandidat" if profile["rejected"] else "Eksisterende referanse"})
        fig.update_layout(template="plotly_dark", title=f"TMG · {symbol} · 10 000 i simulert startkapital",
                          height=430, margin={"l": 45, "r": 20, "t": 50, "b": 40},
                          yaxis_title="Simulert egenkapital", xaxis_title="UTC", legend={"orientation": "h"})
        return "\n\n".join(lines), fig, pd.DataFrame(rows)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"TMG-testresultatene kan ikke vises: {exc}", go.Figure(), pd.DataFrame()
