"""Valgbart gevinstvern og nye kanalfunn, atskilt fra dagens standard og ordremotor."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

import tmg_context_extension as extension
import tmg_daily_dashboard as original
import tmg_daily_confirmation as daily
import tmg_trades_weex as tmg

ROOT = Path(__file__).resolve().parent
DIRECTORY = ROOT / "reports/tmg_context_2026-10-26"
VARIANTS = ("Dagsbekreftelse · dagens standard", "Dagsbekreftelse + gevinstvern · forskning")
DEPLOY_FILES = ["tmg_context_extension.py", "tmg_context_dashboard.py", "tradingview/tmg_daily_profit_guard.pine"] + [
    f"reports/tmg_context_2026-10-26/{name}" for name in
    ("dashboard_data.json", "dashboard_manifest.json", "README.md", "SOURCE_NOTES.md")]


def load():
    manifest = json.loads((DIRECTORY / "dashboard_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Gevinstvernet er skjult fordi koden eller rapporten er endret")
    return json.loads((DIRECTORY / "dashboard_data.json").read_text())


def findings(symbol="BTCUSDT"):
    try:
        data = load()
        if symbol not in original.SYMBOLS:
            raise ValueError("Ukjent marked")
        rows = []
        for row in data["development"]:
            stats = row["datasets"][symbol]
            rows.append({"Testet regel": extension.PROFILES[row["id"]],
                         "2021–23 netto": f"{stats['training']['normal']['return']:+.2%}",
                         "2021–23 stress": f"{stats['training']['stress']['return']:+.2%}",
                         "2024 netto": f"{stats['validation']['normal']['return']:+.2%}",
                         "2024 stress": f"{stats['validation']['stress']['return']:+.2%}",
                         "Utviklingskrav på tvers av markeder": "Bestått" if row["qualified"] else "Ikke bestått"})
        text = ("### Nyeste kanalfunn · 11 faste forsøk\n\n"
                "I [videoen om lukking ved 00:16 og kompresjon ved 03:18](https://www.youtube.com/watch?v=o1Jly6k1BcU&t=16) "
                "skiller TMG en langsiktig markedsoppfatning fra risikoen ved å holde én bestemt handel. "
                "[En annen video ved 08:39](https://www.youtube.com/watch?v=SOJKHYnB0eQ&t=519) drøfter dags-RSI. "
                "Vi testet prisbredde, volum, RSI og tidligere utganger rundt dagens identiske dagsfilter. "
                "Tallene og formlene er våre hypoteser, ikke kanalens oppgitte strategi.\n\n"
                "Bare **gevinstvern ved inngang ±0,3 % etter 2 ATR14 i close-gevinst** bestod utvikling/2024. "
                "På kjent 2025–september 2026 ga det BTC **+31,22 %** mot **−3,91 %**, men ETH falt fra "
                "+161,38 % til +31,98 % og SOL fra +22,07 % til −18,74 %. "
                "**Placebokravet ble ikke bestått:** trenings-stress +135,19 % var under nullvariantenes "
                "75-persentil +139,84 %. Ingen urørt test eller generell forbedring er dokumentert. "
                "Dagens standard beholdes; gevinstvernet kan velges manuelt som forskningsalternativ. "
                "Ingen ordre eller automatisk WEEX-kobling legges til.")
        return text, pd.DataFrame(rows)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"Kanalfunnene kan ikke vises: {exc}", pd.DataFrame()


def chart(points, events, protection, symbol, window):
    fig, table = original.signal_chart(points, events, symbol, window)
    latest = max(pd.Timestamp(p["time"]) for p in points)
    cut = latest - pd.Timedelta(days=90) if window == original.WINDOWS[0] else pd.Timestamp(points[0]["time"])
    line = [p for p in protection if pd.Timestamp(p["time"]) >= cut]
    fig.add_trace(go.Scatter(x=[p["time"] for p in line], y=[p["value"] for p in line], mode="lines",
                             line={"color": "#ffa94d", "width": 2}, connectgaps=False,
                             name="Close-vern · gjelder fra neste bar"))
    fig.update_layout(title=f"TMG Dagsbekreftelse + gevinstvern · {symbol} · forskning")
    return fig, table


def describe(frame, context, flow, start=None, end=None):
    events = original.describe_events(frame, context, flow, start, end)
    extra = {(context.index[i] + pd.Timedelta(hours=1)).isoformat()
             for i in np.flatnonzero(frame.extra_exit.to_numpy(bool))}
    for event in events:
        if event["signal"] == "LUKK" and event["time"] in extra:
            event["reason"] = "Gevinstvern bekreftet ved close. Simulert fyll neste ekte open, ikke på den viste linjen."
    return events


def historical(symbol="BTCUSDT", window=original.WINDOWS[0], variant=VARIANTS[0]):
    if variant == VARIANTS[0]:
        return original.historical(symbol, window)
    try:
        if variant != VARIANTS[1] or symbol not in original.SYMBOLS or window not in original.WINDOWS:
            raise ValueError("Ukjent variant, marked eller tidsvindu")
        data, old = load(), original.load()
        market = data["markets"][symbol]
        fig, signals = chart(old["markets"][symbol]["points"], market["events"], market["protection"], symbol, window)
        rows, capital = [], go.Figure()
        for key, label, color in (("daily", "Dagens dagsfilter · long og short", "#5f9bff"),
                                   ("extension", "Gevinstvern · forskningsalternativ", "#ffa94d")):
            result = market["results"][key]
            rows.append({"Oppsett": label, "Netto": f"{result['normal']['return']:+.2%}",
                         "Kostnadsstress": f"{result['stress']['return']:+.2%}",
                         "Største fall": f"{result['normal']['max_drawdown']:.2%}", "Handler": result["normal"]["trades"]})
            curve = market["curves"][key]
            capital.add_trace(go.Scatter(x=curve["time"], y=[10000 * v for v in curve["equity"]],
                                         name=label, mode="lines", line={"color": color}))
        capital.update_layout(template="plotly_dark", height=360, title="Samme tidsrom, marked, innganger og kostnader",
                              legend={"orientation": "h"}, margin={"l": 50, "r": 20, "t": 55, "b": 40})
        text = ("### TMG Dagsbekreftelse + gevinstvern · forskningsalternativ\n\n"
                "Inngangene følger dagens dagsfilter. Når beste avsluttede close har gått minst **2 ATR14** i riktig "
                "retning fra signalprisen, tegnes et vern ved signalpris **+0,3 % for long / −0,3 % for short**. "
                "Linjen gjelder fra neste bar. **LUKK** bekreftes først hvis close krysser linjen; simulatoren fyller "
                "ved neste ekte open. Ingen sen gjeninntreden på et gammelt signal.\n\n"
                "**Ikke godkjent som ny standard:** BTC ble bedre, ETH ble svakere, SOL tapte og placebokravet feilet. "
                "Dette er ikke dokumentert framtidig lønnsomhet. **Oransje linje er en close-regel, ikke børs-SL "
                "eller garantert breakeven.** Gap, funding og kostnader kan gi tap også etter aktivering.\n\n"
                "**Historisk januar 2025–september 2026 · 1 time · long og short · 1×.** 0,15 % per side + faktisk "
                "funding; stress dobler begge. Kontrollen var kjent, ikke urørt. Ingen native TradingView-/WEEX-fyll "
                "eller test av gearing. Den nye Pine-filen krever samsvarende Binance perpetual/spot og vanlige 1-timeslys. "
                "Ingenting her sender ordre.")
        return text, fig, signals, pd.DataFrame(rows), capital
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"Gevinstvernet kan ikke vises: {exc}", go.Figure(), pd.DataFrame(), pd.DataFrame(), go.Figure()


def current(symbol="BTCUSDT", window=original.WINDOWS[0], variant=VARIANTS[0]):
    if variant == VARIANTS[0]:
        return original.current(symbol, window)
    try:
        load()
        if variant != VARIANTS[1] or symbol not in original.SYMBOLS or window not in original.WINDOWS:
            raise ValueError("Ukjent variant, marked eller tidsvindu")
        now = pd.Timestamp.now(tz="UTC")
        with requests.Session() as session:
            parts = []
            for url, interval, limit, minutes in (
                ("https://fapi.binance.com/fapi/v1/klines", "1h", 1000, 60),
                ("https://fapi.binance.com/fapi/v1/klines", "1d", 300, 1440),
                ("https://api.binance.com/api/v3/klines", "1h", 1000, 60),
            ):
                response = session.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=12)
                response.raise_for_status()
                parts.append(original.parse_klines(response.json(), minutes, now))
        hours, days, spot = parts
        if len(hours) < 900 or len(days) < 200 or not hours.index.equals(spot.index):
            raise ValueError("For kort eller usammenfallende spot-/futureshistorikk")
        context = daily.align_daily(hours.index, pd.Series(days.close.to_numpy(), index=days.index + pd.Timedelta(days=1)))
        flow = pd.DataFrame(index=hours.index)
        flow["spot_share24"] = spot.volume.rolling(24).sum() / hours.volume.rolling(24).sum()
        flow["spot_share720_prior"] = spot.volume.shift(24).rolling(720).sum() / hours.volume.shift(24).rolling(720).sum()
        flow["weekend"] = (hours.index + pd.Timedelta(hours=1)).dayofweek >= 5
        base = tmg.compute(hours[["open", "high", "low", "close", "volume"]], minutes=60)
        long, short = daily.allowed(context, "daily50_weekend", flow)
        filtered = daily.apply_filter(base, long, short)
        frame = extension.apply(filtered, extension.features(hours, filtered, context), "entry_guard")
        start = hours.index[800].strftime("%Y-%m-%d %H:%M:%S")
        events = describe(frame, context, flow, start)
        line = [{"time": (hours.index[i] + pd.Timedelta(hours=1)).isoformat(),
                 "value": float(frame.protection_line.iloc[i]) if np.isfinite(frame.protection_line.iloc[i]) else None}
                for i in range(800, len(frame))]
        fig, signals = chart(original.points_for(hours, context, 800), events, line, symbol, original.WINDOWS[1])
        last = frame.iloc[-1]
        action = "KJØP" if last.buy_signal else "SELG / SHORT" if last.sell_signal else "LUKK" if last.exit_signal else "Ingen nytt signal"
        state = {1: "long", -1: "short", 0: "flat"}[int(last.position)]
        text = (f"### Gevinstvern · siste markedsdata · {symbol}\n\n**{action}** ved siste avsluttede time "
                f"**{(hours.index[-1] + pd.Timedelta(hours=1)).isoformat()}**. Beregnet tilstand: **{state}**; "
                "dette er ikke en faktisk kontoposisjon.\n\n"
                "Oransje linje er et beregnet close-vern fra neste bar, ingen børsordre eller garantert breakeven. "
                "1 000 timer og 300 døgn fra offentlige Binance-data, graf etter 800 timers oppvarming; endelig "
                "historikk kan avvike fra full historikk. **Placebo ikke bestått; SOL tapte i kontrollen.** "
                "Forskningsalternativ, ingen WEEX-kobling eller ordre er aktivert.")
        return text, fig, signals
    except (OSError, ValueError, KeyError, TypeError, requests.RequestException) as exc:
        return f"Ingen oppdatert indikator: {exc}. Ufullstendige/gamle data vises ikke som nye signaler.", go.Figure(), pd.DataFrame()
