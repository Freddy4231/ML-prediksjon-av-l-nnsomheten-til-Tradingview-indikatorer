"""Separat signalindikator og fryst kostnadstest; ingen ordre eller private kontoer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

import tmg_daily_confirmation as daily
import tmg_trades_weex as tmg

ROOT = Path(__file__).resolve().parent
DIRECTORY = ROOT / "reports/tmg_daily_2026-10-26"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
WINDOWS = ("Siste 90 dager", "Hele kontrollperioden")
DEPLOY_FILES = ["tmg_daily_confirmation.py", "tmg_daily_dashboard.py", "tradingview/tmg_daily_confirmation.pine"] + [
    f"reports/tmg_daily_2026-10-26/{name}" for name in
    ("dashboard_data.json", "dashboard_manifest.json", "README.md", "SOURCE_NOTES.md")]


def load():
    manifest = json.loads((DIRECTORY / "dashboard_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Indikatoren er skjult fordi kode, Pine-fil eller rapport er endret.")
    return json.loads((DIRECTORY / "dashboard_data.json").read_text())


def signal_chart(points, events, symbol, window=WINDOWS[0]):
    frame = pd.DataFrame(points)
    clock = pd.to_datetime(frame.time, utc=True)
    cutoff = clock.max() - pd.Timedelta(days=90) if window == WINDOWS[0] else clock.min()
    frame = frame.loc[clock >= cutoff]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=frame.time, open=frame.open, high=frame.high,
                                low=frame.low, close=frame.close, name="Binance perpetual · 1 time"))
    for key, name, color in (("daily_sma50", "Bekreftet SMA50 · dager", "#00dcb9"),
                              ("daily_sma200", "Bekreftet SMA200 · dager", "#ad8bff")):
        fig.add_trace(go.Scatter(x=frame.time, y=frame[key], name=name, mode="lines", line={"color": color}))
    style = {"KJØP": ("triangle-up", "#00dcb9"), "SELG / SHORT": ("triangle-down", "#ff6577"),
             "LUKK": ("x", "#ffd166"), "AVVIST KJØP": ("x", "#838b9c"), "AVVIST SALG": ("x", "#838b9c")}
    events = [e for e in events if pd.Timestamp(e["time"]) >= cutoff]
    for name, (marker, color) in style.items():
        subset = [e for e in events if e["signal"] == name]
        fig.add_trace(go.Scatter(x=[e["time"] for e in subset], y=[e["price"] for e in subset],
                                 text=[e["reason"] for e in subset], name=name, mode="markers",
                                 marker={"symbol": marker, "color": color, "size": 11},
                                 hovertemplate="%{x}<br>%{y:,.4f}<br>%{text}<extra>%{fullData.name}</extra>"))
    fig.update_layout(template="plotly_dark", height=580, title=f"TMG Dagsbekreftelse · {symbol}",
                      xaxis_rangeslider_visible=False, legend={"orientation": "h"},
                      margin={"l": 50, "r": 20, "t": 65, "b": 40})
    table = pd.DataFrame([{"Tid (UTC)": e["time"], "Signal": e["signal"], "Pris ved signal": e["price"],
                           "Begrunnelse": e["reason"]} for e in reversed(events[-40:])],
                         columns=["Tid (UTC)", "Signal", "Pris ved signal", "Begrunnelse"])
    return fig, table


def describe_events(frame, context, flow, start=None, end=None):
    """Markøren tidfestes ved beslutningsclose; simulatorens fyll er neste åpning."""
    clock = context.index + pd.Timedelta(hours=1)
    rows = []
    for key, label in (("exit_signal", "LUKK"), ("buy_signal", "KJØP"), ("sell_signal", "SELG / SHORT"),
                        ("rejected_buy", "AVVIST KJØP"), ("rejected_sell", "AVVIST SALG")):
        for i in np.flatnonzero(frame[key].to_numpy(bool)):
            if start and clock[i] < pd.Timestamp(start, tz="UTC"):
                continue
            if end and clock[i] > pd.Timestamp(end, tz="UTC"):
                continue
            if key == "exit_signal":
                reason = "Grunnmodellens mål ble flatt eller vendte; dette er ingen børsordre."
            else:
                c, ma = context.daily_close.iloc[i], context.daily_sma50.iloc[i]
                direction = "over" if "buy" in key else "under"
                reason = f"Gårsdagens close {c:,.4f}, SMA50 {ma:,.4f}; krever close {direction} SMA50."
                if flow.weekend.iloc[i]:
                    share, prior = flow.spot_share24.iloc[i], flow.spot_share720_prior.iloc[i]
                    reason += f" Helg: spot/perp {share:.3%}, tidligere nivå {prior:.3%}."
                else:
                    reason += " Ukedag: helgekravet brukes ikke."
                if key.startswith("rejected"):
                    reason += " Ett eller flere inngangskrav er ikke oppfylt."
            rows.append({"time": clock[i].isoformat(), "signal": label, "price": float(frame.close.iloc[i]), "reason": reason})
    return sorted(rows, key=lambda e: e["time"])


def points_for(bars, context, first=0):
    result = []
    for i in range(first, len(bars)):
        row = {"time": (bars.index[i] + pd.Timedelta(hours=1)).isoformat()}
        row.update({k: float(bars[k].iloc[i]) for k in ("open", "high", "low", "close")})
        row.update({k: float(context[k].iloc[i]) if np.isfinite(context[k].iloc[i]) else None
                    for k in ("daily_sma50", "daily_sma200")})
        result.append(row)
    return result


def historical(symbol="BTCUSDT", window=WINDOWS[0]):
    try:
        data = load()
        if symbol not in SYMBOLS or window not in WINDOWS:
            raise ValueError("Ukjent marked eller tidsvindu")
        market = data["markets"][symbol]
        fig, signals = signal_chart(market["points"], market["events"], symbol, window)
        rows = []
        capital = go.Figure()
        for key, label, color in (("v2", "TMG V2 · long og short", "#5f9bff"),
                                   ("weekend", "Bare helgefilter · samme retning", "#ad8bff"),
                                   ("daily", "Ny indikator · dagsbekreftelse", "#00dcb9")):
            r = market["results"][key]
            rows.append({"Oppsett": label, "Netto": f"{r['normal']['return']:+.2%}",
                         "Kostnadsstress": f"{r['stress']['return']:+.2%}",
                         "Største fall": f"{r['normal']['max_drawdown']:.2%}", "Handler": r["normal"]["trades"]})
            curve = market["curves"][key]
            capital.add_trace(go.Scatter(x=curve["time"], y=[10000 * x for x in curve["equity"]], name=label,
                                         mode="lines", line={"color": color}))
        capital.update_layout(template="plotly_dark", title="Historisk simulert kapital · samme marked og kostnader",
                              height=360, legend={"orientation": "h"}, margin={"l": 50, "r": 20, "t": 55, "b": 40})
        text = (
            "### Ny indikator: TMG Dagsbekreftelse\n\n"
            "**KJØP** krever et nytt TMG V2-kjøpssignal og gårsdagens close over **50-dagers SMA**. "
            "**SELG / SHORT** krever det motsatte. I helgen må spotdeltakelsen siste døgn også minst matche "
            "de foregående 30 dagene. **LUKK** følger grunnmodellens utgang/vending, selv om motsatt inngang avvises. "
            "Grå kryss er avviste signaler; de blir ikke innganger senere uten et nytt signal.\n\n"
            "Inspirert av [TMGs dagsnivåer ved 01:00 og 06:11](https://www.youtube.com/watch?v=_GqbaF4lvrI&t=60) "
            "og hans volumgjennomgang. Dette er vår fryste regel, ikke kanalens proprietære kode. "
            "200-dagerssnittet vises som kontekst, men inngår **ikke** i den valgte regelens inngangskrav.\n\n"
            "**Forskningsindikator, ikke godkjent til automatisk handel:** 16 nye oppsett ble valgt på 2021–23 "
            "og godkjent i 2024 før kontrollen. Valget bestod også placebokontrollen. BTC tapte fortsatt i kjent "
            "2025–26-historikk, så originalens ordremotor er beholdt.\n\n"
            "**Historiske signaler · 1 time · januar 2025–september 2026.** Markører er beslutninger ved close, "
            "ikke faktiske handler. Kapital: 1× notional, neste ekte åpning, 0,15 % per side + faktisk funding; "
            "stress dobler begge. Ingen urørt test, native TradingView-/WEEX-fyll, gearing eller full børs-SL/TP. "
            "Grunnmodellens volumtrykk er CLV-estimert; det er ikke målt CVD, OI eller likvidasjoner. "
            "Knappen nedenfor kan hente nye offentlige Binance-data; visningen legger ingen ordre.")
        return text, fig, signals, pd.DataFrame(rows), capital
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"Dagsindikatoren kan ikke vises: {exc}", go.Figure(), pd.DataFrame(), pd.DataFrame(), go.Figure()


def parse_klines(rows, minutes, now):
    """Kun komplette, sammenhengende, fysiske lys; ingen reservekilde eller utfylling."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("Kilden returnerte ingen lys")
    if any(not isinstance(r, (list, tuple)) or len(r) < 11 for r in rows):
        raise ValueError("Ugyldig lysformat")
    step = minutes * 60_000
    selected = [r for r in rows if int(r[0]) + step <= int(now.timestamp() * 1000)]
    if not selected:
        raise ValueError("Ingen avsluttede lys")
    times = np.array([int(r[0]) for r in selected], dtype=np.int64)
    if np.any(times % step) or np.any(np.diff(times) != step):
        raise ValueError("Datakilden har hull eller doble lys")
    if any(int(r[6]) != int(r[0]) + step - 1 for r in selected):
        raise ValueError("Uventet lukketid")
    values = np.array([[float(r[k]) for k in (1, 2, 3, 4, 5, 9)] for r in selected])
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4:] < 0).any():
        raise ValueError("Ugyldige priser eller volum")
    o, h, l, c, v, buy = values.T
    if (h < np.maximum(o, c)).any() or (l > np.minimum(o, c)).any() or (h < l).any() or (buy - v > 8 * np.spacing(v)).any():
        raise ValueError("Ugyldige prisgrenser eller kjøpsvolum")
    frame = pd.DataFrame(values, columns=["open", "high", "low", "close", "volume", "taker_buy_volume"],
                         index=pd.to_datetime(times, unit="ms", utc=True).as_unit("ms"))
    if now - (frame.index[-1] + pd.Timedelta(minutes=minutes)) >= pd.Timedelta(minutes=minutes):
        raise ValueError("Markedsdata er for gamle; ingen nytt signal vises")
    return frame


def current(symbol="BTCUSDT", window=WINDOWS[0]):
    """Tre offentlige forespørsler ved eksplisitt knappetrykk, ingen nøkler eller ordre."""
    try:
        data = load()
        if symbol not in SYMBOLS or window not in WINDOWS:
            raise ValueError("Ukjent marked eller tidsvindu")
        # Panelet og det fryste Pine-skriptet har samme konkrete regler.
        if data["selected"] != "daily50_weekend_both":
            raise ValueError("Denne signalvisningen krever ny pakking for et annet fryst valg")
        now = pd.Timestamp.now(tz="UTC")
        with requests.Session() as session:
            frames = []
            for url, interval, limit, minutes in (
                ("https://fapi.binance.com/fapi/v1/klines", "1h", 1000, 60),
                ("https://fapi.binance.com/fapi/v1/klines", "1d", 300, 1440),
                ("https://api.binance.com/api/v3/klines", "1h", 1000, 60),
            ):
                response = session.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=12)
                response.raise_for_status()
                frames.append(parse_klines(response.json(), minutes, now))
        hours, days, spot = frames
        if len(hours) < 900 or len(days) < 200 or not spot.index.equals(hours.index):
            raise ValueError("For kort eller usammenfallende spot-/futureshistorikk")
        context = daily.align_daily(hours.index, pd.Series(days.close.to_numpy(), index=days.index + pd.Timedelta(days=1)))
        flow = pd.DataFrame(index=hours.index)
        flow["spot_share24"] = spot.volume.rolling(24).sum() / hours.volume.rolling(24).sum()
        flow["spot_share720_prior"] = spot.volume.shift(24).rolling(720).sum() / hours.volume.shift(24).rolling(720).sum()
        flow["weekend"] = (hours.index + pd.Timedelta(hours=1)).dayofweek >= 5
        base = tmg.compute(hours[["open", "high", "low", "close", "volume"]], minutes=60)
        l, s = daily.allowed(context, "daily50_weekend", flow)
        filtered = daily.apply_filter(base, l, s)
        # De siste 199 timene har minst 800 timer oppvarming i denne endelige historikken.
        points = points_for(hours, context, 800)
        events = describe_events(filtered, context, flow, start=hours.index[800].strftime("%Y-%m-%d %H:%M:%S"))
        fig, signals = signal_chart(points, events, symbol, WINDOWS[1])
        last = filtered.iloc[-1]
        action = "KJØP" if last.buy_signal else "SELG / SHORT" if last.sell_signal else "LUKK" if last.exit_signal else "Ingen nytt signal"
        target = {1: "long", -1: "short", 0: "flat"}[int(last.position)]
        text = (f"### TMG Dagsbekreftelse · siste markedsdata\n\n**{symbol} · {action}**. "
                f"Siste avsluttede time: **{(hours.index[-1] + pd.Timedelta(hours=1)).isoformat()}**. "
                f"Modellens beregnede tilstand: **{target}**; dette er ingen faktisk kontoposisjon. "
                f"Gårsdagens close: `{context.daily_close.iloc[-1]:,.4f}`, SMA50: `{context.daily_sma50.iloc[-1]:,.4f}`.\n\n"
                "Kun offentlige Binance spot-/perpetualdata. De siste 1 000 timene brukes til V2; de siste 300 døgnene "
                "til dagsnivåer. En beregning med avgrenset historikk kan avvike fra full historikk. "
                "Grafen viser de siste omtrent åtte døgnene etter 800 timers oppvarming. "
                "Forskningsindikator: BTC-kontrollen var negativ. Ingen ordre eller WEEX-kobling er aktivert her.")
        return text, fig, signals
    except (OSError, ValueError, KeyError, TypeError, requests.RequestException) as exc:
        return f"Ingen oppdatert indikator: {exc}. Tidligere/ufullstendige data brukes ikke som nye signaler.", go.Figure(), pd.DataFrame()
