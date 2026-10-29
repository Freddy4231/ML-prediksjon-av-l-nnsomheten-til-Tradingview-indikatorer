"""Fanen «Automatisk handel · WEEX» i visningskopien: skriptenes signaler på WEEX-lys, uten handel.

**Denne kopien kan ikke handle.** Den inneholder ingen ordremotor, ingen signering og leser ingen
WEEX-nøkler. Den henter bare offentlige markedsdata (lys og prissteg) med GET mot en fast liste med
markedsendepunkter. Grafene viser hvor skriptet ville vært long eller short ved lysets lukking.
Start, Stopp og Lukk i skript-feltene driver papirhandel (``papirhandel.py``): simulerte posisjoner og fyll
i minnet på denne maskinen. Kontoens ekte posisjoner, fyll og ordrelogg finnes bare i serverversjonen.

Modulen heter det samme som i hovedprosjektet, fordi skriptmodulene (Clean Wave, SuperTrend, TMG)
henter lysene sine med ``from weex_autotrade_dashboard import market_bars``. Her peker det til de
offentlige lysene under. Ikke finansiell rådgivning.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import requests

import bigscalp_weex
import clean_wave_weex
import papirhandel
import kinetic_scalper_weex
import kivanc_supertrend
import market_oracle_weex
import pro_swing_weex
import tmg_trades_weex

# Offentlige markedsdata fra WEEX futures. Ingen konto-, ordre- eller signerte endepunkter finnes i kopien.
MARKET_DOMAIN = "https://api-contract.weex.com"
PUBLIC_PATHS = frozenset({"/capi/v3/market/klines", "/capi/v3/market/historyKlines",
                          "/capi/v3/market/exchangeInfo"})
SYMBOL = "BTCUSDT"
ALL_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "LTCUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT",
               "SUIUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "NEARUSDT", "TRXUSDT", "TONUSDT")
COINS = {symbol.removesuffix("USDT"): symbol for symbol in ALL_SYMBOLS}
# Tidsintervall i minutter (som TradingViews timeframe.period) → WEEX-intervall og millisekunder.
INTERVALS = {"5": ("5m", 300_000), "15": ("15m", 900_000), "30": ("30m", 1_800_000), "60": ("1h", 3_600_000),
             "240": ("4h", 14_400_000), "1440": ("1d", 86_400_000)}
TIMEFRAMES = {"5 minutter": "5", "15 minutter": "15", "30 minutter": "30", "1 time": "60", "4 timer": "240",
              "1 dag": "1440"}
HISTORY_EXTRA_BARS = 1000
HISTORY_PAGE = 100

CHANDELIER_LABEL = "Chandelier Exit (22, 3) [everget] — Mest lønnsomme skript"
PRO_SWING_LABEL = "Pro Swing (rekonstruksjon · SL/TP)"
KINETIC_LABEL = "Kinetic Scalper (rekonstruksjon · SL/TP)"
ORACLE_LABEL = market_oracle_weex.LABEL
BIGSCALP_LABEL = bigscalp_weex.LABEL
TMG_LABEL = tmg_trades_weex.LABEL
SLOT_STRATEGIES = {
    "Clean Wave Matrix PRO (Dark_Ace_Master)": "clean_wave",
    "SuperTrend STRATEGY (KivancOzbilgic)": "kivanc_st",
    CHANDELIER_LABEL: "chandelier",
    TMG_LABEL: "tmg_trades",
    PRO_SWING_LABEL: "pro_swing",
    KINETIC_LABEL: "kinetic",
    ORACLE_LABEL: "oracle",
    BIGSCALP_LABEL: "bigscalp",
}
DIRECTIONS = {"Long og short (som UT Bot)": "both", "Kun long (SELG lukker)": "long_only"}
# Skript-feltene viser Clean Wave på BUY/SELL-merkene: long når et BUY-merke kommer, short ved SELL-merke.
SLOT_CW_MODE = "badges"
KINDS = {"open": "Inngang", "close": "Utgang", "stop": "Nødstopp"}

SAFETY_TEXT = (
    "**🔒 Papirhandel · ingen ekte handel.** Denne versjonen kan ikke sende ordre: den har ingen ordremotor, ingen "
    "signering og leser ingen WEEX-nøkler. Start, Stopp og Lukk simulerer handelen på offentlige WEEX-priser, i "
    "minnet på denne maskinen. Ikke finansiell rådgivning."
)
PRO_SWING_TEXT = (
    "Uavhengig rekonstruksjon av offentlige utdata fra Pro Swing (ikke den beskyttede originalkoden). "
    "Public reference bruker ATR200 × 8,4, hovednivå på 20 % av kanalbredden, SL på en halv kanalbredde og "
    "TP på 1,5R. Grafen viser kanalen, bekreftede kjøp/salg og referanse-SL/TP. Lønnsomhet er ikke dokumentert."
)
KINETIC_TEXT = (
    "Uavhengig, delvis estimert rekonstruksjon av Kinetic Scalper, ikke den lukkede originalkoden. Den vedlagte "
    "Pine-kilden kjøres med standardinnstillingene på avsluttede WEEX-lys. Grafen viser bekreftede kjøps- og "
    "salgssignaler med referanseinngang, SL, TP1 og TP2. Lønnsomhet er ikke dokumentert."
)
ORACLE_TEXT = (
    "Selvstendig, estimert rekonstruksjon av Market Oracle Pro [ChartPrime], ikke originalkode. Fast Trend-modus, "
    "Tuning 14 og ATR-hysterese. Egen risikoregel: SL 2 × ATR14, TP1 2 × ATR14 og TP2 3 × ATR14. Profilen avviker "
    "fra originalens signaler. Lønnsomhet er ikke dokumentert."
)
BIGSCALP_TEXT = (
    "Delvis estimert rekonstruksjon av Bigscalp Premium fra offentlig beskrivelse: L/S-inngangsmerker, ett ADD og "
    "X-utgang med 5-minutters trendfilter. TP og SL er av, som i originalens standardprofil. Ikke originalkode "
    "og ingen dokumentasjon av lønnsomhet."
)
TMG_TEXT = (
    "Uavhengig kvantitativ rekonstruksjon inspirert av YouTube-kanalen «TMG Trades» (@tmgtrades): Stonksy-ankring "
    "med 50/200 EMA, rullerende VRVP med Value Area, VWAP, CVD-absorpsjon og likvidasjonsfeller. Bare avsluttede "
    "lys brukes. Ikke den lukkede Stonksy-koden."
)

_CACHE = {}
_LOCK = threading.Lock()


class MarketDataError(RuntimeError):
    """Offentlige markedsdata kunne ikke hentes eller leses."""


def _public_get(path, params=None):
    """Bare GET mot de offentlige markedsendepunktene. Ingen nøkler, signatur eller kontodata sendes."""
    if path not in PUBLIC_PATHS:
        raise MarketDataError("Bare offentlige markedsdata kan hentes i denne kopien.")
    try:
        response = requests.get(MARKET_DOMAIN + path, params=params or {}, timeout=(3, 10), allow_redirects=False,
                                headers={"User-Agent": "dat158-visning/1.0", "Accept": "application/json"})
        result = response.json()
    except Exception:
        raise MarketDataError("Ingen kontakt med WEEX sine offentlige markedsdata.") from None
    if response.status_code != 200 or (isinstance(result, dict) and str(result.get("code", "0")) not in ("0", "200")):
        raise MarketDataError(f"WEEX avviste forespørselen (HTTP {response.status_code}).")
    return result


def price_text(value):
    """Norsk prisformat: 81 575,5 (seks gjeldende siffer)."""
    return f"{value:,.6g}".replace(",", " ").replace(".", ",")


def parse_klines(rows, interval_ms, now_ms):
    """WEEX-lys til DataFrame med bare AVSLUTTEDE barer (open_time + intervall ≤ nå)."""
    if not isinstance(rows, list) or not rows:
        raise MarketDataError("WEEX leverte ingen lys.")
    frame = pd.DataFrame([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
                          for r in rows], columns=["open_time", "open", "high", "low", "close", "volume"])
    frame = frame.drop_duplicates("open_time").sort_values("open_time")
    frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
    frame = frame[frame.open_time + interval_ms <= now_ms]
    if frame.empty:
        raise MarketDataError("Ingen avsluttede lys ennå.")
    if (frame.open_time % interval_ms).any():
        raise MarketDataError("Lysene er ikke justert til intervallet.")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("open_time"), unit="ms", utc=True), name="open_time")
    return frame


def market_bars(environment=None, timeframe="5", symbol=SYMBOL, limit=1000):
    """Offentlige WEEX-lys (ingen nøkler), mellomlagret i 30 sekunder.

    ``environment`` tas imot for å passe skriptmodulenes kall, men brukes ikke: kopien leser alltid de offentlige
    produksjonslysene."""
    key = ("bars", str(timeframe), symbol, int(limit))
    with _LOCK:
        cached = _CACHE.get(key)
    if cached and time.time() - cached[0] < 30:
        return cached[1]
    interval, step = INTERVALS[str(timeframe)]
    rows = _public_get("/capi/v3/market/klines", {"symbol": symbol, "interval": interval, "limit": int(limit)})
    bars = parse_klines(rows, step, int(time.time() * 1000))
    with _LOCK:
        _CACHE[key] = (time.time(), bars)
    return bars


def price_step(symbol):
    """WEEX sitt minste prissteg for mynten (Pine-skriptenes syminfo.mintick). Bufret i én time."""
    with _LOCK:
        cached = _CACHE.get("exchange_info")
    if not cached or time.time() - cached[0] >= 3600:
        cached = (time.time(), _public_get("/capi/v3/market/exchangeInfo"))
        with _LOCK:
            _CACHE["exchange_info"] = cached
    chosen = [s for s in (cached[1] or {}).get("symbols") or [] if isinstance(s, dict) and s.get("symbol") == symbol]
    if len(chosen) != 1:
        raise MarketDataError(f"Fant ikke prissteget for {symbol}.")
    return 10.0 ** -int(chosen[0]["pricePrecision"])


def with_history(symbol, timeframe, bars, extra=HISTORY_EXTRA_BARS):
    """Pro Swing: inntil ``extra`` eldre, offentlige lys foran ``bars`` (100 per kall, bufret per mynt og intervall).

    Pro Swing låser kanalbredden fra ATR200; med bare de 1000 siste lysene avhenger signalene av hvor vinduet
    starter. Feiler hentingen, brukes det som finnes."""
    interval, step = INTERVALS[str(timeframe)]
    key = ("history", symbol, str(timeframe))
    first = int(bars.index[0].timestamp() * 1000)
    wanted = first - int(extra) * step
    with _LOCK:
        stored = _CACHE.get(key)
    older = stored[stored.index < bars.index[0]] if stored is not None else bars.iloc[:0]
    if len(older) and int(older.index[-1].timestamp() * 1000) + step != first:
        older = bars.iloc[:0]
    pages = [older]
    oldest = int(older.index[0].timestamp() * 1000) if len(older) else first
    try:
        while oldest > wanted:
            end = oldest - step
            start = max(wanted, end - (HISTORY_PAGE - 1) * step)
            rows = _public_get("/capi/v3/market/historyKlines", {"symbol": symbol, "interval": interval,
                                                                 "startTime": start, "endTime": end,
                                                                 "limit": HISTORY_PAGE})
            page = parse_klines(rows, step, int(time.time() * 1000))
            page = page[page.index < pd.Timestamp(oldest, unit="ms", tz="UTC")]
            if page.empty:
                break
            pages.insert(0, page)
            oldest = int(page.index[0].timestamp() * 1000)
    except (MarketDataError, ValueError, TypeError, KeyError, IndexError):
        pass
    merged = pd.concat([*pages, bars])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged = merged[merged.index >= pd.Timestamp(wanted, unit="ms", tz="UTC")]
    with _LOCK:
        _CACHE[key] = merged
    return merged


def _fill_traces(figure, fills):
    """Papirfyllene som egne merker i grafen (åpnet/lukket long og short, nødstopp)."""
    import plotly.graph_objects as go
    frame = pd.DataFrame(fills)
    if frame.empty:
        return figure
    frame["time_utc"] = pd.to_datetime(frame.time, unit="ms", utc=True)
    styles = {("BUY", "LONG"): ("Åpnet long", "triangle-up", "#0b6b45"),
              ("SELL", "SHORT"): ("Åpnet short", "triangle-down", "#a3222d"),
              ("SELL", "LONG"): ("Lukket long", "x", "#0b6b45"),
              ("BUY", "SHORT"): ("Lukket short", "x", "#a3222d")}
    for (side, position), (name, symbol, color) in styles.items():
        part = frame[(frame.side == side) & (frame.position_side == position)]
        if part.empty:
            continue
        figure.add_trace(go.Scatter(
            x=part.time_utc, y=part.price, mode="markers", name="Papirhandel: " + name,
            marker={"symbol": symbol, "size": 15, "color": color, "line": {"width": 2, "color": "white"}},
            customdata=np.column_stack([part[["qty", "commission", "realized_pnl", "order_id"]].to_numpy(dtype=object),
                                        part.kind.map(KINDS).fillna("Annet").to_numpy(dtype=object)]),
            hovertemplate=("%{x}<br>Papirfyll %{y:.6g}<br>%{customdata[0]:.6g} stk · gebyr %{customdata[1]:.4f} "
                           "USDT<br>Realisert %{customdata[2]:.2f} USDT<br>%{customdata[4]} · %{customdata[3]}"
                           "<extra>" + name + "</extra>")))
    return figure


def _paper_labels(figure):
    """Skriptmodulene kaller fyll for «WEEX-fyll»; i kopien er de simulerte papirfyll."""
    for trace in figure.data if figure is not None else ():
        name = str(trace.name or "")
        if "WEEX" in name and ("fyll" in name or "utførte" in name):
            trace.name = "Papirhandel (simulerte fyll)"
    return figure


def check_public_data():
    """«Kontroller WEEX-tilkobling»: bare offentlige markedsdata, ingen konto eller nøkler."""
    try:
        bars = market_bars(None, "5", SYMBOL)
        step = price_step(SYMBOL)
    except MarketDataError as exc:
        return f"**WEEX offentlige markedsdata:** ikke tilgjengelig akkurat nå ({exc})."
    last = bars.index[-1] + pd.Timedelta(minutes=5)
    return (f"**WEEX offentlige markedsdata:** OK. Siste avsluttede BTC-lys {last:%d.%m.%Y %H:%M} UTC, sluttkurs "
            f"{price_text(float(bars.close.iloc[-1]))} USDT, prissteg {price_text(step)}. Kopien bruker ingen konto "
            "og kan ikke sende ordre.")


# ---------------------------------------------------------------- papirhandel i skript-feltene

def _slot_settings(strategy_label, coin_labels, timeframe_label, direction_label):
    key = SLOT_STRATEGIES.get(strategy_label)
    symbols = [COINS[c] for c in (coin_labels or []) if c in COINS]
    timeframe = TIMEFRAMES.get(timeframe_label, "5")
    return key, symbols, timeframe, int(timeframe), DIRECTIONS.get(direction_label, "both")


def start_slot(slot, strategy_label, timeframe_label, coin_labels, direction_label, margin_pct, leverage, accepted,
               bigscalp_tp=0.0, bigscalp_sl=0.0):
    key, symbols, timeframe, minutes, direction = _slot_settings(strategy_label, coin_labels, timeframe_label,
                                                                  direction_label)
    if key is None:
        return "Velg et skript."
    return papirhandel.start(slot, strategy_label, key, symbols, timeframe, minutes, direction, margin_pct, leverage,
                             accepted, bigscalp_tp, bigscalp_sl)


def stop_slot(slot):
    return papirhandel.stop(slot)


def close_slot(slot):
    return papirhandel.close(slot)


def start_slots(accepted, *settings):
    """«Start alle valgte skript-felt samtidig»: ``settings`` er (skript, intervall, mynter, retning, margin,
    gearing, Bigscalp TP, Bigscalp SL) for hvert felt."""
    messages = []
    for number in range(len(settings) // 8):
        values = settings[number * 8:(number + 1) * 8]
        if values[2]:
            messages.append(start_slot(f"#{number + 1}", values[0], values[1], values[2], values[3], values[4],
                                       values[5], accepted, values[6], values[7]))
    return "\n\n".join(messages) or "Velg mynter i minst ett skript-felt."


def stop_all(slots=("#1", "#2", "#3")):
    for slot in slots:
        papirhandel.stop(slot)
    return "⏹ Alle skript er stoppet. Papirposisjonene beholdes."


def close_all(slots=("#1", "#2", "#3")):
    for slot in slots:
        papirhandel.close(slot)
    return "🛑 Alle papirposisjoner er lukket, og alle skript er stoppet."


def autotrade_view():
    """Status, posisjoner, papirfyll og beslutninger etter at nye avsluttede lys er behandlet."""
    papirhandel.refresh()
    return (papirhandel.status_text(), papirhandel.states_table(), papirhandel.fills_table(),
            papirhandel.decisions_table())


def _bigscalp_profile(symbol):
    """Samme valgfrie TP/SL som et felt som papirhandler Bigscalp på mynten, ellers originalens standard."""
    with papirhandel._LOCK:
        run = next((r for r in papirhandel._STATE["runs"].values()
                    if r.running and r.key == "bigscalp" and symbol in r.tracks), None)
    tp, sl = (run.bigscalp_tp, run.bigscalp_sl) if run else (0.0, 0.0)
    direction = "Long Only" if run and run.direction == "long_only" else "Both"
    return bigscalp_weex.Profile(direction=direction, enable_tp=tp > 0, tp_pct=tp or 1.0, enable_sl=sl > 0,
                                 sl_pct=sl or 0.5)


# ---------------------------------------------------------------- Kinetic Scalper og Market Oracle

KINETIC_TABLE_COLUMNS = ["Signal ved lukking (UTC)", "Signal", "Referanseinngang", "SL", "TP1 (50 %)", "TP2 (resten)"]


def _kinetic_series(frame, key):
    """Les kun Pine-resultatet; ingen separat signalberegning i visningen."""
    aliases = {"buy": ("buy", "Long Signal"), "sell": ("sell", "Short Signal"),
               "entry": ("entry", "Signal Entry"), "sl": ("sl", "Signal SL"),
               "tp1": ("tp1", "Signal TP1"), "tp2": ("tp2", "Signal TP2")}
    for column in aliases.get(key, (key,)):
        if column in frame:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def _kinetic_figure(frame, symbol, minutes=5, product_name="Kinetic Scalper"):
    """Pine-nivåene ved lysets lukking: referanseinngang, SL, TP1, TP2 og bekreftede signaler."""
    import plotly.graph_objects as go
    shown = frame.tail(180)
    closes = shown.index + pd.Timedelta(minutes=minutes)
    fig = go.Figure(go.Candlestick(x=shown.index, open=shown.open, high=shown.high,
                                   low=shown.low, close=shown.close, name=f"WEEX {symbol}"))
    active = (shown["plan_active"].fillna(False).astype(bool) if "plan_active" in shown
              else pd.Series(True, index=shown.index))
    for key, plot_title, label, color in (
        ("entry", "Entry", "Referanseinngang", "#ad8200"),
        ("sl", "Stop Loss", "Skriptets SL (hele posisjonen)", "#bb3545"),
        ("tp1", "TP1", "TP1 (50 %)", "#058c91"),
        ("tp2", "TP2", "TP2 (resten)", "#4164be"),
    ):
        values = shown[plot_title] if plot_title in shown else _kinetic_series(shown, key).where(active)
        fig.add_trace(go.Scatter(x=closes, y=values, mode="lines", name=label, connectgaps=False,
                                 line={"color": color, "width": 1.3}))
    for key, name, marker, color in (("buy", "KJØP / LONG", "triangle-up", "#17845b"),
                                      ("sell", "SELG / SHORT", "triangle-down", "#cf4651")):
        mask = _kinetic_series(shown, key).fillna(0).astype(bool)
        fig.add_trace(go.Scatter(x=closes[mask], y=shown.close[mask], mode="markers", name=name,
                                 marker={"symbol": marker, "size": 11, "color": color},
                                 hovertemplate="%{x}<br>Bekreftet ved close %{y:.6g}<extra>" + name + "</extra>"))
    fig.update_layout(template="plotly_white", height=430, dragmode="pan", hovermode="closest",
                      title=f"{product_name} · {symbol} · {minutes} min",
                      xaxis={"rangeslider": {"visible": False}, "title": "UTC · signaler ved lysets lukking"},
                      yaxis={"title": "Pris (USDT)"}, legend={"orientation": "h", "y": -0.2},
                      margin={"l": 45, "r": 15, "t": 45, "b": 85})
    return fig


def _kinetic_signals_table(frame, minutes, limit=20):
    buy = _kinetic_series(frame, "buy").fillna(0).astype(bool)
    sell = _kinetic_series(frame, "sell").fillna(0).astype(bool)
    selected = frame.loc[buy | sell].tail(limit)
    levels = {key: _kinetic_series(selected, key) for key in ("entry", "sl", "tp1", "tp2")}
    rows = [[(stamp + pd.Timedelta(minutes=minutes)).strftime("%d.%m.%Y %H:%M"),
             "LONG" if buy.loc[stamp] else "SHORT", *[levels[key].loc[stamp] for key in levels]]
            for stamp in reversed(selected.index)]
    return pd.DataFrame(rows, columns=KINETIC_TABLE_COLUMNS)


def _kinetic_text(view, symbol, minutes, name="Kinetic Scalper"):
    active = bool(view.get("plan_active")) and bool(view.get("valid"))
    side = {1: "🟢 **LONG-plan**", -1: "🔴 **SHORT-plan**"}.get(view.get("target"), "venter på signal")
    status = side if active else "⚪ **ingen aktiv referanseplan**"
    frame = view["frame"]
    text = (f"**{symbol} ({minutes} min) · {name} (rekonstruksjon)** · {status} · "
            f"Pris: `{float(frame.close.iloc[-1]):,.6g} USDT`. Signalmerkene står ved lysets lukking.")
    if active:
        text += "\n\n**Pine-referanse:** " + " · ".join(
            f"{label} {price_text(float(view[key]))}" for key, label in
            (("entry", "inngang"), ("sl", "SL"), ("tp1", "TP1 (50 %)"), ("tp2", "TP2 (resten)"))) + "."
    return text


# ---------------------------------------------------------------- grafen i skript-feltene

def coin_chart_view(coin_label="BTC", strategy_label="Clean Wave Matrix PRO (Dark_Ace_Master)",
                    timeframe_label="5 minutter", source_label="WEEX futures"):
    """Sanntidsgraf, kort status og siste signaler for valgt skript og mynt, med eventuelle papirfyll.

    Bare offentlige markedsdata; ingen konto. Papirfyllene kommer fra papirhandelen i skript-feltene."""
    coin = (coin_label or "BTC").strip().upper().removesuffix("USDT")
    symbol = COINS.get(coin, f"{coin}USDT")
    key = SLOT_STRATEGIES.get(strategy_label, "clean_wave")
    tf_code, minutes = kivanc_supertrend.TIMEFRAMES.get(timeframe_label, ("5", 5))
    src = kivanc_supertrend.DATA_SOURCES.get(source_label, "weex")
    fills = papirhandel.fills_for(symbol)
    timing = (" · Signalmerkene står ved lysets lukking; store merker og stjerner = papirfyll (simulert)."
              if fills else " · Signalmerkene står ved lysets lukking. Ingen ordre sendes fra denne kopien.")
    try:
        if key == "bigscalp":
            bars = market_bars(None, tf_code, symbol)
            lower = market_bars(None, "5", symbol) if minutes > 5 else None
            view = bigscalp_weex.evaluate(bars, minutes, filter_bars=lower, profile=_bigscalp_profile(symbol))
            text = bigscalp_weex.dashboard(view, symbol, minutes) + timing + "\n\n" + BIGSCALP_TEXT
            return text, _paper_labels(bigscalp_weex.figure(view["frame"], symbol, fills, minutes)), \
                bigscalp_weex.signals_table(view["frame"], minutes)
        if key in ("kinetic", "oracle"):
            provider = market_oracle_weex if key == "oracle" else kinetic_scalper_weex
            name = "Market Oracle Pro · estimert" if key == "oracle" else "Kinetic Scalper"
            bars = market_bars(None, tf_code, symbol)
            view = provider.evaluate(bars, minutes=minutes, symbol=symbol, mintick=price_step(symbol))
            text = (_kinetic_text(view, symbol, minutes, name=name) + timing + "\n\n"
                    + (ORACLE_TEXT if key == "oracle" else KINETIC_TEXT))
            figure = _kinetic_figure(view["frame"], symbol, minutes, product_name=name)
            shown_from = int(view["frame"].tail(180).index[0].timestamp() * 1000)
            _fill_traces(figure, [f for f in fills if f["time"] >= shown_from])
            return text, figure, _kinetic_signals_table(view["frame"], minutes)
        if key == "pro_swing":
            bars = with_history(symbol, tf_code, market_bars(None, tf_code, symbol))
            frame = pro_swing_weex.compute(bars, minutes=minutes)
            text = pro_swing_weex.dashboard(frame, symbol, minutes=minutes) + timing + "\n\n" + PRO_SWING_TEXT
            figure = pro_swing_weex.figure(frame, symbol=symbol, fills=fills, visible_bars=180, minutes=minutes)
            return text, _paper_labels(figure), pro_swing_weex.recent_signals_table(frame, limit=20, minutes=minutes)
        if key == "tmg_trades":
            bars = tmg_trades_weex.load_bars(symbol, source=src, timeframe=tf_code, minutes=minutes)
            frame = tmg_trades_weex.compute(bars, minutes=minutes, mode="combined")
            s = tmg_trades_weex.summarize(frame, symbol=symbol, timeframe_label=timeframe_label)
            badge = {1: "🟢 **LONG (+1)**", -1: "🔴 **SHORT (-1)**"}.get(s.position, "⚪ **FLAT (0)**")
            background = {1: "🟢 Grønn", -1: "🔴 Rød"}.get(s.bg_regime, "⚪ Hvit")
            poc = f"`{s.poc:,.4f} USDT`" if np.isfinite(s.poc) else "—"
            vwap = f"`{s.vwap:,.4f} USDT`" if np.isfinite(s.vwap) else "—"
            text = (f"**{symbol} ({timeframe_label}, {s.htf_scale}× HTF)** · **{tmg_trades_weex.LABEL} v2** · "
                    f"Posisjon: {badge} · Stonksy BG: **{background}** ({s.osc_color}) · "
                    f"Pris: `{s.close:,.4f} USDT` · VWAP: {vwap} · VRVP POC: {poc} · "
                    f"Buy Pressure: `{s.buy_pressure_pct:.1f} %` (CVD {'stigende' if s.cvd_rising else 'avtagende'})"
                    f"{timing}\n\n{TMG_TEXT}")
            figure = tmg_trades_weex.figure(frame, symbol=symbol, fills=fills, visible_bars=180, minutes=minutes)
            return text, _paper_labels(figure), tmg_trades_weex.recent_signals_table(frame, limit=20)
        if key == "clean_wave":
            bars = clean_wave_weex.load_bars(symbol, source=src, timeframe=tf_code, minutes=minutes)
            frame = clean_wave_weex.compute(bars, minutes=minutes, mode=SLOT_CW_MODE)
            s = clean_wave_weex.summarize(frame, symbol=symbol, timeframe_label=timeframe_label,
                                          source_label=source_label)
            badge = "🟢 **LONG (+1)**" if s.position == 1 else "🔴 **SHORT (-1)**"
            text = (f"**{symbol} ({timeframe_label})** · **Clean Wave Matrix PRO [Dark_Ace_Master]** · "
                    f"handler på BUY/SELL-merkene · Aktiv posisjon: {badge} · Pris: `{s.close:,.4f} USDT` · "
                    f"Trendbølge (ALMA 21): `{s.wave:,.4f} USDT`{timing}")
            figure = clean_wave_weex.figure(frame, symbol=symbol, fills=fills, visible_bars=180, minutes=minutes)
            return text, _paper_labels(figure), clean_wave_weex.recent_signals_table(frame, limit=20, minutes=minutes)
        bars = kivanc_supertrend.load_bars(symbol, source=src, minutes=minutes)
        if key == "chandelier":
            frame = kivanc_supertrend.compute_chandelier(bars, minutes=minutes)
            name, level = "Chandelier Exit (22, 3) [everget]", "Chandelier-stopp"
        else:
            frame = kivanc_supertrend.compute(bars, minutes=minutes)
            name, level = "SuperTrend STRATEGY [KivancOzbilgic]", "SuperTrend-linje"
        s = kivanc_supertrend.summarize(frame, symbol=symbol, timeframe_label=timeframe_label,
                                        source_label=source_label)
        badge = "🟢 **LONG (+1)**" if s.position == 1 else "🔴 **SHORT (-1)**"
        text = (f"**{symbol} ({timeframe_label})** · **{name}** · Aktiv posisjon: {badge} · "
                f"Pris: `{s.close:,.4f} USDT` · {level}: `{s.supertrend_level:,.4f} USDT`{timing}")
        figure = kivanc_supertrend.figure(frame, symbol=symbol, fills=fills, visible_bars=180, minutes=minutes,
                                          title_suffix=" · Chandelier Exit (22, 3)" if key == "chandelier" else "")
        if key == "chandelier" and figure.layout.annotations:
            figure.layout.annotations[0].text = f"{symbol} · Chandelier Exit (22, 3) [everget] ({minutes}m)"
        return text, _paper_labels(figure), kivanc_supertrend.recent_signals_table(frame, limit=20, minutes=minutes)
    except Exception:
        return (f"**{symbol} ({timeframe_label}):** lysene kunne ikke hentes akkurat nå. Prøv «Oppdater» om litt.",
                None, pd.DataFrame())
