"""Papirhandel i visningskopien: skript-feltene følger skriptenes signaler uten konto og uten ordre.

Start, Stopp, Lukk og «Ta profit / lukk alle» virker som i serverversjonen, men alt skjer i minnet på denne
maskinen: **ingen ordre sendes, ingen børskonto brukes og ingen nøkler leses.** Lysene er de samme offentlige
WEEX-lysene som grafene bruker. Tilstanden forsvinner når appen stoppes.

Regler (forenklet etterligning av ordremotoren i serverversjonen):
- Beslutning ved lysets lukking. Papirfyllet skjer til lysets sluttkurs; ordremotoren handler ved neste åpning,
  som normalt er nesten samme pris.
- Clean Wave, SuperTrend, Chandelier og TMG blir med i skriptets nåværende posisjon ved start. Pro Swing,
  Kinetic, Oracle og Bigscalp venter på et nytt signal etter start, som motoren.
- «Kun long»: et SHORT-signal lukker long i stedet for å åpne short.
- Størrelse: margin = valgt prosent av papirsaldoen per mynt; posisjonsverdi = margin × gearing.
- Gebyr 0,08 % av posisjonsverdien per side (WEEX sitt takergebyr). Ingen funding.
- Nødstopp når tapet når 90 % av marginen (som motorens børsnødstopp), sjekket mot lysets høy/lav.
- Skriptenes egne SL/TP følges via målposisjonen ved neste lukking; børsordre, ADD og delvise TP simuleres ikke.

Nye lys behandles hver gang tabellene oppdateres (knapp eller hvert 30. sekund når siden er åpen). Har siden
vært lukket, tas de avsluttede lysene igjen i rekkefølge, så resultatet blir det samme. Ikke finansiell rådgivning.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import bigscalp_weex
import clean_wave_weex
import kinetic_scalper_weex
import kivanc_supertrend
import market_oracle_weex
import pro_swing_weex
import tmg_trades_weex

PAPER_BALANCE = 1000.0          # USDT ved start, felles papirsaldo for alle felt
FEE = 0.0008                    # WEEX takergebyr per side
STOP_SHARE = 0.9                # nødstopp når 90 % av marginen er tapt
JOIN_AT_START = {"clean_wave", "kivanc_st", "chandelier", "tmg_trades"}
MAX_ROWS = 500
FILL_COLUMNS = ["Tid (UTC)", "Mynt", "Handling", "Posisjon", "Årsak", "Antall", "Pris", "Gebyr (USDT)",
                "Realisert (USDT)", "Papir-ID"]
DECISION_COLUMNS = ["Bar slutt (UTC)", "Mynt", "Signal", "Handling", "Status", "Melding"]
STATE_COLUMNS = ["Mynt", "Posisjon", "Neste steg"]
SIDE = {1: "LONG", -1: "SHORT", 0: "FLAT"}


@dataclass
class Track:
    symbol: str
    last_close: int = 0             # siste behandlede lys (lukketid i ms)
    side: int = 0
    qty: float = 0.0
    entry: float = 0.0
    margin: float = 0.0
    waiting: int | None = None      # skriptets mål ved start/nødstopp: vent til det endrer seg
    last_price: float = float("nan")


@dataclass
class Run:
    slot: str
    label: str
    key: str
    timeframe: str
    minutes: int
    direction: str
    margin_pct: float
    leverage: int
    bigscalp_tp: float = 0.0
    bigscalp_sl: float = 0.0
    running: bool = True
    started: int = 0
    tracks: dict = field(default_factory=dict)


_STATE = {"runs": {}, "fills": [], "decisions": [], "seq": 0, "realized": 0.0, "fees": 0.0}
_LOCK = threading.RLock()


def reset():
    """Tøm all papirhandel (brukes av testene)."""
    with _LOCK:
        _STATE.update(runs={}, fills=[], decisions=[], seq=0, realized=0.0, fees=0.0)


def _stamp(ms):
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").strftime("%d.%m.%Y %H:%M")


def _num(value, digits=2):
    value = 0.0 if abs(value) < 0.5 * 10 ** -digits else value      # ingen «−0,00»
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",").replace("-", "−")


def _price(value):
    return f"{value:,.6g}".replace(",", " ").replace(".", ",")


def _coin(symbol):
    return symbol.removesuffix("USDT")


# ---------------------------------------------------------------- skriptenes målposisjon per lys

def targets(key, symbol, timeframe, minutes, bigscalp_tp=0.0, bigscalp_sl=0.0):
    """Avsluttede lys med skriptets mål (+1 long, −1 short, 0 flat) ved lukking. Samme beregning som grafene."""
    import weex_autotrade_dashboard as weex
    if key == "clean_wave":
        frame = clean_wave_weex.compute(clean_wave_weex.load_bars(symbol, source="weex", timeframe=timeframe,
                                                                  minutes=minutes),
                                        minutes=minutes, mode=weex.SLOT_CW_MODE)
        target = frame["position"]
    elif key in ("kivanc_st", "chandelier"):
        bars = kivanc_supertrend.load_bars(symbol, source="weex", minutes=minutes)
        frame = (kivanc_supertrend.compute_chandelier if key == "chandelier" else kivanc_supertrend.compute)(
            bars, minutes=minutes)
        target = frame["position"]
    elif key == "tmg_trades":
        frame = tmg_trades_weex.compute(tmg_trades_weex.load_bars(symbol, source="weex", timeframe=timeframe,
                                                                  minutes=minutes),
                                        minutes=minutes, mode="combined")
        target = frame["position"]
    elif key == "pro_swing":
        frame = pro_swing_weex.compute(weex.with_history(symbol, timeframe, weex.market_bars(None, timeframe, symbol)),
                                       minutes=minutes)
        target = frame["plan_side"].where(frame["plan_active"].astype(bool), 0)
    elif key in ("kinetic", "oracle"):
        provider = market_oracle_weex if key == "oracle" else kinetic_scalper_weex
        frame = provider.evaluate(weex.market_bars(None, timeframe, symbol), minutes=minutes, symbol=symbol,
                                  mintick=weex.price_step(symbol))["frame"]
        target = frame["target"]
    elif key == "bigscalp":
        lower = weex.market_bars(None, "5", symbol) if minutes > 5 else None
        profile = bigscalp_weex.Profile(enable_tp=bigscalp_tp > 0, tp_pct=bigscalp_tp or 1.0,
                                        enable_sl=bigscalp_sl > 0, sl_pct=bigscalp_sl or 0.5)
        frame = bigscalp_weex.evaluate(weex.market_bars(None, timeframe, symbol), minutes,
                                       filter_bars=lower, profile=profile)["frame"]
        target = frame["side"]
    else:
        raise ValueError("Skriptet kan ikke papirhandles.")
    opened = (frame["timestamp"].astype("int64").to_numpy() if "timestamp" in frame
              else pd.DatetimeIndex(frame.index).as_unit("ms").asi8)
    return pd.DataFrame({"close_time": opened + int(minutes) * 60_000,
                         "high": frame["high"].to_numpy(float), "low": frame["low"].to_numpy(float),
                         "close": frame["close"].to_numpy(float),
                         "target": pd.to_numeric(target, errors="coerce").fillna(0).astype(int).to_numpy()})


# ---------------------------------------------------------------- bokføring

def _log(run, track, close_time, signal, action, status, message):
    _STATE["decisions"].append({"time": int(close_time), "symbol": track.symbol, "signal": signal,
                                "action": action, "status": status, "message": f"Skript {run.slot}: {message}"})
    del _STATE["decisions"][:-MAX_ROWS]


def _fill(run, track, close_time, buy, position_side, kind, reason, qty, price, realized):
    _STATE["seq"] += 1
    fee = qty * price * FEE
    _STATE["fees"] += fee
    _STATE["realized"] += realized - fee
    _STATE["fills"].append({"time": int(close_time), "symbol": track.symbol, "side": "BUY" if buy else "SELL",
                            "position_side": position_side, "kind": kind, "reason": reason, "qty": qty,
                            "price": price, "commission": fee, "realized_pnl": realized - fee,
                            "order_id": f"PAPIR-{_STATE['seq']:06d}", "slot": run.slot})
    del _STATE["fills"][:-MAX_ROWS]


def _open(run, track, side, price, close_time, reason):
    track.margin = PAPER_BALANCE * run.margin_pct / 100
    track.qty = track.margin * run.leverage / price
    track.side, track.entry = side, price
    _fill(run, track, close_time, side > 0, SIDE[side], "open", reason, track.qty, price, 0.0)
    _log(run, track, close_time, f"{SIDE[side].lower()}_signal", f"open_{SIDE[side].lower()}", "papirfyll",
         f"{SIDE[side].lower()} {track.qty:.6g} {_coin(track.symbol)} til {_price(price)} ({reason}).")


def _close(run, track, price, close_time, reason, kind="close"):
    realized = (price - track.entry) * track.qty * track.side
    _fill(run, track, close_time, track.side < 0, SIDE[track.side], kind, reason, track.qty, price, realized)
    _log(run, track, close_time, "exit", f"close_{SIDE[track.side].lower()}", "papirfyll",
         f"lukket {SIDE[track.side].lower()} til {_price(price)}, resultat {_num(realized - track.qty * price * FEE)}"
         f" USDT ({reason}).")
    track.side, track.qty, track.entry, track.margin = 0, 0.0, 0.0, 0.0


def _wanted(run, value):
    side = int(np.sign(value))
    return 0 if run.direction == "long_only" and side < 0 else side


def _advance(run, track, data):
    """Behandle nye, avsluttede lys i rekkefølge: nødstopp, ventetid etter start og skriptets mål."""
    for row in data[data.close_time > track.last_close].itertuples(index=False):
        target = _wanted(run, row.target)
        if track.side:
            stop = track.entry * (1 - track.side * STOP_SHARE / run.leverage)
            if (row.low <= stop) if track.side > 0 else (row.high >= stop):
                _close(run, track, stop, row.close_time, "Nødstopp (90 % av marginen)", kind="stop")
                track.waiting = target
                track.last_close = int(row.close_time)
                continue
        if track.waiting is not None:
            if target == track.waiting:
                track.last_close = int(row.close_time)
                continue
            track.waiting = None
        if target != track.side:
            if track.side:
                _close(run, track, row.close, row.close_time, "Snu" if target else "Utgang")
            if target:
                _open(run, track, target, row.close, row.close_time, "Inngang")
        track.last_close = int(row.close_time)
    if len(data):
        track.last_price = float(data.close.iloc[-1])


def _symbols_in_use(exclude):
    return {s for slot, run in _STATE["runs"].items() if slot != exclude and run.running for s in run.tracks}


# ---------------------------------------------------------------- knappene

def start(slot, label, key, coin_symbols, timeframe, minutes, direction, margin_pct, leverage, accepted,
          bigscalp_tp=0.0, bigscalp_sl=0.0):
    """Start papirhandel for ett felt. Returnerer en kort melding til feltet."""
    if not accepted:
        return "Huk av for at du forstår at dette er papirhandel før du starter."
    if not coin_symbols:
        return "Velg minst én kryptovaluta."
    try:
        margin_pct, leverage = float(margin_pct), int(leverage)
    except (TypeError, ValueError):
        return "Margin og gearing må være tall."
    if not 0.01 <= margin_pct <= 10 or not 1 <= leverage <= 50:
        return "Margin må være 0,01–10 % og gearing 1–50×."
    with _LOCK:
        busy = sorted(set(coin_symbols) & _symbols_in_use(slot))
        if busy:
            return f"{', '.join(_coin(s) for s in busy)} papirhandles allerede i et annet felt."
        previous = _STATE["runs"].get(slot)
        if previous is not None:
            for track in previous.tracks.values():
                if track.side:
                    _close(previous, track, track.last_price if np.isfinite(track.last_price) else track.entry,
                           int(time.time() * 1000), "Lukket ved ny start")
        run = Run(slot, label, key, timeframe, int(minutes), direction, margin_pct, leverage,
                  float(bigscalp_tp or 0), float(bigscalp_sl or 0), started=int(time.time() * 1000))
        problems = []
        for symbol in coin_symbols:
            track = Track(symbol)
            try:
                data = targets(key, symbol, timeframe, int(minutes), run.bigscalp_tp, run.bigscalp_sl)
            except Exception:
                problems.append(_coin(symbol))
                continue
            last = data.iloc[-1]
            track.last_close, track.last_price = int(last.close_time), float(last.close)
            first = _wanted(run, last.target)
            if key in JOIN_AT_START and first:
                _open(run, track, first, float(last.close), int(last.close_time), "Inngang (blir med ved start)")
            elif first:
                track.waiting = first
                _log(run, track, last.close_time, f"{SIDE[first].lower()}_plan", "wait", "venter",
                     "skriptet har allerede en plan; venter på et nytt bekreftet signal.")
            run.tracks[symbol] = track
        if not run.tracks:
            return "Lysene kunne ikke hentes akkurat nå. Prøv igjen om litt."
        _STATE["runs"][slot] = run
    note = f" Kunne ikke hente lys for {', '.join(problems)}." if problems else ""
    coins = ", ".join(_coin(s) for s in run.tracks)
    return (f"▶ Papirhandel startet for Skript {slot}: **{label}** på {coins} ({minutes} min), margin "
            f"{_num(margin_pct, 2)} %, gearing {leverage}×. Ingen ordre sendes.{note}")


def stop(slot):
    with _LOCK:
        run = _STATE["runs"].get(slot)
        if run is None or not run.running:
            return f"Skript {slot} kjører ikke."
        run.running = False
    return f"⏹ Skript {slot} er stoppet. Papirposisjonene beholdes til du lukker dem."


def close(slot):
    with _LOCK:
        run = _STATE["runs"].get(slot)
        if run is None:
            return f"Skript {slot} har ingen papirposisjoner."
        for track in run.tracks.values():
            if track.side:
                price = track.last_price if np.isfinite(track.last_price) else track.entry
                _close(run, track, price, int(time.time() * 1000), "Lukket manuelt")
        run.running = False
    return f"✖ Papirposisjonene i Skript {slot} er lukket, og skriptet er stoppet."


def refresh():
    """Behandle nye lys for alle felt som kjører, og oppdater siste pris for stoppede felt med posisjon."""
    with _LOCK:
        runs = list(_STATE["runs"].values())
    for run in runs:
        for track in list(run.tracks.values()):
            if not run.running and not track.side:
                continue
            try:
                data = targets(run.key, track.symbol, run.timeframe, run.minutes, run.bigscalp_tp, run.bigscalp_sl)
            except Exception:
                continue
            with _LOCK:
                if run.running:
                    _advance(run, track, data)
                elif len(data):
                    track.last_price = float(data.close.iloc[-1])


# ---------------------------------------------------------------- tabeller og status

def fills_for(symbol):
    """Papirfyll for én mynt, i samme form som grafene bruker for fyll."""
    with _LOCK:
        return [dict(f) for f in _STATE["fills"] if f["symbol"] == symbol]


def _unrealized(track):
    if not track.side or not np.isfinite(track.last_price):
        return 0.0
    return (track.last_price - track.entry) * track.qty * track.side


def states_table():
    rows = []
    with _LOCK:
        for slot, run in sorted(_STATE["runs"].items()):
            for track in run.tracks.values():
                if not run.running and not track.side:
                    continue
                if track.side:
                    position = (f"{'🟢' if track.side > 0 else '🔴'} {SIDE[track.side]} {track.qty:.6g} "
                                f"{_coin(track.symbol)} til {_price(track.entry)} · urealisert "
                                f"{_num(_unrealized(track))} USDT")
                else:
                    position = "⚪ Ingen posisjon"
                if not run.running:
                    step = "Stoppet. Posisjonen beholdes til du lukker den." if track.side else "Stoppet."
                elif track.waiting is not None:
                    step = "Venter på et nytt bekreftet signal fra skriptet."
                else:
                    step = f"Følger {run.label} ved neste lukking ({run.minutes} min)."
                rows.append([f"{_coin(track.symbol)} · Skript {slot}", position, step])
    return pd.DataFrame(rows, columns=STATE_COLUMNS)


def fills_table():
    with _LOCK:
        fills = list(reversed(_STATE["fills"]))
    rows = [[_stamp(f["time"]), _coin(f["symbol"]), "KJØP" if f["side"] == "BUY" else "SELG",
             f["position_side"].lower(), f["reason"], round(f["qty"], 6), round(f["price"], 6),
             round(f["commission"], 6), round(f["realized_pnl"], 4), f["order_id"]] for f in fills]
    return pd.DataFrame(rows, columns=FILL_COLUMNS)


def decisions_table():
    with _LOCK:
        decisions = list(reversed(_STATE["decisions"]))
    rows = [[_stamp(d["time"]), _coin(d["symbol"]), d["signal"], d["action"], d["status"],
             f"{d['symbol']}: {d['message']}"] for d in decisions]
    return pd.DataFrame(rows, columns=DECISION_COLUMNS)


def status_text():
    with _LOCK:
        runs = sorted(_STATE["runs"].items())
        unrealized = sum(_unrealized(t) for _, run in runs for t in run.tracks.values())
        realized, fees = _STATE["realized"], _STATE["fees"]
    active = [f"Skript {slot}: {run.label} på {', '.join(_coin(s) for s in run.tracks)} ({run.minutes} min)"
              for slot, run in runs if run.running]
    lines = [f"**Papirhandel:** papirsaldo {_num(PAPER_BALANCE, 0)} USDT ved start · realisert {_num(realized)} USDT "
             f"(etter gebyr {_num(fees)} USDT) · urealisert {_num(unrealized)} USDT · egenkapital "
             f"{_num(PAPER_BALANCE + realized + unrealized)} USDT."]
    lines.append("**Aktive felt:** " + ("; ".join(active) if active else "ingen. Velg skript og mynter, og trykk Start."))
    lines.append("Simulert: ingen ordre sendes, og ingen konto eller nøkler brukes. Papirfyll skjer til lysets "
                 "sluttkurs med 0,08 % gebyr per side og nødstopp ved 90 % tapt margin.")
    return "\n\n".join(lines)
