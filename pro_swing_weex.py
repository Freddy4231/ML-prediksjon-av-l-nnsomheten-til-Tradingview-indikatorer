"""Pro Swing: port av den uavhengige offentlige rekonstruksjonen for WEEX.

Kun profilen ``Public reference`` fra ``tradingview/pro_swing_recon.pine`` er støttet:
ATR200 (Wilder) × 8,4, låst kanalbredde, ett forsinket brudd, hovednivå 20 %, EMA8,
minst 20 barer mellom signaler, ett signal per trend og høyst 100 barers trendalder.
SL = en halv kanalbredde og TP = 1,5 ganger denne avstanden, målt fra signalets
hovednivå. Dette er ikke den beskyttede originalkoden til MQLSoftware.

Input inneholder bare avsluttede lys. Et signal blir kjent ved lysets lukking;
referanseinngangen er et tegningsnivå, ikke en historisk markedsfylling. Planen er
avsluttet etter første SL/TP-berøring på et senere lys. Ved begge nivåer på samme
lys gis SL prioritet, som i den innlimte koden. Børsens faktiske fyllinger avgjør
liveposisjonen. Ingen ordre, datahenting eller kontotilgang skjer i denne modulen.

Manglende tidssteg eller lys merket ugyldig/syntetisk nullstiller tilstanden.
Deretter kreves 200 nye sammenhengende virkelige lys. Det hindrer at manglende
markedsdata blir tolket som bekreftede signaler eller brukbare gamle planer.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PINE_FILE = "tradingview/pro_swing_recon.pine"
PINE_SHA256 = "5612eb82ff62093f9725c1dc4d1beb0cdbf4adfaa047551b0b4555d1e5b42574"
PROFILE = "Public reference"
ATR_LENGTH = 200
WIDTH_MULTIPLIER = 8.4
SIGNAL_GAP = 20
MAX_TREND_AGE = 100
ATTRIBUTION = "Pro Swing · uavhengig rekonstruksjon av offentlig utdata (ikke originalkoden)"
SIGNAL_TABLE_COLUMNS = ["Bekreftet (UTC)", "Signal", "Referanseinngang", "SL", "TP", "Planstatus"]


def _prepare(bars: pd.DataFrame, minutes: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Kontroller priser og bevar tidsaksen; lag aldri kunstige lys."""
    if bars is None or bars.empty:
        raise ValueError("Ingen avsluttede prisstolper tilgjengelig.")
    if isinstance(minutes, bool) or int(minutes) != minutes or int(minutes) <= 0:
        raise ValueError("Tidsintervallet må være et positivt helt antall minutter.")
    frame = bars.copy()
    if isinstance(frame.index, pd.DatetimeIndex):
        index = pd.to_datetime(frame.index, utc=True)
    else:
        key = next((c for c in ("timestamp", "open_time", "datetime") if c in frame), None)
        if key is None:
            raise ValueError("Prisstolpene mangler UTC-åpningstid.")
        values = frame[key]
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().all() and not pd.api.types.is_datetime64_any_dtype(values):
            unit = "s" if float(numeric.abs().max()) < 10**11 else "ms"
            index = pd.DatetimeIndex(pd.to_datetime(numeric, unit=unit, utc=True))
        else:
            index = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    if index.hasnans or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Prisstolpene må ha unike, stigende åpningstider.")
    epoch = index.as_unit("ms").asi8
    step = int(minutes) * 60_000
    if np.any(epoch % step != 0):
        raise ValueError("Åpningstidene passer ikke det valgte tidsintervallet.")
    for column in ("open", "high", "low", "close"):
        if column not in frame:
            raise ValueError(f"Prisstolpene mangler {column}.")
        frame[column] = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    values = frame[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("OHLC-priser må være endelige og positive.")
    o, h, l, c = values.T
    if np.any(h < np.maximum(o, c)) or np.any(l > np.minimum(o, c)) or np.any(h < l):
        raise ValueError("Ugyldig OHLC: high/low omslutter ikke open/close.")
    if "volume" not in frame:
        frame["volume"] = 0.0
    volume = pd.to_numeric(frame["volume"], errors="coerce").to_numpy(float)
    if not np.isfinite(volume).all() or np.any(volume < 0):
        raise ValueError("Volum må være endelig og ikke negativt.")
    frame["volume"] = volume
    usable = np.ones(len(frame), dtype=bool)
    for name in ("valid", "available"):
        if name in frame:
            usable &= frame[name].fillna(False).to_numpy(bool)
    for name in ("synthetic", "filled", "missing"):
        if name in frame:
            usable &= ~frame[name].fillna(True).to_numpy(bool)
    frame = frame.drop(columns=[c for c in ("datetime", "timestamp") if c in frame])
    frame.index = pd.DatetimeIndex(index.as_unit("ms"), name="open_time")
    frame["datetime"] = frame.index
    frame["timestamp"] = epoch
    breaks = np.r_[True, np.diff(epoch) != step] | ~usable
    # Første virkelige lys etter et ugyldig lys starter også en helt ny tilstand.
    breaks[1:] |= ~usable[:-1]
    frame["source_valid"] = usable
    return frame, breaks


def compute(bars: pd.DataFrame, minutes: int = 60, mintick: float = 0.0) -> pd.DataFrame:
    """Beregn den faste profilen uten framtidsdata; returner alle lys og låste planer.

    ``mintick`` kan settes til børsens minste prissteg. Standard 0 bevarer rå
    signalnivåer; avrunding til faktisk ordre skjer i WEEX-motoren.
    ``plan_id`` er signalets lukketid i millisekunder og er stabil for samme kilde.
    """
    if not np.isfinite(mintick) or mintick < 0:
        raise ValueError("Minste prissteg må være endelig og ikke negativt.")
    frame, breaks = _prepare(bars, minutes)
    n = len(frame)
    high, low, close = (frame[k].to_numpy(float) for k in ("high", "low", "close"))
    usable = frame["source_valid"].to_numpy(bool)
    arrays = {k: np.full(n, np.nan) for k in ("atr", "upper", "lower", "channel_width", "main", "midline",
                                             "entry", "sl", "tp")}
    integers = {k: np.zeros(n, dtype=np.int64) for k in ("trend", "trend_age", "plan_side", "plan_id")}
    booleans = {k: np.zeros(n, dtype=bool) for k in ("valid", "buy", "sell", "stop_event", "target_event",
                                                   "plan_active", "plan_stopped", "plan_target_hit")}
    consecutive = entries = direction = 0
    tr_sum = 0.0
    atr = upper = lower = width = mid = np.nan
    born = last_entry = None
    plan: dict[str, Any] | None = None
    step = int(minutes) * 60_000
    times = frame["timestamp"].to_numpy(np.int64)
    for i in range(n):
        if breaks[i]:
            consecutive = entries = direction = 0
            tr_sum = 0.0
            atr = upper = lower = width = mid = np.nan
            born = last_entry = plan = None
        if not usable[i]:
            continue
        previous = close[i - 1] if consecutive else np.nan
        tr = high[i] - low[i] if consecutive == 0 else max(high[i] - low[i], abs(high[i] - previous),
                                                         abs(low[i] - previous))
        consecutive += 1
        if consecutive <= ATR_LENGTH:
            tr_sum += tr
            if consecutive == ATR_LENGTH:
                atr = tr_sum / ATR_LENGTH
        else:
            atr = tr / ATR_LENGTH + (1.0 - 1.0 / ATR_LENGTH) * atr
        arrays["atr"][i] = atr
        candidate = atr * WIDTH_MULTIPLIER
        prev_direction = direction
        if np.isfinite(candidate) and candidate > 0:
            if not np.isfinite(width):
                direction = 1 if previous == previous and close[i] >= previous else -1
                width = candidate
                upper = high[i] if direction == 1 else low[i] + width
                lower = high[i] - width if direction == 1 else low[i]
                born = i
            elif (direction == -1 and previous > upper) or (direction == 1 and previous < lower):
                direction = -direction
                width = candidate
                upper = high[i] if direction == 1 else low[i] + width
                lower = high[i] - width if direction == 1 else low[i]
                born = i
            elif direction == 1:
                upper = max(upper, high[i])
                lower = upper - width
            else:
                lower = min(lower, low[i])
                upper = lower + width
        main = upper - width * 0.2 if direction == 1 else lower + width * 0.2
        middle = (upper + lower) * 0.5
        if np.isfinite(middle):
            mid = middle if not np.isfinite(mid) else middle * (2.0 / 9.0) + mid * (7.0 / 9.0)
        ready = bool(direction and np.isfinite(width) and width > mintick)
        age = i - born if ready else 0
        if ready and (not prev_direction or direction != prev_direction):
            entries = 0
        gap_pass = last_entry is None or i - last_entry >= SIGNAL_GAP
        retest = ((previous > main and low[i] <= main and close[i] > main) if direction == 1
                  else (previous < main and high[i] >= main and close[i] < main))
        event = bool(ready and retest and gap_pass and entries < 1 and age <= MAX_TREND_AGE)
        # Den gamle planen prøves først. Ingen SL/TP-berøring på eget signallys.
        if plan is not None and i > plan["born"] and not plan["stopped"]:
            stop = low[i] <= plan["sl"] if plan["side"] == 1 else high[i] >= plan["sl"]
            if stop:
                plan["stopped"] = True
                booleans["stop_event"][i] = True
            elif not plan["hit"] and (high[i] >= plan["tp"] if plan["side"] == 1 else low[i] <= plan["tp"]):
                plan["hit"] = True
                booleans["target_event"][i] = True
        if event:
            entries += 1
            last_entry = i
            distance = max(width * 0.5, mintick)
            plan = {"born": i, "side": direction, "id": int(times[i] + step), "entry": main,
                    "sl": main - direction * distance, "tp": main + direction * distance * 1.5,
                    "stopped": False, "hit": False}
            booleans["buy" if direction == 1 else "sell"][i] = True
        for key, value in (("upper", upper), ("lower", lower), ("channel_width", width), ("main", main),
                           ("midline", mid)):
            arrays[key][i] = value
        integers["trend"][i] = direction
        integers["trend_age"][i] = age
        booleans["valid"][i] = ready
        if plan is not None:
            for key in ("entry", "sl", "tp"):
                arrays[key][i] = plan[key]
            integers["plan_side"][i] = plan["side"]
            integers["plan_id"][i] = plan["id"]
            booleans["plan_active"][i] = not plan["stopped"] and not plan["hit"]
            booleans["plan_stopped"][i] = plan["stopped"]
            booleans["plan_target_hit"][i] = plan["hit"]
    for name, values in {**arrays, **integers, **booleans}.items():
        frame[name] = values
    frame["buy_signal"] = frame["buy"]
    frame["sell_signal"] = frame["sell"]
    frame["plan_time"] = pd.to_datetime(np.where(frame["plan_id"] > 0, frame["plan_id"], np.nan), unit="ms", utc=True)
    frame["bar_close"] = frame.index + pd.Timedelta(minutes=int(minutes))
    frame.attrs.update(profile=PROFILE, pine_sha256=PINE_SHA256, minutes=int(minutes), gap_policy="reset_200_real_bars")
    return frame


def evaluate(bars: pd.DataFrame, minutes: int = 60, mintick: float = 0.0) -> dict[str, Any]:
    """Siste bekreftede signal/plan, med separat friskhet og stabil identitet.

    En aktiv gammel plan er nyttig for visning, men ``fresh`` må være sann før
    motoren åpner en ny handel. Etter børsens SL/TP må motoren også huske utført
    ``plan_id``; gamle lys eller en annen prisfeed kan ikke autorisere gjenåpning.
    """
    frame = compute(bars, minutes=minutes, mintick=mintick)
    row = frame.iloc[-1]
    levels = np.array([row["entry"], row["sl"], row["tp"]], dtype=float)
    side = int(row["plan_side"])
    bracket = bool(np.isfinite(levels).all() and (levels > 0).all() and
                   (row["sl"] < row["entry"] < row["tp"] if side == 1 else
                    row["tp"] < row["entry"] < row["sl"] if side == -1 else False))
    valid = bool(row["valid"])
    active = bool(valid and bracket and row["plan_active"])
    fresh = bool(valid and bracket and (row["buy"] or row["sell"]))
    reason = ("Oppvarming eller manglende sammenhengende lys" if not valid else
              "SL eller TP er berørt" if side and not row["plan_active"] else
              "Ingen bekreftet handelsplan" if not side else
              "Handelsplanen har ugyldige prisnivåer" if not bracket else
              "Nytt bekreftet signal" if fresh else "Aktiv tidligere plan; vent på nytt signal for ny inngang")
    return {"valid": valid, "available": valid, "target": side if active else 0, "fresh": fresh,
            "plan_active": active, "plan_id": int(row["plan_id"]), "plan_time": row["plan_time"],
            "signal_close": row["plan_time"], "bar_open": frame.index[-1], "bar_close": row["bar_close"],
            "entry": float(row["entry"]), "sl": float(row["sl"]), "tp": float(row["tp"]),
            "close": float(row["close"]), "trend": int(row["trend"]), "reason": reason,
            "profile": PROFILE}


def recent_signals_table(frame: pd.DataFrame, limit: int = 20, minutes: int | None = None) -> pd.DataFrame:
    """Vis signalenes bekreftelsestid og de opprinnelige, absolutte nivåene."""
    rows = []
    signals = frame.loc[frame["valid"] & (frame["buy"] | frame["sell"])].tail(max(1, int(limit))).iloc[::-1]
    for _, row in signals.iterrows():
        same = frame.loc[frame["plan_id"] == row["plan_id"]].iloc[-1]
        status = "SL berørt" if same["plan_stopped"] else "TP berørt" if same["plan_target_hit"] else "Aktiv / erstattet av nyere signal"
        rows.append({"Bekreftet (UTC)": row["plan_time"].strftime("%Y-%m-%d %H:%M"),
                     "Signal": "KJØP / LONG" if row["buy"] else "SELG / SHORT",
                     "Referanseinngang": float(row["entry"]), "SL": float(row["sl"]), "TP": float(row["tp"]),
                     "Planstatus": status})
    return pd.DataFrame(rows, columns=SIGNAL_TABLE_COLUMNS)


def dashboard(frame: pd.DataFrame, symbol: str, minutes: int = 60) -> str:
    """Kort forklaring av profilen og siste plan; ingen påstand om avkastning."""
    row = frame.iloc[-1]
    signal = "Ingen signal ennå" if not row["plan_id"] else ("LONG" if row["plan_side"] == 1 else "SHORT")
    state = "Oppvarming" if not row["valid"] else "Aktiv" if row["plan_active"] else "Avsluttet / venter på signal"
    levels = "—" if not row["plan_id"] else f"{row['entry']:,.6g} · SL {row['sl']:,.6g} · TP {row['tp']:,.6g}"
    return (f"### Pro Swing · {symbol} · {int(minutes)} min\n"
            f"**{PROFILE}** · {signal} · **{state}**  \nReferanseinngang og nivåer: **{levels}**.\n\n"
            "ATR200 × 8,4 · hovednivå 20 % · én inngang per trend · SL ½ kanalbredde · TP 1,5 R. "
            "Signaler bekreftes når lyset lukker. Markedsordren kan fylles til en annen pris enn referansen. "
            "Kun denne faste profilen er koblet til motoren. Uavhengig rekonstruksjon av offentlige utdata; "
            "den beskyttede originalkoden er ikke brukt.")


def figure(frame: pd.DataFrame, symbol: str, fills: list[dict[str, Any]] | None = None,
           visible_bars: int = 180, minutes: int = 60) -> go.Figure:
    """Kanal, bekreftede kjøp/salg og aktive SL/TP med faktiske fyll som egne stjerner."""
    view = frame.tail(max(int(visible_bars), 40))
    fig = go.Figure(go.Candlestick(x=view.index, open=view["open"], high=view["high"], low=view["low"],
                                  close=view["close"], name="Pris", increasing_line_color="#00DCB9",
                                  decreasing_line_color="#EB5564"))
    for column, label, color, dash in (("upper", "Øvre kanal", "#538da8", None),
                                       ("lower", "Nedre kanal", "#538da8", None),
                                       ("main", "Hovednivå 20 %", "#d4b64c", "dot"),
                                       ("midline", "Midtlinje EMA8", "#ae8bf2", None)):
        fig.add_trace(go.Scatter(x=view.index, y=view[column], mode="lines", name=label,
                                 line={"color": color, "width": 1.2, "dash": dash}))
    for column, label, color, marker in (("buy", "KJØP / LONG", "#00DCB9", "triangle-up"),
                                        ("sell", "SELG / SHORT", "#EB5564", "triangle-down")):
        points = view.loc[view[column]]
        fig.add_trace(go.Scatter(x=points["bar_close"], y=points["entry"], mode="markers", name=label,
                                 marker={"color": color, "size": 12, "symbol": marker},
                                 hovertemplate=label + " bekreftet %{x|%d.%m %H:%M} UTC<br>Referanse %{y:.6g}<extra></extra>"))
    # Ny plan eller avslutning bryter linjen, slik at forskjellige planer ikke bindes sammen.
    for column, label, color in (("sl", "SL", "#EB5564"), ("tp", "TP", "#00DCB9")):
        x: list[Any] = []
        y: list[Any] = []
        last_id = None
        for index, row in view.iterrows():
            if row["plan_active"]:
                if row["plan_id"] != last_id:
                    x.append(None)
                    y.append(None)
                x.append(index + pd.Timedelta(minutes=int(minutes)))
                y.append(row[column])
                last_id = row["plan_id"]
            else:
                x.append(None)
                y.append(None)
                last_id = None
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label,
                                 line={"color": color, "width": 1.5, "dash": "dash"}, connectgaps=False))
    if fills:
        points = pd.DataFrame(fills)
        if {"time", "price"}.issubset(points.columns):
            when = pd.to_datetime(points["time"], unit="ms", utc=True, errors="coerce")
            price = pd.to_numeric(points["price"], errors="coerce")
            keep = when.notna() & price.notna() & (when >= view.index[0])
            fig.add_trace(go.Scatter(x=when[keep], y=price[keep], mode="markers", name="Faktiske WEEX-fyll",
                                     marker={"color": "#ffd600", "symbol": "star", "size": 11}))
    fig.update_layout(template="plotly_dark", title=f"Pro Swing · Public reference · {symbol} · {int(minutes)} min",
                      height=600, margin={"l": 50, "r": 25, "t": 55, "b": 35}, hovermode="x unified",
                      legend={"orientation": "h"}, xaxis_rangeslider_visible=False, yaxis_title="Pris (USDT)")
    return fig
