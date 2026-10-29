"""Uavhengig Bigscalp-rekonstruksjon fra offentlig beskrivelse og grafverdier.

Ren signalberegning på avsluttede WEEX-lys; ingen konto, HTTP eller ordre.
Python-reglene kontrolleres mot vedlagt Pine-kilde og den offentlige 5m-grafen.
For 15m–1d kommer filterets 5m-lys eksplisitt fra samme WEEX-kilde. Manglende
historisk filterdata gir ingen innganger; en manglende siste filterbar avviser
hele beslutningen. Alternative moduser og HTF-formelen er åpne tolkninger.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
PINE_FILE = "tradingview/bigscalp_premium_reconstruction.pine"
PINE_SHA256 = "21cd16fb129aa51bb78ee1d070c0b254cd7efb81cf6cc8316a253445b360dcef"
LABEL = "Bigscalp Premium (rekonstruksjon · ADD/X)"
MIN_BARS = 5


@dataclass(frozen=True)
class Profile:
    """Offentlige standarder, med valgfri overstyring for dokumenterte tester."""
    entry_mode: str = "Normal"
    htf_minutes: int = 5
    direction: str = "Both"
    add_mode: str = "One ADD"
    min_extension: float = 0.0
    exit_mode: str = "Normal"
    enable_tp: bool = False
    tp_pct: float = 1.0
    enable_sl: bool = False
    sl_pct: float = 0.5

    def validate(self):
        if (self.entry_mode not in ("Conservative", "Normal", "Aggressive")
                or self.direction not in ("Both", "Long Only", "Short Only")
                or self.add_mode not in ("Disabled", "One ADD", "Unlimited")
                or self.exit_mode not in ("Tight", "Normal", "Wide")):
            raise ValueError("Ukjent Bigscalp-modus.")
        if type(self.htf_minutes) is not int or self.htf_minutes < 1:
            raise ValueError("Bigscalp-filteret må være et positivt antall minutter.")
        for name, minimum in (("min_extension", 0), ("tp_pct", .1), ("sl_pct", .1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < minimum:
                raise ValueError("Ugyldig Bigscalp-verdi: " + name)
        if type(self.enable_tp) is not bool or type(self.enable_sl) is not bool:
            raise ValueError("TP/SL må slås eksplisitt av eller på.")


def _prepare(bars, minutes):
    """Ikke fyll hull, sorter om priser eller finn på volum i motoren."""
    if type(minutes) is not int or minutes < 1:
        raise ValueError("Bigscalp-intervallet må være et positivt helt antall minutter.")
    if not isinstance(bars, pd.DataFrame) or bars.empty or not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("Bigscalp trenger faktiske WEEX-lys med tidsindeks.")
    if bars.index.tz is None:
        raise ValueError("WEEX-lysenes tidssone mangler.")
    frame = bars.copy()
    frame.index = pd.to_datetime(frame.index, utc=True).as_unit("ms")
    t = frame.index.asi8
    if (frame.index.hasnans or frame.index.has_duplicates or not frame.index.is_monotonic_increasing
            or np.any(t % (minutes * 60_000)) or np.any(np.diff(t) != minutes * 60_000)):
        raise ValueError("Hull eller ugyldig tidsjustering i Bigscalp-lysene.")
    try:
        values = frame[["open", "high", "low", "close", "volume"]].to_numpy(float)
    except (KeyError, TypeError, ValueError):
        raise ValueError("Bigscalp trenger open, high, low, close og volume.") from None
    if not np.isfinite(values).all():
        raise ValueError("Bigscalp-lysene inneholder ugyldige priser eller volum.")
    o, h, l, c, v = values.T
    if (np.any(values[:, :4] <= 0) or np.any(v < 0) or np.any(h < np.maximum(o, c))
            or np.any(l > np.minimum(o, c)) or np.any(h < l)):
        raise ValueError("Ugyldige OHLCV-verdier i Bigscalp-lysene.")
    for name in ("valid", "available"):
        if name in frame and not frame[name].fillna(False).to_numpy(bool).all():
            raise ValueError("Ugyldige eller utilgjengelige WEEX-lys.")
    for name in ("synthetic", "filled", "missing"):
        if name in frame and frame[name].fillna(True).to_numpy(bool).any():
            raise ValueError("Bigscalp handler ikke på syntetiske eller utfylte lys.")
    frame[["open", "high", "low", "close", "volume"]] = values
    return frame


def _filter_values(frame, minutes, profile, filter_bars):
    target = profile.htf_minutes
    if target == minutes:
        other = frame
    elif target > minutes:
        if target % minutes:
            raise ValueError("HTF må være et helt multiplum av basisintervallet.")
        epoch = frame.index.asi8
        step = target * 60_000
        anchor = 4 * 86_400_000 if target == 10080 else 0
        buckets = (epoch - anchor) // step * step + anchor
        grouped = frame.groupby(buckets, sort=True)
        other = grouped.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                            close=("close", "last"), volume=("volume", "sum"))
        counts = grouped.size()
        other = other.loc[counts == target // minutes].copy()
        other.index = pd.to_datetime(other.index, unit="ms", utc=True)
    else:
        if filter_bars is None:
            raise ValueError(f"Bigscalp trenger egne avsluttede WEEX-lys på {target} min til filteret.")
        other = _prepare(filter_bars, target)
    out = np.full((len(frame), 4), np.nan)
    if other.empty:
        return out
    # Høyere rammer: kun forrige ferdige HTF-lys ved basislysets åpning.
    # Lavere/lik ramme: siste ferdige intralys ved basislysets lukking.
    query = frame.index.asi8 + (0 if target > minutes else minutes * 60_000)
    closes = other.index.as_unit("ms").asi8 + target * 60_000
    j = np.searchsorted(closes, query, side="right") - 1
    valid = j >= 1
    if target <= minutes:
        # Ingen videreføring av et gammelt filterlys inn i et datahull.
        valid &= closes[np.maximum(j, 0)] == query
    for k in np.flatnonzero(valid):
        at = j[k]
        out[k] = (other.close.iloc[at], other.close.iloc[at - 1],
                  other.high.iloc[at - 1], other.low.iloc[at - 1])
    return out


def compute(bars, minutes=5, *, filter_bars=None, profile=None):
    """L/S = faktiske innganger; rå Long/Short-varsler er egne kolonner."""
    p = profile or Profile()
    p.validate()
    frame = _prepare(bars, minutes)
    source = (ROOT / PINE_FILE).read_bytes()
    if hashlib.sha256(source).hexdigest() != PINE_SHA256:
        raise ValueError("Bigscalp-kilden stemmer ikke med den testede motorprofilen.")
    fc, fp, fh, fl = _filter_values(frame, minutes, p, filter_bars).T
    o, h, l, c = (frame[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    prev_h, prev_l = np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]]
    bl, bs = c > prev_h, c < prev_l
    up, down = fc > fp, fc < fp
    if p.entry_mode == "Conservative":
        lf = up & (fc > fh) & np.r_[False, bl[:-1]]
        sf = down & (fc < fl) & np.r_[False, bs[:-1]]
    elif p.entry_mode == "Aggressive":
        lf = sf = np.ones(len(frame), dtype=bool)
    else:
        lf, sf = up, down
    lc = bl & lf & (p.direction != "Short Only")
    sc = bs & sf & (p.direction != "Long Only")
    la, sa = lc & ~np.r_[False, lc[:-1]], sc & ~np.r_[False, sc[:-1]]
    if p.exit_mode == "Tight":
        red, green = c < o, c > o
        xl, xs = red & np.r_[False, red[:-1]], green & np.r_[False, green[:-1]]
    elif p.exit_mode == "Wide":
        xl = c < np.minimum(prev_l, np.r_[np.nan, np.nan, l[:-2]])
        xs = c > np.maximum(prev_h, np.r_[np.nan, np.nan, h[:-2]])
    else:
        xl, xs = c < prev_l, c > prev_h
    fields = ("buy", "sell", "add", "exit", "tp_hit", "sl_hit", "side", "entry", "sl", "tp", "trigger",
              "adds", "plan_id", "event_side")
    rows = []
    side, entry, trigger, born, adds, plan_id = 0, np.nan, np.nan, -1, 0, np.nan
    for k in range(len(frame)):
        buy = sell = added = exited = tp_hit = sl_hit = False
        event_side = side
        if side:
            sl = entry * (1 - side * p.sl_pct / 100) if p.enable_sl else np.nan
            tp = entry * (1 + side * p.tp_pct / 100) if p.enable_tp else np.nan
            sl_hit = bool(p.enable_sl and k > born and (l[k] <= sl if side == 1 else h[k] >= sl))
            target_hit = bool(p.enable_tp and k > born and (h[k] >= tp if side == 1 else l[k] <= tp))
            if sl_hit or target_hit or (xl[k] if side == 1 else xs[k]):
                exited, tp_hit = True, bool(not sl_hit and target_hit)
                side, adds, trigger = 0, 0, np.nan
            else:
                extension = c[k] > trigger * (1 + p.min_extension / 100) if side == 1 else c[k] < trigger * (1 - p.min_extension / 100)
                trend = p.entry_mode == "Aggressive" or (up[k] if side == 1 else down[k])
                allowed = p.add_mode == "Unlimited" or (p.add_mode == "One ADD" and adds == 0)
                if allowed and extension and trend and k > born:
                    added, adds = True, adds + 1
                    trigger = h[k] if side == 1 else l[k]
        elif lc[k] or sc[k]:
            side = 1 if lc[k] else -1
            buy, sell = side == 1, side == -1
            entry, trigger, born, adds = c[k], h[k] if side == 1 else l[k], k, 0
            plan_id = int(frame.index.asi8[k]) + minutes * 60_000
            event_side = side
        sl = entry * (1 - side * p.sl_pct / 100) if side and p.enable_sl else np.nan
        tp = entry * (1 + side * p.tp_pct / 100) if side and p.enable_tp else np.nan
        rows.append((buy, sell, added, exited, tp_hit, sl_hit, side, entry, sl, tp, trigger, adds, plan_id, event_side))
    frame[list(fields)] = pd.DataFrame(rows, index=frame.index, columns=fields)
    frame["long_alert"], frame["short_alert"] = la, sa
    frame["exit_long_rule"], frame["exit_short_rule"] = xl, xs
    frame["context_ready"] = np.isfinite(fc) & np.isfinite(fp)
    frame["long_condition"], frame["short_condition"] = lc, sc
    return frame


def evaluate(bars, minutes=5, *, filter_bars=None, profile=None):
    """Fryst standardprofil til WEEX; boolsk fresh gjelder kun siste faktiske inngang."""
    frame = compute(bars, minutes, filter_bars=filter_bars, profile=profile)
    row = frame.iloc[-1]
    valid = len(frame) >= MIN_BARS and bool(row.context_ready)
    signal_ms = int(frame.index.as_unit("ms").asi8[-1]) + minutes * 60_000
    return {"valid": valid, "fresh": valid and bool(row.buy or row.sell),
            "target": int(row.side), "signal_close": signal_ms,
            "plan_id": int(row.plan_id) if np.isfinite(row.plan_id) else None,
            "entry": float(row.entry) if np.isfinite(row.entry) else None,
            "add": valid and bool(row['add']), "exit": bool(row['exit']),
            "exit_long_rule": bool(row.exit_long_rule), "exit_short_rule": bool(row.exit_short_rule),
            "frame": frame, "source_sha256": PINE_SHA256,
            "reason": "Bekreftede Bigscalp-signaler." if valid else "Bigscalp mangler ferske 5-minutters filterlys."}


def figure(frame, symbol, fills=(), minutes=5):
    """Signalmerker ved beslutningens close; fyll er egne spor fra kontoens logg."""
    shown = frame.tail(180)
    times = shown.index + pd.Timedelta(minutes=minutes)
    fig = go.Figure(go.Candlestick(x=times, open=shown.open, high=shown.high, low=shown.low, close=shown.close,
                                  name="WEEX"))
    for key, title, shape, color in (("buy", "L", "triangle-up", "#16a34a"),
                                    ("sell", "S", "triangle-down", "#dc2626"),
                                    ("add", "ADD", "diamond", "#2563eb"),
                                    ("exit", "X", "x", "#ea580c")):
        mask = shown[key].to_numpy(bool)
        fig.add_trace(go.Scatter(x=times[mask], y=shown.close.to_numpy()[mask], mode="markers", name=title,
                                 marker={"symbol": shape, "size": 10, "color": color}))
    for key, color in (("entry", "#64748b"), ("sl", "#dc2626"), ("tp", "#16a34a")):
        fig.add_trace(go.Scatter(x=times, y=shown[key].where(shown.side.ne(0)), name=key.upper(),
                                 line={"color": color, "dash": "dot"}, connectgaps=False))
    if fills:
        from weex_autotrade_dashboard import _fill_traces
        recent = [f for f in fills if f.get("time") and int(f['time']) >= int(shown.index[0].timestamp() * 1000)]
        if recent:
            _fill_traces(fig, recent)
    fig.update_layout(template="plotly_white", height=430, xaxis_rangeslider_visible=False,
                      title=f"{symbol} · Bigscalp-rekonstruksjon · {minutes}m", margin={"l": 40, "r": 20, "t": 40, "b": 30})
    return fig


def signals_table(frame, minutes=5):
    events = frame.loc[frame.buy | frame.sell | frame['add'] | frame['exit']].tail(20).iloc[::-1]
    rows = []
    for time_, row in events.iterrows():
        event = "L" if row.buy else "S" if row.sell else "ADD" if row['add'] else "SL" if row.sl_hit else "TP" if row.tp_hit else "X"
        rows.append([(time_ + pd.Timedelta(minutes=minutes)).strftime("%d.%m.%Y %H:%M"), event, float(row.close)])
    return pd.DataFrame(rows, columns=["Bekreftet ved close (UTC)", "Signal", "Referansepris"])


def dashboard(view, symbol, minutes=5):
    row = view['frame'].iloc[-1]
    side = {1: "LONG", -1: "SHORT", 0: "FLAT"}[int(row.side)]
    context = "klart" if view['valid'] else "mangler siste ferdige 5m-lys; ingen ny inngang"
    return (f"**{symbol} · Bigscalp-rekonstruksjon · {minutes} min** · Signalposisjon **{side}** · "
            f"5m-filter: {context}. Normal inngang/utgang, ett ADD. TP/SL er av i originalens standardprofil; "
            "motoren har fortsatt børsnødstopp. Grafens referanseposisjon er ikke en fyllbekreftelse.")
