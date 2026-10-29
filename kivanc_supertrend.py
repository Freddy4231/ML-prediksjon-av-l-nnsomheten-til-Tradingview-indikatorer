"""SuperTrend STRATEGY [KivancOzbilgic] — sanntidsindikator, visning og kontinuerlig WEEX-handel.

Denne modulen porterer brukerens Pine Script® v4-strategi:
«SuperTrend STRATEGY»
© KivancOzbilgic — lisensiert under Mozilla Public License 2.0 (https://mozilla.org/MPL/2.0/).

Alle de originale komponentene og standardinnstillingene fra Pine-skriptet er ivaretatt:
1. ATR Period (standard 10), Source (`hl2`), ATR Multiplier (standard 3.0)
2. Change ATR Calculation Method (`changeATR`, standard `true` = Wilder RMA `atr(Periods)`, `false` = `sma(tr, Periods)`)
3. SuperTrend-bånd:
   - `up = src - Multiplier * atr`, `up := close[1] > up1 ? max(up, up1) : up`
   - `dn = src + Multiplier * atr`, `dn := close[1] < dn1 ? min(dn, dn1) : dn`
   - `trend := trend == -1 and close > dn1 ? 1 : trend == 1 and close < up1 ? -1 : trend`
4. Signaler og Stop-and-Reverse (ta profitt og snu posisjon):
   - `buySignal = trend == 1 and trend[1] == -1` → lukk eventuell SHORT (ta profitt) og gå inn i LONG (`BUY`).
   - `sellSignal = trend == -1 and trend[1] == 1` → lukk eventuell LONG (ta profitt) og gå inn i SHORT (`SELL`).
5. Highlighter-fyll mellom `mPlot` (`ohlc4`) og `upPlot` / `dnPlot` samt `barcolor` basert på `barssince(buySignal)` vs `barssince(sellSignal)`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research.clean_wave import pine_atr

ATTRIBUTION = "SuperTrend STRATEGY © KivancOzbilgic (Mozilla Public License 2.0)"
PINE_FILE = "tradingview/supertrend_kivanc_strategy.pine"
PINE_LICENSE_HEADER = (
    "// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/\n"
    "// © KivancOzbilgic"
)

DEFAULT_PARAMS: dict[str, Any] = {
    "periods": 10,
    "multiplier": 3.0,
    "change_atr": True,
    "show_signals": True,
    "highlighting": True,
    "bar_coloring": True,
}

TIMEFRAMES: dict[str, tuple[str, int]] = {
    "5 minutter": ("5", 5),
    "15 minutter": ("15", 15),
    "30 minutter": ("30", 30),
    "1 time": ("60", 60),
    "4 timer": ("240", 240),
    "1 dag": ("1440", 1440),
}

# Signalet er bekreftet først når lyset lukker; motoren handler da (ca. 10 s etter). Tabellene viser derfor
# lukketidspunktet, som er det samme som «Tid (UTC)» i tabellen over utførte handler.
SIGNAL_TABLE_COLUMNS = ["Handel ved lukking (UTC)", "Signallys åpnet (UTC)", "Signal", "Handling", "Pris (USDT)",
                        "SuperTrend-nivå"]


def bar_step(frame: pd.DataFrame, minutes: int | None = None) -> pd.Timedelta:
    """Lengden på ett lys: oppgitt intervall, ellers vanligste avstand mellom lysene."""
    if minutes:
        return pd.Timedelta(minutes=int(minutes))
    diffs = pd.to_datetime(frame["datetime"], utc=True).diff().dropna()
    return diffs.mode().iloc[0] if not diffs.empty else pd.Timedelta(minutes=5)


DATA_SOURCES: dict[str, str] = {
    "WEEX futures": "weex",
    "Binance spot": "binance",
}


@dataclass(frozen=True)
class KivancSupertrendSummary:
    symbol: str
    timeframe_label: str
    source_label: str
    bar_time: str
    close: float
    supertrend_level: float
    trend: int
    position: int
    action: str
    last_signal_time: str
    last_signal_price: float
    buy_count: int
    sell_count: int
    periods: int
    multiplier: float
    change_atr: bool


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.r_[np.nan, close[:-1]]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    if len(tr) > 0:
        tr[0] = high[0] - low[0]
    return tr


def prepare(bars: pd.DataFrame, minutes: int = 5, gap_bars: int = 200) -> tuple[pd.DataFrame, bool]:
    """Rens og fyll eventuelle manglende tidssteg slik at ATR og SuperTrend-tilstand blir presis."""
    if bars is None or bars.empty:
        raise ValueError("Ingen prisstolper tilgjengelig.")

    df = bars.copy()
    if "timestamp" not in df.columns:
        if "open_time" in df.columns:
            ts_num = pd.to_numeric(df["open_time"], errors="coerce")
            if ts_num.notna().all():
                arr = ts_num.to_numpy(dtype=np.int64)
                df["timestamp"] = np.where(arr < 10**11, arr * 1000, arr).astype(np.int64)
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            else:
                dt_ser = pd.to_datetime(df["open_time"], utc=True)
                df["datetime"] = dt_ser
                df["timestamp"] = dt_ser.dt.as_unit("ms").astype("int64").to_numpy(dtype=np.int64)
        elif isinstance(df.index, pd.DatetimeIndex):
            idx_utc = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            df["timestamp"] = idx_utc.as_unit("ms").asi8.astype(np.int64)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            raise ValueError("Prisstolpene mangler tidsstempel.")
    else:
        ts_num = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts_num.notna().all():
            arr = ts_num.to_numpy(dtype=np.int64)
            df["timestamp"] = np.where(arr < 10**11, arr * 1000, arr).astype(np.int64)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            dt_ser = pd.to_datetime(df["timestamp"], utc=True)
            df["datetime"] = dt_ser
            df["timestamp"] = dt_ser.dt.as_unit("ms").astype("int64").to_numpy(dtype=np.int64)

    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    if df.empty:
        raise ValueError("Ingen gyldige OHLC-stolper etter rensing.")

    df["timestamp"] = df["timestamp"].astype(np.int64)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    step_ms = max(int(minutes), 1) * 60_000
    diffs = np.diff(df["timestamp"].to_numpy(dtype=np.int64))
    recent_tail = diffs[-min(len(diffs), 60):] if len(diffs) > 0 else np.array([], dtype=np.int64)
    recent_gap = bool(np.any(recent_tail > step_ms * max(int(gap_bars), 2)))

    df["valid"] = True
    max_gap_ms = step_ms * max(int(gap_bars), 2)
    if len(diffs) > 0 and np.any(diffs > max_gap_ms):
        last_gap_pos = int(np.flatnonzero(diffs > max_gap_ms)[-1]) + 1
        df.loc[: last_gap_pos - 1, "valid"] = False

    return df, recent_gap


def compute(
    bars: pd.DataFrame,
    minutes: int = 5,
    periods: int = 10,
    multiplier: float = 3.0,
    change_atr: bool = True,
) -> pd.DataFrame:
    """Beregn KivancOzbilgic SuperTrend STRATEGY eksakt etter Pine v4-koden."""
    frame, _ = prepare(bars, minutes=minutes)
    valid_mask = frame["valid"].to_numpy(dtype=bool)
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) == 0:
        raise ValueError("Ingen sammenhengende prisdata å beregne SuperTrend på.")

    sub = frame.iloc[valid_indices].copy()
    n = len(sub)
    open_p = sub["open"].to_numpy(dtype=float)
    high_p = sub["high"].to_numpy(dtype=float)
    low_p = sub["low"].to_numpy(dtype=float)
    close_p = sub["close"].to_numpy(dtype=float)
    ts = sub["timestamp"].to_numpy(dtype=np.int64)

    p_len = max(1, int(periods))
    mult = float(multiplier)
    src = (high_p + low_p) / 2.0
    ohlc4 = (open_p + high_p + low_p + close_p) / 4.0

    if change_atr:
        atr = pine_atr(high_p, low_p, close_p, p_len)
    else:
        tr = _true_range(high_p, low_p, close_p)
        atr = pd.Series(tr).rolling(p_len, min_periods=p_len).mean().to_numpy(dtype=float)

    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    trend = np.ones(n, dtype=int)
    buy_signal = np.zeros(n, dtype=bool)
    sell_signal = np.zeros(n, dtype=bool)
    signal_epoch = np.zeros(n, dtype=np.int64)
    bar_color = [""] * n

    last_buy_idx = -1
    last_sell_idx = -1
    curr_epoch = ts[0] if n > 0 else 0

    for i in range(n):
        if np.isnan(atr[i]):
            up[i] = np.nan
            dn[i] = np.nan
            trend[i] = trend[i - 1] if i > 0 else 1
            signal_epoch[i] = curr_epoch
            continue

        raw_up = src[i] - mult * atr[i]
        raw_dn = src[i] + mult * atr[i]

        up1 = up[i - 1] if (i > 0 and not np.isnan(up[i - 1])) else raw_up
        dn1 = dn[i - 1] if (i > 0 and not np.isnan(dn[i - 1])) else raw_dn

        if i > 0 and close_p[i - 1] > up1:
            up[i] = max(raw_up, up1)
        else:
            up[i] = raw_up

        if i > 0 and close_p[i - 1] < dn1:
            dn[i] = min(raw_dn, dn1)
        else:
            dn[i] = raw_dn

        prev_trend = trend[i - 1] if i > 0 else 1
        if prev_trend == -1 and close_p[i] > dn1:
            curr_trend = 1
        elif prev_trend == 1 and close_p[i] < up1:
            curr_trend = -1
        else:
            curr_trend = prev_trend

        trend[i] = curr_trend
        if i > 0 and curr_trend == 1 and prev_trend == -1:
            buy_signal[i] = True
            last_buy_idx = i
            curr_epoch = ts[i]
        elif i > 0 and curr_trend == -1 and prev_trend == 1:
            sell_signal[i] = True
            last_sell_idx = i
            curr_epoch = ts[i]

        signal_epoch[i] = curr_epoch

        # Pine: buy1 = barssince(buySignal), sell1 = barssince(sellSignal), color1 bruker buy1[1] vs sell1[1]
        if last_buy_idx >= 0 and last_sell_idx >= 0:
            bar_color[i] = "green" if last_buy_idx > last_sell_idx else "red"
        elif last_buy_idx >= 0:
            bar_color[i] = "green"
        elif last_sell_idx >= 0:
            bar_color[i] = "red"
        else:
            bar_color[i] = "green" if curr_trend == 1 else "red"

    st_line = np.where(trend == 1, up, dn)
    up_plot = np.where(trend == 1, up, np.nan)
    dn_plot = np.where(trend == -1, dn, np.nan)

    for name, arr in {
        "atr": atr,
        "up": up,
        "dn": dn,
        "up_plot": up_plot,
        "dn_plot": dn_plot,
        "supertrend": st_line,
        "ohlc4": ohlc4,
    }.items():
        full = np.full(len(frame), np.nan)
        full[valid_indices] = arr
        frame[name] = full

    for name, arr in {
        "buy_signal": buy_signal,
        "sell_signal": sell_signal,
    }.items():
        full = np.zeros(len(frame), dtype=bool)
        full[valid_indices] = arr
        frame[name] = full

    full_trend = np.ones(len(frame), dtype=int)
    full_trend[valid_indices] = trend
    frame["trend"] = full_trend
    frame["position"] = full_trend

    full_epoch = np.zeros(len(frame), dtype=np.int64)
    full_epoch[valid_indices] = signal_epoch
    frame["signal_epoch"] = full_epoch

    full_color = ["green"] * len(frame)
    for local_i, global_i in enumerate(valid_indices):
        full_color[global_i] = bar_color[local_i]
    frame["bar_color"] = full_color

    return frame


def summarize(
    frame: pd.DataFrame,
    symbol: str,
    timeframe_label: str = "5 minutter",
    source_label: str = "WEEX futures",
    periods: int = 10,
    multiplier: float = 3.0,
    change_atr: bool = True,
) -> KivancSupertrendSummary:
    valid = frame[frame["valid"] & frame["supertrend"].notna()]
    if valid.empty:
        raise ValueError("For få stolper til å oppsummere SuperTrend STRATEGY.")

    last = valid.iloc[-1]
    trend = int(last["trend"])
    if bool(last["buy_signal"]):
        action = "TA PROFITT SHORT → GÅ INN LONG (BUY)"
    elif bool(last["sell_signal"]):
        action = "TA PROFITT LONG → GÅ INN SHORT (SELL)"
    elif trend == 1:
        action = "HOLD LONG (til neste Sell-signal)"
    else:
        action = "HOLD SHORT (til neste Buy-signal)"

    sig_rows = valid[valid["buy_signal"] | valid["sell_signal"]]
    if not sig_rows.empty:
        last_sig = sig_rows.iloc[-1]
        last_sig_time = pd.Timestamp(last_sig["datetime"]).strftime("%Y-%m-%d %H:%M UTC")
        last_sig_price = float(last_sig["close"])
    else:
        last_sig_time = "—"
        last_sig_price = float(last["close"])

    bar_time = pd.Timestamp(last["datetime"]).strftime("%Y-%m-%d %H:%M UTC")
    return KivancSupertrendSummary(
        symbol=symbol,
        timeframe_label=timeframe_label,
        source_label=source_label,
        bar_time=bar_time,
        close=float(last["close"]),
        supertrend_level=float(last["supertrend"]),
        trend=trend,
        position=trend,
        action=action,
        last_signal_time=last_sig_time,
        last_signal_price=last_sig_price,
        buy_count=int(valid["buy_signal"].sum()),
        sell_count=int(valid["sell_signal"].sum()),
        periods=int(periods),
        multiplier=float(multiplier),
        change_atr=bool(change_atr),
    )


def dashboard(
    frame: pd.DataFrame,
    symbol: str,
    source: str = "weex",
    minutes: int = 5,
    periods: int = 10,
    multiplier: float = 3.0,
    change_atr: bool = True,
) -> str:
    tf_label = next((k for k, v in TIMEFRAMES.items() if v[1] == int(minutes)), f"{minutes} minutter")
    src_label = next((k for k, v in DATA_SOURCES.items() if v == source), source)
    summary = summarize(
        frame,
        symbol=symbol,
        timeframe_label=tf_label,
        source_label=src_label,
        periods=periods,
        multiplier=multiplier,
        change_atr=change_atr,
    )
    pos_badge = "🟢 **LONG (+1)**" if summary.trend == 1 else "🔴 **SHORT (-1)**"
    atr_method = "Wilder RMA (`atr(Periods)`)" if change_atr else "SMA av True Range (`sma(tr, Periods)`)"

    return (
        f"### SuperTrend STRATEGY [KivancOzbilgic] (`{symbol}` · {tf_label})\n"
        f"*{PINE_LICENSE_HEADER}*\n\n"
        f"| Felt | Verdi |\n"
        f"| :--- | :--- |\n"
        f"| **Aktiv posisjon og handling** | {pos_badge} — {summary.action} |\n"
        f"| **Siste Buy/Sell-vending** | `{summary.last_signal_time}` @ `{summary.last_signal_price:,.4f} USDT` |\n"
        f"| **Siste lukkede stolpe** | `{summary.bar_time}` — Pris: `{summary.close:,.4f} USDT` |\n"
        f"| **Aktiv SuperTrend-linje** | `{summary.supertrend_level:,.4f} USDT` ({'Up Trend (støtte)' if summary.trend == 1 else 'Down Trend (motstand)'}) |\n"
        f"| **Parametre** | ATR Period: `{int(periods)}` · Multiplier: `{float(multiplier):.1f}` · ATR-metode: {atr_method} |\n"
        f"| **Totalt bekreftede signaler i vinduet** | Buy-signaler: `{summary.buy_count}` · Sell-signaler: `{summary.sell_count}` · Kilde: {src_label} |\n"
    )


def figure(
    frame: pd.DataFrame,
    symbol: str,
    fills: list[dict[str, Any]] | None = None,
    visible_bars: int = 180,
    minutes: int = 5,
    title_suffix: str = "",
) -> go.Figure:
    valid = frame[frame["valid"] & frame["supertrend"].notna()].copy()
    if valid.empty:
        valid = frame.copy()
    view_df = valid.tail(max(int(visible_bars), 40)).reset_index(drop=True)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.78, 0.22],
        subplot_titles=(
            f"{symbol} · SuperTrend STRATEGY [KivancOzbilgic] ({minutes}m){title_suffix}",
            "Kontinuerlig posisjonsretning (+1 = LONG, -1 = SHORT)",
        ),
    )

    bull_mask = (view_df["bar_color"] == "green").to_numpy(dtype=bool)
    bear_mask = ~bull_mask
    for mask, name, color in [
        (bull_mask, "UpTrend-lys (grønn)", "#00c853"),
        (bear_mask, "DownTrend-lys (rød)", "#ff1744"),
    ]:
        if not np.any(mask):
            continue
        sub = view_df[mask]
        fig.add_trace(
            go.Candlestick(
                x=sub["datetime"],
                open=sub["open"],
                high=sub["high"],
                low=sub["low"],
                close=sub["close"],
                name=name,
                increasing_line_color=color,
                increasing_fillcolor=color,
                decreasing_line_color=color,
                decreasing_fillcolor=color,
            ),
            row=1,
            col=1,
        )

    # Up Trend + Highlighter fill til ohlc4
    fig.add_trace(
        go.Scatter(
            x=view_df["datetime"],
            y=view_df["ohlc4"],
            mode="lines",
            line={"color": "rgba(255,255,255,0)", "width": 0},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=view_df["datetime"],
            y=view_df["up_plot"],
            mode="lines",
            name="Up Trend (grønn)",
            line={"color": "#00e676", "width": 2.5},
            fill="tonexty",
            fillcolor="rgba(0, 230, 118, 0.12)",
        ),
        row=1,
        col=1,
    )

    # Down Trend + Highlighter fill til ohlc4
    fig.add_trace(
        go.Scatter(
            x=view_df["datetime"],
            y=view_df["ohlc4"],
            mode="lines",
            line={"color": "rgba(255,255,255,0)", "width": 0},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=view_df["datetime"],
            y=view_df["dn_plot"],
            mode="lines",
            name="Down Trend (rød)",
            line={"color": "#ff1744", "width": 2.5},
            fill="tonexty",
            fillcolor="rgba(255, 23, 68, 0.12)",
        ),
        row=1,
        col=1,
    )

    # Signalmerket står ved lysets lukking: da er signalet bekreftet og motoren handler.
    step = bar_step(view_df, minutes)
    buys = view_df[view_df["buy_signal"]]
    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys["datetime"] + step,
                y=buys["up"],
                mode="markers+text",
                name="Buy (lukk short / gå long)",
                text=["Buy"] * len(buys),
                textposition="bottom center",
                marker={"color": "#00c853", "size": 11, "symbol": "triangle-up", "line": {"color": "#ffffff", "width": 1}},
                hovertemplate="<b>Buy</b> bekreftet ved lukking %{x|%d.%m %H:%M} UTC · motoren går long nå"
                              "<extra></extra>",
            ),
            row=1,
            col=1,
        )

    sells = view_df[view_df["sell_signal"]]
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells["datetime"] + step,
                y=sells["dn"],
                mode="markers+text",
                name="Sell (lukk long / gå short)",
                text=["Sell"] * len(sells),
                textposition="top center",
                marker={"color": "#d50000", "size": 11, "symbol": "triangle-down", "line": {"color": "#ffffff", "width": 1}},
                hovertemplate="<b>Sell</b> bekreftet ved lukking %{x|%d.%m %H:%M} UTC · motoren går short nå"
                              "<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if fills:
        fill_df = pd.DataFrame(fills)
        if not fill_df.empty and "time" in fill_df.columns and "price" in fill_df.columns:
            fill_df["dt"] = pd.to_datetime(fill_df["time"], unit="ms", utc=True)
            min_dt = view_df["datetime"].min()
            fill_df = fill_df[fill_df["dt"] >= min_dt]
            if not fill_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=fill_df["dt"],
                        y=fill_df["price"].astype(float),
                        mode="markers",
                        name="WEEX utførte handler (faktisk tid)",
                        marker={"color": "#ffd600", "size": 11, "symbol": "star", "line": {"color": "#000000", "width": 1}},
                    ),
                    row=1,
                    col=1,
                )

    fig.add_trace(
        go.Scatter(
            x=view_df["datetime"],
            y=view_df["position"],
            mode="lines",
            name="Aktiv posisjon (+1 Long / -1 Short)",
            line={"color": "#29b6f6", "width": 2, "shape": "hv"},
            fill="tozeroy",
            fillcolor="rgba(41, 182, 246, 0.15)",
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Pris (USDT)", row=1, col=1, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(
        title_text="Retning",
        tickvals=[-1, 1],
        ticktext=["SHORT (-1)", "LONG (+1)"],
        range=[-1.35, 1.35],
        row=2,
        col=1,
        gridcolor="rgba(255,255,255,0.08)",
    )
    fig.update_xaxes(rangeslider_visible=False, gridcolor="rgba(255,255,255,0.06)")
    fig.update_layout(
        template="plotly_dark",
        height=660,
        margin={"l": 50, "r": 30, "t": 55, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        hovermode="x unified",
    )
    return fig


def recent_signals_table(frame: pd.DataFrame, limit: int = 20, minutes: int | None = None) -> pd.DataFrame:
    valid = frame[frame["valid"] & (frame["buy_signal"] | frame["sell_signal"])].copy()
    if valid.empty:
        return pd.DataFrame(columns=SIGNAL_TABLE_COLUMNS)
    step = bar_step(frame, minutes)
    tail = valid.tail(limit).iloc[::-1]
    rows = []
    for _, row in tail.iterrows():
        is_buy = bool(row["buy_signal"])
        rows.append(
            {
                "Handel ved lukking (UTC)": (pd.Timestamp(row["datetime"]) + step).strftime("%Y-%m-%d %H:%M"),
                "Signallys åpnet (UTC)": pd.Timestamp(row["datetime"]).strftime("%Y-%m-%d %H:%M"),
                "Signal": "BUY (LONG)" if is_buy else "SELL (SHORT)",
                "Handling": "Ta profitt på short → åpne long" if is_buy else "Ta profitt på long → åpne short",
                "Pris (USDT)": round(float(row["close"]), 4),
                "SuperTrend-nivå": round(float(row["supertrend"]), 4) if pd.notna(row["supertrend"]) else None,
            }
        )
    return pd.DataFrame(rows)


def load_bars(
    symbol: str,
    source: str = "weex",
    minutes: int = 5,
    timeframe: str | None = None,
    limit: int = 1000,
    **_kwargs,
) -> pd.DataFrame:
    import triple_supertrend as ts
    from weex_autotrade_dashboard import market_bars

    tf_str = str(timeframe) if timeframe is not None else str(int(minutes))
    if source == "binance":
        return ts.fetch_binance(symbol, minutes=int(minutes))
    try:
        bars = market_bars(os.environ.get("WEEX_ENV", "simulation"), tf_str, symbol)
        return bars.tail(int(limit)) if limit and len(bars) > int(limit) else bars
    except Exception:
        bars = ts.fetch_binance(symbol, minutes=int(minutes))
        return bars.tail(int(limit)) if limit and len(bars) > int(limit) else bars


def view(
    coin: str = "SOL",
    timeframe_label: str = "5 minutter",
    source_label: str = "WEEX futures",
    periods: int = 10,
    multiplier: float = 3.0,
    change_atr: bool = True,
) -> tuple[str, go.Figure | None, pd.DataFrame]:
    from weex_autotrade_dashboard import COINS

    symbol = COINS.get(coin, coin if str(coin).endswith("USDT") else "SOLUSDT")
    _, minutes = TIMEFRAMES.get(timeframe_label, ("5", 5))
    source = DATA_SOURCES.get(source_label, "weex")
    try:
        bars = load_bars(symbol, source=source, minutes=minutes)
        frame = compute(
            bars,
            minutes=minutes,
            periods=int(periods),
            multiplier=float(multiplier),
            change_atr=bool(change_atr),
        )
        md = dashboard(
            frame,
            symbol=symbol,
            source=source,
            minutes=minutes,
            periods=int(periods),
            multiplier=float(multiplier),
            change_atr=bool(change_atr),
        )
        fig = figure(frame, symbol=symbol, visible_bars=180, minutes=minutes)
        table = recent_signals_table(frame, limit=20, minutes=minutes)
        return md, fig, table
    except Exception:
        return (
            "**SuperTrend STRATEGY kan ikke vises nå:** lysene kunne ikke hentes. Prøv igjen om litt.",
            None,
            pd.DataFrame(columns=SIGNAL_TABLE_COLUMNS),
        )


def compute_chandelier(
    bars: pd.DataFrame,
    minutes: int = 5,
    length: int = 22,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Beregn Chandelier Exit (22, 3) [everget] — det mest lønnsomme åpne skriptet (+174,8 % før 2025, +25,4 % i 2025–2026)."""
    frame, _ = prepare(bars, minutes=minutes)
    valid_mask = frame["valid"].to_numpy(dtype=bool)
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) == 0:
        raise ValueError("Ingen sammenhengende prisdata å beregne Chandelier Exit på.")

    sub = frame.iloc[valid_indices].copy()
    n = len(sub)
    open_p = sub["open"].to_numpy(dtype=float)
    high_p = sub["high"].to_numpy(dtype=float)
    low_p = sub["low"].to_numpy(dtype=float)
    close_p = sub["close"].to_numpy(dtype=float)
    ts = sub["timestamp"].to_numpy(dtype=np.int64)

    p = max(int(length), 1)
    mult = float(multiplier)
    atr_val = mult * pine_atr(high_p, low_p, close_p, p)
    long_raw = pd.Series(close_p).rolling(p, min_periods=1).max().to_numpy(dtype=float) - atr_val
    short_raw = pd.Series(close_p).rolling(p, min_periods=1).min().to_numpy(dtype=float) + atr_val

    up = np.copy(long_raw)
    dn = np.copy(short_raw)
    trend = np.ones(n, dtype=int)
    buy_sig = np.zeros(n, dtype=bool)
    sell_sig = np.zeros(n, dtype=bool)

    for i in range(n):
        if np.isnan(long_raw[i]) or np.isnan(short_raw[i]):
            continue
        if i == 0:
            trend[i] = 1
            continue
        up1 = up[i - 1] if not np.isnan(up[i - 1]) else long_raw[i]
        dn1 = dn[i - 1] if not np.isnan(dn[i - 1]) else short_raw[i]
        up[i] = max(long_raw[i], up1) if close_p[i - 1] > up1 else long_raw[i]
        dn[i] = min(short_raw[i], dn1) if close_p[i - 1] < dn1 else short_raw[i]
        prev_t = trend[i - 1]
        if close_p[i] > dn1:
            trend[i] = 1
        elif close_p[i] < up1:
            trend[i] = -1
        else:
            trend[i] = prev_t
        buy_sig[i] = (trend[i] == 1) and (prev_t == -1)
        sell_sig[i] = (trend[i] == -1) and (prev_t == 1)

    up_plot = np.where(trend == 1, up, np.nan)
    dn_plot = np.where(trend == 1, np.nan, dn)
    supertrend_line = np.where(trend == 1, up, dn)
    ohlc4 = (open_p + high_p + low_p + close_p) / 4.0

    signal_epoch = np.zeros(n, dtype=np.int64)
    last_epoch = ts[0] if n > 0 else 0
    for i in range(n):
        if buy_sig[i] or sell_sig[i]:
            last_epoch = ts[i]
        signal_epoch[i] = last_epoch

    out = frame.copy()
    for col, default in [
        ("atr", np.nan),
        ("up", np.nan),
        ("dn", np.nan),
        ("up_plot", np.nan),
        ("dn_plot", np.nan),
        ("supertrend", np.nan),
        ("ohlc4", np.nan),
        ("trend", 1),
        ("position", 1),
        ("buy_signal", False),
        ("sell_signal", False),
        ("signal_epoch", 0),
        ("bar_color", "green"),
    ]:
        out[col] = default

    idx = sub.index
    out.loc[idx, "atr"] = atr_val
    out.loc[idx, "up"] = up
    out.loc[idx, "dn"] = dn
    out.loc[idx, "up_plot"] = up_plot
    out.loc[idx, "dn_plot"] = dn_plot
    out.loc[idx, "supertrend"] = supertrend_line
    out.loc[idx, "ohlc4"] = ohlc4
    out.loc[idx, "trend"] = trend
    out.loc[idx, "position"] = trend
    out.loc[idx, "buy_signal"] = buy_sig
    out.loc[idx, "sell_signal"] = sell_sig
    out.loc[idx, "signal_epoch"] = signal_epoch
    out.loc[idx, "bar_color"] = np.where(trend == 1, "green", "red")
    return out


def view_chandelier(
    coin: str = "ETH",
    timeframe_label: str = "5 minutter",
    source_label: str = "WEEX futures",
    length: int = 22,
    multiplier: float = 3.0,
) -> tuple[str, go.Figure | None, pd.DataFrame]:
    from weex_autotrade_dashboard import COINS

    symbol = COINS.get(coin, coin if str(coin).endswith("USDT") else "ETHUSDT")
    _, minutes = TIMEFRAMES.get(timeframe_label, ("5", 5))
    source = DATA_SOURCES.get(source_label, "weex")
    try:
        bars = load_bars(symbol, source=source, minutes=minutes)
        frame = compute_chandelier(bars, minutes=minutes, length=int(length), multiplier=float(multiplier))
        s = summarize(frame, symbol=symbol, timeframe_label=timeframe_label, source_label=source_label,
                      periods=int(length), multiplier=float(multiplier), change_atr=True)
        pos_badge = "🟢 **LONG (+1)**" if s.position == 1 else "🔴 **SHORT (-1)**"
        md = (
            f"**{symbol} ({timeframe_label})** · **Chandelier Exit (22, 3) [everget]** · "
            f"Aktiv posisjon: {pos_badge} · Pris: `{s.close:,.4f} USDT` · "
            f"Chandelier-stopp: `{s.supertrend_level:,.4f} USDT`"
        )
        fig = figure(frame, symbol=symbol, visible_bars=180, minutes=minutes, title_suffix=" · Chandelier Exit (22, 3)")
        if fig.layout.annotations:
            fig.layout.annotations[0].text = f"{symbol} · Chandelier Exit (22, 3) [everget] ({minutes}m)"
        table = recent_signals_table(frame, limit=20, minutes=minutes)
        return md, fig, table
    except Exception:
        return (
            "**Chandelier Exit kan ikke vises nå:** lysene kunne ikke hentes. Prøv igjen om litt.",
            None,
            pd.DataFrame(columns=SIGNAL_TABLE_COLUMNS),
        )

