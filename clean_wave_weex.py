"""SMC Institutional Clean Wave & Structure PRO — sanntidsindikator og WEEX-signalmotor.

Original Pine Script® v6-indikator:
    "SMC Institutional Clean Wave & Structure PRO" ("Clean Wave Matrix PRO")
    © Dark_Ace_Master — lisensiert under Mozilla Public License 2.0
    https://mozilla.org/MPL/2.0/
Originalkoden ligger uendret i ``tradingview/clean_wave_original_indicator.pine``.

Denne modulen gjenbruker den eksakte, ikke-ommålende tilstandsmaskinen i
``research.clean_wave.clean_wave_state`` for sanntidsvisning (5m, 15m, 30m,
1t, 4t, 1d) og kontinuerlig LONG/SHORT-handel på WEEX.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research.clean_wave import CW_PARAMS, clean_wave_state

PINE_FILE = "tradingview/clean_wave_original_indicator.pine"
PINE_LICENSE_HEADER = (
    "SMC Institutional Clean Wave & Structure PRO (Clean Wave Matrix PRO) — "
    "© Dark_Ace_Master · Mozilla Public License 2.0 (https://mozilla.org/MPL/2.0/)"
)
ATTRIBUTION = PINE_LICENSE_HEADER

DEFAULT_PARAMS: dict[str, Any] = {
    "wave_len": 21,
    "wave_offset": 0.85,
    "wave_sigma": 6.0,
    "smc_sens": 7,
    "sig_sens": 10,
    "sd_len": 20,
    "sd_mult": 2.5,
    "atr_len": 14,
}

CW_MODES: dict[str, str] = {
    "combined": "BUY/SELL-merker + BOS/CHoCH-struktur + ALMA-bølge (anbefalt)",
    "badges": "Kun bekreftede BUY / SELL-merker (svingpivot 10)",
    "structure": "Kun markedsstruktur (BOS og CHoCH, pivot 7)",
    "wave": "Dynamisk trendbølge (ALMA 21: grønn = long, rød = short)",
}

TIMEFRAMES: dict[str, tuple[str, int]] = {
    "5 minutter": ("5", 5),
    "15 minutter": ("15", 15),
    "30 minutter": ("30", 30),
    "1 time": ("60", 60),
    "4 timer": ("240", 240),
    "1 dag": ("1440", 1440),
}

DATA_SOURCES: dict[str, str] = {
    "WEEX futures": "weex",
    "Binance spot (som TradingView BINANCE:BTCUSDT)": "binance",
}


@dataclass(frozen=True)
class CleanWaveSummary:
    symbol: str
    timeframe_label: str
    source_label: str
    mode: str
    bar_time: str
    close: float
    wave: float
    wave_up: bool
    trend_dir: int
    position: int
    action: str
    signal_reason: str
    sd_target: float
    inside_bar: bool
    last_signal_time: str
    last_signal_price: float
    bull_struct_count: int
    bear_struct_count: int
    buy_badge_count: int
    sell_badge_count: int


def prepare(bars: pd.DataFrame, gap_bars: int = 200, minutes: int = 5) -> pd.DataFrame:
    """Rens OHLCV-tabell (enten med DatetimeIndex eller timestamp-kolonne) og marker gyldige stolper."""
    if bars is None or bars.empty:
        raise ValueError("Ingen prisdata tilgjengelig.")

    frame = bars.copy()
    if "timestamp" not in frame.columns:
        if "open_time" in frame.columns:
            frame["timestamp"] = pd.to_numeric(frame["open_time"], errors="coerce")
        elif isinstance(frame.index, pd.DatetimeIndex):
            idx_utc = frame.index.tz_localize("UTC") if frame.index.tz is None else frame.index.tz_convert("UTC")
            frame["timestamp"] = (idx_utc.as_unit("ms").asi8).astype(np.int64)
        else:
            raise ValueError("Mangler tidskolonne (timestamp eller DatetimeIndex) i prisdata.")

    required = ["timestamp", "open", "high", "low", "close"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Mangler kolonner i prisdata: {', '.join(missing)}")

    for col in required:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    else:
        frame["volume"] = 0.0

    frame = frame.dropna(subset=required).sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Prisdata inneholder ingen gyldige rader.")

    frame["timestamp"] = frame["timestamp"].astype(np.int64)
    step_ms = max(int(minutes), 1) * 60_000
    diffs = frame["timestamp"].diff()
    gap_split = diffs > (step_ms * max(int(gap_bars), 2))
    segment_id = gap_split.cumsum()
    latest_segment = segment_id.iloc[-1]
    frame["valid"] = segment_id == latest_segment
    frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame.index = pd.DatetimeIndex(frame["datetime"], name="open_time")
    return frame


def compute(
    bars: pd.DataFrame,
    minutes: int = 5,
    mode: str = "combined",
    gap_bars: int = 200,
    **overrides: Any,
) -> pd.DataFrame:
    """Beregn Clean Wave Matrix PRO-indikatoren og kontinuerlige LONG/SHORT-signaler.

    Parametere:
        bars: OHLCV-DataFrame med DatetimeIndex eller ``timestamp``-kolonne.
        minutes: Stolpeintervall i minutter (standard 5).
        mode: Signalmodus fra ``CW_MODES`` (``combined``, ``badges``, ``structure``, ``wave``).
        gap_bars: Maks tillatt opphold i stolper før ny oppvarming.
        overrides: Valgfrie overstyringer av ``CW_PARAMS`` (f.eks. ``wave_len``, ``smc_sens``, ``sig_sens``).
    """
    if mode not in CW_MODES:
        raise ValueError(f"Ugyldig Clean Wave-modus: {mode}")

    params = dict(CW_PARAMS)
    params.update({k: v for k, v in overrides.items() if v is not None})

    frame = prepare(bars, gap_bars=gap_bars, minutes=minutes)
    valid_mask = frame["valid"].to_numpy(dtype=bool)
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) == 0:
        raise ValueError("Ingen sammenhengende prisdata å beregne Clean Wave på.")

    sub = frame.iloc[valid_indices].copy()
    state = clean_wave_state(sub[["open", "high", "low", "close", "volume"]], params=params)

    n = len(sub)
    high = sub["high"].to_numpy(dtype=float)
    low = sub["low"].to_numpy(dtype=float)
    ts = sub["timestamp"].to_numpy(dtype=np.int64)

    wave = state["wave"].to_numpy(dtype=float)
    wave_up = state["wave_up"].to_numpy(dtype=bool)
    wave_up_flip = state["wave_up_flip"].to_numpy(dtype=bool)
    wave_down_flip = state["wave_down_flip"].to_numpy(dtype=bool)
    inside = state["inside"].to_numpy(dtype=bool)
    ph_level = state["last_ph"].to_numpy(dtype=float)
    pl_level = state["last_pl"].to_numpy(dtype=float)
    struct_bull = state["struct_bull"].to_numpy(dtype=bool)
    struct_bear = state["struct_bear"].to_numpy(dtype=bool)
    struct_bull_choch = state["struct_bull_choch"].to_numpy(dtype=bool)
    struct_bear_choch = state["struct_bear_choch"].to_numpy(dtype=bool)
    trend_dir = state["trend_dir"].to_numpy(dtype=int)
    swing_high_conf = state["swing_high_confirmed"].to_numpy(dtype=bool)
    swing_low_conf = state["swing_low_confirmed"].to_numpy(dtype=bool)
    sd_mean = state["sma"].to_numpy(dtype=float)
    sd_std = state["sd"].to_numpy(dtype=float)
    sd_target = state["sd_target"].to_numpy(dtype=float)
    atr14 = state["atr"].to_numpy(dtype=float)

    # Finn eksakt nivå som ble brutt ved BOS / CHoCH for visning i grafen
    bos_level = np.full(n, np.nan)
    bos_label = [""] * n
    for i in range(n):
        if struct_bull[i]:
            bos_level[i] = ph_level[i]
            bos_label[i] = "CHoCH" if struct_bull_choch[i] else "BOS"
        if struct_bear[i]:
            bos_level[i] = pl_level[i]
            bos_label[i] = "CHoCH" if struct_bear_choch[i] else "BOS"

    # Kontinuerlig LONG (+1) / SHORT (-1) tilstandsmaskin uten ommaling
    warmup = max(int(params["wave_len"]), int(params["sd_len"]), int(params["sig_sens"]) * 2 + 1)
    position = np.zeros(n, dtype=int)
    buy_signal = np.zeros(n, dtype=bool)
    sell_signal = np.zeros(n, dtype=bool)
    signal_reason = [""] * n
    signal_epoch = np.zeros(n, dtype=np.int64)

    curr_pos = 0
    curr_epoch = np.int64(0)

    for i in range(n):
        if np.isnan(wave[i]) or i < min(warmup, n - 1):
            continue

        target = curr_pos
        reason = ""

        if mode == "wave":
            target = 1 if wave_up[i] else -1
            if target != curr_pos:
                reason = "ALMA-bølge opp" if target == 1 else "ALMA-bølge ned"
        elif mode == "badges":
            if swing_low_conf[i] and not swing_high_conf[i]:
                target = 1
                reason = "BUY-merke (bekreftet svingbunn)"
            elif swing_high_conf[i] and not swing_low_conf[i]:
                target = -1
                reason = "SELL-merke (bekreftet svingtopp)"
            elif curr_pos == 0:
                target = 1 if wave_up[i] else -1
                reason = "Oppstart fra ALMA-bølge"
        elif mode == "structure":
            if struct_bull[i] and not struct_bear[i]:
                target = 1
                reason = "Bullish CHoCH" if struct_bull_choch[i] else "Bullish BOS"
            elif struct_bear[i] and not struct_bull[i]:
                target = -1
                reason = "Bearish CHoCH" if struct_bear_choch[i] else "Bearish BOS"
            elif curr_pos == 0:
                if trend_dir[i] != 0:
                    target = int(trend_dir[i])
                    reason = "Aktiv markedsstruktur"
                else:
                    target = 1 if wave_up[i] else -1
                    reason = "Oppstart fra ALMA-bølge"
        else:
            # "combined": BUY/SELL-merker + BOS/CHoCH-struktur, støttet av ALMA-bølge ved oppstart/konflikt
            bull_hit = bool(swing_low_conf[i] or struct_bull[i])
            bear_hit = bool(swing_high_conf[i] or struct_bear[i])
            if bull_hit and not bear_hit:
                target = 1
                if swing_low_conf[i] and struct_bull[i]:
                    reason = "BUY-merke + " + ("CHoCH" if struct_bull_choch[i] else "BOS")
                elif swing_low_conf[i]:
                    reason = "BUY-merke (bekreftet svingbunn)"
                else:
                    reason = "Bullish CHoCH" if struct_bull_choch[i] else "Bullish BOS"
            elif bear_hit and not bull_hit:
                target = -1
                if swing_high_conf[i] and struct_bear[i]:
                    reason = "SELL-merke + " + ("CHoCH" if struct_bear_choch[i] else "BOS")
                elif swing_high_conf[i]:
                    reason = "SELL-merke (bekreftet svingtopp)"
                else:
                    reason = "Bearish CHoCH" if struct_bear_choch[i] else "Bearish BOS"
            elif bull_hit and bear_hit:
                target = 1 if wave_up[i] else -1
                reason = "ALMA-bølge (konfliktløser)"
            elif curr_pos == 0:
                if trend_dir[i] != 0:
                    target = int(trend_dir[i])
                    reason = "Aktiv markedsstruktur"
                else:
                    target = 1 if wave_up[i] else -1
                    reason = "Oppstart fra ALMA-bølge"

        if target != curr_pos:
            if target == 1:
                buy_signal[i] = True
            elif target == -1:
                sell_signal[i] = True
            curr_pos = target
            curr_epoch = ts[i]
            signal_reason[i] = reason or ("LONG" if target == 1 else "SHORT")

        position[i] = curr_pos
        signal_epoch[i] = curr_epoch

    # Plasseringsankere for historiske BUY / SELL-etiketter (som i Pine: idx = bar_index - sig_sens)
    sig_sens = int(params["sig_sens"])
    badge_buy_price = np.full(n, np.nan)
    badge_sell_price = np.full(n, np.nan)
    confirm_buy_price = np.full(n, np.nan)
    confirm_sell_price = np.full(n, np.nan)
    for i in range(n):
        atr_pad = (atr14[i] if not np.isnan(atr14[i]) else (high[i] - low[i])) * 0.3
        if swing_low_conf[i]:
            confirm_buy_price[i] = low[i] - atr_pad
            piv_idx = i - sig_sens
            if piv_idx >= 0:
                badge_buy_price[piv_idx] = low[piv_idx] - atr_pad
        if swing_high_conf[i]:
            confirm_sell_price[i] = high[i] + atr_pad
            piv_idx = i - sig_sens
            if piv_idx >= 0:
                badge_sell_price[piv_idx] = high[piv_idx] + atr_pad

    cols_float = {
        "wave": wave,
        "ph_level": ph_level,
        "pl_level": pl_level,
        "bos_level": bos_level,
        "sd_mean": sd_mean,
        "sd_std": sd_std,
        "sd_target": sd_target,
        "atr14": atr14,
        "badge_buy_price": badge_buy_price,
        "badge_sell_price": badge_sell_price,
        "confirm_buy_price": confirm_buy_price,
        "confirm_sell_price": confirm_sell_price,
    }
    for name, arr in cols_float.items():
        full = np.full(len(frame), np.nan)
        full[valid_indices] = arr
        frame[name] = full

    cols_bool = {
        "wave_up": wave_up,
        "wave_up_flip": wave_up_flip,
        "wave_down_flip": wave_down_flip,
        "inside": inside,
        "struct_bull": struct_bull,
        "struct_bear": struct_bear,
        "struct_bull_choch": struct_bull_choch,
        "struct_bear_choch": struct_bear_choch,
        "swing_high_confirmed": swing_high_conf,
        "swing_low_confirmed": swing_low_conf,
        "buy_signal": buy_signal,
        "sell_signal": sell_signal,
    }
    for name, arr in cols_bool.items():
        full = np.zeros(len(frame), dtype=bool)
        full[valid_indices] = arr
        frame[name] = full

    full_trend = np.zeros(len(frame), dtype=int)
    full_trend[valid_indices] = trend_dir
    frame["trend_dir"] = full_trend

    full_pos = np.zeros(len(frame), dtype=int)
    full_pos[valid_indices] = position
    frame["position"] = full_pos

    full_epoch = np.zeros(len(frame), dtype=np.int64)
    full_epoch[valid_indices] = signal_epoch
    frame["signal_epoch"] = full_epoch

    full_bos_lbl = [""] * len(frame)
    full_reason = [""] * len(frame)
    for local_i, global_i in enumerate(valid_indices):
        full_bos_lbl[global_i] = bos_label[local_i]
        full_reason[global_i] = signal_reason[local_i]
    frame["bos_label"] = full_bos_lbl
    frame["signal_reason"] = full_reason

    return frame


def summarize(
    frame: pd.DataFrame,
    symbol: str,
    timeframe_label: str = "5 minutter",
    source_label: str = "WEEX futures",
    mode: str = "combined",
) -> CleanWaveSummary:
    """Oppsummer siste stolpe og signalhistorikk for dashbord og WEEX-handel."""
    valid = frame[frame["valid"] & frame["wave"].notna()]
    if valid.empty:
        raise ValueError("For få stolper til å oppsummere Clean Wave Matrix PRO.")

    last = valid.iloc[-1]
    pos = int(last["position"])
    if bool(last["buy_signal"]):
        action = "NY LONG (BUY)"
    elif bool(last["sell_signal"]):
        action = "NY SHORT (SELL)"
    elif pos == 1:
        action = "HOLD LONG"
    elif pos == -1:
        action = "HOLD SHORT"
    else:
        action = "AVVENT"

    sig_rows = valid[valid["buy_signal"] | valid["sell_signal"]]
    if not sig_rows.empty:
        last_sig = sig_rows.iloc[-1]
        last_sig_time = pd.Timestamp(last_sig["datetime"]).strftime("%Y-%m-%d %H:%M UTC")
        last_sig_price = float(last_sig["close"])
        reason = str(last_sig["signal_reason"] or action)
    else:
        last_sig_time = "—"
        last_sig_price = float(last["close"])
        reason = str(last["signal_reason"] or "—")

    bar_time = pd.Timestamp(last["datetime"]).strftime("%Y-%m-%d %H:%M UTC")
    return CleanWaveSummary(
        symbol=symbol,
        timeframe_label=timeframe_label,
        source_label=source_label,
        mode=mode,
        bar_time=bar_time,
        close=float(last["close"]),
        wave=float(last["wave"]),
        wave_up=bool(last["wave_up"]),
        trend_dir=int(last["trend_dir"]),
        position=pos,
        action=action,
        signal_reason=reason,
        sd_target=float(last["sd_target"]) if pd.notna(last["sd_target"]) else float("nan"),
        inside_bar=bool(last["inside"]),
        last_signal_time=last_sig_time,
        last_signal_price=last_sig_price,
        bull_struct_count=int(valid["struct_bull"].sum()),
        bear_struct_count=int(valid["struct_bear"].sum()),
        buy_badge_count=int(valid["swing_low_confirmed"].sum()),
        sell_badge_count=int(valid["swing_high_confirmed"].sum()),
    )


def dashboard(
    frame: pd.DataFrame,
    symbol: str,
    mode: str = "combined",
    source: str = "weex",
    minutes: int = 5,
    wave_len: int = 21,
    smc_sens: int = 7,
    sig_sens: int = 10,
    sd_len: int = 20,
) -> str:
    """Markdown-oppsummering av Clean Wave Matrix PRO for både indikatorvisning og WEEX-panelet."""
    tf_label = next((k for k, v in TIMEFRAMES.items() if v[1] == int(minutes)), f"{minutes} minutter")
    src_label = next((k for k, v in DATA_SOURCES.items() if v == source), source)
    summary = summarize(
        frame,
        symbol=symbol,
        timeframe_label=tf_label,
        source_label=src_label,
        mode=mode,
    )
    pos_badge = (
        "🟢 **LONG (+1)**"
        if summary.position == 1
        else ("🔴 **SHORT (-1)**" if summary.position == -1 else "⚪ **FLAT (0)**")
    )
    wave_badge = "🟢 Bullish (pris ≥ ALMA)" if summary.wave_up else "🔴 Bearish (pris < ALMA)"
    struct_badge = (
        "🟢 Bullish (+1)"
        if summary.trend_dir > 0
        else ("🔴 Bearish (-1)" if summary.trend_dir < 0 else "⚪ Nøytral (0)")
    )
    inside_txt = "Ja (`#b2dfdb`)" if summary.inside_bar else "Nei"

    return (
        f"### SMC Institutional Clean Wave & Structure PRO (`{symbol}` · {tf_label})\n"
        f"*{PINE_LICENSE_HEADER}*\n\n"
        f"| Felt | Verdi |\n"
        f"| :--- | :--- |\n"
        f"| **Aktiv WEEX-posisjonsretning** | {pos_badge} ({summary.action}) |\n"
        f"| **Siste signal og årsak** | `{summary.last_signal_time}` @ `{summary.last_signal_price:,.4f}` — {summary.signal_reason} |\n"
        f"| **Siste lukkede stolpe** | `{summary.bar_time}` — Pris: `{summary.close:,.4f} USDT` |\n"
        f"| **Dynamisk trendbølge (ALMA {int(wave_len)})** | `{summary.wave:,.4f}` — {wave_badge} |\n"
        f"| **Markedsstruktur (BOS / CHoCH, pivot {int(smc_sens)})** | {struct_badge} (Bullish brudd: {summary.bull_struct_count}, Bearish brudd: {summary.bear_struct_count}) |\n"
        f"| **Bekreftede svingmerker (pivot {int(sig_sens)})** | BUY-merker: {summary.buy_badge_count} · SELL-merker: {summary.sell_badge_count} |\n"
        f"| **Neste -2.5 SD målnivå (SMA/SD {int(sd_len)})** | `{summary.sd_target:,.4f} USDT` |\n"
        f"| **Inside Bar / konsolidering** | {inside_txt} |\n"
        f"| **Signalmodus** | {CW_MODES.get(mode, mode)} · Kilde: {src_label} |\n"
        + weex_strategies_markdown()
    )


def weex_strategies_markdown() -> str:
    """Viser forskningsoversikten over andre lønnsomme strategier på WEEX (fra reports/weex_strategies/)."""
    import json
    from pathlib import Path

    sel_path = Path("reports/weex_strategies/selection.json")
    ret_path = Path("reports/weex_strategies/retrospective.json")
    if not (sel_path.exists() and ret_path.exists()):
        return ""
    try:
        sel = json.loads(sel_path.read_text())["results"]
        ret = json.loads(ret_path.read_text())["methods"]
    except Exception:
        return ""

    labels = {
        "hybrid_carry_breakout": "**1. Kombinert Kanalbrudd + Funding-Carry** *(anbefalt)*",
        "clean_wave_trend": "**2. Clean Wave Matrix PRO + 20d Trendregime**",
        "multi_horizon_breakout": "**3. Fler-horisont Donchian-ensemble (10d/20d/40d)**",
        "cross_sectional_rotation": "**4. Relativ styrke-rotasjon (topp 3 av 7 WEEX-mynter)**",
        "channel_baseline": "Referanse: 20-dagers kanalbrudd (kun long)",
        "clean_wave_raw_sar": "Advarsel: Ufiltrert Clean Wave SAR (kontinuerlig flip)",
        "buy_and_hold": "Referanse: Kjøp og hold",
    }
    rows = [
        "\n#### Andre strategier for å tjene penger på WEEX (testet 2020–2024, kontrollert 2025–2026)",
        "| Strategi | Kvalifisert før 2025 | BTC 2020–2024 | Pos. år | Værste DD | BTC 2025–2026 | 7 WEEX-mynter 2025–2026 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for key, label in labels.items():
        if key not in sel or key not in ret:
            continue
        s_btc = sel[key]["coins"]["BTCUSDT"]
        r_m = ret[key]
        r_btc = r_m["coins"]["BTCUSDT"]["normal"]
        qual = "Ja" if sel[key]["qualified"] else "Nei"
        rows.append(
            f"| {label} | {qual} | `{s_btc['compound']*100:+.1f} %` | `{s_btc['positive_years']}/5` | "
            f"`{s_btc['worst_drawdown']*100:.1f} %` | `{r_btc['total_return']*100:+.1f} %` | "
            f"`{r_m['portfolio_7_return']*100:+.1f} %` (DD `{r_m['portfolio_7_drawdown']*100:.1f} %`) |"
        )
    return "\n".join(rows) + "\n"



def figure(
    frame: pd.DataFrame,
    symbol: str,
    fills: list[dict[str, Any]] | None = None,
    visible_bars: int = 180,
    minutes: int = 5,
    title_suffix: str = "",
) -> go.Figure:
    """Lag interaktiv Plotly-graf som gjenskaper Clean Wave Matrix PRO-indikatoren."""
    valid = frame[frame["valid"]].copy()
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
            f"{symbol} · SMC Institutional Clean Wave & Structure PRO ({minutes}m){title_suffix}",
            "Kontinuerlig posisjonsretning (+1 = LONG, -1 = SHORT)",
        ),
    )

    # 1. Smart Candle Coloring Engine (grønn, rød og konsolidering / inside bar #b2dfdb)
    inside_mask = view_df["inside"].to_numpy(dtype=bool)
    bull_mask = (~inside_mask) & (view_df["close"] >= view_df["open"]).to_numpy(dtype=bool)
    bear_mask = (~inside_mask) & (view_df["close"] < view_df["open"]).to_numpy(dtype=bool)

    for mask, name, color in [
        (bull_mask, "Bullish lys (#00c853)", "#00c853"),
        (bear_mask, "Bearish lys (#d50000)", "#d50000"),
        (inside_mask, "Inside / konsolidering (#b2dfdb)", "#b2dfdb"),
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
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # 2. Dynamic Trend Wave (ALMA 21) + bakgrunnsfyll mellom bølge og pris
    up_wave = view_df["wave"].where(view_df["wave_up"])
    down_wave = view_df["wave"].where(~view_df["wave_up"])

    fig.add_trace(
        go.Scatter(
            x=view_df["datetime"],
            y=view_df["close"],
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
            y=up_wave,
            mode="lines",
            name="Trendbølge (bullish #00e676)",
            line={"color": "#00e676", "width": 2.5},
            fill="tonexty",
            fillcolor="rgba(0, 230, 118, 0.10)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=view_df["datetime"],
            y=view_df["close"],
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
            y=down_wave,
            mode="lines",
            name="Trendbølge (bearish #ff1744)",
            line={"color": "#ff1744", "width": 2.5},
            fill="tonexty",
            fillcolor="rgba(255, 23, 68, 0.10)",
        ),
        row=1,
        col=1,
    )

    # 3. Market Structure Shifts (BOS & CHoCH)
    for idx, row in view_df[view_df["struct_bull"] | view_df["struct_bear"]].iterrows():
        level = row["bos_level"]
        if pd.isna(level):
            continue
        is_choch = bool(row["struct_bull_choch"] or row["struct_bear_choch"])
        col = "#7b1fa2" if is_choch else "#0288d1"
        lbl = str(row["bos_label"] or ("CHoCH" if is_choch else "BOS"))
        start_idx = max(0, int(idx) - 5)
        x0 = view_df.loc[start_idx, "datetime"]
        x1 = row["datetime"]
        fig.add_shape(
            type="line",
            x0=x0,
            x1=x1,
            y0=float(level),
            y1=float(level),
            line={"color": col, "width": 1.5, "dash": "dash"},
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=x1,
            y=float(level),
            text=lbl,
            showarrow=False,
            font={"color": col, "size": 10},
            yshift=10 if bool(row["struct_bull"]) else -10,
            row=1,
            col=1,
        )

    # 4. Confirmed Major Swing Signals (BUY / SELL badges). Merket står på svingbunnen/-toppen, men er først
    # kjent sig_sens lys senere (pivoten må ha sig_sens lys på hver side). Motoren handler ved bekreftelsen.
    pivot_known = pd.Timedelta(minutes=int(minutes)) * (int(CW_PARAMS.get("sig_sens", 10)) + 1)
    buy_pivots = view_df[view_df["badge_buy_price"].notna()]
    if not buy_pivots.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_pivots["datetime"],
                y=buy_pivots["badge_buy_price"],
                mode="markers+text",
                name="BUY-sving (tegnet i ettertid, ikke inngang)",
                text=["BUY"] * len(buy_pivots),
                textposition="bottom center",
                marker={"symbol": "triangle-up-open", "size": 11, "color": "#00c853"},
                textfont={"color": "#00c853", "size": 10},
                customdata=(buy_pivots["datetime"] + pivot_known).dt.strftime("%d.%m %H:%M"),
                hovertemplate="<b>BUY-sving</b> (bunn %{x|%d.%m %H:%M} UTC)<br>Kjent først %{customdata} UTC, "
                              "når svingen er bekreftet.<br>Tegnes bakover i tid; ikke et inngangstidspunkt.<extra></extra>",
            ),
            row=1,
            col=1,
        )

    sell_pivots = view_df[view_df["badge_sell_price"].notna()]
    if not sell_pivots.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_pivots["datetime"],
                y=sell_pivots["badge_sell_price"],
                mode="markers+text",
                name="SELL-sving (tegnet i ettertid, ikke inngang)",
                text=["SELL"] * len(sell_pivots),
                textposition="top center",
                marker={"symbol": "triangle-down-open", "size": 11, "color": "#d50000"},
                textfont={"color": "#d50000", "size": 10},
                customdata=(sell_pivots["datetime"] + pivot_known).dt.strftime("%d.%m %H:%M"),
                hovertemplate="<b>SELL-sving</b> (topp %{x|%d.%m %H:%M} UTC)<br>Kjent først %{customdata} UTC, "
                              "når svingen er bekreftet.<br>Tegnes bakover i tid; ikke et inngangstidspunkt.<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Markør ved lukkingen av bekreftelsesstolpen: da er signalet kjent og boten utfører LONG / SHORT.
    step = pd.Timedelta(minutes=int(minutes))
    exec_buys = view_df[view_df["buy_signal"]]
    if not exec_buys.empty:
        fig.add_trace(
            go.Scatter(
                x=exec_buys["datetime"] + step,
                y=exec_buys["low"] * 0.9985,
                mode="markers",
                name="WEEX LONG-inngang (bekreftet)",
                marker={
                    "symbol": "circle",
                    "size": 9,
                    "color": "#00e676",
                    "line": {"color": "#ffffff", "width": 1.5},
                },
                customdata=exec_buys["signal_reason"],
                hovertemplate="<b>LONG</b> bekreftet ved lukking %{x|%d.%m %H:%M} UTC (motoren handler nå)<br>"
                              "Årsak: %{customdata}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    exec_sells = view_df[view_df["sell_signal"]]
    if not exec_sells.empty:
        fig.add_trace(
            go.Scatter(
                x=exec_sells["datetime"] + step,
                y=exec_sells["high"] * 1.0015,
                mode="markers",
                name="WEEX SHORT-inngang (bekreftet)",
                marker={
                    "symbol": "circle",
                    "size": 9,
                    "color": "#ff1744",
                    "line": {"color": "#ffffff", "width": 1.5},
                },
                customdata=exec_sells["signal_reason"],
                hovertemplate="<b>SHORT</b> bekreftet ved lukking %{x|%d.%m %H:%M} UTC (motoren handler nå)<br>"
                              "Årsak: %{customdata}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # 5. Target Standard Deviation Level (-2.5 SD Next Target)
    last_row = view_df.iloc[-1]
    if pd.notna(last_row["sd_target"]):
        sd_val = float(last_row["sd_target"])
        start_sd_idx = max(0, len(view_df) - 30)
        x_sd_start = view_df.loc[start_sd_idx, "datetime"]
        x_sd_end = last_row["datetime"]
        fig.add_shape(
            type="line",
            x0=x_sd_start,
            x1=x_sd_end,
            y0=sd_val,
            y1=sd_val,
            line={"color": "#ff9100", "width": 2, "dash": "dash"},
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=x_sd_end,
            y=sd_val,
            text=f"Next -2.5 SD Target ({sd_val:,.2f})",
            showarrow=False,
            xanchor="right",
            yshift=-12,
            font={"color": "#ff9100", "size": 11},
            row=1,
            col=1,
        )

    # Faktiske WEEX-utførelser (støtter både Ledger-fyll med «time» og enkle lister med «ts»)
    if fills:
        fill_times = []
        fill_prices = []
        fill_labels = []
        fill_colors = []
        min_ts = int(view_df["timestamp"].iloc[0])
        for item in fills:
            ts_val = int(item.get("time") or item.get("ts") or 0)
            price_val = item.get("price")
            if ts_val >= min_ts and price_val:
                fill_times.append(datetime.fromtimestamp(ts_val / 1000.0, tz=timezone.utc))
                fill_prices.append(float(price_val))
                if "side" in item and "position_side" in item:
                    act = f"{item['side']} {item['position_side']} ({item.get('qty', '')})"
                    is_long = item["position_side"] == "LONG"
                else:
                    act = str(item.get("action") or "ordre")
                    is_long = "long" in act.lower()
                fill_labels.append(act)
                fill_colors.append("#00e676" if is_long else "#ff1744")
        if fill_times:
            fig.add_trace(
                go.Scatter(
                    x=fill_times,
                    y=fill_prices,
                    mode="markers",
                    name="Utførte WEEX-ordre",
                    marker={
                        "symbol": "diamond",
                        "size": 12,
                        "color": fill_colors,
                        "line": {"color": "#ffffff", "width": 1.5},
                    },
                    text=fill_labels,
                    hovertemplate="<b>WEEX-fyll: %{text}</b><br>%{x}<br>Pris: %{y:,.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # Rad 2: Kontinuerlig posisjonsretning (+1 / -1)
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
        tickvals=[-1, 0, 1],
        ticktext=["SHORT (-1)", "FLAT (0)", "LONG (+1)"],
        range=[-1.35, 1.35],
        row=2,
        col=1,
        gridcolor="rgba(255,255,255,0.08)",
    )
    fig.update_xaxes(rangeslider_visible=False, gridcolor="rgba(255,255,255,0.06)")
    fig.update_layout(
        template="plotly_dark",
        height=680,
        margin={"l": 50, "r": 30, "t": 55, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        hovermode="x unified",
    )
    return fig


SIGNAL_TABLE_COLUMNS = ["Handel ved lukking (UTC)", "Signallys åpnet (UTC)", "Signal", "Årsak", "Pris (USDT)",
                        "ALMA-bølge", "-2.5 SD mål"]


def recent_signals_table(frame: pd.DataFrame, limit: int = 20, minutes: int | None = None) -> pd.DataFrame:
    """Siste signaler. Handelstiden er lysets lukking (signalet bekreftet), som i tabellen over utførte handler."""
    valid = frame[frame["valid"] & (frame["buy_signal"] | frame["sell_signal"])].copy()
    if valid.empty:
        return pd.DataFrame(columns=SIGNAL_TABLE_COLUMNS)
    if minutes:
        step = pd.Timedelta(minutes=int(minutes))
    else:
        diffs = pd.to_datetime(frame["datetime"], utc=True).diff().dropna()
        step = diffs.mode().iloc[0] if not diffs.empty else pd.Timedelta(minutes=5)
    tail = valid.tail(limit).iloc[::-1]
    rows = []
    for _, row in tail.iterrows():
        rows.append(
            {
                "Handel ved lukking (UTC)": (pd.Timestamp(row["datetime"]) + step).strftime("%Y-%m-%d %H:%M"),
                "Signallys åpnet (UTC)": pd.Timestamp(row["datetime"]).strftime("%Y-%m-%d %H:%M"),
                "Signal": "LONG (BUY)" if bool(row["buy_signal"]) else "SHORT (SELL)",
                "Årsak": str(row["signal_reason"]),
                "Pris (USDT)": round(float(row["close"]), 4),
                "ALMA-bølge": round(float(row["wave"]), 4) if pd.notna(row["wave"]) else None,
                "-2.5 SD mål": round(float(row["sd_target"]), 4) if pd.notna(row["sd_target"]) else None,
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
    """Hent siste 1000 avsluttede lys fra WEEX futures eller Binance spot."""
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
    coin: str = "BTC",
    timeframe_label: str = "5 minutter",
    source_label: str = "WEEX futures",
    mode_label: str = "BUY/SELL-merker + BOS/CHoCH-struktur + ALMA-bølge (anbefalt)",
    wave_len: int = 21,
    smc_sens: int = 7,
    sig_sens: int = 10,
    sd_len: int = 20,
) -> tuple[str, go.Figure | None, pd.DataFrame]:
    """Bygg sanntids Clean Wave Matrix PRO-visning for Gradio."""
    from weex_autotrade_dashboard import COINS

    symbol = COINS.get(coin, coin if str(coin).endswith("USDT") else "BTCUSDT")
    _, minutes = TIMEFRAMES.get(timeframe_label, ("5", 5))
    source = DATA_SOURCES.get(source_label, "weex")
    mode = next((k for k, v in CW_MODES.items() if v == mode_label or k == mode_label), "combined")

    try:
        bars = load_bars(symbol, source=source, minutes=minutes)
        frame = compute(
            bars,
            minutes=minutes,
            mode=mode,
            wave_len=int(wave_len),
            smc_sens=int(smc_sens),
            sig_sens=int(sig_sens),
            sd_len=int(sd_len),
        )
        md = dashboard(
            frame,
            symbol=symbol,
            mode=mode,
            source=source,
            minutes=minutes,
            wave_len=int(wave_len),
            smc_sens=int(smc_sens),
            sig_sens=int(sig_sens),
            sd_len=int(sd_len),
        )
        fig = figure(frame, symbol=symbol, visible_bars=180, minutes=minutes)
        table = recent_signals_table(frame, limit=20, minutes=minutes)
        return md, fig, table
    except Exception:
        return (
            "**Clean Wave Matrix PRO kan ikke vises nå:** lysene kunne ikke hentes. Prøv igjen om litt.",
            None,
            pd.DataFrame(columns=SIGNAL_TABLE_COLUMNS),
        )
