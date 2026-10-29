"""TMG Trades v2 [That Martini Guy / @tmgtrades & nioctib.live] — Indikator og WEEX-strategi.

Kvantitativ, strengt kausal (ikke-repaintende) v2-rekonstruksjon av metodikken fra YouTube-kanalen
«TMG Trades» (@tmgtrades / Jordan) og hans ordreflyt-/analyseplattform (nioctib.live & Stonksy):

1. **Multi-Timeframe Stonksy-ankring («Slow Context First»):**
   - Forankrer Stonksy-trendregimet (50/200 EMA), 4-farget momentumoscillator (12/26/9) og VRVP
     til en roligere 1,5–2-timers struktur på intradag-grafer (skalafaktor opptil ``4×``, maks 800 stolper
     slik at hele modellen kjører sømløst innenfor WEEX sine 1000 live-klines og TradingView Pine v6).
   - Kutter mikro-støy og overtrading dramatisk på 15m/30m/1t samtidig som store trendbølger fanges.

2. **Rullerende VRVP (Volume Profile: 70 % Value Area, POC / VAH / VAL) & «No Man's Land»-filter:**
   - **POC (Point of Control):** Prisnivået med høyest omsatt volum — fungerer som likviditetsmagnet
     og Take Profit-anker ved likviditetsfeller.
   - **VAH / VAL:** Øvre og nedre grense for 70 % av omsatt volum.
   - **«No Man's Land»-filter:** Unngår å jage nye trendhandler midt inne i trang konsolidering rundt POC;
     krever enten prisaksept utenfor Value Area (over ``VAH`` / under ``VAL``) eller sterk strukturell
     trendspredning (``|EMA50 - EMA200| / ATR >= nml_threshold``).

3. **Institusjonell rullerende VWAP + Nioctib CVD-absorpsjon, Divergens & Buy Pressure %:**
   - **Sesjons-VWAP:** Volumvektet snittpris som institusjonelt referansenivå (bull over VWAP, bear under).
   - **Nioctib CVD & Buy Pressure %:** Kumulativ volumdelta (fra ekte ``taker_buy_volume`` når tilgjengelig,
     ellers intrabar CLV-fordeling) måler om aggressive kjøpere/selgere har kontroll, samt oppdager
     **absorpsjonsdivergens** (lavere prisbunn mens CVD holder høyere bunn, eller omvendt).

4. **Ikke-avbrytende Liquidation Cascade Traps -> 88 % VRVP POC + Break-Even ved 50 %:**
   - Utløses kun når motoren er **flat** (avbryter aldri en pågående Stonksy-trendbølge!).
   - Krever likvidasjonskaskade-volumspike (``vol >= 1,5 × SMA20``), stop-hunt utenfor nylig svinglikviditet
     og utenfor ``VAL``/``VAH``, tydelig avvisningsveke (``>= 40 %``), RSI-ekstrem og CVD-/ordreflytabsorpsjon.
   - **Front-running POC TP (88 %):** Tar gevinst litt før selve ``POC`` («take profit a little bit too soon»).
   - **Break-Even Stop:** Flytter automatisk stop-loss opp til inngang (+ gebyrbuffer) så snart prisen når
     halvveis (50 %) mot POC-målet.

Kun avsluttede stolper brukes; ingen fremtidige data lekker inn i signalene.
Ikke finansiell rådgivning. Uavhengig kvantitativ modell basert på offentlig kanal- og verktøyinnhold.
"""
from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LABEL = "TMG Trades [Stonksy + VRVP Liquidity Trap]"
PINE_FILE = "tradingview/tmg_trades_strategy.pine"

TMG_MODES: dict[str, str] = {
    "combined": "TMG Hybrid v2: HTF Stonksy + VWAP/CVD + Cascade Trap (anbefalt)",
    "stonksy_trend": "Kun HTF Stonksy Trend, VWAP & CVD (best i sterke trendmarkeder)",
    "liquidity_trap": "Kun Liquidation Cascade Trap -> 88 % VRVP POC (best i range/chop)",
}

TIMEFRAMES: dict[str, tuple[str, int]] = {
    "1 minutt": ("1", 1),
    "5 minutter": ("5", 5),
    "15 minutter": ("15", 15),
    "30 minutter": ("30", 30),
    "1 time": ("60", 60),
    "4 timer": ("240", 240),
}

DATA_SOURCES: dict[str, str] = {
    "WEEX futures": "weex",
    "WEEX futures (direkte fra børsen)": "weex",
    "Binance spot": "binance",
    "Binance spot (TradingView-referanse)": "binance",
}


@dataclass(frozen=True)
class TmgSummary:
    symbol: str
    timeframe_label: str
    mode: str
    close: float
    position: int
    bg_regime: int
    osc_value: float
    osc_color: str
    buy_pressure_pct: float
    poc: float
    vah: float
    val: float
    ema50: float
    ema200: float
    stop_price: float | None
    tp_price: float | None
    action: str
    last_signal_time: str
    last_reason: str
    vwap: float = 0.0
    cvd_rising: bool = False
    htf_scale: int = 1
    setup_kind: str = ""


def _ema(arr: np.ndarray, length: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    alpha = 2.0 / (max(int(length), 1) + 1.0)
    prev = np.nan
    for i in range(n):
        x = arr[i]
        if not np.isfinite(x):
            out[i] = prev
            continue
        prev = x if not np.isfinite(prev) else alpha * x + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _rma(arr: np.ndarray, length: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    length = max(int(length), 1)
    alpha = 1.0 / length
    acc = 0.0
    count = 0
    prev = np.nan
    for i in range(n):
        x = arr[i]
        if not np.isfinite(x):
            out[i] = prev
            continue
        if count < length:
            acc += x
            count += 1
            if count == length:
                prev = acc / length
                out[i] = prev
        else:
            prev = alpha * x + (1.0 - alpha) * prev
            out[i] = prev
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    n = len(close)
    tr = np.empty(n, dtype=float)
    tr[0] = high[0] - low[0]
    if n > 1:
        prev_c = close[:-1]
        tr[1:] = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - prev_c), np.abs(low[1:] - prev_c)])
    return _rma(tr, length)


def _rsi(close: np.ndarray, length: int = 14) -> np.ndarray:
    n = len(close)
    diff = np.zeros(n, dtype=float)
    if n > 1:
        diff[1:] = np.diff(close)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    avg_g = _rma(gains, length)
    avg_l = _rma(losses, length)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_g / avg_l
        rsi = np.where(avg_l == 0, 100.0, 100.0 - 100.0 / (1.0 + rs))
    return rsi


def _rolling_vrvp(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int = 120,
    n_bins: int = 32,
    va_fraction: float = 0.70,
    vp_step: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rask, strengt kausal rullerende VRVP (POC, VAH, VAL) uten fremtidslekkasje."""
    n = len(close)
    poc = np.full(n, np.nan, dtype=float)
    vah = np.full(n, np.nan, dtype=float)
    val = np.full(n, np.nan, dtype=float)
    hlc3 = (high + low + close) / 3.0
    eff_vol = np.where(volume > 0, volume, 1.0)
    lb = max(int(lookback), 20)
    nb = max(int(n_bins), 10)
    step_bars = max(int(vp_step), 1)

    last_poc = last_vah = last_val = np.nan
    for i in range(lb - 1, n):
        if (i - (lb - 1)) % step_bars != 0 and np.isfinite(last_poc):
            poc[i] = last_poc
            vah[i] = last_vah
            val[i] = last_val
            continue
        s = i - lb + 1
        p_min = float(np.min(low[s : i + 1]))
        p_max = float(np.max(high[s : i + 1]))
        if not np.isfinite(p_min) or not np.isfinite(p_max) or p_max <= p_min:
            last_poc = last_vah = last_val = close[i]
            poc[i] = vah[i] = val[i] = close[i]
            continue
        span = p_max - p_min
        idx = np.clip(((hlc3[s : i + 1] - p_min) / span * nb).astype(int), 0, nb - 1)
        hist = np.bincount(idx, weights=eff_vol[s : i + 1], minlength=nb)
        poc_bin = int(np.argmax(hist))
        step = span / nb
        last_poc = p_min + (poc_bin + 0.5) * step

        target_vol = float(np.sum(hist)) * va_fraction
        cum_vol = float(hist[poc_bin])
        lo_b = hi_b = poc_bin
        while cum_vol < target_vol and (lo_b > 0 or hi_b < nb - 1):
            down_v = float(hist[lo_b - 1]) if lo_b > 0 else -1.0
            up_v = float(hist[hi_b + 1]) if hi_b < nb - 1 else -1.0
            if up_v >= down_v:
                hi_b += 1
                cum_vol += up_v
            else:
                lo_b -= 1
                cum_vol += down_v
        last_val = p_min + lo_b * step
        last_vah = p_min + (hi_b + 1) * step
        poc[i] = last_poc
        vah[i] = last_vah
        val[i] = last_val

    return poc, vah, val


def _htf_scale(minutes: int, htf_anchor_minutes: int = 100) -> int:
    """Multi-timeframe skalafaktor (1..4) uten tvetydig 0,5-avrunding mellom Python og Pine v6."""
    if htf_anchor_minutes <= 0:
        return 1
    m = max(int(minutes), 1)
    return int(np.clip(round(float(htf_anchor_minutes) / float(m)), 1, 4))


def prepare(bars: pd.DataFrame, minutes: int = 15, gap_bars: int = 200) -> tuple[pd.DataFrame, bool]:
    """Rens og normaliser OHLCV-data med millisekund-presisjon."""
    if bars is None or bars.empty:
        raise ValueError("Ingen prisstolper tilgjengelig.")
    df = bars.copy()
    if "timestamp" not in df.columns:
        if "open_time" in df.columns:
            ts_num = pd.to_numeric(df["open_time"], errors="coerce")
            if ts_num.notna().all() and not pd.api.types.is_datetime64_any_dtype(df["open_time"]):
                arr = ts_num.to_numpy(dtype=np.int64)
                df["timestamp"] = np.where(arr < 10**11, arr * 1000, arr).astype(np.int64)
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            else:
                dt_ser = pd.to_datetime(df["open_time"], utc=True)
                df["datetime"] = dt_ser
                df["timestamp"] = dt_ser.dt.as_unit("ms").asi8.astype(np.int64)
        elif isinstance(df.index, pd.DatetimeIndex):
            idx_utc = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            df["timestamp"] = idx_utc.as_unit("ms").asi8.astype(np.int64)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            raise ValueError("Prisstolpene mangler tidsstempel.")
    else:
        ts_num = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts_num.notna().all() and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            arr = ts_num.to_numpy(dtype=np.int64)
            df["timestamp"] = np.where(arr < 10**11, arr * 1000, arr).astype(np.int64)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            dt_ser = pd.to_datetime(df["timestamp"], utc=True)
            df["datetime"] = dt_ser
            df["timestamp"] = dt_ser.dt.as_unit("ms").asi8.astype(np.int64)

    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0) if "volume" in df.columns else 0.0

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    if df.empty:
        raise ValueError("Ingen gyldige OHLC-stolper etter rensing.")

    df["timestamp"] = df["timestamp"].astype(np.int64)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    step_ms = max(int(minutes), 1) * 60_000
    diffs = np.diff(df["timestamp"].to_numpy(dtype=np.int64))
    max_gap_ms = step_ms * max(int(gap_bars), 2)
    recent_tail = diffs[-min(len(diffs), 60) :] if len(diffs) > 0 else np.array([], dtype=np.int64)
    recent_gap = bool(np.any(recent_tail > max_gap_ms))

    df["valid"] = True
    if len(diffs) > 0 and np.any(diffs > max_gap_ms):
        last_gap_pos = int(np.flatnonzero(diffs > max_gap_ms)[-1]) + 1
        df.loc[: last_gap_pos - 1, "valid"] = False
    return df, recent_gap


def compute(
    bars: pd.DataFrame,
    minutes: int = 15,
    mode: str = "combined",
    vp_lookback: int = 100,
    vp_rows: int = 32,
    liq_lookback: int = 72,
    bg_threshold: float = 0.50,
    max_sl_pct: float = 0.025,
    use_macro_filter: bool = True,
    htf_anchor_minutes: int = 100,
    use_vwap_filter: bool = True,
    use_cvd_filter: bool = True,
    nml_threshold: float = 0.85,
    poc_tp_ratio: float = 0.88,
) -> pd.DataFrame:
    """Beregn alle TMG Trades v2-komponenter (HTF Stonksy, VRVP, VWAP, Nioctib CVD og Cascade Traps)."""
    if mode not in TMG_MODES:
        mode = "combined"
    df, _ = prepare(bars, minutes=minutes)
    n = len(df)
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    valid = df["valid"].to_numpy(dtype=bool)

    scale = _htf_scale(minutes, htf_anchor_minutes)

    atr_bar = _atr(h, l, c, 14)
    atr_htf = _atr(h, l, c, 14 * scale)
    ema21 = _ema(c, 21 * scale)
    ema50 = _ema(c, 50 * scale)
    ema55 = _ema(c, 55 * scale)
    ema200 = _ema(c, 200 * scale)
    rsi14 = _rsi(c, 14)

    # Institusjonell rullerende sesjons-VWAP (24t kappet til 16..96 stolper)
    hlc3 = (h + l + c) / 3.0
    eff_v = np.where(v > 0, v, 1.0)
    vwap_len = int(np.clip(round(1440.0 / max(int(minutes), 1)), 16, 96))
    vwap = (
        pd.Series(hlc3 * eff_v).rolling(vwap_len, min_periods=1).sum().to_numpy()
        / np.maximum(pd.Series(eff_v).rolling(vwap_len, min_periods=1).sum().to_numpy(), 1e-12)
    )

    # 1. HTF-ankret Stonksy Oscillator & 4-farget momentumtilstand
    hlcc4 = (h + l + 2.0 * c) / 4.0
    fast_m = _ema(hlcc4, 12 * scale)
    slow_m = _ema(hlcc4, 26 * scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_mom = np.where(atr_htf > 0, (fast_m - slow_m) / atr_htf, 0.0)
    osc = _ema(raw_mom, 9 * scale) * 100.0
    osc_prev = np.r_[np.nan, osc[:-1]]
    osc_state = np.zeros(n, dtype=int)
    osc_state = np.where((osc > 0) & (osc > osc_prev), 2, osc_state)
    osc_state = np.where((osc > 0) & (osc <= osc_prev), 1, osc_state)
    osc_state = np.where((osc < 0) & (osc < osc_prev), -2, osc_state)
    osc_state = np.where((osc < 0) & (osc >= osc_prev), -1, osc_state)

    # 2. Stonksy Background Regime (skalert 50 EMA mot 200 EMA normalisert med HTF ATR + hysterese)
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = np.where(atr_htf > 0, (ema50 - ema200) / atr_htf, 0.0)
    bg = np.zeros(n, dtype=int)
    cur_bg = 0
    for i in range(n):
        sp = spread[i]
        if not np.isfinite(sp) or not np.isfinite(ema50[i]) or not np.isfinite(ema200[i]):
            bg[i] = 0
            cur_bg = 0
            continue
        if cur_bg == 1:
            if sp < -bg_threshold and c[i] < ema200[i]:
                cur_bg = -1
            elif sp <= 0.0:
                cur_bg = 0
        elif cur_bg == -1:
            if sp > bg_threshold and c[i] > ema200[i]:
                cur_bg = 1
            elif sp >= 0.0:
                cur_bg = 0
        else:
            if sp > bg_threshold and c[i] > ema50[i]:
                cur_bg = 1
            elif sp < -bg_threshold and c[i] < ema50[i]:
                cur_bg = -1
        bg[i] = cur_bg

    # 3. Nioctib Ordreflyt: Buy Pressure %, CVD og Absorpsjonsdivergens
    if "taker_buy_volume" in df.columns and df["taker_buy_volume"].notna().all():
        v_buy = pd.to_numeric(df["taker_buy_volume"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        rng = np.maximum(h - l, 1e-12)
        w_buy = np.clip((c - l) / rng, 0.0, 1.0)
        v_buy = v * w_buy
    eff_vb = np.where(v > 0, v_buy, 0.5 * eff_v)
    flow_len = 12 * scale
    roll_vb = pd.Series(eff_vb).rolling(flow_len, min_periods=1).sum().to_numpy()
    roll_v = pd.Series(eff_v).rolling(flow_len, min_periods=1).sum().to_numpy()
    buy_pct = 100.0 * roll_vb / np.maximum(roll_v, 1e-12)

    bar_delta = 2.0 * eff_vb - eff_v
    cvd = np.cumsum(bar_delta)
    cvd_look = 6 * scale
    cvd_slope = cvd - pd.Series(cvd).shift(cvd_look).to_numpy()
    vol_norm_win = pd.Series(eff_v).rolling(cvd_look, min_periods=1).sum().to_numpy()
    cvd_norm = np.where(np.isfinite(cvd_slope), 50.0 * cvd_slope / np.maximum(vol_norm_win, 1e-12), 0.0)

    vol_ma20 = pd.Series(eff_v).rolling(20, min_periods=5).mean().to_numpy()
    vol_spike = eff_v >= 1.5 * vol_ma20

    # 4. Rullerende VRVP (70 % Value Area: POC, VAH, VAL — maks 200 stolper for rask og kausal oppvarming)
    eff_vp_lb = min(max(int(vp_lookback) * scale, 40), 200)
    eff_vp_step = max(4, 2 * scale)
    poc, vah, val = _rolling_vrvp(h, l, c, v, lookback=eff_vp_lb, n_bins=vp_rows, va_fraction=0.70, vp_step=eff_vp_step)

    # 5. Liquidation Cascade Sweeps (Stop-Hunt med volumkaskade + CVD-absorpsjon -> 88 % VRVP POC)
    eff_liq_lb = min(max(int(liq_lookback) * scale, 20), 144)
    liq_high = pd.Series(h).rolling(eff_liq_lb).max().shift(1).to_numpy()
    liq_low = pd.Series(l).rolling(eff_liq_lb).min().shift(1).to_numpy()

    candle_rng = np.maximum(h - l, 1e-12)
    lower_wick_ratio = (np.minimum(o, c) - l) / candle_rng
    upper_wick_ratio = (h - np.maximum(o, c)) / candle_rng

    div_lb = 10 * scale
    bull_div = (l < pd.Series(l).shift(div_lb).to_numpy()) & (cvd > pd.Series(cvd).shift(div_lb).to_numpy())
    bear_div = (h > pd.Series(h).shift(div_lb).to_numpy()) & (cvd < pd.Series(cvd).shift(div_lb).to_numpy())

    bull_sweep = (
        np.isfinite(liq_low)
        & (l < liq_low)
        & (c > liq_low)
        & (lower_wick_ratio >= 0.40)
        & ((bar_delta > 0) | bull_div)
        & vol_spike
        & (rsi14 < 40.0)
    )
    bear_sweep = (
        np.isfinite(liq_high)
        & (h > liq_high)
        & (c < liq_high)
        & (upper_wick_ratio >= 0.40)
        & ((bar_delta < 0) | bear_div)
        & vol_spike
        & (rsi14 > 60.0)
    )

    # 6. Tilstandsmaskin for TMG Trades v2 posisjon, break-even SL og POC TP (uten fremtidslekkasje)
    pos = np.zeros(n, dtype=int)
    buy_sig = np.zeros(n, dtype=bool)
    sell_sig = np.zeros(n, dtype=bool)
    exit_sig = np.zeros(n, dtype=bool)
    sl_arr = np.full(n, np.nan, dtype=float)
    tp_arr = np.full(n, np.nan, dtype=float)
    reason_arr = [""] * n
    kind_arr = [""] * n
    sig_epoch = np.zeros(n, dtype=np.int64)

    cur_pos = 0
    cur_kind = ""  # "trend" eller "trap"
    entry_px = np.nan
    cur_sl = np.nan
    cur_tp = np.nan
    last_epoch = 0
    last_reason = ""
    traded_bg_dir = 0

    warmup = max(eff_vp_lb, 200 if use_macro_filter else 75)
    tp_frac = float(np.clip(poc_tp_ratio, 0.50, 1.0))

    for i in range(n):
        if i < warmup or not valid[i] or not np.isfinite(atr_htf[i]) or not np.isfinite(poc[i]):
            pos[i] = 0
            cur_pos = 0
            entry_px = cur_sl = cur_tp = np.nan
            continue

        if bg[i] == 0:
            traded_bg_dir = 0
        elif traded_bg_dir != 0 and bg[i] != traded_bg_dir:
            traded_bg_dir = 0

        # A. Sjekk utgang / break-even-justering på eksisterende posisjon
        if cur_pos == 1:
            if cur_kind == "trap":
                if np.isfinite(entry_px) and np.isfinite(cur_tp) and h[i] >= entry_px + 0.5 * (cur_tp - entry_px):
                    cur_sl = max(cur_sl, entry_px * 1.0015)
                hit_sl = np.isfinite(cur_sl) and c[i] <= cur_sl
                hit_tp = np.isfinite(cur_tp) and h[i] >= cur_tp
                if hit_sl or hit_tp:
                    cur_pos = 0
                    cur_kind = ""
                    is_be = hit_sl and np.isfinite(entry_px) and cur_sl >= entry_px
                    entry_px = cur_sl = cur_tp = np.nan
                    exit_sig[i] = True
                    last_reason = "88 % VRVP POC nådd (TP)" if hit_tp else ("Break-Even stopp" if is_be else "Trap Stop Loss truffet")
            else:
                if bg[i] <= 0:
                    cur_pos = 0
                    cur_kind = ""
                    entry_px = cur_sl = cur_tp = np.nan
                    exit_sig[i] = True
                    last_reason = "Stonksy Hvit konsolidering (exit)"
        elif cur_pos == -1:
            if cur_kind == "trap":
                if np.isfinite(entry_px) and np.isfinite(cur_tp) and l[i] <= entry_px - 0.5 * (entry_px - cur_tp):
                    cur_sl = min(cur_sl, entry_px * 0.9985)
                hit_sl = np.isfinite(cur_sl) and c[i] >= cur_sl
                hit_tp = np.isfinite(cur_tp) and l[i] <= cur_tp
                if hit_sl or hit_tp:
                    cur_pos = 0
                    cur_kind = ""
                    is_be = hit_sl and np.isfinite(entry_px) and cur_sl <= entry_px
                    entry_px = cur_sl = cur_tp = np.nan
                    exit_sig[i] = True
                    last_reason = "88 % VRVP POC nådd (TP)" if hit_tp else ("Break-Even stopp" if is_be else "Trap Stop Loss truffet")
            else:
                if bg[i] >= 0:
                    cur_pos = 0
                    cur_kind = ""
                    entry_px = cur_sl = cur_tp = np.nan
                    exit_sig[i] = True
                    last_reason = "Stonksy Hvit konsolidering (exit)"

        # B. Sjekk nye inngangssignaler ved close[i]
        macro_bull = (not use_macro_filter) or (c[i] > ema200[i])
        macro_bear = (not use_macro_filter) or (c[i] < ema200[i])
        vwap_bull = (not use_vwap_filter) or (c[i] >= vwap[i])
        vwap_bear = (not use_vwap_filter) or (c[i] <= vwap[i])
        flow_bull = (not use_cvd_filter) or (buy_pct[i] >= 50.0) or (np.isfinite(cvd_slope[i]) and cvd_slope[i] > 0)
        flow_bear = (not use_cvd_filter) or (buy_pct[i] <= 50.0) or (np.isfinite(cvd_slope[i]) and cvd_slope[i] < 0)

        if nml_threshold > 0:
            nml_bull = (c[i] >= vah[i]) or (spread[i] >= nml_threshold)
            nml_bear = (c[i] <= val[i]) or (spread[i] <= -nml_threshold)
        else:
            nml_bull = c[i] >= poc[i]
            nml_bear = c[i] <= poc[i]

        trend_long = (
            mode in ("combined", "stonksy_trend")
            and bg[i] == 1
            and traded_bg_dir != 1
            and macro_bull
            and vwap_bull
            and flow_bull
            and osc_state[i] == 2
            and c[i] > ema50[i]
            and c[i] >= poc[i]
            and nml_bull
        )
        trend_short = (
            mode in ("combined", "stonksy_trend")
            and bg[i] == -1
            and traded_bg_dir != -1
            and macro_bear
            and vwap_bear
            and flow_bear
            and osc_state[i] == -2
            and c[i] < ema50[i]
            and c[i] <= poc[i]
            and nml_bear
        )

        # Kritisk v2-regel: Likviditetsfeller utløses KUN når motoren er flat (cur_pos == 0),
        # slik at et mot-trend veke-sveip aldri kutter en aktiv Stonksy-trendbølge!
        trap_long = (
            mode in ("combined", "liquidity_trap")
            and cur_pos == 0
            and bull_sweep[i]
            and c[i] < val[i]
            and np.isfinite(atr_bar[i])
            and (poc[i] - c[i]) > 1.3 * atr_bar[i]
        )
        trap_short = (
            mode in ("combined", "liquidity_trap")
            and cur_pos == 0
            and bear_sweep[i]
            and c[i] > vah[i]
            and np.isfinite(atr_bar[i])
            and (c[i] - poc[i]) > 1.3 * atr_bar[i]
        )

        if trend_long and cur_pos != 1:
            cur_pos = 1
            cur_kind = "trend"
            traded_bg_dir = 1
            entry_px = c[i]
            buy_sig[i] = True
            last_epoch = ts[i]
            cur_sl = max(ema200[i] - atr_htf[i], c[i] * (1.0 - max(max_sl_pct, 0.06)))
            cur_tp = np.nan
            last_reason = "HTF Stonksy Grønn + VWAP/CVD + VAH-utbrudd (LONG)"
        elif trend_short and cur_pos != -1:
            cur_pos = -1
            cur_kind = "trend"
            traded_bg_dir = -1
            entry_px = c[i]
            sell_sig[i] = True
            last_epoch = ts[i]
            cur_sl = min(ema200[i] + atr_htf[i], c[i] * (1.0 + max(max_sl_pct, 0.06)))
            cur_tp = np.nan
            last_reason = "HTF Stonksy Rød + VWAP/CVD + VAL-brudd (SHORT)"
        elif trap_long and cur_pos == 0:
            cur_pos = 1
            cur_kind = "trap"
            entry_px = c[i]
            buy_sig[i] = True
            last_epoch = ts[i]
            cur_sl = max(l[i] - 0.5 * atr_bar[i], c[i] * (1.0 - max_sl_pct))
            cur_tp = c[i] + tp_frac * (poc[i] - c[i])
            last_reason = "Cascade Liquidity Trap under VAL -> 88 % VRVP POC (LONG)"
        elif trap_short and cur_pos == 0:
            cur_pos = -1
            cur_kind = "trap"
            entry_px = c[i]
            sell_sig[i] = True
            last_epoch = ts[i]
            cur_sl = min(h[i] + 0.5 * atr_bar[i], c[i] * (1.0 + max_sl_pct))
            cur_tp = c[i] - tp_frac * (c[i] - poc[i])
            last_reason = "Cascade Liquidity Trap over VAH -> 88 % VRVP POC (SHORT)"

        pos[i] = cur_pos
        sl_arr[i] = cur_sl
        tp_arr[i] = cur_tp
        sig_epoch[i] = last_epoch
        reason_arr[i] = last_reason
        kind_arr[i] = cur_kind

    df["atr"] = atr_bar
    df["atr_htf"] = atr_htf
    df["ema21"] = ema21
    df["ema50"] = ema50
    df["ema55"] = ema55
    df["ema200"] = ema200
    df["vwap"] = vwap
    df["rsi14"] = rsi14
    df["osc"] = osc
    df["osc_state"] = osc_state
    df["bg_regime"] = bg
    df["buy_pct"] = buy_pct
    df["cvd"] = cvd
    df["cvd_norm"] = cvd_norm
    df["bull_div"] = bull_div
    df["bear_div"] = bear_div
    df["vol_spike"] = vol_spike
    df["poc"] = poc
    df["vah"] = vah
    df["val"] = val
    df["bull_sweep"] = bull_sweep
    df["bear_sweep"] = bear_sweep
    df["position"] = pos
    df["buy_signal"] = buy_sig
    df["sell_signal"] = sell_sig
    df["exit_signal"] = exit_sig
    df["sl"] = sl_arr
    df["tp"] = tp_arr
    df["signal_epoch"] = sig_epoch
    df["signal_reason"] = reason_arr
    df["setup_kind"] = kind_arr
    df["htf_scale"] = scale
    return df


def summarize(
    frame: pd.DataFrame,
    symbol: str = "BTCUSDT",
    timeframe_label: str = "15 minutter",
    mode: str = "combined",
) -> TmgSummary:
    last = frame.iloc[-1]
    pos_val = int(last["position"])
    osc_st = int(last["osc_state"])
    osc_color = {
        2: "Mørkegrønn (styrkende bull)",
        1: "Lysegrønn (avtagende bull)",
        -1: "Lyserød (avtagende bear)",
        -2: "Mørkerød (styrkende bear)",
    }.get(osc_st, "Nøytral")
    if bool(last["buy_signal"]):
        action = "NY LONG (TMG BUY)"
    elif bool(last["sell_signal"]):
        action = "NY SHORT (TMG SELL)"
    elif pos_val == 1:
        action = "HOLD LONG"
    elif pos_val == -1:
        action = "HOLD SHORT"
    else:
        action = "FLAT / VENTER PÅ OPPSETT"

    epoch = int(last["signal_epoch"])
    sig_time = pd.Timestamp(epoch, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M UTC") if epoch > 0 else "—"
    vwap_val = float(last["vwap"]) if "vwap" in last and pd.notna(last["vwap"]) else float(last["close"])
    cvd_up = bool(float(last["cvd_norm"]) > 0.0) if "cvd_norm" in last and pd.notna(last["cvd_norm"]) else False
    scale_val = int(last["htf_scale"]) if "htf_scale" in last and pd.notna(last["htf_scale"]) else 1
    kind_val = str(last["setup_kind"] or "") if "setup_kind" in last else ""
    return TmgSummary(
        symbol=symbol,
        timeframe_label=timeframe_label,
        mode=mode,
        close=float(last["close"]),
        position=pos_val,
        bg_regime=int(last["bg_regime"]),
        osc_value=float(last["osc"]) if pd.notna(last["osc"]) else 0.0,
        osc_color=osc_color,
        buy_pressure_pct=float(last["buy_pct"]) if pd.notna(last["buy_pct"]) else 50.0,
        poc=float(last["poc"]) if pd.notna(last["poc"]) else float(last["close"]),
        vah=float(last["vah"]) if pd.notna(last["vah"]) else float(last["close"]),
        val=float(last["val"]) if pd.notna(last["val"]) else float(last["close"]),
        ema50=float(last["ema50"]) if pd.notna(last["ema50"]) else float(last["close"]),
        ema200=float(last["ema200"]) if pd.notna(last["ema200"]) else float(last["close"]),
        stop_price=float(last["sl"]) if pd.notna(last["sl"]) else None,
        tp_price=float(last["tp"]) if pd.notna(last["tp"]) else None,
        action=action,
        last_signal_time=sig_time,
        last_reason=str(last["signal_reason"] or "—"),
        vwap=vwap_val,
        cvd_rising=cvd_up,
        htf_scale=scale_val,
        setup_kind=kind_val,
    )


def figure(
    frame: pd.DataFrame,
    symbol: str = "BTCUSDT",
    fills: list | tuple = (),
    visible_bars: int = 180,
    minutes: int = 15,
) -> go.Figure:
    sub = frame.tail(max(int(visible_bars), 40)).copy()
    dt = pd.to_datetime(sub["datetime"], utc=True)
    scale_val = int(sub["htf_scale"].iloc[-1]) if "htf_scale" in sub.columns and len(sub) else 1

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.70, 0.30],
        subplot_titles=(
            f"{symbol} ({minutes}m, {scale_val}× HTF) — TMG Trades v2: Stonksy + VRVP (POC/VAH/VAL) + VWAP + Cascade Traps",
            "HTF Stonksy 4-farget Oscillator, Nioctib Buy Pressure % & CVD-impuls",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=dt,
            open=sub["open"],
            high=sub["high"],
            low=sub["low"],
            close=sub["close"],
            name="Pris",
            increasing_line_color="#00c853",
            decreasing_line_color="#ff1744",
        ),
        row=1,
        col=1,
    )

    # VRVP POC, VAH, VAL, Institusjonell VWAP og HTF 50/200 EMA
    fig.add_trace(
        go.Scatter(x=dt, y=sub["poc"], mode="lines", name="VRVP POC (Likviditetsmagnet)",
                   line={"color": "#ffab00", "width": 2.2}),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dt, y=sub["vah"], mode="lines", name="VRVP VAH (70 % topp)",
                   line={"color": "#29b6f6", "width": 1.2, "dash": "dot"}),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dt, y=sub["val"], mode="lines", name="VRVP VAL (70 % bunn)",
                   line={"color": "#29b6f6", "width": 1.2, "dash": "dot"}),
        row=1, col=1,
    )
    if "vwap" in sub.columns:
        fig.add_trace(
            go.Scatter(x=dt, y=sub["vwap"], mode="lines", name="Institusjonell VWAP",
                       line={"color": "#e0f7fa", "width": 1.5, "dash": "dash"}),
            row=1, col=1,
        )
    fig.add_trace(
        go.Scatter(x=dt, y=sub["ema50"], mode="lines", name="HTF 50 EMA (TMG Grønn linje)",
                   line={"color": "#00e676", "width": 1.8}),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dt, y=sub["ema200"], mode="lines", name="HTF 200 EMA (Makro trend)",
                   line={"color": "#ab47bc", "width": 1.8}),
        row=1, col=1,
    )

    buys = sub[sub["buy_signal"]]
    if not buys.empty:
        buy_labels = [
            "TRAP LONG" if str(k) == "trap" else "TMG BUY"
            for k in (buys["setup_kind"] if "setup_kind" in buys.columns else [""] * len(buys))
        ]
        fig.add_trace(
            go.Scatter(
                x=buys["datetime"],
                y=buys["low"] * 0.997,
                mode="markers+text",
                text=buy_labels,
                textposition="bottom center",
                name="TMG LONG-signal",
                marker={"symbol": "triangle-up", "size": 13, "color": "#00e676", "line": {"width": 1, "color": "#fff"}},
            ),
            row=1,
            col=1,
        )

    sells = sub[sub["sell_signal"]]
    if not sells.empty:
        sell_labels = [
            "TRAP SHORT" if str(k) == "trap" else "TMG SELL"
            for k in (sells["setup_kind"] if "setup_kind" in sells.columns else [""] * len(sells))
        ]
        fig.add_trace(
            go.Scatter(
                x=sells["datetime"],
                y=sells["high"] * 1.003,
                mode="markers+text",
                text=sell_labels,
                textposition="top center",
                name="TMG SHORT-signal",
                marker={"symbol": "triangle-down", "size": 13, "color": "#ff1744", "line": {"width": 1, "color": "#fff"}},
            ),
            row=1,
            col=1,
        )

    if fills:
        t_min = int(sub["timestamp"].iloc[0])
        recent = [f for f in fills if f.get("time") and int(f["time"]) >= t_min]
        if recent:
            f_df = pd.DataFrame(recent)
            f_dt = pd.to_datetime(f_df["time"], unit="ms", utc=True)
            fig.add_trace(
                go.Scatter(
                    x=f_dt,
                    y=f_df["price"],
                    mode="markers",
                    name="Faktiske WEEX-fyll",
                    marker={"symbol": "star", "size": 14, "color": "#ffd54f", "line": {"width": 1.5, "color": "#111"}},
                ),
                row=1,
                col=1,
            )

    # Panel 2: Stonksy 4-farget Oscillator + Nioctib Buy Pressure % + CVD-impuls
    color_map = {2: "#00c853", 1: "#b9f6ca", -1: "#ff8a80", -2: "#d50000"}
    bar_colors = [color_map.get(int(s), "#9e9e9e") for s in sub["osc_state"]]
    fig.add_trace(
        go.Bar(x=dt, y=sub["osc"], name="Stonksy Oscillator", marker_color=bar_colors),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dt,
            y=sub["buy_pct"] - 50.0,
            mode="lines",
            name="Ordreflyt Buy% − 50%",
            line={"color": "#ffd54f", "width": 1.5},
        ),
        row=2,
        col=1,
    )
    if "cvd_norm" in sub.columns:
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=sub["cvd_norm"],
                mode="lines",
                name="Nioctib CVD-impuls",
                line={"color": "#00e5ff", "width": 1.3, "dash": "dot"},
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        height=660,
        margin={"l": 45, "r": 25, "t": 50, "b": 35},
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return fig


def recent_signals_table(frame: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    sigs = frame[frame["buy_signal"] | frame["sell_signal"]].tail(max(int(limit), 1)).iloc[::-1]
    rows = []
    for _, row in sigs.iterrows():
        is_buy = bool(row["buy_signal"])
        rows.append(
            {
                "Tidspunkt (UTC)": pd.Timestamp(row["datetime"]).strftime("%Y-%m-%d %H:%M"),
                "Signal": "TMG BUY (LONG)" if is_buy else "TMG SELL (SHORT)",
                "Oppsett / Årsak": str(row["signal_reason"]),
                "Pris (USDT)": round(float(row["close"]), 4),
                "VWAP": round(float(row["vwap"]), 4) if "vwap" in row and pd.notna(row["vwap"]) else None,
                "VRVP POC": round(float(row["poc"]), 4) if pd.notna(row["poc"]) else None,
                "Stop Loss": round(float(row["sl"]), 4) if pd.notna(row["sl"]) else None,
                "Take Profit": round(float(row["tp"]), 4) if pd.notna(row["tp"]) else None,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["Tidspunkt (UTC)", "Signal", "Oppsett / Årsak", "Pris (USDT)", "VWAP", "VRVP POC", "Stop Loss", "Take Profit"]
        )
    return pd.DataFrame(rows)


def load_bars(
    symbol: str,
    source: str = "weex",
    minutes: int = 15,
    timeframe: str | None = None,
    limit: int = 1000,
    **_kwargs,
) -> pd.DataFrame:
    import triple_supertrend as ts
    from weex_autotrade_dashboard import market_bars

    tf_str = str(timeframe) if timeframe is not None else str(int(minutes))
    if source == "binance":
        bars = ts.fetch_binance(symbol, minutes=int(minutes))
        return bars.tail(int(limit)) if limit and len(bars) > int(limit) else bars
    try:
        bars = market_bars(os.environ.get("WEEX_ENV", "simulation"), tf_str, symbol)
        return bars.tail(int(limit)) if limit and len(bars) > int(limit) else bars
    except Exception:
        bars = ts.fetch_binance(symbol, minutes=int(minutes))
        return bars.tail(int(limit)) if limit and len(bars) > int(limit) else bars


def view(
    coin: str = "BTC",
    timeframe_label: str = "15 minutter",
    source_label: str = "WEEX futures",
    mode_label: str = "TMG Hybrid v2: HTF Stonksy + VWAP/CVD + Cascade Trap (anbefalt)",
) -> tuple[str, go.Figure | None, pd.DataFrame]:
    from weex_autotrade_dashboard import COINS

    symbol = COINS.get(coin, coin if str(coin).endswith("USDT") else "BTCUSDT")
    tf_code, minutes = TIMEFRAMES.get(timeframe_label, ("15", 15))
    source = DATA_SOURCES.get(source_label, "weex")
    mode = next((k for k, v in TMG_MODES.items() if v == mode_label or k == mode_label), "combined")
    bars = load_bars(symbol, source=source, timeframe=tf_code, minutes=minutes)
    frame = compute(bars, minutes=minutes, mode=mode)
    s = summarize(frame, symbol=symbol, timeframe_label=timeframe_label, mode=mode)
    bg_txt = {1: "🟢 Grønn (Bull)", -1: "🔴 Rød (Bear)", 0: "⚪ Hvit (Range/Nøytral)"}[s.bg_regime]
    pos_txt = {1: "🟢 **LONG (+1)**", -1: "🔴 **SHORT (-1)**", 0: "⚪ **FLAT (0)**"}[s.position]
    cvd_txt = "stigende" if s.cvd_rising else "avtagende"
    md = (
        f"**{symbol} ({timeframe_label}, {s.htf_scale}× HTF)** · **{LABEL} v2** · Posisjon: {pos_txt} · "
        f"Stonksy-bakgrunn: {bg_txt} · Oscillator: `{s.osc_value:+.1f}` ({s.osc_color}) · "
        f"VWAP: `{s.vwap:,.4f}` · Buy Pressure: `{s.buy_pressure_pct:.1f} %` (CVD {cvd_txt}) · "
        f"VRVP POC: `{s.poc:,.4f}` (VAL `{s.val:,.4f}` – VAH `{s.vah:,.4f}`)"
    )
    return md, figure(frame, symbol=symbol, minutes=minutes), recent_signals_table(frame)
