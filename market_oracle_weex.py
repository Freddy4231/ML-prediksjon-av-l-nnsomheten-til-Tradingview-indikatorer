"""Estimert Oracle-profil fra samme Pine-kilde på avsluttede WEEX-lys.

Ingen HTTP, konto eller ordre. Filter, støtteverktøy og ATR-risikoplan er
selvstendige tolkninger av offentlig funksjonalitet, ikke ChartPrimes kode.
Deler bare OHLCV-validering og avlesning av den eksplisitte plotkontrakten
med Kinetic-adapteren; Oracle har egen kilde, SHA og signalberegning.
"""
import hashlib
import time
from pathlib import Path

import numpy as np

import kinetic_scalper_weex as shared
from pine_engine import runner
from pine_engine.compiler import compile_source
from pine_engine.parser import PineError
from pine_engine.runtime import PineTimeout

ROOT = Path(__file__).resolve().parent
PINE_FILE = "tradingview/market_oracle_reconstruction.pine"
PINE_SHA256 = "89aed6fc7d3b93ee074f0224479381d23bf4655df55055962f231b05b4727a90"
LABEL = "Market Oracle Pro (estimert rekonstruksjon · SL/TP)"
MIN_BARS = 100
_COMPILED = {}


def evaluate(bars, minutes=5, symbol="BTCUSDT", mintick=0.01):
    """Fast Trend/14-profil, SL 2 ATR, halvpart ved 2 ATR, rest ved 3 ATR.

    Bare et nytt signal på siste avsluttede lys er fresh. Referanseplanen er
    ikke en bekreftelse på børsfyll. Manglende/endrede data avviser beregningen.
    """
    if not isinstance(symbol, str) or not symbol or not symbol.isalnum():
        raise ValueError("Oracle krever et enkelt WEEX-tickernavn.")
    if not np.isfinite(mintick) or mintick <= 0:
        raise ValueError("Ugyldig prissteg for Oracle.")
    try:
        frame = shared._prepare(bars, minutes)
    except ValueError as exc:
        raise ValueError(str(exc).replace("Kinetic", "Oracle")) from None
    empty = {"target": 0, "fresh": False, "signal_close": None, "plan_id": None,
             "valid": False, "entry": None, "sl": None, "tp1": None, "tp2": None,
             "plan_active": False, "frame": frame, "source_sha256": PINE_SHA256}
    source = (ROOT / PINE_FILE).read_text(encoding="utf-8")
    if hashlib.sha256(source.encode()).hexdigest() != PINE_SHA256:
        raise ValueError("Oracle-kilden er endret siden motorprofilen ble kontrollert.")
    if len(frame) < MIN_BARS:
        return {**empty, "reason": f"Oracle trenger minst {MIN_BARS} sammenhengende WEEX-lys."}
    try:
        if PINE_SHA256 not in _COMPILED:
            _COMPILED[PINE_SHA256] = compile_source(source)
        compiled = _COMPILED[PINE_SHA256]
        inputs, sources = runner.resolve_inputs(compiled)
        main = shared._to_bars(frame, minutes, symbol.upper(), float(mintick))
        result = runner.run_pass(compiled, main, inputs=inputs, sources=sources,
                                 deadline=time.monotonic() + 15, capture_plots=True)
        frame = shared._plot_frame(result, frame)
    except (PineError, PineTimeout) as exc:
        raise ValueError(f"Oracle kunne ikke kjøres: {exc}") from None
    empty['frame'] = frame
    signals = np.flatnonzero((frame.buy ^ frame.sell).to_numpy())
    if not len(signals):
        return {**empty, "reason": "Oracle venter på et nytt bekreftet signal."}
    k = int(signals[-1])
    row = frame.iloc[k]
    side = 1 if row.buy else -1
    entry, sl, tp1, tp2 = levels = row[['entry', 'sl', 'tp1', 'tp2']].to_numpy(float)
    valid = bool(np.isfinite(levels).all() and np.all(levels > 0)
                 and (sl < entry < tp1 < tp2 if side == 1 else tp2 < tp1 < entry < sl))
    stamp = int(main.TC[k])
    return {**empty, "valid": valid, "fresh": bool(valid and k == len(frame)-1),
            "target": side if valid else 0, "signal_close": stamp, "plan_id": stamp,
            "entry": float(entry) if valid else None, "sl": float(sl) if valid else None,
            "tp1": float(tp1) if valid else None, "tp2": float(tp2) if valid else None,
            "plan_active": bool(valid and frame.side.iloc[-1] == side),
            "reason": "Oracle · estimert fast Trend/14-profil på bekreftede WEEX-lys."}
