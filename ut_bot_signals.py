"""Kausal UT Bot-port med samme ATR-stopp og kryss som brukerens Pine v4.

Input må inneholde bare avsluttede barer. Modulen henter ingen data og sender
ingen ordre. ATR bruker ekte OHLC også når Heikin Ashi er valgt som signalkilde.
Første ATR er SMA av ``atr_period`` true-range-verdier, deretter Wilder-RMA.
Stoppens historikk starter på første innlastede bar; samme sluttutsnitt alene
kan derfor gi andre signaler enn hele historikken.
"""

from numbers import Integral, Real

import numpy as np
import pandas as pd


OHLC_COLUMNS = ("open", "high", "low", "close")
SIGNAL_COLUMNS = ("src", "atr", "stop", "buy", "sell", "side")


def _validated_ohlc(ohlcv):
    """Krev ekte priser og regulære tidspunkter; fyll aldri et datahull."""
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("OHLC-data må være en DataFrame.")
    missing = set(OHLC_COLUMNS).difference(ohlcv.columns)
    if missing:
        raise ValueError("OHLC-data mangler kolonner: " + ", ".join(sorted(missing)))
    if "open_time" in ohlcv.columns:
        raw_index = ohlcv["open_time"]
    elif isinstance(ohlcv.index, pd.DatetimeIndex):
        raw_index = ohlcv.index
    else:
        raise ValueError("OHLC-data må ha DatetimeIndex eller open_time-kolonne.")
    if pd.api.types.is_numeric_dtype(raw_index):
        raise ValueError("open_time må være dato/tid, ikke tall uten tidsenhet.")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(raw_index, utc=True, errors="raise"), name="open_time")
    except (TypeError, ValueError) as exc:
        raise ValueError("open_time inneholder ugyldig dato/tid.") from exc
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("open_time må være unik, stigende og uten manglende tidspunkt.")
    if len(index) > 2:
        # Timedelta-differanser beholder nanosekund/mikrosekund-enheten i pandas.
        differences = index[1:] - index[:-1]
        if not np.all(differences == differences[0]):
            raise ValueError("OHLC-data har tidsluker eller uregelmessig intervall; hull fylles ikke.")
    try:
        prices = ohlcv.loc[:, OHLC_COLUMNS].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("OHLC-priser må være tall.") from exc
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("OHLC-priser må være positive og endelige; hull fylles ikke.")
    open_, high, low, close = prices.T
    if ((high < low) | (high < open_) | (high < close) | (low > open_) | (low > close)).any():
        raise ValueError("OHLC-prisene ligger utenfor barens high/low.")
    return index, open_, high, low, close


def calculate_ut_bot(ohlcv, key_value=1.0, atr_period=10, heikin_ashi=False):
    """Beregn UT Bot-signaler på avsluttede barer uten å endre input.

    ``ohlcv`` har en UTC-tolket ``open_time``-kolonne eller DatetimeIndex og
    positive OHLC-priser. Naive tidspunkt tolkes som UTC. Alle intervaller må
    være like; manglende barer gir en feil i stedet for syntetiske priser.
    ``heikin_ashi`` velger (open + high + low + close) / 4 som ``src``. ATR
    bruker fortsatt originalens high/low/close. ``buy`` og ``sell`` er faktiske
    kryss mot stoppserien, inkludert forrige bars stopp, som ``crossover``.
    ``side`` er 1 over stoppen, −1 under og 0 ved likhet/ukjent stopp. Det er
    en indikatorfarge, ikke en porteføljeposisjon eller en utført ordre.
    """
    if isinstance(key_value, (bool, np.bool_)) or not isinstance(key_value, Real):
        raise ValueError("Nøkkelverdi må være et positivt, endelig tall.")
    if not np.isfinite(key_value) or key_value <= 0:
        raise ValueError("Nøkkelverdi må være et positivt, endelig tall.")
    if isinstance(atr_period, (bool, np.bool_)) or not isinstance(atr_period, Integral) or atr_period < 1:
        raise ValueError("ATR-perioden må være et positivt heltall.")
    if not isinstance(heikin_ashi, (bool, np.bool_)):
        raise ValueError("Heikin Ashi må være True eller False.")
    index, open_, high, low, close = _validated_ohlc(ohlcv)
    n = len(index)
    src = (open_ + high + low + close) / 4.0 if heikin_ashi else close.copy()
    atr = np.full(n, np.nan)
    stop = np.full(n, np.nan)
    buy = np.zeros(n, dtype=bool)
    sell = np.zeros(n, dtype=bool)
    if n:
        tr = high - low
        tr[1:] = np.maximum.reduce((tr[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        alpha = 1.0 / atr_period
        if n >= atr_period:
            atr[atr_period - 1] = tr[:atr_period].mean()
            for i in range(atr_period, n):
                atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
        for i in range(n):
            # Dette tilsvarer nz(xATRTrailingStop[1], 0), også før ATR-seed.
            previous_stop = stop[i - 1] if i and np.isfinite(stop[i - 1]) else 0.0
            previous_src = src[i - 1] if i else np.nan
            loss = key_value * atr[i]
            if src[i] > previous_stop and previous_src > previous_stop:
                stop[i] = max(previous_stop, src[i] - loss) if np.isfinite(loss) else np.nan
            elif src[i] < previous_stop and previous_src < previous_stop:
                stop[i] = min(previous_stop, src[i] + loss) if np.isfinite(loss) else np.nan
            else:
                stop[i] = src[i] - loss if src[i] > previous_stop else src[i] + loss
            if i and np.isfinite(stop[i - 1]) and np.isfinite(stop[i]):
                # ta.ema(src, 1) = src. Forrige sammenligning inkluderer likhet.
                buy[i] = src[i] > stop[i] and src[i - 1] <= stop[i - 1]
                sell[i] = src[i] < stop[i] and src[i - 1] >= stop[i - 1]
    side = np.where(src > stop, 1, np.where(src < stop, -1, 0)).astype(np.int8)
    return pd.DataFrame({"src": src, "atr": atr, "stop": stop,
                         "buy": buy, "sell": sell, "side": side}, index=index)
