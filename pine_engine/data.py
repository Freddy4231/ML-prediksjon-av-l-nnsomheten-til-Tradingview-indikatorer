"""Markedsdata til Pine-testen.

- **Krypto (Binance spot):** månedsarkiv og dagsarkiv fra data.binance.vision (SHA-256 mot
  Binances CHECKSUM, lagret i ``data/pine_cache/``), resten av inneværende døgn fra REST-API-et.
  Full historikk fra 2017. Faktisk funding hentes fra Binance USDT-perp når paret finnes der.
- **Aksjer, indekser, valuta og råvarer (Yahoo Finance):** gratis, men begrenset historikk for
  korte intervaller (1 min: ca. 30 dager, 2–90 min: ca. 60 dager, 1 time: ca. 2 år, 1 dag: alt).

Intervaller Binance/Yahoo ikke har (f.eks. 45 min), bygges av det største intervallet som går opp
i det (45 min = 3 × 15 min). Lysene starter ved UTC-midnatt for krypto (som TradingView) og ved
øktens første lys for aksjer. Et lys som ikke er ferdig ved slutten av perioden, tas ikke med.
"""
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from download_data import DEFAULT_BASE_URL, fetch_archive
from pine_engine.runner import Bars
from pine_engine.runtime import MONTH, is_months

CACHE = Path("data/pine_cache")
BINANCE_INTERVALS = {1: "1m", 3: "3m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h", 360: "6h",
                     480: "8h", 720: "12h", 1440: "1d"}
YAHOO_INTERVALS = {1: "1m", 2: "2m", 5: "5m", 15: "15m", 30: "30m", 60: "60m", 90: "90m", 1440: "1d"}
YAHOO_MAX_DAYS = {1: 29, 2: 59, 5: 59, 15: 59, 30: 59, 60: 729, 90: 59, 1440: None}
MAX_BARS = 400_000
HEADERS = {"User-Agent": "Mozilla/5.0 (dat158-pine-test)"}
SYMBOL = re.compile(r"^[A-Z0-9]{2,20}$")
YAHOO_SYMBOL = re.compile(r"^[A-Za-z0-9.^=\-]{1,20}$")
_INFO_CACHE, _FUNDING_CACHE = {}, {}
_LOCK = threading.Lock()


class DataError(Exception):
    """Data kunne ikke hentes (ukjent symbol, for lang periode, nettverksfeil)."""


def base_interval(minutes, table):
    options = [b for b in table if minutes % b == 0]
    if not options:
        raise DataError(f"intervallet {minutes} min kan ikke bygges av kildens intervaller")
    return max(options)


def _ms(stamp):
    stamp = pd.Timestamp(stamp)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return int(stamp.value // 1_000_000)


# ---------------------------------------------------------------- Binance

def binance_info(symbol, session=None):
    """Symbolinfo (tick-størrelse, basis- og kursvaluta) fra Binance; feiler for ukjente symboler."""
    symbol = symbol.upper().strip()
    if not SYMBOL.match(symbol):
        raise DataError("Binance-symbolet må være bokstaver/tall, f.eks. BTCUSDT")
    with _LOCK:
        if symbol in _INFO_CACHE:
            return _INFO_CACHE[symbol]
    session = session or requests.Session()
    try:
        response = session.get("https://api.binance.com/api/v3/exchangeInfo", params={"symbol": symbol},
                               headers=HEADERS, timeout=15)
    except requests.RequestException as exc:
        raise DataError(f"Binance svarer ikke ({exc.__class__.__name__})") from None
    if response.status_code == 400:
        raise DataError(f"«{symbol}» finnes ikke på Binance spot")
    response.raise_for_status()
    item = response.json()["symbols"][0]
    tick = next((float(f["tickSize"]) for f in item.get("filters", []) if f.get("filterType") == "PRICE_FILTER"),
                0.01)
    info = {"symbol": symbol, "base": item.get("baseAsset", ""), "quote": item.get("quoteAsset", "USDT"),
            "mintick": tick or 0.01}
    with _LOCK:
        _INFO_CACHE[symbol] = info
    return info


def _rest_klines(session, symbol, interval, start_ms, end_ms):
    rows, cursor = [], start_ms
    while cursor < end_ms:
        response = session.get("https://api.binance.com/api/v3/klines", headers=HEADERS, timeout=15,
                               params={"symbol": symbol, "interval": interval, "startTime": cursor,
                                       "endTime": end_ms - 1, "limit": 1000})
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 1
        if len(batch) < 1000:
            break
    if not rows:
        return None
    frame = pd.DataFrame([[float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows],
                         columns=["open", "high", "low", "close", "volume"],
                         index=pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True))
    return frame


def binance_klines(symbol, minutes, start_ms, end_ms, progress=None):
    """Binance-lys for ``[start_ms, end_ms)`` på et innebygd intervall, fra arkiv + REST."""
    interval = BINANCE_INTERVALS[minutes]
    raw_dir = CACHE / "binance" / symbol / interval
    raw_dir.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    today = now.normalize()
    start = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end = min(pd.Timestamp(end_ms, unit="ms", tz="UTC"), now)
    jobs = []
    for month in pd.period_range(start.tz_localize(None).to_period("M"), end.tz_localize(None).to_period("M"),
                                 freq="M"):
        month_start = pd.Timestamp(month.start_time, tz="UTC")
        month_end = pd.Timestamp(month.end_time, tz="UTC").normalize() + pd.Timedelta(days=1)
        if month_end <= today - pd.Timedelta(days=1) and month_end <= now:
            jobs.append(("month", month.strftime("%Y-%m"), month_start, month_end))
        else:
            first_day = max(month_start, start.normalize())
            for day in pd.date_range(first_day, min(month_end, today) - pd.Timedelta(days=1), freq="D"):
                jobs.append(("day", day.strftime("%Y-%m-%d"), day, day + pd.Timedelta(days=1)))
    frames = []

    def fetch(job, session):
        kind, period, begin, finish = job
        frame = fetch_archive(session, DEFAULT_BASE_URL, symbol, interval, period, raw_dir, daily=kind == "day")
        if frame is not None:
            return [frame]
        if kind == "month":
            out = []
            for day in pd.date_range(begin, min(finish, today) - pd.Timedelta(days=1), freq="D"):
                part = fetch_archive(session, DEFAULT_BASE_URL, symbol, interval, day.strftime("%Y-%m-%d"), raw_dir,
                                     daily=True)
                out.append(part if part is not None else
                           _rest_klines(session, symbol, interval, _ms(day), _ms(day + pd.Timedelta(days=1))))
            return [f for f in out if f is not None]
        rest = _rest_klines(session, symbol, interval, _ms(begin), _ms(finish))
        return [rest] if rest is not None else []

    sessions = threading.local()

    def worker(job):
        if not hasattr(sessions, "s"):
            sessions.s = requests.Session()
            sessions.s.headers.update(HEADERS)
        return fetch(job, sessions.s)

    done = 0
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for parts in pool.map(worker, jobs):
                frames.extend(parts)
                done += 1
                if progress:
                    progress(done / max(len(jobs), 1))
        with requests.Session() as session:
            tail = _rest_klines(session, symbol, interval, _ms(max(today, start)), _ms(end) + 1) \
                if end > today else None
    except requests.RequestException as exc:
        raise DataError(f"kunne ikke hente Binance-data ({exc.__class__.__name__})") from None
    except ValueError as exc:
        raise DataError(f"Binance-arkivet er ugyldig: {exc}") from None
    if tail is not None:
        frames.append(tail)
    if not frames:
        raise DataError(f"ingen Binance-data for {symbol} i perioden")
    data = pd.concat([f[["open", "high", "low", "close", "volume"]] for f in frames]).sort_index()
    data = data[~data.index.duplicated(keep="last")]
    stamps = data.index.as_unit("ms").asi8
    return data[(stamps >= start_ms) & (stamps < end_ms)]


def binance_funding(symbol, start_ms, end_ms):
    """Faktiske funding-rater (tid i ms, rate) fra Binance USDT-perp; ``None`` hvis paret ikke finnes der."""
    key = symbol
    with _LOCK:
        cached = _FUNDING_CACHE.get(key)
    if cached is not None and cached[0] <= start_ms and cached[1] >= min(end_ms, int(time.time() * 1000) - 3_600_000):
        times, rates = cached[2], cached[3]
        keep = (times >= start_ms) & (times < end_ms)
        return times[keep], rates[keep]
    rows, cursor = [], start_ms
    try:
        with requests.Session() as session:
            while cursor < end_ms:
                response = session.get("https://fapi.binance.com/fapi/v1/fundingRate", headers=HEADERS, timeout=15,
                                       params={"symbol": symbol, "startTime": cursor, "endTime": end_ms,
                                               "limit": 1000})
                if response.status_code == 400:
                    return None
                response.raise_for_status()
                batch = response.json()
                if not isinstance(batch, list):
                    return None
                if not batch:
                    break
                rows.extend(batch)
                cursor = int(batch[-1]["fundingTime"]) + 1
                if len(batch) < 1000:
                    break
    except requests.RequestException:
        return None
    if not rows:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    times = np.array([int(r["fundingTime"]) // 60_000 * 60_000 for r in rows], dtype=np.int64)
    rates = np.array([float(r["fundingRate"]) for r in rows], dtype=float)
    with _LOCK:
        _FUNDING_CACHE[key] = (start_ms, end_ms, times, rates)
    return times, rates


def funding_per_bar(bars, times, rates):
    """Rate per lys: funding med tidspunkt i (forrige åpning, denne åpningen] belastes ved denne åpningen."""
    out = np.zeros(len(bars))
    if times is None or len(times) == 0:
        return out.tolist()
    position = np.searchsorted(bars.T, times, side="left")
    keep = (position > 0) & (position < len(bars))
    np.add.at(out, position[keep], rates[keep])
    return out.tolist()


# ---------------------------------------------------------------- Yahoo Finance

def yahoo_klines(symbol, minutes, start_ms, end_ms):
    """Yahoo-lys med begrenset historikk for korte intervaller; returnerer også børsens tidssone."""
    if not YAHOO_SYMBOL.match(symbol or ""):
        raise DataError("Yahoo-symbolet kan bare inneholde bokstaver, tall og . ^ = -")
    interval = YAHOO_INTERVALS[minutes]
    limit = YAHOO_MAX_DAYS[minutes]
    now_ms = int(time.time() * 1000)
    if limit is not None:
        earliest = now_ms - limit * 86_400_000
        start_ms = max(start_ms, earliest)
    if start_ms >= end_ms:
        raise DataError(f"Yahoo har bare de siste {limit} dagene med {interval}-lys")
    chunk = 7 * 86_400_000 if minutes == 1 else end_ms - start_ms
    frames, tz = [], "UTC"
    cursor = start_ms
    try:
        with requests.Session() as session:
            while cursor < end_ms:
                stop = min(cursor + chunk, end_ms)
                response = session.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                                       headers=HEADERS, timeout=20,
                                       params={"period1": cursor // 1000, "period2": stop // 1000,
                                               "interval": interval, "includePrePost": "false"})
                if response.status_code == 404:
                    raise DataError(f"«{symbol}» finnes ikke på Yahoo Finance")
                response.raise_for_status()
                result = (response.json().get("chart") or {}).get("result")
                if not result:
                    raise DataError(f"Yahoo Finance ga ingen data for «{symbol}»")
                result = result[0]
                tz = result.get("meta", {}).get("exchangeTimezoneName") or tz
                stamps = result.get("timestamp") or []
                quote = (result.get("indicators", {}).get("quote") or [{}])[0]
                if stamps:
                    frame = pd.DataFrame({k: quote.get(k) for k in ("open", "high", "low", "close", "volume")},
                                         index=pd.to_datetime(np.asarray(stamps, dtype=np.int64), unit="s", utc=True))
                    frames.append(frame)
                cursor = stop
    except requests.RequestException as exc:
        raise DataError(f"Yahoo Finance svarer ikke ({exc.__class__.__name__})") from None
    if not frames:
        raise DataError(f"ingen Yahoo-data for «{symbol}» i perioden")
    data = pd.concat(frames).sort_index()
    data = data[~data.index.duplicated(keep="last")]
    data = data.apply(pd.to_numeric, errors="coerce").dropna(subset=["open", "high", "low", "close"])
    data["volume"] = data["volume"].fillna(0.0)
    data = data[(data[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return data, tz


# ---------------------------------------------------------------- sammenslåing

DAY = 86_400_000


def _day_buckets(day, minutes):
    """Start på dag-/uke-/månedslyset for hver dag (ms ved midnatt): uker starter mandag, måneder den 1."""
    if is_months(minutes):
        stamps = pd.to_datetime(day, unit="ms", utc=True)
        n = minutes // MONTH
        index = (stamps.year.to_numpy() * 12 + stamps.month.to_numpy() - 1) // n * n
        starts = pd.to_datetime(pd.DataFrame({"year": index // 12, "month": index % 12 + 1, "day": 1}), utc=True)
        return starts.dt.as_unit("ms").astype("int64").to_numpy()
    if minutes % 10080 == 0:
        weekday = (day // DAY + 3) % 7                    # 1970-01-01 var en torsdag
        monday = day - weekday * DAY
        return monday - (monday // (7 * DAY) % (minutes // 10080)) * 7 * DAY
    return day - (day // DAY % (minutes // 1440)) * DAY


def bucket_end(opened, minutes):
    if is_months(minutes):
        stamps = pd.to_datetime(opened, unit="ms", utc=True)
        n = minutes // MONTH
        index = stamps.year.to_numpy() * 12 + stamps.month.to_numpy() - 1 + n
        ends = pd.to_datetime(pd.DataFrame({"year": index // 12, "month": index % 12 + 1, "day": 1}), utc=True)
        return ends.dt.as_unit("ms").astype("int64").to_numpy()
    return opened + minutes * 60_000


def aggregate(data, minutes, base, tz=None, end_ms=None):
    """Base-lys → ``minutes``-lys. Krypto: UTC-midnatt som anker. Aksjer: øktens første lys hver dag.
    Uker starter mandag og måneder den 1. (som TradingView)."""
    t = data.index.as_unit("ms").asi8.astype(np.int64)
    step = minutes * 60_000
    if minutes == base:
        bucket = t
    elif minutes >= 1440:
        if tz is not None:
            local = data.index.tz_convert(tz)
            day = local.normalize().tz_convert("UTC").as_unit("ms").asi8.astype(np.int64)
        else:
            day = t - t % DAY
        bucket = day if minutes == 1440 else _day_buckets(day, minutes)
    elif tz is None:
        day = t - t % DAY
        bucket = day + (t - day) // step * step
    else:
        local_day = data.index.tz_convert(tz).normalize()
        first = pd.Series(t, index=local_day).groupby(level=0).transform("min").to_numpy()
        bucket = first + (t - first) // step * step
    starts = np.flatnonzero(np.r_[True, bucket[1:] != bucket[:-1]])
    o = data["open"].to_numpy(float)[starts]
    h = np.maximum.reduceat(data["high"].to_numpy(float), starts)
    l = np.minimum.reduceat(data["low"].to_numpy(float), starts)
    ends = np.r_[starts[1:], len(t)] - 1
    c = data["close"].to_numpy(float)[ends]
    v = np.add.reduceat(data["volume"].to_numpy(float), starts)
    opened = bucket[starts]
    if tz is None or minutes >= 1440:
        closed = bucket_end(opened, minutes)
        if len(opened) and t[ends[-1]] + base * 60_000 < closed[-1]:
            keep = slice(0, len(opened) - 1)       # siste lys er ikke ferdig
            opened, closed, o, h, l, c, v = (a[keep] for a in (opened, closed, o, h, l, c, v))
    else:
        closed = np.minimum(opened + step, t[ends] + base * 60_000)
    if end_ms is not None:
        keep = closed <= end_ms
        opened, closed, o, h, l, c, v = (a[keep] for a in (opened, closed, o, h, l, c, v))
    return opened, closed, o, h, l, c, v


# ---------------------------------------------------------------- inngang

def load_bars(source, symbol, minutes, start, end, warmup_bars=1500, progress=None):
    """Lys for ``[start, end)`` med ``warmup_bars`` ekstra lys før start (for indikatorenes oppvarming)."""
    start_ms, end_ms = _ms(start), _ms(end)
    now_ms = int(time.time() * 1000)
    end_ms = min(end_ms, now_ms)
    if end_ms <= start_ms:
        raise DataError("sluttdatoen må være etter startdatoen")
    need = (end_ms - start_ms) // (minutes * 60_000) + warmup_bars
    if need > MAX_BARS:
        days = (MAX_BARS - warmup_bars) * minutes // 1440
        raise DataError(f"perioden gir {need:,} lys på {minutes} min; maks er {MAX_BARS:,} (ca. {days} dager)"
                        .replace(",", " "))
    fetch_from = start_ms - warmup_bars * minutes * 60_000
    if source == "binance":
        info = binance_info(symbol)
        base = 1440 if is_months(minutes) else base_interval(minutes, BINANCE_INTERVALS)
        data = binance_klines(info["symbol"], base, max(fetch_from, _ms("2017-07-01")), end_ms, progress)
        tz = None
        bars_meta = dict(tickerid=f"BINANCE:{info['symbol']}", mintick=info["mintick"], currency=info["quote"],
                         basecurrency=info["base"], symtype="crypto", prefix="BINANCE")
        symbol = info["symbol"]
    elif source == "yahoo":
        base = 1440 if is_months(minutes) else base_interval(minutes, YAHOO_INTERVALS)
        data, tz = yahoo_klines(symbol, base, fetch_from, end_ms)
        bars_meta = dict(tickerid=f"YAHOO:{symbol}", mintick=0.01, currency="USD", basecurrency=symbol,
                         symtype="stock", prefix="YAHOO")
    else:
        raise DataError("ukjent datakilde")
    if data.empty:
        raise DataError("ingen lys i perioden")
    opened, closed, o, h, l, c, v = aggregate(data, minutes, base, tz=tz, end_ms=end_ms)
    if len(opened) < 50:
        raise DataError(f"for få lys ({len(opened)}) for {minutes} min i perioden")
    bars = Bars(symbol, minutes, opened, closed, o, h, l, c, v, **bars_meta)
    first = int(np.searchsorted(opened, start_ms))
    if first >= len(opened) - 1:
        raise DataError("ingen lys etter startdatoen")
    return bars, first, base
