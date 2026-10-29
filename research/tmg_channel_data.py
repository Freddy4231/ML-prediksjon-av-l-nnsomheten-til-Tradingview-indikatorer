"""Isolert futures-/ordreflythistorikk til TMG-kanalanalysen. Offentlige SHA-kontrollerte arkiv."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tmg_channel_2026-10-26"
REPORT = ROOT / "reports/tmg_channel_2026-10-26"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def parse(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [x for x in archive.namelist() if x.endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Uventet arkivinnhold")
        raw = pd.read_csv(archive.open(names[0]), header=None, dtype=str)
    if not raw.iloc[0, 0].strip().isdigit():
        raw = raw.iloc[1:]
    timestamps = raw.iloc[:, 0].astype("int64").to_numpy()
    index = pd.to_datetime(np.where(timestamps > 10**14, timestamps // 1000, timestamps), utc=True, unit="ms")
    columns = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"]
    out = raw.iloc[:, [1, 2, 3, 4, 5, 7, 8, 9, 10]].astype(float)
    out.columns = columns
    out.index = index
    out.index.name = "open_time"
    if not np.isfinite(out.to_numpy()).all() or (out < 0).any().any():
        raise ValueError("Ugyldig pris eller volum")
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Pris må være positiv")
    if (out.high < out[["open", "close"]].max(axis=1)).any() or (out.low > out[["open", "close"]].min(axis=1)).any():
        raise ValueError("Ugyldig OHLC")
    for buy, volume in (("taker_buy_volume", "volume"), ("taker_buy_quote_volume", "quote_volume")):
        if (out[buy].to_numpy() - out[volume].to_numpy() > 8 * np.spacing(out[volume].to_numpy())).any():
            raise ValueError("Kjøpsvolum overstiger totalvolum")
    if out.index.has_duplicates or not out.index.is_monotonic_increasing or not (out.index == out.index.floor("h")).all():
        raise ValueError("Ugyldig tidsakse")
    return out


def archive(task):
    symbol, month = task
    name = f"{symbol}-1h-{month}.zip"
    url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/1h/{name}"
    local = DATA / "raw" / symbol / name
    old = ROOT / "data/raw_futures" / name
    source = local if local.exists() else old if old.exists() else None
    checksum = local.with_suffix(".zip.CHECKSUM")
    last_error = None
    for attempt in range(3):
        try:
            with requests.Session() as session:
                if source is None:
                    response = session.get(url, timeout=(10, 25))
                    if response.status_code == 404:
                        return symbol, month, None, {"month": month, "status": 404, "url": url}
                    response.raise_for_status()
                    payload = response.content
                else:
                    payload = source.read_bytes()
                if checksum.exists():
                    check = checksum.read_bytes()
                else:
                    response = session.get(url + ".CHECKSUM", timeout=(10, 25))
                    response.raise_for_status()
                    check = response.content
            expected = check.decode().split()[0].lower()
            if len(payload) > 8 * 1024 * 1024 or sha(payload) != expected:
                raise ValueError("Feil arkivstørrelse eller SHA-256")
            frame = parse(payload)
            start = pd.Timestamp(month, tz="UTC")
            end = start + pd.offsets.MonthBegin()
            if frame.index[0] < start or frame.index[-1] >= end:
                raise ValueError("Arkivet inneholder data utenfor måneden")
            local.parent.mkdir(parents=True, exist_ok=True)
            if source is None:
                local.write_bytes(payload)
                source = local
            checksum.write_bytes(check)
            return symbol, month, frame, {"month": month, "url": url, "status": 200,
                "archive_path": str(source.relative_to(ROOT)), "archive_sha256": expected,
                "checksum_path": str(checksum.relative_to(ROOT)), "checksum_sha256": sha(check), "rows": len(frame)}
        except (requests.RequestException, ValueError, zipfile.BadZipFile) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"{symbol} {month}: {last_error}")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    tasks = [(s, str(m)) for s in SYMBOLS for m in pd.period_range("2020-01", "2026-09", freq="M")]
    frames, details = {s: [] for s in SYMBOLS}, {s: [] for s in SYMBOLS}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number, (symbol, month, frame, info) in enumerate(pool.map(archive, tasks), 1):
            details[symbol].append(info)
            if frame is not None:
                frames[symbol].append(frame)
            if number % 20 == 0 or number == len(tasks):
                print(f"Arkiv kontrollert: {number}/{len(tasks)}", flush=True)
    manifest = {"source": "Binance USD-M perpetual 1h, offentlige månedsarkiv", "symbols": {}}
    for symbol in SYMBOLS:
        found = False
        for row in details[symbol]:
            if row["status"] == 200:
                found = True
            elif found:
                raise ValueError(f"Manglende hel måned etter kontraktstart: {symbol} {row['month']}")
        frame = pd.concat(frames[symbol]).sort_index()
        if frame.index.has_duplicates or frame.index[-1] != pd.Timestamp("2026-09-30 23:00", tz="UTC"):
            raise ValueError("Uventet slutt eller overlapp")
        path = DATA / f"{symbol}_perp_1h.csv"
        content = frame.to_csv().encode()
        if path.exists() and path.read_bytes() != content:
            raise ValueError("Nekter å overskrive et annet datasett")
        path.write_bytes(content)
        manifest["symbols"][symbol] = {"path": str(path.relative_to(ROOT)), "sha256": sha(content),
            "rows": len(frame), "start": str(frame.index[0]), "end": str(frame.index[-1]),
            "missing_hours": int((frame.index[-1] - frame.index[0]) / pd.Timedelta(hours=1)) + 1 - len(frame),
            "archives": details[symbol]}
        print(symbol, len(frame), "timer", flush=True)
    path = REPORT / "data_manifest.json"
    content = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != content:
        raise ValueError("Et tidligere datamanifest kan ikke endres")
    path.write_text(content)


if __name__ == "__main__":
    main()
