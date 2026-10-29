"""Separat, SHA-kontrollert open-interest-grunnlag til TMG-forskning.

Originale tidsstempler beholdes. Motstridende duplikater blir utilgjengelige.
Arkivet dokumenterer ikke historisk API-publisering; det er en begrensning,
selv om forskningen legger til en konservativ forsinkelse på 65 minutter.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import threading
import time
import zipfile

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tmg_oi_2026-10-26"
REPORT = ROOT / "reports/tmg_oi_2026-10-26"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
LOCAL = threading.local()


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def parse(payload, symbol, day):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [n for n in archive.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Uventet OI-arkiv")
        frame = pd.read_csv(archive.open(names[0]))
    if not {"create_time", "symbol", "sum_open_interest"}.issubset(frame.columns):
        raise ValueError("Mangler OI-kolonner")
    if not frame.symbol.eq(symbol).all():
        raise ValueError("Feil OI-marked")
    index = pd.DatetimeIndex(pd.to_datetime(frame.create_time, utc=True, format="mixed"))
    start = pd.Timestamp(day, tz="UTC")
    if index.hasnans or (index < start).any() or (index > start + pd.Timedelta(days=1)).any():
        raise ValueError("OI-tidspunkt utenfor arkivdøgnet")
    values = pd.to_numeric(frame.sum_open_interest, errors="coerce").to_numpy(float)
    valid = np.isfinite(values) & (values > 0)
    return pd.DataFrame({"open_interest": values[valid]}, index=index[valid]), int((~valid).sum())


def merge(frames):
    """Ikke velg en gunstig verdi når samme tidsstempel har motstridende versjoner."""
    data = pd.concat(frames).sort_index()
    conflicts = data.groupby(level=0).open_interest.nunique().gt(1)
    bad = conflicts.index[conflicts]
    data = data.loc[~data.index.isin(bad)]
    data = data.loc[~data.index.duplicated()]
    data.index.name = "observed_time"
    return data, [t.isoformat() for t in bad]


def align(index, snapshots):
    """Kjent ved signalclose: arkivtid +65min; maks to timer etter dette tidspunktet.

    Dette er en konservativ antakelse om tilgang, ikke dokumentert historisk
    publisering eller garanti for en uendret TradingView-feed.
    """
    if index.tz is None or not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("Ugyldig prisakse")
    if not snapshots.index.is_monotonic_increasing or not snapshots.index.is_unique:
        raise ValueError("Ugyldig OI-akse")
    available = snapshots.copy()
    available["observed_time"] = snapshots.index
    available.index = snapshots.index + pd.Timedelta(minutes=65)
    closes = index + pd.Timedelta(hours=1)
    out = available.reindex(closes, method="ffill")
    age = closes - (pd.DatetimeIndex(out.observed_time) + pd.Timedelta(minutes=65))
    out.loc[(age < pd.Timedelta(0)) | (age >= pd.Timedelta(hours=2)), "open_interest"] = np.nan
    out.index = index
    return out


def archive(task):
    symbol, day = task
    name = f"{symbol}-metrics-{day}.zip"
    url = f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{name}"
    path = DATA / "raw" / symbol / name
    check_path = path.with_suffix(".zip.CHECKSUM")
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    session = LOCAL.session
    for attempt in range(3):
        try:
            if path.exists() and check_path.exists():
                payload, check = path.read_bytes(), check_path.read_bytes()
            else:
                r = session.get(url, timeout=(10, 25))
                if r.status_code == 404:
                    return symbol, None, {"day":day,"url":url,"status":404}
                r.raise_for_status()
                payload = r.content
                r = session.get(url + ".CHECKSUM", timeout=(10, 25))
                r.raise_for_status()
                check = r.content
            expected = check.decode().split()[0].lower()
            if len(payload) > 8*1024*1024 or sha(payload) != expected:
                raise ValueError("Feil OI-arkivhash eller størrelse")
            frame, rejected = parse(payload, symbol, day)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(payload)
                check_path.write_bytes(check)
            return symbol,frame,{"day":day,"url":url,"status":200,"sha256":expected,
                                 "checksum_sha256":sha(check),"rows":len(frame),"invalid_rows":rejected}
        except (requests.RequestException,ValueError,zipfile.BadZipFile) as exc:
            if attempt == 2:
                raise RuntimeError(f"OI-arkiv {symbol} {day}: {exc}") from exc
            time.sleep(attempt+1)


def main(start, end):
    tasks = [(s,t.strftime("%Y-%m-%d")) for t in pd.date_range(start,end,inclusive="left",freq="1D") for s in SYMBOLS]
    frames,details = {s:[]for s in SYMBOLS},{s:[]for s in SYMBOLS}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for n,(symbol,frame,info) in enumerate(pool.map(archive,tasks),1):
            details[symbol].append(info)
            if frame is not None and len(frame):
                frames[symbol].append(frame)
            if n%150 == 0 or n==len(tasks):
                print(f"OI-arkiv {n}/{len(tasks)}",flush=True)
    manifest = {"start":start,"end_exclusive":end,"alignment":"Arkivtid +65min; maks2h gammel. Antatt, ikke verifisert historisk tilgjengelighet.",
                "source_sha256":sha(Path(__file__).read_bytes()),"symbols":{}}
    for s in SYMBOLS:
        merged,bad = merge(frames[s])
        path=DATA/f"{s}_open_interest.csv"
        payload=merged.to_csv().encode()
        if path.exists() and path.read_bytes()!=payload:
            raise ValueError("Nekter å overskrive et annet OI-grunnlag")
        path.write_bytes(payload)
        manifest["symbols"][s]={"path":str(path.relative_to(ROOT)),"sha256":sha(payload),"rows":len(merged),
                                 "conflicting_times":bad,"archives":details[s],"missing_days":sum(r["status"]==404 for r in details[s])}
        print(s,len(merged),"observasjoner, konflikter",len(bad),"manglende døgn",manifest["symbols"][s]["missing_days"],flush=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    path=REPORT/"data_manifest.json"
    payload=json.dumps(manifest,indent=2,ensure_ascii=False)+"\n"
    if path.exists() and path.read_text()!=payload:
        raise ValueError("Datamanifestet kan ikke overskrives")
    path.write_text(payload)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--start",default="2021-01-01")
    p.add_argument("--end",default="2025-01-01")
    args=p.parse_args()
    main(args.start,args.end)
