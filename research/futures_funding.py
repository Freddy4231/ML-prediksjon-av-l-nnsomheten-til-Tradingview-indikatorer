"""Historiske funding-rater for USDT-perpetuals (Binance som offentlig kilde).

WEEX publiserer ikke lang nok funding-historikk. Binance sine rater brukes derfor som
nærmeste offentlige proxy; de faktiske WEEX-ratene kan avvike. Filene skrives til
``data/funding/`` (git-ignorert) og SHA-256 lagres i forsøkets låste plan.

Kjør:  python -m research.futures_funding
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://fapi.binance.com/fapi/v1/fundingRate"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "LTCUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT")
DIRECTORY = Path("data/funding")
# Samme siste time som prosjektets BTC- og altkoinfiler. Binance-perpetuals startet september 2019;
# startTime=0 tolkes av API-et som «siste 1000», så en eksplisitt start er nødvendig.
START = pd.Timestamp("2019-09-01", tz="UTC")
END = pd.Timestamp("2026-10-11 00:00", tz="UTC")


def path_for(symbol, directory=DIRECTORY):
    return Path(directory) / f"{symbol.lower()}_funding.csv"


def download(symbol, end=END, session=None):
    """Hent alle rater fra første tilgjengelige til ``end`` (1000 per kall)."""
    session = session or requests.Session()
    start, rows = int(START.timestamp() * 1000), []
    end_ms = int(end.timestamp() * 1000)
    while True:
        response = session.get(URL, params={"symbol": symbol, "startTime": start, "limit": 1000}, timeout=20)
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list):
            raise ValueError(f"{symbol}: uventet svar fra funding-kilden.")
        if not batch:
            break
        last = int(batch[-1]["fundingTime"])
        rows += [r for r in batch if int(r["fundingTime"]) <= end_ms]
        if len(batch) < 1000 or last >= end_ms:
            break
        start = last + 1
        time.sleep(0.3)
    frame = pd.DataFrame({"funding_time": pd.to_datetime([int(r["fundingTime"]) for r in rows], unit="ms", utc=True),
                          "funding_rate": [float(r["fundingRate"]) for r in rows]})
    frame = frame.drop_duplicates("funding_time").sort_values("funding_time")
    if frame.empty or not frame.funding_rate.abs().lt(0.05).all():
        raise ValueError(f"{symbol}: tom eller urimelig funding-serie.")
    return frame


def load(symbol, directory=DIRECTORY):
    frame = pd.read_csv(path_for(symbol, directory))
    frame["funding_time"] = pd.to_datetime(frame.funding_time, utc=True, format="ISO8601")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=list(SYMBOLS))
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    for symbol in args.symbols:
        frame = download(symbol)
        frame.to_csv(path_for(symbol), index=False)
        print(f"{symbol}: {len(frame)} rater {frame.funding_time.iloc[0]} → {frame.funding_time.iloc[-1]}, "
              f"snitt {frame.funding_rate.mean() * 100:.4f} % per 8 t")


if __name__ == "__main__":
    main()
