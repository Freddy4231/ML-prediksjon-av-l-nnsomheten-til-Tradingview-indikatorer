"""Laster ned historiske timesdata for BTC/USDT fra Binance sitt offentlige arkiv.

Kilde: https://data.binance.vision (gratis, ingen API-nøkkel).
Én zip-fil per måned lastes ned, sjekksummen kontrolleres, og alt samles i
data/btcusdt_1h.csv. Nedlastede zip-filer caches i data/raw/, så en ny kjøring
bare henter det som mangler. Andre symboler kan hentes med --symbol (f.eks. ETHUSDT).

Med --market futures hentes perpetual futures (USDⓈ-M) i stedet. De finnes fra januar 2020
og lagres i data/btcusdt_perp_1h.csv (cache i data/raw_futures/).

Kjør:  python download_data.py
       python download_data.py --start 2020-01 --end 2026-09
       python download_data.py --market futures
"""

import argparse
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DEFAULT_BASE_URL = "https://data.binance.vision"
DEFAULT_START = "2017-08"  # BTCUSDT ble lansert på Binance i august 2017
# Fast sluttmåned gjør at alle i gruppa (og sensor) får nøyaktig samme datasett.
# Oppdater den hvis dere vil trene på nyere data.
DEFAULT_END = "2026-09"

# Kolonnene vi tar vare på, med posisjonen i Binance-formatet. Kolonne 6 er sluttidspunktet og
# 7 er volum i USDT, de trengs ikke. taker_buy_volume er volumet fra aggressive kjøp
# (ordre som handler mot ordreboken med en gang), og brukes til ordreflyt-features.
KLINE_COLUMNS = {
    "open_time": 0, "open": 1, "high": 2, "low": 3, "close": 4, "volume": 5,
    "trades": 8, "taker_buy_volume": 9,
}
COLUMNS = list(KLINE_COLUMNS)
MARKET_PATHS = {"spot": "spot", "futures": "futures/um"}


def parse_kline_csv(fileobj) -> pd.DataFrame:
    """Leser én måneds-CSV fra Binance (kolonneformatet til /api/v3/klines)."""
    raw = pd.read_csv(fileobj, header=None, dtype=str)
    # Noen filer har en overskriftsrad. Den kjennes igjen på at første celle ikke er et tall.
    if not raw.iloc[0, 0].strip().isdigit():
        raw = raw.iloc[1:]
    raw = raw.iloc[:, list(KLINE_COLUMNS.values())]
    raw.columns = COLUMNS

    timestamps = raw["open_time"].astype("int64").to_numpy()
    # Fra 1. januar 2025 bruker Binance mikrosekunder i stedet for millisekunder for spot-data
    timestamps_ms = np.where(timestamps > 10**14, timestamps // 1000, timestamps)

    df = raw[COLUMNS[1:]].astype(float)
    df.index = pd.to_datetime(timestamps_ms, unit="ms", utc=True)
    df.index.name = "open_time"
    # Sikkerhetssjekk i tilfelle Binance endrer tidsformatet igjen
    if df.index.min() < pd.Timestamp("2009-01-01", tz="UTC"):
        raise ValueError(f"Uventet tidsformat i Binance-filen (første tid: {df.index.min()})")
    return df


def fetch_archive(session, base_url, symbol, interval, period, raw_dir: Path, market="spot", daily=False):
    """Henter en måneds- eller dagsfil med SHA-256-kontroll før den caches."""
    filename = f"{symbol}-{interval}-{period}.zip"
    local_zip = raw_dir / filename

    if not local_zip.exists():
        frequency = "daily" if daily else "monthly"
        url = f"{base_url}/data/{MARKET_PATHS[market]}/{frequency}/klines/{symbol}/{interval}/{filename}"
        response = session.get(url, timeout=20)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        content = response.content

        # Kontroller at filen ikke er ødelagt underveis (Binance publiserer SHA-256 per fil)
        checksum = session.get(url + ".CHECKSUM", timeout=20)
        checksum.raise_for_status()
        expected = checksum.text.split()[0].strip().lower()
        actual = hashlib.sha256(content).hexdigest()
        if expected != actual:
            raise ValueError(f"Feil sjekksum for {filename}")
        local_zip.write_bytes(content)

    with zipfile.ZipFile(local_zip) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as f:
            return parse_kline_csv(io.TextIOWrapper(f, encoding="utf-8"))


def fetch_month(session, base_url, symbol, interval, month, raw_dir: Path, market="spot"):
    """Henter én måned. Bevarer grensesnittet for eksisterende kode."""
    return fetch_archive(session, base_url, symbol, interval, month, raw_dir, market)


def update_recent(args, symbol, out_path: Path):
    """Legger til avsluttede dager uten å erstatte eller forkorte tidligere historikk.

    Månedlige arkiv er ikke publisert for måneden som pågår. Dagsarkiv gir derfor
    ferskere trening. Gårsdagens UTC-dato er øvre grense, også ved eksplisitt --through.
    En manglende dagsfil stopper oppdateringen, så hull ikke skjules som nye data.
    """
    if not out_path.exists():
        raise SystemExit("--update krever en eksisterende CSV. Kjør først vanlig nedlasting.")
    existing = pd.read_csv(out_path, index_col="open_time")
    existing.index = pd.to_datetime(existing.index, utc=True, format="ISO8601")
    existing = existing.sort_index()
    yesterday = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
    through = pd.Timestamp(args.through, tz="UTC") if args.through else yesterday
    if through > yesterday:
        raise SystemExit("--through må være en avsluttet UTC-dag (senest i går).")
    first = existing.index[-1].normalize()
    if existing.index[-1].hour == 23:
        first += pd.Timedelta(days=1)
    raw_dir = out_path.parent / ("raw" if args.market == "spot" else f"raw_{args.market}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = [existing]
    with requests.Session() as session:
        session.headers["User-Agent"] = "dat158-btc-prediksjon/1.0"
        for day in pd.date_range(first, through, freq="D"):
            period = day.strftime("%Y-%m-%d")
            frame = fetch_archive(session, args.base_url.rstrip("/"), symbol, args.interval,
                                  period, raw_dir, args.market, daily=True)
            if frame is None:
                print(f"{period}: dagsarkivet er ikke publisert ennå. Stopper her.")
                break
            expected = pd.date_range(day, day + pd.Timedelta(hours=23), freq="h")
            if not frame.index.equals(expected):
                raise ValueError(f"Ufullstendig dagsarkiv {period}: forventet 24 sammenhengende timer.")
            frames.append(frame)
            print(f"{symbol} {period}: {len(frame)} timer", flush=True)
    if len(frames) == 1:
        print("Ingen nye avsluttede dager tilgjengelig. Eksisterende historikk er beholdt.")
        return
    data = pd.concat(frames).sort_index()
    data = data[~data.index.duplicated(keep="first")]
    data.index.name = "open_time"
    temporary = out_path.with_suffix(".csv.tmp")
    data.to_csv(temporary)
    temporary.replace(out_path)
    print(f"La til {len(data) - len(existing)} timer. Historikken slutter {data.index[-1]}.")


def download(args, symbol, out_path: Path):
    """Laster ned (eller leser fra cache) alle månedene for ett symbol og lagrer én CSV."""
    raw_dir = out_path.parent / ("raw" if args.market == "spot" else f"raw_{args.market}")
    raw_dir.mkdir(parents=True, exist_ok=True)

    months = [p.strftime("%Y-%m") for p in pd.period_range(args.start, args.end, freq="M")]
    frames, missing = [], []
    with requests.Session() as session:
        session.headers["User-Agent"] = "dat158-btc-prediksjon/1.0"
        for i, month in enumerate(months, start=1):
            df = fetch_month(session, args.base_url.rstrip("/"), symbol, args.interval, month, raw_dir, args.market)
            if df is None:
                missing.append(month)
                print(f"[{symbol} {i}/{len(months)}] {month}: finnes ikke (hoppet over)")
            else:
                frames.append(df)
                print(f"[{symbol} {i}/{len(months)}] {month}: {len(df)} rader")

    if not frames:
        raise SystemExit(f"Fant ingen data for {symbol}. Sjekk symbol, intervall og datoer.")

    data = pd.concat(frames).sort_index()
    data = data[~data.index.duplicated(keep="last")]
    expected_hours = int((data.index[-1] - data.index[0]) / pd.Timedelta(hours=1)) + 1
    data.to_csv(out_path)

    print()
    print(f"Lagret {len(data):,} timer til {out_path}")
    print(f"Periode: {data.index[0]} til {data.index[-1]}")
    print(f"Manglende timer i perioden: {expected_hours - len(data)} (fylles inn i features.regularize_hourly)")
    if missing:
        print(f"Måneder uten data: {', '.join(missing)}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", action="append",
                        help="kan gis flere ganger, f.eks. --symbol BTCUSDT --symbol ETHUSDT. Standard: BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start", default=DEFAULT_START, help="første måned, f.eks. 2017-08")
    parser.add_argument("--end", default=DEFAULT_END, help="siste måned, f.eks. 2026-09")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--market", choices=list(MARKET_PATHS), default="spot")
    parser.add_argument("--update", action="store_true", help="legg til nye avsluttede dager i eksisterende CSV")
    parser.add_argument("--through", help="siste UTC-dag ved --update, f.eks. 2026-10-10 (standard: i går)")
    parser.add_argument("--out", default=None,
                        help="bare med ett symbol. Standard: data/<symbol>_1h.csv eller data/<symbol>_perp_1h.csv")
    args = parser.parse_args()

    if args.interval != "1h":
        raise SystemExit("Modellen bruker timesdata. --interval må være 1h.")
    if args.through and not args.update:
        raise SystemExit("--through brukes sammen med --update.")

    if args.market == "futures" and args.start == DEFAULT_START:
        args.start = "2020-01"  # perpetual futures finnes i arkivet fra januar 2020
    symbols = args.symbol or ["BTCUSDT"]
    if args.out and len(symbols) > 1:
        raise SystemExit("--out kan bare brukes med ett symbol.")
    for symbol in symbols:
        suffix = "_1h.csv" if args.market == "spot" else "_perp_1h.csv"
        output = Path(args.out or f"data/{symbol.lower()}{suffix}")
        (update_recent if args.update else download)(args, symbol, output)


if __name__ == "__main__":
    main()
