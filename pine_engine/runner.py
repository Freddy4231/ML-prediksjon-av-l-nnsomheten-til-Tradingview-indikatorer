"""Kjører et oversatt Pine-skript på lys og gjør hendelser/ordre om til handler.

Én kjøring per tidsramme. ``request.security`` mot høyere tidsramme, Heikin Ashi eller et annet
symbol løses med en egen hjelpekjøring av hele skriptet på de lysene; verdien leses uten
fremtidslekkasje (``lookahead_off``: lysets verdi blir kjent ved lysets close).
"""
import re
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pine_engine import runtime as R
from pine_engine.broker import Broker, NullBroker
from pine_engine.parser import PineError

SAFE_BUILTINS = {"len": len, "abs": abs, "float": float, "True": True, "False": False, "None": None}
LONG = re.compile(r"(?<![a-zæøå])(buy|long|bull|bullish|kj[øo]p|kjop|uptrend begins|up trend begins|to uptrend|"
                  r"to up trend|trend up|oppover|opp-signal)(?![a-zæøå])", re.IGNORECASE)
SHORT = re.compile(r"(?<![a-zæøå])(sell|short|bear|bearish|selg|downtrend begins|down trend begins|to downtrend|"
                   r"to down trend|trend down|nedover|ned-signal)(?![a-zæøå])", re.IGNORECASE)
EXCLUDE = re.compile(r"(?<![a-zæøå])(close|exit|lukk|tp\d*|sl|stop|take profit|target|hit|trend change|"
                     r"invalidat\w*)(?![a-zæøå])", re.IGNORECASE)
PRIORITY = {"plotshape": 1, "plotchar": 1, "alertcondition": 2, "label.new": 3, "plotarrow": 4, "alert": 5}
KIND_LABEL = {"plotshape": "plotshape", "plotchar": "plotchar", "alertcondition": "alertcondition",
              "label.new": "label.new", "plotarrow": "plotarrow", "alert": "alert()"}


@dataclass
class Bars:
    """Lys for én tidsramme. Tider i millisekunder (UTC), ``T`` = åpning, ``TC`` = close."""
    symbol: str
    minutes: int
    T: np.ndarray
    TC: np.ndarray
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray
    tickerid: str = ""
    mintick: float = 0.01
    currency: str = "USDT"
    basecurrency: str = ""
    symtype: str = "crypto"
    prefix: str = "BINANCE"

    def __len__(self):
        return len(self.C)

    def heikin_ashi(self):
        """Heikin Ashi-lys av de samme lysene (som ``ticker.heikinashi``)."""
        ha_close = (self.O + self.H + self.L + self.C) / 4
        ha_open = np.empty(len(self))
        ha_open[0] = (self.O[0] + self.C[0]) / 2
        for k in range(1, len(self)):
            ha_open[k] = (ha_open[k - 1] + ha_close[k - 1]) / 2
        ha_high = np.maximum.reduce([self.H, ha_open, ha_close])
        ha_low = np.minimum.reduce([self.L, ha_open, ha_close])
        return Bars(self.symbol, self.minutes, self.T, self.TC, ha_open, ha_high, ha_low, ha_close, self.V,
                    self.tickerid, self.mintick, self.currency, self.basecurrency, self.symtype, self.prefix)


class NeedContext(Exception):
    def __init__(self, key):
        self.key = key
        super().__init__(str(key))


class RT:
    """Alt den oversatte koden leser: priser, kalender, input, symbolinfo og hjelpeobjekter."""

    def __init__(self, bars, inputs, sources, events, broker, security, clock, main_minutes=None):
        n = len(bars)
        self.minutes = bars.minutes
        self.O, self.H, self.L, self.C, self.V = (a.astype(float).tolist() for a in (bars.O, bars.H, bars.L, bars.C,
                                                                                     bars.V))
        self.T, self.TC = bars.T.astype(float).tolist(), bars.TC.astype(float).tolist()
        h, l, c, o = bars.H, bars.L, bars.C, bars.O
        self.HL2, self.HLC3 = ((h + l) / 2).tolist(), ((h + l + c) / 3).tolist()
        self.OHLC4, self.HLCC4 = ((o + h + l + c) / 4).tolist(), ((h + l + c + c) / 4).tolist()
        stamps = pd.to_datetime(bars.T, unit="ms", utc=True)
        self.YEAR = stamps.year.astype(float).tolist()
        self.MONTH = stamps.month.astype(float).tolist()
        self.DOM = stamps.day.astype(float).tolist()
        self.DOW = (((stamps.dayofweek + 1) % 7) + 1).astype(float).tolist()
        self.HOUR = stamps.hour.astype(float).tolist()
        self.MINUTE = stamps.minute.astype(float).tolist()
        self.SECOND = stamps.second.astype(float).tolist()
        self.WOY = stamps.isocalendar().week.astype(float).tolist()
        self.TDAY = (bars.T - bars.T % 86_400_000).astype(float).tolist()
        self.LAST, self.LAST_TIME = n - 1, float(bars.T[-1])
        self.IN = inputs
        self.SRCS = [getattr(self, R_SOURCE[s]) if s else None for s in sources]
        self.EV, self.events = events, events
        drawings = R.Drawings()
        self.DR = drawings.new
        self.BR = broker
        self.SEC = security
        self.CLK = clock.loop
        self.clock = clock
        self.reg = R.Registry()
        self.REG = self.reg.series
        self.SITE = self._site
        self.TICKERID = bars.tickerid or f"{bars.prefix}:{bars.symbol}"
        self.TICKER = bars.symbol
        self.MINTICK = float(bars.mintick)
        self.PRICESCALE = round(1 / bars.mintick) if bars.mintick > 0 else 100.0
        self.CURRENCY, self.BASECURRENCY = bars.currency, bars.basecurrency
        self.SYMTYPE, self.PREFIX = bars.symtype, bars.prefix
        self.TF_PERIOD = R.tf_period(bars.minutes)
        self.TF_MAIN = R.tf_period(main_minutes or bars.minutes)
        self.TF_MULT = float(bars.minutes if bars.minutes < 1440 else bars.minutes // 1440)
        self.TF_INTRADAY, self.TF_DAILY = bars.minutes < 1440, bars.minutes >= 1440

    def _site(self, q):
        cls = R.SITES.get(q)
        if cls is None:
            raise PineError(f"intern feil: mangler {q}")
        return cls(self)


R_SOURCE = {"open": "O", "high": "H", "low": "L", "close": "C", "volume": "V", "hl2": "HL2", "hlc3": "HLC3",
            "ohlc4": "OHLC4", "hlcc4": "HLCC4"}


def namespace():
    """Bare navnene den genererte koden får bruke (se ``compiler._validate``)."""
    from pine_engine.compiler import PURE_IMPLS, RUNTIME_NAMES
    ns = {name: getattr(R, name) for name in RUNTIME_NAMES | PURE_IMPLS if hasattr(R, name)}
    ns["__builtins__"] = SAFE_BUILTINS
    return ns


_CODE_CACHE = {}


def make_factory(compiled):
    key = compiled.code
    if key not in _CODE_CACHE:
        ns = namespace()
        exec(compile(compiled.code, "<pine>", "exec"), ns)   # noqa: S102 – koden er generert og validert av oss
        if len(_CODE_CACHE) > 20:
            _CODE_CACHE.clear()
        _CODE_CACHE[key] = ns["make_script"]
    return _CODE_CACHE[key]


def resolve_inputs(compiled, overrides=None):
    """Input-verdier: skriptets standard, eventuelt overstyrt fra nettsiden (nøkkel = indeks)."""
    values, sources = [], []
    overrides = overrides or {}
    for item in compiled.inputs:
        value = overrides.get(item["index"], item["default"])
        kind = item["kind"]
        if kind == "source":
            name = str(value).strip().lower()
            if name not in R_SOURCE:
                raise PineError(f"input «{item['title']}»: kilden må være en av {', '.join(R_SOURCE)}")
            values.append(R.NAN)
            sources.append(name)
            continue
        sources.append(None)
        if kind == "bool":
            value = value if isinstance(value, bool) else str(value).strip().lower() in ("true", "ja", "1", "on")
        elif kind in ("int", "float"):
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise PineError(f"input «{item['title']}» må være et tall (fikk «{value}»)") from None
            if kind == "int":
                value = float(int(value))
        elif kind in ("string", "timeframe", "color"):
            value = "" if value is None else str(value)
        values.append(value)
    return values, sources


class Security:
    """``request.security`` i hovedkjøringen (oppslag) eller i en hjelpekjøring (lagrer verdiene)."""

    def __init__(self, bars, contexts, capture_key=None):
        self.bars, self.contexts, self.capture_key = bars, contexts, capture_key
        self.captured = {}
        self.current = ("self", bars.minutes)

    def key(self, symbol, tf):
        if symbol is None or R.isna(symbol) or symbol in ("", self.bars.tickerid, self.bars.symbol,
                                                          f"{self.bars.prefix}:{self.bars.symbol}"):
            sym = "self"
        elif str(symbol).startswith("HA:"):
            base = str(symbol)[3:]
            if base not in ("", self.bars.tickerid, self.bars.symbol, f"{self.bars.prefix}:{self.bars.symbol}"):
                raise R.PineRuntimeError("Heikin Ashi av et annet symbol støttes ikke")
            sym = "ha"
        else:
            sym = str(symbol).split(":")[-1].upper()
        minutes = R.tf_minutes(tf, self.bars.minutes)
        if minutes < self.bars.minutes and self.capture_key is None:
            raise R.PineRuntimeError("request.security mot lavere tidsramme enn grafen støttes ikke")
        return sym, minutes

    def __call__(self, site, i, symbol, tf, value, gaps="barmerge.gaps_off", lookahead="barmerge.lookahead_off"):
        key = self.key(symbol, tf)
        if self.capture_key is not None:
            # I hjelpekjøringen er det forespurte fremmede symbolet blitt kontekstens
            # eget symbol. Da normaliseres nøkkelen til «self» og må fortsatt fanges.
            if key == self.capture_key or key == ("self", self.bars.minutes):
                values = self.captured.setdefault(site, [R.NAN] * len(self.bars))
                values[i] = value
            return value
        if key == self.current:
            return value
        context = self.contexts.get(key)
        if context is None:
            raise NeedContext(key)
        # Tupler ([a, b] = request.security(...)) får na i hvert felt når verdien mangler. ``[a, b]`` oversettes
        # til en Python-liste, så begge former må gjenkjennes (ellers krasjer utpakkingen før første HTF-lys).
        sequence = isinstance(value, (tuple, list))
        missing = tuple(R.NAN for _ in value) if sequence else R.NAN
        values = context["values"].get(site)
        if values is None:
            return missing
        j, last, first = context["index"][i], context["is_last"][i], context["first"][i]
        if j < 0:
            return missing
        gaps_on = gaps == "barmerge.gaps_on"
        if lookahead == "barmerge.lookahead_on":
            out = missing if gaps_on and not first else values[j]
        elif last:
            out = values[j]
        elif gaps_on or j < 1:
            out = missing
        else:
            out = values[j - 1]
        return missing if sequence and not isinstance(out, (tuple, list)) else out


def _line_of(compiled, exc):
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        if frame.filename == "<pine>":
            return compiled.line_map.get(frame.lineno)
    return None


@dataclass
class PassResult:
    events: R.Events
    broker: object
    bars: Bars
    seconds: float
    captured: dict = field(default_factory=dict)


def run_pass(compiled, bars, *, inputs, sources, broker=None, contexts=None, capture_key=None, deadline=None,
             record_events=True, main_minutes=None, capture_plots=False):
    """Én kjøring av skriptet over alle lys. Megleren (strategi) handler før og etter hvert lys."""
    events = R.Events(enabled=record_events, capture_plots=capture_plots)
    clock = R.Clock(deadline)
    security = Security(bars, contexts or {}, capture_key)
    broker_obj = broker if broker is not None else NullBroker()
    rt = RT(bars, inputs, sources, events, broker_obj, security, clock, main_minutes)
    started = time.monotonic()
    try:
        bar = make_factory(compiled)(rt)
        on_open = broker.on_open if broker is not None else None
        on_close = broker.on_close if broker is not None else None
        trim = rt.reg.trim
        check = clock.check
        for i in range(len(bars)):
            if on_open is not None:
                on_open(i)
            bar(i)
            if on_close is not None:
                on_close(i)
            if not i & 1023:
                check()
                if not i & 4095:
                    trim()
    except NeedContext:
        raise
    except (R.PineTimeout, PineError):
        raise
    except R.PineRuntimeError as exc:
        raise PineError(f"kjøringsfeil: {exc}", _line_of(compiled, exc)) from None
    except RecursionError:
        raise PineError("skriptet er for dypt nøstet") from None
    except Exception as exc:   # noqa: BLE001 – alle feil i brukerkoden vises som Pine-feil
        raise PineError(f"kjøringsfeil ({type(exc).__name__}): {exc}", _line_of(compiled, exc)) from None
    return PassResult(events, broker, bars, time.monotonic() - started, security.captured)


def context_mapping(main, other):
    """For hvert hovedlys: indeks i den andre lysrekken, om hovedlyset avslutter det, og om det starter det."""
    j = np.searchsorted(other.T, main.T, side="right") - 1
    valid = j >= 0
    j = np.where(valid, j, -1)
    close_other = np.where(valid, other.TC[np.clip(j, 0, None)], np.inf)
    is_last = valid & (main.TC >= close_other)
    first = np.r_[True, j[1:] != j[:-1]] & valid
    return j.tolist(), is_last.tolist(), first.tolist()


def run_with_security(compiled, bars, *, inputs, sources, broker_factory=None, provider=None, deadline=None,
                      capture_plots=False):
    """Hovedkjøringen; ved behov kjøres skriptet først på lysene ``request.security`` ber om."""
    contexts = {}
    for _ in range(6):
        broker = broker_factory() if broker_factory else None
        try:
            return run_pass(compiled, bars, inputs=inputs, sources=sources, broker=broker, contexts=contexts,
                            deadline=deadline, capture_plots=capture_plots), contexts
        except NeedContext as need:
            key = need.key
            sym, minutes = key
            if sym == "ha":
                other = bars.heikin_ashi() if minutes == bars.minutes else None
                if other is None:
                    raise PineError("Heikin Ashi på en annen tidsramme støttes ikke") from None
            elif provider is None:
                raise PineError("request.security trenger data som ikke er tilgjengelig") from None
            else:
                other = provider(None if sym == "self" else sym, minutes)
            capture = run_pass(compiled, other, inputs=inputs, sources=sources, capture_key=key, deadline=deadline,
                               record_events=False, main_minutes=bars.minutes)
            index, is_last, first = context_mapping(bars, other)
            contexts[key] = {"values": capture.captured, "index": index, "is_last": is_last, "first": first}
    raise PineError("for mange forskjellige request.security-kall")


# ---------------------------------------------------------------- signaler fra indikatorer

def classify(kind, text, sample=None):
    probe = sample if kind == "alert" and sample else text
    if not probe or EXCLUDE.search(probe):
        return None
    long_hit, short_hit = bool(LONG.search(probe)), bool(SHORT.search(probe))
    if long_hit and not short_hit:
        return "long"
    if short_hit and not long_hit:
        return "short"
    return None


def candidates(events):
    """Alle hendelseskilder med antall og automatisk tolkning (long/short/ukjent)."""
    out = []
    for key, hits in events.hits.items():
        kind, site, text = key
        sample = events.samples.get(key)
        out.append({"key": f"{kind}|{site}|{text}", "kind": kind, "site": site, "text": text, "count": len(hits),
                    "class": classify(kind, text, sample), "sample": sample,
                    "label": f"{KIND_LABEL.get(kind, kind)} «{text}»" + (f" · {sample[:60]}" if sample else "")})
    out.sort(key=lambda c: (PRIORITY.get(c["kind"], 9), c["site"], c["text"]))
    return out


def choose(cands, side):
    """Automatisk valg: høyest prioritet (plotshape > alertcondition > label > plotarrow > alert), så flest treff."""
    options = [c for c in cands if c["class"] == side]
    if not options:
        return None
    return min(options, key=lambda c: (PRIORITY.get(c["kind"], 9), -c["count"]))["key"]


def signal_flags(events, key, n):
    flags = np.zeros(n, dtype=bool)
    if key:
        kind, site, text = key.split("|", 2)
        hits = events.hits.get((kind, int(site), text), [])
        flags[np.asarray(hits, dtype=int)] = True
    return flags


def simulate_signals(bars, first, long_flags, short_flags, cost, funding, long_only, mintick):
    """Indikator: long ved long-signal, short (eller ut, kun long) ved short-signal; fyll ved neste åpning."""
    broker = Broker(bars, first, cost=cost, funding=funding, long_only=long_only, mintick=mintick)
    both = long_flags & short_flags
    longs = (long_flags & ~both).tolist()
    shorts = (short_flags & ~both).tolist()
    for k in range(len(bars)):
        broker.on_open(k)
        if longs[k]:
            broker.entry(k, "Long", True)
        elif shorts[k]:
            broker.entry(k, "Short", False)
        broker.on_close(k)
    broker.finish(len(bars) - 1)
    return broker
