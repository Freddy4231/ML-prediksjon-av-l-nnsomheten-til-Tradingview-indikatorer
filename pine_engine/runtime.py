"""Kjøremiljø for oversatt Pine: innebygde funksjoner, ``ta.*``-kallsteder og signalregistrering.

Pine-semantikk som følges:

- ``na`` er ``NaN`` for alle typer; aritmetikk med ``na`` gir ``na``, sammenligninger gir ``false``.
- Hvert kallsted for en ``ta.*``-funksjon har egen tilstand og egen historikk som oppdateres hver
  gang kallet faktisk kjøres (slik Pine gjør; kall inne i betingelser oppdateres bare når de kjøres).
- ``ta.ema`` starter på første verdi, ``ta.rma`` på SMA av de første ``length`` verdiene,
  ``ta.stdev`` bruker Pines nullingsregel (|avvik| ≤ 1e-10 → 0), pivoter: nyeste like verdi vinner.
- Historikk ut over 5000 lys kastes (Pines standard ``max_bars_back``).
"""
import math
import re
import time as _time
from datetime import datetime, timezone

NAN = float("nan")
MAX_BACK = 5000
MAX_ARRAY = 100_000
MAX_STRING = 40_960
COLORS = {
    "aqua": "#00BCD4", "black": "#363A45", "blue": "#2962FF", "fuchsia": "#E040FB", "gray": "#787B86",
    "green": "#4CAF50", "lime": "#00E676", "maroon": "#880E4F", "navy": "#311B92", "olive": "#808000",
    "orange": "#FF9800", "purple": "#9C27B0", "red": "#F23645", "silver": "#B2B5BE", "teal": "#089981",
    "white": "#FFFFFF", "yellow": "#FDD835",
}


class PineRuntimeError(Exception):
    """``runtime.error()`` i skriptet eller en ugyldig operasjon under kjøring."""


class PineTimeout(Exception):
    """Kjøringen brukte lengre tid eller flere løkkesteg enn tillatt."""


# ---------------------------------------------------------------- grunnleggende hjelpere

def isna(x):
    return x is None or (x.__class__ is float and x != x)


def t_(x):
    """Sannhetsverdi i Pine: ``na`` og 0 er usann."""
    if x is True:
        return True
    if x is False or x is None:
        return False
    try:
        return x == x and x != 0
    except Exception:
        return bool(x)


def add_(a, b):
    if a.__class__ is str or b.__class__ is str:
        if isna(a) or isna(b):
            return NAN
        out = str(a) + str(b)
        if len(out) > MAX_STRING:
            raise PineRuntimeError("teksten ble for lang")
        return out
    return a + b


def div_(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return NAN


def mod_(a, b):
    try:
        if a != a or b != b:
            return NAN
        return math.fmod(a, b)
    except (ZeroDivisionError, ValueError):
        return NAN


def hist_(series, k):
    """``x[k]`` for en variabel lagret som liste (siste element er nåværende verdi)."""
    if k != k:
        return NAN
    k = int(k)
    if k < 0:
        raise PineRuntimeError("historikkreferanse kan ikke være negativ")
    return series[-1 - k] if len(series) > k else NAN


def bhist_(array, i, k):
    """``close[k]`` for innebygde prisserier (indeksert på lysnummer)."""
    if k != k:
        return NAN
    k = int(k)
    if k < 0:
        raise PineRuntimeError("historikkreferanse kan ikke være negativ")
    return array[i - k] if i >= k else NAN


def hpush_(series, value, k):
    """Historikk av et sammensatt uttrykk (``ta.sma(close, 10)[1]``): lagrer verdien og leser bakover."""
    series.append(value)
    return hist_(series, k)


def nz_(x, y=0.0):
    return y if isna(x) else x


def na_(x):
    return isna(x)


def int_(x):
    if isna(x):
        return NAN
    try:
        return float(math.trunc(x))
    except (TypeError, ValueError, OverflowError):
        return NAN


def float_(x):
    if isna(x):
        return NAN
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def bool_(x):
    return t_(x)


def iff_(cond, a, b):
    return a if t_(cond) else b


# ---------------------------------------------------------------- math.*

def _nan_args(args):
    return any(isna(a) for a in args)


def m_max(*args):
    return NAN if _nan_args(args) else max(args)


def m_min(*args):
    return NAN if _nan_args(args) else min(args)


def m_abs(x):
    return NAN if isna(x) else abs(x)


def m_avg(*args):
    return NAN if _nan_args(args) else sum(args) / len(args)


def m_round(x, precision=NAN):
    """``math.round``: nærmeste heltall, midtverdier rundes opp (som Pine)."""
    if isna(x) or math.isinf(x):
        return NAN if isna(x) else x
    if isna(precision):
        return float(math.floor(x + 0.5))
    factor = 10 ** int(precision)
    return math.floor(x * factor + 0.5) / factor


def m_floor(x):
    return NAN if isna(x) or math.isinf(x) else float(math.floor(x))


def m_ceil(x):
    return NAN if isna(x) or math.isinf(x) else float(math.ceil(x))


def m_sqrt(x):
    return NAN if isna(x) or x < 0 else math.sqrt(x)


def m_pow(x, y):
    if isna(x) or isna(y):
        return NAN
    try:
        out = math.pow(x, y)
    except (OverflowError, ValueError):
        return NAN
    return out


def m_log(x):
    return NAN if isna(x) or x <= 0 else math.log(x)


def m_log10(x):
    return NAN if isna(x) or x <= 0 else math.log10(x)


def m_exp(x):
    if isna(x):
        return NAN
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def m_sign(x):
    return NAN if isna(x) else (1.0 if x > 0 else -1.0 if x < 0 else 0.0)


def _unary(fn):
    def call(x):
        if isna(x):
            return NAN
        try:
            return fn(x)
        except (ValueError, OverflowError):
            return NAN
    return call


m_sin, m_cos, m_tan = _unary(math.sin), _unary(math.cos), _unary(math.tan)
m_asin, m_acos, m_atan = _unary(math.asin), _unary(math.acos), _unary(math.atan)
m_todegrees, m_toradians = _unary(math.degrees), _unary(math.radians)


class Random:
    """``math.random`` med fast frø, så kjøringer kan gjentas."""

    def __init__(self):
        self.state = 12345

    def __call__(self, i, lo=0.0, hi=1.0, seed=NAN):
        self.state = (1103515245 * self.state + 12345) % 2 ** 31
        return lo + (hi - lo) * self.state / 2 ** 31


# ---------------------------------------------------------------- str.*

def _num_text(x, fmt=None):
    if x is True or x is False:
        return "true" if x else "false"
    if isna(x):
        return "NaN"
    if x.__class__ is str:
        return x
    if isinstance(x, PineArray):
        return "[" + ", ".join(_num_text(v) for v in x.items) + "]"
    if fmt.__class__ is str and fmt:
        if fmt == "format.percent":
            return f"{x:.2f}%"
        if fmt.startswith("format."):
            return f"{x:.2f}"
        if "#" in fmt or "0" in fmt:
            decimals = len(fmt.split(".")[1]) if "." in fmt else 0
            text = f"{x:.{decimals}f}"
            required = fmt.split(".")[1].count("0") if "." in fmt else 0
            if "." in text and "#" in fmt:
                whole, frac = text.split(".")
                frac = frac.rstrip("0").ljust(required, "0")
                text = whole + ("." + frac if frac else "")
            return text
    if float(x).is_integer() and abs(x) < 1e15:
        return str(int(x))
    return f"{x:.8f}".rstrip("0").rstrip(".")


def s_tostring(x, fmt=None):
    return _num_text(x, fmt)


def s_format(fmt, *args):
    if isna(fmt):
        return NAN
    out = str(fmt)
    for k, value in enumerate(args):
        text = _num_text(value)
        out = out.replace("{" + str(k) + "}", text)
        start = out.find("{" + str(k) + ",")
        while start >= 0:
            end = out.find("}", start)
            if end < 0:
                break
            out = out[:start] + text + out[end + 1:]
            start = out.find("{" + str(k) + ",")
    if len(out) > MAX_STRING:
        raise PineRuntimeError("teksten ble for lang")
    return out


def s_length(s):
    return NAN if isna(s) else float(len(str(s)))


def s_contains(s, sub):
    return False if isna(s) or isna(sub) else str(sub) in str(s)


def s_startswith(s, sub):
    return False if isna(s) or isna(sub) else str(s).startswith(str(sub))


def s_endswith(s, sub):
    return False if isna(s) or isna(sub) else str(s).endswith(str(sub))


def s_pos(s, sub):
    if isna(s) or isna(sub):
        return NAN
    k = str(s).find(str(sub))
    return NAN if k < 0 else float(k)


def s_replace_all(s, old, new):
    return NAN if isna(s) else str(s).replace(str(old), str(new))


def s_replace(s, old, new, occurrence=0.0):
    if isna(s):
        return NAN
    text, old, new = str(s), str(old), str(new)
    k, count = -1, int(occurrence)
    for _ in range(count + 1):
        k = text.find(old, k + 1)
        if k < 0:
            return text
    return text[:k] + new + text[k + len(old):]


def s_lower(s):
    return NAN if isna(s) else str(s).lower()


def s_upper(s):
    return NAN if isna(s) else str(s).upper()


def s_trim(s):
    return NAN if isna(s) else str(s).strip()


def s_substring(s, begin, end=NAN):
    if isna(s):
        return NAN
    return str(s)[int(begin):] if isna(end) else str(s)[int(begin):int(end)]


def s_tonumber(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return NAN


def s_split(s, sep):
    return PineArray("string", [] if isna(s) else str(s).split(str(sep)) if sep else list(str(s)))


def s_repeat(s, n, sep=""):
    out = str(sep).join([str(s)] * max(0, int(n)))
    if len(out) > MAX_STRING:
        raise PineRuntimeError("teksten ble for lang")
    return out


def s_format_time(t, fmt="yyyy-MM-dd'T'HH:mm:ssZ", tz=None):
    if isna(t):
        return NAN
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- farger

def c_new(color, transp=0.0):
    return color


def c_rgb(r, g, b, transp=0.0):
    try:
        return "#%02X%02X%02X" % (int(r), int(g), int(b))
    except (TypeError, ValueError):
        return NAN


def _channel(color, k):
    try:
        return float(int(str(color)[1 + 2 * k:3 + 2 * k], 16))
    except (TypeError, ValueError):
        return NAN


def c_r(color):
    return _channel(color, 0)


def c_g(color):
    return _channel(color, 1)


def c_b(color):
    return _channel(color, 2)


def c_t(color):
    return 0.0


def c_from_gradient(value, bottom, top, bottom_color, top_color):
    return bottom_color if not t_(value >= (bottom + top) / 2) else top_color


# ---------------------------------------------------------------- tid

def timestamp_(*args):
    """``timestamp(år, måned, dag, time, minutt, sekund)``, med valgfri tidssone først, eller datotekst."""
    args = list(args)
    if args and args[0].__class__ is str and len(args) == 1:
        text = args[0].strip().replace("Z", "").replace("+0000", "").strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%d %b %Y %H:%M",
                    "%d %b %Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp() * 1000
            except ValueError:
                continue
        return NAN
    if args and args[0].__class__ is str:
        args = args[1:]
    if not args or any(isna(a) for a in args):
        return NAN
    year, month, day, hour, minute, second = ([int(a) for a in args] + [1, 1, 0, 0, 0])[:6] if len(args) < 3 \
        else ([int(a) for a in args] + [0, 0, 0])[:6]
    year = max(1970, min(year, 9999))
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    try:
        stamp = datetime(min(year, 9999), month, 1, tzinfo=timezone.utc).timestamp()
    except (ValueError, OverflowError):
        return NAN
    return (stamp + (day - 1) * 86400 + hour * 3600 + minute * 60 + second) * 1000.0


_ZONES = {}


def _zone(tz):
    """Pine-tidssone: «UTC», «GMT+2», «UTC-5» eller IANA-navn som «America/New_York»."""
    if tz is None or isna(tz) or str(tz).strip().upper() in ("", "UTC", "GMT", "ETC/UTC", "SYMINFO.TIMEZONE"):
        return timezone.utc
    text = str(tz).strip()
    if text not in _ZONES:
        match = re.fullmatch(r"(?:UTC|GMT)\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?", text, re.IGNORECASE)
        if match:
            sign = 1 if match.group(1) == "+" else -1
            minutes = int(match.group(2)) * 60 + int(match.group(3) or 0)
            from datetime import timedelta
            _ZONES[text] = timezone(sign * timedelta(minutes=minutes))
        else:
            try:
                from zoneinfo import ZoneInfo
                _ZONES[text] = ZoneInfo(text)
            except Exception:
                raise PineRuntimeError(f"ukjent tidssone «{text}»") from None
    return _ZONES[text]


def _dt(t, tz=None):
    return datetime.fromtimestamp(t / 1000, tz=_zone(tz))


def year_(t, tz=None):
    return NAN if isna(t) else float(_dt(t, tz).year)


def month_(t, tz=None):
    return NAN if isna(t) else float(_dt(t, tz).month)


def dayofmonth_(t, tz=None):
    return NAN if isna(t) else float(_dt(t, tz).day)


def dayofweek_(t, tz=None):
    return NAN if isna(t) else float((_dt(t, tz).weekday() + 1) % 7 + 1)


def hour_(t, tz=None):
    return NAN if isna(t) else float(_dt(t, tz).hour)


def minute_(t, tz=None):
    return NAN if isna(t) else float(_dt(t, tz).minute)


def second_(t, tz=None):
    return NAN if isna(t) else float(_dt(t, tz).second)


def weekofyear_(t, tz=None):
    return NAN if isna(t) else float(_dt(t, tz).isocalendar()[1])


def in_session(t, session):
    """Om tidspunktet ``t`` (ms, UTC) er innenfor økten «HHMM-HHMM[:dager]»."""
    if isna(t):
        return False
    text = str(session or "")
    if not text or text in ("24x7", "0000-0000"):
        return True
    spec, _, days = text.partition(":")
    stamp = _dt(t)
    if days and str((stamp.weekday() + 1) % 7 + 1) not in days:
        return False
    try:
        start, end = spec.split("-")
        now = stamp.hour * 60 + stamp.minute
        a = int(start[:2]) * 60 + int(start[2:4])
        b = int(end[:2]) * 60 + int(end[2:4])
    except (ValueError, IndexError):
        return True
    return a <= now < b if a < b else (now >= a or now < b)


# ---------------------------------------------------------------- arrays

class PineArray:
    __slots__ = ("typ", "items")

    def __init__(self, typ, items):
        if len(items) > MAX_ARRAY:
            raise PineRuntimeError("array er for stor (maks 100 000 elementer)")
        self.typ, self.items = typ, list(items)

    def __repr__(self):
        return f"array<{self.typ}>({len(self.items)})"


def _arr(a):
    if not isinstance(a, PineArray):
        raise PineRuntimeError("forventet en array (er den na?)")
    return a


def _idx(a, k):
    if isna(k):
        raise PineRuntimeError("array-indeks er na")
    k = int(k)
    if k < 0:
        k += len(a.items)
    if not 0 <= k < len(a.items):
        raise PineRuntimeError(f"array-indeks {k} er utenfor [0, {len(a.items) - 1}]")
    return k


def a_new(typ="float"):
    def make(size=0.0, initial=NAN):
        size = 0 if isna(size) else int(size)
        if size > MAX_ARRAY:
            raise PineRuntimeError("array er for stor (maks 100 000 elementer)")
        return PineArray(typ, [initial] * size)
    return make


def a_from(*items):
    return PineArray("any", list(items))


def a_push(a, x):
    a = _arr(a)
    if len(a.items) >= MAX_ARRAY:
        raise PineRuntimeError("array er for stor (maks 100 000 elementer)")
    a.items.append(x)


def a_unshift(a, x):
    a = _arr(a)
    if len(a.items) >= MAX_ARRAY:
        raise PineRuntimeError("array er for stor (maks 100 000 elementer)")
    a.items.insert(0, x)


def a_pop(a):
    a = _arr(a)
    if not a.items:
        raise PineRuntimeError("kan ikke ta ut fra tom array")
    return a.items.pop()


def a_shift(a):
    a = _arr(a)
    if not a.items:
        raise PineRuntimeError("kan ikke ta ut fra tom array")
    return a.items.pop(0)


def a_get(a, k):
    a = _arr(a)
    return a.items[_idx(a, k)]


def a_set(a, k, x):
    a = _arr(a)
    a.items[_idx(a, k)] = x


def a_size(a):
    return NAN if isna(a) else float(len(_arr(a).items))


def a_clear(a):
    _arr(a).items.clear()


def a_insert(a, k, x):
    a = _arr(a)
    if len(a.items) >= MAX_ARRAY:
        raise PineRuntimeError("array er for stor (maks 100 000 elementer)")
    a.items.insert(int(k), x)


def a_remove(a, k):
    a = _arr(a)
    return a.items.pop(_idx(a, k))


def a_includes(a, x):
    return x in _arr(a).items


def a_indexof(a, x):
    items = _arr(a).items
    return float(items.index(x)) if x in items else -1.0


def a_lastindexof(a, x):
    items = _arr(a).items
    for k in range(len(items) - 1, -1, -1):
        if items[k] == x:
            return float(k)
    return -1.0


def _finite(a):
    return [x for x in _arr(a).items if not isna(x)]


def a_sum(a):
    values = _finite(a)
    return float(sum(values)) if values else NAN


def a_avg(a):
    values = _finite(a)
    return sum(values) / len(values) if values else NAN


def a_max(a, nth=0.0):
    values = sorted(_finite(a), reverse=True)
    return values[int(nth)] if len(values) > int(nth) else NAN


def a_min(a, nth=0.0):
    values = sorted(_finite(a))
    return values[int(nth)] if len(values) > int(nth) else NAN


def a_median(a):
    values = sorted(_finite(a))
    if not values:
        return NAN
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def a_variance(a, biased=True):
    values = _finite(a)
    if len(values) < 2:
        return NAN
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / (len(values) if t_(biased) else len(values) - 1)


def a_stdev(a, biased=True):
    v = a_variance(a, biased)
    return NAN if isna(v) else math.sqrt(v)


def a_sort(a, order="order.ascending"):
    _arr(a).items.sort(key=lambda x: (isna(x), x if not isna(x) else 0), reverse=order == "order.descending")


def a_reverse(a):
    _arr(a).items.reverse()


def a_copy(a):
    a = _arr(a)
    return PineArray(a.typ, a.items)


def a_slice(a, start, end):
    a = _arr(a)
    return PineArray(a.typ, a.items[int(start):int(end)])


def a_concat(a, b):
    a = _arr(a)
    a.items.extend(_arr(b).items)
    if len(a.items) > MAX_ARRAY:
        raise PineRuntimeError("array er for stor (maks 100 000 elementer)")
    return a


def a_first(a):
    a = _arr(a)
    if not a.items:
        raise PineRuntimeError("tom array")
    return a.items[0]


def a_last(a):
    a = _arr(a)
    if not a.items:
        raise PineRuntimeError("tom array")
    return a.items[-1]


def a_fill(a, x, start=0.0, end=NAN):
    a = _arr(a)
    stop = len(a.items) if isna(end) else int(end)
    for k in range(int(start), stop):
        a.items[k] = x


def a_join(a, sep=","):
    return str(sep).join(_num_text(x) for x in _arr(a).items)


def a_range(a):
    values = _finite(a)
    return max(values) - min(values) if values else NAN


ARRAY_METHODS = {
    "push": a_push, "unshift": a_unshift, "pop": a_pop, "shift": a_shift, "get": a_get, "set": a_set,
    "size": a_size, "clear": a_clear, "insert": a_insert, "remove": a_remove, "includes": a_includes,
    "indexof": a_indexof, "lastindexof": a_lastindexof, "sum": a_sum, "avg": a_avg, "max": a_max, "min": a_min,
    "median": a_median, "variance": a_variance, "stdev": a_stdev, "sort": a_sort, "reverse": a_reverse,
    "copy": a_copy, "slice": a_slice, "concat": a_concat, "first": a_first, "last": a_last, "fill": a_fill,
    "join": a_join, "range": a_range,
}


class Drawing(float):
    """Id for label/line/box/table: et vanlig tall (ikke na), slik at ``na(lbl)`` virker som i Pine."""


DRAWING_METHODS = {"delete", "copy", "get_x", "get_y", "get_x1", "get_x2", "get_y1", "get_y2", "get_price",
                   "get_text", "get_left", "get_right", "get_top", "get_bottom", "get_line1", "get_line2"}


def method_(obj, name, *args):
    """``obj.metode(...)`` på brukerobjekter: arrays får array-funksjonen, tegneobjekter gjør ingenting."""
    if isinstance(obj, PineArray):
        fn = ARRAY_METHODS.get(name)
        if fn is None:
            raise PineRuntimeError(f"array-metoden «{name}» støttes ikke")
        return fn(obj, *args)
    if isinstance(obj, PineMap):
        fn = MAP_METHODS.get(name)
        if fn is None:
            raise PineRuntimeError(f"map-metoden «{name}» støttes ikke")
        return fn(obj, *args)
    if obj.__class__ is PineObject:
        if name == "copy":
            return copy_obj_(obj)
        if name == "delete":
            return NAN
        raise PineRuntimeError(f"typen {obj.kind} har ingen metode «{name}»")
    if isinstance(obj, Drawing) or isna(obj):
        return "" if name == "get_text" else NAN
    if obj.__class__ is str:
        text_methods = {"length": s_length, "contains": s_contains, "upper": s_upper, "lower": s_lower,
                        "startswith": s_startswith, "endswith": s_endswith, "tonumber": s_tonumber}
        if name in text_methods:
            return text_methods[name](obj, *args)
    raise PineRuntimeError(f"metoden «{name}» støttes ikke for denne verdien")


# ---------------------------------------------------------------- serier og kallsteder

class Registry:
    """Alle historikklister i en kjøring; de trimmes til ``MAX_BACK`` lys med jevne mellomrom."""

    def __init__(self):
        self.lists = []

    def series(self):
        out = []
        self.lists.append(out)
        return out

    def trim(self):
        for values in self.lists:
            if len(values) > MAX_BACK + 1000:
                del values[:len(values) - MAX_BACK - 1]


def _window(buf, n):
    return buf[-n:] if len(buf) >= n else None


def n_(length):
    """Lengdeargument som heltall ≥ 1 (Pine stopper skriptet ved ugyldig lengde)."""
    if isna(length):
        raise PineRuntimeError("lengden er na")
    n = int(length)
    if n < 1:
        raise PineRuntimeError(f"lengden må være minst 1 (fikk {n})")
    if n > MAX_BACK:
        raise PineRuntimeError(f"lengden kan være høyst {MAX_BACK}")
    return n


class SMA:
    def __init__(self, rt):
        self.buf, self.n, self.total, self.nans, self.count = rt.reg.series(), None, 0.0, 0, 0

    def __call__(self, i, src, length):
        buf = self.buf
        buf.append(src)
        n = n_(length)
        if len(buf) < n:
            return NAN
        self.count += 1
        if n != self.n or self.count % 1000 == 0:
            self.n = n
            window = buf[-n:]
            self.nans = sum(1 for x in window if x != x)
            self.total = math.fsum(x for x in window if x == x)
        else:
            if src == src:
                self.total += src
            else:
                self.nans += 1
            if len(buf) > n:
                old = buf[-n - 1]
                if old == old:
                    self.total -= old
                else:
                    self.nans -= 1
        return NAN if self.nans else self.total / n


class SUM(SMA):
    def __call__(self, i, src, length):
        value = SMA.__call__(self, i, src, length)
        return NAN if isna(value) else self.total


class EMA:
    def __init__(self, rt):
        self.prev = NAN

    def __call__(self, i, src, length):
        if src != src:
            return self.prev
        prev = self.prev
        if prev != prev:
            self.prev = src
        else:
            alpha = 2.0 / (length + 1)
            self.prev = alpha * src + (1 - alpha) * prev
        return self.prev


class RMA:
    def __init__(self, rt):
        self.prev, self.buf = NAN, rt.reg.series()

    def __call__(self, i, src, length):
        self.buf.append(src)
        n = n_(length)
        prev = self.prev
        if prev != prev:
            window = _window(self.buf, n)
            if window is not None and all(x == x for x in window):
                self.prev = sum(window) / n
            return self.prev
        alpha = 1.0 / n
        self.prev = alpha * src + (1 - alpha) * prev
        return self.prev


class WMA:
    def __init__(self, rt):
        self.buf = rt.reg.series()

    def __call__(self, i, src, length):
        self.buf.append(src)
        n = n_(length)
        window = _window(self.buf, n)
        if window is None:
            return NAN
        total = 0.0
        for k, x in enumerate(window, 1):
            total += k * x
        return total / (n * (n + 1) / 2)


class VWMA:
    def __init__(self, rt):
        self.num, self.den, self.rt = SMA(rt), SMA(rt), rt

    def __call__(self, i, src, length):
        volume = self.rt.V[i]
        return div_(self.num(i, src * volume, length), self.den(i, volume, length))


class HMA:
    def __init__(self, rt):
        self.half, self.full, self.out = WMA(rt), WMA(rt), WMA(rt)

    def __call__(self, i, src, length):
        n = n_(length)
        half = self.half(i, src, n // 2)
        full = self.full(i, src, n)
        return self.out(i, 2 * half - full, int(math.floor(math.sqrt(n))))


class SWMA:
    def __init__(self, rt):
        self.buf = rt.reg.series()

    def __call__(self, i, src):
        self.buf.append(src)
        window = _window(self.buf, 4)
        if window is None:
            return NAN
        return window[0] / 6 + window[1] * 2 / 6 + window[2] * 2 / 6 + window[3] / 6


class ALMA:
    def __init__(self, rt):
        self.buf, self.key, self.weights = rt.reg.series(), None, None

    def __call__(self, i, src, length, offset=0.85, sigma=6.0, floor=False):
        self.buf.append(src)
        n = n_(length)
        key = (n, offset, sigma, bool(floor))
        if key != self.key:
            m = math.floor(offset * (n - 1)) if t_(floor) else offset * (n - 1)
            s = n / sigma
            weights = [math.exp(-((k - m) ** 2) / (2 * s * s)) for k in range(n)]
            total = sum(weights)
            self.key, self.weights = key, [w / total for w in weights]
        window = _window(self.buf, n)
        if window is None:
            return NAN
        out = 0.0
        for w, x in zip(self.weights, window):
            out += w * x
        return out


def _pine_sum(a, b):
    res = a + b
    return 0.0 if abs(res) <= 1e-10 else res


class STDEV:
    def __init__(self, rt):
        self.buf = rt.reg.series()

    def __call__(self, i, src, length, biased=True):
        self.buf.append(src)
        n = n_(length)
        window = _window(self.buf, n)
        if window is None or any(x != x for x in window):
            return NAN
        mean = sum(window) / n
        total = 0.0
        for x in window:
            d = _pine_sum(x, -mean)
            total += d * d
        if not t_(biased):
            return math.sqrt(total / (n - 1)) if n > 1 else NAN
        return math.sqrt(total / n)


class VARIANCE(STDEV):
    def __call__(self, i, src, length, biased=True):
        value = STDEV.__call__(self, i, src, length, biased)
        return value * value


class DEV:
    def __init__(self, rt):
        self.buf = rt.reg.series()

    def __call__(self, i, src, length):
        self.buf.append(src)
        n = n_(length)
        window = _window(self.buf, n)
        if window is None or any(x != x for x in window):
            return NAN
        mean = sum(window) / n
        return sum(abs(x - mean) for x in window) / n


class HIGHEST:
    sign = 1

    def __init__(self, rt):
        self.buf = rt.reg.series()

    def __call__(self, i, src, length):
        self.buf.append(src)
        n = n_(length)
        window = _window(self.buf, n)
        if window is None or any(x != x for x in window):
            return NAN
        return max(window) if self.sign > 0 else min(window)


class LOWEST(HIGHEST):
    sign = -1


class HIGHESTBARS:
    sign = 1

    def __init__(self, rt):
        self.buf = rt.reg.series()

    def __call__(self, i, src, length):
        self.buf.append(src)
        n = n_(length)
        window = _window(self.buf, n)
        if window is None or any(x != x for x in window):
            return NAN
        best, offset = window[-1], 0
        for k in range(1, n):
            x = window[-1 - k]
            if (x > best) if self.sign > 0 else (x < best):
                best, offset = x, k
        return -float(offset)


class LOWESTBARS(HIGHESTBARS):
    sign = -1


class RANGE:
    def __init__(self, rt):
        self.hi, self.lo = HIGHEST(rt), LOWEST(rt)

    def __call__(self, i, src, length):
        return self.hi(i, src, length) - self.lo(i, src, length)


class CHANGE:
    def __init__(self, rt):
        self.buf = rt.reg.series()

    def __call__(self, i, src, length=1.0):
        self.buf.append(src)
        return src - hist_(self.buf, length)


class ROC(CHANGE):
    def __call__(self, i, src, length):
        self.buf.append(src)
        prev = hist_(self.buf, length)
        return div_(100 * (src - prev), prev)


class RSI:
    def __init__(self, rt):
        self.prev, self.up, self.down = NAN, RMA(rt), RMA(rt)

    def __call__(self, i, src, length):
        prev, self.prev = self.prev, src
        u = NAN if prev != prev or src != src else max(src - prev, 0.0)
        d = NAN if prev != prev or src != src else max(prev - src, 0.0)
        up, down = self.up(i, u, length), self.down(i, d, length)
        if up != up or down != down:
            return NAN
        if down == 0:
            return 100.0
        if up == 0:
            return 0.0
        return 100 - 100 / (1 + up / down)


def tr_(rt, i, handle_na=False):
    h, l = rt.H[i], rt.L[i]
    if i == 0:
        return h - l if t_(handle_na) else NAN
    c = rt.C[i - 1]
    return max(h - l, abs(h - c), abs(l - c))


class ATR:
    def __init__(self, rt):
        self.rt, self.rma = rt, RMA(rt)

    def __call__(self, i, length):
        return self.rma(i, tr_(self.rt, i, True), length)


class CROSSOVER:
    kind = "over"

    def __init__(self, rt):
        self.pa = self.pb = NAN

    def __call__(self, i, a, b):
        pa, pb = self.pa, self.pb
        self.pa, self.pb = a, b
        if self.kind == "over":
            return a > b and pa <= pb
        if self.kind == "under":
            return a < b and pa >= pb
        return (a > b and pa <= pb) or (a < b and pa >= pb)


class CROSSUNDER(CROSSOVER):
    kind = "under"


class CROSS(CROSSOVER):
    kind = "cross"


class PIVOT:
    kind = "high"

    def __init__(self, rt):
        self.buf = rt.reg.series()

    def __call__(self, i, src, left, right):
        self.buf.append(src)
        left, right = int(left), int(right)
        window = _window(self.buf, left + right + 1)
        if window is None or any(x != x for x in window):
            return NAN
        c = window[left]
        if self.kind == "high":
            ok = all(c >= x for x in window[:left]) and all(c > x for x in window[left + 1:])
        else:
            ok = all(c <= x for x in window[:left]) and all(c < x for x in window[left + 1:])
        return c if ok else NAN


class PIVOTLOW(PIVOT):
    kind = "low"


class VALUEWHEN:
    def __init__(self, rt):
        self.values = rt.reg.series()

    def __call__(self, i, cond, src, occurrence=0.0):
        if t_(cond):
            self.values.append(src)
        return hist_(self.values, occurrence)


class BARSSINCE:
    def __init__(self, rt):
        self.count = NAN

    def __call__(self, i, cond):
        if t_(cond):
            self.count = 0.0
        elif self.count == self.count:
            self.count += 1
        return self.count


class CUM:
    def __init__(self, rt):
        self.total = 0.0

    def __call__(self, i, src):
        if src == src:
            self.total += src
        return self.total


class FIXNAN:
    def __init__(self, rt):
        self.last = NAN

    def __call__(self, i, x):
        if not isna(x):
            self.last = x
        return self.last


class PERCENTRANK:
    def __init__(self, rt):
        self.buf = rt.reg.series()

    def __call__(self, i, src, length):
        self.buf.append(src)
        n = n_(length)
        window = _window(self.buf, n + 1)
        if window is None or any(x != x for x in window):
            return NAN
        return sum(1 for x in window[:-1] if x <= src) / n * 100


class LINREG:
    def __init__(self, rt):
        self.buf = rt.reg.series()

    def __call__(self, i, src, length, offset=0.0):
        self.buf.append(src)
        n = n_(length)
        window = _window(self.buf, n)
        if window is None or any(x != x for x in window):
            return NAN
        mean_x = (n - 1) / 2
        mean_y = sum(window) / n
        sxy = sum((k - mean_x) * (y - mean_y) for k, y in enumerate(window))
        sxx = sum((k - mean_x) ** 2 for k in range(n))
        slope = sxy / sxx if sxx else 0.0
        intercept = mean_y - slope * mean_x
        return intercept + slope * (n - 1 - int(offset))


class CORRELATION:
    def __init__(self, rt):
        self.a, self.b = rt.reg.series(), rt.reg.series()

    def __call__(self, i, a, b, length):
        self.a.append(a)
        self.b.append(b)
        n = n_(length)
        wa, wb = _window(self.a, n), _window(self.b, n)
        if wa is None or any(x != x for x in wa) or any(x != x for x in wb):
            return NAN
        ma, mb = sum(wa) / n, sum(wb) / n
        cov = sum((x - ma) * (y - mb) for x, y in zip(wa, wb))
        va = sum((x - ma) ** 2 for x in wa)
        vb = sum((y - mb) ** 2 for y in wb)
        return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else NAN


class MEDIAN:
    def __init__(self, rt):
        self.buf = rt.reg.series()

    def __call__(self, i, src, length):
        self.buf.append(src)
        window = _window(self.buf, n_(length))
        if window is None or any(x != x for x in window):
            return NAN
        values = sorted(window)
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


class PERCENTILE:
    linear = True

    def __init__(self, rt):
        self.buf = rt.reg.series()

    def __call__(self, i, src, length, percentage):
        self.buf.append(src)
        window = _window(self.buf, n_(length))
        if window is None or any(x != x for x in window):
            return NAN
        values = sorted(window)
        if not self.linear:
            k = max(0, math.ceil(percentage / 100 * len(values)) - 1)
            return values[min(k, len(values) - 1)]
        pos = percentage / 100 * (len(values) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)


class PERCENTILE_NEAREST(PERCENTILE):
    linear = False


class RISING:
    sign = 1

    def __init__(self, rt):
        self.buf = rt.reg.series()

    def __call__(self, i, src, length):
        self.buf.append(src)
        n = n_(length)
        window = _window(self.buf, n + 1)
        if window is None:
            return False
        if self.sign > 0:
            return all(src > x for x in window[:-1])
        return all(src < x for x in window[:-1])


class FALLING(RISING):
    sign = -1


class ALLMAX:
    sign = 1

    def __init__(self, rt):
        self.best = NAN

    def __call__(self, i, src):
        if src == src and (self.best != self.best or (src > self.best if self.sign > 0 else src < self.best)):
            self.best = src
        return self.best


class ALLMIN(ALLMAX):
    sign = -1


class VWAP:
    """Daglig forankret VWAP (UTC-døgn), med valgfrie standardavviksbånd."""

    def __init__(self, rt):
        self.rt, self.day, self.pv, self.vol, self.pv2 = rt, None, 0.0, 0.0, 0.0

    def __call__(self, i, src=None, anchor=None, stdev_mult=NAN):
        rt = self.rt
        src = rt.HLC3[i] if src is None else src
        day = rt.T[i] // 86_400_000
        if day != self.day:
            self.day, self.pv, self.vol, self.pv2 = day, 0.0, 0.0, 0.0
        volume = rt.V[i]
        if src == src:
            self.pv += src * volume
            self.vol += volume
            self.pv2 += src * src * volume
        vwap = self.pv / self.vol if self.vol > 0 else NAN
        if isna(stdev_mult):
            return vwap
        dev = math.sqrt(max(self.pv2 / self.vol - vwap * vwap, 0.0)) if self.vol > 0 else NAN
        return vwap, vwap + stdev_mult * dev, vwap - stdev_mult * dev


class OBV:
    def __init__(self, rt):
        self.rt, self.total = rt, 0.0

    def __call__(self, i):
        rt = self.rt
        if i > 0:
            change = rt.C[i] - rt.C[i - 1]
            self.total += rt.V[i] * (1 if change > 0 else -1 if change < 0 else 0)
        return self.total


class ACCDIST:
    def __init__(self, rt):
        self.rt, self.total = rt, 0.0

    def __call__(self, i):
        rt = self.rt
        h, l, c, v = rt.H[i], rt.L[i], rt.C[i], rt.V[i]
        if h != l:
            self.total += ((c - l) - (h - c)) / (h - l) * v
        return self.total


SITES = {
    "ta.sma": SMA, "ta.ema": EMA, "ta.rma": RMA, "ta.wma": WMA, "ta.vwma": VWMA, "ta.hma": HMA, "ta.swma": SWMA,
    "ta.alma": ALMA, "ta.stdev": STDEV, "ta.variance": VARIANCE, "ta.dev": DEV, "ta.highest": HIGHEST,
    "ta.lowest": LOWEST, "ta.highestbars": HIGHESTBARS, "ta.lowestbars": LOWESTBARS, "ta.range": RANGE,
    "ta.change": CHANGE, "ta.mom": CHANGE, "ta.roc": ROC, "ta.rsi": RSI, "ta.atr": ATR, "ta.crossover": CROSSOVER,
    "ta.crossunder": CROSSUNDER, "ta.cross": CROSS, "ta.pivothigh": PIVOT, "ta.pivotlow": PIVOTLOW,
    "ta.valuewhen": VALUEWHEN, "ta.barssince": BARSSINCE, "ta.cum": CUM, "math.sum": SUM, "fixnan": FIXNAN,
    "ta.percentrank": PERCENTRANK, "ta.linreg": LINREG, "ta.correlation": CORRELATION, "ta.median": MEDIAN,
    "ta.percentile_linear_interpolation": PERCENTILE, "ta.percentile_nearest_rank": PERCENTILE_NEAREST,
    "ta.rising": RISING, "ta.falling": FALLING, "ta.max": ALLMAX, "ta.min": ALLMIN, "ta.vwap": VWAP,
    "ta.obv": OBV, "ta.accdist": ACCDIST, "math.random": Random,
}


# ---------------------------------------------------------------- signaler og tegninger

class Events:
    """Hendelser fra ``plotshape``/``plotchar``/``plotarrow``/``alertcondition``/``alert``/``label.new``."""

    def __init__(self, enabled=True, capture_plots=False):
        self.enabled = enabled
        self.capture_plots = bool(capture_plots)
        self.hits = {}          # (kind, site, tekst) -> [lysnummer]
        self.samples = {}       # eksempel på alert()-melding per kallsted
        self.series_values = {}  # valgfritt: (linje, tittel) -> [(lysnummer, verdi)]

    def add(self, kind, site, text, i):
        if not self.enabled:
            return
        key = (kind, site, text)
        hits = self.hits.get(key)
        if hits is None:
            if sum(1 for k in self.hits if k[1] == site and k[0] == kind) >= 25:
                key = (kind, site, f"{kind} #{site + 1} (andre tekster)")
                hits = self.hits.get(key)
                if hits is None:
                    self.hits[key] = [i]
                    return
            else:
                self.hits[key] = [i]
                return
        if hits[-1] != i:
            hits.append(i)

    def add_alert(self, site, message, i):
        if not self.enabled:
            return
        key = ("alert", site, f"alert() #{site + 1}")
        if key not in self.hits:
            self.hits[key] = [i]
            self.samples[key] = message[:300]
        elif self.hits[key][-1] != i:
            self.hits[key].append(i)


class Drawings:
    def __init__(self):
        self.next = 0

    def new(self):
        self.next += 1
        return Drawing(self.next)


class Clock:
    """Tidsfrist og løkkegrense for én kjøring."""

    def __init__(self, deadline, loop_limit=50_000_000):
        self.deadline, self.loops, self.limit = deadline, 0, loop_limit

    def loop(self):
        self.loops += 1
        if self.loops > self.limit:
            raise PineTimeout("for mange løkkesteg (over 50 millioner)")
        if self.loops % 100_000 == 0:
            self.check()

    def check(self):
        if self.deadline is not None and _time.monotonic() > self.deadline:
            raise PineTimeout("kjøringen tok for lang tid")


def runtime_error(message=""):
    raise PineRuntimeError(f"runtime.error i skriptet: {message}")


# ---------------------------------------------------------------- hjelpere brukt av oversatt kode

LABEL_NUMBER = None


def _signal_text(value):
    """Kort, stabil tekst for en hendelse: tall byttes ut med #, så «SL 81234.5» blir én nøkkel."""
    import re
    global LABEL_NUMBER
    if LABEL_NUMBER is None:
        LABEL_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)*%?")
    text = _num_text(value) if value.__class__ is not str else value
    text = LABEL_NUMBER.sub("#", text).strip()
    return text[:80]


def out_signal(events, kind, site, i, series, title, text, extras=()):
    if t_(series):
        name = title if title.__class__ is str and title else text if text.__class__ is str and text else ""
        events.add(kind, site, _signal_text(name) or f"{kind} #{site + 1}", i)
    return NAN


def out_plot(events, site, i, series, title, extras=()):
    """Valgfri serieinnsamling; endrer aldri signalenes kallsted eller plot-returverdien."""
    if events.capture_plots:
        key = (site, title if isinstance(title, str) else "")
        events.series_values.setdefault(key, []).append((i, series))
    return NAN


def out_arrow(events, site, i, series, title, extras=()):
    if not isna(series) and series:
        name = title if title.__class__ is str and title else f"plotarrow #{site + 1}"
        events.add("plotarrow", site, _signal_text(name) + (" ↑" if series > 0 else " ↓"), i)
    return NAN


def out_alert(events, site, i, message, extras=()):
    events.add_alert(site, _num_text(message) if message.__class__ is not str else message, i)
    return NAN


def out_label(events, new, site, i, x, y, text, extras=()):
    events.add("label.new", site, _signal_text(text) or f"label #{site + 1}", i)
    return new()


def noop_(*args):
    return NAN


def draw_(new, *args):
    return new()


def ident_(x):
    return x


def unpack_(value, count):
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise PineRuntimeError(f"forventet {count} verdier å pakke ut")
    return value


def items_(arr):
    return list(_arr(arr).items)


def eitems_(arr):
    return [(float(k), v) for k, v in enumerate(_arr(arr).items)]


def ticker_ha(symbol):
    return f"HA:{symbol}"


def ticker_new(prefix, ticker, session=None, adjustment=None):
    return f"{prefix}:{ticker}" if prefix and not isna(prefix) else str(ticker)


def ticker_modify(tickerid, session=None, adjustment=None):
    return tickerid


def tf_minutes(tf, current):
    """Pine-tidsramme («5», «60», «240», «D», «1D», «W») → minutter; tom tekst = gjeldende."""
    if tf is None or isna(tf) or tf == "":
        return current
    text = str(tf).strip().upper()
    if text in ("D", "1D"):
        return 1440
    if text in ("W", "1W"):
        return 10080
    if text.endswith("D") and text[:-1].isdigit():
        return int(text[:-1]) * 1440
    if text.endswith("W") and text[:-1].isdigit():
        return int(text[:-1]) * 10080
    if text.isdigit():
        return int(text)
    if text.endswith("H") and text[:-1].isdigit():
        return int(text[:-1]) * 60
    if text.endswith("M") and (text[:-1].isdigit() or text == "M"):
        return int(text[:-1] or 1) * MONTH          # kalendermåneder (egen kode, se is_months)
    if text.endswith("S"):
        raise PineRuntimeError(f"tidsrammen «{tf}» (sekunder) støttes ikke")
    raise PineRuntimeError(f"ukjent tidsramme «{tf}»")


MONTH = 43_800          # «minutter» som betyr én kalendermåned (ikke et fast antall minutter)


def is_months(minutes):
    return minutes >= MONTH and minutes % MONTH == 0


def tf_period(minutes):
    if is_months(minutes):
        return "M" if minutes == MONTH else f"{minutes // MONTH}M"
    if minutes % 10080 == 0:
        return "W" if minutes == 10080 else f"{minutes // 10080}W"
    if minutes % 1440 == 0:
        return "D" if minutes == 1440 else f"{minutes // 1440}D"
    return str(minutes)


def tf_seconds_of(tf):
    text = "" if tf is None or isna(tf) else str(tf).strip().upper()
    if text.endswith("M") and (text[:-1].isdigit() or text == "M"):
        return float(int(text[:-1] or 1) * 2_628_003)        # Pine: 1 måned = 2 628 003 sekunder
    return float(tf_minutes(tf, 1) * 60)


def tf_from_seconds(seconds):
    return tf_period(max(1, int(seconds) // 60))


def _bucket(t, minutes):
    if is_months(minutes):
        stamp = _dt(t)
        index = (stamp.year * 12 + stamp.month - 1) // (minutes // MONTH) * (minutes // MONTH)
        return int(datetime(index // 12, index % 12 + 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    day = t - t % 86_400_000
    if minutes >= 1440:
        if minutes == 10080:
            weekday = int((day // 86_400_000 + 3) % 7)     # 1970-01-01 var torsdag; uke starter mandag
            return day - weekday * 86_400_000
        return day - (day // 86_400_000 % (minutes // 1440)) * 86_400_000
    return day + (t - day) // (minutes * 60_000) * (minutes * 60_000)


def time_fn(rt, i, tf=None, session=None, is_close=False):
    """``time(tf, session)``: starttid for lyset i tidsrammen ``tf``; ``na`` utenfor økten."""
    t = rt.T[i]
    if session is not None and not isna(session) and not in_session(t, session):
        return NAN
    if tf is None or isna(tf) or tf == "":
        return rt.TC[i] if is_close else t
    minutes = tf_minutes(tf, rt.minutes)
    start = _bucket(int(t), minutes)
    return float(start + minutes * 60_000) if is_close else float(start)


class TFCHANGE:
    def __init__(self, rt):
        self.rt, self.last = rt, None

    def __call__(self, i, tf):
        bucket = _bucket(int(self.rt.T[i]), tf_minutes(tf, self.rt.minutes))
        changed = self.last is not None and bucket != self.last
        self.last = bucket
        return changed


SITES["timeframe.change"] = TFCHANGE


# ---------------------------------------------------------------- egne typer, map og chart.point

class PineObject:
    """Objekt av en egen type (``type Navn``). Felt lagres i en ordbok; ``na`` er NaN som ellers."""
    __slots__ = ("kind", "fields")

    def __init__(self, kind, fields):
        self.kind, self.fields = kind, fields

    def __repr__(self):
        return f"{self.kind}({', '.join(f'{k}={v!r}' for k, v in self.fields.items())})"


def new_obj_(kind, names, values):
    return PineObject(kind, dict(zip(names, values)))


def getf_(obj, name):
    if obj.__class__ is PineObject:
        try:
            return obj.fields[name]
        except KeyError:
            raise PineRuntimeError(f"typen {obj.kind} har ikke feltet «{name}»") from None
    if isna(obj):
        raise PineRuntimeError(f"kan ikke lese feltet «{name}» fra et na-objekt")
    if isinstance(obj, Drawing):
        return NAN
    raise PineRuntimeError(f"verdien har ikke feltet «{name}»")


def setf_(obj, name, value):
    if obj.__class__ is PineObject:
        if name not in obj.fields:
            raise PineRuntimeError(f"typen {obj.kind} har ikke feltet «{name}»")
        obj.fields[name] = value
        return value
    if isna(obj):
        raise PineRuntimeError(f"kan ikke endre feltet «{name}» på et na-objekt")
    raise PineRuntimeError(f"verdien har ikke feltet «{name}»")


def copy_obj_(obj):
    if obj.__class__ is PineObject:
        return PineObject(obj.kind, dict(obj.fields))
    if isinstance(obj, PineArray):
        return a_copy(obj)
    if isinstance(obj, PineMap):
        return PineMap(dict(obj.items))
    return obj


def type_key(obj):
    if obj.__class__ is PineObject:
        return obj.kind
    if isinstance(obj, PineArray):
        return "array"
    if isinstance(obj, PineMap):
        return "map"
    if isinstance(obj, Drawing):
        return "drawing"
    if obj.__class__ is str:
        return "string"
    if obj is True or obj is False:
        return "bool"
    return "float"


def mcall_(obj, table, i, *args):
    """Egen ``method`` med samme navn for flere typer: velg etter objektets type."""
    fn = table.get(type_key(obj))
    if fn is None and type_key(obj) == "float":
        fn = table.get("int")
    if fn is None:
        if isna(obj):
            raise PineRuntimeError("kan ikke kalle en metode på et na-objekt")
        raise PineRuntimeError(f"ingen metode for typen {type_key(obj)}")
    return fn(i, obj, *args)


class PineMap:
    __slots__ = ("items",)

    def __init__(self, items=None):
        self.items = items if items is not None else {}


def _map(m):
    if not isinstance(m, PineMap):
        raise PineRuntimeError("forventet et map (er det na?)")
    return m


def m_new(*args):
    return PineMap()


def m_put(m, key, value):
    m = _map(m)
    old = m.items.get(key, NAN)
    if key not in m.items and len(m.items) >= 50_000:
        raise PineRuntimeError("map er for stort (maks 50 000 nøkler)")
    m.items[key] = value
    return old


def m_get(m, key):
    return _map(m).items.get(key, NAN)


def m_contains(m, key):
    return key in _map(m).items


def m_remove(m, key):
    return _map(m).items.pop(key, NAN)


def m_size(m):
    return float(len(_map(m).items))


def m_clear(m):
    _map(m).items.clear()


def m_keys(m):
    return PineArray("any", list(_map(m).items.keys()))


def m_values(m):
    return PineArray("any", list(_map(m).items.values()))


def m_copy(m):
    return PineMap(dict(_map(m).items))


def m_put_all(m, other):
    _map(m).items.update(_map(other).items)


MAP_METHODS = {"put": m_put, "get": m_get, "contains": m_contains, "remove": m_remove, "size": m_size,
               "clear": m_clear, "keys": m_keys, "values": m_values, "copy": m_copy, "put_all": m_put_all}


def point_new(time=NAN, index=NAN, price=NAN):
    return PineObject("chart.point", {"time": time, "index": index, "price": price})


def point_from_index(index, price):
    return PineObject("chart.point", {"time": NAN, "index": index, "price": price})


def point_from_time(time, price):
    return PineObject("chart.point", {"time": time, "index": NAN, "price": price})


USD_LIKE = {"USD", "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI"}


def currency_rate_(source, target, ignore_invalid_currency=False):
    """``request.currency_rate``: 1 mellom dollar-valutaer (USD/USDT/USDC …); andre kurser hentes ikke."""
    a = "" if isna(source) else str(source).upper()
    b = "" if isna(target) else str(target).upper()
    if a == b or (a in USD_LIKE and b in USD_LIKE):
        return 1.0
    if t_(ignore_invalid_currency):
        return NAN
    raise PineRuntimeError(f"valutakursen {a}→{b} er ikke tilgjengelig i Pine-testen")
