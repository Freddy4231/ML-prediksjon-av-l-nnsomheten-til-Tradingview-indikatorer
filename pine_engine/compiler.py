"""Oversetter Pine-syntakstreet til Python som kjøres lys for lys.

Sikkerhet: brukerens navn kommer aldri inn i Python-koden. Variabler blir ``S12``/``v12``,
tekst blir ``repr``, tall blir ``float``, og funksjoner slås opp i en fast tabell. Den ferdige
koden sjekkes med ``ast`` (bare kjente navn og attributter) før den kjøres med tomme
``__builtins__``.

Ytelse: alt lages som lukninger (``make_script`` → ``bar(i)``), variabler uten historikk blir
vanlige lokale Python-variabler, og bare variabler med ``x[n]``, ``var`` eller bruk i egne
funksjoner lagres som lister.
"""
import ast
import re
from dataclasses import dataclass, field

from pine_engine import parser as P
from pine_engine.parser import PineError

SOURCE_NAMES = {"open": "O", "high": "H", "low": "L", "close": "C", "volume": "V", "hl2": "HL2", "hlc3": "HLC3",
                "ohlc4": "OHLC4", "hlcc4": "HLCC4"}
BAR_VARS = {**{k: f"{v}[i]" for k, v in SOURCE_NAMES.items()}, "time": "T[i]", "time_close": "TC[i]",
            "bar_index": "i", "last_bar_index": "LAST", "last_bar_time": "LAST_TIME", "timenow": "LAST_TIME",
            "year": "YEAR[i]", "month": "MONTH[i]", "dayofmonth": "DOM[i]", "dayofweek": "DOW[i]",
            "hour": "HOUR[i]", "minute": "MINUTE[i]", "second": "SECOND[i]", "weekofyear": "WOY[i]",
            "time_tradingday": "TDAY[i]"}
HIST_ARRAYS = {**SOURCE_NAMES, "time": "T", "time_close": "TC", "year": "YEAR", "month": "MONTH",
               "dayofmonth": "DOM", "dayofweek": "DOW", "hour": "HOUR", "minute": "MINUTE"}
QUALIFIED_VARS = {
    "syminfo.tickerid": "TICKERID", "syminfo.ticker": "TICKER", "syminfo.mintick": "MINTICK",
    "syminfo.pointvalue": "1.0", "syminfo.currency": "CURRENCY", "syminfo.basecurrency": "BASECURRENCY",
    "syminfo.type": "SYMTYPE", "syminfo.timezone": "'Etc/UTC'", "syminfo.root": "TICKER", "syminfo.prefix": "PREFIX",
    "syminfo.description": "TICKER", "syminfo.session": "'24x7'", "syminfo.volumetype": "'base'",
    "syminfo.minmove": "1.0", "syminfo.pricescale": "PRICESCALE",
    "timeframe.period": "TF_PERIOD", "timeframe.main_period": "TF_MAIN", "timeframe.multiplier": "TF_MULT",
    "timeframe.isintraday": "TF_INTRADAY", "timeframe.isdaily": "TF_DAILY", "timeframe.isweekly": "False",
    "timeframe.ismonthly": "False", "timeframe.isminutes": "TF_INTRADAY", "timeframe.isseconds": "False",
    "timeframe.isdwm": "TF_DAILY", "timeframe.isticks": "False",
    "barstate.isconfirmed": "True", "barstate.islast": "(i == LAST)", "barstate.isfirst": "(i == 0)",
    "barstate.ishistory": "True", "barstate.isrealtime": "False", "barstate.isnew": "True",
    "barstate.islastconfirmedhistory": "(i == LAST)",
    "chart.is_standard": "True", "chart.is_heikinashi": "False", "chart.is_renko": "False",
    "chart.is_kagi": "False", "chart.is_linebreak": "False", "chart.is_pnf": "False", "chart.is_range": "False",
    "chart.bg_color": "'#FFFFFF'", "chart.fg_color": "'#000000'", "chart.left_visible_bar_time": "T[0]",
    "chart.right_visible_bar_time": "LAST_TIME",
    "strategy.position_size": "BR.pos", "strategy.position_avg_price": "BR.avg_price()",
    "strategy.opentrades": "BR.opentrades()", "strategy.closedtrades": "BR.closedtrades()",
    "strategy.equity": "BR.equity_now(i)", "strategy.netprofit": "BR.netprofit()",
    "strategy.openprofit": "BR.openprofit(i)", "strategy.initial_capital": "BR.initial",
    "strategy.wintrades": "BR.wintrades()", "strategy.losstrades": "BR.losstrades()",
    "strategy.eventrades": "0.0", "strategy.grossprofit": "BR.grossprofit()", "strategy.grossloss": "BR.grossloss()",
    "strategy.position_entry_name": "BR.entry_name()", "strategy.max_drawdown": "0.0",
    "strategy.long": "True", "strategy.short": "False",
    "math.pi": "3.141592653589793", "math.e": "2.718281828459045", "math.phi": "1.618033988749895",
    "math.rphi": "0.6180339887498949",
    "session.ismarket": "True", "session.ispremarket": "False", "session.ispostmarket": "False",
    "session.isfirstbar": "False", "session.islastbar": "False",
    "dayofweek.sunday": "1.0", "dayofweek.monday": "2.0", "dayofweek.tuesday": "3.0", "dayofweek.wednesday": "4.0",
    "dayofweek.thursday": "5.0", "dayofweek.friday": "6.0", "dayofweek.saturday": "7.0",
    "ta.tr": "tr_(RT, i, False)", "tr": "tr_(RT, i, False)",
}
STATE_VARS = {"ta.vwap": "ta.vwap", "vwap": "ta.vwap", "ta.obv": "ta.obv", "obv": "ta.obv",
              "ta.accdist": "ta.accdist", "accdist": "ta.accdist"}
CONST_NAMESPACES = {"color", "shape", "location", "size", "plot", "line", "label", "text", "position", "extend",
                    "xloc", "yloc", "display", "format", "alert", "order", "currency", "scale", "font", "hline",
                    "barmerge", "session", "dividends", "earnings", "splits", "adjustment", "backadjustment",
                    "strategy", "box", "table", "settlement_as_close", "input"}
V4_ALIASES = {name: f"ta.{name}" for name in (
    "sma", "ema", "rma", "wma", "vwma", "swma", "alma", "hma", "linreg", "stdev", "variance", "dev", "highest",
    "lowest", "highestbars", "lowestbars", "atr", "rsi", "crossover", "crossunder", "cross", "change", "mom", "roc",
    "rising", "falling", "pivothigh", "pivotlow", "valuewhen", "barssince", "cum", "macd", "bb", "bbw", "kc", "kcw",
    "stoch", "cci", "mfi", "wpr", "sar", "supertrend", "dmi", "tsi", "cmo", "percentrank", "correlation", "median",
    "percentile_linear_interpolation", "percentile_nearest_rank", "range", "cog")}
V4_ALIASES.update({name: f"math.{name}" for name in (
    "abs", "max", "min", "round", "floor", "ceil", "sqrt", "pow", "log", "log10", "exp", "sign", "avg", "sum",
    "sin", "cos", "tan", "asin", "acos", "atan", "todegrees", "toradians")})
V4_ALIASES.update({"security": "request.security", "study": "indicator", "tostring": "str.tostring",
                   "tickerid": "ticker.new", "heikinashi": "ticker.heikinashi"})

REQ = object()


@dataclass
class Spec:
    kind: str                # pure, var, site, prelude, input, output, drawing, noop, strategy, security, decl
    impl: str = ""
    params: list = field(default_factory=list)   # [(navn, standardkode eller REQ)]
    aliases: dict = field(default_factory=dict)


def _p(*names, **defaults):
    return [(n, defaults.get(n, REQ)) for n in names]


PURE = {
    "math.abs": ("m_abs", _p("number")), "math.round": ("m_round", _p("number", "precision", precision="NAN")),
    "math.floor": ("m_floor", _p("number")), "math.ceil": ("m_ceil", _p("number")),
    "math.sqrt": ("m_sqrt", _p("number")), "math.pow": ("m_pow", _p("base", "exponent")),
    "math.log": ("m_log", _p("number")), "math.log10": ("m_log10", _p("number")),
    "math.exp": ("m_exp", _p("number")), "math.sign": ("m_sign", _p("number")),
    "math.sin": ("m_sin", _p("angle")), "math.cos": ("m_cos", _p("angle")), "math.tan": ("m_tan", _p("angle")),
    "math.asin": ("m_asin", _p("angle")), "math.acos": ("m_acos", _p("angle")), "math.atan": ("m_atan", _p("angle")),
    "math.todegrees": ("m_todegrees", _p("radians")), "math.toradians": ("m_toradians", _p("degrees")),
    "nz": ("nz_", _p("source", "replacement", replacement="0.0")), "na": ("na_", _p("x")),
    "iff": ("iff_", _p("condition", "then", "_else")), "int": ("int_", _p("x")), "float": ("float_", _p("x")),
    "bool": ("bool_", _p("x")), "color": ("ident_", _p("x")), "string": ("ident_", _p("x")),
    "str.tostring": ("s_tostring", _p("value", "format", format="None")), "str.length": ("s_length", _p("string")),
    "str.contains": ("s_contains", _p("source", "str")), "str.startswith": ("s_startswith", _p("source", "str")),
    "str.endswith": ("s_endswith", _p("source", "str")), "str.pos": ("s_pos", _p("source", "str")),
    "str.replace_all": ("s_replace_all", _p("source", "target", "replacement")),
    "str.replace": ("s_replace", _p("source", "target", "replacement", "occurrence", occurrence="0.0")),
    "str.lower": ("s_lower", _p("source")), "str.upper": ("s_upper", _p("source")),
    "str.trim": ("s_trim", _p("source")),
    "str.substring": ("s_substring", _p("source", "begin_pos", "end_pos", end_pos="NAN")),
    "str.tonumber": ("s_tonumber", _p("string")), "str.split": ("s_split", _p("string", "separator")),
    "str.repeat": ("s_repeat", _p("source", "repeat", "separator", separator="''")),
    "str.format_time": ("s_format_time", _p("time", "format", "timezone", format="''", timezone="None")),
    "color.new": ("c_new", _p("color", "transp", transp="0.0")),
    "color.rgb": ("c_rgb", _p("red", "green", "blue", "transp", transp="0.0")),
    "color.r": ("c_r", _p("color")), "color.g": ("c_g", _p("color")), "color.b": ("c_b", _p("color")),
    "color.t": ("c_t", _p("color")),
    "color.from_gradient": ("c_from_gradient", _p("value", "bottom_value", "top_value", "bottom_color",
                                                  "top_color")),
    "year": ("year_", _p("time", "timezone", timezone="None")),
    "month": ("month_", _p("time", "timezone", timezone="None")),
    "dayofmonth": ("dayofmonth_", _p("time", "timezone", timezone="None")),
    "dayofweek": ("dayofweek_", _p("time", "timezone", timezone="None")),
    "hour": ("hour_", _p("time", "timezone", timezone="None")),
    "minute": ("minute_", _p("time", "timezone", timezone="None")),
    "second": ("second_", _p("time", "timezone", timezone="None")),
    "weekofyear": ("weekofyear_", _p("time", "timezone", timezone="None")),
    "ticker.heikinashi": ("ticker_ha", _p("symbol")), "ticker.standard": ("ident_", _p("symbol")),
    "ticker.new": ("ticker_new", _p("prefix", "ticker", "session", "adjustment", session="None",
                                    adjustment="None")),
    "ticker.modify": ("ticker_modify", _p("tickerid", "session", "adjustment", session="None",
                                          adjustment="None")),
    "runtime.error": ("runtime_error", _p("message", message="''")),
    "timeframe.in_seconds": ("tf_seconds_of", _p("timeframe", timeframe="TF_PERIOD")),
    "timeframe.from_seconds": ("tf_from_seconds", _p("seconds")),
    "array.from": ("a_from", "var"), "array.size": ("a_size", _p("id")),
    "array.push": ("a_push", _p("id", "value")), "array.unshift": ("a_unshift", _p("id", "value")),
    "array.pop": ("a_pop", _p("id")), "array.shift": ("a_shift", _p("id")),
    "array.get": ("a_get", _p("id", "index")), "array.set": ("a_set", _p("id", "index", "value")),
    "array.clear": ("a_clear", _p("id")), "array.insert": ("a_insert", _p("id", "index", "value")),
    "array.remove": ("a_remove", _p("id", "index")), "array.includes": ("a_includes", _p("id", "value")),
    "array.indexof": ("a_indexof", _p("id", "value")), "array.lastindexof": ("a_lastindexof", _p("id", "value")),
    "array.sum": ("a_sum", _p("id")), "array.avg": ("a_avg", _p("id")),
    "array.max": ("a_max", _p("id", "nth", nth="0.0")), "array.min": ("a_min", _p("id", "nth", nth="0.0")),
    "array.median": ("a_median", _p("id")), "array.variance": ("a_variance", _p("id", "biased", biased="True")),
    "array.stdev": ("a_stdev", _p("id", "biased", biased="True")),
    "array.sort": ("a_sort", _p("id", "order", order="'order.ascending'")), "array.reverse": ("a_reverse", _p("id")),
    "array.copy": ("a_copy", _p("id")), "array.slice": ("a_slice", _p("id", "index_from", "index_to")),
    "array.concat": ("a_concat", _p("id1", "id2")), "array.first": ("a_first", _p("id")),
    "array.last": ("a_last", _p("id")),
    "array.fill": ("a_fill", _p("id", "value", "index_from", "index_to", index_from="0.0", index_to="NAN")),
    "array.join": ("a_join", _p("id", "separator", separator="','")), "array.range": ("a_range", _p("id")),
    "map.put": ("m_put", _p("id", "key", "value")), "map.get": ("m_get", _p("id", "key")),
    "map.contains": ("m_contains", _p("id", "key")), "map.remove": ("m_remove", _p("id", "key")),
    "map.size": ("m_size", _p("id")), "map.clear": ("m_clear", _p("id")), "map.keys": ("m_keys", _p("id")),
    "map.values": ("m_values", _p("id")), "map.copy": ("m_copy", _p("id")),
    "map.put_all": ("m_put_all", _p("id", "id2")), "map.new": ("m_new", []),
    "chart.point.new": ("point_new", _p("time", "index", "price", time="NAN", index="NAN", price="NAN")),
    "chart.point.from_index": ("point_from_index", _p("index", "price")),
    "chart.point.from_time": ("point_from_time", _p("time", "price")),
    "chart.point.copy": ("copy_obj_", _p("id")), "copy": ("copy_obj_", _p("id")),
    "request.currency_rate": ("currency_rate_", _p("from", "to", "ignore_invalid_currency",
                                                   ignore_invalid_currency="False")),
}
VARARGS = {"math.max": "m_max", "math.min": "m_min", "math.avg": "m_avg", "str.format": "s_format",
           "timestamp": "timestamp_", "array.from": "a_from"}
ARRAY_NEW = {"array.new_float": "float", "array.new_int": "int", "array.new_bool": "bool",
             "array.new_string": "string", "array.new_color": "color", "array.new_line": "line",
             "array.new_label": "label", "array.new_box": "box", "array.new_table": "table",
             "array.new_linefill": "linefill", "array.new": "any"}
SITE_PARAMS = {
    "ta.sma": _p("source", "length"), "ta.ema": _p("source", "length"), "ta.rma": _p("source", "length"),
    "ta.wma": _p("source", "length"), "ta.vwma": _p("source", "length"), "ta.hma": _p("source", "length"),
    "ta.swma": _p("source"),
    "ta.alma": _p("series", "length", "offset", "sigma", "floor", floor="False"),
    "ta.stdev": _p("source", "length", "biased", biased="True"),
    "ta.variance": _p("source", "length", "biased", biased="True"), "ta.dev": _p("source", "length"),
    "ta.highest": _p("source", "length"), "ta.lowest": _p("source", "length"),
    "ta.highestbars": _p("source", "length"), "ta.lowestbars": _p("source", "length"),
    "ta.range": _p("source", "length"), "ta.change": _p("source", "length", length="1.0"),
    "ta.mom": _p("source", "length", length="1.0"), "ta.roc": _p("source", "length"),
    "ta.rsi": _p("source", "length"), "ta.atr": _p("length"), "ta.crossover": _p("source1", "source2"),
    "ta.crossunder": _p("source1", "source2"), "ta.cross": _p("source1", "source2"),
    "ta.pivothigh": _p("source", "leftbars", "rightbars"), "ta.pivotlow": _p("source", "leftbars", "rightbars"),
    "ta.valuewhen": _p("condition", "source", "occurrence", occurrence="0.0"), "ta.barssince": _p("condition"),
    "ta.cum": _p("source"), "math.sum": _p("source", "length"), "fixnan": _p("source"),
    "ta.percentrank": _p("source", "length"), "ta.linreg": _p("source", "length", "offset", offset="0.0"),
    "ta.correlation": _p("source1", "source2", "length"), "ta.median": _p("source", "length"),
    "ta.percentile_linear_interpolation": _p("source", "length", "percentage"),
    "ta.percentile_nearest_rank": _p("source", "length", "percentage"),
    "ta.rising": _p("source", "length"), "ta.falling": _p("source", "length"), "ta.max": _p("source"),
    "ta.min": _p("source"), "ta.vwap": _p("source", "anchor", "stdev_mult", source="None", anchor="None",
                                          stdev_mult="NAN"),
    "math.random": _p("min", "max", "seed", min="0.0", max="1.0", seed="NAN"),
    "timeframe.change": _p("timeframe"),
}
OPTIONAL_SOURCE = {"ta.highest": "H", "ta.lowest": "L", "ta.highestbars": "H", "ta.lowestbars": "L"}
OUTPUTS = {
    "plotshape": _p("series", "title", "style", "location", "color", "offset", "text", "textcolor",
                    title="None", style="None", location="None", color="None", offset="None", text="None",
                    textcolor="None"),
    "plotchar": _p("series", "title", "char", "location", "color", "offset", "text", title="None", char="None",
                   location="None", color="None", offset="None", text="None"),
    "plotarrow": _p("series", "title", title="None"),
    "alertcondition": _p("condition", "title", "message", title="None", message="None"),
    "alert": _p("message", "freq", freq="None"),
}
DRAWING_NEW = {"label.new", "line.new", "box.new", "table.new", "linefill.new", "polyline.new", "label.copy",
               "line.copy", "box.copy"}
NOOP_PREFIXES = ("plot", "bgcolor", "barcolor", "fill", "hline", "label.", "line.", "box.", "table.", "linefill.",
                 "polyline.", "log.", "strategy.risk.", "max_bars_back")
STRATEGY = {
    "strategy.entry": ("entry", _p("id", "direction", "qty", "limit", "stop", "oca_name", "oca_type", "comment",
                                   "alert_message", "disable_alert", "when", qty="NAN", limit="NAN", stop="NAN",
                                   oca_name="None", oca_type="None", comment="None", alert_message="None",
                                   disable_alert="None", when="True"), {"long": "direction"}),
    "strategy.order": ("order", _p("id", "direction", "qty", "limit", "stop", "oca_name", "oca_type", "comment",
                                   "alert_message", "disable_alert", "when", qty="NAN", limit="NAN", stop="NAN",
                                   oca_name="None", oca_type="None", comment="None", alert_message="None",
                                   disable_alert="None", when="True"), {"long": "direction"}),
    "strategy.exit": ("exit", _p("id", "from_entry", "qty", "qty_percent", "profit", "limit", "loss", "stop",
                                 "trail_price", "trail_points", "trail_offset", "oca_name", "comment",
                                 "comment_profit", "comment_loss", "comment_trailing", "alert_message",
                                 "alert_profit", "alert_loss", "alert_trailing", "disable_alert", "when",
                                 from_entry="''", qty="NAN", qty_percent="NAN", profit="NAN", limit="NAN",
                                 loss="NAN", stop="NAN", trail_price="NAN", trail_points="NAN", trail_offset="NAN",
                                 oca_name="None", comment="None", comment_profit="None", comment_loss="None",
                                 comment_trailing="None", alert_message="None", alert_profit="None",
                                 alert_loss="None", alert_trailing="None", disable_alert="None", when="True"), {}),
    "strategy.close": ("close", _p("id", "comment", "qty", "qty_percent", "alert_message", "immediately",
                                   "disable_alert", "when", comment="None", qty="NAN", qty_percent="NAN",
                                   alert_message="None", immediately="False", disable_alert="None", when="True"), {}),
    "strategy.close_all": ("close_all", _p("comment", "alert_message", "immediately", "disable_alert", "when",
                                           comment="None", alert_message="None", immediately="False",
                                           disable_alert="None", when="True"), {}),
    "strategy.cancel": ("cancel", _p("id", "when", when="True"), {}),
    "strategy.cancel_all": ("cancel_all", _p("when", when="True"), {}),
}
INPUTS = {"input": "auto", "input.int": "int", "input.float": "float", "input.bool": "bool",
          "input.string": "string", "input.source": "source", "input.timeframe": "timeframe",
          "input.color": "color", "input.session": "string", "input.symbol": "string", "input.price": "float",
          "input.time": "float", "input.text_area": "string"}
DECLS = {"indicator", "strategy", "library"}
UNSUPPORTED = {"request.security_lower_tf": "request.security_lower_tf",
               "request.financial": "request.financial", "request.economic": "request.economic",
               "request.quandl": "request.quandl", "request.dividends": "request.dividends",
               "request.earnings": "request.earnings", "request.splits": "request.splits",
               "request.seed": "request.seed",
               "strategy.opentrades.entry_price": "strategy.opentrades.*",
               "strategy.closedtrades.profit": "strategy.closedtrades.*"}

PRELUDE = r"""
__ta_macd(source, fastlen, slowlen, siglen) =>
    fastMA = ta.ema(source, fastlen)
    slowMA = ta.ema(source, slowlen)
    macdLine = fastMA - slowMA
    signalLine = ta.ema(macdLine, siglen)
    [macdLine, signalLine, macdLine - signalLine]
__ta_bb(series, length, mult) =>
    basis = ta.sma(series, length)
    dev = mult * ta.stdev(series, length)
    [basis, basis + dev, basis - dev]
__ta_bbw(series, length, mult) =>
    basis = ta.sma(series, length)
    dev = mult * ta.stdev(series, length)
    ((basis + dev) - (basis - dev)) / basis
__ta_kc(series, length, mult, useTrueRange = true) =>
    basis = ta.ema(series, length)
    span = useTrueRange ? ta.tr(true) : high - low
    rangeEma = ta.ema(span, length)
    [basis, basis + rangeEma * mult, basis - rangeEma * mult]
__ta_kcw(series, length, mult, useTrueRange = true) =>
    basis = ta.ema(series, length)
    span = useTrueRange ? ta.tr(true) : high - low
    rangeEma = ta.ema(span, length)
    ((basis + rangeEma * mult) - (basis - rangeEma * mult)) / basis
__ta_stoch(source, high, low, length) =>
    lowest = ta.lowest(low, length)
    highest = ta.highest(high, length)
    100 * (source - lowest) / (highest - lowest)
__ta_cci(source, length) =>
    ma = ta.sma(source, length)
    (source - ma) / (0.015 * ta.dev(source, length))
__ta_mfi(series, length) =>
    ch = ta.change(series)
    upper = math.sum(volume * (ch <= 0.0 ? 0.0 : series), length)
    lower = math.sum(volume * (ch >= 0.0 ? 0.0 : series), length)
    100.0 - (100.0 / (1.0 + upper / lower))
__ta_wpr(length) =>
    hi = ta.highest(high, length)
    lo = ta.lowest(low, length)
    100 * (close - hi) / (hi - lo)
__ta_tsi(source, short_length, long_length) =>
    pc = ta.change(source)
    ta.ema(ta.ema(pc, long_length), short_length) / ta.ema(ta.ema(math.abs(pc), long_length), short_length)
__ta_cmo(series, length) =>
    mom = ta.change(series)
    sm1 = math.sum(mom >= 0 ? mom : 0.0, length)
    sm2 = math.sum(mom >= 0 ? 0.0 : -mom, length)
    100 * (sm1 - sm2) / (sm1 + sm2)
__ta_supertrend(factor, atrPeriod) =>
    src = hl2
    atr = ta.atr(atrPeriod)
    upperBand = src + factor * atr
    lowerBand = src - factor * atr
    prevLowerBand = nz(lowerBand[1])
    prevUpperBand = nz(upperBand[1])
    lowerBand := lowerBand > prevLowerBand or close[1] < prevLowerBand ? lowerBand : prevLowerBand
    upperBand := upperBand < prevUpperBand or close[1] > prevUpperBand ? upperBand : prevUpperBand
    int direction = na
    float superTrend = na
    prevSuperTrend = superTrend[1]
    if na(atr[1])
        direction := 1
    else if prevSuperTrend == prevUpperBand
        direction := close > upperBand ? -1 : 1
    else
        direction := close < lowerBand ? 1 : -1
    superTrend := direction == -1 ? lowerBand : upperBand
    [superTrend, direction]
__ta_dmi(diLength, adxSmoothing) =>
    up = ta.change(high)
    down = -ta.change(low)
    plusDM = na(up) ? na : (up > down and up > 0 ? up : 0)
    minusDM = na(down) ? na : (down > up and down > 0 ? down : 0)
    trur = ta.rma(ta.tr, diLength)
    plus = fixnan(100 * ta.rma(plusDM, diLength) / trur)
    minus = fixnan(100 * ta.rma(minusDM, diLength) / trur)
    sum = plus + minus
    adx = 100 * ta.rma(math.abs(plus - minus) / (sum == 0 ? 1 : sum), adxSmoothing)
    [plus, minus, adx]
__ta_sar(start, inc, max) =>
    var float result = na
    var float maxMin = na
    var float acceleration = na
    var bool isBelow = false
    bool isFirstTrendBar = false
    if bar_index == 1
        if close > close[1]
            isBelow := true
            maxMin := high
            result := low[1]
        else
            isBelow := false
            maxMin := low
            result := high[1]
        isFirstTrendBar := true
        acceleration := start
    result := result + acceleration * (maxMin - result)
    if isBelow
        if result > low
            isFirstTrendBar := true
            isBelow := false
            result := math.max(high, maxMin)
            maxMin := low
            acceleration := start
    else
        if result < high
            isFirstTrendBar := true
            isBelow := true
            result := math.min(low, maxMin)
            maxMin := high
            acceleration := start
    if not isFirstTrendBar
        if isBelow
            if high > maxMin
                maxMin := high
                acceleration := math.min(acceleration + inc, max)
        else
            if low < maxMin
                maxMin := low
                acceleration := math.min(acceleration + inc, max)
    if isBelow
        result := math.min(result, low[1])
        if bar_index > 1
            result := math.min(result, low[2])
    else
        result := math.max(result, high[1])
        if bar_index > 1
            result := math.max(result, high[2])
    result
"""
PRELUDE_NAMES = {"ta.macd": "__ta_macd", "ta.bb": "__ta_bb", "ta.bbw": "__ta_bbw", "ta.kc": "__ta_kc",
                 "ta.kcw": "__ta_kcw", "ta.stoch": "__ta_stoch", "ta.cci": "__ta_cci", "ta.mfi": "__ta_mfi",
                 "ta.wpr": "__ta_wpr", "ta.tsi": "__ta_tsi", "ta.cmo": "__ta_cmo",
                 "ta.supertrend": "__ta_supertrend", "ta.dmi": "__ta_dmi", "ta.sar": "__ta_sar"}
_PRELUDE_CACHE = {}


def _prelude():
    if not _PRELUDE_CACHE:
        program = P.parse("//@version=5\n" + PRELUDE)
        _PRELUDE_CACHE.update({f.name: f for f in program.body})
    return _PRELUDE_CACHE


# ---------------------------------------------------------------- analyse

def _walk(node):
    """Alle noder i et (del)tre."""
    stack = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
            continue
        if isinstance(item, tuple):
            stack.extend(item)
            continue
        if not hasattr(item, "__dataclass_fields__"):
            continue
        yield item
        for name in item.__dataclass_fields__:
            value = getattr(item, name)
            if isinstance(value, (list, tuple)) or hasattr(value, "__dataclass_fields__"):
                stack.append(value)


def _needs_list(body):
    """Navn som må lagres som lister: brukt med historikk ``x[n]`` eller erklært med ``var``."""
    names = set()
    for node in _walk(body):
        if isinstance(node, P.Index) and isinstance(node.expr, P.Name):
            names.add(node.expr.id)
        elif isinstance(node, P.Decl) and node.mode:
            names.add(node.name)
    return names


def _qualified(node):
    parts = []
    while isinstance(node, P.Attr):
        parts.append(node.attr)
        node = node.obj
    if not isinstance(node, P.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _const(node):
    """Kompileringstidskonstant (tall, tekst, bool, farge, ``na``, ``-tall``, ``[liste]``) eller ``None``."""
    if isinstance(node, P.Num):
        return node.value
    if isinstance(node, (P.Str, P.Color)):
        return node.value
    if isinstance(node, P.Bool):
        return node.value
    if isinstance(node, P.Na):
        return float("nan")
    if isinstance(node, P.UnaryOp) and node.op in "-+" and isinstance(node.operand, P.Num):
        return -node.operand.value if node.op == "-" else node.operand.value
    if isinstance(node, P.TupleLit):
        items = [_const(x) for x in node.items]
        return None if any(x is None for x in items) else items
    if isinstance(node, P.Attr):
        q = _qualified(node)
        if q and q.split(".")[0] in ("color",):
            return q
        if q and q.split(".")[0] in CONST_NAMESPACES:
            return q
    return None


# ---------------------------------------------------------------- kodegenerering

@dataclass
class Binding:
    kind: str                # list, local, loop, param
    code: str                # Python-navn
    hist: str | None = None  # liste for parameterhistorikk
    is_global: bool = False


class Unit:
    """Én kodeenhet: hovedskriptet eller en egen funksjon (fabrikk med egen tilstand)."""

    def __init__(self, name, is_function, list_names):
        self.name, self.is_function, self.list_names = name, is_function, list_names
        self.lists, self.sites, self.instances, self.body = [], [], [], []
        self.scopes = [{}]
        self.indent = 1


class Compiler:
    def __init__(self, program):
        self.program, self.version = program, program.version
        self.counter = 0
        self.functions = {}       # pine-navn -> (FuncDef, fabrikknavn, parameterliste)
        self.factories = []       # ferdig genererte fabrikker (kodelinjer)
        self.inputs, self.outputs, self.securities, self.warnings = [], [], [], []
        self.decl = {"kind": "indicator", "title": "", "args": {}}
        self.types = {}           # egne typer: navn -> [(felt, standarduttrykk)]
        self.enums = {}           # enum: navn -> {medlem: verdi}
        self.methods = {}         # metodenavn -> [(mottakertype, nøkkel i self.functions)]
        self.uses_strategy = False
        self.line_map = {}
        self.global_unit = None
        self.function_refs = set()
        for node in program.body:
            if isinstance(node, P.FuncDef):
                for sub in _walk(node.body):
                    if isinstance(sub, P.Name):
                        self.function_refs.add(sub.id)

    def fresh(self, prefix):
        self.counter += 1
        return f"{prefix}{self.counter}"

    # ------------------------------------------------ utskrift

    def emit(self, unit, text, line=0):
        unit.body.append(("    " * unit.indent + text, line))

    # ------------------------------------------------ navneoppslag

    def lookup(self, unit, name):
        for scope in reversed(unit.scopes):
            if name in scope:
                return scope[name]
        if unit.is_function and self.global_unit is not None:
            binding = self.global_unit.scopes[0].get(name)
            if binding is not None:
                if binding.kind != "list":
                    raise PineError(f"intern feil: global «{name}» er ikke tilgjengelig i funksjonen")
                return Binding("list", binding.code, is_global=True)
        return None

    def declare(self, unit, name, force_list=False):
        as_list = force_list or name in unit.list_names or \
            (not unit.is_function and len(unit.scopes) == 1 and name in self.function_refs)
        if as_list:
            code = self.fresh("S")
            unit.lists.append(code)
            binding = Binding("list", code)
        else:
            binding = Binding("local", self.fresh("v"))
        unit.scopes[-1][name] = binding
        return binding

    def read(self, binding):
        if binding.kind == "list":
            return f"{binding.code}[-1]"
        return binding.code

    # ------------------------------------------------ uttrykk

    def expr(self, unit, node):
        if isinstance(node, P.Num):
            return repr(float(node.value))
        if isinstance(node, (P.Str, P.Color)):
            return repr(str(node.value))
        if isinstance(node, P.Bool):
            return "True" if node.value else "False"
        if isinstance(node, P.Na):
            return "NAN"
        if isinstance(node, P.Name):
            return self.name(unit, node)
        if isinstance(node, P.Attr):
            return self.attr(unit, node)
        if isinstance(node, P.Call):
            return self.call(unit, node)
        if isinstance(node, P.Index):
            return self.index(unit, node)
        if isinstance(node, P.BinOp):
            a, b = self.expr(unit, node.left), self.expr(unit, node.right)
            if node.op == "and":
                return f"(t_({a}) and t_({b}))" if self.version >= 6 else f"(t_({a}) & t_({b}))"
            if node.op == "or":
                return f"(t_({a}) or t_({b}))" if self.version >= 6 else f"(t_({a}) | t_({b}))"
            if node.op == "+":
                return f"add_({a}, {b})"
            if node.op == "/":
                return f"div_({a}, {b})"
            if node.op == "%":
                return f"mod_({a}, {b})"
            return f"({a} {node.op} {b})"
        if isinstance(node, P.UnaryOp):
            inner = self.expr(unit, node.operand)
            if node.op == "not":
                return f"(not t_({inner}))"
            return f"(-{inner})" if node.op == "-" else inner
        if isinstance(node, P.Ternary):
            return f"({self.expr(unit, node.a)} if t_({self.expr(unit, node.cond)}) else {self.expr(unit, node.b)})"
        if isinstance(node, P.TupleLit):
            return "[" + ", ".join(self.expr(unit, x) for x in node.items) + "]"
        if isinstance(node, P.BlockExpr):
            target = self.fresh("_t")
            self.block_value(unit, [node.stmt], target)
            return target
        raise PineError(f"uttrykket støttes ikke ({type(node).__name__})")

    def name(self, unit, node):
        binding = self.lookup(unit, node.id)
        if binding is not None:
            return self.read(binding)
        if node.id in BAR_VARS:
            return BAR_VARS[node.id]
        if node.id in QUALIFIED_VARS:
            return QUALIFIED_VARS[node.id]
        if node.id in STATE_VARS:
            return self.site_call(unit, STATE_VARS[node.id], [], node.line, state_var=True)
        if node.id in self.functions:
            raise PineError(f"«{node.id}» er en funksjon og må kalles med ()", node.line)
        raise PineError(f"ukjent navn «{node.id}» (ikke deklarert før bruk?)", node.line)

    def attr(self, unit, node):
        q = _qualified(node)
        if q in QUALIFIED_VARS and isinstance(self._root(node), P.Name) and \
                self.lookup(unit, self._root(node).id) is None:
            # Innebygde konstanter som dayofweek.monday, også når roten er en innebygd serie.
            if q.startswith("strategy."):
                self.uses_strategy = True
            return QUALIFIED_VARS[q]
        if not self._is_namespace(unit, node.obj):
            self._check_member(unit, node.obj, node.attr, node.line)
            return f"getf_({self.expr(unit, node.obj)}, {node.attr!r})"
        q = _qualified(node)
        root = q.split(".")[0] if q else None
        if root in self.enums and q.count(".") == 1:
            members = self.enums[root]
            if node.attr not in members:
                raise PineError(f"enum {root} har ikke medlemmet «{node.attr}»", node.line)
            return repr(members[node.attr])
        if q is None:
            raise PineError("ukjent felt", node.line)
        if q in QUALIFIED_VARS:
            if q.startswith("strategy."):
                self.uses_strategy = True
            return QUALIFIED_VARS[q]
        if q in STATE_VARS:
            return self.site_call(unit, STATE_VARS[q], [], node.line, state_var=True)
        root, _, rest = q.partition(".")
        if root == "color" and rest in COLOR_NAMES:
            return repr(COLOR_NAMES[rest])
        if root in CONST_NAMESPACES:
            return repr(q)
        raise PineError(f"ukjent navn «{q}»", node.line)

    def index(self, unit, node):
        offset_node = node.offset
        k = _const(offset_node)
        offset = repr(int(k)) if isinstance(k, float) and k == k else self.expr(unit, offset_node)
        target = node.expr
        if isinstance(target, P.Name):
            binding = self.lookup(unit, target.id)
            if binding is not None:
                if binding.kind == "list":
                    if isinstance(k, float) and k == k and k >= 0:
                        n = int(k)
                        return f"{binding.code}[-1]" if n == 0 else \
                            f"({binding.code}[-{n + 1}] if len({binding.code}) > {n} else NAN)"
                    return f"hist_({binding.code}, {offset})"
                if binding.kind == "param" and binding.hist:
                    return f"hist_({binding.hist}, {offset})"
                raise PineError(f"historikk av «{target.id}» støttes ikke her", node.line)
            if target.id in HIST_ARRAYS:
                arr = HIST_ARRAYS[target.id]
                if isinstance(k, float) and k == k and k >= 0:
                    n = int(k)
                    return f"{arr}[i]" if n == 0 else f"({arr}[i - {n}] if i >= {n} else NAN)"
                return f"bhist_({arr}, i, {offset})"
            if target.id == "bar_index":
                return f"(i - {offset})"
        hidden = self.fresh("H")
        unit.lists.append(hidden)
        return f"hpush_({hidden}, {self.expr(unit, target)}, {offset})"

    # ------------------------------------------------ kall

    def bind(self, name, params, node, aliases=None, extras_ok=False, unit=None):
        """Posisjons- og navngitte argumenter → koden for hver parameter i rekkefølge (+ ekstra uttrykk)."""
        aliases = aliases or {}
        values, extras = {}, []
        for k, arg in enumerate(node.args):
            if k < len(params):
                values[params[k][0]] = self.expr(unit, arg)
            elif extras_ok:
                extras.append(self.expr(unit, arg))
            else:
                raise PineError(f"for mange argumenter til {name}", node.line)
        for key, arg in node.kwargs:
            key = aliases.get(key, key)
            if any(p[0] == key for p in params):
                values[key] = self.expr(unit, arg)
            elif extras_ok:
                extras.append(self.expr(unit, arg))
            else:
                raise PineError(f"{name} har ingen parameter «{key}»", node.line)
        out = []
        for pname, default in params:
            if pname in values:
                out.append(values[pname])
            elif default is REQ:
                raise PineError(f"{name} mangler argumentet «{pname}»", node.line)
            else:
                out.append(default)
        return out, extras

    def call(self, unit, node):
        func = node.func
        if isinstance(func, P.Attr) and not self._is_namespace(unit, func.obj):
            self._check_member(unit, func.obj, func.attr, node.line)
        if isinstance(func, P.Attr) and not self._is_namespace(unit, func.obj) and func.attr in self.methods:
            return self.method_call(unit, func.attr, self.expr(unit, func.obj), node.args, node.kwargs, node.line)
        if isinstance(func, P.Attr) and not self._is_namespace(unit, func.obj):
            obj = self.expr(unit, func.obj)
            args = [self.expr(unit, a) for a in node.args] + [self.expr(unit, v) for _, v in node.kwargs]
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", func.attr):
                raise PineError("ugyldig metodenavn", node.line)
            return f"method_({', '.join([obj, repr(func.attr)] + args)})"
        q = _qualified(func)
        if q is None:
            raise PineError("kun navngitte funksjoner kan kalles", node.line)
        if q in self.functions:
            return self.user_call(unit, q, node)
        if q in self.methods and node.args:
            return self.method_call(unit, q, self.expr(unit, node.args[0]), node.args[1:], node.kwargs, node.line)
        root, _, member = q.partition(".")
        if root in self.types and member in ("new", "copy"):
            return self.construct(unit, root, node) if member == "new" else \
                f"copy_obj_({self.expr(unit, node.args[0])})"
        if q == "chart.point.now":
            args, _ = self.bind(q, _p("price", price="C[i]"), node, unit=unit)
            return f"point_new(T[i], float(i), {args[0]})"
        if q == "input.enum":
            return self.input_call(unit, "input.string", node, enum=True)
        if "." not in q and q in V4_ALIASES:
            q = V4_ALIASES[q]
        if q in UNSUPPORTED:
            raise PineError(f"{UNSUPPORTED[q]} støttes ikke i Pine-testen", node.line)
        if q in DECLS:
            return self.declaration_call(unit, q, node)
        if q in INPUTS:
            return self.input_call(unit, q, node)
        if q in PRELUDE_NAMES:
            return self.prelude_call(unit, q, node)
        if q == "math.round_to_mintick":
            args, _ = self.bind(q, _p("number"), node, unit=unit)
            return f"(m_round(div_({args[0]}, MINTICK)) * MINTICK)"
        if q == "ta.tr":
            args, _ = self.bind(q, _p("handle_na", handle_na="False"), node, unit=unit)
            return f"tr_(RT, i, {args[0]})"
        if q in OPTIONAL_SOURCE and len(node.args) == 1 and not node.kwargs:
            return self.site_call(unit, q, [f"{OPTIONAL_SOURCE[q]}[i]", self.expr(unit, node.args[0])], node.line)
        if q in ("ta.pivothigh", "ta.pivotlow") and len(node.args) == 2 and not node.kwargs:
            src = "H[i]" if q == "ta.pivothigh" else "L[i]"
            return self.site_call(unit, q, [src] + [self.expr(unit, a) for a in node.args], node.line)
        if q in SITE_PARAMS:
            args, _ = self.bind(q, SITE_PARAMS[q], node, unit=unit)
            return self.site_call(unit, q, args, node.line)
        if q in VARARGS:
            args = [self.expr(unit, a) for a in node.args] + [self.expr(unit, v) for _, v in node.kwargs]
            return f"{VARARGS[q]}({', '.join(args)})"
        if q in PURE:
            impl, params = PURE[q]
            args, _ = self.bind(q, params, node, unit=unit)
            return f"{impl}({', '.join(args)})"
        if q in ARRAY_NEW:
            params = _p("size", "initial_value", size="0.0", initial_value="NAN")
            args, _ = self.bind(q, params, node, unit=unit)
            typ = ARRAY_NEW[q] if q != "array.new" else (node.type_args[0] if node.type_args else "any")
            return f"a_new({typ!r})({', '.join(args)})"
        if q in ("request.security",):
            return self.security_call(unit, node)
        if q in ("time", "time_close"):
            params = _p("timeframe", "session", "timezone", timeframe="None", session="None", timezone="None")
            args, _ = self.bind(q, params, node, unit=unit)
            return f"time_fn(RT, i, {args[0]}, {args[1]}, {q == 'time_close'})"
        if q in STRATEGY:
            return self.strategy_call(unit, q, node)
        if q == "strategy.risk.allow_entry_in":
            args, _ = self.bind(q, _p("value"), node, unit=unit)
            self.uses_strategy = True
            return f"BR.allow_entry_in({args[0]})"
        if q in OUTPUTS:
            return self.output_call(unit, q, node)
        if q in DRAWING_NEW:
            return self.drawing_call(unit, q, node)
        if q == "plot":
            args, extras = self.bind(q, _p("series", "title", title="''"), node,
                                     extras_ok=True, unit=unit)
            extra = "(" + "".join(e + ", " for e in extras) + ")"
            return f"out_plot(EV, {node.line}, i, {args[0]}, {args[1]}, {extra})"
        if q.startswith(NOOP_PREFIXES):
            args = [self.expr(unit, a) for a in node.args] + [self.expr(unit, v) for _, v in node.kwargs]
            return f"noop_({', '.join(args)})"
        if q.startswith("matrix."):
            raise PineError("matrix støttes ikke ennå", node.line)
        raise PineError(f"funksjonen «{q}» støttes ikke (ennå)", node.line)

    def _check_member(self, unit, obj, name, line):
        """Felt/metoder bare på brukerverdier, og aldri navn som starter med «__»."""
        if name.startswith("__") or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise PineError(f"ugyldig felt- eller metodenavn «{name}»", line)
        if isinstance(obj, P.Name) and self.lookup(unit, obj.id) is None and obj.id in BAR_VARS:
            raise PineError(f"«{obj.id}» er en innebygd serie og har ikke feltet «{name}»", line)

    def _is_namespace(self, unit, obj):
        if isinstance(obj, P.Name):
            return self.lookup(unit, obj.id) is None and obj.id not in BAR_VARS
        if isinstance(obj, P.Attr):
            q = _qualified(obj)
            return q is not None and self._is_namespace(unit, self._root(obj))
        return False

    @staticmethod
    def _root(node):
        while isinstance(node, P.Attr):
            node = node.obj
        return node

    def site_call(self, unit, q, args, line, state_var=False):
        site = self.fresh("K")
        unit.sites.append((site, q))
        return f"{site}(i{''.join(', ' + a for a in args)})"

    def declaration_call(self, unit, q, node):
        args = {}
        for k, arg in enumerate(node.args):
            args[["title", "shorttitle", "overlay"][k] if k < 3 else f"arg{k}"] = _const(arg)
        for key, arg in node.kwargs:
            args[key] = _const(arg)
        self.decl = {"kind": q, "title": args.get("title") or "", "args": args}
        if q == "strategy":
            self.uses_strategy = True
        return "None"

    def input_call(self, unit, q, node, enum=False):
        kind = INPUTS[q]
        params = ["defval", "title"]
        values = {}
        for k, arg in enumerate(node.args[:2]):
            values[params[k]] = arg
        for key, arg in node.kwargs:
            values[key] = arg
        default = values.get("defval")
        if default is None:
            raise PineError(f"{q} mangler standardverdi", node.line)
        index = len(self.inputs)
        title = _const(values["title"]) if "title" in values else None
        options = _const(values["options"]) if "options" in values and not enum else None
        if kind == "auto":
            typ = _qualified(values["type"]) if "type" in values else None
            kind = {"input.integer": "int", "input.float": "float", "input.bool": "bool", "input.string": "string",
                    "input.source": "source", "input.resolution": "timeframe", "input.session": "string",
                    "input.symbol": "string", "input.color": "color", "input.time": "float",
                    "input.price": "float"}.get(typ)
        if isinstance(default, P.Name) and default.id in SOURCE_NAMES:
            kind = "source"
            value = default.id
        elif enum or (isinstance(default, P.Attr) and _qualified(default) and
                      _qualified(default).split(".")[0] in self.enums):
            enum_name = _qualified(default).split(".")[0] if isinstance(default, P.Attr) else None
            if enum_name not in self.enums or default.attr not in self.enums[enum_name]:
                raise PineError("input.enum trenger et enum-medlem som standardverdi", node.line)
            kind, value = "string", self.enums[enum_name][default.attr]
            options = options or list(self.enums[enum_name].values())
        else:
            value = _const(default)
            if value is None and kind == "color":
                value = "#787B86"       # fargen påvirker ikke handlene
            if value is None:
                raise PineError(f"standardverdien i {q} må være en konstant", node.line)
            if kind is None:
                kind = "bool" if isinstance(value, bool) else "string" if isinstance(value, str) else \
                    "int" if isinstance(value, float) and value == int(value) else "float"
        self.inputs.append({"index": index, "kind": kind, "title": title or f"Input {index + 1}", "default": value,
                            "options": options, "line": node.line})
        if kind == "source":
            return f"SRCS[{index}][i]"
        return f"IN[{index}]"

    def output_call(self, unit, q, node):
        args, extras = self.bind(q, OUTPUTS[q], node, extras_ok=True, unit=unit)
        site = len(self.outputs)
        values = dict(zip([p[0] for p in OUTPUTS[q]], args))
        title = next((_const(v) for k, v in node.kwargs if k == "title"), None)
        if title is None and len(node.args) > 1 and q != "alert":
            title = _const(node.args[1])
        self.outputs.append({"site": site, "kind": q, "title": title if isinstance(title, str) else None,
                             "line": node.line})
        extra = "(" + "".join(e + ", " for e in extras) + ")"
        if q == "alert":
            return f"out_alert(EV, {site}, i, {values['message']}, {extra})"
        if q == "alertcondition":
            return f"out_signal(EV, 'alertcondition', {site}, i, {values['condition']}, {values['title']}, " \
                   f"{values['message']}, {extra})"
        if q == "plotarrow":
            return f"out_arrow(EV, {site}, i, {values['series']}, {values['title']}, {extra})"
        text = values.get("text", "None")
        return f"out_signal(EV, {q!r}, {site}, i, {values['series']}, {values['title']}, {text}, {extra})"

    def drawing_call(self, unit, q, node):
        if q == "label.new":
            params = _p("x", "y", "text", title="None", text="''", x="NAN", y="NAN")
            args, extras = self.bind(q, params, node, extras_ok=True, unit=unit)
            site = len(self.outputs)
            self.outputs.append({"site": site, "kind": "label.new", "title": None, "line": node.line})
            extra = "(" + "".join(e + ", " for e in extras) + ")"
            return f"out_label(EV, DR, {site}, i, {args[0]}, {args[1]}, {args[2]}, {extra})"
        args = [self.expr(unit, a) for a in node.args] + [self.expr(unit, v) for _, v in node.kwargs]
        return f"draw_(DR{''.join(', ' + a for a in args)})"

    def strategy_call(self, unit, q, node):
        self.uses_strategy = True
        method, params, aliases = STRATEGY[q]
        args, _ = self.bind(q, params, node, aliases=aliases, unit=unit)
        values = dict(zip([p[0] for p in params], args))
        if method in ("entry", "order"):
            return f"BR.{method}(i, {values['id']}, {values['direction']}, {values['limit']}, {values['stop']}, " \
                   f"{values['when']})"
        if method == "exit":
            keys = ("id", "from_entry", "qty_percent", "profit", "limit", "loss", "stop", "trail_price",
                    "trail_points", "trail_offset", "when")
            return f"BR.exit(i, {', '.join(values[k] for k in keys)})"
        if method == "close":
            return f"BR.close(i, {values['id']}, {values['qty_percent']}, {values['immediately']}, {values['when']})"
        if method == "close_all":
            return f"BR.close_all(i, {values['immediately']}, {values['when']})"
        if method == "cancel":
            return f"BR.cancel(i, {values['id']}, {values['when']})"
        return f"BR.cancel_all(i, {values['when']})"

    def security_call(self, unit, node):
        params = _p("symbol", "timeframe", "expression", "gaps", "lookahead", "ignore_invalid_symbol", "currency",
                    "calc_bars_count", gaps="'barmerge.gaps_off'", lookahead="'barmerge.lookahead_off'",
                    ignore_invalid_symbol="False", currency="None", calc_bars_count="None")
        args, _ = self.bind("request.security", params, node, aliases={"resolution": "timeframe"}, unit=unit)
        site = len(self.securities)
        expression = node.args[2] if len(node.args) > 2 else dict(node.kwargs).get("expression")
        lookahead = next((v for k, v in node.kwargs if k == "lookahead"), node.args[4] if len(node.args) > 4 else None)
        look_on = _qualified(lookahead) == "barmerge.lookahead_on" if lookahead is not None else False
        shifted = self._shifted(expression)
        if look_on and not shifted:
            self.warnings.append(f"Linje {node.line}: request.security med lookahead_on uten [1] ser inn i "
                                 "fremtiden på historiske lys. Resultatet er ikke realistisk.")
        self.securities.append({"site": site, "line": node.line, "lookahead_on": look_on})
        return f"SEC({site}, i, {', '.join(args[:5])})"

    def call_args(self, unit, q, params, args, kwargs, line, pre=()):
        """Argumentkode i parameterrekkefølge; ``pre`` er ferdig oversatte første argumenter (mottaker)."""
        values = {params[k][0]: code for k, code in enumerate(pre)}
        offset = len(pre)
        for k, arg in enumerate(args):
            if k + offset >= len(params):
                raise PineError(f"for mange argumenter til {q}", line)
            values[params[k + offset][0]] = self.expr(unit, arg)
        for key, arg in kwargs:
            if not any(p[0] == key for p in params):
                raise PineError(f"{q} har ingen parameter «{key}»", line)
            values[key] = self.expr(unit, arg)
        out = []
        for pname, default in params:
            if pname in values:
                out.append(values[pname])
            elif default is not None:
                out.append(self.expr(unit, default))
            else:
                raise PineError(f"{q} mangler argumentet «{pname}»", line)
        return out

    def _shifted(self, expression, depth=0):
        """Om uttrykket bare bruker forrige lys (``x[1]``), direkte eller som siste linje i en egen funksjon."""
        if isinstance(expression, P.Index):
            offset = _const(expression.offset)
            return isinstance(offset, float) and offset >= 1
        if isinstance(expression, P.TupleLit):
            return bool(expression.items) and all(self._shifted(x, depth + 1) for x in expression.items)
        if isinstance(expression, P.Call) and depth < 5:
            name = _qualified(expression.func)
            entry = self.functions.get(name)
            if entry is not None and entry[0].body and isinstance(entry[0].body[-1], P.ExprStmt):
                return self._shifted(entry[0].body[-1].expr, depth + 1)
        return False

    def user_call(self, unit, q, node):
        func, factory, params = self.functions[q]
        instance = self.fresh("U")
        unit.instances.append((instance, factory))
        args = self.call_args(unit, q, params, node.args, node.kwargs, node.line)
        return f"{instance}(i{''.join(', ' + a for a in args)})"

    def method_call(self, unit, name, receiver, args, kwargs, line):
        """``obj.metode(...)``: én kandidat kalles direkte, flere velges etter objektets type ved kjøring."""
        candidates = self.methods[name]
        _, key = candidates[0]
        params = self.functions[key][2]
        codes = self.call_args(unit, name, params, args, kwargs, line, pre=["_RECV_"])[1:]
        if len(candidates) == 1:
            instance = self.fresh("U")
            unit.instances.append((instance, self.functions[key][1]))
            return f"{instance}(i, {receiver}{''.join(', ' + c for c in codes)})"
        table = []
        for typ, key in candidates:
            instance = self.fresh("U")
            unit.instances.append((instance, self.functions[key][1]))
            table.append(f"{typ!r}: {instance}")
        return f"mcall_({receiver}, {{{', '.join(table)}}}, i{''.join(', ' + c for c in codes)})"

    def construct(self, unit, typ, node):
        fields = self.types[typ]
        names = [f for f, _ in fields]
        values = {}
        for k, arg in enumerate(node.args):
            if k >= len(names):
                raise PineError(f"for mange argumenter til {typ}.new", node.line)
            values[names[k]] = self.expr(unit, arg)
        for key, arg in node.kwargs:
            if key not in names:
                raise PineError(f"typen {typ} har ikke feltet «{key}»", node.line)
            values[key] = self.expr(unit, arg)
        codes = [values[f] if f in values else (self.expr(unit, d) if d is not None else "NAN") for f, d in fields]
        return f"new_obj_({typ!r}, {tuple(names)!r}, ({''.join(c + ', ' for c in codes)}))"

    def prelude_call(self, unit, q, node):
        name = PRELUDE_NAMES[q]
        if name not in self.functions:
            self.function(_prelude()[name])
        return self.user_call(unit, name, node)

    # ------------------------------------------------ setninger

    def statements(self, unit, body):
        if not body:
            self.emit(unit, "pass")
        for stmt in body:
            self.statement(unit, stmt)

    def scoped(self, unit, body):
        unit.scopes.append({})
        self.statements(unit, body)
        unit.scopes.pop()

    def statement(self, unit, node):
        line = getattr(node, "line", 0)
        if isinstance(node, P.FuncDef):
            if unit.is_function or len(unit.scopes) > 1:
                raise PineError("funksjoner kan bare defineres på øverste nivå", line)
            self.function(node)
        elif isinstance(node, P.TypeDef):
            if node.name in self.types:
                raise PineError(f"typen «{node.name}» er definert to ganger", line)
            self.types[node.name] = node.fields
        elif isinstance(node, P.EnumDef):
            values = {}
            for member, title in node.members:
                values[member] = _const(title) if title is not None else member
                if values[member] is None:
                    raise PineError("enum-titler må være tekst", line)
            self.enums[node.name] = values
        elif isinstance(node, P.FieldAssign):
            obj = self.fresh("_t")
            self.emit(unit, f"{obj} = {self.expr(unit, node.obj)}", line)
            value = self.value_code(unit, node.expr, line)
            if node.op != ":=":
                current = f"getf_({obj}, {node.field!r})"
                op = node.op[0]
                value = f"add_({current}, {value})" if op == "+" else f"div_({current}, {value})" if op == "/" \
                    else f"mod_({current}, {value})" if op == "%" else f"({current} {op} {value})"
            self.emit(unit, f"setf_({obj}, {node.field!r}, {value})", line)
        elif isinstance(node, P.Decl):
            self.declaration(unit, node)
        elif isinstance(node, P.TupleDecl):
            value = self.value_code(unit, node.expr, line)
            temp = self.fresh("_t")
            self.emit(unit, f"{temp} = unpack_({value}, {len(node.names)})", line)
            for k, name in enumerate(node.names):
                if name == "_":
                    continue
                if node.reassign:
                    self.assign_value(unit, name, f"{temp}[{k}]", line)
                else:
                    binding = self.declare(unit, name)
                    self.store_new(unit, binding, f"{temp}[{k}]", line)
        elif isinstance(node, P.Assign):
            value = self.value_code(unit, node.expr, line)
            if node.op != ":=":
                binding = self.lookup(unit, node.name)
                if binding is None:
                    raise PineError(f"«{node.name}» er ikke deklarert", line)
                current = self.read(binding)
                op = node.op[0]
                value = f"add_({current}, {value})" if op == "+" else f"div_({current}, {value})" if op == "/" \
                    else f"mod_({current}, {value})" if op == "%" else f"({current} {op} {value})"
            self.assign_value(unit, node.name, value, line)
        elif isinstance(node, P.ExprStmt):
            if isinstance(node.expr, P.BlockExpr):
                self.block_value(unit, [node.expr.stmt], None)
            else:
                self.emit(unit, self.expr(unit, node.expr), line)
        elif isinstance(node, P.If):
            self.if_stmt(unit, node, None)
        elif isinstance(node, P.Switch):
            self.switch_stmt(unit, node, None)
        elif isinstance(node, P.For):
            self.for_stmt(unit, node)
        elif isinstance(node, P.ForIn):
            self.forin_stmt(unit, node)
        elif isinstance(node, P.While):
            self.emit(unit, f"while t_({self.expr(unit, node.cond)}):", line)
            unit.indent += 1
            self.emit(unit, "CLK()", line)
            self.scoped(unit, node.body)
            unit.indent -= 1
        elif isinstance(node, P.Break):
            self.emit(unit, "break", line)
        elif isinstance(node, P.Continue):
            self.emit(unit, "continue", line)
        else:
            raise PineError(f"setningen støttes ikke ({type(node).__name__})", line)

    def value_code(self, unit, expr, line):
        if isinstance(expr, P.BlockExpr):
            target = self.fresh("_t")
            self.block_value(unit, [expr.stmt], target)
            return target
        return self.expr(unit, expr)

    def declaration(self, unit, node):
        line = node.line
        if node.mode:
            binding = self.declare(unit, node.name, force_list=True)
            self.emit(unit, f"if {binding.code}:", line)
            self.emit(unit, f"    {binding.code}.append({binding.code}[-1])", line)
            self.emit(unit, "else:", line)
            unit.indent += 1
            value = self.value_code(unit, node.expr, line)
            self.emit(unit, f"{binding.code}.append({value})", line)
            unit.indent -= 1
            return binding
        value = self.value_code(unit, node.expr, line)
        binding = self.declare(unit, node.name)
        self.store_new(unit, binding, value, line)
        return binding

    def store_new(self, unit, binding, value, line):
        if binding.kind == "list":
            self.emit(unit, f"{binding.code}.append({value})", line)
        else:
            self.emit(unit, f"{binding.code} = {value}", line)

    def assign_value(self, unit, name, value, line):
        binding = self.lookup(unit, name)
        if binding is None:
            raise PineError(f"«{name}» er ikke deklarert (bruk = for ny variabel)", line)
        if binding.kind == "loop":
            raise PineError(f"løkkevariabelen «{name}» kan ikke endres", line)
        if binding.is_global:
            raise PineError(f"en funksjon kan ikke endre den globale variabelen «{name}»", line)
        if binding.kind == "list":
            self.emit(unit, f"{binding.code}[-1] = {value}", line)
        elif binding.kind == "param":
            raise PineError(f"parameteren «{name}» kan ikke endres", line)
        else:
            self.emit(unit, f"{binding.code} = {value}", line)

    def block_value(self, unit, body, target):
        """Kjør ``body``; siste uttrykk (også i ``if``/``switch``) blir verdien i ``target``."""
        if target is not None:
            self.emit(unit, f"{target} = NAN")
        if not body:
            return
        for stmt in body[:-1]:
            self.statement(unit, stmt)
        last = body[-1]
        line = getattr(last, "line", 0)
        if isinstance(last, P.ExprStmt):
            if isinstance(last.expr, P.BlockExpr):
                inner = last.expr.stmt
                if isinstance(inner, P.If):
                    self.if_stmt(unit, inner, target)
                else:
                    self.switch_stmt(unit, inner, target)
            elif isinstance(last.expr, P.TupleLit):
                items = ", ".join(self.expr(unit, x) for x in last.expr.items)
                self.emit(unit, f"{target} = ({items},)" if target else f"({items},)", line)
            elif target is not None:
                self.emit(unit, f"{target} = {self.expr(unit, last.expr)}", line)
            else:
                self.emit(unit, self.expr(unit, last.expr), line)
        elif isinstance(last, P.If):
            self.if_stmt(unit, last, target)
        elif isinstance(last, P.Switch):
            self.switch_stmt(unit, last, target)
        elif isinstance(last, P.Decl):
            binding = self.declaration(unit, last)
            if target is not None:
                self.emit(unit, f"{target} = {self.read(binding)}", line)
        elif isinstance(last, P.Assign):
            self.statement(unit, last)
            if target is not None:
                self.emit(unit, f"{target} = {self.read(self.lookup(unit, last.name))}", line)
        else:
            self.statement(unit, last)

    def branch(self, unit, body, target):
        unit.indent += 1
        unit.scopes.append({})
        if target is None:
            self.statements(unit, body)
        else:
            self.block_value(unit, body, target)
        unit.scopes.pop()
        unit.indent -= 1

    def if_stmt(self, unit, node, target):
        if target is not None:
            self.emit(unit, f"{target} = NAN", node.line)
        self.emit(unit, f"if t_({self.expr(unit, node.cond)}):", node.line)
        self.branch(unit, node.body, target)
        if node.elifs or node.orelse is not None:
            self.emit(unit, "else:", node.line)
            unit.indent += 1
            if node.elifs:
                cond, body = node.elifs[0]
                rest = P.If(cond, body, node.elifs[1:], node.orelse, node.line)
                self.if_stmt(unit, rest, target)
            else:
                unit.indent -= 1
                self.branch(unit, node.orelse, target)
                unit.indent += 1
            unit.indent -= 1

    def switch_stmt(self, unit, node, target):
        if target is not None:
            self.emit(unit, f"{target} = NAN", node.line)
        subject = None
        if node.subject is not None:
            subject = self.fresh("_s")
            self.emit(unit, f"{subject} = {self.expr(unit, node.subject)}", node.line)
        first = True
        for key, body in node.cases:
            if key is None:
                self.emit(unit, "else:" if not first else "if True:", node.line)
            else:
                cond = f"{subject} == {self.expr(unit, key)}" if subject else f"t_({self.expr(unit, key)})"
                self.emit(unit, f"{'if' if first else 'elif'} {cond}:", node.line)
            self.branch(unit, body, target)
            first = False
            if key is None:
                break

    def for_stmt(self, unit, node):
        line = node.line
        a, b, s, var = self.fresh("_a"), self.fresh("_b"), self.fresh("_x"), self.fresh("L")
        self.emit(unit, f"{a} = float_({self.expr(unit, node.start)})", line)
        self.emit(unit, f"{b} = float_({self.expr(unit, node.end)})", line)
        step = f"abs(float_({self.expr(unit, node.step)}))" if node.step is not None else "1.0"
        self.emit(unit, f"{s} = ({step}) * (1.0 if {a} <= {b} else -1.0)", line)
        self.emit(unit, f"{var} = {a} - {s}", line)
        self.emit(unit, f"while {a} == {a} and {b} == {b} and {s} == {s} and {s} != 0:", line)
        unit.indent += 1
        self.emit(unit, f"{var} += {s}", line)
        self.emit(unit, f"if ({var} > {b}) if {s} > 0 else ({var} < {b}):", line)
        self.emit(unit, "    break", line)
        self.emit(unit, "CLK()", line)
        unit.scopes.append({node.var: Binding("loop", var)})
        self.statements(unit, node.body)
        unit.scopes.pop()
        unit.indent -= 1

    def forin_stmt(self, unit, node):
        line = node.line
        iterable = self.expr(unit, node.iterable)
        names = [self.fresh("L") for _ in node.names]
        if len(names) == 1:
            self.emit(unit, f"for {names[0]} in items_({iterable}):", line)
        else:
            self.emit(unit, f"for {names[0]}, {names[1]} in eitems_({iterable}):", line)
        unit.indent += 1
        self.emit(unit, "CLK()", line)
        unit.scopes.append({n: Binding("loop", code) for n, code in zip(node.names, names)})
        self.statements(unit, node.body)
        unit.scopes.pop()
        unit.indent -= 1

    # ------------------------------------------------ funksjoner

    def function(self, node):
        key = node.name
        if node.method_of is not None:
            receiver = node.method_of
            receiver = "drawing" if receiver in ("line", "label", "box", "table", "linefill", "polyline") else receiver
            key = f"{node.name}@{receiver}"
            self.methods.setdefault(node.name, []).append((receiver, key))
        if key in self.functions:
            raise PineError(f"funksjonen «{node.name}» er definert to ganger", node.line)
        factory = self.fresh("F")
        params = node.params
        self.functions[key] = (node, factory, params)
        unit = Unit(factory, True, _needs_list(node.body))
        hist_params = _needs_list(node.body)
        args = []
        for pname, _ in params:
            code = self.fresh("PA")
            hist = None
            if pname in hist_params:
                hist = self.fresh("PS")
                unit.lists.append(hist)
            unit.scopes[0][pname] = Binding("param", code, hist)
            args.append((code, hist))
        unit.indent = 2
        for code, hist in args:
            if hist:
                self.emit(unit, f"{hist}.append({code})", node.line)
        result = self.fresh("_t")
        self.block_value(unit, node.body, result)
        self.emit(unit, f"return {result}", node.line)
        lines = [(f"    def {factory}():", node.line)]
        lines += [(f"        {name} = REG()", node.line) for name in unit.lists]
        lines += [(f"        {name} = SITE({q!r})", node.line) for name, q in unit.sites]
        lines += [(f"        {name} = {fac}()", node.line) for name, fac in unit.instances]
        lines.append((f"        def call(i{''.join(', ' + c for c, _ in args)}):", node.line))
        lines += [("    " + text, ln) for text, ln in unit.body]
        lines.append(("        return call", node.line))
        self.factories.append(lines)

    # ------------------------------------------------ helhet

    def compile(self):
        unit = Unit("main", False, _needs_list([n for n in self.program.body if not isinstance(n, P.FuncDef)]))
        self.global_unit = unit
        unit.indent = 2
        for stmt in self.program.body:
            self.statement(unit, stmt)
        lines = [("def make_script(RT):", 0)]
        for name in ("O", "H", "L", "C", "V", "T", "TC", "HL2", "HLC3", "OHLC4", "HLCC4", "YEAR", "MONTH", "DOM",
                     "DOW", "HOUR", "MINUTE", "SECOND", "WOY", "TDAY", "LAST", "LAST_TIME", "IN", "SRCS", "EV",
                     "DR", "BR", "SEC", "CLK", "REG", "SITE", "TICKERID", "TICKER", "MINTICK", "CURRENCY",
                     "BASECURRENCY", "SYMTYPE", "PREFIX", "PRICESCALE", "TF_PERIOD", "TF_MAIN", "TF_MULT",
                     "TF_INTRADAY", "TF_DAILY"):
            lines.append((f"    {name} = RT.{name}", 0))
        lines += [(f"    {name} = REG()", 0) for name in unit.lists]
        lines += [(f"    {name} = SITE({q!r})", 0) for name, q in unit.sites]
        for factory in self.factories:
            lines += factory
        lines += [(f"    {name} = {fac}()", 0) for name, fac in unit.instances]
        lines.append(("    def bar(i):", 0))
        lines += unit.body or [("        pass", 0)]
        lines.append(("    return bar", 0))
        code = "\n".join(text for text, _ in lines) + "\n"
        self.line_map = {k + 1: ln for k, (_, ln) in enumerate(lines)}
        _validate(code)
        return code


COLOR_NAMES = {
    "aqua": "#00BCD4", "black": "#363A45", "blue": "#2962FF", "fuchsia": "#E040FB", "gray": "#787B86",
    "green": "#4CAF50", "lime": "#00E676", "maroon": "#880E4F", "navy": "#311B92", "olive": "#808000",
    "orange": "#FF9800", "purple": "#9C27B0", "red": "#F23645", "silver": "#B2B5BE", "teal": "#089981",
    "white": "#FFFFFF", "yellow": "#FDD835",
}
RUNTIME_NAMES = {
    "t_", "add_", "div_", "mod_", "hist_", "bhist_", "hpush_", "nz_", "na_", "int_", "float_", "bool_", "iff_",
    "ident_", "unpack_", "items_", "eitems_", "method_", "noop_", "draw_", "out_signal", "out_alert", "out_arrow",
    "out_label", "out_plot", "time_fn", "tr_", "ticker_ha", "ticker_new", "ticker_modify", "runtime_error",
    "tf_seconds_of", "tf_from_seconds", "timestamp_", "a_new", "NAN", "len", "abs", "True", "False", "None",
    "new_obj_", "getf_", "setf_", "copy_obj_", "mcall_", "point_new", "float",
}
RT_ATTRS = {"O", "H", "L", "C", "V", "T", "TC", "HL2", "HLC3", "OHLC4", "HLCC4", "YEAR", "MONTH", "DOM", "DOW",
            "HOUR", "MINUTE", "SECOND", "WOY", "TDAY", "LAST", "LAST_TIME", "IN", "SRCS", "EV", "DR", "BR", "SEC",
            "CLK", "REG", "SITE", "TICKERID", "TICKER", "MINTICK", "CURRENCY", "BASECURRENCY", "SYMTYPE", "PREFIX",
            "PRICESCALE", "TF_PERIOD", "TF_MAIN", "TF_MULT", "TF_INTRADAY", "TF_DAILY"}
BROKER_ATTRS = {"pos", "avg_price", "opentrades", "closedtrades", "equity_now", "netprofit", "openprofit",
                "initial", "wintrades", "losstrades", "grossprofit", "grossloss", "entry_name", "entry", "order",
                "exit", "close", "close_all", "cancel", "cancel_all", "allow_entry_in"}
GENERATED = re.compile(r"(S|v|L|K|U|H|F|PA|PS|_t|_s|_a|_b|_x)\d+")
PURE_IMPLS = {impl for impl, _ in PURE.values()} | set(VARARGS.values())


def _validate(code):
    """Forsvar i dybden: den genererte koden får bare bruke kjente navn og attributter."""
    tree = ast.parse(code, filename="<pine>")
    allowed_names = RUNTIME_NAMES | PURE_IMPLS | RT_ATTRS | {"RT", "i", "call", "bar", "make_script"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Lambda, ast.ClassDef,
                             ast.With, ast.Try, ast.Raise, ast.Yield, ast.Await, ast.Delete)):
            raise PineError("intern sikkerhetssjekk: ulovlig konstruksjon i oversatt kode")
        if isinstance(node, ast.Attribute):
            base = node.value.id if isinstance(node.value, ast.Name) else None
            ok = (base == "RT" and node.attr in RT_ATTRS) or (base == "BR" and node.attr in BROKER_ATTRS) or \
                (node.attr == "append" and base is not None and GENERATED.fullmatch(base))
            if not ok or node.attr.startswith("_"):
                raise PineError(f"intern sikkerhetssjekk: attributtet «{node.attr}» er ikke tillatt")
        if isinstance(node, ast.Name) and node.id not in allowed_names and not GENERATED.fullmatch(node.id):
            raise PineError(f"intern sikkerhetssjekk: navnet «{node.id}» er ikke tillatt")
        if isinstance(node, ast.FunctionDef) and not (node.name in ("make_script", "bar", "call") or
                                                      GENERATED.fullmatch(node.name)):
            raise PineError("intern sikkerhetssjekk: ukjent funksjonsdefinisjon")


@dataclass
class Compiled:
    code: str
    version: int
    decl: dict
    inputs: list
    outputs: list
    securities: list
    warnings: list
    uses_strategy: bool
    line_map: dict
    cleaned: str = ""         # Pine-koden som faktisk ble lest (etter rensing av innlimt tekst)
    clean_note: str = ""

    @property
    def is_strategy(self):
        return self.decl.get("kind") == "strategy"

    @property
    def title(self):
        return self.decl.get("title") or ""


def compile_source(source):
    """Pine-kilde → ``Compiled`` (Python-kode og informasjon om input, signaler og deklarasjon)."""
    cleaned, note = P.extract_pine(source)
    program = P.parse(cleaned)
    compiler = Compiler(program)
    code = compiler.compile()
    return Compiled(code, program.version, compiler.decl, compiler.inputs, compiler.outputs, compiler.securities,
                    compiler.warnings, compiler.uses_strategy, compiler.line_map, cleaned, note or "")
