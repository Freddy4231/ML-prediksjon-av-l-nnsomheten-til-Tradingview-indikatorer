"""Pine-test nederst på ML-siden: lim inn kildekoden til et Pine Script og se hvor lønnsomt det er på f.eks. BTC.

Enkel bruk: lim inn koden, velg marked (BTC som standard) og periode, trykk én knapp. Svaret vises
i klartekst: tjente skriptet penger, og slo det å bare kjøpe og holde? Intervaller, kostnad,
retning, signalvalg og skriptets input ligger under «Avanserte valg». Skriptet kjøres lys for lys
med ``pine_engine`` (neste åpning, kostnad per side, faktisk funding for krypto, hele kapitalen
per posisjon, 1×). Maskinlæringsdelen forutsier om hver handel skriptet gjør, blir lønnsom.
Ingen ordre sendes noe sted; visningskopien har ingen kobling til noen børskonto.
"""
import time
from datetime import date, timedelta
from pathlib import Path

import gradio as gr
import pandas as pd

from pine_engine import backtest as B
from pine_engine import ml as ML
from pine_engine.data import DataError, SYMBOL
from pine_engine import parser as P
from pine_engine.parser import PineError

ROOT = Path(__file__).resolve().parent
SIMPLE_EXAMPLE = """//@version=5
strategy("EMA-kryss 9/21", overlay=true)
fastLen = input.int(9, "Rask EMA")
slowLen = input.int(21, "Treg EMA")
fast = ta.ema(close, fastLen)
slow = ta.ema(close, slowLen)
if ta.crossover(fast, slow)
    strategy.entry("Long", strategy.long)
if ta.crossunder(fast, slow)
    strategy.entry("Short", strategy.short)
plot(fast, "Rask", color.green)
plot(slow, "Treg", color.red)
"""
EXAMPLES = {
    "TMG Trades · Stonksy + VRVP POC & Liquidity Trap (v6)": "tradingview/tmg_trades_strategy.pine",
    "Market Oracle Pro · estimert rekonstruksjon (v6)": "tradingview/market_oracle_reconstruction.pine",
    "Kinetic Scalper · uavhengig, delvis estimert rekonstruksjon (v6)": "tradingview/kinetic_scalper_reconstruction.pine",
    "Momentum Market Structure · uavhengig rekonstruksjon, strategi (v6)":
        "tradingview/momentum_market_structure_recon.pine",
    "Pro Swing · Public Output Reconstruction (v6)": "tradingview/pro_swing_recon.pine",
    "Pro Swing · strategi med stopp/mål, snu eller følg kanalen (v6)": "tradingview/pro_swing_strategy.pine",
    "PRL Recon · long når kanalen blir blå, short når den blir rød (v6)": "tradingview/prl_recon_kanalfarge.pine",
    "Chandelier Exit (22, 3) · samme regel som skript-feltene (v5)": "tradingview/chandelier_exit.pine",
    "Enkel EMA-kryss-strategi 9/21 (v5)": None,
    "UT Bot Alerts (indikator, v6)": "tradingview/ut_bot_alerts.pine",
    "SuperTrend STRATEGY · KivancOzbilgic (v4)": "tradingview/supertrend_kivanc_strategy.pine",
    "Clean Wave Matrix PRO · Dark_Ace_Master (v6)": "tradingview/clean_wave_original_indicator.pine",
    "Triple Supertrend Confluence · MarkitTick (v6)": "tradingview/triple_supertrend_confluence.pine",
}
MARKET_CHOICES = [
    ("BTC · Bitcoin", "binance:BTCUSDT"), ("ETH · Ethereum", "binance:ETHUSDT"), ("SOL · Solana", "binance:SOLUSDT"),
    ("BNB", "binance:BNBUSDT"), ("XRP", "binance:XRPUSDT"), ("ADA · Cardano", "binance:ADAUSDT"),
    ("DOGE · Dogecoin", "binance:DOGEUSDT"), ("LTC · Litecoin", "binance:LTCUSDT"),
    ("AVAX · Avalanche", "binance:AVAXUSDT"), ("LINK · Chainlink", "binance:LINKUSDT"),
    ("S&P 500 (SPY)", "yahoo:SPY"), ("Nasdaq 100 (QQQ)", "yahoo:QQQ"), ("Apple (AAPL)", "yahoo:AAPL"),
    ("Nvidia (NVDA)", "yahoo:NVDA"), ("Tesla (TSLA)", "yahoo:TSLA"), ("Gull (GC=F)", "yahoo:GC=F"),
    ("Olje (CL=F)", "yahoo:CL=F"), ("EUR/USD", "yahoo:EURUSD=X"), ("Equinor (EQNR.OL)", "yahoo:EQNR.OL"),
]
MARKET_NAMES = {value: label.split(" · ")[0].split(" (")[0] for label, value in MARKET_CHOICES}
PERIODS = ["Siste 30 dager", "Siste 90 dager", "Siste 180 dager", "Siste år", "2025", "2026 hittil",
           "Egne datoer (under avanserte valg)"]
DEFAULT_TFS = ["1 min", "5 min", "30 min", "45 min", "1 time"]
AUTO_LABEL = "Automatisk"
NONE_LABEL = "Ingen"
INTRO = """
## Pine-test · test et Pine Script på BTC

**1.** Lim inn Pine-koden fra TradingView (Pine Editor → merk alt → kopier). **2.** Velg marked og periode.
**3.** Trykk **Test lønnsomhet**. Du får svar på om skriptet tjente penger etter gebyr, og om det slo
å bare kjøpe og holde, på 1, 5, 30, 45 og 60 minutters lys.
"""
HOW = """
**Slik testes skriptet:** det kjøres lys for lys på ekte priser. Strategier (`strategy(...)`) handler med sine
egne `strategy.entry/exit/close`-ordre. Indikatorer handler på signalene de viser (`plotshape`, `alertcondition`,
`label.new` med BUY/SELL eller lignende): long ved kjøpssignal, short ved salgssignal. Ordren fylles ved neste
åpning etter signallyset (som TradingView), med 0,10 % kostnad per side, faktisk funding for krypto og hele
kapitalen per handel (ingen giring). Signaler som tegnes bakover i tid, telles først når de faktisk dukker opp.

**Støttes:** variabler, `var`, `:=`, tupler, `if`/`for`/`while`/`switch`, egne funksjoner, egne typer (`type`)
med felt, `method`, `enum`, historikk `x[n]`, de fleste `ta.*`-, `math.*`-, `str.*`-, `array.*`- og `map.*`-funksjoner,
`chart.point`, `input.*`, `strategy.*` og `request.security` mot høyere tidsramme eller Heikin Ashi.
**Støttes ikke ennå:** `import` av biblioteker og `matrix`. **Data:** Binance har full historikk fra 2017. Yahoo har ca. 30 dager med 1 min, 60 dager med
2–90 min og 2 år med 1 time. Maks 400 000 lys per intervall (1 min ≈ 9 måneder).

> **Historisk test, ikke en garanti.** Velger du intervallet eller perioden med best resultat i etterkant, er det
> kurvetilpasning. Ikke finansiell rådgivning.
"""


def _pct(value, digits=2, sign=True):
    if value is None:
        return "–"
    if value == float("inf"):
        return "∞"
    text = f"{value * 100:+.{digits}f} %" if sign else f"{value * 100:.{digits}f} %"
    return text.replace(".", ",").replace("-", "−")


def _num(value, digits=2):
    if value is None:
        return "–"
    if value == float("inf"):
        return "∞"
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def load_example(name):
    path = EXAMPLES.get(name)
    if name not in EXAMPLES:
        return gr.update()
    return SIMPLE_EXAMPLE if path is None else (ROOT / path).read_text(encoding="utf-8")


def preset_dates(preset):
    """Periodevalget → (fra, til) som ÅÅÅÅ-MM-DD. «Egne datoer» returnerer None (bruk feltene)."""
    today = date.today()
    tomorrow = (today + timedelta(days=1)).isoformat()
    if preset == "2025":
        return "2025-01-01", "2026-01-01"
    if preset == "2026 hittil":
        return "2026-01-01", tomorrow
    days = {"Siste 30 dager": 30, "Siste 90 dager": 90, "Siste 180 dager": 180, "Siste år": 365}.get(preset)
    if days is None:
        return None
    return (today - timedelta(days=days)).isoformat(), tomorrow


def parse_market(value):
    """Markedsvalget → (kilde, symbol). Egen tekst: «SOLUSDT» = Binance, ellers Yahoo (f.eks. «AAPL»)."""
    text = (value or "binance:BTCUSDT").strip()
    if ":" in text and text.split(":", 1)[0] in ("binance", "yahoo"):
        source, symbol = text.split(":", 1)
        return source, symbol
    upper = text.upper().replace("/", "").replace("-", "")
    if SYMBOL.match(upper) and upper.endswith(("USDT", "USDC", "BTC", "ETH", "FDUSD")):
        return "binance", upper
    if SYMBOL.match(upper + "USDT") and len(upper) <= 6 and upper in {"BTC", "ETH", "SOL", "BNB", "XRP", "ADA",
                                                                       "DOGE", "LTC", "AVAX", "LINK", "DOT", "TRX",
                                                                       "SUI", "NEAR", "TON", "PEPE", "SHIB"}:
        return "binance", upper + "USDT"
    return "yahoo", text


def _inputs_frame(compiled):
    rows = [{"Nr": item["index"] + 1, "Navn": item["title"], "Type": item["kind"],
             "Verdi": item["default"] if not isinstance(item["default"], float) or item["default"] != int(item["default"])
             else int(item["default"])} for item in compiled.inputs if item["kind"] not in ("color",)]
    return pd.DataFrame(rows, columns=["Nr", "Navn", "Type", "Verdi"])


def _signal_choices(candidates=None, compiled=None):
    choices = [(AUTO_LABEL, B.AUTO), (NONE_LABEL, "")]
    seen = set()
    for c in candidates or []:
        tag = {"long": " → long", "short": " → short"}.get(c["class"], "")
        choices.append((f"{c['label'][:70]} · {c['count']} treff{tag}", c["key"]))
        seen.add(c["key"])
    return choices


def read_script(code):
    """Les skriptet med en gang det limes inn: kort bekreftelse (eller feil med linjenummer) og input-tabellen.

    Er hele TradingView-siden limt inn (overskrift, «Copy code», linjenummer), byttes feltet ut med selve koden.
    """
    empty = pd.DataFrame(columns=["Nr", "Navn", "Type", "Verdi"])
    if not code or not code.strip():
        return "", empty, gr.update(), gr.update(), gr.update()
    cleaned, note = P.extract_pine(code)
    new_code = gr.update(value=cleaned) if note and cleaned.strip() != code.strip() else gr.update()
    prefix = f"✂️ {note}  \n" if note else ""
    try:
        compiled = B.analyze(cleaned)
    except PineError as exc:
        return f"{prefix}❌ **Skriptet kan ikke leses:** {exc}", empty, gr.update(), gr.update(), new_code
    kind = "strategi – testes med skriptets egne kjøp/salg-ordre" if compiled.is_strategy else \
        "indikator – testes på kjøps- og salgssignalene den viser"
    lines = [f"{prefix}✅ Lest: **{compiled.title or 'skript uten tittel'}** (Pine v{compiled.version}, {kind})."]
    lines += [f"⚠️ {w}" for w in compiled.warnings]
    choices = _signal_choices()
    return "  \n".join(lines), _inputs_frame(compiled), \
        gr.update(choices=choices, value=B.AUTO), gr.update(choices=choices, value=B.AUTO), new_code


def _overrides(compiled, frame):
    if frame is None or len(frame) == 0:
        return {}
    by_index = {item["index"]: item for item in compiled.inputs}
    out = {}
    for _, row in pd.DataFrame(frame).iterrows():
        try:
            index = int(row["Nr"]) - 1
        except (TypeError, ValueError):
            continue
        item = by_index.get(index)
        if item is None:
            continue
        value = row["Verdi"]
        default = item["default"]
        if str(value).strip() != str(default if not isinstance(default, float) or default != int(default)
                                     else int(default)).strip():
            out[index] = value
    return out


def _figure(results):
    import plotly.graph_objects as go
    fig = go.Figure()
    for item in results:
        c = item["curve"]
        fig.add_trace(go.Scatter(x=c["time"], y=[(v - 1) * 100 for v in c["equity"]], mode="lines",
                                 name=item["label"]))
    if results:
        c = min(results, key=lambda r: r["minutes"])["curve"]
        fig.add_trace(go.Scatter(x=c["time"], y=[(v - 1) * 100 for v in c["buy_hold"]], mode="lines",
                                 name="Kjøp og hold", line={"dash": "dash", "color": "#787B86"}))
    fig.update_layout(template="plotly_white", height=420, margin={"l": 40, "r": 20, "t": 30, "b": 40},
                      yaxis_title="Avkastning (%)", legend={"orientation": "h", "y": -0.2}, hovermode="x unified")
    return fig


def _results_frame(results):
    rows = []
    for item in results:
        s = item["stats"]
        rows.append({
            "Intervall": item["label"], "Resultat etter gebyr": _pct(s["net"]),
            "Lønnsom?": "Ja" if s["net"] > 0 else "Nei", "Kjøp og hold": _pct(s["buy_hold"]),
            "Bedre enn å holde?": "Ja" if s["net"] > s["buy_hold"] else "Nei",
            "Handler": s["trades"], "Treffrate": _pct(s["win_rate"], 0, sign=False),
            "Største fall": _pct(s["max_drawdown"], 1),
        })
    return pd.DataFrame(rows)


def _details_frame(results):
    rows = []
    for item in results:
        s = item["stats"]
        rows.append({
            "Intervall": item["label"], "Fra (UTC)": item["start"][:16], "Til (UTC)": item["end"][:16],
            "Lys": item["bars"], "Long / short": f"{s['long_trades']} / {s['short_trades']}",
            "Profittfaktor": _num(s["profit_factor"]), "Gebyr": _pct(-s["fees"], 2),
            "Funding": _pct(-s["funding"], 2), "Tid i markedet": _pct(s["exposure"], 0, sign=False),
            "Sharpe (dag)": _num(s["sharpe"]), "Data": f"{item['base']} min → {item['minutes']} min",
        })
    return pd.DataFrame(rows)


def verdict(results, title, market_name, period_text, chosen=None, long_only=False):
    """Kort svar i klartekst øverst: tjente skriptet penger, og slo det kjøp og hold?"""
    n = len(results)
    winners = [r for r in results if r["stats"]["net"] > 0]
    beat = [r for r in results if r["stats"]["net"] > r["stats"]["buy_hold"]]
    best = max(results, key=lambda r: r["stats"]["net"])
    worst = min(results, key=lambda r: r["stats"]["net"])
    if len(winners) == n:
        head = f"🟢 Ja – skriptet tjente penger på alle {n} intervallene"
    elif winners:
        head = f"🟡 Delvis – skriptet tjente penger på {len(winners)} av {n} intervaller"
    else:
        head = f"🔴 Nei – skriptet tapte penger på alle {n} intervallene"
    hold = best["stats"]["buy_hold"]
    lines = [f"### {head}",
             f"**{title or 'Skriptet'}** på **{market_name}**, {period_text}:",
             f"- Best: **{best['label']}** med **{_pct(best['stats']['net'])}** etter gebyr "
             f"({best['stats']['trades']} handler). Å bare kjøpe og holde {market_name} ga {_pct(hold)}.",
             f"- Slo kjøp og hold på **{len(beat)} av {n}** intervaller."]
    if worst is not best:
        lines.append(f"- Dårligst: {worst['label']} med {_pct(worst['stats']['net'])}"
                     + (" (tapte hele kapitalen)." if worst["stats"]["bankrupt"] else "."))
    if chosen and results[0]["mode"] == "signaler":
        cand = {c["key"]: c for c in results[0]["candidates"]}
        long_label = cand[chosen["long"]]["label"] if chosen.get("long") in cand else "ingen"
        short_label = cand[chosen["short"]]["label"] if chosen.get("short") in cand else "ingen"
        lines.append(f"- Signaler brukt: kjøp = {long_label}, {'ut' if long_only else 'salg/short'} = {short_label} "
                     "(kan endres under «Avanserte valg»).")
    return "\n".join(lines)


def _settings(code, market, period, start, end, timeframes, direction, cost, funding, use_orders, long_sel,
              short_sel, inputs_frame):
    """Valgene på siden → (oversatt skript, Settings, markedsnavn, periodetekst). Felles for test og ML."""
    dates = preset_dates(period)
    if dates is not None:
        start, end = dates
    source, symbol = parse_market(market)
    compiled = B.analyze(code)
    pd.Timestamp(start), pd.Timestamp(end)
    settings = B.Settings(
        source=source, symbol=symbol, start=start, end=end,
        timeframes=[B.TIMEFRAMES[t] for t in timeframes if t in B.TIMEFRAMES],
        long_only=direction == "Kun long", cost=max(0.0, float(cost or 0)) / 100,
        funding=bool(funding) and source == "binance", overrides=_overrides(compiled, inputs_frame),
        long_key=long_sel if long_sel is not None else B.AUTO,
        short_key=short_sel if short_sel is not None else B.AUTO, use_orders=bool(use_orders))
    period_text = period.lower() if dates is not None else f"{start} til {end}"
    return compiled, settings, MARKET_NAMES.get(market, symbol), period_text


def run_test(code, market, period, start, end, timeframes, direction, cost, funding, use_orders, long_sel,
             short_sel, inputs_frame, progress=gr.Progress()):
    empty = pd.DataFrame()
    hidden = gr.update(visible=False)
    nothing = ("", hidden, hidden, empty, gr.update(), empty, None, gr.update(), gr.update())
    if not code or not code.strip():
        return ("Lim inn Pine-koden i feltet over først.",) + nothing[1:]
    if not timeframes:
        return ("Velg minst ett intervall under «Avanserte valg».",) + nothing[1:]
    try:
        compiled, settings, market_name, period_text = _settings(
            code, market, period, start, end, timeframes, direction, cost, funding, use_orders, long_sel, short_sel,
            inputs_frame)
        report = B.run(code, settings, progress=progress)
    except PineError as exc:
        return (f"❌ **Skriptet kan ikke testes:** {exc}",) + nothing[1:]
    except (ValueError, DataError) as exc:
        return (f"❌ **Feil i valgene:** {exc}",) + nothing[1:]
    results = report["results"]
    if not results:
        problems = "\n".join(f"- {label}: {error}" for label, error in report["errors"])
        return (f"❌ **Ingen intervaller kunne testes.**\n{problems}",) + nothing[1:]
    lines = [verdict(results, compiled.title, market_name, period_text, report["chosen"], settings.long_only)]
    lines += [f"- ❌ {label}: {error}" for label, error in report["errors"]]
    lines += [f"- ⚠️ {w}" for w in compiled.warnings]
    lines.append("\n*Historisk test med 0,10 % kostnad per side"
                 + (" og faktisk funding" if settings.funding else "") + ". Ikke finansiell rådgivning.*")
    labels = [r["label"] for r in results]
    state = {r["label"]: r["trades"] for r in results}
    choices = _signal_choices(results[0]["candidates"])
    return ("\n".join(lines), gr.update(value=_results_frame(results), visible=True),
            gr.update(value=_figure(results), visible=True), _details_frame(results),
            gr.update(choices=labels, value=labels[0]), pd.DataFrame(results[0]["trades"]), state,
            gr.update(choices=choices), gr.update(choices=choices))


# ---------------------------------------------------------------- maskinlæring på skriptets handler

ML_MODES = {"Voksende vindu (all kjent historikk)": "expanding", "Rullerende vindu (siste N handler)": "rolling"}
ML_COLUMNS = ["Modell", "Handler forutsagt", "Treffsikkerhet (grunnrate)", "MSE / Brier (grunnrate)",
              "Log loss (grunnrate)", "AUC", "Treningstid (median)", "Svartid per prediksjon",
              "Gjennomstrømning", "Falske pos. / neg.", "Som filter (alle handler)", "Treninger"]
ML_INTRO = """
Hver handel skriptet gjør, blir ett eksempel. Modellene får bare det som var kjent ved signallyset (pris, volatilitet,
RSI, trend, volum, tid og forrige handel) og skal anslå sannsynligheten for at handelen gir gevinst etter gebyr.

**Ingen statisk trening:** modellene trenes på nytt fortløpende fremover i tid. Hver gang brukes bare handler som var
avsluttet før signalet, i et voksende eller rullerende vindu. Alle tall gjelder handler modellen ikke hadde sett.
Velg lang periode (minst 40 handler); «Grunnrate» er å gjette andelen lønnsomme handler uten modell.
"""
ML_HOW = """
- **Treffsikkerhet (accuracy):** andel riktige ja/nei (P ≥ 0,5). Sammenlign med grunnraten; ved f.eks. 35 % lønnsomme
  handler gir «alltid nei» 65 %.
- **MSE / Brier:** kvadratisk avvik mellom sannsynlighet og fasit (0 = perfekt, 0,25 = myntkast). Lavere enn
  grunnraten = modellen vet noe. Log loss straffer sikre feil ekstra.
- **Latency og throughput:** treningstid per ny trening, svartid for én prediksjon og prediksjoner per sekund.
- **Feil-prediksjoner:** falske positive (sa gevinst, tapte) og falske negative (sa tap, tjente); kostnaden vises.
- **Feature importance:** permutasjon på ukjente handler (hvor mye MSE øker når featuren stokkes) og modellens egen.
- **Forbedring:** fjern features som ikke hjelper (feature-utvalg i hvert treningssteg gjør det automatisk med bare
  treningsdata), hev terskelen hvis falske positive koster mest, bruk rullerende vindu hvis treffsikkerheten faller
  over tid, og velg modellen med lavest MSE (ikke høyest filterresultat, som er kurvetilpasning).
"""


ML_MODELS = """
**Hvilke modeller brukes?** Klassifisering: hver modell anslår sannsynligheten for at en handel gir gevinst etter gebyr.

| Modell | Innstillinger |
|---|---|
| **Logistisk regresjon** | Manglende verdier fylles med median, standardisering, C = 0,5 |
| **Random Forest** | 200 trær, minst 5 handler per blad, tilfeldig utvalg av egenskaper (√n) per splitt |
| **Gradient boosting** (scikit-learns HistGradientBoosting, samme histogrammetode som LightGBM) | Dybde 3, læringsrate 0,05, 150 runder, minst 10 handler per blad, L2-regularisering 1,0 |
| **XGBoost / LightGBM** | Vises bare når pakkene er installert |

**Data:** skriptet kjøres lys for lys på ekte priser (fyll ved neste lys, 0,10 % kostnad per side, faktisk funding).
**Hver handel blir ett eksempel** med 20 egenskaper fra signallyset (lyset før fyllet, bare kjent historikk):
avkastning over 1, 4, 12 og 48 lys, volatilitet over 12 og 48 lys og forholdet mellom dem, ATR mot pris, RSI 14,
avstand til glidende snitt 20 og 100, plass i 48-lysers spenn, volum mot snitt, tid på døgnet og ukedag, retning
(long/short), forrige avsluttede handels resultat og tid siden den. **Fasit:** 1 hvis handelen ga gevinst etter gebyr
og funding, ellers 0.

**Trening (ikke statisk):** modellene trenes på nytt fortløpende fremover i tid (standard hver 5. handel), i voksende
eller rullerende vindu. Bare handler som var avsluttet før signalet brukes. Minst 30 treningshandler før første
spådom og minst 40 handler totalt.
"""


def _ml_table(reports, threshold):
    rows = []
    for name, r in reports.items():
        if not r.get("predicted"):
            rows.append({"Modell": name, "Handler forutsagt": 0})
            continue
        rows.append({
            "Modell": name, "Handler forutsagt": r["predicted"],
            "Treffsikkerhet (grunnrate)": f"{_pct(r['accuracy'], 1, sign=False)} ({_pct(r['baseline_accuracy'], 1, sign=False)})",
            "MSE / Brier (grunnrate)": f"{_num(r['mse'], 4)} ({_num(r['baseline_mse'], 4)})",
            "Log loss (grunnrate)": f"{_num(r['log_loss'], 3)} ({_num(r['baseline_log_loss'], 3)})",
            "AUC": _num(r["auc"], 3),
            "Treningstid (median)": f"{_num(r['fit_ms_median'], 1)} ms",
            "Svartid per prediksjon": f"{_num(r['latency_ms'], 2)} ms",
            "Gjennomstrømning": f"{r['throughput']:,.0f}/s".replace(",", " ") if r["throughput"] else "–",
            "Falske pos. / neg.": f"{r['fp']} / {r['fn']}",
            "Som filter (alle handler)": f"{_pct(r['filtered'], 1)} på {r['taken']} ({_pct(r['all_trades'], 1)})",
            "Treninger": (f"{r['refits']} (hver {r['requested_step']}.–{r['step']}., tidsmål)" if r.get("adjusted")
                          else f"{r['refits']} (hver {r['step']}.)")})
    return pd.DataFrame(rows, columns=ML_COLUMNS)


def _ml_importance_figure(result):
    import plotly.graph_objects as go
    best = result["importance"][result["best"]]
    order = list(best.index[:12])[::-1]
    fig = go.Figure()
    for name, series in result["importance"].items():
        if name.endswith("+ feature-utvalg"):
            continue
        fig.add_trace(go.Bar(y=[ML.FEATURE_LABELS.get(f, f) for f in order], orientation="h", name=name,
                             x=[float(series.get(f, 0.0)) * 1000 for f in order]))
    fig.update_layout(template="plotly_white", height=460, barmode="group", margin={"l": 40, "r": 20, "t": 30, "b": 40},
                      xaxis_title="Økning i MSE × 1000 når featuren stokkes (positiv = nyttig)",
                      legend={"orientation": "h", "y": -0.25})
    return fig


def _ml_drift_figure(result, window=20):
    import plotly.graph_objects as go
    fig = go.Figure()
    shown = False
    for name, wf in result["runs"].items():
        if name.endswith("+ feature-utvalg") or not result["reports"][name].get("predicted"):
            continue
        times, hits, base = ML.rolling_accuracy(result["table"], wf, window)
        fig.add_trace(go.Scatter(x=times, y=hits * 100, mode="lines", name=name))
        if not shown:
            fig.add_trace(go.Scatter(x=times, y=base * 100, mode="lines", name="Grunnrate",
                                     line={"dash": "dash", "color": "#787B86"}))
            shown = True
    fig.update_layout(template="plotly_white", height=380, margin={"l": 40, "r": 20, "t": 30, "b": 40},
                      yaxis_title=f"Treffsikkerhet siste {window} handler (%)", legend={"orientation": "h", "y": -0.25})
    return fig


def run_ml(code, market, period, start, end, direction, cost, funding, use_orders, long_sel, short_sel,
           inputs_frame, interval, models, mode, window, refit, min_train, threshold, select, progress=gr.Progress()):
    """Backtest på valgt intervall → fortløpende trente modeller på skriptets handler → mål, viktighet og råd."""
    hidden = gr.update(visible=False)
    nothing = (hidden, hidden, hidden, pd.DataFrame())
    if not code or not code.strip():
        return ("Lim inn Pine-koden først.",) + nothing
    models = [m for m in (models or []) if m in ML.available_models()]
    if not models:
        return ("Velg minst én modell.",) + nothing
    minutes = B.TIMEFRAMES.get(interval)
    if minutes is None:
        return ("Velg et intervall.",) + nothing
    deadline = time.monotonic() + 300
    try:
        compiled, settings, market_name, period_text = _settings(
            code, market, period, start, end, [interval], direction, cost, funding, use_orders, long_sel, short_sel,
            inputs_frame)
        progress(0.05, desc=f"{interval}: kjører skriptet")
        tested = B.run_timeframe(compiled, settings, minutes, deadline, keep_raw=True)
        raw = tested["raw"]
        progress(0.3, desc="trener modellene fortløpende")
        result = ML.run(raw["bars"], raw["trades"], raw["first"], models=models, mode=ML_MODES.get(mode, "expanding"),
                        window=int(window or 200), refit=max(1, int(refit or 5)), min_train=max(10, int(min_train or 30)),
                        threshold=float(threshold or 0.5), select=bool(select), deadline=deadline)
    except PineError as exc:
        return (f"❌ **Skriptet kan ikke testes:** {exc}",) + nothing
    except (ValueError, DataError) as exc:
        return (f"❌ **Kan ikke trene modellene:** {exc}",) + nothing
    return ml_outputs(result, compiled.title, market_name, interval, period_text, threshold)


def ml_outputs(result, title, market_name, interval, period_text, threshold):
    """Tekst, tabell, figurer og feil-prediksjoner fra ``ML.run``. Brukes også av ML-siden (/ml-skript)."""
    reports, best = result["reports"], result["best"]
    rep = reports[best]
    beats = rep["mse"] < rep["baseline_mse"]
    head = (f"### {'🟢' if beats else '🔴'} {'Beste modell slår grunnraten' if beats else 'Ingen modell slår grunnraten'}"
            f" ({best}: MSE {_num(rep['mse'], 4)} mot {_num(rep['baseline_mse'], 4)})")
    lines = [head,
             f"**{title or 'Skriptet'}** på **{market_name}**, {interval}, {period_text}: "
             f"{len(result['table'])} handler, {rep['predicted']} forutsagt uten å være sett (andel lønnsomme "
             f"{_pct(rep['base_rate'], 0, sign=False)}). Trent på nytt {rep['refits']} ganger ("
             + (f"hver {rep['requested_step']}. handel, økt underveis til hver {rep['step']}. for å holde tidsmålet"
                if rep.get("adjusted") else f"hver {rep['step']}. handel")
             + f"), treningsvindu {rep['train_sizes'][0]}–{rep['train_sizes'][1]} handler.",
             "", "**Hva tallene betyr og hvordan modellene kan forbedres:**"]
    lines += [f"- {tip}" for tip in result["tips"]]
    missing = [m for m in (ML.XGB, ML.LGBM) if m not in ML.available_models()]
    if missing:
        lines.append(f"\n*{' og '.join(missing)} er ikke installert på serveren; gradient boosting kjøres med "
                     "scikit-learns HistGradientBoosting (samme histogrammetode som LightGBM). De dukker opp i listen "
                     "når pakkene installeres.*")
    lines.append("\n*Historisk test; feil-prediksjoner under gjelder beste modell. Ikke finansiell rådgivning.*")
    return ("\n".join(lines), gr.update(value=_ml_table(reports, threshold), visible=True),
            gr.update(value=_ml_importance_figure(result), visible=True),
            gr.update(value=_ml_drift_figure(result), visible=True), result["errors"])


def show_trades(label, state):
    if not state or label not in state:
        return pd.DataFrame()
    return pd.DataFrame(state[label])


def add_section():
    """Pine-test som egen seksjon nederst på ML-siden (``script_profit_dashboard.add_page``). Ingen ordre."""
    today = date.today()
    with gr.Group():
        gr.Markdown(INTRO)
        example = gr.Dropdown(choices=list(EXAMPLES), value=None, label="…eller last inn et ferdig skript")
        code = gr.Textbox(lines=25, max_lines=1000, show_label=True, label="Pine-kode (opptil 1000 linjer)",
                          placeholder="Lim inn hele Pine Script-koden her, f.eks.\n\n//@version=5\n"
                                      "strategy(\"Min strategi\", overlay=true)\n...")
        script_info = gr.Markdown()
        with gr.Row(equal_height=True):
            market = gr.Dropdown(choices=MARKET_CHOICES, value="binance:BTCUSDT", allow_custom_value=True,
                                 label="Marked (eller skriv f.eks. PEPEUSDT eller AAPL)", scale=2)
            period = gr.Dropdown(choices=PERIODS, value="Siste 90 dager", label="Periode", scale=2)
            run_button = gr.Button("Test lønnsomhet", variant="primary", size="lg", scale=1)
    summary = gr.Markdown()
    results = gr.Dataframe(interactive=False, wrap=True, label="Resultat per intervall", visible=False)
    chart = gr.Plot(label="Avkastning gjennom perioden (stiplet = kjøp og hold)", visible=False)
    with gr.Accordion("Detaljer og alle handler", open=False):
        details = gr.Dataframe(interactive=False, wrap=True, label="Flere tall per intervall")
        trades_for = gr.Dropdown(choices=[], label="Vis handler for intervall")
        trades = gr.Dataframe(interactive=False, wrap=True,
                              label="Handler, nyeste først (tid = åpningen etter signallyset, da ordren fylles)")
    state = gr.State()
    with gr.Accordion("Avanserte valg", open=False):
        timeframes = gr.CheckboxGroup(list(B.TIMEFRAMES), value=DEFAULT_TFS, label="Tidsintervaller")
        with gr.Row():
            start = gr.Textbox(value=(today - timedelta(days=90)).isoformat(),
                               label="Egne datoer: fra (ÅÅÅÅ-MM-DD, UTC)")
            end = gr.Textbox(value=(today + timedelta(days=1)).isoformat(), label="til (ÅÅÅÅ-MM-DD, UTC)")
        with gr.Row():
            direction = gr.Radio(["Long og short", "Kun long"], value="Long og short", label="Retning")
            cost = gr.Number(value=0.10, minimum=0, maximum=2, step=0.01, label="Kostnad per side (%)")
            funding = gr.Checkbox(value=True, label="Trekk faktisk funding (krypto)")
            use_orders = gr.Checkbox(value=True, label="Strategier: bruk skriptets egne ordre")
        with gr.Row():
            long_signal = gr.Dropdown(choices=_signal_choices(), value=B.AUTO, allow_custom_value=True,
                                      label="Kjøpssignal (indikatorer)")
            short_signal = gr.Dropdown(choices=_signal_choices(), value=B.AUTO, allow_custom_value=True,
                                       label="Salgssignal (indikatorer)")
        inputs = gr.Dataframe(headers=["Nr", "Navn", "Type", "Verdi"], interactive=True, wrap=True,
                              label="Skriptets innstillinger (endre «Verdi» for å teste andre verdier)")
    with gr.Accordion("🤖 Maskinlæring: kan lønnsomheten til skriptets handler forutsies?", open=True):
        gr.Markdown(ML_INTRO)
        gr.Markdown(ML_MODELS)
        with gr.Row():
            ml_interval = gr.Dropdown(list(B.TIMEFRAMES), value="1 time", label="Intervall", scale=1)
            ml_period = gr.Dropdown(choices=PERIODS, value="Siste år", label="Periode for ML", scale=1)
            ml_models = gr.CheckboxGroup(ML.available_models(), value=[ML.LR, ML.RF, ML.HGB], label="Modeller",
                                         scale=3)
        with gr.Row():
            ml_mode = gr.Radio(list(ML_MODES), value=list(ML_MODES)[0], label="Treningsvindu", scale=2)
            ml_window = gr.Number(value=200, minimum=20, maximum=5000, step=10, precision=0,
                                  label="Rullerende vindu (handler)", scale=1)
            ml_refit = gr.Number(value=5, minimum=1, maximum=200, step=1, precision=0,
                                 label="Tren på nytt hver N. handel", scale=1)
            ml_min_train = gr.Number(value=30, minimum=10, maximum=1000, step=5, precision=0,
                                     label="Minste antall treningshandler", scale=1)
        with gr.Row():
            ml_threshold = gr.Slider(0.5, 0.8, value=0.5, step=0.05, label="Terskel for «ta handelen» (P ≥)",
                                     scale=2)
            ml_select = gr.Checkbox(value=False, label="Forbedring: feature-utvalg i hvert treningssteg "
                                                       "(bare treningsdata)", scale=2)
            ml_button = gr.Button("Tren og test modellene", variant="primary", scale=1)
        ml_summary = gr.Markdown()
        ml_table = gr.Dataframe(interactive=False, wrap=True, visible=False,
                                label="Mål på handler modellen ikke hadde sett (grunnrate i parentes)")
        ml_importance = gr.Plot(label="Feature importance (permutasjon på ukjente handler)", visible=False)
        ml_drift = gr.Plot(label="Treffsikkerhet over tid (rullerende)", visible=False)
        ml_errors = gr.Dataframe(interactive=False, wrap=True,
                                 label="Feil-prediksjoner der modellen var mest sikker (beste modell)")
        with gr.Accordion("Hva måles, og hvordan kan det forbedre modellene?", open=False):
            gr.Markdown(ML_HOW)
    with gr.Accordion("Hvordan testes skriptet, og hva støttes?", open=False):
        gr.Markdown(HOW)
    code.change(read_script, inputs=[code], outputs=[script_info, inputs, long_signal, short_signal, code],
                trigger_mode="always_last", show_progress="hidden")
    example.change(load_example, inputs=[example], outputs=[code])
    run_button.click(run_test, inputs=[code, market, period, start, end, timeframes, direction, cost, funding,
                                       use_orders, long_signal, short_signal, inputs],
                     outputs=[summary, results, chart, details, trades_for, trades, state, long_signal,
                              short_signal],
                     api_name="pine_test", concurrency_limit=1, concurrency_id="pine_test")
    trades_for.change(show_trades, inputs=[trades_for, state], outputs=[trades])
    ml_button.click(run_ml, inputs=[code, market, ml_period, start, end, direction, cost, funding, use_orders,
                                    long_signal, short_signal, inputs, ml_interval, ml_models, ml_mode, ml_window,
                                    ml_refit, ml_min_train, ml_threshold, ml_select],
                    outputs=[ml_summary, ml_table, ml_importance, ml_drift, ml_errors],
                    api_name="pine_ml", concurrency_limit=1, concurrency_id="pine_test")
