"""Clean Wave Matrix PRO oversatt til Python, uten fremtidslekkasje.

Originalen er «SMC Institutional Clean Wave & Structure PRO» (kort tittel
«Clean Wave Matrix PRO») av Dark_Ace_Master, lisensiert under Mozilla Public
License 2.0. Standardinnstillingene (21, 2, 90, 7, small, 10, 20, Dashed, 2)
gir bølgeperiode 21, strukturfølsomhet 7, signalfølsomhet 10 og SD-periode 20.
Linjetykkelse, gjennomsiktighet, etikettstørrelse og linjestil er bare grafikk.

Bare deler som kan påvirke en handel er tatt med:

- Trendbølge: ``ta.alma(close, 21, 0.85, 6)``; opptrend når close >= bølgen.
- Innsidebarer: high <= high[1] og low >= low[1] (nå eller forrige bar).
- Struktur (BOS/CHoCH): pivoter med 7 barer på hver side. Brudd når close
  krysser siste bekreftede pivot. CHoCH hvis trenden var motsatt, ellers BOS.
- Svingmerker KJØP/SELG: pivoter med 10 barer på hver side.
- −2,5 SD-nivå: SMA(20) − 2,5 · populasjons-standardavvik(20).

Viktig om KJØP/SELG-merkene: originalen tegner merket ``sig_sens`` barer
TILBAKE i tid når pivoten blir bekreftet. På en historisk graf ser merkene da
ut til å treffe bunner og topper perfekt, men i sanntid vet man det først 10
barer senere. Her er de bekreftede hendelser på bekreftelsesbaren.
``hindsight_pivots`` gir den tilbakedaterte plasseringen, bare for å
dokumentere illusjonen. Den skal aldri brukes som feature eller handelssignal.

Pine-semantikk som er kopiert eksakt:

- ``ta.pivothigh``/``ta.pivotlow``: ved like verdier vinner den nyeste baren
  (som i PyneCore). En topp må være >= barene til venstre og > barene til høyre.
  En bunn må være <= til venstre og < til høyre. Verdien rapporteres på
  bekreftelsesbaren, ``right`` barer etter selve pivoten.
- ``ta.atr`` er ``ta.rma`` av true range, startet med SMA av de første verdiene.
- ``ta.stdev`` har ``biased=true`` som standard (deler på N).
- ``ta.crossover(close, last_ph)`` bruker verdien av ``last_ph`` da funksjonen
  ble kalt på forrige bar. Skriptet nullstiller ``last_ph`` etter et brudd. Med
  Pine v6 sin late evaluering av ``and`` gir det samme resultat som her, fordi en
  nylig bekreftet topp alltid ligger over close på bekreftelsesbaren.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


CW_PARAMS = {
    "wave_len": 21, "alma_offset": 0.85, "alma_sigma": 6.0,
    "smc_sens": 7, "sig_sens": 10, "sd_len": 20, "sd_mult": 2.5, "atr_len": 14,
}
# Alder (barer siden en hendelse) kappes, så gamle hendelser ikke gir ekstreme verdier.
AGE_CAP = 500
# Antall barer før alle Clean Wave-tilstander er stabile (ATR/RMA, pivoter, struktur).
CW_WARMUP_BARS = 1000

CW_FEATURE_COLUMNS = [
    # Avstand fra close til trendbølgen, i ATR
    "cw_dist_wave_atr",
    # Bølgens stigning siste 3 barer, i ATR
    "cw_wave_slope_atr",
    # 1 hvis close >= bølgen (grønn bølge), ellers 0
    "cw_wave_up",
    # log(1 + barer siden bølgen sist skiftet farge)
    "cw_wave_age",
    # Strukturtrend etter siste BOS/CHoCH: 1 opp, −1 ned, 0 ingen ennå
    "cw_trend_dir",
    # log(1 + barer siden siste BOS/CHoCH)
    "cw_struct_age",
    # 1 hvis siste strukturbrudd var CHoCH (trendskifte), 0 hvis BOS
    "cw_last_choch",
    # Avstand fra close til siste bekreftede strukturtopp/-bunn, i ATR
    "cw_dist_ph_atr", "cw_dist_pl_atr",
    # log(1 + barer siden pivot-baren til siste bekreftede svingbunn/-topp)
    "cw_swing_low_age", "cw_swing_high_age",
    # Avstand fra close til siste bekreftede svingbunn/-topp, i ATR
    "cw_dist_swing_low_atr", "cw_dist_swing_high_atr",
    # (close − SMA20) / standardavvik20; −2,5 tilsvarer skriptets SD-mål
    "cw_sd_z",
    # 1 hvis baren eller forrige bar er en innsidebar (konsolidering)
    "cw_inside",
]

# Inngangskandidater som skriptet selv viser, alle kjent ved bekreftelsesbarens close.
ENTRY_EVENTS = ("struct_bull", "wave_up_flip", "swing_low_confirmed")
EXIT_EVENTS = ("struct_bear", "wave_down_flip", "swing_high_confirmed")


def alma(values, length, offset, sigma):
    """Pine ``ta.alma`` med floor=false. Vekt nr. 0 gjelder eldste verdi i vinduet."""
    values = np.asarray(values, dtype=float)
    m = offset * (length - 1)
    s = length / sigma
    weights = np.exp(-((np.arange(length) - m) ** 2) / (2 * s * s))
    out = np.full(len(values), np.nan)
    if len(values) >= length:
        out[length - 1:] = sliding_window_view(values, length) @ weights / weights.sum()
    return out


def pine_rma(values, length):
    """Pine ``ta.rma``: første verdi er SMA av ``length`` verdier, så Wilder-utjevning."""
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    alpha, prev = 1.0 / length, np.nan
    for t in range(len(values)):
        if np.isnan(prev):
            if t >= length - 1 and np.isfinite(values[t - length + 1:t + 1]).all():
                prev = values[t - length + 1:t + 1].mean()
        else:
            prev = alpha * values[t] + (1 - alpha) * prev
        out[t] = prev
    return out


def pine_atr(high, low, close, length):
    """Pine ``ta.atr``: RMA av true range; første bar bruker high − low."""
    prev_close = np.r_[np.nan, close[:-1]]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    tr[0] = high[0] - low[0]
    return pine_rma(tr, length)


def pivots(values, left, right, kind):
    """Pine ``ta.pivothigh``/``ta.pivotlow``, justert til bekreftelsesbaren."""
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) < left + right + 1:
        return out
    win = sliding_window_view(values, left + right + 1)
    c = win[:, left:left + 1]
    if kind == "high":
        ok = (c >= win[:, :left]).all(axis=1) & (c > win[:, left + 1:]).all(axis=1)
    else:
        ok = (c <= win[:, :left]).all(axis=1) & (c < win[:, left + 1:]).all(axis=1)
    ok &= np.isfinite(win).all(axis=1)
    out[left + right:][ok] = c[ok, 0]
    return out


def _age(t, last):
    return np.log1p(min(t - last, AGE_CAP)) if last >= 0 else np.nan


def clean_wave_state(data, params=None):
    """Alle Clean Wave-tilstander per bar, beregnet bare fra kjent historikk.

    Hver rad gjelder baren som lukkes ved ``open_time + 1 t``. Hendelsene er
    kjent ved denne barens close og kan tidligst handles ved neste åpning.
    """
    p = {**CW_PARAMS, **(params or {})}
    open_, high, low, close = (data[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
    n = len(data)
    wave = alma(close, p["wave_len"], p["alma_offset"], p["alma_sigma"])
    atr_values = pine_atr(high, low, close, p["atr_len"])
    is_up = close >= wave  # NaN-bølge gir False, som i Pine (na-sammenligning er false)
    wave_ready = np.isfinite(wave)
    ph_smc = pivots(high, p["smc_sens"], p["smc_sens"], "high")
    pl_smc = pivots(low, p["smc_sens"], p["smc_sens"], "low")
    ph_sig = pivots(high, p["sig_sens"], p["sig_sens"], "high")
    pl_sig = pivots(low, p["sig_sens"], p["sig_sens"], "low")

    trend_dir = np.zeros(n)
    events = {name: np.zeros(n, dtype=bool) for name in
              ("struct_bull", "struct_bear", "struct_bull_choch", "struct_bear_choch")}
    last_ph_any = np.full(n, np.nan)
    last_pl_any = np.full(n, np.nan)
    struct_age = np.full(n, np.nan)
    last_choch = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    swing_high = np.full(n, np.nan)
    swing_low_age = np.full(n, np.nan)
    swing_high_age = np.full(n, np.nan)
    wave_age = np.full(n, np.nan)

    last_ph = last_pl = np.nan          # skriptets variabler, nullstilles ved brudd
    call_ph_prev = call_pl_prev = np.nan  # verdien ta.crossover/crossunder så forrige bar
    direction, last_event, last_event_choch = 0, -1, np.nan
    ph_value = pl_value = np.nan
    sl_value = sh_value = np.nan
    sl_bar = sh_bar = -1
    flip_bar = -1
    for t in range(n):
        if not np.isnan(ph_smc[t]):
            last_ph = ph_smc[t]
            ph_value = ph_smc[t]
        if not np.isnan(pl_smc[t]):
            last_pl = pl_smc[t]
            pl_value = pl_smc[t]
        call_ph, call_pl = last_ph, last_pl
        if (not np.isnan(call_ph) and t > 0 and not np.isnan(call_ph_prev)
                and close[t] > call_ph and close[t - 1] <= call_ph_prev):
            events["struct_bull"][t] = True
            events["struct_bull_choch"][t] = direction == -1
            last_event, last_event_choch = t, float(direction == -1)
            direction = 1
            last_ph = np.nan
        if (not np.isnan(call_pl) and t > 0 and not np.isnan(call_pl_prev)
                and close[t] < call_pl and close[t - 1] >= call_pl_prev):
            events["struct_bear"][t] = True
            events["struct_bear_choch"][t] = direction == 1
            last_event, last_event_choch = t, float(direction == 1)
            direction = -1
            last_pl = np.nan
        call_ph_prev, call_pl_prev = call_ph, call_pl
        trend_dir[t] = direction
        last_ph_any[t], last_pl_any[t] = ph_value, pl_value
        struct_age[t] = _age(t, last_event)
        last_choch[t] = last_event_choch

        if not np.isnan(pl_sig[t]):
            sl_value, sl_bar = pl_sig[t], t - p["sig_sens"]
        if not np.isnan(ph_sig[t]):
            sh_value, sh_bar = ph_sig[t], t - p["sig_sens"]
        swing_low[t], swing_high[t] = sl_value, sh_value
        swing_low_age[t], swing_high_age[t] = _age(t, sl_bar), _age(t, sh_bar)

        if t > 0 and wave_ready[t] and wave_ready[t - 1] and is_up[t] != is_up[t - 1]:
            flip_bar = t
        wave_age[t] = _age(t, flip_bar)

    prev_up = np.r_[False, is_up[:-1]]
    prev_ready = np.r_[False, wave_ready[:-1]]
    inside = np.r_[False, (high[1:] <= high[:-1]) & (low[1:] >= low[:-1])]
    was_inside = np.r_[False, inside[:-1]]
    sma = pd.Series(close).rolling(p["sd_len"]).mean().to_numpy()
    sd = pd.Series(close).rolling(p["sd_len"]).std(ddof=0).to_numpy()

    state = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "wave": wave, "atr": atr_values, "wave_up": is_up & wave_ready,
        "wave_up_flip": wave_ready & prev_ready & is_up & ~prev_up,
        "wave_down_flip": wave_ready & prev_ready & ~is_up & prev_up,
        "trend_dir": trend_dir, **events,
        "swing_low_confirmed": ~np.isnan(pl_sig), "swing_high_confirmed": ~np.isnan(ph_sig),
        "last_ph": last_ph_any, "last_pl": last_pl_any,
        "swing_low": swing_low, "swing_high": swing_high,
        "struct_age": struct_age, "last_choch": last_choch,
        "swing_low_age": swing_low_age, "swing_high_age": swing_high_age, "wave_age": wave_age,
        "inside": inside | was_inside, "sma": sma, "sd": sd,
        "sd_target": sma - p["sd_mult"] * sd,
    }, index=data.index)
    return state


def clean_wave_features(state):
    """Skalafrie features (ATR-, SD- eller log-alder), like på tvers av børser."""
    atr_values = state["atr"].where(state["atr"] > 0)
    close = state["close"]
    f = pd.DataFrame(index=state.index)
    f["cw_dist_wave_atr"] = (close - state["wave"]) / atr_values
    f["cw_wave_slope_atr"] = (state["wave"] - state["wave"].shift(3)) / atr_values
    f["cw_wave_up"] = state["wave_up"].astype(float).where(state["wave"].notna())
    f["cw_wave_age"] = state["wave_age"]
    f["cw_trend_dir"] = state["trend_dir"]
    f["cw_struct_age"] = state["struct_age"]
    f["cw_last_choch"] = state["last_choch"]
    f["cw_dist_ph_atr"] = (close - state["last_ph"]) / atr_values
    f["cw_dist_pl_atr"] = (close - state["last_pl"]) / atr_values
    f["cw_swing_low_age"] = state["swing_low_age"]
    f["cw_swing_high_age"] = state["swing_high_age"]
    f["cw_dist_swing_low_atr"] = (close - state["swing_low"]) / atr_values
    f["cw_dist_swing_high_atr"] = (close - state["swing_high"]) / atr_values
    f["cw_sd_z"] = ((close - state["sma"]) / state["sd"]).where(state["sd"] > 0, 0.0).where(state["sd"].notna())
    f["cw_inside"] = state["inside"].astype(float)
    # Før oppvarmingen er ferdig er RMA/struktur ikke stabile nok til å brukes.
    f.iloc[:CW_WARMUP_BARS] = np.nan
    return f[CW_FEATURE_COLUMNS]


def hindsight_pivots(state, params=None):
    """KUN ILLUSTRASJON (fremtidslekkasje): svingmerkene plassert på pivot-baren,
    slik originalen tegner dem i ettertid. Kan ikke handles i sanntid."""
    shift = {**CW_PARAMS, **(params or {})}["sig_sens"]
    return pd.DataFrame({
        "buy_badge_hindsight": state["swing_low_confirmed"].shift(-shift, fill_value=False).astype(bool),
        "sell_badge_hindsight": state["swing_high_confirmed"].shift(-shift, fill_value=False).astype(bool),
    }, index=state.index)
