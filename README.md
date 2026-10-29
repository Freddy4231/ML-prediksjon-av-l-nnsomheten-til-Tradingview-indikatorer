# ML-prediksjon av lønnsomheten til TradingView-indikatorer

ML assignment 2 for Gabriel og Fredrik. Innleveringsrapporten ligger i `Mal DAT158 innlevering 2.docx`.

> **▶ Åpne appen i nettleseren:** kjør `python App.py --share` (se «Kjør lokalt»). Gradio skriver da en offentlig
> lenke, `https://….gradio.live`, som alle kan åpne så lenge appen kjører på maskinen. Lenken er ny ved hver
> oppstart og slutter å virke etter en uke eller når appen stoppes. På egen maskin: http://127.0.0.1:7862.

En Gradio-app i to deler:

1. **Automatisk handel · WEEX (papirhandel)** – tre skript-felt med tekniske indikatorer (Clean Wave Matrix PRO,
   SuperTrend, Chandelier Exit, TMG Trades, Pro Swing, Kinetic Scalper, Market Oracle og Bigscalp) på offentlige
   WEEX-lys for 15 kryptovalutaer. Hvert felt har sanntidsgraf, valg av mynter, intervall, retning, margin og
   gearing, og knappene Start, Stopp og Lukk. Under feltene vises posisjoner, utførte handler og en beslutningslogg.
   TMG-nedtrekkene viser forskningsresultatene bak TMG-indikatorene.
2. **ML-vurdering av skript** (`/ml-skript`) – velg en teknisk indikator og en ML-modell (logistisk regresjon/Ridge,
   Random Forest eller gradient boosting). Modellen forutsier sannsynligheten for at indikatoren blir lønnsom neste
   dag, uke eller måned, og meravkastningen mot vanlig kjøp og hold. Modellen trenes fortløpende fremover i tid
   (walk-forward), og siden viser treffsikkerhet, latency, throughput og feature importance.

> **Ikke finansiell rådgivning.** Alle resultater er historiske simuleringer og usikre modellanslag.

## Denne kopien kan ikke handle

Repoet er en offentlig visningskopi. Den private versjonen med ekte ordremotor er ikke med.

- Det finnes **ingen ordremotor, ingen signering og ingen kontokall** i koden, og ingen kode leser WEEX-nøkler.
- `weex_autotrade_dashboard.py` henter bare offentlige markedsdata (lys og prissteg) med GET mot en fast liste
  med markedsendepunkter (`PUBLIC_PATHS`).
- Start, Stopp, Lukk og «Ta profit / lukk alle» driver **papirhandel** (`papirhandel.py`): simulerte posisjoner og
  fyll i minnet på din egen maskin, på offentlige priser. Ingen ordre sendes noe sted. Tilstanden forsvinner når
  appen stoppes.
- `tests/test_kopi_uten_handel.py` kontrollerer at ingen fil inneholder ordre- eller kontoendepunkter, nøkkelnavn
  eller signering, at bare tillatte miljøvariabler leses, og at klienten bare sender GET uten nøkler.

Papirhandelen er en forenklet etterligning av ordremotoren. Den beslutter ved lysets lukking og fyller til
sluttkursen med 0,08 % gebyr per side. Nødstoppen utløses når 90 % av marginen er tapt. Clean Wave, SuperTrend,
Chandelier og TMG blir med i skriptets posisjon ved start, mens de andre skriptene venter på et nytt signal.
Skriptenes egne SL/TP følges via målposisjonen; børsordre, ADD og delvise TP simuleres ikke.

## Kjør lokalt

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python App.py
```

Åpne http://127.0.0.1:7862 (ML-siden: http://127.0.0.1:7862/ml-skript). Port 7862 er valgt fordi 7860 ofte er i
bruk. En annen port settes med `GRADIO_SERVER_PORT=7870 python App.py`.

Med `python App.py --share` lager Gradio i tillegg en midlertidig offentlig lenke som kan deles. Papirhandelen er da
felles for alle som bruker lenken, og den sender fortsatt ingen ordre.

Grafene og papirhandelen trenger internett for å hente offentlige WEEX-lys. ML-siden virker uten nett med
«Lagret eksempel · 11. oktober 2026»; «Ferske BTC-data» henter offentlige Binance-lys.

## Tester

```bash
python -m unittest discover -s tests
```

## ML-siden: data, modeller og mål

- **Data:** Binance BTCUSDT-lys 2017–2026 på 30 min, 1 time, 4 timer og 1 dag, og Binance-funding.
  Hver rad i CSV-tabellene i `reports/script_profit_website_2026-10-27/` er ett døgn for én indikator på ett
  tidsintervall. Raden har 32 relative egenskaper (avkastning, volatilitet, RSI, glidende snitt, volum, tid og
  indikatorens tilstand). Fasit er om indikatoren ga gevinst etter gebyr og funding, og meravkastningen mot kjøp
  og hold.
- **Modeller:** logistisk regresjon/Ridge (med standardisering), Random Forest og HistGradientBoosting.
  XGBoost og LightGBM vises hvis pakkene er installert. De lagrede, tunede modellene i `reports/script_ml_check/`
  vises som statisk referanse.
- **Trening:** walk-forward fra 2019 med ny trening hver måned, kvartal, halvår eller år. Bare eksempler med kjent
  fasit før treningstidspunktet brukes. «Automatisk» holder hele beregningen på omtrent 10–20 sekunder.
- **Mål:** treffsikkerhet mot grunnlinjen, AUC, falske positive og negative, latency per prediksjon, throughput og
  feature importance. 2019–2024 brukes til vurdering; 2025–2026 er kjent kontrollhistorikk, ikke en urørt test.
- Datasett, modeller og kode er bundet sammen med SHA-256 i `manifest.json`; endres en fil, sperres vurderingen.

## Filer

| Fil | Rolle |
|---|---|
| `App.py` | Hovedsiden (fanen «Automatisk handel · WEEX») og ML-undersiden |
| `weex_autotrade_dashboard.py` | Grafene i skript-feltene og offentlige WEEX-markedsdata |
| `papirhandel.py` | Papirhandel for Start/Stopp/Lukk, posisjoner, fyll og beslutningslogg |
| `script_profit_dashboard.py`, `script_model_compare.py`, `script_profit_predictor.py`, `script_profit_ml.py` | ML-undersiden |
| `clean_wave_weex.py`, `kivanc_supertrend.py`, `triple_supertrend.py`, `ut_bot_signals.py`, `tmg_trades_weex.py`, `pro_swing_weex.py`, `kinetic_scalper_weex.py`, `market_oracle_weex.py`, `bigscalp_weex.py` | Indikatorene |
| `pine_engine/` | Pine Script-tolk som kjører Kinetic- og Oracle-rekonstruksjonene |
| `tmg_*.py`, `research/tmg_*.py`, `reports/tmg_*`, `data/` | TMG-forskningspanelene med SHA-kontrollerte data |
| `tradingview/` | Pine-kildene til indikatorene |

## Lisenser og kreditering

- *SMC Institutional Clean Wave & Structure PRO* © Dark_Ace_Master, Mozilla Public License 2.0.
- *SuperTrend STRATEGY* © KivancOzbilgic, Mozilla Public License 2.0.
- *Triple Supertrend Confluence* © MarkitTick, Creative Commons BY-NC-SA 4.0. Python-porten
  (`triple_supertrend.py`) deler samme lisens og brukes ikke-kommersielt.
- *Chandelier Exit* følger everget sin regel (22, 3) i egen implementasjon.
- Pro Swing, Kinetic Scalper, Market Oracle Pro og Bigscalp Premium er **uavhengige rekonstruksjoner** fra offentlige
  beskrivelser og grafer, ikke originalkode. TMG-indikatorene er egne tolkninger inspirert av YouTube-kanalen
  «TMG Trades» (@tmgtrades).
- Markedsdata fra Binance og WEEX sine offentlige API-er.
