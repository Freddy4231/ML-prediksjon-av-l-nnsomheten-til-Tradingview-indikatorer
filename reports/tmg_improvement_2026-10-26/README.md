# TMG · 120 forbedringsforsøk, 26. oktober 2026

**Ingen ny strategi er godkjent. Original TMG V2 og Pine-filen er beholdt.**
Alle forsøk er utført uten subagenter. Ingen konto, margin, gearing eller handelsmotor er aktivert eller endret.

## Ekte historikk og sammenlignbare kostnader

Binance spot-OHLCV fra `data/30m/` og `data/tv_search_2026-10-21/*_raw_1h.csv`;
faktiske fundinghendelser fra samme sistnevnte mappe. Eksakte filhasher er låst før søket.
2019 brukes til oppvarming. Felles sluttdato er 11. oktober 2026 kl. 00:00 UTC, eksklusiv,
fordi 30-minuttersfilene slutter der. Nyere timesdata finnes, men er ikke blandet inn i sammenligningen.

Alle simulerte posisjoner bruker 1× notional og fast mengde fra inngang til utgang.
Markedsordre fylles først ved neste ekte åpning etter signalets lukking. Kostnaden er
0,15 % per omsatt side pluss funding; stress er 0,30 % per side og funding ×2. Full kostnad
betales også ved vending og terminal lukking. Ingen handel på syntetiske lys, med 800 barers
inngangssperre etter hull. En gammel posisjon åpnes ikke uten et nytt signal.

Dette er en simulering av TMG-portens målposisjon, ikke native TradingView eller faktiske WEEX-fyll:

- Python-portens POC-treff handles ved neste åpning; Pine bruker en hvilende limitordre.
- Trendens strukturelle børsstopp, WEEX-nødstopp, forsinkelse, ordrebok og gearing/likvidasjon er ikke simulert.
- Funding belastes ved første baråpning etter det faktiske hendelsestidspunktet, med denne åpningens pris som proxy.
- Volumdelta er konsekvent estimert fra close innenfor high–low, slik som i Pine/WEEX. Det er ikke målt CVD.
- Lengre EMA-vinduer er en HTF-proxy; de er ikke faktiske aggregerte HTF-lys.
- Før første tilgjengelige fundinghendelse regnes funding som null; særlig SOLs første spotmåned i 2020
  faller før fundinghistorikken. Resultater fra denne starten kan ikke tolkes som gjennomførbar perp-handel.
- Live bruker et rullerende historikkvindu og kan starte med annen indikatortilstand enn en lang avspilling.

## Forhåndslåste forsøk

1. **48 parameteroppsett:** original/hybrid/trend, rask/standard/langsom kontekst, med/uten
   volumdelta-proxy som filter, sterkere regimeskifte, 30 min/1 t/4 t og begge retninger/kun long.
   Valg på BTC/ETH/SOL 2020–23, deretter godkjenning i 2024, med identisk V2 som referanse.
   Vinner før kontroll: `slow_price_flow_30_long_only`. På BTC ga den +129,88 % i utvikling og
   +52,28 % i 2024. Den tapte **14,20 %** i 2025–26 og ble forkastet. Sammenlignbar V2 kun long
   ga +17,45 %. De øvrige kandidatene ble ikke rangert på kontrollen.
2. **48 utgangsvarianter:** gevinstsikring 4/6/8 ATR, aktivering etter gevinst på 2/3/4 ATR,
   én inngang per regime eller ny momentumovergang etter minst 24 barer, to kontekster,
   30 min/1 t, begge retninger/kun long. Gevinststoppen gjelder først etter at den er beregnet,
   og utgangen bruker close og neste åpning. Ingen bestod kravene i 2020–24; ingen kontrollrangering.
3. **24 ML-filtre:** to regulariserte logistiske modeller og et beslutningstre med maks dybde 2,
   to terskler, 30 min/1 t og begge retninger/kun long. Tolv relative pris-, momentum-, volum-
   og regimevariabler. Modellene oppdateres ved årsskiftet og bruker bare fullførte handler
   før treningsfristen. Terminalt avkuttede handler brukes aldri som treningsfasit.
   Testårene er 2022, 2023 og 2024; trening starter i 2020 på BTC/ETH/SOL samlet.
   Ingen bestod kravene. Et filter som avviser alle handler får ikke passere.

Retur, kostnadsstress, handler, år og drawdown inngår i kravene. Valgkriterier og terskler er
låst i hver `locked_plan.json`. En feil ved lesing av blandet tidsstempelpresisjon ble rettet
etter to baseline-rader, før fullført første søk; uendret forskningsplan og kildehash før/etter
er dokumentert i `implementation_repairs.json`.

**2025–26 har vært sett i tidligere prosjektarbeid og er kjent historisk kontroll.** Etter første
TMG-forsøk er perioden også kjent for denne indikatoren. Det finnes ingen urørt holdout eller
forward-test i denne rapporten. Ingen nye terskler eller parametre er valgt fra kontrolltallene.

## V2 i kjent kontroll · 1. januar 2025–11. oktober 2026

BTC, long/short, 1× notional, etter kostnader/funding:

| Intervall | Netto | Stress | Største fall | Handler |
|---|---:|---:|---:|---:|
| 30 min | +30,68 % | +10,92 % | −27,98 % | 52 |
| 1 time | −9,26 % | −21,07 % | −35,50 % | 43 |

På 30 min ga ETH +1,17 %, SOL −16,67 %, BNB +7,73 %, XRP −59,09 %, ADA −92,09 % og LTC −85,24 %.
Tallene støtter ingen generell påstand om lønnsomhet. Myntene utenom BTC/ETH/SOL ble bare brukt
til kontroll, ikke til å velge en ny variant.

## Artefakter og etterprøving

- `locked_plan.json`, `development.json`, `frozen_choice.json`, `known_control.json` og `release_decision.json`.
- `../tmg_guard_2026-10-26/` og `../tmg_meta_filter_2026-10-26/`: egne låste forsøk og beslutninger.
- `before/`: uendret kode og Pine som referanse; separate CSV-er har alle simulerte handler og kapitalforløp.
- `dashboard_data.json` er en liten, skrivebeskyttet pakke for nettsiden. `dashboard_manifest.json`
  validerer denne mot TMG-/Pine-kilden før visning. Den pakkes eksplisitt, aldri ved appoppstart.

Regnskap, kostnader, signal/fylltid, vending, funding, hull, avviste signaler, kausalitet og
treningsfrister er kontrollert med tester. Ingen native TradingView-kompilering eller ordretest med penger er utført.

Kjør fra prosjektroten med `gradio-env/bin/python -m unittest tests.test_tmg_trades tests.test_tmg_improvement
tests.test_tmg_profit tests.test_tmg_meta_filter tests.test_tmg_research_dashboard`. Forskningsfilene ligger i
`research/tmg_improvement.py`, `research/tmg_guard_search.py` og `research/tmg_meta_filter.py`.
Kontrollfilene overskrives ikke ved ny kjøring.
