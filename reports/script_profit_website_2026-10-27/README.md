# ML-vurdering av skript · nettside for prosjektoppgaven

Egen side `/ml-skript`: brukeren velger teknisk indikator, intervall, periode, **ML-modell** og kapital og trykker
«Beregn med ML». Valgt modell trenes fortløpende fremover i tid og forutsier sannsynligheten for at indikatoren blir
lønnsom og meravkastningen mot vanlig kjøp og hold, med regneeksempel i kroner. Kapitalen skalerer bare regneeksemplet.

Fem indikatorer: UT Bot, SuperTrend Kivanc, Chandelier, Clean Wave og Triple Supertrend.
Fire intervaller: 30 min, 1 time, 4 timer og 1 dag. Tre horisonter: dag, uke og måned.
De 60 lagrede modellparene fra `reports/script_ml_check` vises bare som statisk referanselinje og endres ikke.
Vilkårlig innlimt Pine-kode må få eget datasett og modell før den kan støttes.

## Slik kan løsningen demonstreres

1. Åpne «ML-vurdering av skript» i nettsidens navigasjon.
2. Velg UT Bot, 1 time, neste uke, en ML-modell, 10 000 kr og «Lagret eksempel».
3. Trykk «Beregn med ML». Modellen får faktiske historiske egenskaper fra 11. oktober 2026.
4. Bytt indikator, intervall eller modell og beregn igjen. Prognose, treffsikkerhet og hastighet endres.
5. «Ferske BTC-data» henter bare offentlige lys ved siste komplette UTC-døgn og bruker de samme egenskapsfunksjonene.
   Nettverksfeil, hull, uavsluttede lys eller feil modellpakke gir ingen prognose; gammelt eksempel erstatter aldri live-data stille.

## Datasettet

Siden viser ikke lenger datasettet, egenskapene, feil-prediksjoner eller forklaringslistene (fjernet etter brukerens
ønske 28. oktober 2026). Datagrunnlaget ligger fortsatt i denne mappen og brukes av modellen:

- Hele det regulariserte OHLCV-grunnlaget på alle fire intervaller, med `filled` for syntetiske lys.
- Daglige ML-eksempler per indikator og intervall: 32 historiske egenskaper, senere nettoavkastning,
  gevinst/tap-label, tidspunkt fasiten ble kjent og markering av radene brukt ved siste lagrede trening.
- Faktisk Binance USDT-M-funding som inngår i fasiten, i egen CSV.
- Tom fasit betyr ukjent. `script_profit_predictor.dataset_view` kan fortsatt lese datoavgrensede utdrag.

Datasettene eksporteres med samme funksjoner og antall eksempler som originaltreningen.
`manifest.json` binder runtime, modell/resultater, demonstrasjonsinput og CSV-er med SHA-256.
Gamle modellartefakter og treningskilder endres ikke. Pakking: `python -m research.script_profit_website`.

## Hva er ML-resultatet?

Klassifisering: P(simulert skriptavkastning > 0) etter kostnader og funding.
Regresjon: anslag på denne avkastningen. Modellfamilier er logistisk regresjon/Ridge, Random Forest og gradient boosting.
Fasit kommer fra å følge indikatorens long/short-posisjon fra neste open over valgt periode.
Barvis avkastning ved omtrent 1×, kostnad 0,10 % per omsatt side og historisk funding.
Dette er en spotpris-proxy for en strategi, ikke faktisk WEEX-/TradingView-bokføring, SL/TP, gearing eller bekreftede fyll.
Fremtidige fundingrater brukes i treningsfasiten, ikke som kjente egenskaper ved prediksjon.

Trening og modellvalg bruker årlige tidsfolder i 2019–2024, med ukjent fasit fjernet fra trening før teståret.
2025–2026 er kjent kontrollhistorikk brukt tidligere i prosjektet. De viste utviklingstallene er også brukt til modellvalg;
de er ikke urørt test. Fasitvinduer overlapper. Nye data her gir ingen ny måling av ML-presisjon.
Det lagrede eksemplet viser funksjonen og er ikke bevis for senere profitt. Modellene er ikke dokumentert sannsynlighetskalibrert.
Svak AUC, dårlig log loss mot basisrate og regresjonsfeil mot historisk snitt vises tydelig, også når anslaget er positivt.
Gjennomsnittlig absolutt feil er ingen konfidensgrense og brukes ikke som kunstig usikkerhetsintervall.

## Kilde til ferske lys

[Binance Spot API · klines](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#klinecandlestick-data):
åpningstid, OHLC, volum og lukketid. Maksimalt to sider med 1000 lys, fast BTCUSDT og støttede intervaller.
Forsinket døgnavslutning, duplikatkonflikter, hull og ugyldig OHLC avvises.
Kortere oppvarmingshistorikk live kan påvirke en tilstandsfull indikators signaler sammenlignet med full treningshistorikk.
Siden kjører bare prediksjon; den har ingen WEEX-kobling eller ordrehandling.

Tre faktiske modellkall med ulike brukerinput er dokumentert i [demonstration.json](demonstration.json).

## Indikator + ML-modell med fortløpende trening (28. oktober 2026)

Siden er nå én flyt (`script_model_compare.py`): brukeren velger teknisk indikator, intervall, periode og **ML-modell**
(logistisk regresjon/Ridge, Random Forest eller gradient boosting HistGB; XGBoost/LightGBM når pakkene er installert),
og «Beregn med ML» lar valgt modell forutsi indikatorens lønnsomhet. Modellen trenes på de samme daglige eksemplene.
Under «Avanserte ML-valg» velges trening, vindu og egenskaper, og om de andre modellene skal tas med til sammenligning.
Den lagrede, tunede modellen fra prosjektets trening vises som statisk referanse i teksten; de 60 modellparene endres ikke.

- **Ingen statisk trening:** ny trening hver måned, kvartal, halvår eller år fra 2019 (walk-forward), voksende eller
  rullerende toårsvindu. Bare eksempler med `known_*` ≤ treningstidspunktet brukes. Minst 200 eksempler per trening.
  «Automatisk» velger den hyppigste treningen som holder hele beregningen innenfor omtrent 10–20 sekunder.
- **Treffsikkerhet:** andelen perioder der P(lønnsom) ≥ 50 % stemte, på valgt tidsintervall, mot grunnlinjen
  (andelen lønnsomme perioder i treningsvinduet). Standardfeil justeres for overlappende fasitperioder.
- **Kvadratisk avvik (beregnes, vises ikke lenger i tabellen etter brukerens ønske):** Brier for sannsynligheten, og
  MSE (prosentpoeng²) for anslått meravkastning mot vanlig kjøp og
  hold: skriptets avkastning minus BTC-kursendringen mellom de samme åpningene (`open` i prisfilen ved beslutning og
  ved `known_*`). Grunnlinjen er treningsvinduets snitt. MSE brukes til feature importance og til å finne modellen
  med lavest feil. Snittprofitt mot kjøp og hold vises per periode.
- **Latency og throughput:** svartid per prediksjon (median av 15 kall) og prediksjoner per sekund (2000 rader), samt
  treningstid per gang og totalt. Beregningstiden for hele sammenligningen vises mot målet.
- **Feature importance:** permutasjonsviktighet (økning i Brier) på ukjente perioder i 2019–2024 for valgt modell.
  Kalibrering og rullerende treffsikkerhet vises.

2025–2026 vises separat og er kjent kontrollhistorikk. Velg aldri modell, intervall eller innstillinger etter den.
Resultatene mellomlagres i minnet per valg. Datasettene leses bare når SHA-256 stemmer med manifestet.
Kodehashene oppdateres eksplisitt med `python -m research.script_profit_website --refresh-runtime`, som stopper hvis
datasett, modeller eller demonstrasjonsinput er endret.
