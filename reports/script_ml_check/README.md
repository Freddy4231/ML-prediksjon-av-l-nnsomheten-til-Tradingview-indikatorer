# ML-prediksjon av skriptenes lønnsomhet: én tunet modell per skript og tidsintervall (19. oktober 2026)

ML endrer ikke skriptene og velger ikke handler. Den gir bare en prediksjon av hvor lønnsomt hvert skript blir
neste dag, uke og måned (sjanse for gevinst og forventet avkastning, 1×, etter kostnad).

- Resultatene, tuning-panelet og live-prediksjonen vises nederst på hovedsiden (`models_overview.py`).
- Trening og tuning: `python -m research.script_ml_check` (ca. 15 min).
- Felles beregning for trening og nettside: `script_profit_ml.py`.
- Lagrede modeller: `models.joblib`, med SHA i `results.json`.

## Datasett

- **Kilde:** Binance BTCUSDT spot 2017-08-17 til 2026-10-10.
  - `data/btcusdt_latest_1h.csv` (1 time);
  - `data/30m/btcusdt_30m.csv` (30 minutter);
  - 4 timer og 1 dag er slått sammen fra timesdataene.
- **Funding:** Binance USDT-M fra 2019-09.
- **Skriptene:** Clean Wave Matrix PRO, SuperTrend [KivancOzbilgic], Chandelier Exit (22, 3), UT Bot Alerts og
  Triple Supertrend [MarkitTick]. Hvert skript kjøres på hvert intervall som motoren:
  - long etter kjøp, short etter salg, og posisjonen snus ved motsatt signal;
  - ordre ved neste open, 1×, 0,10 % per side og funding.
- **Eksempler:** ett per døgn (døgnets siste close), ca. 3 300 per skript, intervall og horisont.
- **Fasit:** skriptets faktiske avkastning neste dag, uke eller måned.

## Rensing, feature engineering og skalering (tuning)

| Steg | Valg |
|---|---|
| Rensing av data | Av, eller på: fjern treningseksempler der featurevinduet (192 barer) eller fasiten berører utfylte barer (børsstans), og klipp hver feature til 1.–99. persentil i treningen |
| Feature engineering | Grunnleggende 27 (23 markedsfeatures + posisjon, skriptavkastning 7 og 30 dager, timer siden signal) eller utvidet 32 (+ signaler siste 7 dager, skriptets fall fra 30-dagers topp, kort/lang volatilitet, trendeffektivitet, \|avstand fra SMA168\|) |
| Skalering | Ingen, StandardScaler eller RobustScaler (bare den lineære modellen) |
| Modeller og hyperparametre | Logistisk regresjon C ∈ {0,01; 0,1; 1}; Random Forest (dybde, blad) ∈ {(4, 20), (6, 50)}; HistGradientBoosting (dybde, læringsrate) ∈ {(2; 0,05), (3; 0,05)} |

- **Omfang:** 52 kombinasjoner per skript, intervall og horisont (60 grupper). Vinneren er den med lavest snitt log
  loss i gående-fremover-testene 2019–2024.
- **Purging:** modellen trenes bare på eksempler med kjent fasit før teståret.
- **Utgangspunkt** (før tuning): grunnleggende features, ingen rensing, StandardScaler og standard hyperparametre.
- **Kontroll:** 2025–2026 er en kjent kontroll som ikke er brukt i tuningen.
- **Forventet avkastning:** samme rensing, features og skalering, med modellfamilie valgt på MAE.

## Resultat

**Effekt av tuningstegene, snitt over 60 grupper:**

| Steg | Endring AUC 2019–24 | Endring AUC 2025–26 | Bedre log loss (andel grupper) |
|---|---|---|---|
| Rensing på mot av | −0,007 | +0,000 | 32 % |
| Utvidet mot grunnleggende features | −0,002 | +0,001 | 28 % |
| StandardScaler mot ingen (lineær) | −0,003 | −0,001 | 33 % |
| RobustScaler mot StandardScaler (lineær) | +0,002 | +0,001 | 87 % |
| Automatisk tuning mot utgangspunkt | +0,006 | −0,003 | 100 % (valgt der) |

Tuningen forbedrer utviklingsårene litt, siden den er valgt der, men ikke kontrollperioden. Gevinsten er støy.
Rensing og feature engineering hjalp ikke i snitt. RobustScaler var marginalt bedre enn StandardScaler. Der det ikke
finnes signal, velger tuningen ofte en sterkt regularisert logistisk modell, som i praksis gir baseline.

**Tunet AUC 2019–24 / 2025–26, neste måned (✓ = minst 0,55 i begge):**

| Skript | 30 min | 1 time | 4 timer | 1 dag |
|---|---|---|---|---|
| Clean Wave | 0,50 / 0,33 | 0,46 / 0,53 | 0,55 / 0,58 ✓ | 0,56 / 0,56 ✓ |
| SuperTrend Kivanc | 0,54 / 0,46 | 0,61 / 0,66 ✓ | 0,66 / 0,72 ✓ | 0,57 / 0,61 ✓ |
| Chandelier | 0,51 / 0,48 | 0,59 / 0,70 ✓ | 0,59 / 0,73 ✓ | 0,54 / 0,46 |
| UT Bot | 0,54 / – | 0,58 / 0,48 | 0,59 / 0,66 ✓ | 0,60 / 0,66 ✓ |
| Triple Supertrend | 0,53 / 0,56 | 0,49 / 0,49 | 0,55 / 0,68 | 0,54 / 0,64 |

- **Best:** trendskriptene Kivanc og Chandelier på 1 og 4 timer, neste uke og måned (AUC opp mot 0,66–0,73).
- **Dårligst:** 30 minutter. Der faller AUC under 0,5 i 2025–26.
- **Treffsikkerhet:** riktig gevinst/tap-gjetning ved 50 % slår ikke alltid baselinen, fordi sannsynlighetene er
  dårlig kalibrert.

## Begrensninger

- 2019–2024 brukes både til tuning og til å vise utviklingsresultatet. Bare 2025–2026 er utenfor valget, og den
  perioden er sett før i prosjektet.
- Live kjører skriptene på 5 minutter med gearing. Den historikken er ikke lastet ned.
- Ikke finansiell rådgivning.
