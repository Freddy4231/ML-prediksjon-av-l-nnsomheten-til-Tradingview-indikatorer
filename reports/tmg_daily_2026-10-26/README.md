# TMG Dagsbekreftelse · ny, separat signalindikator

**26. oktober 2026. Relativ forbedring, fortsatt negativ BTC-kontroll. Ingen aktiv motorregel er erstattet.**

## Nye funn og regel

To nye videoer ble lest via YouTubes transkripsjonspanel, særlig dagsclose og 50/200-dagersnivå
i [golden-cross-videoen](https://www.youtube.com/watch?v=_GqbaF4lvrI&t=60).
Se [kildenotatene](SOURCE_NOTES.md) for tidsstempler og skillet mellom kanalens omtale og vår regel.
Dette er vår uavhengige modell, ikke formlene til Stonksy eller dokumentasjon av TMGs faktiske kontogevinster.

Den fryste regelen er **`daily50_weekend_both`**, 1 time, både long og short:

- **KJØP:** ferskt TMG V2-kjøpssignal og gårsdagens ferdige UTC-close over SMA50 på dagsdata.
- **SELG / SHORT:** ferskt V2-salgssignal og gårsdagens close under SMA50.
- I helgen kreves i begge retninger at spot/perp-basevolum siste 24 timer minst matcher de
  foregående, ikke-overlappende 720 timene. Dette er én børs og samme mynt, ikke global USD-ordreflyt.
- **LUKK:** originalens mål blir flatt eller vender. Et avvist motsatt signal lukker gammel posisjon.
  Et avvist signal blir aldri en forsinket inngang uten et nytt signal.
- 200-dagers SMA tegnes som kontekst; det inngår **ikke** i valgt inngangskrav.

Bare gårsdagens døgn er tilgjengelig ved timebars åpning. Timebaren 23:00–00:00 bruker fortsatt
foregående døgn; det nyss avsluttede døgnet brukes fra 00:00-baren og besluttes ved 01:00-close.
Dette tilsvarer Pine `[close[1], ta.sma(close,50)[1]]` med `lookahead_on`, ikke dagens uferdige dagslys.
Hvert døgn krever 24 ekte futures-timer. Manglende døgn fylles ikke inn i snittene.

## Valg før kontroll

**16 faste kandidater:** SMA50, SMA200, 50/200-kryss og full close/50/200-rekkefølge,
hver med/uten helgekrav og long/short eller kun long. De opprinnelige referansene er ikke nye kandidater.
Ingen ML eller ekstra tuning er lagt til for å løfte resultatet i de allerede kjente kontrollårene.

Trening/valg på **2021–2023**, godkjenning i **2024**, BTC/ETH/SOL.
Høyest median trenings-Sharpe med doble kostnader blant kvalifiserte avgjorde valget.
Valget bestod åtte placeboer med filteret forskjøvet bakover 720–3240 timer.
Alle kilde-/datahash, terskler, valg og kontroll ligger separat i denne mappen.

BTC trening: **+99,05 %**, stress **+72,32 %**. BTC 2024: **+46,30 %**, stress **+29,99 %**.
ETH var derimot negativ i både trening og 2024; dette er ikke et filter som var positivt i alle markeder.

## Kjent kontroll · januar 2025–september 2026

Samme SHA-kontrollerte Binance USD-M-data som forrige forsøk: 235 arkiver, **171 185 futures-timer**.
**1 time, begge retninger, 1× notional, 0,15 % kostnad per side + faktisk funding.**
Stress bruker 0,30 % per side og funding ×2. Fast mengde per handel, beslutning ved close og fyll neste ekte open.

| Marked | V2 netto | Ny indikator | V2 stress | Ny stress |
|---|---:|---:|---:|---:|
| BTC | −8,38 % | **−3,91 %** | −20,44 % | **−11,97 %** |
| ETH | +146,34 % | **+161,38 %** | +120,66 % | **+146,70 %** |
| SOL | +13,45 % | **+22,07 %** | −2,45 % | **+12,54 %** |

Dette er **begge retninger**. Tallene må ikke sammenlignes direkte med forrige rapports kun-long-tabell.
Appen viser også helgefilteret alene i identisk retning/tidsrom, uten å bruke denne kontrollen til nytt valg.

BTCs positivkrav feilet, selv om relativ forbedring, drawdownkravet, placebo og øvrige sluttkrav bestod.
**`released=false`.** Den nye indikatoren er tilgjengelig for visuell undersøkelse, og er ikke ny motorstandard.
ETH/SOL-kontrollen brukes ikke til egne markedsspesifikke regler.

## Bruk på nettsiden

Åpne «Ny indikator · TMG Dagsbekreftelse (kjøp/selg · 1 time)» ved de andre indikatorene.
Velg BTC/ETH/SOL og historisk vindu. Grafen viser timeslys, dagsnivåer, KJØP, SELG/SHORT, LUKK og
grå avviste signaler, med begrunnelse i tabellen. Markører er signaler ved close, ikke faktiske fyll.
Kapitalgraf/tabell er alltid tydelig merket som historisk kontroll.

«Hent siste markedsdata» gjør tre offentlige Binance-kall først ved knappetrykk:
1 000 futures-timer, 300 futures-døgn og 1 000 spot-timer. Bare avsluttede, sammenhengende lys godtas.
De siste ca. åtte døgnene tegnes etter 800 timers oppvarming. Endelig historikk kan avvike fra full historikk.
Feil/stale data gir ingen nye signaler. Ingen nøkler, private kontoer, ordre eller automatisk oppstart brukes.

## Pine og begrensninger

`tradingview/tmg_daily_confirmation.pine` er en **egen Pine v6-indikator** med kjøp/salg/lukk og alarmer,
uten `strategy.*`-ordre eller ordrewebhook. Velg standard 1-times Binance perpetual og samme mynts
spotsymbol i input. Original TMG-Pine og `tmg_trades_weex.py` er uendret.

Indikatoren bruker en eksplisitt close-tilstand for V2 og samme filter som Python. Den nye Pine-porten
har Python-portens oppdateringsanker for volumprofilen og RSI-start, og bruker ingen brokeremulator
til å avgjøre grunnsignalene. Dette gjelder den nye filen; originalstrategien er ikke endret.
Lokale paritetstester betyr ikke at faktisk kompilering/kjøring hos TradingView er verifisert.
Separat kontroll av januar–april 2024 ga **0 signal-/posisjonsavvik på 8 712 ekte timer**
fordelt på BTC/ETH/SOL (2 904 hver), inkludert dagsnivåer. Se `pine_parity.json`.
Stigende og fallende syntetiske forløp tester også kjøp, short, lukk og avviste signaler.

Grunnmodellens CLV-volumtrykk er prisestimert, ikke målt taker-CVD. Ingen OI, likvidasjonskart eller ETF-data
er rekonstruert fra OHLCV. Reelle børs-SL/TP, ordrebok, forsinkelse, gearing og likvidasjon er ikke fullstendig
modellert. 1×-tall kan ikke skaleres lineært til 50×. Pine-signaler er ikke native backtestresultater.
Kontrollhistorikken var kjent fra tidligere forsøk; videoene er publisert etter mye av historikken.
Dette er retrospektivt og har ingen urørt holdout eller bekreftet framtidig lønnsomhet.
