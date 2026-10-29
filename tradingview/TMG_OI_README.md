# TMG OI-bekreftelse · forskningsprototype

Ny separat fil: `tmg_oi_confirmation.pine`. Dagens Dagsbekreftelse og
Gevinstvern beholdes. Ingen ordre eller kobling til WEEX er lagt til.

1. Velg vanlige 1-timeslys på BINANCE:BTCUSDT.P, ETHUSDT.P eller SOLUSDT.P.
2. Lim inn hele filen i Pine Editor og legg indikatoren på grafen.
3. Sett «Spotsymbol» til samme mynt, f.eks. BINANCE:ETHUSDT for ETH-grafen.
4. VENT betyr en plan som venter på pris/OI-bekreftelse. KJØP / SELG gjelder
   bare ved avsluttet timebar. LUKK følger grunnmodellens endrede mål.
   Varsler velges som «Once per bar close». En utløpt plan åpner ikke senere.

Regel: uendret dagsfilter → fersk plan → stigende OI og pris i signalretning
over fire timer, innen seks timer. Dette er vår tallfestede hypotese fra
videoene, ikke kanalens proprietære indikator eller en kopi av private trades.

**Dette er ikke en bekreftet identisk TradingView-backtest.** Python bruker
historiske Binance-metrics med arkivtid +65 minutter. Pine bruker
TradingViews egen OI-ticker for samme kontrakt, f.eks.
BINANCE:BTCUSDT.P_OI, og siste 5m-underbars `open[12]` / `time[12]`.
På en komplett tidsakse er dette 65 minutter gammelt ved 1h-close.
For gammel eller manglende OI gir ingen erstatning med pris/volum.

Manglende underbarer, ulike OI-verdier/tidsstempler og forskjellig tilgjengelig
historikk kan gi andre signaler. Sikkerhetsfilteret ved pris-/spotgap kan
også gi forskjell. TradingView har begrenset underbarhistorikk avhengig av
abonnement; eldre bars uten data får ingen nye OI-innganger.
[Offisiell forklaring av lavere tidsrammer](https://www.tradingview.com/pine-script-docs/concepts/other-timeframes-and-data/#lower-timeframes)
og [open interest](https://www.tradingview.com/support/solutions/43000685269-open-interest/).

Appens lokale Pine-test og WEEX-Pine-adapter støtter ikke lavere tidsrammer.
Denne filen skal derfor prøves direkte i native TradingView. Den er ikke
lagt i motorens innebygde skriptvalg. Automatisk handel er ikke aktivert.

Statisk lokal kompilering og isolert ventetilstand er kontrollert. På 6 552
ekte timer var det null avvik i posisjon, ventende retning, kjøp, salg og
lukk ved injisert kjent OI og grunnmål. Samplingens 65-minutters tidsregel
er en separat lokal kontroll, ikke bevis på identiske native markedsdata.
Faktisk native TradingView-kompilering/feed og framtidig lønnsomhet er ikke
verifisert. Se den komplette Python-testen i reports/tmg_oi_2026-10-26.
