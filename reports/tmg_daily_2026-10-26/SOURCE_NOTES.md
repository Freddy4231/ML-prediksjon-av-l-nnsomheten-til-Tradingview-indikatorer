# Videre TMG-undersøkelse · 26. oktober 2026

To nye videoer er lest i YouTubes vanlige transkripsjonspanel. Automatiske
transkripsjoner kan inneholde feil; ingen makronyheter eller personlige gevinstpåstander
brukes som bekreftede data. Tidligere tre videoer og volumforsøket beholdes separat.

## Bekreftet dagsnivå framfor et kort kryss

[Bitcoin Golden Cross](https://www.youtube.com/watch?v=_GqbaF4lvrI), publisert 14. juli 2026:

- **01:00–01:43:** daglig 50- og 200-dagerssnitt, og lukking over dagsnivå som viktig bekreftelse.
- **06:11–07:09:** et bullish firetimerskryss kan etterfølges av fall; drøfter dagsnivåer og bruk av markedsdata sammen med teknisk inngang.
- **15:06–15:23:** vent på lukking og eventuelt stå over etter ugyldiggjøring; analysen er skjønnsmessig og dynamisk.

Vår avgrensede tolkning er et dagsfilter på **ferske** TMG V2-signaler. Fire faste
profiler: siste komplette dagsclose mot SMA50, mot SMA200, SMA50 mot SMA200 og
samstemt close/SMA50/SMA200. Hver testes med/uten forrige helgefilter og med begge
retninger/kun long. 16 kandidater, uten ny søking på kontrollen.
Vi velger SMA som en eksplisitt implementasjonsantakelse, ikke en påstand om den skjulte Stonksy-formelen.

## Data, press og urealiserte gevinster

[Bitcoin $43K Liquidity Grab](https://www.youtube.com/watch?v=kPYrp8Wzg3Q):

- **02:06–02:27:** døgnets spot/futures-forhold sammenlignes med 30 dager.
- **04:01–04:16:** se etter retningen på volumet rundt US-åpningen.
- **07:07–07:22:** nivåer fra likvidasjonskart må vurderes sammen med posisjonsdata.
- **14:03–14:15:** urealiserte gevinster er ikke det samme som lukkede handler.

Dette støtter etterprøvbar filtrering og kostnadsregnskap. Opptrapping av gearing/DCA
fra videoen kopieres ikke inn som en dokumentert forbedring. Ingen påstand om at
hele kanalen er gjennomgått eller om at kanalens faktiske handel er verifisert.

## Kausalitet og overføring

Dagsfilteret bruker gårsdagens ferdige UTC-døgn ved starten av neste timebar.
På Pine-siden tilsvarer det `request.security` med uttrykk forskjøvet én dag og
`lookahead_on`, slik at inneværende dags uferdige close aldri kommer inn i historikken.
Se [TradingViews offisielle forklaring](https://www.tradingview.com/pine-script-docs/concepts/other-timeframes-and-data/).
Et døgn i forskningen må ha alle 24 ekte timer. Manglende døgn fylles ikke med et falskt close.

Felles data/priser/kostnader fra `reports/tmg_channel_2026-10-26` brukes uendret.
Kontrollhistorikken er allerede kjent, og videoene er publisert etter store deler av
historikken. Dette er et retrospektivt forsøk, ikke en urørt eller framoverskuende test.
