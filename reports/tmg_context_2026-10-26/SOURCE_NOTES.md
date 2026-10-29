# Mer fra TMG-kanalen · 26. oktober 2026

To ytterligere videoer er lest i YouTubes ordinære transkripsjonspanel. Makropåstander,
markedsprognoser, personlige kontosaldoer og gevinstpåstander er ikke verifisert eller brukt som testdata.
Ingen grupper er kontaktet, kontoer opprettet eller betalt innhold åpnet.

## 1. Faktisk lukking, motstridende signaler og kompresjon

[I Closed My Bitcoin Short Position](https://www.youtube.com/watch?v=o1Jly6k1BcU):

- **00:16–00:32:** beskriver lukking fordi en begrenset motbevegelse ville ha gitt et stort kontanttap.
- **01:30–01:47:** motstridende observasjoner gjør det vanskeligere å holde store posisjoner.
- **02:50–03:18:** sammenligner spot/futures-deltakelse og retningen på volumet.
- **03:18–04:01:** avtagende volum, stadig trangere prisområde og historisk volatilitet før mulig ekspansjon.
- **04:33–05:15:** skiller kortsiktig motbevegelse fra en langsiktig markedsoppfatning.
- **06:36–06:52:** likvidasjonskart viser utsatte nivåer, ikke en sikker retning.

Vår operasjonalisering er prisbredde, døgnvolum, retningsandel og ekstra close-utganger
etter gevinst eller ugyldig dagskontekst. **Alle vinduer, ATR-tall, kvantiler og kostbuffere er våre**;
videoen gir ingen slik komplett algoritme. Bollinger-bredde er ikke kanalens volatilitetsside,
og tidlig exit/ATR-linje er ikke en rekonstruksjon av hans nyhetsavgjørelse.

## 2. Flere markeder og dags-RSI

[My $6 Million Bitcoin Short Is Now in Profit](https://www.youtube.com/watch?v=SOJKHYnB0eQ):

- **04:49–05:20:** ser på Ethereum når Bitcoin-data gir et uklart bilde.
- **06:21–06:49:** skiller pris og open interest; OI alene sier ikke prisretning.
- **07:38–07:52:** drøfter bekreftet tap av et støttenivå på timesgrafen.
- **08:39–09:08:** bruker dags-RSI til å drøfte en strukket oppgang.

Vi tester derfor også et avgrenset dags-RSI-filter (ingen nye long ved RSI≥70 eller short ved RSI≤30).
Dette er en antakelse med faste standardnivåer, ikke en eksplisitt regel som TMG oppgir.
BTC/ETH/SOL analyseres likt; vi velger ikke et annet marked etter kontrollresultatet.

## Testgrense

11 faste kandidater legges **rundt dagens fryste `daily50_weekend_both`** på 1 time.
Samme ekte perpetualdata, kostnader, funding, neste-open-fyll og retning for alle.
Valg 2021–23, godkjenning2024; ingen ekstra 2025–26-rangering. Dags-RSI bruker bare gårsdagens
close. Gevinstlinjer bygges fra beste avsluttede close og gjelder fra neste bar, ikke fra tidligere intrabar-ekstremer.
Ingen økning i gearing, DCA, nyhets-, OI-, ETF- eller likvidasjonsproxies.
Videoene og kontrollhistorikken er retrospektivt kjent. Ingen urørt test eller sikker lønnsomhet påstås.
