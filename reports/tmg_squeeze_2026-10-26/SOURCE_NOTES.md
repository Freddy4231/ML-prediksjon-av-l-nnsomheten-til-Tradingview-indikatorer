# Nye TMG-videoer · spotstøtte og vending etter OI-fall

Lest 26. oktober 2026 via YouTubes offentlige transkripsjon. Ingen private grupper,
kontodata eller betalte formler. Notatene er egne sammendrag. Transkripsjoner kan inneholde feil.

- [BITCOIN LIQUIDATION WIPEOUT!!! FED DECISION BITCOIN WARNING!! I'M $5.5M SHORT](https://www.youtube.com/watch?v=BesZsOHBLWA)
  (vist som premiere to timer før lesingen; nettlesertittel hadde en annen TUESDAY-tittel).
  02:20–03:24: funding tolkes sammen med spot og OI. 04:43–05:41: relativ spotdeltakelse og
  forskjellig kjøps-/salgspress på BTC og ETH. 08:16–09:18: pris og OI på 15 minutter,
  stigende pris med fallende OI nær toppen av et intervall og tålmodig inngang etter bevegelsen.
  Videoens sikkerhet om hvem som handler/likvideres kan ikke fastslås fra samlet OI alene.
- [BITCOIN HEATMAP TRAP!! MSTR 12% YIELD WARNING... WATCH BEFORE TUESDAY US OPEN!](https://www.youtube.com/watch?v=gY46g0pz7To)
  publisert 17. august 2026. 05:05–05:30: Asia-/USA-sesjoner. 07:12–09:03: intervall,
  oppbygging og avvikling av posisjoner, og egen dealer-delta-proxy uten full offentlig formel.
  13:08–13:48: sammenligner spot/futures-volum med tidligere måned og spør om brudd støttes
  av spot. 15:04–15:40: OI faller mens prisen stiger, mulig avvikling av shortposisjoner.
- [How I Turned Open Interest Into Consistent Profits](https://www.youtube.com/watch?v=ChYk8XWXlgE)
  publisert 16. oktober 2026, 40 sekunder. Lest hele transkripsjonen; påstander om egne
  resultater og en introduksjon, ingen ny presis regel. Brukes ikke som profittbevis.

## Vår egen operasjonalisering, låst før resultater

Åtte hypoteser på 1 time. Samme dagsfilter og seks timers gyldighet for et nytt signal.
Referanser: dagsfilteret og forrige OI-kandidat `pending_oi4`. Alle retninger, fast 1×.
1. OI/pris-bekreftelse + ekte spotkjøpsandel på riktig side av 50 % på 4 timer.
2. Som 1, men 24 timer.
3. OI/pris-bekreftelse + spot/perp-basevolum siste24t minst som foregående720t (ikke overlapp), alle ukedager.
4. OI/pris-bekreftelse + close utenfor high/low fra foregående24t.
5. Alternativ bekreftelse: OI4t faller, pris4t går motsatt grunnsignalet, pris1t vender med signalet.
6. Alternativ bekreftelse: OI4t faller og gjeldende time bryter foregående24t ytterpunkt og lukker tilbake innenfor nivået.
7. Opprinnelig OI-inngang; lukk når OI4t faller, pris4t fortsatt med posisjonen, men pris1t går motsatt.
8. Opprinnelig OI-inngang; lukk når ekte spotpress4/24t og pris1t går motsatt posisjonen.

Dette er eksplorative tolkninger, ikke TMGs egen kildekode eller en gjenskaping av dealer-delta.
Ingen etterfølgende terskeljustering. Oppsett 5/6 erstatter selve OI-bekreftelsen, øvrige utvider den.
Ingen likvidasjonskart, ETF-/børsbeholdning, makronyheter eller USDC-aggregat blir fabrikkert.
1t-regler gjenskaper ikke videoens 5/15-minutters timing. Kun Binance USDT-paret.

## Datagrunnlag og begrensninger

Eksisterende SHA-kontrollerte BTC/ETH/SOL perpetual-, spot-, funding- og OI-filer brukes.
OI observeres med 65 minutters antatt publiseringsforsinkelse og maksimalt to timer framfyll.
Historisk publisering/versjoner er fortsatt ubekreftet. Ingen ny 2025–26-kjøring i dette forsøket.
2021–23 brukes til rangering, 2024 til godkjenning. Begge er gjenbrukt historikk; dette er ikke urørt holdout.
Posisjoner utføres neste ekte open; 0,15 %/side + faktisk funding, stress dobler begge.

[TradingViews OI-definisjon](https://www.tradingview.com/support/solutions/43000685269-open-interest/)
angir antall utestående kontrakter, ikke hvilken aktør som åpner/lukker en bestemt side.
Kildebegrensningene for historisk OI fra forrige forsøk gjelder uendret.
