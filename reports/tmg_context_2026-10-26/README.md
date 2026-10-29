# TMG Dagsbekreftelse + gevinstvern · valgbart forskningsalternativ

**26. oktober 2026. Bedre kjent BTC-resultat, svakere ETH/SOL og mislykket placebokrav. Ingen ny motorstandard.**

## Kanalundersøkelse og test

To nye videoer ble lest i YouTubes ordinære transkripsjonspanel: lukking ved uklart bilde,
volum som tørker opp, priskompresjon, forskjeller mellom markeder og dags-RSI.
Se [kildenotatene](SOURCE_NOTES.md). Gevinstpåstander, nyhetsfortellinger og kontosaldoer i videoene
ble ikke bekreftet eller brukt som treningsdata. Vi bruker ikke oppdiktet OI eller likvidasjonskart.

**11 faste kandidater**, alle rundt dagens uendrede `daily50_weekend_both`, 1 time, begge retninger:
prisbredde mot tidligere kvartil25, økende bredde, bredde+volum, retningsandel, dags-RSI70/30,
dagskontekst som utgang, EMA50/VWAP etter gevinst, inngangsvern og to ATR-trailinger.
Formlene og tallene er våre tolkninger, ikke en algoritme som kanalen oppgir.

Samme **171 185 SHA-kontrollerte futures-timer** på BTC/ETH/SOL som tidligere.
Valg på 2021–23, godkjenning i 2024. Bare én kandidat bestod: **`entry_guard`**.
Ingen av de øvrige kandidatene er rangert på 2025–26. Plan, datakildehash, utvikling,
fryst valg, placebo og kontroll er bevart separat.

## Den valgte regelen

Inngangene bruker fortsatt dagsclose/SMA50, helgens spotdeltakelse og ferske V2-signaler.
Fra signalprisen måles beste **avsluttede close** i riktig retning. Når dette har gitt
minst **2 ATR14 fra inngangslyset**, settes en fast linje:

- Long: signalpris ×1,003.
- Short: signalpris ×0,997.
- Linjen gjelder fra **neste bar**. LUKK avgjøres først på close som bryter linjen.
- Fyll skjer neste ekte open i simuleringen, aldri til en bedre, hypotetisk pris på linjen.
- Et gammelt signal åpnes ikke på nytt etter slik lukking.

Dette er en **close-regel**, ikke en børs-SL, hvilende limit eller garantert breakeven.
Gap, funding og kostnader kan gi tap selv etter aktivering. Linjen bygges fra signalprisen,
ikke faktisk børsfyll. Ingen økt gearing, ekstra innskudd eller DCA.

## Kostnadsregnskap og kjent kontroll

Januar 2025–september 2026, begge retninger, **1× notional**, fast mengde per handel.
0,15 % kostnad per side + faktisk funding; stress dobler begge. Samme neste-open-motor i begge rader.

| Marked | Dagsfilter | Med vern | Dagsfilter stress | Med vern stress | Fall før | Fall med vern |
|---|---:|---:|---:|---:|---:|---:|
| BTC | −3,91% | **+31,22%** | −11,97% | **+20,93%** | −29,83% | −17,02% |
| ETH | +161,38% | **+31,98%** | +146,70% | **+24,51%** | −39,44% | −24,77% |
| SOL | +22,07% | **−18,74%** | +12,54% | **−24,19%** | −37,77% | −46,25% |

BTC hadde 23 handler, ETH 17 og SOL 20 i begge varianter; vernets endring gjelder først og fremst utganger.
Dette er ikke en generell forbedring: vernet tar også bort store vinnerhandler i ETH og forverrer SOL.
Det er ikke valgt egne regler for de tre markedene etter å ha sett dette.

## Hvorfor dette ikke blir ny standard

Valget forbedret BTC i trening og 2024 og bestod de opprinnelige utviklingskravene.
Kjent kontroll bestod positiv BTC i normal/stress, relativ forbedring, median, to positive markeder og drawdown.
**Placebokravet feilet likevel.** Åtte kontekstserier ble forskjøvet bakover 720–3240 timer:
BTC trenings-stress var **+135,19 %**, under deres 75-persentil **+139,84 %**.
2024-stress var +45,34%, over placeboenes median +35,76%.

Dette er en følsomhetskontroll, ikke en formell signifikansberegning.
Vi senker ikke kravet etterpå. **`released=false`**, dagens variant forblir standard.
Gevinstvernet vises som et eksplisitt valgbart forskningsalternativ. Ordremotoren er uendret.

## Nettside og Pine

Under «TMG Dagsbekreftelse» velges «Dagsbekreftelse + gevinstvern · forskning».
Grafen viser kjøp/salg/lukk og oransje close-vern. Tabellen viser samme kostnader og tidsrom
for begge varianter. Dagens standard finnes fortsatt i samme velger. Alle 11 utviklingsforsøk
og nye videolenker vises i en egen accordion.

Knappen for siste markedsdata bruker bare offentlige Binance-data (1 000 futures-timer,
300 døgn og 1 000 spottimer), bare ferdige/sammenhengende lys. Ingen nøkler, private kontoer eller ordre.
Avgrenset live-historikk kan avvike fra hele forskningshistorikken; grafen vises etter 800 timers oppvarming.

`tradingview/tmg_daily_profit_guard.pine` er en egen signalindikator uten brokerordre/webhook.
Velg **vanlige Binance perpetual-lys, 1 time, og samme mynts spotsymbol**. Manglende samsvar
sperrer signalene i denne nye Pine-filen. Originale TMG-/Dagsbekreftelse-filer beholdes uendret.
Pine og Python kontrolleres lokalt mot samme signaler og vernnivåer; dette bekrefter ikke faktisk
kompilering eller native strategiresultater hos TradingView. Separate vinduer på BTC/ETH/SOL
(januar–april 2024, totalt **8 712 timer**) gav 0 avvik i signaler, målposisjon og vernnivå.

Historikken har vært brukt i flere forsøk, videoene er retrospektive, og det finnes ingen urørt holdout.
Ingen full ordrebok-/WEEX-SL/TP-/gearing-/likvidasjonsmodell. 1×-tall kan ikke skaleres lineært til 50×.
