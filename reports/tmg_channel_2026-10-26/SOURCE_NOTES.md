# TMG Trades · primærkilder lest 26. oktober 2026

Kanal: [TMG Trades](https://www.youtube.com/@tmgtrades). Tre videoer er gjennomgått via
YouTubes synlige transkripsjonspanel og videobeskrivelse. Den automatiske eksporten svarte
først at transkripsjon manglet; den vanlige «Show transcript»-visningen fungerte.
Transkripsjonene kan inneholde ASR-feil. Ingen påstand om at hele kanalen er gjennomgått.
Kanalens oppgitte personlige gevinster er ikke kontrollert eller brukt som evidens.

## 1. US-åpning og motstridende volum

[BLACK MONDAY?? BITCOIN MONDAY US OPEN LIQUIDATION TRAP!!!!!](https://www.youtube.com/watch?v=6JiCtRBSsGo)

- **03:54:** drøfter delvis gevinsttaking før en mulig motbevegelse.
- **05:37–06:47:** sammenligner spot/futures-volum, siste døgn mot omtrent én måned,
  og vekter spottrykk høyere når deltakelsen er sterk. Futureskjøp kan også skyldes shorts som lukkes.
- **08:40–11:14:** bruker pris sammen med open interest til å tolke posisjonsbygging og lukking.
- **14:17:** beskriver en stopp nær inngang etter at gevinst allerede er tatt.

Vår testbare tolkning: filtre for reelt spottrykk, enighet mellom markedene og sterkere
spotdeltakelse. Tall, vinduer og terskler i testen er våre operasjonaliseringer; videoen gir ingen komplett algoritme.

## 2. Helgefeller og likviditetskart

[BITCOIN WEEKEND LIQUIDATION TRAP!? BITCOIN HEATMAP WARNING!!](https://www.youtube.com/watch?v=OCnkVk2KG7o), publisert 23. oktober 2026.

- **04:00–04:45:** sammenligner døgnets spot/futures-forhold med 30 dager og ser etter enighet i salgstrykket.
- **06:37:** beskriver likvidasjonskart som nivåer der posisjoner kan rammes, ikke som retningsprognoser.
- **07:04–08:40:** kombinerer prisendring og open interest for å tolke en mulig utløsning av posisjoner.

Vår testbare tolkning: sterkere volumkrav for nye helgeinnganger. Dette er en hypotese
inspirert av konteksten, ikke en regel han eksplisitt oppgir. OHLCV-sveip og VRVP er ikke reelle likvidasjonskart.

## 3. Spotdeltakelse og korte vinduer

[BITCOIN HEATMAP LIQUIDATION WIPEOUT?! URGENT OPEN INTEREST WARNING!!](https://www.youtube.com/watch?v=5pBOHaOeu4U), publisert 25. september 2026.

- **00:42–01:10:** omtaler stopp etterfulgt av en ny inngang; ikke belegg for å fjerne stopp.
- **03:38–05:10:** legger vekt på spottrykk sammen med deltakelse mot 30-dagersnivået.
  Nevner også fire timers grafhistorikk med 15 minutters rullering i sitt analyseverktøy.
- **10:25–11:10:** understreker at kartene ikke angir retning, og drøfter pris sammen med OI rundt US-åpningen.

Vår testbare tolkning: korte/lange trykkvinduer, og en separat hypotese om å vente gjennom
første time etter New York-åpningen. Sistnevnte er vår regel, ikke et ordrett råd fra kanalen.

## Hva som kan testes og overføres

| Idé | Datatilgang | Forbehold |
|---|---|---|
| Aggressivt kjøps-/salgsvolum | Ekte Binance `taker_buy_volume` og totalvolum | Én børs; ikke global ni-octib-aggregasjon |
| Spotdeltakelse mot futures | Samme mynts spot-/perp-basevolum | Ikke kanalens globale USD-måling |
| Fundingposisjonering | Faktiske historiske fundingrater | For inngang brukes bare publiserte, minst én time gamle rater |
| US-åpning | Lokal børs-klokke `America/New_York` | Sommertid håndteres; ingen historisk nyhetskalender |
| OI/likvidasjonskart/ETF | Ikke komplett tidsstemplet historikk i prosjektet | Ikke testet og aldri rekonstruert fra pris/volum som om det var målt |

[Stonksys beskrivelse](https://www.stonksy.io/about) omtaler bakgrunn og momentumlinje som ulike
signaler, uten å oppgi de lukkede formlene. Dagens TMG er vår uavhengige rekonstruksjon.
[nioctib.live](https://nioctib.live/) beskriver aggregert ordreflyt fra flere børser.

Binances [offisielle dataformat](https://github.com/binance/binance-public-data/) skiller mellom
totalvolum og taker-kjøp. Det gir målt aggressivt kjøpsvolum; lysfarge alene gjør ikke det.
TradingViews [footprint-dokumentasjon](https://www.tradingview.com/pine-script-docs/concepts/other-timeframes-and-data/#requestfootprint)
beskriver derimot klassifisering fra intrabar prisbevegelse. Det er en annen datakilde enn
Binances taker-flagg og kan ikke presenteres som en identisk Pine-port av denne testen.

Videoene er publisert etter store deler av testhistorikken. Historisk testing av disse
ideene er derfor retrospektiv og ikke dokumentasjon av en strategi som var tilgjengelig da.
