# Nye videoideer · posisjoner, uke og bekreftelse

Gjennomgått via offentlige YouTube-transkripsjoner 26. oktober 2026:

- [The secret to timing the crypto dip](https://www.youtube.com/watch?v=3iSVeAV3Dfk),
  publisert 29. august 2026: 00:15–00:29 funding og ønsket om å vente med
  short, 00:38–00:48 spotdeltakelse sammen med salgspress.
- [Bitcoin huge liquidity gap](https://www.youtube.com/watch?v=2PUWkW400mk),
  premiere 4. oktober 2026: 00:00–00:16 lavere topp og delvis gevinstuttak;
  03:35–04:25 spot/futures-deltakelse og press; 05:03–05:35 sammenligning
  av sju dagers ordreflyt; 06:38–08:22 prisintervall og topper;
  10:44–12:33 pris sammen med OI og vurdering av om en handel skal holdes.
  01:16 nevner «1248», men en full oppskrift gis ikke i videoen. Vi
  rekonstruerer ikke denne og øker ikke posisjonsstørrelsen.

Videoene gir skjønnsmessige markedsvurderinger. Vi tester egne tallfestede
hypoteser: pris/OI over 4/24t, motsatt spot/futures-press, uken som vindu,
to bekreftede UTC-4t-blokker og eksplisitt ventetid høyst seks timer etter
et nytt grunnsignal. Ventetid er en ny regel, ikke gjenbruk av et signal uten
utløpsdato. Maks seks timer, kansellering ved nytt/endrede grunnmål. Ingen
privat gruppe, påstått gevinst, makronyhet eller konto brukes som fasit.

Datagrunnlag: [Binances offentlige arkiv](https://data.binance.vision/?prefix=data/futures/um/daily/metrics/BTCUSDT/),
daglige metrics-ZIP med publisher CHECKSUM. sum_open_interest brukes som en
egen samlet OI-serie, med prosentendring innen samme symbol, aldri som målt
åpning av bare long eller short. OI har alltid begge kontraktssider. Dette
er én børs, ikke kanalens aggregerte OI eller globale spotpress.

[Binances åpne sak 491](https://github.com/binance/binance-public-data/issues/491)
og [sak 509](https://github.com/binance/binance-public-data/issues/509) beskriver
uavklarte historiske tids-, enhets- og versjonsspørsmål. Disse er rapporter og
forespørsler, ikke bekreftede datakorreksjoner fra børsen. Vi forskyver ikke
enkelte perioder ut fra påstandene. Vi beholder originalenes hash/tidsstempler,
utelater motstridende duplikater og bruker arkivtid +65 minutter som en
konservativ antakelse. Data får deretter ikke fylles fram mer enn to timer.
Dette dokumenterer ikke faktisk historisk API-tilgjengelighet eller fravær
av tilbakefylling. Alle OI-resultater er derfor eksplorative, og en OI-vinner
kan ikke godkjennes som produksjon bare på grunnlag av denne testen.

[TradingView dokumenterer OI for kryptoderivater](https://www.tradingview.com/support/solutions/43000685269-open-interest/).
Datafeedens historie, enheter og tidspunkt må verifiseres separat før en
Binance-arkivtest kan kalles en identisk TradingView-port. Ingen OHLCV-proxy
merkes som ekte OI, og ingen native TradingView-kjøring er bekreftet.
