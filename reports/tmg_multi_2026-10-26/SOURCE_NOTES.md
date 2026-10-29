# Flere TMG-videoer · 26. oktober 2026

Kildene ble lest via YouTubes offentlige transkripsjon. Vi har ikke verifisert
kanalens gevinster eller gjenskapt private signaler. Tallfestede regler er våre
testbare hypoteser, fastsatt før dette forsøket. Ingen konto, Telegram-gruppe,
verving, innskudd, identitetsløsning eller andre handlinger i videoene er brukt.

- [Trading challenge, 17. oktober 2026](https://www.youtube.com/watch?v=1O0XCCfjvkU):
  02:06–02:22 drøfter handelskostnader; 09:18–09:33 forklarer forholdsmessig
  størrelse; 12:42–12:58 handler om å lukke tap; 14:17–14:52 sammenholder
  spotpress i flere tidsvinduer og prisretning. Størrelse/gearing er ikke økt.
- [Heatmap trap, 16. august 2026](https://www.youtube.com/watch?v=wkgWeJafDV8):
  02:16–02:44 bruker funding sammen med posisjonskontekst; 03:00–03:52
  vurderer spot/futures-deltakelse; 08:01–09:21 skiller kjøp fra åpning av long
  og drøfter finansieringskostnader; 16:58–17:11 tilpasser aktivitet til
  tydelige/uklare forhold. Dette er ikke en oppgitt presis fundingformel.

Vi tester derfor åtte avgrensede hypoteser rundt det eksisterende dagsfilteret:
4/24 timers spotenighet som inngangs- eller utgangskontekst, separat med ekte
Binance-taker-volum og med CLV fra OHLCV, samt trehendelsers kjent funding.
Funding er minst én time gammel. Fremtidig funding, OI, heatmaps, ETF, globale
volum eller private data utledes ikke fra lys. CLV er ikke ekte kjøp/salg-volum.

Ekte taker-flagg og realisert funding kan ikke uten videre overføres til en
vanlig Pine-OHLCV-serie. En lønnsom Python-test av dem ville ikke dokumentert
en identisk TradingView-indikator. En egen CLV-test undersøker den portable
pristolkningen uten å blande seriene. Kanalens andre påstander er ikke
kontrollert og brukes ikke som fakta eller signaler.
