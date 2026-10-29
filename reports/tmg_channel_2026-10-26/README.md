# TMG-kanalen · ekte ordreflyt og helgefilter

**26. oktober 2026. Relativ forbedring funnet, men ingen godkjent erstatning av TMG V2.**
Arbeidet er gjort uten subagenter. Original Pine, signalmotor, konto, gearing og ordreutførelse er beholdt.

## Hva kanalundersøkelsen faktisk omfatter

Tre av kanalens videoer er lest via YouTubes vanlige transkripsjonspanel, supplert med
videobeskrivelser og de offisielle sidene til Stonksy og nioctib.live. Se
[kildenotatene](SOURCE_NOTES.md) for direkte videolenker og tidsstempler.
Kanalens påstander om egne gevinster er ikke verifisert og inngår ikke som testresultater.

Nye, målbare hypoteser er faktisk spottrykk, spot/perp-enighet, spotdeltakelse mot en lengre
referanse, fundingposisjonering, helgekontekst og New York-åpningen. Vi har ikke rekonstruert
OI, ETF-strømmer eller likvidasjonskart fra OHLCV. Manglende slike dataserier er ikke skjult
med prisproxies. Korte prisbevegelser kan ikke alene fortelle hvem som åpner eller lukker posisjoner.

## Datagrunnlag

- **235 offentlige Binance USD-M-futuresarkiver**, hvert kontrollert mot Binances SHA-256.
  Åtte forespurte SOL-måneder før kontraktstart mangler. Ingen hel måned mangler etter første arkiv.
- **171 185 ekte futures-timer**: BTC 59 160, ETH 59 160, SOL 52 865. Nye ETH-/SOL-perpetualserier
  gir 112 025 timer som ikke fantes i prosjektets tidligere futuresfil. Eksisterende filer er ikke overskrevet.
- Ekte taker-kjøpsvolum og totalvolum er bevart, både base- og quotevolum for perpetuals.
  Kjøp/salg i trykkfiltrene kommer fra taker-flagg, ikke lysfarge. Spotgrunnlaget er de tidligere
  SHA-bundne timesfilene med ekte `taker_buy_volume`.
- Deltakelse måles i samme mynts **basevolum på én børs**. Dette er en testbar tilnærming til
  kanalens tema, ikke en kopi av det globale USD-aggregatet på nioctib.live.
- Alle sammenligninger her bruker **ekte perpetualpriser**, ikke spot som prisproxy.
  Felles sluttdato er 1. oktober 2026 kl. 00:00 UTC, eksklusiv; siste komplette futuresmåned er september.

Data og arkivkontroll: `data_manifest.json`. Data er lagret isolert i `data/tmg_channel_2026-10-26/`.

## Test som ble låst før resultater

Ti profiler × long/short eller kun long = **20 oppsett**, inkludert to uendrede V2-referanser.
Fire- og 24-timers trykkfiltre er egne faste hypoteser; kanalens omtale av et 15-minutters
rullerende analysevindu er ikke testet på timesdata som om minutthistorikk var tilgjengelig.

Valg på **2021–2023**, godkjenning i **2024**, på BTC/ETH/SOL. 2020 gir oppvarming og sikrer
at SOL har en hel første evalueringsperiode etter kontraktstart. Kravene omfatter positiv avkastning
med kostnadsstress, antall handler, positive år, drawdown og forbedring mot identisk V2.
Høyest median stress-Sharpe i utviklingsperioden avgjør blant godkjente oppsett.

Valget ble **`weekend_confidence_long_only`**: ved et nytt long-signal i helgen kreves
`sum(spotvolum, 24 t) / sum(perpvolum, 24 t) >= sum(spotvolum, foregående 720 t) / sum(perpvolum, foregående 720 t)`.
Referansevinduet ligger før siste 24 timer og overlapper ikke disse. Ukedager følger V2.
En avvist inngang åpnes aldri forsinket på et gammelt signal. Utgangene er uendret.

Dette er vår operasjonalisering av videofunnene, ikke en formel som TMG oppgir.

Før kontrollen ble valgt filter sammenlignet med åtte nullvarianter forskjøvet **bakover i tid**
720–3240 timer. Den bestod de forhåndslåste kravene: treningsavkastning i stress over placeboenes
75-persentil og 2024 over deres median. Dette er en følsomhetskontroll, ikke et bevis på statistisk signifikans.

På BTC ga det +48,71 % i 2021–23 og +63,08 % i 2024, mot +37,26 % / +42,93 % for identisk V2 kun long.
Deretter ble den kjente kontrollperioden kjørt én gang. Ingen andre kandidater ble rangert på den.

## Kjent kontroll · 1. januar 2025–30. september 2026

**1 time, kun long, 1× notional, 0,15 % per side + faktisk Binance-funding.**
Stress er 0,30 % per side og funding ×2. Fast kontraktsmengde per handel.

| Mynt | V2 netto | Med helgefilter | V2 stress | Med filter stress | V2 handler | Filter handler |
|---|---:|---:|---:|---:|---:|---:|
| BTC | −4,84 % | **−0,59 %** | −13,95 % | −8,88 % | 21 | 18 |
| ETH | +42,35 % | **+52,82 %** | +29,58 % | +40,59 % | 19 | 17 |
| SOL | +10,62 % | **+14,07 %** | +1,84 % | +5,89 % | 20 | 18 |

BTCs største fall ble redusert fra **−31,69 % til −28,48 %**. ETHs største fall ble litt verre
(−49,72 % → −50,23 %), og SOLs bedre (−50,72 % → −48,93 %).

**BTC er fortsatt negativ, også i stress. Kravet om positiv BTC-avkastning ble ikke bestått.**
De øvrige sluttkravene inngår i `known_control.json`; den samlede beslutningen er `released=false`.
Filteret blir ikke ny standard og kobles ikke til motoren. Resultatene viser et mulig nyttig filter
som trenger mer uavhengig testing, ikke en lønnsomhetsgaranti. Ingen markedsspesifikke varianter
ble valgt fra ETH/SOL-kontrollen.

## Hva simuleringen ikke viser

Alle ordre avgjøres ved close og fylles ved neste ekte åpning. Innganger er sperret i 800 timer
etter et kjent hull i spot eller futures, likt for kandidat og referanse. Manglende volum fylles ikke.
Funding som forklaringsvariabel må være minst én time gammel. Sommertid ved US-åpningen bruker
`America/New_York`; helgen defineres fra signalets lukking i UTC.

Dette er fortsatt en sammenligning av Python-portens målposisjon, **ikke native TradingView eller
faktiske WEEX-fyll**. Full børs-SL/TP, nødstopp, gearing/likvidasjon, ordrebok og nettverksforsinkelse
er ikke modellert. Python-POC handles neste åpning, mens Pine bruker en hvilende limitordre.
1×-tallene kan ikke omregnes lineært til et resultat ved 50× gearing.

Historikken 2025–26 er kjent fra tidligere arbeid. Videoene er dessuten publisert etter store deler
av utviklingsperioden. Forsøket er derfor retrospektivt og har **ingen urørt holdout/forward-test**.

## Visning og TradingView-panel

Nettsidens «Nye funn fra TMG-kanalen» viser identiske kapitalforløp før/etter og bekreftede
grunnsignaler, med røde kryss for avviste innganger. Det er historiske simuleringer, ikke live handler.
Visningspakken kontrolleres mot SHA av data, TMG-kilde og det nye panelet. Ingen trening/henting ved visning.

`tradingview/tmg_spot_participation.pine` er et **separat dataverktøy uten ordre og uten kjøps-/salgssignaler**.
På 1 time viser det basevolumforholdene og helgens volumkrav. Velg spot/perp for samme mynt.
Det måler ikke taker-kjøp, OI eller likvidasjoner. Volumseriene er kontrollert mot Python i prosjektets
Pine-tolk; faktisk kompilering i TradingView er ikke bekreftet. Original TMG-Pine er uendret.

Forskningskoden er `research/tmg_channel_data.py`, `tmg_channel_rules.py`, `tmg_channel_search.py`
og `tmg_channel_report.py`. Låst plan, kildehash, utvikling, valg, placebo og kontroll ligger her.
Testene dekker ekte volum mot candle-proxy, manglende data, publiseringstid, kausalitet, sommertid,
avviste innganger, kildevalidering og Pine-serier uten ordre.

Ved denne kontrollen ble en feil i den lokale Pine-tolkens kryssymbol-oppslag rettet: det forespurte
symbolet blir `self` i hjelpekonteksten, men verdien ble ikke fanget tilbake til hovedkjøringen.
Dette gav tomme serier ved korrekt `request.security` mot et annet symbol. Rettingen påvirker ikke
de fryste TMG-testresultatene, som beregnes direkte i Python uten Pine-tolken.
