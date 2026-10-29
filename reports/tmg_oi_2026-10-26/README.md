# TMG OI-bekreftelse: lovende historisk kandidat

Tolv regler ble låst og sammenlignet på BTC, ETH og SOL. Nye primærkilder:
YouTube 3iSVeAV3Dfk og 2PUWkW400mk, relevante seksjoner via transkripsjoner;
se SOURCE_NOTES.md. 3 715 publisher-SHA-kontrollerte metrics-arkiver ga
1 068 226 OI-observasjoner for 2021–2024. ETH og SOL mangler 334 tidlige
arkivdøgn hver. De fylles ikke med kunstig OI. Ingen motstridende duplikater.

Valgt på trening 2021–23 med godkjenning i 2024: **pending_oi4**. Et ferskt
signal fra uendret daily50_weekend_both oppretter en plan. Inngang krever
stigende OI siste fire timer og prisendring i signalretning siste fire timer.
Planen kan vente høyst seks timer, erstattes av et nytt signal og kanselleres
ved endret grunnmål, utløp eller hullsperre. Utgang følger grunnmålet. Ingen
gjeninntreden etter utløpt plan, størrelseøkning eller nye kontoinnstillinger.

| Marked | Referanse 2024 | Kandidat 2024 | Referanse stress | Kandidat stress |
|---|---:|---:|---:|---:|
| BTC | +46,30 % | +65,44 % | +29,99 % | +50,31 % |
| ETH | −3,92 % | +12,54 % | −13,24 % | +2,71 % |
| SOL | +30,51 % | +38,90 % | +15,79 % | +23,55 % |

BTC trening: +99,05 % → +120,51 %, stress +72,32 % → +95,27 %,
28 → 19 handler. BTC største fall 2024: −36,38 % → −27,20 %.
ETH-trening er fortsatt negativ (−39,01 %), SOL-trening har store fall
(−63,58 %). Dette er ikke en generelt robust eller risikofri modell.

Kandidaten bestod de forhåndslåste utviklingskravene og placebokontrollen:
BTC trenings-stress +95,27 % mot forskjøvede konteksters 75-persentil
+55,65 %; 2024 stress +50,31 % mot median +23,36 %. Ingen annen kandidat
ble valgt fra 2025–26. Ingen ny kontroll på nyere OI ble kjørt: låst plan
krever verifisert OI-tilgjengelighet/feed før videre godkjenning. released=false.

En ekstra etterkontroll avdekket at forskyvning ga mindre dekning tidlig i
treningen. `placebo_coverage_audit.json` bruker derfor samme kausale
tilgjengelighetsmaske for kandidaten og alle åtte placeboene (20 145 av
26 280 treningstimer; 8 667 av 8 784 timer i 2024). Kandidaten består også
denne sammenligningen: BTC trenings-stress +64,69 % mot 75-persentil
+50,91 %; 2024 +50,31 % mot median +23,36 %. Dette er en etterkontroll,
ikke en ny forhåndslåst godkjenning eller ny utvelgelse.

Alle tall er Python-simulering: signalclose, neste ekte åpning, 1× fast mengde
per handel, 0,15 % per side og faktisk funding; stress dobler begge. Eksisterende
regel for hull og oppvarming beholdes. Historikken er kjent fra tidligere
forsøk, ikke urørt. Native TradingView-/WEEX-ordre, gearing og likvidasjon
er ikke testet. Historisk OI er en arkivserie med uavklarte tilgjengelighets-
og versjonsspørsmål; +65 min er en antakelse, ikke dokumentert publisering.

Ny nettsidevisning: dagens signaler og valgt kandidat i historisk 2024-graf,
OI-mengde, endring 4/24 timer, dato for kjent OI ved hvert signal og alle
tolv forsøk. Ingen trening eller datahenting ved visning. Kildeendringer
skjuler rapporten gjennom et SHA-manifest.

Separat `tradingview/tmg_oi_confirmation.pine` er en forskningsprototype med
KJØP, SELG/SHORT, LUKK og VENT, ikke en dokumentert identisk datakobling.
Les `tradingview/TMG_OI_README.md`. Isolert Pine-ventetilstand på 6 552 ekte
timer ga null avvik i posisjon, plan, kjøp, salg og lukk mot Python. Dette
tester ikke hele skriptets native request.security/feed eller kompilering
i TradingView. Originale indikatorer og ordremotoren er uendret.

Reproduser: `research.tmg_oi_data` (offentlig nedlasting),
`research.tmg_position_search --develop`, deretter `--control`,
`research.export_tmg_oi`, `research.tmg_oi_pine_audit` og
`research.tmg_position_report`. Plan og tidligere kontroll overskrives ikke.
