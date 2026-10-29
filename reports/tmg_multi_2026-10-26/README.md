# Flere TMG-videoer: ingen godkjent erstatning

Åtte regler ble låst før kjøring på BTC, ETH og SOL, med ekte timespriser fra
Binance USD-M, ekte spot-taker-volum og historisk funding. Trening 2021–23,
godkjenning 2024. Ingen ny rangering eller evaluering av kandidatene i 2025–26
ble kjørt, fordi ingen kvalifiserte seg. Historikken er kjent fra tidligere
arbeid; dette er ikke en urørt test. Se SOURCE_NOTES.md for videoer/tidsstempler.

På BTC ga 4/24 timers ekte spotenighet +140,28 % i trening og +77,81 % i 2024
(stress +122,24 % / +63,80 %), mot dagsfilterets +99,05 % / +46,30 %
(stress +72,32 % / +29,99 %). Men det var bare 14 treningshandler, under det
forhåndslåste minimumet 15. Vi senker ikke kravet etter å ha sett resultatet.
CLV-enighet, som kan beregnes fra Pine-OHLCV, tapte 20,31 % på BTC i 2024
(stress −22,29 %). Bedre ekte-volumtest er ikke bevis for en CLV-port.

Resultater: development.json (alle åtte og referansen, alle markeder),
frozen_choice.json (selected=null), known_control.json (ingen ny kontroll),
locked_plan.json (regler, krav, data- og kildehash). Grafpakken inneholder
2024-close, eksisterende dagsfilter-signaler, ekte spotandel 4/24 t, separat
CLV 4/24 t og minst én time gammelt snitt av tre publiserte fundinghendelser.

Fyll neste ekte åpning etter signalclose; 1× fast mengde per handel; 0,15 %
kostnad per side pluss faktisk funding. Stress dobler begge. Hullregler og
oppvarming er identiske med den eksisterende dagsfiltertesten. Ingen native
TradingView-fyll, børs-SL/TP, ordrebok eller gearing/likvidasjon testes.

Ingen signalregel eller konto er endret. Nettsiden får en historisk
forklaringsvisning med alle testfeil synlig, ingen ny påstand om profitt.
Vi eksporterer ikke en villedende Pine-kopi av ekte taker/funding.

Kjør: `gradio-env/bin/python -m research.tmg_multi_search --develop`, deretter
`--control`, og `-m research.tmg_multi_report`. Plan/kontroll overskrives ikke.
