# Videre videotest · spotstøtte og squeeze

Låst og evaluert 27. oktober 2026. Arbeidsmappen ble opprettet 26. oktober.

Ingen av åtte bestod alle utviklingskrav. OI-kandidaten beholdes. Ingen2025–26-kjøring.

Alle tall er Python-simulering ved 1×, etter kostnader og faktisk funding.
0,15 % per side, stress 0,30 % og dobbel funding. Ingen verifiserte TradingView-/WEEX-fyll.

## BTC · alle forhåndslåste kandidater

| Regel | 2021–23 netto | 2021–23 stress | Handler | 2024 netto | 2024 stress | Bestått |
|---|---:|---:|---:|---:|---:|---|
| Dagsfilter | +99.05% | +72.32% | 28 | +46.30% | +29.99% | referanse |
| OI-referanse | +120.51% | +95.27% | 19 | +65.44% | +50.31% | referanse |
| OI + sterk spotdeltakelse alle ukedager | +148.58% | +125.42% | 10 | +32.21% | +27.00% | nei |
| OI + ekte spotpress i samme retning · 4 t | +109.00% | +89.36% | 16 | +65.02% | +49.93% | nei |
| OI + ekte spotpress i samme retning · 24 t | +76.99% | +60.61% | 16 | +81.19% | +64.98% | nei |
| Vent på fallende OI, motsatt 4 t-pris og 1 t-vending | +29.12% | +24.48% | 12 | -16.40% | -19.08% | nei |
| OI-inngang; lukk ved OI-fall og kort prisvending | +13.71% | +7.40% | 19 | -7.13% | -9.20% | nei |
| OI-inngang; lukk ved motsatt spotpress på 4 og 24 t | +28.04% | +20.52% | 19 | -0.91% | -3.09% | nei |
| OI + close utenfor forrige 24-timersintervall | +11.69% | +7.32% | 5 | +41.14% | +36.63% | nei |
| Vent på retur innenfor 24 t-intervall med fallende OI | -2.48% | -3.08% | 3 | -2.25% | -2.54% | nei |

## Begrensninger

- Kjent, ofte gjenbrukt historikk. Tilleggsreglene bygger på tidligere valg.
- OI-arkivtid+65min er antakelse, ikke historisk publiseringsbevis.
- 1× Python-simulering, ikke TradingView-/WEEX-fyll eller likvidasjon.
- Ekte spot-takervolum finnes ikke som identisk serie i dagens Pine.
- Ingen kandidat kvalifisert: ingen ny placebo eller 2025–26-kjøring i dette forsøket.
- Tidligere Pine-kilder og aktiv motor beholdes. Nytt panel er historisk sammenligning.
- Alle åtte forsøk og feilede krav finnes i development.json; data-/kodehash i locked_plan.json.
- Videoenes OI-tolkning beviser ikke bestemte market maker-posisjoner eller likvidasjoner.

Kilder og eksakte egne regler: [SOURCE_NOTES.md](SOURCE_NOTES.md).
