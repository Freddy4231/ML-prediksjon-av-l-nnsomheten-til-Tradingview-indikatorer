# ML-prediksjon av lønnsomheten til TradingView-indikatorer

ML assignment 2 for Fredrik, Gabriel og Lloyd. Innleveringsrapporten ligger i Mal DAT158 innlevering 2.docx ovenfor README.md.

En Gradio-app bestående av 2 element: 

**Automatisk handel · WEEX (papirhandel)** – tre skript-felt med tekniske indikatorer (Clean Wave Matrix PRO,
SuperTrend, Chandelier Exit, TMG Trades, Pro Swing, Kinetic Scalper, Market Oracle og Bigscalp) på offentlige
WEEX-lys for 15 kryptovalutaer. Hvert felt har sanntidsgraf, valg av mynter, intervall, retning, margin og
gearing, og knappene Start, Stopp og Lukk. Under feltene vises posisjoner, utførte handler og en beslutningslogg.
TMG-indikatorene TMG Trades og TMG Dagsbekreftelse har egne nedtrekk med Pine-kilde, signaler og historisk kostnadstest.

Papirhandelen er en forenklet etterligning av ordremotoren. Den beslutter ved lysets lukking og fyller til
sluttkursen med 0,08 % gebyr per side. Nødstoppen utløses når 90 % av marginen er tapt. Clean Wave, SuperTrend,
Chandelier og TMG blir med i skriptets posisjon ved start, mens de andre skriptene venter på et nytt signal.
Skriptenes egne SL/TP følges via målposisjonen; børsordre, ADD og delvise TP simuleres ikke.

**ML-vurdering av skript** (`/ml-skript`) – velg en teknisk indikator og en ML-modell (logistisk regresjon/Ridge,
Random Forest eller gradient boosting). Modellen forutsier sannsynligheten for at indikatoren blir lønnsom neste
dag, uke eller måned, og meravkastningen mot vanlig kjøp og hold. Modellen trenes fortløpende fremover i tid
(walk-forward), og siden viser treffsikkerhet, latency, throughput og feature importance.
**Nederst på ML-siden ligger Pine-testen:** lim inn et hvilket som helst Pine Script fra TradingView, velg marked
og periode, og se om det tjente penger etter gebyr på 1–60 minutters lys. Maskinlæringsdelen forutsier om hver
handel skriptet gjør, blir lønnsom (logistisk regresjon, Random Forest og gradient boosting).





