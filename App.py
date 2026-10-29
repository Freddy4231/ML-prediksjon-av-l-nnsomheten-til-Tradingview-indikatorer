"""DAT158 · BTC-prognose (visningskopi): skriptenes signaler på WEEX-lys, papirhandel og ML-vurdering av skript.

Kjør lokalt:  python App.py   →   http://127.0.0.1:7862   (ML-siden: http://127.0.0.1:7862/ml-skript)

Hovedsiden har fanen «Automatisk handel · WEEX» med de samme feltene som serverversjonen: tre skript-felt med
sanntidsgrafer, Start/Stopp/Lukk, posisjoner, fyll og beslutningslogg. **Kopien kan ikke handle**: den har ingen
ordremotor, ingen signering og leser ingen WEEX-nøkler. Knappene driver papirhandel i minnet (papirhandel.py) på
offentlige WEEX-priser (weex_autotrade_dashboard.py). Undersiden «ML-vurdering av skript» lar en valgt ML-modell
forutsi lønnsomheten til en teknisk indikator (script_profit_dashboard.py). Ikke finansiell rådgivning.
"""
import os
from pathlib import Path

import gradio as gr

import script_profit_dashboard
import tmg_channel_dashboard
import tmg_context_dashboard
import tmg_daily_dashboard
import tmg_multi_dashboard
import tmg_position_dashboard
import tmg_research_dashboard
import tmg_squeeze_dashboard
import weex_autotrade_dashboard as weex

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7862"))   # 7860 er ofte tatt av en SSH-tunnel til serveren
APP_CSS = """
.gradio-container { max-width: 1520px !important; }
#intro { padding-bottom: 0.4rem; }
.skript-kort {
    padding: 1.15rem;
    border: 1.5px solid var(--border-color-primary);
    border-radius: 14px;
    background: var(--background-fill-secondary);
}
"""
SLOTS = (
    ("### 🟢 Skript-felt #1\n*Clean Wave Matrix PRO [Dark_Ace_Master]* — long ved BUY-merke og short ved SELL-merke, "
     "i det merket dukker opp (svingen er bekreftet).", "Clean Wave Matrix PRO (Dark_Ace_Master)", "BTC", ""),
    ("### 🟣 Skript-felt #2\n*SuperTrend STRATEGY [KivancOzbilgic]* — ATR(10), `hl2`, mult 3,0 (+464,1 % før 2025).",
     "SuperTrend STRATEGY (KivancOzbilgic)", "SOL", ""),
    ("### 🏆 Skript-felt #3 · Mest lønnsomme skript\n*Chandelier Exit (22, 3) [everget]* — **+174,8 %** (2020–24) og "
     "**+25,4 %** (2025–26) i long & short.", weex.CHANDELIER_LABEL, "ETH", " (Mest lønnsomme skript)"),
)
SCRIPT_INFO = (
    ("Pro Swing · nytt skript med SL og TP på WEEX", weex.PRO_SWING_LABEL, weex.PRO_SWING_TEXT, "pro_swing_recon.pine"),
    ("Kinetic Scalper · SL og to delmål på WEEX", weex.KINETIC_LABEL, weex.KINETIC_TEXT,
     "kinetic_scalper_reconstruction.pine"),
    ("Market Oracle Pro · estimert rekonstruksjon og WEEX-plan", weex.ORACLE_LABEL, weex.ORACLE_TEXT,
     "market_oracle_reconstruction.pine"),
    ("Bigscalp Premium · L/S, ADD og X på WEEX", weex.BIGSCALP_LABEL, weex.BIGSCALP_TEXT,
     "bigscalp_premium_reconstruction.pine"),
)


def _sync_coin(coins, current):
    """Grafen følger mynten i feltet: vis første valgte mynt hvis den viste ikke er valgt."""
    return current if not coins or current in coins else coins[0]


def _bigscalp_options(strategy_label):
    return gr.update(visible=strategy_label == weex.BIGSCALP_LABEL)


with gr.Blocks(title="Automatisk handel · WEEX og ML") as demo:
    gr.Markdown(
        "# ⚡ Automatisk handel · WEEX & BTC Maskinlæring\n"
        "**Sett opp skriptene dine i egne felt og test dem med papirhandel på WEEX-priser (standard 5 minutters "
        "intervall), eller la en ML-modell forutsi lønnsomheten til en teknisk indikator på undersiden.**\n\n"
        "[ML-vurdering av skript · velg indikator og ML-modell, og få en prognose](/ml-skript)",
        elem_id="intro",
    )
    with gr.Tabs(selected="ut-bot"):
        with gr.Tab("Automatisk handel · WEEX", id="ut-bot"):
            # --- Topplinje: bekreftelse + felles hurtigknapper (papirhandel, ingen ordre) ---
            with gr.Group(elem_classes="skript-kort"):
                gr.Markdown(weex.SAFETY_TEXT)
                with gr.Row():
                    slot_accept = gr.Checkbox(
                        value=True, scale=3,
                        label="Jeg forstår at dette er papirhandel: ingen ordre sendes, og ingen konto brukes",
                    )
                    weex_check = gr.Button("🔌 Kontroller WEEX-tilkobling", scale=1)
                with gr.Row():
                    start_all_btn = gr.Button("🚀 Start alle valgte skript-felt samtidig (Skript #1 + #2 + #3)",
                                              variant="primary")
                    stop_all_btn = gr.Button("⏹ Stopp alle skript (behold posisjoner)")
                    close_all_btn = gr.Button("🛑 Ta profit / lukk ALLE posisjoner nå og stopp", variant="stop")
                weex_status = gr.Markdown("**Papirhandel:** klar. Velg skript og mynter i feltene under.")

            for title, label, text, pine in SCRIPT_INFO:
                with gr.Accordion(title, open=False):
                    gr.Markdown(f"Velg **{label}** i ett av skript-feltene under.\n\n{text}")
                    gr.File(value=str(ROOT / "tradingview" / pine), label="Pine-kilde", interactive=False)

            with gr.Accordion("TMG Trades · Stonksy + VRVP POC & Liquidity Trap (@tmgtrades)", open=False):
                gr.Markdown(f"Velg **{weex.TMG_LABEL}** i Skript #1, #2 eller #3. Se testresultatene før valg av "
                            f"tidsintervall.\n\n{weex.TMG_TEXT}")
                gr.File(value=str(ROOT / "tradingview/tmg_trades_strategy.pine"),
                        label="TMG Trades · Pine v6-strategi", interactive=False)
                tmg_review_text, tmg_review_chart, tmg_review_rows = tmg_research_dashboard.view("BTCUSDT")
                tmg_review_coin = gr.Dropdown(list(tmg_research_dashboard.SYMBOLS), value="BTCUSDT",
                                              label="TMG-test · velg marked")
                tmg_review_status = gr.Markdown(tmg_review_text)
                tmg_review_figure = gr.Plot(value=tmg_review_chart, label="TMG · historisk simulert kapital")
                tmg_review_table = gr.Dataframe(value=tmg_review_rows, label="TMG · netto, kostnadsstress og største fall",
                                                interactive=False)
                tmg_review_coin.change(fn=tmg_research_dashboard.view, inputs=[tmg_review_coin],
                                       outputs=[tmg_review_status, tmg_review_figure, tmg_review_table])
                gr.File(value=str(ROOT / "reports/tmg_improvement_2026-10-26/README.md"),
                        label="TMG · metode og resultater fra 120 forsøk", interactive=False)
                with gr.Accordion("Nye funn fra TMG-kanalen · ekte futuresdata og helgevolum", open=False):
                    tmg_channel_text, tmg_channel_chart, tmg_channel_rows = tmg_channel_dashboard.view("BTCUSDT")
                    tmg_channel_coin = gr.Dropdown(list(tmg_channel_dashboard.SYMBOLS), value="BTCUSDT",
                                                   label="TMG-kanalen · ekte futuresdata")
                    tmg_channel_status = gr.Markdown(tmg_channel_text)
                    tmg_channel_figure = gr.Plot(value=tmg_channel_chart,
                                                 label="Kanalfunn · kapital og filtrerte signaler")
                    tmg_channel_table = gr.Dataframe(value=tmg_channel_rows, label="Samme tidsrom, mynt og retning",
                                                     interactive=False)
                    tmg_channel_coin.change(fn=tmg_channel_dashboard.view, inputs=[tmg_channel_coin],
                                            outputs=[tmg_channel_status, tmg_channel_figure, tmg_channel_table])
                    gr.File(value=[str(ROOT / "tradingview/tmg_spot_participation.pine"),
                                   str(ROOT / "reports/tmg_channel_2026-10-26/SOURCE_NOTES.md"),
                                   str(ROOT / "reports/tmg_channel_2026-10-26/README.md")],
                            label="TradingView-volumpanel uten ordre · kilder og metode", file_count="multiple",
                            interactive=False)

            with gr.Accordion("Ny indikator · TMG Dagsbekreftelse (kjøp/selg · 1 time)", open=False):
                (tmg_daily_text, tmg_daily_chart, tmg_daily_rows, tmg_daily_metrics,
                 tmg_daily_equity) = tmg_daily_dashboard.historical()
                with gr.Row():
                    tmg_daily_coin = gr.Dropdown(list(tmg_daily_dashboard.SYMBOLS), value="BTCUSDT",
                                                 label="TMG Dagsbekreftelse · marked")
                    tmg_daily_window = gr.Dropdown(list(tmg_daily_dashboard.WINDOWS),
                                                   value=tmg_daily_dashboard.WINDOWS[0], label="Historisk signalvindu")
                tmg_daily_variant = gr.Dropdown(list(tmg_context_dashboard.VARIANTS),
                                                value=tmg_context_dashboard.VARIANTS[0],
                                                label="Variant av indikatoren · gevinstvern er et forskningsalternativ")
                tmg_daily_status = gr.Markdown(tmg_daily_text)
                with gr.Row():
                    tmg_daily_history = gr.Button("Vis historiske signaler")
                    tmg_daily_refresh = gr.Button("Hent siste markedsdata · ingen ordre")
                tmg_daily_figure = gr.Plot(value=tmg_daily_chart, label="Ny indikator · kjøp, salg, lukk og dagsnivåer")
                tmg_daily_signals = gr.Dataframe(value=tmg_daily_rows, interactive=False,
                                                 label="Siste signaler i grafvinduet · tidspunkt ved close")
                gr.Markdown("**Historisk kostnadstest · januar 2025–september 2026.** Tabellen og kapitalgrafen gjelder "
                            "denne perioden også når siste markedsdata vises over.")
                tmg_daily_table = gr.Dataframe(value=tmg_daily_metrics, interactive=False,
                                               label="Historisk simulering · 1× og faktisk funding")
                tmg_daily_capital = gr.Plot(value=tmg_daily_equity, label="Historisk simulert kapital")
                tmg_daily_outputs = [tmg_daily_status, tmg_daily_figure, tmg_daily_signals, tmg_daily_table,
                                     tmg_daily_capital]
                tmg_daily_inputs = [tmg_daily_coin, tmg_daily_window, tmg_daily_variant]
                for control in (tmg_daily_coin, tmg_daily_window, tmg_daily_variant):
                    control.change(fn=tmg_context_dashboard.historical, inputs=tmg_daily_inputs,
                                   outputs=tmg_daily_outputs)
                tmg_daily_history.click(fn=tmg_context_dashboard.historical, inputs=tmg_daily_inputs,
                                        outputs=tmg_daily_outputs)
                tmg_daily_refresh.click(fn=tmg_context_dashboard.current, inputs=tmg_daily_inputs,
                                        outputs=[tmg_daily_status, tmg_daily_figure, tmg_daily_signals],
                                        concurrency_limit=1)
                gr.File(value=[str(ROOT / "tradingview/tmg_daily_confirmation.pine"),
                               str(ROOT / "reports/tmg_daily_2026-10-26/README.md"),
                               str(ROOT / "reports/tmg_daily_2026-10-26/SOURCE_NOTES.md")],
                        label="TMG Dagsbekreftelse · Pine-indikator og testgrunnlag", file_count="multiple",
                        interactive=False)
                with gr.Accordion("Nyeste kanalfunn · volatilitet, dags-RSI og gevinstvern", open=False):
                    tmg_context_text, tmg_context_rows = tmg_context_dashboard.findings()
                    tmg_context_status = gr.Markdown(tmg_context_text)
                    tmg_context_table = gr.Dataframe(
                        value=tmg_context_rows, interactive=False,
                        label="11 låste forsøk · bare utvikling og 2024, ingen rangering på kontrollen")
                    tmg_daily_coin.change(fn=tmg_context_dashboard.findings, inputs=[tmg_daily_coin],
                                          outputs=[tmg_context_status, tmg_context_table])
                gr.File(value=[str(ROOT / "tradingview/tmg_daily_profit_guard.pine"),
                               str(ROOT / "reports/tmg_context_2026-10-26/README.md"),
                               str(ROOT / "reports/tmg_context_2026-10-26/SOURCE_NOTES.md")],
                        label="Valgfritt gevinstvern · Pine-indikator, nye kilder og hele kostnadstesten",
                        file_count="multiple", interactive=False)
                with gr.Accordion("Flere videofunn · ekte spotpress, Pine-proxy og funding", open=False):
                    tmg_multi_window = gr.Dropdown(list(tmg_multi_dashboard.WINDOWS),
                                                   value=tmg_multi_dashboard.WINDOWS[0],
                                                   label="Historisk kontekst · 2024, ikke sanntid")
                    tmg_multi_text, tmg_multi_rows, tmg_multi_chart, tmg_multi_signals = tmg_multi_dashboard.view()
                    tmg_multi_status = gr.Markdown(tmg_multi_text)
                    tmg_multi_table = gr.Dataframe(value=tmg_multi_rows, interactive=False,
                                                   label="Åtte nye tester og dagens referanse · alle avvisningsgrunner")
                    tmg_multi_figure = gr.Plot(value=tmg_multi_chart,
                                               label="Historiske signaler med separat ekte spotpress, CLV og funding")
                    tmg_multi_events = gr.Dataframe(value=tmg_multi_signals, interactive=False,
                                                    label="Eksisterende kjøp/selg · kjent kontekst ved signalclose")
                    tmg_multi_outputs = [tmg_multi_status, tmg_multi_table, tmg_multi_figure, tmg_multi_events]
                    for control in (tmg_daily_coin, tmg_multi_window):
                        control.change(fn=tmg_multi_dashboard.view, inputs=[tmg_daily_coin, tmg_multi_window],
                                       outputs=tmg_multi_outputs)
                    gr.File(value=[str(ROOT / "reports/tmg_multi_2026-10-26/README.md"),
                                   str(ROOT / "reports/tmg_multi_2026-10-26/SOURCE_NOTES.md")],
                            label="Nye videokilder og hele metoden", file_count="multiple", interactive=False)
                with gr.Accordion("Ny forskningsindikator · OI, ukespress og bekreftet inngang", open=False) \
                        as tmg_position_review:
                    tmg_position_window = gr.Dropdown(list(tmg_position_dashboard.WINDOWS),
                                                      value=tmg_position_dashboard.WINDOWS[0],
                                                      label="Historiske kandidatsignaler · 2024, ikke sanntid")
                    tmg_position_status = gr.Markdown("Åpne panelet for den nye OI-kandidaten, testresultater og "
                                                      "kjøp/selg. Ingen ordre.")
                    tmg_position_table = gr.Dataframe(interactive=False,
                                                      label="Tolv låste regler og dagens referanse · kostnader inkludert")
                    tmg_position_figure = gr.Plot(label="Dagens signaler og OI-kandidat · historisk 2024")
                    tmg_position_events = gr.Dataframe(interactive=False, label="Kjøp/selg og OI-observasjonens tidspunkt")
                    tmg_position_inputs = [tmg_daily_coin, tmg_position_window]
                    tmg_position_outputs = [tmg_position_status, tmg_position_table, tmg_position_figure,
                                            tmg_position_events]
                    tmg_position_review.expand(fn=tmg_position_dashboard.view, inputs=tmg_position_inputs,
                                               outputs=tmg_position_outputs)
                    for control in (tmg_position_window, tmg_daily_coin):
                        control.change(fn=tmg_position_dashboard.view, inputs=tmg_position_inputs,
                                       outputs=tmg_position_outputs)
                    gr.File(value=[str(ROOT / "tradingview/tmg_oi_confirmation.pine"),
                                   str(ROOT / "tradingview/TMG_OI_README.md"),
                                   str(ROOT / "reports/tmg_oi_2026-10-26/README.md"),
                                   str(ROOT / "reports/tmg_oi_2026-10-26/SOURCE_NOTES.md")],
                            label="TMG OI · Pine-prototype for TradingView · separat feed, ikke koblet til handel",
                            file_count="multiple", interactive=False)
                with gr.Accordion("Videre videotest · spotstøtte og squeeze", open=False) as tmg_squeeze_review:
                    tmg_squeeze_rule = gr.Dropdown(tmg_squeeze_dashboard.CHOICES, value="oi",
                                                   label="Historisk regel å undersøke · aktiverer ingen handler")
                    tmg_squeeze_status = gr.Markdown("Åpne for å se åtte nye videoregler testet mot OI-kandidaten på "
                                                     "ekte historikk.")
                    tmg_squeeze_table = gr.Dataframe(interactive=False,
                                                     label="Alle åtte regler og to referanser · kostnader inkludert")
                    tmg_squeeze_figure = gr.Plot(label="Historiske kjøp/selg/lukk og netto kapital · 2024")
                    tmg_squeeze_events = gr.Dataframe(interactive=False, label="Signal ved close og observert spotpress/OI")
                    tmg_squeeze_inputs = [tmg_daily_coin, tmg_squeeze_rule]
                    tmg_squeeze_outputs = [tmg_squeeze_status, tmg_squeeze_table, tmg_squeeze_figure,
                                           tmg_squeeze_events]
                    tmg_squeeze_review.expand(fn=tmg_squeeze_dashboard.view, inputs=tmg_squeeze_inputs,
                                              outputs=tmg_squeeze_outputs)
                    for control in (tmg_squeeze_rule, tmg_daily_coin):
                        control.change(fn=tmg_squeeze_dashboard.view, inputs=tmg_squeeze_inputs,
                                       outputs=tmg_squeeze_outputs)
                    gr.File(value=[str(ROOT / "reports/tmg_squeeze_2026-10-26/README.md"),
                                   str(ROOT / "reports/tmg_squeeze_2026-10-26/SOURCE_NOTES.md")],
                            label="Nye videofunn · regler og full sammenligning", file_count="multiple",
                            interactive=False)

            # --- Skript-feltene side om side ---
            slots = []
            with gr.Row():
                for number, (heading, script, coin, suffix) in enumerate(SLOTS, start=1):
                    with gr.Column(scale=1, elem_classes="skript-kort"):
                        gr.Markdown(heading)
                        strategy = gr.Dropdown(list(weex.SLOT_STRATEGIES), value=script,
                                               label=f"Skript i Felt #{number}{suffix}")
                        coins = gr.CheckboxGroup(list(weex.COINS), value=[coin],
                                                 label=f"Kryptovalutaer for Skript #{number} (15 tokens)")
                        with gr.Row():
                            timeframe = gr.Dropdown(list(weex.TIMEFRAMES), value="5 minutter", label="Tidsintervall")
                            direction = gr.Dropdown(list(weex.DIRECTIONS), value="Long og short (som UT Bot)",
                                                    label="Retning (Long ↔ Short)")
                        with gr.Row():
                            margin = gr.Number(value=10, minimum=0.01, maximum=10, label="Margin (% av USDT · maks 10)")
                            leverage = gr.Number(value=5, minimum=1, maximum=50, precision=0, label="Gearing (1–50×)")
                        with gr.Accordion("Bigscalp · valgfri TP/SL", open=False,
                                          visible=script == weex.BIGSCALP_LABEL) as bigscalp_options:
                            gr.Markdown("Gjelder Bigscalp ved neste start. 0 = av, som originalens standard. "
                                        "Nødstoppen beholdes.")
                            with gr.Row():
                                bigscalp_tp = gr.Number(value=0, minimum=0, maximum=100, step=0.1,
                                                        label="Bigscalp TP (%) · 0 = av")
                                bigscalp_sl = gr.Number(value=0, minimum=0, maximum=99.9, step=0.1,
                                                        label="Bigscalp SL (%) · 0 = av")
                        with gr.Row():
                            start = gr.Button(f"▶ Start Skript #{number}", variant="primary")
                            stop = gr.Button(f"⏹ Stopp #{number}")
                            close = gr.Button(f"✖ Lukk #{number}", variant="stop")
                        message = gr.Markdown(f"Klar til oppstart for Skript #{number}.")
                        with gr.Row():
                            chart_coin = gr.Dropdown(list(weex.COINS), value=coin,
                                                     label="Vis graf for kryptovaluta (alle 15 tokens)")
                            refresh = gr.Button("🔄 Oppdater")
                        chart_text = gr.Markdown("Leser 5-minutterslys …")
                        chart_plot = gr.Plot(label=f"Sanntidsgraf · Skript #{number}")
                        chart_table = gr.Dataframe(interactive=False, visible=False)
                    slots.append({"slot": f"#{number}", "strategy": strategy, "coins": coins, "timeframe": timeframe,
                                  "direction": direction, "margin": margin, "leverage": leverage,
                                  "bigscalp_options": bigscalp_options, "bigscalp_tp": bigscalp_tp,
                                  "bigscalp_sl": bigscalp_sl, "start": start, "stop": stop, "close": close,
                                  "message": message, "chart_coin": chart_coin, "refresh": refresh,
                                  "chart": [chart_text, chart_plot, chart_table]})

            # --- Papirhandel: posisjoner, simulerte fyll og beslutningslogg ---
            with gr.Group(elem_classes="skript-kort"):
                with gr.Row():
                    gr.Markdown("### 💼 Papirhandel: posisjoner, utførte handler (simulerte fyll) og beslutningslogg")
                    auto_refresh = gr.Button("🔄 Oppdater papirhandel nå", scale=0)
                auto_status = gr.Markdown("Henter status …")
                auto_states = gr.Dataframe(interactive=False, wrap=True, label="Posisjon og neste steg per mynt")
                auto_fills = gr.Dataframe(interactive=False, label="Utførte handler (papirfyll, simulert)")
                auto_decisions = gr.Dataframe(interactive=False, wrap=True, label="Motorens beslutninger per bar (papir)")
                auto_timer = gr.Timer(30)

    # Grafer og papirhandel hentes når siden åpnes og når et valg endres. Bare offentlige markedsdata, ingen ordre.
    chart_event = {"api_visibility": "private", "concurrency_limit": 3, "concurrency_id": "weex_graf"}
    paper_event = {"api_visibility": "private", "concurrency_limit": 1, "concurrency_id": "papirhandel"}
    paper_outputs = [auto_status, auto_states, auto_fills, auto_decisions]
    demo.load(weex.autotrade_view, outputs=paper_outputs, **paper_event)
    auto_refresh.click(weex.autotrade_view, outputs=paper_outputs, **paper_event)
    auto_timer.tick(weex.autotrade_view, outputs=paper_outputs, **paper_event)
    weex_check.click(weex.check_public_data, outputs=[weex_status], **chart_event)

    def chart_inputs(slot):
        return [slot["chart_coin"], slot["strategy"], slot["timeframe"]]

    for slot in slots:
        inputs = chart_inputs(slot)
        demo.load(weex.coin_chart_view, inputs=inputs, outputs=slot["chart"], **chart_event)
        slot["refresh"].click(weex.coin_chart_view, inputs=inputs, outputs=slot["chart"], **chart_event)
        for control in inputs:
            control.change(weex.coin_chart_view, inputs=inputs, outputs=slot["chart"], **chart_event)
        slot["coins"].change(_sync_coin, inputs=[slot["coins"], slot["chart_coin"]], outputs=[slot["chart_coin"]],
                             api_visibility="private")
        slot["strategy"].change(_bigscalp_options, inputs=[slot["strategy"]], outputs=[slot["bigscalp_options"]],
                                api_visibility="private")
        settings = [slot["strategy"], slot["timeframe"], slot["coins"], slot["direction"], slot["margin"],
                    slot["leverage"], slot_accept, slot["bigscalp_tp"], slot["bigscalp_sl"]]

        def start_one(*values, name=slot["slot"]):
            return weex.start_slot(name, *values)

        def stop_one(name=slot["slot"]):
            return weex.stop_slot(name)

        def close_one(name=slot["slot"]):
            return weex.close_slot(name)

        slot["start"].click(start_one, inputs=settings, outputs=[slot["message"]], **paper_event) \
            .then(weex.autotrade_view, outputs=paper_outputs, **paper_event) \
            .then(weex.coin_chart_view, inputs=inputs, outputs=slot["chart"], **chart_event)
        slot["stop"].click(stop_one, outputs=[slot["message"]], **paper_event) \
            .then(weex.autotrade_view, outputs=paper_outputs, **paper_event)
        slot["close"].click(close_one, outputs=[slot["message"]], **paper_event) \
            .then(weex.autotrade_view, outputs=paper_outputs, **paper_event) \
            .then(weex.coin_chart_view, inputs=inputs, outputs=slot["chart"], **chart_event)

    all_settings = [slot_accept] + [component for slot in slots for component in
                                    (slot["strategy"], slot["timeframe"], slot["coins"], slot["direction"],
                                     slot["margin"], slot["leverage"], slot["bigscalp_tp"], slot["bigscalp_sl"])]
    start_all = start_all_btn.click(weex.start_slots, inputs=all_settings, outputs=[weex_status], **paper_event) \
        .then(weex.autotrade_view, outputs=paper_outputs, **paper_event)
    stop_all = stop_all_btn.click(weex.stop_all, outputs=[weex_status], **paper_event) \
        .then(weex.autotrade_view, outputs=paper_outputs, **paper_event)
    close_all = close_all_btn.click(weex.close_all, outputs=[weex_status], **paper_event) \
        .then(weex.autotrade_view, outputs=paper_outputs, **paper_event)
    for event in (start_all, close_all):
        for slot in slots:
            event.then(weex.coin_chart_view, inputs=chart_inputs(slot), outputs=slot["chart"], **chart_event)

if demo.pages and demo.pages[0] == ("", "Home", True):
    demo.pages[0] = ("", "Automatisk handel · WEEX", True)

# Egen underside /ml-skript: valgt ML-modell forutsier lønnsomheten til valgt teknisk indikator.
script_profit_inputs = script_profit_dashboard.add_page(demo)


if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), server_port=PORT,
                theme=gr.themes.Soft(), css=APP_CSS)
