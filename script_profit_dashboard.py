"""Egen, enkel nettside: teknisk indikator + ML-modell → forutsagt lønnsomhet."""
from __future__ import annotations
from pathlib import Path
import script_profit_ml as spm
import script_profit_predictor as predictor
import script_model_compare as compare

ROOT=Path(__file__).resolve().parent
DIRECTORY=ROOT/'reports/script_profit_website_2026-10-27'
INTRO='''# ML-vurdering av skript
Velg en **teknisk indikator** og en **ML-modell**. Modellen forutsier **sannsynligheten for at indikatoren blir lønnsom**
i valgt periode og **meravkastningen mot vanlig kjøp og hold** på BTCUSDT. Kapitalen oversetter anslaget til kroner.

Modellen trenes fortløpende fremover i tid (ikke statisk) og bare med fasit som var kjent da. Resultatet viser
treffsikkerhet, latency, throughput og feature importance; hele beregningen sikter mot 10–20
sekunder. Bruk det lagrede eksemplet uten nettverk, eller hent siste avsluttede UTC-døgn. Indikatoren simuleres
long/short ved 1×, 0,10 % per side og funding.
'''


def view(indicator='ut_bot',minutes='60',horizon='Neste uke',model=compare.RF,capital=10000,source=predictor.DEMO,
         cadence=compare.AUTO,window=compare.EXPANDING,feature_set='Utvidet · 32 egenskaper',compare_all=True,
         progress=None):
    """«Beregn med ML»: valgt modell forutsier lønnsomheten til valgt indikator (se ``script_model_compare.view``)."""
    return compare.view(indicator,minutes,horizon,model,capital,source,cadence,window,feature_set,compare_all,progress)


def deployment_files():
    """Alle pakkefiler, også hele de synlige CSV-datasettene."""
    import json
    manifest=json.loads((DIRECTORY/'manifest.json').read_text())
    return sorted(set([*manifest['runtime_sha256'],*manifest['dataset_sha256'],
                       'reports/script_profit_website_2026-10-27/manifest.json']))


def add_page(demo):
    import gradio as gr
    with demo.route('ML-vurdering av skript','ml-skript'):
        gr.Markdown(INTRO)
        with gr.Row():
            indicator=gr.Dropdown([(v,k)for k,v in spm.NAMES.items()],value='ut_bot',label='1. Teknisk indikator')
            minutes=gr.Dropdown([(v,k)for k,v in spm.TIMEFRAMES.items()],value='60',label='2. Tidsintervall på lysene')
            horizon=gr.Radio(list(spm.HORIZONS),value='Neste uke',label='3. Periode som ML skal anslå')
        with gr.Row():
            model=gr.Radio(compare.available_models(),value=compare.RF,label='4. ML-modell som forutsier lønnsomheten')
        with gr.Row():
            capital=gr.Number(value=10000,minimum=1,maximum=10000000,label='5. Kapital i kroner · regneeksempel ved 1×')
            source=gr.Radio(list(predictor.SOURCES),value=predictor.DEMO,label='6. Data til prediksjonen')
        with gr.Accordion('Avanserte ML-valg · trening, vindu og egenskaper',open=False):
            with gr.Row():
                cadence=gr.Radio(list(compare.CADENCES),value=compare.AUTO,label='Ny trening (walk-forward)')
                window=gr.Radio(list(compare.WINDOWS),value=compare.EXPANDING,label='Treningsvindu')
            with gr.Row():
                feature_set=gr.Radio(list(compare.FEATURE_SETS),value='Utvidet · 32 egenskaper',label='Egenskaper')
                compare_all=gr.Checkbox(value=True,label='Sammenlign med de andre modellene (samme data og trening)')
        run=gr.Button('Beregn med ML',variant='primary',size='lg')
        status=gr.Markdown('Velg indikator og ML-modell, og trykk «Beregn med ML». Det lagrede eksemplet virker uten nettverk.')
        predictions=gr.Dataframe(interactive=False,label='Prediksjon for valgt input')
        quality=gr.Dataframe(interactive=False,label='Treffsikkerhet på perioder modellen ikke hadde sett')
        speed=gr.Dataframe(interactive=False,label='Latency og throughput')
        drift=gr.Plot(label='Treffsikkerhet over tid og kalibrering')
        importance=gr.Plot(label='Feature importance')

        def run_view(indicator,minutes,horizon,model,capital,source,cadence,window,feature_set,compare_all,
                     progress=gr.Progress()):
            return view(indicator,minutes,horizon,model,capital,source,cadence,window,feature_set,compare_all,progress)

        outputs=[status,predictions,quality,speed,drift,importance]
        run.click(run_view,inputs=[indicator,minutes,horizon,model,capital,source,cadence,window,feature_set,compare_all],
                  outputs=outputs,api_name='ml_script_assess',concurrency_limit=1,concurrency_id='ml_compare')
        # Når valgene endres skal gamle resultater fjernes, så input aldri ser ut til å høre til forrige prognose.
        for control in (indicator,minutes,horizon,model,capital,source,cadence,window,feature_set,compare_all):
            control.change(lambda:('Valget er endret. Trykk «Beregn med ML» for en ny vurdering.',*compare.empty()),
                           outputs=outputs,api_visibility='private')
        return {'indicator':indicator,'minutes':minutes,'horizon':horizon,'model':model,'capital':capital,'source':source}
