"""Pakk ekte modellinput og synlige datasett uten å endre eller trene gamle modeller."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import numpy as np
import script_profit_ml as spm
import script_profit_predictor as predictor
from research import script_ml_check as training

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,doc):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


OUTPUT=ROOT/'reports/script_profit_website_2026-10-27'
RESULTS=ROOT/'reports/script_ml_check/results.json'
# Kode som siden kjører. Modell, resultater og demonstrasjonsinput legges til i build/refresh_runtime.
RUNTIME=['script_profit_predictor.py','script_profit_dashboard.py','script_profit_ml.py','script_model_compare.py',
         'features.py','clean_wave_weex.py','kivanc_supertrend.py','triple_supertrend.py','ut_bot_signals.py',
         'research/clean_wave.py','research/futures_funding.py','reports/script_ml_check/results.json',
         'research/script_ml_check.py','research/script_profit_website.py',
         'reports/script_profit_website_2026-10-27/demo_inputs.json',
         'reports/script_profit_website_2026-10-27/README.md']
FROZEN=('reports/script_ml_check/results.json','reports/script_ml_check/models.joblib',
        'reports/script_profit_website_2026-10-27/demo_inputs.json',
        'reports/script_profit_website_2026-10-27/demonstration.json')


def build():
    results=json.loads(RESULTS.read_text())
    for path,expected in results['sources'].items():assert sha(ROOT/path)==expected,path
    assert sha(ROOT/results['model_file']['path'])==results['model_file']['sha256']
    funding=training.load_funding()
    assert funding is not None,'Datapakken krever samme faktiske funding som treningen'
    funding_path=OUTPUT/'BTCUSDT_funding.csv'
    funding.to_frame('funding_rate').to_csv(funding_path,index_label='funding_time')
    snapshot={'features':spm.ALL_COLUMNS,'timeframes':{}}
    datasets={};paths=[funding_path]
    for minutes in spm.TIMEFRAMES:
        _,bars,rates=training.load(minutes,funding)
        features=spm.make_features(bars)
        positions=spm.positions(bars,minutes)
        bar_path=OUTPUT/f'BTCUSDT_{minutes}_bars.csv'
        bars.to_csv(bar_path,index_label='open_time');paths.append(bar_path)
        datasets[minutes]={'bars':len(bars),'filled_bars':int(bars.filled.sum()),
                           'first':bars.index[0].isoformat(),'last':bars.index[-1].isoformat()}
        snapshot['timeframes'][minutes]={'scripts':{}}
        for key,position in positions.items():
            eq=spm.equity(spm.bar_returns(bars,position))
            last=len(bars)-1
            decision=bars.index[last]+pd.Timedelta(minutes=int(minutes))
            assert decision==decision.normalize()
            vector=spm.sample_features(features,position,eq,[last],minutes,bars.close.to_numpy()).iloc[0]
            assert np.isfinite(vector.to_numpy(float)).all()
            snapshot['timeframes'][minutes]['scripts'][key]={
                'values':vector.tolist(),'position':int(position[last]),'decision_time':decision.isoformat(),'bar_count':len(bars)}
            examples=None
            for horizon,hours in spm.HORIZONS.items():
                x,y,known,clean=training.dataset(bars,features,position,rates,hours,minutes)
                h=results['timeframes'][minutes]['scripts'][key]['horizons'][horizon]
                assert len(y)==h['samples'],(minutes,key,horizon)
                assert np.isclose(np.mean(y),h['mean_return'],atol=1e-12,rtol=1e-12),(minutes,key,horizon,'Fasitgrunnlag endret')
                assert np.isclose(np.mean(y>0),h['profitable_share'],atol=1e-12,rtol=1e-12)
                assert np.isclose(np.mean(clean),h['clean_share'],atol=1e-12,rtol=1e-12)
                if examples is None:examples=x.copy()
                selected=clean if h['chosen']['cleaning']=='på'else np.ones(len(x),dtype=bool)
                hz={'Neste dag':'day','Neste uke':'week','Neste måned':'month'}[horizon]
                examples[f'return_{hz}']=pd.Series(y,index=x.index)
                examples[f'profit_{hz}']=pd.Series((y>0).astype(int),index=x.index)
                examples[f'known_{hz}']=pd.Series([t.isoformat()for t in known],index=x.index)
                examples[f'clean_{hz}']=pd.Series(clean,index=x.index)
                examples[f'used_training_{hz}']=False
                examples.loc[x.index,f'used_training_{hz}']=selected
            examples.insert(0,'decision_time',(examples.index+pd.Timedelta(minutes=int(minutes))).map(lambda t:t.isoformat()))
            path=OUTPUT/f'{minutes}_{key}_examples.csv'
            examples.to_csv(path,index_label='open_time');paths.append(path)
        print(minutes,'prisrader',len(bars),'ML-eksempler eksportert',flush=True)
    write(OUTPUT/'demo_inputs.json',snapshot)
    runtime=[*RUNTIME,results['model_file']['path']]
    source_paths=['data/btcusdt_latest_1h.csv','data/30m/btcusdt_30m.csv',
                  str(training.futures_funding.path_for('BTCUSDT'))]
    write(OUTPUT/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
          'model_path':results['model_file']['path'],'results_path':'reports/script_ml_check/results.json',
          'runtime_sha256':{p:sha(ROOT/p)for p in runtime},
          'dataset_sha256':{str(p.relative_to(ROOT)):sha(p)for p in paths},'datasets':datasets,
          'input_files_sha256':{p:sha(ROOT/p)for p in source_paths},
          'funding_file':'reports/script_profit_website_2026-10-27/BTCUSDT_funding.csv',
          'training':'Eksisterende modeller; ingen ny modelltilpasning eller valg. Full historikk brukes som i originaltreningen.',
          'demo':'Ekte historiske egenskaper ved11.okt2026, kjøres gjennom ekte estimators ved klikk. Ingen ny måling av presisjon.',
          'live':'Siste avsluttede UTC-døgn, ekte spotlys uten hull. Forkortet oppvarming kan endre indikatorens tilstand.'})
    demonstrations=[]
    for indicator,minutes,horizon in [('ut_bot','60','Neste uke'),('chandelier','240','Neste måned'),('kivanc_st','60','Neste dag')]:
        result=predictor.assess(indicator,minutes,horizon,10000)
        demonstrations.append({k:result[k]for k in ('indicator','name','minutes','horizon','capital','source',
            'decision_time','p_profit','expected_return','expected_pnl_nok','classifier','regressor','feature_count','model_sha256')})
    proof=OUTPUT/'demonstration.json'
    write(proof,{'method':'Faktiske predict_proba/predict-kall mot lagrede modeller; ingen konstruerte ML-svar.',
                 'examples':demonstrations})
    manifest=json.loads((OUTPUT/'manifest.json').read_text())
    manifest['runtime_sha256'][str(proof.relative_to(ROOT))]=sha(proof)
    write(OUTPUT/'manifest.json',manifest)


def refresh_runtime():
    """Etter en kodeendring på siden: oppdater bare kodehashene. Datasett, modell og demo-input må være uendret."""
    path=OUTPUT/'manifest.json'
    manifest=json.loads(path.read_text())
    for name,expected in manifest['dataset_sha256'].items():
        assert sha(ROOT/name)==expected,f'Datasett endret: {name}'
    for name in FROZEN:
        assert sha(ROOT/name)==manifest['runtime_sha256'][name],f'Fryst artefakt endret: {name}'
    manifest['runtime_sha256']={p:sha(ROOT/p)for p in [*RUNTIME,*FROZEN]}
    manifest['runtime_updated_utc']=datetime.now(timezone.utc).isoformat()
    manifest['comparison']=('Modellsammenligningen trener fortløpende (walk-forward) ved klikk på de samme '
                            'datasettene; lagrede modeller og demonstrasjonsinput er uendret.')
    write(path,manifest)
    return manifest


if __name__=='__main__':
    import sys
    refresh_runtime()if '--refresh-runtime'in sys.argv else build()
