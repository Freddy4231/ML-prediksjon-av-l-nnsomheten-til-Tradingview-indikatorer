"""Brukerinput → ekte lagret ML-modell → vurdering av simulert skriptavkastning.

Kun BTCUSDT og fem ferdigtrente indikatorer. Ingen trening eller ordre ved serving.
Døgnets siste close brukes, som i treningsdataenes daglige eksempler.
"""
from __future__ import annotations
import hashlib
import json
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import script_profit_ml as spm

ROOT=Path(__file__).resolve().parent
DIRECTORY=ROOT/'reports/script_profit_website_2026-10-27'
DEMO='Lagret eksempel · 11. oktober 2026'
LIVE='Ferske BTC-data · siste UTC-døgn'
SOURCES=(DEMO,LIVE)
INTERVALS={'30':'30m','60':'1h','240':'4h','1440':'1d'}
URL='https://api.binance.com/api/v3/klines'
_CACHE={}
_LOCK=threading.Lock()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    manifest=json.loads((DIRECTORY/'manifest.json').read_text())
    # Små servingfiler kontrolleres hver gang. Store datasett kontrolleres ved lesing/nedlasting.
    for name,expected in manifest['runtime_sha256'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError('Modell, funksjoner eller datakontrakt er endret. Vurderingen er skjult.')
    results=json.loads((ROOT/manifest['results_path']).read_text())
    if results['dataset']['features']!=spm.ALL_COLUMNS:
        raise ValueError('Modellens egenskaper har feil rekkefølge.')
    return manifest,results


def bundle(manifest):
    import joblib
    name=manifest['model_path']; expected=manifest['runtime_sha256'][name]
    with _LOCK:
        if _CACHE.get('sha')!=expected:
            model=joblib.load(ROOT/name)
            if model['features']!=spm.ALL_COLUMNS:
                raise ValueError('Modellfilens egenskaper samsvarer ikke.')
            _CACHE.clear();_CACHE.update(sha=expected,model=model)
        return _CACHE['model']


def validate_bars(bars,minutes):
    if not isinstance(bars.index,pd.DatetimeIndex) or bars.index.tz is None or bars.empty:
        raise ValueError('Prisdata må ha UTC-tid og minst én rad.')
    bars=bars.copy();bars.index=bars.index.tz_convert('UTC')
    if bars.index.has_duplicates or not bars.index.is_monotonic_increasing:
        raise ValueError('Dupliserte eller usorterte prisdata.')
    step=pd.Timedelta(minutes=int(minutes))
    if (bars.index.as_unit('ms').asi8 % (int(minutes)*60000)).any():
        raise ValueError('Prisdata er ikke på riktig UTC-intervall.')
    if len(bars)>1 and not (bars.index.to_series().diff().dropna()==step).all():
        raise ValueError('Prisdata har hull. Ingen kunstig utfylling brukes i ny prediksjon.')
    x=bars[['open','high','low','close','volume']].to_numpy(float)
    if not np.isfinite(x).all() or (x[:,:4]<=0).any() or (x[:,4]<0).any():
        raise ValueError('Prisdata har ugyldige tall.')
    if ((bars.high<bars[['open','low','close']].max(axis=1))|
        (bars.low>bars[['open','high','close']].min(axis=1))).any():
        raise ValueError('Prisdata har ugyldig OHLC-spenn.')
    return bars


def fetch_bars(minutes,session=None,now=None,count=2000):
    """Bare offentlige spotlys. Fast slutt ved siste UTC-døgn, høyst to sider."""
    import requests
    if minutes not in INTERVALS or not 1<=count<=2000:
        raise ValueError('Ukjent intervall eller for mange lys.')
    now=pd.Timestamp.now(tz='UTC')if now is None else pd.Timestamp(now)
    if now.tzinfo is None:raise ValueError('Klokken må ha tidssone.')
    boundary=now.tz_convert('UTC').normalize()
    end=int(boundary.timestamp()*1000)-1
    step=int(minutes)*60000
    session=session or requests.Session()
    seen={}
    for _ in range(2):
        response=session.get(URL,params={'symbol':'BTCUSDT','interval':INTERVALS[minutes],
                             'endTime':end,'limit':min(1000,count-len(seen))},timeout=(3,12))
        response.raise_for_status();page=response.json()
        if not isinstance(page,list)or not page:raise ValueError('Tomt svar fra datakilden.')
        first=None
        for row in page:
            if not isinstance(row,list)or len(row)<7:raise ValueError('Ugyldig lysformat.')
            opening=int(row[0]);closing=int(row[6])
            if opening%step or closing!=opening+step-1 or closing>end:
                raise ValueError('Uavsluttet lys eller feil tidsstempel fra datakilden.')
            item=tuple(map(float,row[1:6]))
            if opening in seen and seen[opening]!=item:raise ValueError('Motstridende duplikater fra datakilden.')
            seen[opening]=item
            first=opening if first is None else min(first,opening)
        if len(seen)>=count:break
        if first is None or first-1>=end:raise ValueError('Datakilden går ikke bakover i tid.')
        end=first-1
    if len(seen)<count:raise ValueError('For lite historikk fra datakilden.')
    times=sorted(seen)[-count:]
    frame=pd.DataFrame([seen[t]for t in times],columns=['open','high','low','close','volume'],
                       index=pd.to_datetime(times,unit='ms',utc=True))
    frame=validate_bars(frame,minutes)
    if frame.index[-1]+pd.Timedelta(minutes=int(minutes))!=boundary:
        raise ValueError('Datakilden mangler siste avsluttede UTC-døgn.')
    return frame


def live_input(minutes,key,bars=None):
    bars=fetch_bars(minutes)if bars is None else validate_bars(bars,minutes)
    if len(bars)<max(801,spm.bars_for(720,minutes)+1):
        raise ValueError('For lite historikk til indikator og ML-egenskaper.')
    decision=bars.index[-1]+pd.Timedelta(minutes=int(minutes))
    if decision!=decision.normalize():raise ValueError('ML-modellen vurderer bare ved UTC-døgnskifte.')
    # Samme historikkfunksjoner som under trening; bare den valgte indikatorens rad brukes.
    when,inputs=spm.live_inputs(bars,minutes)
    item=inputs[key]
    if not item['clean'] or not np.isfinite(item['x'].to_numpy(float)).all():
        raise ValueError('Ingen prediksjon på utfylte lys eller ufullstendige ML-egenskaper.')
    return {'decision_time':decision.isoformat(),'position':item['position'],
            'values':item['x'].iloc[0].tolist(),'bar_count':len(bars)}


def assess(indicator='ut_bot',minutes='60',horizon='Neste uke',capital=10000,source=DEMO,*,bars=None):
    """Hvert klikk kjører predict_proba/predict; kapital er bare et regneeksempel."""
    if indicator not in spm.NAMES or minutes not in spm.TIMEFRAMES or horizon not in spm.HORIZONS or source not in SOURCES:
        raise ValueError('Velg en trent indikator, tidsramme, periode og datakilde.')
    if isinstance(capital,bool):raise ValueError('Kapital må være et positivt tall.')
    try:amount=float(capital)
    except(TypeError,ValueError):raise ValueError('Kapital må være et positivt tall.')from None
    if not np.isfinite(amount)or not 1<=amount<=10000000:
        raise ValueError('Kapital må være fra 1 til 10 000 000 kroner.')
    manifest,results=load()
    if source==DEMO:
        snapshot=json.loads((DIRECTORY/'demo_inputs.json').read_text())
        item=snapshot['timeframes'][minutes]['scripts'][indicator]
    else:item=live_input(minutes,indicator,bars)
    x=np.asarray(item['values'],dtype=float).reshape(1,-1)
    if x.shape!=(1,len(spm.ALL_COLUMNS))or not np.isfinite(x).all():
        raise ValueError('Ugyldig modellinput.')
    model=bundle(manifest)['models'][f'{minutes}|{indicator}|{horizon}']
    probability=float(model['clf'].predict_proba(x)[0,1])
    expected=float(model['reg'].predict(x)[0])
    if not np.isfinite([probability,expected]).all()or not 0<=probability<=1 or expected<=-1:
        raise ValueError('Modellen returnerte ugyldig anslag.')
    stats=results['timeframes'][minutes]['scripts'][indicator]['horizons'][horizon]
    columns=spm.FEATURE_SETS[stats['chosen']['features']]
    return {'indicator':indicator,'name':spm.NAMES[indicator],'minutes':minutes,'horizon':horizon,
            'capital':amount,'source':source,'decision_time':item['decision_time'],
            'prediction_end':(pd.Timestamp(item['decision_time'])+pd.Timedelta(hours=spm.HORIZONS[horizon])).isoformat(),
            'p_profit':probability,'expected_return':expected,'expected_pnl_nok':amount*expected,
            'historical_profitable_share':stats['profitable_share'],'position':item['position'],
            'classifier':model['clf_name'],'regressor':model['reg_name'],'stats':stats,
            'feature_count':len(columns),'features':{k:float(x[0,spm.ALL_COLUMNS.index(k)])for k in columns},
            'model_sha256':manifest['runtime_sha256'][manifest['model_path']]}


def dataset_view(indicator='ut_bot',minutes='60',horizon='Neste uke',start='2026-09-01',end='2026-10-11'):
    """Avgrenset tabell, fullstendige kildekontrollerte CSV-filer til nedlasting."""
    if indicator not in spm.NAMES or minutes not in spm.TIMEFRAMES or horizon not in spm.HORIZONS:
        raise ValueError('Velg en trent indikator, tidsramme og periode.')
    first=pd.Timestamp(start,tz='UTC');last=pd.Timestamp(end,tz='UTC')
    if first!=first.normalize()or last!=last.normalize()or first>=last:
        raise ValueError('Bruk datoer ÅÅÅÅ-MM-DD; til-datoen må være etter fra-datoen.')
    manifest,results=load()
    paths=[f'reports/script_profit_website_2026-10-27/BTCUSDT_{minutes}_bars.csv',
           f'reports/script_profit_website_2026-10-27/{minutes}_{indicator}_examples.csv',manifest['funding_file']]
    frames=[]
    for path in paths:
        if digest(ROOT/path)!=manifest['dataset_sha256'][path]:raise ValueError('Datasettet er endret. Visning og nedlasting er sperret.')
        frames.append(pd.read_csv(ROOT/path))
    raw,examples,funding=frames
    raw_times=pd.to_datetime(raw.open_time,utc=True)
    ex_times=pd.to_datetime(examples.decision_time,utc=True)
    hz={'Neste dag':'day','Neste uke':'week','Neste måned':'month'}[horizon]
    raw_filtered=raw.loc[(raw_times>=first)&(raw_times<last)]
    ex_filtered=examples.loc[(ex_times>=first)&(ex_times<last)]
    cols=['decision_time','position','rsi_14','ret_24h','script_ret_7d','script_ret_30d',
          f'return_{hz}',f'profit_{hz}',f'known_{hz}',f'used_training_{hz}']
    meta=manifest['datasets'][minutes]
    text=(f"**Datasett: Binance BTCUSDT spot · {spm.TIMEFRAMES[minutes]}.** {meta['bars']:,} prisrader fra "
          f"{meta['first']} til {meta['last']}; {meta['filled_bars']} utfylte lys er merket i `filled`. "
          f"{spm.NAMES[indicator]} har {len(examples):,} daglige ML-eksempler på tvers av horisontene. "
          f"Valgt modell er faktisk trent på **{int(examples[f'used_training_{hz}'].fillna(False).sum()):,}** av disse.\n\n"
          f"Valgt datoperiode har {len(raw_filtered):,} prisrader og {len(ex_filtered):,} ML-eksempler. "
          "Tabellene viser de siste 500 radene. Til-dato er eksklusiv. CSV-filene inneholder hele datasettet, "
          "alle 32 egenskaper og fasit/tilgjengelighet for dag, uke og måned. `used_training_*` viser hvilke rader "
          "som faktisk ble brukt ved siste modelltilpasning; andre rader kan være urene eller mangle senere fasit.\n\n"
          "`profit_*` er 1 når den simulerte nettoavkastningen er positiv, ellers 0. Tom fasit betyr ukjent. "
          "`known_*` angir når fasiten ble kjent. `ret_24h` er 24 **lys** på valgt intervall, slik treningskoden definerer det. "
          f"Fundingfilen inneholder {len(funding):,} faktiske Binance USDT-M-hendelser og kan også lastes ned. "
          "BTC-spotprisene er en proxy i en long/short-simulering med funding; dette er ikke faktiske WEEX-handler.")
    return text,raw_filtered.tail(500),ex_filtered[cols].tail(500),[str(ROOT/p)for p in paths]
