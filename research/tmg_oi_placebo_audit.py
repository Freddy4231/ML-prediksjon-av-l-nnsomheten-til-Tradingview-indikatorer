"""Etterkontroll av lik kalenderdekning; ingen ny kandidat eller parameterjakt."""
import json
import numpy as np
import pandas as pd
from research import tmg_position_search as search
from research import tmg_channel_search as channel
from research import tmg_improvement as base
import tmg_position_context as rules


def main():
    path=search.OUTPUT/'placebo_coverage_audit.json'
    if path.exists():raise ValueError('Etterkontrollen finnes allerede')
    choice=json.loads((search.OUTPUT/'frozen_choice.json').read_text())
    assert choice['selected']['id']=='pending_oi4'
    data,frames,values=search.contexts()
    symbol='BTCUSDT'
    original=values[symbol]
    contexts=[original]+[original.shift(h)for h in channel.SHIFTS]
    common=original.eligible.copy()
    for value in contexts:
        common &= value[['oi4','price4']].notna().all(axis=1)
    results=[]
    for h,value in zip((0,*channel.SHIFTS),contexts):
        value=value.copy()
        value['eligible']=common
        frame=rules.apply(frames[symbol],value,'pending_oi4')
        results.append({'hours':h,'training':base.evaluate(data[symbol],frame,search.TRAIN,'both'),
                        'validation':base.evaluate(data[symbol],frame,search.VALID,'both')})
    coverage={}
    for label,period in (('training',search.TRAIN),('validation',search.VALID)):
        mask=(original.index>=pd.Timestamp(period[0],tz='UTC'))&(original.index<pd.Timestamp(period[1],tz='UTC'))
        dates=original.index[mask&common]
        coverage[label]={'eligible_hours':len(dates),'calendar_hours':int(mask.sum()),'first_eligible':dates[0].isoformat()if len(dates)else None}
    actual=results[0]
    gates={}
    thresholds={}
    for period,q in (('training',.75),('validation',.5)):
        threshold=float(np.quantile([r[period]['stress']['return']for r in results[1:]],q))
        thresholds[period]=threshold
        gates[period]=bool(actual[period]['stress']['return']>threshold)
    document={'choice_sha256':base.sha(search.OUTPUT/'frozen_choice.json'),'source_sha256':base.sha(__file__),
              'after_selection':True,'method':'Samme kausale tilgjengelighetsmaske for valgt regel og alle8placeboer. Kun oi4/price4 fra faktisk/fortidsskift; faktiske hullsperrer beholdt. Inngangsplaner utenfor fellesmasken opprettes ikke eller kanselleres. Ingen ny utvelgelse.',
              'coverage':coverage,'results':results,'thresholds':thresholds,'gates':gates,'passed':all(gates.values()),
              'limitations':'Etterkontroll, ikke forhåndslåst ny test. Løser ikke historisk publisering/native feed eller stor utvalgsusikkerhet.'}
    base.write(path,document)
    print(json.dumps({'passed':document['passed'],'coverage':coverage,'actual_stress':{p:actual[p]['stress']['return']for p in ('training','validation')},'thresholds':thresholds},indent=2))


if __name__=='__main__':main()
