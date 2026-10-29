"""Ekte ML-respons, treningsdata og stengte feilstier uten ordre (sidens rute testes i test_kopi_uten_handel)."""
import json
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import script_profit_predictor as model
import script_profit_dashboard as page
import script_profit_ml as spm


class MLWebsiteTests(unittest.TestCase):
    def test_real_saved_models_for_every_indicator_and_horizon_and_amount_scales_only_money(self):
        with patch('requests.sessions.Session.request',side_effect=AssertionError('Ingen HTTP')):
            for minutes in spm.TIMEFRAMES:
                for indicator in spm.NAMES:
                    for horizon in spm.HORIZONS:
                        r=model.assess(indicator,minutes,horizon,10000)
                        self.assertTrue(0<=r['p_profit']<=1)
                        self.assertTrue(np.isfinite(r['expected_return']))
                        self.assertEqual(r['feature_count'],len(r['features']))
                        self.assertEqual(pd.Timestamp(r['decision_time']).hour,0)
            a=model.assess('ut_bot','60','Neste uke',1000)
            b=model.assess('ut_bot','60','Neste uke',2000)
            self.assertEqual(a['p_profit'],b['p_profit'])
            self.assertAlmostEqual(a['expected_return'],b['expected_return'],places=14)
            self.assertAlmostEqual(b['expected_pnl_nok'],2*a['expected_pnl_nok'])
            c=model.assess('chandelier','240','Neste måned',1000)
            self.assertNotEqual((a['p_profit'],a['expected_return']),(c['p_profit'],c['expected_return']))
            out=page.view('ut_bot','60','Neste uke','Random Forest',10000,compare_all=False,cadence='Hvert år')
            text,predictions,quality,speed,drift,importance=out
            self.assertIn('Random Forest anslår',text)
            self.assertIn('ikke en garanti eller finansiell rådgivning',text)
            self.assertIn('lagrede, tunede modellen fra prosjektets trening (statisk) 26,6 %',text)
            self.assertEqual(len(predictions),1)

    def test_visible_datasets_have_labels_known_times_training_flags_and_full_downloads(self):
        manifest,results=model.load()
        for minutes in spm.TIMEFRAMES:
            for indicator in spm.NAMES:
                for horizon,hz in zip(spm.HORIZONS,('day','week','month')):
                    text,raw,examples,downloads=model.dataset_view(indicator,minutes,horizon,'2026-01-01','2026-10-11')
                    self.assertEqual(len(downloads),3)
                    self.assertLessEqual(len(raw),500)
                    self.assertLessEqual(len(examples),500)
                    self.assertIn('filled',raw.columns)
                    self.assertIn('known_'+hz,examples.columns)
                    self.assertIn('used_training_'+hz,examples.columns)
                    self.assertTrue(all(Path(p).is_file()for p in downloads))
                    known=pd.to_datetime(examples['known_'+hz],utc=True)
                    decisions=pd.to_datetime(examples.decision_time,utc=True)
                    available=known.notna()
                    self.assertTrue((known[available]>=decisions[available]+pd.Timedelta(hours=spm.HORIZONS[horizon])).all())
                    self.assertTrue((examples.loc[available,'profit_'+hz]==examples.loc[available,'return_'+hz].gt(0).astype(int)).all())
                    self.assertTrue(examples.loc[~available,'profit_'+hz].isna().all())
                    full=pd.read_csv(downloads[1])
                    stats=results['timeframes'][minutes]['scripts'][indicator]['horizons'][horizon]
                    expected=round(stats['samples']*stats['clean_share'])if stats['chosen']['cleaning']=='på'else stats['samples']
                    self.assertEqual(int(full['used_training_'+hz].sum()),expected)
        _,raw,examples,_=model.dataset_view(start='2100-01-01',end='2100-02-01')
        self.assertTrue(raw.empty and examples.empty)

    def test_bad_input_and_network_failure_never_fall_back_to_historical_prediction(self):
        for fields in ({'indicator':'arbitrary pine'},{'minutes':'5'},{'horizon':'Neste år'},{'capital':-1},
                       {'capital':float('nan')},{'capital':True},{'capital':None},{'source':'ukjent'}):
            with self.assertRaises(ValueError):model.assess(**fields)
        import script_model_compare as compare
        with patch.object(model,'fetch_bars',side_effect=ValueError('Datakilden er utilgjengelig')),\
                patch.dict(compare._LIVE,clear=True):
            text,predictions,*_=page.view(source=model.LIVE,compare_all=False,cadence='Hvert år')
            self.assertIn('Ingen prediksjon for valgt input',text)
            self.assertTrue(predictions.empty)

    def test_tampered_data_or_model_contract_fails_closed(self):
        manifest,_=model.load()
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            for name in [*manifest['runtime_sha256'],*manifest['dataset_sha256'],
                         'reports/script_profit_website_2026-10-27/manifest.json']:
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
                # CSV-ene trenger ikke kopieres fysisk før én av dem muteres.
                target.symlink_to(model.ROOT/name)
            dataset=root/'reports/script_profit_website_2026-10-27/BTCUSDT_60_bars.csv'
            dataset.unlink();dataset.write_text('endret')
            with patch.object(model,'ROOT',root),patch.object(model,'DIRECTORY',root/'reports/script_profit_website_2026-10-27'):
                with self.assertRaisesRegex(ValueError,'Datasettet er endret'):model.dataset_view()
                code=root/'script_profit_ml.py';code.unlink();code.write_text('endret')
                with self.assertRaisesRegex(ValueError,'endret'):model.assess()

    def test_public_data_parser_rejects_gaps_conflicts_stale_and_unfinished_bars(self):
        class Reply:
            def __init__(self,rows):self.rows=rows
            def raise_for_status(self):pass
            def json(self):return self.rows
        class Session:
            def __init__(self,rows):self.rows=rows;self.calls=[]
            def get(self,url,**kwargs):self.calls.append((url,kwargs));return Reply(self.rows)
        end=pd.Timestamp('2026-10-27',tz='UTC')
        def row(t):
            ms=int(t.timestamp()*1000)
            return [ms,'100','102','99','101','5',ms+3600000-1]
        good=[row(end-pd.Timedelta(hours=i))for i in (3,2,1)]
        session=Session(good)
        parsed=model.fetch_bars('60',session=session,now=end+pd.Timedelta(hours=12),count=3)
        self.assertEqual(parsed.index[-1]+pd.Timedelta(hours=1),end)
        self.assertTrue(all(call[1]['params']['symbol']=='BTCUSDT'for call in session.calls))
        wrongs=[good[:1]+good[2:],good[:-1]+[row(end)],good[:-1]+[row(end-pd.Timedelta(hours=4))]]
        conflict=good+[list(good[-1])];conflict[-1][4]='100'
        wrongs.append(conflict)
        for rows in wrongs:
            with self.assertRaises(ValueError):model.fetch_bars('60',session=Session(rows),now=end,count=3)
        bad=parsed.copy();bad.iloc[-1,bad.columns.get_loc('high')]=1
        with self.assertRaises(ValueError):model.validate_bars(bad,'60')
        with self.assertRaises(ValueError):model.live_input('60','ut_bot',parsed)


if __name__=='__main__':unittest.main()
