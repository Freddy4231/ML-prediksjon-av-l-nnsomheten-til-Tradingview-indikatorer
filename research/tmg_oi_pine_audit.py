"""Kontroller Pine-ventetilstanden med ferdig justert OI, ikke native TV-feeden."""
from pathlib import Path
import numpy as np
import pandas as pd
from pine_engine import runner
from pine_engine.compiler import compile_source
import tmg_position_context as rules
from research import tmg_improvement as base

SOURCE=Path(__file__).resolve().parents[1]/"tradingview/tmg_oi_confirmation.pine"


def run_core(frame,values,index):
    source=SOURCE.read_text()
    start=source.index("// ---------------- Ventetilstand")
    end=source.index("// ---------------- Slutt på OI-ventetilstand")
    prefix='''//@version=6
indicator("Isolert OI-tilstandstest", overlay=true)
bool validTf = true
bool oiEligible = true
int filteredPos = int(open)
bool filteredBuy = high > 0
bool filteredSell = low > 0
float oiKnown = volume
bool oiWindow = not na(oiKnown) and not na(oiKnown[1]) and not na(oiKnown[2]) and not na(oiKnown[3]) and not na(oiKnown[4])
float oiChange4 = oiWindow ? oiKnown / oiKnown[4] - 1.0 : na
float priceChange4 = not na(close[4]) ? close / close[4] - 1.0 : na
bool oiLongAllowed = not na(oiChange4) and oiChange4 > 0 and priceChange4 > 0
bool oiShortAllowed = not na(oiChange4) and oiChange4 > 0 and priceChange4 < 0
'''
    capture='\nplot(oiBuy ? 1 : 0, "Test buy")\nplot(oiSell ? 1 : 0, "Test sell")\nplot(oiExit ? 1 : 0, "Test exit")\n'
    compiled=compile_source(prefix+source[start:end]+capture)
    times=index.as_unit("ms").asi8
    # Kanalene injiserer ferdig grunnmål og OI; disse er ikke vanlige OHLCV-lys.
    bars=runner.Bars("BTCUSDT.P",60,times,times+3600000,frame.position.to_numpy(float),
                     frame.buy_signal.to_numpy(float),frame.sell_signal.to_numpy(float),
                     frame.close.to_numpy(float),values.oi.to_numpy(float))
    inputs,sources=runner.resolve_inputs(compiled)
    result,_=runner.run_with_security(compiled,bars,inputs=inputs,sources=sources,capture_plots=True)
    outputs={title:np.array([v for _,v in pairs],dtype=float)for(_,title),pairs in result.events.series_values.items()}
    return result,outputs


def main():
    from research import tmg_position_search as search
    data,frames,values=search.contexts()
    checks=[]
    for symbol in search.SYMBOLS:
        mask=(data[symbol]["bars"].index>=pd.Timestamp("2024-01-01",tz="UTC"))&(data[symbol]["bars"].index<pd.Timestamp("2024-04-01",tz="UTC"))
        frame=frames[symbol].loc[mask].reset_index(drop=True)
        value=values[symbol].loc[mask].reset_index(drop=True)
        index=data[symbol]["bars"].index[mask]
        assert value.eligible.all() and not(frame.buy_signal.iloc[:4]|frame.sell_signal.iloc[:4]).any()
        expected=rules.apply(frame,value,"pending_oi4")
        result,plots=run_core(frame,value,index)
        mismatches={}
        for name,field in (("OI-bekreftet målposisjon","position"),("Ventende retning · maks6timer","waiting_direction"),
                           ("Test buy","buy_signal"),("Test sell","sell_signal"),("Test exit","exit_signal")):
            mismatches[field]=int(np.count_nonzero(plots[name]!=expected[field].to_numpy()))
        assert not any(mismatches.values()),symbol
        checks.append({"symbol":symbol,"hours":len(index),"mismatches":mismatches})
    base.write(search.OUTPUT/"pine_logic_audit.json",{"checks":checks,"source_sha256":base.sha(SOURCE),
               "scope":"Kun valgt Pine-ventetilstand, injisert kjent OI/grunnmål og faktiske 1h-close. Ingen native TV-feed eller full request.security mot lavere TF.",
               "native_feed_verified":False,"native_compilation_verified":False})
    print(checks)


if __name__=="__main__":main()
