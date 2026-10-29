"""Separat Pine-prototype av valgt venteregel; TV-feeden er ikke arkivvalidert."""
import json
from research import tmg_position_search as search
from research import tmg_improvement as base


def build():
    choice=json.loads((search.OUTPUT/"frozen_choice.json").read_text())
    assert choice["selected"]["id"]=="pending_oi4"
    plan=search.lock()
    for p,h in plan["sha256"].items():assert base.sha(base.ROOT/p)==h,p
    text=(base.ROOT/"tradingview/tmg_daily_confirmation.pine").read_text()
    text=text.replace('indicator("TMG Dagsbekreftelse · kjøp/selg (DAT158)", "TMG Dagsbekreftelse",',
                      'indicator("TMG OI-bekreftelse · prototype", "TMG OI · forskning",')
    text=text.replace("// Fryst valg: daily50_weekend_both.",
        "// Valgt regel: daily50_weekend_both + pending_oi4. Kun forskningsprototype.\n"
        "// Python-arkivets avkastning er IKKE et verifisert TradingView-resultat.\n"
        "// TradingView bruker sin egen OI-feed. Arkivets historiske tilgjengelighet er uavklart.\n"
        "// Siste avsluttede 5m-underbar med OI open[12] gir minst65min forsinkelse.\n"
        "// Datamangler eller forskjellig sampling kan gi andre signaler enn Python.")
    text=text.replace("bool validTf = timeframe.in_seconds() == 3600 and chart.is_standard",
        'bool testedCoin = syminfo.ticker == "BTCUSDT.P" or syminfo.ticker == "ETHUSDT.P" or syminfo.ticker == "SOLUSDT.P"\nbool validMarket = testedCoin and syminfo.prefix == "BINANCE" and spotSymbol == "BINANCE:" + str.replace_all(syminfo.ticker, ".P", "")\nbool validTf = timeframe.in_seconds() == 3600 and chart.is_standard and validMarket')
    marker="// Et avvist signal blir aldri inngang senere uten et nytt grunnsignal."
    additions='''// ---------------- OI-feed · separat TradingView-prototype ----------------
string oiSymbol = syminfo.prefix + ":" + syminfo.ticker + "_OI"
// Historisk request.security velger siste5m-underbar. Bare endelig1h-close handler.
// Originale5m-tider beholdes; ingen pris-/volumserie brukes i stedet for OI.
[oiSample, oiObservedTime] = request.security(oiSymbol, "5", [open[12], time[12]], gaps=barmerge.gaps_on, lookahead=barmerge.lookahead_off, ignore_invalid_symbol=true)
float oiAge = time_close - oiObservedTime
float oiKnown = not na(oiSample) and oiSample > 0 and not na(oiAge) and oiAge >= 3900000 and oiAge < 11100000 ? oiSample : na
bool oiWindow = not na(oiKnown) and not na(oiKnown[1]) and not na(oiKnown[2]) and not na(oiKnown[3]) and not na(oiKnown[4])
float oiChange4 = oiWindow ? oiKnown / oiKnown[4] - 1.0 : na
float priceChange4 = not na(close[4]) ? close / close[4] - 1.0 : na
bool oiLongAllowed = not na(oiChange4) and oiChange4 > 0 and priceChange4 > 0
bool oiShortAllowed = not na(oiChange4) and oiChange4 > 0 and priceChange4 < 0
[oiSpotVolume, oiSpotTime] = request.security(spotSymbol, "60", [volume, time], gaps=barmerge.gaps_on, lookahead=barmerge.lookahead_off)
var int oiLastBadTime = na
if (not na(time[1]) and time - time[1] != 3600000) or na(oiSpotVolume) or oiSpotTime != time
    oiLastBadTime := time
bool oiEligible = na(oiLastBadTime) or time - oiLastBadTime > 2880000000

// ---------------- Ventetilstand · testes separat mot Python ----------------
var int oiPos = 0
var int oiPending = 0
var int oiExpires = -1
bool oiBuy = false
bool oiSell = false
bool oiExit = false
bool oiRejected = false
if barstate.isconfirmed and validTf
    if oiPos != 0 and filteredPos != oiPos
        oiPos := 0
        oiExit := true
    if oiPending != 0 and (filteredPos != oiPending or time_close > oiExpires or not oiEligible)
        oiPending := 0
    int oiFresh = filteredBuy ? 1 : filteredSell ? -1 : 0
    if oiFresh != 0 and oiEligible
        oiPending := oiFresh
        oiExpires := time_close + 21600000
    bool oiPermit = oiPending == 1 ? oiLongAllowed : oiPending == -1 ? oiShortAllowed : false
    if oiPos == 0 and oiPending != 0 and oiEligible and oiPermit
        oiPos := oiPending
        oiBuy := oiPos == 1
        oiSell := oiPos == -1
        oiPending := 0
    else if oiFresh != 0 and oiPos == 0
        oiRejected := true
    if oiPos != 0
        oiPending := 0
plot(oiPos, "OI-bekreftet målposisjon", display=display.none)
plot(oiPending, "Ventende retning · maks6timer", display=display.none)
plot(oiKnown, "Kjent OI · separat TV-feed", display=display.data_window)
plot(oiChange4 * 100, "OI-endring4t · prosent", display=display.data_window)
plotshape(oiPending == 1 and not oiBuy, title="VENT KJØP", text="VENT", style=shape.circle, location=location.belowbar, color=#FBBF24, size=size.tiny)
plotshape(oiPending == -1 and not oiSell, title="VENT SELG", text="VENT", style=shape.circle, location=location.abovebar, color=#FBBF24, size=size.tiny)
// ---------------- Slutt på OI-ventetilstand ----------------

'''
    assert marker in text
    text=text.replace(marker,additions+marker)
    text=text.replace(marker,"// Bare en registrert plan kan vente. Etter seks timer kreves et nytt signal.")
    for old,new in (("plotshape(filteredBuy","plotshape(oiBuy"),("plotshape(filteredSell","plotshape(oiSell"),
                    ("plotshape(filteredExit","plotshape(oiExit"),("alertcondition(filteredBuy","alertcondition(oiBuy"),
                    ("alertcondition(filteredSell","alertcondition(oiSell"),("alertcondition(filteredExit","alertcondition(oiExit")):
        text=text.replace(old,new)
    text=text.replace('"TMG Dagsbekreftelse · forskning"','"TMG OI · prototype"')
    text=text.replace('"Velg vanlige 1-timerslys"','"Krever samsvarende Binance-perp/spot og vanlige1t-lys"')
    text=text.replace('"Ingen ordre · se kostnadstesten i appen"','"TV-feed ikke arkivvalidert · ingen ordre"')
    text=text.replace('"Velg samme mynt spot og futures"','na(oiKnown) ? "OI mangler · ingen nye innganger" : "Separat TV OI-feed"')
    for action in ("KJØP","SELG","LUKK"):
        text=text.replace("TMG Dagsbekreftelse · "+action,"TMG OI · "+action)
    target=base.ROOT/"tradingview/tmg_oi_confirmation.pine"
    target.write_text(text)
    print(target)


if __name__=="__main__":build()
