"""TMG-hypoteser om OI, ukespress og bekreftet prisstruktur.

OI er målt samlet kontraktsinteresse, aldri antall nye long/short alene.
Venteregler er eksplisitt nye: bare et ferskt signal kan opprette en plan,
den utløper etter seks timer og kanselleres hvis grunnmålet endrer retning.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PROFILES = {
    "oi_confirm4": "Inngang: stigende OI og pris i signalretning · 4 t",
    "oi_confirm24": "Inngang: stigende OI og pris i signalretning · 24 t",
    "oi_avoid_cover4": "Inngang: unngå prisbevegelse med fallende OI · 4 t",
    "oi_fade4": "Inngang: spot med signalet, pris motsatt og stigende OI · 4 t",
    "oi_spot_crowd24": "Inngang: spot med signalet, futures motsatt og stigende OI · 24 t",
    "oi_cover_exit4": "Utgang: motsatt pris og fallende OI · 4 t",
    "oi_new_exit4": "Utgang: motsatt pris og stigende OI · 4 t",
    "structure4": "Inngang: to avsluttede 4-timersblokker i signalretning",
    "pending_oi4": "Vent inntil 6 t på pris-/OI-bekreftelse · 4 t",
    "pending_structure4": "Vent inntil 6 t på bekreftet 4-timersstruktur",
    "real_weekly": "Inngang: ekte spotpress siste 7 døgn med signalet",
    "proxy_weekly": "Inngang: separat CLV-proxy siste 7 døgn med signalet",
}


def features(bars, spot, flow, aligned_oi, blocked=None):
    if not all(bars.index.equals(x.index) for x in (spot,flow,aligned_oi)):
        raise ValueError("Pris, spot og OI må ha samme tidsakse")
    out = pd.DataFrame(index=bars.index)
    out["eligible"] = True if blocked is None else ~np.asarray(blocked,dtype=bool)
    out["oi"] = aligned_oi.open_interest
    out["oi_observed_time"] = aligned_oi.observed_time
    for h in (4,24):
        valid = out.oi.notna().rolling(h+1).sum().eq(h+1)
        out[f"oi{h}"] = (out.oi/out.oi.shift(h)-1).where(valid)
        out[f"price{h}"] = bars.close/bars.close.shift(h)-1
    # Bare 4h-lys som var lukket ved timebarens åpning. Samme sikre HTF-regel som Pine [1].
    grouped = bars.close.resample("4h",origin="epoch",label="right").agg(["last","count"])
    close4 = grouped["last"].where(grouped["count"].eq(4))
    up = close4.gt(close4.shift(1)) & close4.shift(1).gt(close4.shift(2))
    down = close4.lt(close4.shift(1)) & close4.shift(1).lt(close4.shift(2))
    out["structure_long"] = up.reindex(bars.index,method="ffill").fillna(False).to_numpy(bool)
    out["structure_short"] = down.reindex(bars.index,method="ffill").fillna(False).to_numpy(bool)
    for k in ("spot_buy4","spot_buy24","perp_buy24"):
        out[k]=flow[k]
    volume,buy=spot.volume,spot.taker_buy_volume
    valid = (volume>0) & (buy>=0) & (buy<=volume*(1+1e-12))
    out["spot_buy168"] = buy.where(valid).rolling(168).sum()/volume.where(valid).rolling(168).sum()
    span=spot.high-spot.low
    clv=((2*spot.close-spot.high-spot.low)/span.where(span>0)).where(span!=0,0.)
    valid_proxy=spot[["high","low","close"]].notna().all(axis=1)&volume.gt(0)
    out["proxy168"]=(clv.clip(-1,1)*volume).where(valid_proxy).rolling(168).sum()/volume.where(valid_proxy).rolling(168).sum()
    return out.replace([np.inf,-np.inf],np.nan)


def allowed(values, profile, side):
    if profile not in PROFILES or side not in (-1,1):
        raise ValueError("Ukjent kontekst eller retning")
    h=24 if profile in ("oi_confirm24","oi_spot_crowd24") else 4
    oi,price=values[f"oi{h}"],values[f"price{h}"]
    if profile in ("oi_confirm4","oi_confirm24","pending_oi4"):
        permit=oi.gt(0)&(side*price).gt(0)
    elif profile=="oi_avoid_cover4":
        permit=oi.notna()&price.notna()&~(oi.lt(0)&(side*price).gt(0))
    elif profile=="oi_fade4":
        permit=oi.gt(0)&(side*price).lt(0)&(side*(values.spot_buy4-.5)).gt(0)
    elif profile=="oi_spot_crowd24":
        permit=oi.gt(0)&(side*(values.spot_buy24-.5)).gt(0)&(side*(values.perp_buy24-.5)).lt(0)
    elif profile in ("structure4","pending_structure4"):
        permit=values["structure_long" if side==1 else "structure_short"].eq(True)
    elif profile=="real_weekly":
        permit=(side*(values.spot_buy168-.5)).gt(0)
    elif profile=="proxy_weekly":
        permit=(side*values.proxy168).gt(0)
    else:
        permit=pd.Series(True,index=values.index)
    return permit.fillna(False)


def apply(frame, values, profile):
    if len(frame)!=len(values):
        raise ValueError("Ulik signallengde")
    allow={s:allowed(values,profile,s).to_numpy(bool)for s in (-1,1)}
    target=frame.position.to_numpy(int)
    fresh=np.where(frame.buy_signal,1,np.where(frame.sell_signal,-1,0))
    eligible=values.eligible.to_numpy(bool)
    oi,price=values.oi4.to_numpy(float),values.price4.to_numpy(float)
    n=len(frame)
    pos,buy,sell,exit_,extra,reject,waiting=[np.zeros(n,dtype=t)for t in (int,bool,bool,bool,bool,bool,int)]
    side,pending,expires=0,0,-1
    for i in range(n):
        if side and target[i]!=side:
            side=0
            exit_[i]=True
        if pending and (target[i]!=pending or i>expires or not eligible[i]):
            pending=0
        if side and profile in ("oi_cover_exit4","oi_new_exit4"):
            trigger=np.isfinite(oi[i]) and np.isfinite(price[i]) and side*price[i]<0
            trigger=trigger and (oi[i]<0 if profile=="oi_cover_exit4" else oi[i]>0)
            if trigger:
                side=0
                exit_[i]=extra[i]=True
        if fresh[i] and eligible[i]:
            pending=int(fresh[i])
            expires=i+(6 if profile.startswith("pending_")else 0)
        if not side and pending and not extra[i] and eligible[i] and allow[pending][i]:
            side=pending
            buy[i],sell[i]=side==1,side==-1
            pending=0
        elif fresh[i] and not side:
            reject[i]=True
        if not profile.startswith("pending_") or side:
            pending=0
        pos[i],waiting[i]=side,pending
    out=frame.copy()
    out["position"],out["buy_signal"],out["sell_signal"],out["exit_signal"]=pos,buy,sell,exit_
    out["position_context_exit"],out["position_context_rejected"],out["waiting_direction"]=extra,reject,waiting
    return out
