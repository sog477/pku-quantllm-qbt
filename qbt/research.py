"""Matched execution experiments and reviewable unresolved target attempts.

This layer uses the existing broker/account; it never fabricates terminal trades.
Constraint flags are concurrent diagnostics, not mutually exclusive causal shares.
"""
from dataclasses import replace
import numpy as np
from .broker import Broker
from .strategy import Target
from ._validation import boolean_mask, integer


def exit_schedule(target, dates, end, sessions=20):
    """Freeze last pre-exit membership and taper gross on a known daily calendar.

    The first exit day's target is set on the preceding close. No future price or
    capacity enters the schedule. Final session target is zero; inventory may remain.
    Stops/weights retain their original meaning; score is frozen with membership.
    """
    dates=np.asarray(dates)
    if dates.ndim!=1 or len(np.unique(dates.astype('datetime64[D]')))!=len(dates):
        raise ValueError('exit_schedule requires daily calendar rows')
    count=integer(sessions,1,len(dates),'sessions')
    end_day=np.datetime64(end,'D')
    hits=np.flatnonzero(dates.astype('datetime64[D]')==end_day)
    if len(hits)!=1:raise ValueError('end must be an explicit session in the supplied calendar')
    last=int(hits[0]);first=last-count+1
    if first<1:raise ValueError('not enough pre-exit history')
    sel=boolean_mask(target.sel,'target.sel').copy()
    if sel.ndim!=2 or sel.shape[0]!=len(dates):raise ValueError('exit_schedule requires a (T,N) target')
    keep=sel.copy() if target.keep is None else boolean_mask(target.keep,'target.keep',sel.shape).copy()
    gross=np.broadcast_to(np.asarray(target.gross,float),(len(dates),)).copy()
    if not np.isfinite(gross).all() or (gross<0).any():raise ValueError('invalid gross')
    anchor=first-1
    # Freeze target membership; run_traced(stop_buys_from=...) also prevents adding
    # previously unfilled target stocks and rebalancing buys during the exit period.
    members=sel[anchor].copy();retained=keep[anchor].copy();base=float(gross[anchor])
    sel[anchor:]=members;keep[anchor:]=retained
    score=None if target.score is None else np.asarray(target.score,float).copy()
    if score is not None:
        if score.shape!=sel.shape:raise ValueError('score must be an aligned panel')
        score[anchor:]=score[anchor].copy()
    # Orders on row i consume the signal on row i-1.
    gross[anchor:last]=base*np.arange(count-1,-1,-1)/count
    gross[last:]=0
    return replace(target,sel=sel,keep=keep,gross=gross,score=score,
        note=target.note+f'; scheduled exit {dates[first]}..{dates[last]}, inventory not forced'),dict(
            first_execution=str(dates[first]),last_execution=str(dates[last]),sessions=count,
            assumption='known session calendar; freeze pre-exit membership; no forced cash settlement')


class TraceBroker(Broker):
    """Optional target-broker trace; all quantities come from actual execute calls."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not self.record_trades:raise ValueError('TraceBroker requires record_trades=True')
        self.attempts=[]
        self.stop_buy_index=None
        self.suppressed_buy_requests=0

    def execute(self,acct,i,want_qty,day,phase='open'):
        held=acct.shares.copy();want=np.asarray(want_qty).copy()
        if self.stop_buy_index is not None and i>=self.stop_buy_index:
            self.suppressed_buy_requests+=int((want>held).sum())
            want=np.minimum(want,held)
            want_qty=want
        available=np.asarray(self.account_rules.sellable(acct)).copy()
        cash=float(self.account_rules.buying_power(acct))
        used=self.begin_capacity(i).copy()
        capacity=np.maximum(self.capacity_amount(i)-used,0)
        info=super().execute(acct,i,want_qty,day,phase)
        filled={}
        for _,j,side,q,price,fee in info['fills']:
            filled[(j,side)]=filled.get((j,side),0)+q
        off_buy=self.asset_rules.can_buy(self.m,i,'off')
        off_sell=self.asset_rules.can_sell(self.m,i,'off')
        for j in np.flatnonzero(want!=held):
            side=1 if want[j]>held[j] else -1
            requested=int(abs(want[j]-held[j]));done=int(filled.get((j,side),0))
            if done==requested:continue
            price=float(info['px_buy' if side==1 else 'px_sell'][j])
            off=off_buy if side==1 else off_sell
            limit_flag=bool(info['buy_mask' if side==1 else 'sell_mask'][j])
            flags=[]
            if not np.isfinite(self.m.ds.amt[i,j]) or self.m.ds.amt[i,j]<=0:flags.append('no_traded_amount')
            if not bool(off[j]):flags.append('unavailable_quote_or_state')
            if limit_flag:flags.append('limit_gate')
            if side<0 and available[j]<requested:flags.append('frozen_or_restricted_inventory')
            if not np.isfinite(price) or price<=0:flags.append('invalid_execution_price')
            elif capacity[j]+1e-7<requested*price:flags.append('capacity_below_requested')
            if side>0 and np.isfinite(price) and cash<requested*price:flags.append('initial_cash_below_requested_notional')
            flags.append('rounding_cash_allocation_or_other_rules_possible')
            self.attempts.append(dict(date=str(day),index=int(i),code=self.m.ds.codes[j],
                phase=phase,side='buy' if side==1 else 'sell',requested=requested,
                filled=done,unfilled=requested-done,status='partial' if done else 'unfilled',
                execution_price=price,capacity_before= float(capacity[j]),
                constraints=';'.join(flags)))
        return info


def run_traced(ds,strategy,*,stop_buys_from=None,**options):
    """Return (Result, incomplete_attempts); does not alter qbt.run defaults."""
    from .engine import run
    if 'broker_factory' in options:raise ValueError('run_traced supplies TraceBroker')
    brokers=[]
    def factory(*a,**kw):
        b=TraceBroker(*a,**kw)
        if stop_buys_from is not None:
            hits=np.flatnonzero(ds.dates.astype('datetime64[D]')==np.datetime64(stop_buys_from,'D'))
            if len(hits)!=1:raise ValueError('stop_buys_from must be an explicit session')
            b.stop_buy_index=int(hits[0])
        brokers.append(b);return b
    result=run(ds,strategy,broker_factory=factory,**options)
    result.target_attempts=brokers[0].attempts
    result.scheduled_suppressed_buy_requests=brokers[0].suppressed_buy_requests
    return result,result.target_attempts


def residual_inventory(result):
    """Marked residuals, last source dates and frozen stock; no fictitious sales."""
    a=result.account;d=result.ds;rows=[]
    for j in np.flatnonzero(a.shares>0):
        t=int(result.step_indices[-1]);valid=np.isfinite(d.close[:t+1,j])&(d.close[:t+1,j]>0)
        last=np.flatnonzero(valid)
        price=float(result.final_prices[j]);qty=int(a.shares[j])
        rows.append(dict(code=d.codes[j],shares=qty,frozen=int(a.frozen[j]),
            restricted=int(a.restricted[j]),valuation_price=price,marked_value=qty*price,
            last_raw_quote=str(d.dates[last[-1]]) if len(last) else '',
            stale_sessions=int(a.stale[j]),last_recorded_price=float(a.last_px[j]),
            last_price_value=qty*float(a.last_px[j]),
            valuation_write_down=bool(np.isfinite(price) and price==0),
            actual_cash_proceeds=0.))
    return rows


def data_quality(ds):
    """Internal consistency, not vendor authenticity or historical-universe proof."""
    o,h,l,c=[np.asarray(getattr(ds,k),float) for k in ['open','high','low','close']]
    observed=np.isfinite(o)&np.isfinite(h)&np.isfinite(l)&np.isfinite(c)
    violations=observed&((l>h)|(o<l)|(o>h)|(c<l)|(c>h))
    factor=np.asarray(ds.cumadj_ff,float)
    with np.errstate(invalid='ignore',divide='ignore'):
        delta=np.abs(factor[1:]/factor[:-1]-1)
    examples=[dict(date=str(ds.dates[i]),code=ds.codes[j]) for i,j in np.argwhere(violations)[:20]]
    return dict(rows=ds.T,stocks=ds.N,complete_ohlc=int(observed.sum()),
        inconsistent_ohlc=int(violations.sum()),inconsistent_examples=examples,
        missing_by_field={k:int((~np.isfinite(getattr(ds,k))).sum()) for k in ['open','high','low','close','amt','cumadj']},
        factor_small_nonzero_changes=int(((delta>0)&(delta<=1e-4)).sum()),
        factor_changes_above_tolerance=int((delta>1e-4).sum()),
        limitation='internal checks only; no true volume/VWAP, ST or external price verification inferred')
