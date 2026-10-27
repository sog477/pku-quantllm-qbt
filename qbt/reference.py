"""Versioned, local-file point-in-time reference data and explicit equity rights.

CSV records: code,field,value,effective_from,known_at[,effective_until].
JSON manifest: profile,sessions,securities,records,actions,settlements,fees.
Files are inputs, never executable configuration. No network is used by loaders.
"""
import copy
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .actions import CorporateAction, CorporateActions, Settlement, TaxAssessment
from .markets import market_profile, SessionCalendar, local_timestamp
from .tradability import MarketMetadata, build_market, base_band, is_main_board
from ._validation import boolean_mask, integer


class ReferenceData:
    def __init__(self, profile, sessions, securities, *, records=(), actions=(), settlements=(),
                 fees=(), source='user supplied', periods_per_year=None, tick_rule=None,
                 factor_mode='derive', strict=True, taxes=(), quote_time=None, max_quote_age=None, settlement_sessions=None):
        self.profile=market_profile(profile)
        self.calendar=sessions if isinstance(sessions,SessionCalendar) else SessionCalendar(sessions,self.profile,source)
        if self.calendar.profile != self.profile:
            raise ValueError('reference calendar belongs to another market')
        if settlement_sessions is not None:
            self.calendar=SessionCalendar([dict(date=str(r[0]),open=str(r[1]),close=str(r[2]),
                break_start=None if r[3] is None else str(r[3]),break_end=None if r[4] is None else str(r[4]))
                for r in self.calendar.rows],self.profile,self.calendar.source,settlement_sessions)
        self.securities=tuple(dict(s) for s in securities)
        self.records=tuple(dict(s) for s in records)
        self.actions=tuple(actions)
        self.settlements=tuple(settlements)
        self.taxes=tuple(taxes)
        self.quote_time,self.max_quote_age=quote_time,max_quote_age
        self.fees=tuple(dict(r) for r in fees)
        self.source,self.periods_per_year,self.tick_rule=source,periods_per_year,tick_rule
        if factor_mode not in ('derive','validate_split'):
            raise ValueError('factor_mode must be derive/validate_split')
        self.factor_mode,self.strict=factor_mode,bool(boolean_mask(strict,'strict',()))
        if tick_rule is not None and not callable(tick_rule): raise ValueError('tick_rule must be a callable')
        self.boards=None
        self.assumptions=[]

    @classmethod
    def from_json(cls, path):
        path=Path(path)
        raw=json.loads(path.read_text(encoding='utf-8-sig'))
        # Optional CSV paths are relative to the manifest and cannot execute code.
        for k in ('sessions','securities','records','actions','settlements','fees','taxes'):
            if isinstance(raw.get(k),str):
                with (path.parent/raw[k]).open(encoding='utf-8-sig',newline='') as f:
                    raw[k]=list(csv.DictReader(f))
        for k,typ in [('actions',CorporateAction),('settlements',Settlement),('taxes',TaxAssessment)]:
            items=[]
            for r in raw.get(k,()):
                r={a:b for a,b in r.items() if b!=''}
                for name in ('cash_dividend','share_ratio','cash_in_lieu_price','withholding_rate','price','amount'):
                    if name in r: r[name]=float(r[name])
                items.append(typ(**r))
            raw[k]=items
        raw.setdefault('source',str(path.resolve()))
        return cls(**raw)

    def provenance(self):
        payload=dict(profile=self.profile.code,source=self.source,securities=self.securities,
                     records=self.records,actions=[vars(a) for a in self.actions],
                     settlements=[vars(a) for a in self.settlements],fees=self.fees,
                     taxes=[vars(a) for a in self.taxes],strict=self.strict,
                     quote_time=None if self.quote_time is None else np.asarray(self.quote_time).astype(str).tolist(),
                     max_quote_age=str(self.max_quote_age),profile_settings=vars(self.profile),periods_per_year=self.periods_per_year,
                     sessions=[[str(x) for x in r] for r in self.calendar.rows],factor_mode=self.factor_mode)
        payload['settlement_sessions']=self.calendar.settlement_days.astype(str).tolist()
        from .provenance import callable_provenance
        payload['tick_rule']=None if self.tick_rule is None else callable_provenance(self.tick_rule)
        return dict(source=self.source,sha256=hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest())

    def _normalize_event(self, ev, ds):
        from dataclasses import replace
        i=ds.calendar.index(ev.date,'exact')
        opening=self.calendar.row(ds.dates[i])[1]
        if self.strict and ev.known_at is None:
            raise ValueError('explicit rights require known_at in strict reference mode')
        distribution=(isinstance(ev,CorporateAction) and ev.cash_dividend>0 or isinstance(ev,Settlement) and ev.price>0)
        if self.strict and distribution and ev.withholding_rate is None:
            raise ValueError('cash rights require an explicit withholding_rate, including explicit 0 for exempt/gross input')
        if self.strict and isinstance(ev,CorporateAction) and ev.share_ratio>1 and ev.share_date is None:
            raise ValueError('share events require explicit share_date; do not assume immediate tradability')
        known=None if ev.known_at is None else local_timestamp(ev.known_at,self.profile)
        if known is not None and known>opening:
            raise ValueError(f'{ev.code} event not known before effective session open')
        # The legacy daily ledger compares known_at with its midnight date label;
        # reference validation above uses the actual exchange opening, then retains
        # detailed provenance in ReferenceData rather than reinterpreting that label.
        kw={'known_at':None}
        if isinstance(ev,(CorporateAction,Settlement)) and ev.withholding_rate is None:
            kw['withholding_rate']=0.
        return replace(ev,**kw)

    def prepare(self, ds, profile=None):
        profile=self.profile if profile is None else market_profile(profile)
        if profile != self.profile:
            raise ValueError('reference profile mismatch')
        self.calendar.validate(ds)
        codes={c:j for j,c in enumerate(ds.codes)}
        if len(self.securities)!=ds.N or {r['code'] for r in self.securities}!=set(codes):
            raise ValueError('security master must match every dataset code exactly')
        sec={r['code']:r for r in self.securities}
        shape=(ds.T,ds.N)
        st=np.zeros(shape,bool); suspended=np.zeros(shape,bool); no_limit=np.zeros(shape,bool)
        reference=np.full(shape,np.nan); age=np.full(shape,-1,np.int32)
        upper,lower=np.full(shape,np.nan),np.full(shape,np.nan)
        band=base_band(ds.codes,ds.dates) if profile.code=='CN' else np.full(shape,30 if profile.code=='KR' else 0,np.int8)
        first_open=self.calendar.row(ds.dates[0])[1]
        listing={}
        self.boards=[sec[c].get('board','KOSPI' if profile.code=='KR' else 'UNKNOWN') for c in ds.codes]
        for code,j in codes.items():
            r=sec[code]
            if r.get('currency',profile.currency)!=profile.currency:
                raise ValueError('one market run cannot mix quote currencies')
            listing[code]=np.datetime64(r['listing_date'],'D')
            if np.isnat(listing[code]): raise ValueError('missing true listing_date')
            if self.strict and not r.get('known_at'): raise ValueError('security master requires known_at')
            known=local_timestamp(r.get('known_at','1900-01-01'),profile)
            if known>first_open:
                raise ValueError('security master was unknown at dataset start; use earlier PIT baseline')
            for i,t in enumerate(ds.dates):
                day=t.astype('datetime64[D]')
                # For listings before calendar coverage, an explicit listed_sessions_at_start
                # is needed for exact age; otherwise use a declared lower bound.
                if day>=listing[code]:
                    p0=np.searchsorted(self.calendar.days,listing[code])
                    p=np.searchsorted(self.calendar.days,day)
                    value=r.get('listed_sessions_at_start',0)
                    if isinstance(value,str): value=int(value)
                    offset=integer(value,0,2**31-1,'listed_sessions_at_start')
                    age[i,j]=int(p-p0+offset)
                    if listing[code]<self.calendar.days[0] and 'listed_sessions_at_start' not in r:
                        age[i,j]=int(p-p0)  # conservative lower bound, never "very old".
                else: suspended[i,j]=True
                if r.get('delisting_date') and day>=np.datetime64(r['delisting_date'],'D'):
                    suspended[i,j]=True
        # Revisions: only a record already published at this opening can set that
        # session's state. Later corrections never rewrite an earlier row.
        parsed=[]; seen=set()
        allowed={'st','suspended','no_limit','reference','band','limit_up','limit_down'}
        for r in self.records:
            if r['code'] not in codes or r['field'] not in allowed:
                raise ValueError('unknown reference field or instrument')
            a=local_timestamp(r['effective_from'],profile)
            end=r.get('effective_until') or '2260-01-01'
            b=local_timestamp(end,profile)
            if len(str(end))==10: b+=np.timedelta64(1,'D')-np.timedelta64(1,'ns')
            known=local_timestamp(r['known_at'],profile)
            if np.isnat(a) or np.isnat(b) or b<a: raise ValueError('invalid reference interval')
            key=(r['code'],r['field'],a,known)
            if key in seen: raise ValueError('ambiguous reference revision')
            seen.add(key)
            parsed.append((r,a,b,known))
        for i,t in enumerate(ds.dates):
            day=t.astype('datetime64[D]')
            opening=t if getattr(ds,'intraday',False) else self.calendar.row(day)[1]
            selected={}
            for r,a,b,known in parsed:
                if a<=opening<=b and known<=opening:
                    key=(r['code'],r['field']); priority=(a,known)
                    if key not in selected or priority>selected[key][0]: selected[key]=(priority,r)
            for (code,field),(_,r) in selected.items():
                j=codes[code]; val=r['value']
                if field in ('st','suspended','no_limit'):
                    if isinstance(val,str):
                        if val.lower() not in ('true','false','0','1'): raise ValueError('invalid boolean reference value')
                        val=val.lower() in ('true','1')
                    if val not in (True,False,0,1): raise ValueError('invalid boolean reference value')
                    if field=='suspended': suspended[i,j] |= bool(val)
                    elif field=='st': st[i,j]=bool(val)
                    else: no_limit[i,j]=bool(val)
                elif field=='band':
                    v=float(val)
                    if v not in (0,5,10,20,30): raise ValueError('unsupported band; supply explicit limit prices for special sessions')
                    band[i,j]=int(v)
                else:
                    v=float(val)
                    if not np.isfinite(v) or v<=0: raise ValueError('invalid official reference')
                    {'reference':reference,'limit_up':upper,'limit_down':lower}[field][i,j]=v
            if self.strict:
                for code in codes:
                    # Baselines with explicit false values distinguish no-event from missing data.
                    for field in ('suspended',)+(('st',) if profile.code=='CN' else ()):
                        if (code,field) not in selected:
                            raise ValueError(f'missing PIT {field} state for {code} on {day}')
                    if profile.code=='KR' and day>=listing[code]:
                        for field in ('limit_up','limit_down'):
                            if (code,field) not in selected:
                                raise ValueError('strict KR reference requires official daily limit_up/limit_down, including IPO exceptions')
                    if profile.code=='CN' and self.calendar.days[0]<=listing[code]<=day:
                        first_sessions=self.calendar.days[(self.calendar.days>=listing[code]) & (self.calendar.days<=day)]
                        if len(first_sessions)<=5 and not ((code,'no_limit') in selected or
                                (code,'limit_up') in selected and (code,'limit_down') in selected):
                            raise ValueError('IPO sessions require explicit no_limit or official limit prices; history must not be inferred from first quote')
        if profile.code=='CN':
            st_band=np.where(ds.dates.astype('datetime64[D]')[:,None]>=np.datetime64('2026-07-06'),10,5)
            band=np.where(st & is_main_board(ds.codes)[None,:],st_band,band)
        band=np.where(no_limit,0,band)
        events=[self._normalize_event(ev,ds) for ev in self.actions]
        terminals=[self._normalize_event(ev,ds) for ev in self.settlements]
        assessments=[self._normalize_event(ev,ds) for ev in self.taxes]
        action_source=CorporateActions(events,terminals,assessments)
        prepared=copy.copy(ds); prepared._c={}; prepared._raw=dict(ds._raw)
        # Clone cached source attributes, never mutate source data. Cash dividends
        # stay cash: only share ratios enter trading/stop price adjustment factors.
        factor=np.ones(shape,float); cash=np.zeros(shape,float); ratios=np.ones(shape,float)
        for ev in events:
            i,j=ds.calendar.index(ev.date,'exact'),codes[ev.code]
            if ratios[i,j]!=1 or cash[i,j]!=0: raise ValueError('merge same-day same-code rights before import')
            ratios[i,j]=ev.share_ratio; cash[i,j]=ev.cash_dividend*(1-ev.withholding_rate)
        for i in range(ds.T): factor[i]=ratios[i] if i==0 else factor[i-1]*ratios[i]
        if not np.isfinite(factor).all() or (factor<=0).any(): raise ValueError('explicit share factors overflow')
        source_changes=np.abs(ds.cumadj_ff[1:]/ds.cumadj_ff[:-1]-1)>1e-4
        covered={(ds.calendar.index(ev.date,'exact'),codes[ev.code]) for ev in events+terminals}
        missing=[(int(i)+1,int(j)) for i,j in np.argwhere(source_changes) if (int(i)+1,int(j)) not in covered]
        if missing:
            raise ValueError(f'source factor jumps lack explicit rights events: {missing[:5]}')
        prepared.factor_diagnostics=dict(source_jump_cells=int(source_changes.sum()),
            mode=self.factor_mode,explicit_events=len(events),source_factor_basis='not assumed; execution uses explicit share ratios')
        if self.factor_mode=='validate_split':
            supplied=ds.cumadj_ff/ds.cumadj_ff[0]
            if not np.allclose(supplied,factor/factor[0],rtol=1e-8,atol=1e-10,equal_nan=False):
                raise ValueError('provided split factors disagree with explicit share events')
        prepared._raw['cumadj']=factor
        prepared.cumadj=factor
        prepared.explicit_split_basis=True
        prepared.market_code=profile.code
        prepared.market_boards=np.asarray(self.boards,dtype=str)
        qt=None
        fresh=np.ones(shape,bool)
        if self.quote_time is not None:
            from .tradability import _quote_evidence
            raw=np.asarray(self.quote_time)
            if raw.shape!=shape: raise ValueError('quote_time must align with dataset')
            qt=np.array([np.datetime64('NaT','ns') if str(v)=='NaT' else local_timestamp(v,profile) for v in raw.flat]).reshape(shape)
            _,fresh=_quote_evidence(prepared,MarketMetadata(quote_time=qt,max_quote_age=self.max_quote_age),{})
        # Ex-rights reference uses gross distribution, not investor withholding.
        last=np.full(ds.N,np.nan); filled=np.empty(shape,float); tri=np.empty(shape,float)
        prev=np.full(ds.N,np.nan); level=np.ones(ds.N,float)
        gross=np.zeros(shape,float)
        for ev in events: gross[ds.calendar.index(ev.date,'exact'),codes[ev.code]]=ev.cash_dividend
        prepared._cash_dividends,prepared._share_ratios=gross,ratios
        terminal_prices=np.full(shape,np.nan); closed=np.zeros(shape,bool)
        for ev in terminals:
            row,j=ds.calendar.index(ev.date,'exact'),codes[ev.code]
            terminal_prices[row,j]=ev.price*(1-ev.withholding_rate)
            closed[row:,j]=True
        suspended |= closed
        prepared.factor_diagnostics['quotes_on_or_after_termination']=int((closed & np.isfinite(ds.close)).sum())
        for i in range(ds.T):
            old=last.copy()
            theoretical=np.maximum(last-gross[i],0)/ratios[i]
            valid=np.isfinite(ds.close[i]) & (ds.close[i]>0) & fresh[i] & ~closed[i]
            last=np.where(valid,ds.close[i],theoretical)
            last=np.where(np.isfinite(terminal_prices[i]),terminal_prices[i],last)
            filled[i]=last
            if i:
                reference[i]=np.where(np.isfinite(reference[i]),reference[i],theoretical)
                with np.errstate(invalid='ignore',divide='ignore'):
                    ret=(ratios[i]*last+cash[i])/old
                level*=np.where(np.isfinite(ret),ret,1.)
            tri[i]=np.where(np.isfinite(last),level,np.nan)
        prepared._c['close_ff']=filled
        # A total return index is a signal/benchmark series, never a trading price.
        prepared.total_return_close=tri
        prepared._c['close_adj']=tri
        prepared._listed_age=age
        meta=MarketMetadata(st=st,no_limit=band==0,suspended=suspended,reference=reference,
            limit_up=upper,limit_down=lower,quote_time=qt,max_quote_age=self.max_quote_age)
        m=build_market(prepared,band_override=band,metadata=meta,st=False,log=None)
        self.assumptions=[
            'Reference states are evaluated by effective date AND publication time before session open',
            'Raw OHLC are execution prices; split factors from explicit share events; ctx.ret/vol use tax-net distribution total return',
            'Price-stop basis includes distributions earned on current holdings; no automatic dividend reinvestment',
            'Listing age before calendar coverage is a lower bound unless listed_sessions_at_start is supplied',
            'Investor-specific withholding and later tax assessments are explicit inputs; unknown investor tax is not inferred',
            'Daily OHLC execution does not identify exchange queue position; L3 replay is a separate entry point',
            f'Settlement dates: {self.calendar.settlement_source}; provide settlement_sessions for banking holidays distinct from trading holidays',
            'Daily source timestamps use daily bar labels; intraday physical source times require timestamp bars',
        ]
        if not self.strict: self.assumptions.append('Non-strict mode: missing reference states assume false; not a complete PIT security master')
        return prepared,m,action_source,self.calendar
