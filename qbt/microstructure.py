"""Single-session shadow L3 replay with exchange-order queue evidence.

Requires sequenced order-by-order add/cancel/execute messages. Public historical
messages are replayed unchanged; own orders are counterfactual shadow orders.
This can estimate queue fills under displayed FIFO, not prove historical fills
or predict other participants' response to our orders. Hidden liquidity,
auctions, cross-venue routing and pro-rata priority are deliberately unsupported.
"""
from dataclasses import dataclass, asdict
from types import MappingProxyType, SimpleNamespace
import csv
import heapq
import hashlib
import json
from pathlib import Path
import numpy as np
from .account import Account
from .costs import evaluate_cost
from .orders import OrderRequest
from .markets import market_profile, local_timestamp, AStockRules, EquityRules, ScheduledCosts
from ._validation import integer


@dataclass(frozen=True)
class L3Event:
    timestamp: object
    sequence: int
    kind: str
    code: str
    order_id: str
    side: str
    price: float
    quantity: int

    def __post_init__(self):
        object.__setattr__(self,'sequence',integer(self.sequence,1,2**63-1,'sequence'))
        object.__setattr__(self,'quantity',integer(self.quantity,1,2**63-1,'quantity'))
        if self.kind not in ('add','cancel','execute') or self.side not in ('buy','sell'):
            raise ValueError('L3 kind=add/cancel/execute; side is the RESTING order side')
        if not isinstance(self.order_id,str) or not self.order_id or not np.isfinite(self.price) or self.price<=0:
            raise ValueError('invalid L3 order id or price')


def load_l3_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:
        return tuple(L3Event(r['timestamp'],int(r['sequence']),r['kind'],r['code'],r['order_id'],
                             r['side'],float(r['price']),int(r['quantity'])) for r in csv.DictReader(f))


class L3Context:
    def __init__(self,replay,timestamp):
        self.now=timestamp
        self.cash=replay.account.cash
        self.positions=MappingProxyType({c:int(replay.account.shares[j]) for c,j in replay.codes.items()})
        self.book=MappingProxyType({oid:MappingProxyType(dict(r)) for oid,r in replay.public.items()})
        self.order=replay.submit
        self.cancel=replay.cancel


class L3Result:
    def __init__(self,replay):
        self.account=replay.account
        self.orders=tuple(dict(o,ahead=dict(o['ahead'])) for o in replay.orders.values())
        self.fills=tuple(dict(f) for f in replay.fills)
        self.events=tuple(dict(f) for f in replay.events)
        self.currency=replay.profile.currency
        self.source_sha256=replay.source_hasher.hexdigest()
        self.strategy_provenance=replay.strategy_provenance
        self.assumptions=[__doc__,f'Latency {replay.latency}; exchange event wins timestamp ties',
            'Aggressive fills consume displayed liquidity once per public order; passive fills share each print volume',
            'Detectable sequence gaps, invalid cancellations or non-FIFO executions abort; upstream completeness remains a source contract',
            'Single session only; use the bar engine for multi-day corporate actions and cash settlement']

    def save_json(self,path):
        orders=[dict(o,request=asdict(o['request'])) for o in self.orders]
        Path(path).write_text(json.dumps(dict(currency=self.currency,orders=orders,fills=self.fills,
            events=self.events,account=self.account.snapshot(),source_sha256=self.source_sha256,
            strategy_provenance=self.strategy_provenance,assumptions=self.assumptions),default=str,indent=2),encoding='utf-8')
        return str(Path(path).resolve())


class _Replay:
    def __init__(self,profile,calendar,codes,capital,costs,latency,initial_holdings,admission):
        self.profile,self.calendar=market_profile(profile),calendar
        if calendar.profile!=self.profile: raise ValueError('L3 calendar/profile mismatch')
        self.codes={c:j for j,c in enumerate(codes)}
        if not self.codes or len(self.codes)!=len(codes): raise ValueError('codes must be unique')
        self.account=Account(capital,len(codes))
        self.costs=costs
        self.latency=np.timedelta64(latency)
        if np.isnat(self.latency) or self.latency<np.timedelta64(0,'ns'): raise ValueError('invalid latency')
        self.public,self.orders,self.used={}, {}, {}
        self.pending,self.fills,self.events=[],[],[]
        self.now,self.day,self.seq=None,None,0
        self.unsettled=0.
        self.admission=admission
        self.source_hasher=hashlib.sha256()
        self.assets=None
        for code,(qty,price) in (initial_holdings or {}).items():
            j=self.codes[code]; qty=integer(qty,0,2**63-1,'initial holding')
            if not np.isfinite(price) or price<=0: raise ValueError('initial cost must be positive')
            self.account.shares[j]=qty
            self.account.cost_raw[j]=self.account.cost_adj[j]=self.account.last_px[j]=price

    def log(self,o,reason):
        self.events.append(dict(order_id=o['id'],timestamp=self.now,status=o['status'],reason=reason))

    def submit(self,request):
        if not isinstance(request,OrderRequest): raise TypeError('L3 order takes OrderRequest')
        if request.tif=='GTD': raise ValueError('single-session L3 uses DAY/GTC; cancel explicitly for deadlines')
        oid=len(self.orders)+1
        o=dict(id=oid,request=request,status='PENDING',remaining=request.quantity,filled=0,amount=0.,fee=0.,ahead={},active_at=None)
        self.orders[oid]=o
        j=self.codes.get(request.code)
        if j is None or not self.assets.valid_order_quantity(j,request.quantity,request.side,self.account.shares[j]) or (
                request.limit_price is not None and not self.assets.valid_order_price(j,request.limit_price)):
            o['status']='REJECTED'; self.log(o,'invalid_instrument_quantity_or_price')
        else:
            heapq.heappush(self.pending,(self.now+self.latency,oid,False))
            self.log(o,'submitted')
        return oid

    def cancel(self,oid):
        if oid not in self.orders: return False
        heapq.heappush(self.pending,(self.now+self.latency,oid,True))
        return True

    def fill(self,o,qty,price,evidence):
        req=o['request']; j=self.codes[req.code]
        qty=min(qty,o['remaining'])
        if req.side=='sell': qty=min(qty,int(self.account.shares[j]-self.account.frozen[j]-self.account.restricted[j]))
        fn=self.costs.buy if req.side=='buy' else self.costs.sell
        def fee(q):
            n=float(evaluate_cost(fn,o['amount']+q*price,self.day,codes=req.code,quantity=o['filled']+q)-
                    evaluate_cost(fn,o['amount'],self.day,codes=req.code,quantity=o['filled']))
            if n < -1e-8 or not np.isfinite(n): raise ValueError('L3 cost must have finite nonnegative increments')
            return max(n,0.)
        if req.side=='buy':
            lo,hi=0,qty
            while lo<hi:
                mid=(lo+hi+1)//2
                if mid*price+fee(mid)<=self.account.cash-self.unsettled+1e-9: lo=mid
                else: hi=mid-1
            qty=lo
        if qty<=0 or req.side=='sell' and qty*price+self.account.cash<fee(qty)-1e-9: return 0
        cost=fee(qty)
        if req.side=='buy':
            self.account.buy(j,qty,price,cost,1.)
            if self.profile.code!='CN': self.account.frozen[j]-=qty
        else:
            net=self.account.sell(j,qty,price,cost,1.)
            if self.profile.code!='CN': self.unsettled+=max(0.,net)
        o['remaining']-=qty; o['filled']+=qty; o['amount']+=qty*price; o['fee']+=cost
        o['status']='FILLED' if not o['remaining'] else 'PARTIAL'
        self.fills.append(dict(order_id=o['id'],timestamp=self.now,code=req.code,side=req.side,
            quantity=qty,price=price,fee=cost,evidence=evidence))
        self.log(o,'fill')
        return qty

    def activate(self,o):
        if o['status']!='PENDING': return
        o['status']='OPEN'; o['active_at']=self.now
        req=o['request']
        if self.admission is not None and not self.admission(self.now,req):
            o['status']='REJECTED'; self.log(o,'admission_filter_at_arrival'); return
        opposite='sell' if req.side=='buy' else 'buy'
        candidates=[(oid,p) for oid,p in self.public.items() if p['code']==req.code and p['side']==opposite and
                    (req.limit_price is None or (p['price']<=req.limit_price if req.side=='buy' else p['price']>=req.limit_price))]
        candidates.sort(key=lambda x:((x[1]['price'] if req.side=='buy' else -x[1]['price']),x[1]['priority']))
        for oid,p in candidates:
            available=max(0,p['quantity']-self.used.get(oid,0))
            filled=self.fill(o,available,p['price'],f'displayed_order:{oid}')
            self.used[oid]=self.used.get(oid,0)+filled
            if not o['remaining']: return
        if req.limit_price is None:
            o['status']='CANCELED'; self.log(o,'market_remainder_no_displayed_liquidity')
            return
        o['ahead']={oid:p['quantity'] for oid,p in self.public.items() if p['code']==req.code and
                    p['side']==req.side and p['price']==req.limit_price}
        self.log(o,'resting_displayed_FIFO')

    def process_pending(self,before,inclusive):
        while self.pending and (self.pending[0][0]<before or inclusive and self.pending[0][0]==before):
            t,oid,cancel=heapq.heappop(self.pending); self.now=t
            o=self.orders[oid]
            if cancel:
                if o['status'] in ('PENDING','OPEN','PARTIAL'):
                    o['status']='CANCELED'; self.log(o,'cancel_arrived')
            else: self.activate(o)

    def public_event(self,ev):
        oid=ev.order_id
        if ev.kind=='add':
            if oid in self.public: raise ValueError('duplicate public order id')
            self.public[oid]=dict(code=ev.code,side=ev.side,price=ev.price,quantity=ev.quantity,priority=ev.sequence)
            return
        p=self.public.get(oid)
        if p is None or any(p[k]!=getattr(ev,k) for k in ('code','side','price')) or ev.quantity>p['quantity']:
            raise ValueError('missing or inconsistent L3 order history')
        budget=ev.quantity
        if ev.kind=='execute':
            better=[q for q in self.public.values() if q['code']==ev.code and q['side']==ev.side and
                    (q['price']>ev.price if ev.side=='buy' else q['price']<ev.price)]
            earlier=[q for q in self.public.values() if q['code']==ev.code and q['side']==ev.side and
                     q['price']==ev.price and q['priority']<p['priority']]
            if better or earlier: raise ValueError('non-FIFO/hidden/auction execution unsupported')
            # Already consumed shadow liquidity cannot also drive passive fills.
            consumed=min(budget,self.used.get(oid,0)); budget-=consumed
            self.used[oid]=max(0,self.used.get(oid,0)-consumed)
            active=sorted((o for o in self.orders.values() if o['status'] in ('OPEN','PARTIAL') and
                o['request'].code==ev.code and o['request'].side==ev.side and o['request'].limit_price==ev.price),
                key=lambda o:(o['active_at'],o['id']))
            for o in active:
                if budget and not sum(o['ahead'].values()):
                    budget-=self.fill(o,budget,ev.price,f'public_execute:{ev.sequence}/{oid}')
        p['quantity']-=ev.quantity
        for o in self.orders.values():
            if oid in o['ahead']:
                o['ahead'][oid]=max(0,o['ahead'][oid]-ev.quantity)
        if p['quantity']==0:
            del self.public[oid]; self.used.pop(oid,None)


def run_l3(events,strategy,*,profile,calendar,codes,capital,costs,latency=np.timedelta64(0,'ns'),initial_holdings=None,admission=None,warmup_events=0):
    """Replay an exact, complete, single-session displayed FIFO L3 stream.

    strategy(ctx) uses ctx.order(OrderRequest(...)), ctx.cancel(id). Called after
    each public event; ctx.book contains only already observed messages.
    """
    replay=_Replay(profile,calendar,codes,capital,costs,latency,initial_holdings,admission)
    warmup_events=integer(warmup_events,0,2**63-1,'warmup_events')
    from .provenance import callable_provenance
    replay.strategy_provenance=callable_provenance(strategy)
    previous=None
    count=0
    from itertools import groupby
    # All public messages at the same timestamp have priority over arrivals.
    for _,group in groupby(events,key=lambda ev:local_timestamp(ev.timestamp,replay.profile)):
        for ev in group:
            count+=1
            if not isinstance(ev,L3Event): raise TypeError('events must be L3Event')
            t=local_timestamp(ev.timestamp,replay.profile); day=t.astype('datetime64[D]')
            if previous is not None and (t<previous or ev.sequence!=replay.seq+1):
                raise ValueError('L3 time went backwards or sequence gap')
            if ev.code not in replay.codes: raise ValueError('unknown L3 instrument')
            _,o,c,a,b=calendar.row(day)
            if not(o<=t<=c) or a is not None and a<t<b: raise ValueError('L3 outside continuous session')
            if replay.day is None:
                replay.day=day
                ds=SimpleNamespace(N=len(codes),codes=list(codes),dates=np.array([t]))
                replay.assets=(AStockRules() if replay.profile.code=='CN' else EquityRules(replay.profile)).bind(ds)
                replay.assets.day=day
            if day!=replay.day: raise ValueError('L3 is single-session; split multi-day replay explicitly')
            replay.process_pending(t,False)
            replay.now=t
            replay.public_event(ev)
            replay.source_hasher.update(json.dumps(asdict(ev),sort_keys=True,default=str).encode())
            if ev.kind=='execute':
                marks=replay.account.last_px.copy(); marks[replay.codes[ev.code]]=ev.price
                replay.account.mark(marks,advance=False)
            replay.now=t
            if count>warmup_events:
                strategy(L3Context(replay,t))
            replay.seq,previous=ev.sequence,t
            replay.account.check_cash()
        replay.process_pending(t,True)
    if previous is None: raise ValueError('empty L3 stream')
    if warmup_events>=count: raise ValueError('no live events after L3 snapshot warmup')
    # No evidence after the last event: do not invent fills or process future arrivals.
    replay.now=previous
    for order in replay.orders.values():
        if order['status'] in ('OPEN','PARTIAL','PENDING'):
            order['status']='EVIDENCE_ENDED'; replay.log(order,'end_of_available_evidence')
    return L3Result(replay)
