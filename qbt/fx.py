"""As-of FX reporting for independently funded market accounts."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FXQuote:
    currency: str
    timestamp: object
    published_at: object
    rate: float


class FXRates:
    """Rates are BASE units per one foreign unit; times are explicit UTC instants."""
    def __init__(self, base, quotes, max_age=np.timedelta64(3,'D')):
        from datetime import datetime, timezone
        def utc(v):
            if isinstance(v,np.datetime64):
                raise ValueError('FX times require timezone-aware ISO strings, not naive datetime64')
            d=v if isinstance(v,datetime) else datetime.fromisoformat(str(v).replace('Z','+00:00'))
            if d.tzinfo is None or d.utcoffset() is None: raise ValueError('FX time must include timezone')
            return np.datetime64(d.astimezone(timezone.utc).replace(tzinfo=None),'ns')
        self.base,self.quotes,self.max_age=str(base),{},np.timedelta64(max_age)
        if np.isnat(self.max_age) or self.max_age<np.timedelta64(0,'ns'): raise ValueError('invalid FX age')
        seen=set()
        for q in quotes:
            t,p=utc(q.timestamp),utc(q.published_at)
            if not np.isfinite(q.rate) or q.rate<=0: raise ValueError('invalid FX rate')
            key=(q.currency,t,p)
            if key in seen: raise ValueError('duplicate FX revision')
            seen.add(key)
            self.quotes.setdefault(q.currency,[]).append((t,p,float(q.rate)))

    def at(self, currency, utc_time):
        if currency==self.base: return 1.
        t=np.datetime64(utc_time,'ns')
        eligible=[q for q in self.quotes.get(currency,()) if q[0]<=t and q[1]<=t and t-q[0]<=self.max_age]
        if not eligible: raise ValueError(f'no timely published FX quote for {currency} at {t}')
        return max(eligible,key=lambda q:(q[0],q[1]))[2]


def aggregate_results(results, fx, valuation_times):
    """As-of UTC portfolio NAV. Missing first mark/FX is an error, never backfilled.

    Accounts are independently funded. No cross-account buying power or automatic
    reinvestment of stale proceeds is implied by this reporting aggregation.
    """
    times=np.array(valuation_times,dtype='datetime64[ns]')
    if times.ndim!=1 or np.isnat(times).any() or (np.diff(times)<=np.timedelta64(0,'ns')).any():
        raise ValueError('UTC valuation times must be increasing and valid')
    values=np.zeros(len(times)); components={}
    for name,r in results.items():
        if not hasattr(r,'session_calendar'): raise ValueError('aggregation requires a market-profile result')
        from datetime import datetime,timezone
        from zoneinfo import ZoneInfo
        def instant(v):
            if not getattr(r.ds,'intraday',False): return r.session_calendar.utc_close(v)
            d=datetime.fromisoformat(str(v)).replace(tzinfo=ZoneInfo(r.timezone))
            return np.datetime64(d.astimezone(timezone.utc).replace(tzinfo=None),'ns')
        observed=np.array([instant(t) for t in r.dates])
        p=np.searchsorted(observed,times,side='right')-1
        if (p<0).any(): raise ValueError('valuation precedes first available account mark')
        v=np.array([r.equity[j]*fx.at(r.currency,t) for j,t in zip(p,times)])
        components[name]=v; values+=v
    return dict(times=times,equity=values,currency=fx.base,components=components,
                assumptions=['independently funded accounts; latest completed local mark; as-of published FX'])
