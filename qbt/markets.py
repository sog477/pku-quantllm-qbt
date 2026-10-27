"""Dated cash-equity market profiles, exchange sessions and settled buying power.

Internal bar timestamps are local, timezone-free labels. UTC conversion happens only
at the boundary. Each run has one currency; aggregate_results converts independent
accounts using point-in-time FX, without lending cash between accounts.
"""
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from zoneinfo import ZoneInfo
import copy
import numpy as np

from .clocks import DailyClock, BarClock
from .rules import AStockRules, CashAssetRules, CashAccountRules
from .costs import Costs


@dataclass(frozen=True)
class MarketProfile:
    code: str
    currency: str
    timezone: str
    calendar: str
    periods_per_year: int
    cash_delay: int

    def settlement_delay(self, date):
        if self.code == 'US':
            return 1 if np.datetime64(date, 'D') >= np.datetime64('2024-05-28') else 2
        return self.cash_delay

    def sessions(self, start, end):
        """Optional exchange_calendars adapter; never extrapolate outside its coverage."""
        import exchange_calendars as xc
        cal = xc.get_calendar(self.calendar, start=str(start), end=str(end))
        return SessionCalendar.from_exchange(cal.schedule, self, f'exchange_calendars {xc.__version__}/{self.calendar}')


PROFILES = {
    'CN': MarketProfile('CN', 'CNY', 'Asia/Shanghai', 'XSHG', 242, 1),
    'US': MarketProfile('US', 'USD', 'America/New_York', 'XNYS', 252, 1),
    'KR': MarketProfile('KR', 'KRW', 'Asia/Seoul', 'XKRX', 250, 2),
}


def market_profile(value):
    if isinstance(value, MarketProfile):
        return value
    try:
        return PROFILES[str(value).upper()]
    except KeyError:
        raise ValueError('market_profile must be CN, US or KR') from None


def local_timestamp(value, profile):
    """Aware input -> market wall time; naive input explicitly means local time."""
    if isinstance(value, np.datetime64):
        result = value.astype('datetime64[ns]')
    else:
        d = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if d.tzinfo is not None:
            d = d.astimezone(ZoneInfo(market_profile(profile).timezone)).replace(tzinfo=None)
        result = np.datetime64(d, 'ns')
    if np.isnat(result):
        raise ValueError('timestamp cannot be NaT')
    return result


class SessionCalendar:
    """Explicit exchange sessions, including closures, lunch and early/special hours."""
    def __init__(self, rows, profile, source='user supplied', settlement_days=None):
        self.profile, self.source = market_profile(profile), str(source)
        self.rows = []
        for row in rows:
            day = np.datetime64(row['date'], 'D')
            o, c = [local_timestamp(row[k], self.profile) for k in ('open', 'close')]
            a, b = [None if not row.get(k) else local_timestamp(row[k], self.profile)
                    for k in ('break_start', 'break_end')]
            if (np.isnat(day) or o >= c or o.astype('datetime64[D]') != day or
                    c.astype('datetime64[D]') != day or (a is None) != (b is None) or
                    (a is not None and not (o < a < b < c))):
                raise ValueError('invalid session hours or lunch break')
            if self.rows and day <= self.rows[-1][0]:
                raise ValueError('session dates must be unique and increasing')
            self.rows.append((day, o, c, a, b))
        if not self.rows:
            raise ValueError('empty exchange calendar')
        self.days = np.array([r[0] for r in self.rows])
        self.days.setflags(write=False)
        self.settlement_days=np.array(self.days if settlement_days is None else settlement_days,dtype='datetime64[D]')
        if self.settlement_days.ndim!=1 or not len(self.settlement_days) or np.isnat(self.settlement_days).any() or (np.diff(self.settlement_days)<=np.timedelta64(0,'D')).any():
            raise ValueError('settlement calendar must be increasing and valid')
        self.settlement_source='trading-session proxy' if settlement_days is None else 'explicit settlement sessions'

    @classmethod
    def from_exchange(cls, schedule, profile, source):
        import pandas as pd
        rows = []
        for day, r in schedule.iterrows():
            row = {'date': str(day.date())}
            for k in ('open', 'close', 'break_start', 'break_end'):
                row[k] = None if pd.isna(r[k]) else r[k].isoformat()
            rows.append(row)
        return cls(rows, profile, source)

    def row(self, day):
        day = np.datetime64(day, 'D')
        i = int(np.searchsorted(self.days, day))
        if i == len(self.days) or self.days[i] != day:
            raise ValueError(f'{day} is not a session in {self.source}')
        return self.rows[i]

    def validate(self, ds, complete=True):
        days = ds.dates.astype('datetime64[D]')
        observed = np.unique(days)
        if complete:
            expected = self.days[(self.days >= days[0]) & (self.days <= days[-1])]
            if not np.array_equal(expected, observed):
                raise ValueError('missing or unexpected exchange sessions in market data')
        for t in ds.dates:
            _, o, c, a, b = self.row(t)
            if getattr(ds, 'intraday', False) and not (o <= t <= c and not (a is not None and a < t < b)):
                raise ValueError(f'bar {t} outside exchange session')

    def due(self, day, delay):
        self.row(day)
        if delay==0: return np.datetime64(day,'D')
        p=int(np.searchsorted(self.settlement_days,np.datetime64(day,'D'),side='right'))+delay-1
        return self.settlement_days[p] if p<len(self.settlement_days) else None

    def utc_close(self, day):
        t = self.row(day)[2]
        local = datetime.fromisoformat(str(t)).replace(tzinfo=ZoneInfo(self.profile.timezone))
        from datetime import timezone
        return np.datetime64(local.astimezone(timezone.utc).replace(tzinfo=None), 'ns')


class SessionClock(BarClock):
    def __init__(self, calendar, intraday=False, periods_per_year=None):
        if intraday and periods_per_year is None:
            raise ValueError('intraday reference data requires explicit periods_per_year')
        super().__init__(calendar.profile.periods_per_year if periods_per_year is None else periods_per_year)
        self.calendar, self.intraday = calendar, intraday

    def events(self, ds, start=None, end=None):
        self.calendar.validate(ds)
        return super().events(ds, start, end)


class SettledCashRules(CashAccountRules):
    """US/KR conservative cash account: only settled proceeds finance purchases.

    Cash in the ledger includes unsettled sale claims; pending is excluded from
    buying power and withdrawal. No margin, good-faith-credit or implicit FX.
    CN sale proceeds remain reusable for buying on trade date.
    """
    name = 'settled_cash'

    def __init__(self, profile, calendar):
        self.profile, self.calendar = market_profile(profile), calendar
        self.pending, self.day = [], None

    def reset(self):
        self.pending,self.day=[],None

    def start_bar(self, account, event, asset_rules):
        self.day = np.datetime64(event.session if event.session is not None else event.timestamp, 'D')
        if hasattr(asset_rules,'day'):
            asset_rules.day=self.day
        self.pending = [(d, amount) for d, amount in self.pending if d is None or d > self.day]
        super().start_bar(account, event, asset_rules)

    def unsettled(self):
        return sum(amount for _, amount in self.pending)

    def buying_power(self, account):
        return max(0., account.cash-account.tax_payables.sum() - (0. if self.profile.code == 'CN' else self.unsettled()))

    def withdrawable(self, account):
        return max(0., account.cash-account.tax_payables.sum()-self.unsettled())

    def sell(self, account, j, qty, price, fee, factor, asset_rules):
        if self.day is None:
            raise ValueError('cash settlement requires an active session')
        due = self.calendar.due(self.day, self.profile.settlement_delay(self.day))
        net = account.sell(j, qty, price, fee, factor)
        self.pending.append((due, max(0., net)))


class EquityRules(CashAssetRules):
    """US common stocks and post-2023 KRX common stocks; dated/custom tick overrides."""
    def __init__(self, profile, boards=None, ticks=None):
        self.profile = market_profile(profile)
        if self.profile.code not in ('US','KR'):
            raise ValueError('custom markets require explicit asset_rules; no foreign tick regime is guessed')
        self.boards, self.tick_override = boards, ticks
        super().__init__(settlement_days=0, price_limits=self.profile.code == 'KR')

    def bind(self, ds):
        obj = super().bind(ds)
        obj.day = ds.dates[0].astype('datetime64[D]')
        obj.boards_bound = np.broadcast_to(np.asarray('KOSPI' if self.boards is None else self.boards), (ds.N,)).copy()
        if self.profile.code == 'KR' and not np.isin(obj.boards_bound, ['KOSPI', 'KOSDAQ']).all():
            raise ValueError('KR boards must be KOSPI/KOSDAQ common equities')
        return obj

    def tick(self, j, price):
        if self.tick_override is not None:
            return float(self.tick_override(self.day, j, price))
        if self.profile.code == 'US':
            if self.day>=np.datetime64('2027-11-01'):
                raise ValueError('US post-2027 tick regime requires an explicit dated per-symbol tick rule')
            return .01 if price >= 1 else .0001
        if self.day < np.datetime64('2023-01-25'):
            raise ValueError('pre-2023 KRX ticks require a dated tick override')
        # Common-equity tick tiers, KRX 2023-01-25 reform.
        bounds, sizes = [2000,5000,20000,50000,200000,500000], [1,5,10,50,100,500,1000]
        return sizes[int(np.searchsorted(bounds, price, side='right'))]

    def valid_order_price(self, j, price):
        tick = self.tick(j, price)
        if not np.isfinite(tick) or tick <= 0:
            raise ValueError('invalid tick rule')
        return bool(np.isclose(price/tick, np.rint(price/tick), rtol=0, atol=1e-7))

    def can_buy(self,market,i,rule):
        if self.profile.code!='US': return super().can_buy(market,i,rule)
        p=market.ds.open[i]
        # LULD bounds/halts, when supplied, do not imply an A-share queue lock
        # merely because a quote is equal to the band.
        return ~market.quote_suspended[i] & (p>=market.limit_down[i]) & (p<=market.limit_up[i])

    def can_sell(self,market,i,rule):
        return self.can_buy(market,i,rule) if self.profile.code=='US' else super().can_sell(market,i,rule)


class ScheduledCosts:
    """Explicit dated per-market fee schedule; rates are in local currency.

    commission/minimum are broker-specific; sell_tax includes statutory charges
    that are proportional to notional. Per-share/venue charges can use a custom
    Costs component. Coverage never extends silently beyond effective_until.
    """
    def __init__(self, rows):
        self.rows = tuple(dict(r) for r in rows)
        if not self.rows:
            raise ValueError('fee schedule is required')
        previous_end = None
        for r in self.rows:
            allowed={'effective_from','effective_until','commission','minimum','buy_tax','sell_tax','slippage_bps',
                     'buy_per_share','sell_per_share','buy_per_share_cap','sell_per_share_cap','source','description'}
            if set(r)-allowed: raise ValueError(f'unknown fee fields: {sorted(set(r)-allowed)}')
            a,b = np.datetime64(r['effective_from'],'D'),np.datetime64(r['effective_until'],'D')
            if np.isnat(a) or np.isnat(b) or a>b or previous_end is not None and a<=previous_end:
                raise ValueError('fee schedule must be ordered and non-overlapping')
            previous_end=b
            for k in ('commission','minimum','buy_tax','sell_tax','slippage_bps','buy_per_share','sell_per_share','buy_per_share_cap','sell_per_share_cap'):
                v=float(r.get(k,0.))
                if not np.isfinite(v) or v<0:
                    raise ValueError('invalid fee schedule')

    def _fee(self, amount, day, side, quantity=None):
        day=np.datetime64(day,'D')
        matches=[r for r in self.rows if np.datetime64(r['effective_from'],'D')<=day<=np.datetime64(r['effective_until'],'D')]
        if len(matches)!=1:
            raise ValueError(f'fee schedule does not cover {day}')
        r=matches[0]; a=np.abs(np.asarray(amount,float))
        out=np.maximum(a*float(r.get('commission',0)),float(r.get('minimum',0)))
        out+=a*(float(r.get(side+'_tax',0))+float(r.get('slippage_bps',0))*1e-4)
        per_share=float(r.get(side+'_per_share',0))
        if per_share:
            if quantity is None: raise ValueError('per-share fees require execution quantity')
            q=np.asarray(quantity,float)
            if not np.isfinite(q).all() or (q<0).any() or (q!=np.floor(q)).any(): raise ValueError('invalid fee quantity')
            charge=q*per_share
            if side+'_per_share_cap' in r: charge=np.minimum(charge,float(r[side+'_per_share_cap']))
            out+=charge
        return np.where(a>0,out,0.)

    def buy(self, amount, day=None, *, quantity=None): return self._fee(amount,day,'buy',quantity)
    def sell(self, amount, day=None, *, quantity=None): return self._fee(amount,day,'sell',quantity)
    buy.supports_quantity=True
    sell.supports_quantity=True
    def __repr__(self): return f'ScheduledCosts({self.rows!r})'


def prepare_market_options(ds, options):
    kw=dict(options)
    profile=kw.pop('market_profile',None)
    reference=kw.pop('reference_data',None)
    if isinstance(kw.get('market'),str):
        if profile is not None:
            raise ValueError('provide market or market_profile, not both')
        profile=kw.pop('market')
    if profile is None and reference is None:
        return ds,kw,None,None,None
    profile=market_profile(reference.profile if profile is None else profile)
    if reference is None:
        raise ValueError('market switching requires ReferenceData with sessions and fee schedule; see examples/market_switch.py')
    if kw.get('market') is not None or kw.get('market_kw') is not None:
        raise ValueError('ReferenceData owns the market metadata; do not combine with market/market_kw')
    prepared, market, actions, calendar = reference.prepare(ds,profile)
    kw['market']=market
    if kw.get('clock') is None:
        kw['clock']=SessionClock(calendar,getattr(prepared,'intraday',False),reference.periods_per_year)
    if kw.get('asset_rules') is None:
        kw['asset_rules']=AStockRules() if profile.code=='CN' else EquityRules(profile,reference.boards,reference.tick_rule)
    if kw.get('account_rules') is None:
        kw['account_rules']=SettledCashRules(profile,calendar)
    if kw.get('costs') is None:
        kw['costs']=Costs() if profile.code=='CN' and not reference.fees else ScheduledCosts(reference.fees)
    if kw.get('corporate_actions') is not None:
        raise ValueError('supply rights through ReferenceData to keep signals and ledger consistent')
    if kw.get('div_policy') not in (None,'events'):
        raise ValueError('ReferenceData uses explicit equity events; use a separately declared reference scenario for counterfactual policies')
    kw['corporate_actions']=actions
    kw['div_policy']='events'
    return prepared,kw,profile,reference,calendar


def market_entry(function):
    """Opt-in adapter shared by both existing engines."""
    @wraps(function)
    def wrapped(ds,*args,**kw):
        prepared,kw,profile,reference,calendar=prepare_market_options(ds,kw)
        result=function(prepared,*args,**kw)
        from .provenance import callable_provenance
        result.execution_provenance['market_adapter']=callable_provenance(wrapped)
        if profile is None:
            return result
        rules=kw['account_rules']
        result.market_profile,result.currency=profile.code,profile.currency
        result.timezone,result.session_calendar=profile.timezone,calendar
        result.unsettled_cash=rules.unsettled() if isinstance(rules,SettledCashRules) else None
        result.withdrawable_cash=rules.withdrawable(result.account) if isinstance(rules,SettledCashRules) else None
        result.reference_provenance=reference.provenance()
        result.input_provenance=getattr(prepared,'source_provenance',{'source':'in-memory arrays; input snapshot not attached'})
        result.factor_diagnostics=prepared.factor_diagnostics
        result.assumptions.extend(reference.assumptions)
        result.assumptions.append(f'{profile.code} / {profile.currency} / {profile.timezone}; {calendar.source}; conservative settled-cash account')
        return result
    return wrapped


def from_market_arrays(profile, timestamps, codes, *, intraday=False, **panels):
    """Explicit timezone normalization before constructing the normal data object."""
    from .data import Dataset
    from .bars import BarDataset
    times=np.array([local_timestamp(t,profile) for t in timestamps])
    if not intraday:
        times=times.astype('datetime64[D]')
    return (BarDataset if intraday else Dataset).from_arrays(times,codes,**panels)
