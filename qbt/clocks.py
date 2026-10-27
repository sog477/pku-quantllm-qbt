"""行情行与交易会话分离。回调只能看见本事件之前已经完成的行。"""
from dataclasses import dataclass
import numpy as np
from ._validation import boolean_mask


@dataclass(frozen=True)
class ClockEvent:
    index: int
    timestamp: object
    previous_timestamp: object
    new_session: bool
    session: object = None

    def __post_init__(self):
        flag = boolean_mask(self.new_session,'ClockEvent.new_session',())
        object.__setattr__(self,'new_session',bool(flag))


class BarClock:
    def __init__(self, periods_per_year, stride=1):
        if not np.isfinite(periods_per_year) or periods_per_year <= 0 or int(stride) != stride or stride < 1:
            raise ValueError("periods_per_year 必须为正，stride 必须为正整数")
        self.periods_per_year, self.stride = float(periods_per_year), int(stride)

    def session_key(self, timestamp):
        return timestamp.astype("datetime64[D]")

    def is_session_end(self, ds, i):
        return i+1 == ds.T or self.session_key(ds.dates[i+1]) != self.session_key(ds.dates[i])

    def events(self, ds, start=None, end=None):
        a, b = ds.calendar.slice_of(start, end)
        previous = ds.dates[max(0, a-1)]
        for i in range(max(1, a), b, self.stride):
            timestamp = ds.dates[i]
            yield ClockEvent(i, timestamp, ds.dates[i-1],
                self.session_key(timestamp) != self.session_key(previous), self.session_key(timestamp))
            previous = timestamp


class DailyClock(BarClock):
    def __init__(self, periods_per_year=242):
        super().__init__(periods_per_year)

    def events(self, ds, start=None, end=None):
        days = ds.dates.astype("datetime64[D]")
        if len(np.unique(days)) != len(days):
            raise ValueError("DailyClock 不接受日内多行数据，请显式提供 BarClock 和年化频率")
        return super().events(ds, start, end)


def clock_events(clock, ds, start, end):
    if not np.isfinite(clock.periods_per_year) or clock.periods_per_year <= 0:
        raise ValueError("时钟 periods_per_year 必须是正的有限数")
    events = tuple(clock.events(ds, start, end))
    indices = np.array([e.index for e in events], dtype=int)
    if not events or indices.min() < 1 or indices.max() >= ds.T or (np.diff(indices) <= 0).any():
        raise ValueError("时钟必须产生非空、递增且有前一行历史的合法行情行")
    for e in events:
        boolean_mask(e.new_session,'clock.new_session',())
        if e.timestamp != ds.dates[e.index]:
            raise ValueError("时钟时间戳必须与行情行一致")
        if e.previous_timestamp != ds.dates[e.index-1]:
            raise ValueError("回调时点必须是执行行之前最近已完成的行情行")
    return events, indices


def benchmark_at_steps(ds, indices, market=None):
    from .engine import ew_market_returns
    adjusted = None
    if market is not None and market.stats.get('quote_time_source') == 'explicit' and not hasattr(ds,'total_return_close'):
        observed = getattr(market,'observed_price',None)
        if callable(observed):
            adjusted = observed('close',True)*ds.cumadj_ff
    r = np.nan_to_num(ew_market_returns(ds,adjusted), nan=0.)
    nav = np.cumprod(1+r)
    selected = nav[np.r_[indices[0]-1, indices]]
    selected = selected / selected[0]
    return selected[1:]/selected[:-1]-1, selected


def row_event(clock, ds, i):
    """底层市场行的会话边界；稀疏策略时钟仍须推进中间账户状态。"""
    if not callable(getattr(clock,'session_key',None)):
        raise ValueError('稀疏时钟须提供session_key(timestamp)，才能推进中间行的结算会话')
    current, previous = ds.dates[i], ds.dates[i-1]
    session = clock.session_key(current)
    return ClockEvent(i,current,previous,session != clock.session_key(previous),session)
