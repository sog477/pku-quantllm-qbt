"""时间戳型 OHLC 面板，支持用户提供分钟/小时数据，不虚构日线内部路径。"""
import numpy as np
from .data import Dataset, FILES, _validate_panel, _numeric_panel
from ._validation import naive_datetime_input, naive_datetime_scalar


class BarCalendar:
    def __init__(self, timestamps):
        d=np.asarray(timestamps,dtype='datetime64[ns]').copy()
        if d.ndim != 1 or np.isnat(d).any() or (np.diff(d)<=np.timedelta64(0,'ns')).any():
            raise ValueError('日内日历须严格递增且无NaT')
        d.setflags(write=False)
        self.days=d

    def __len__(self):
        return len(self.days)

    def index(self, value, side="left"):
        original = naive_datetime_scalar(value,'查询时刻',keep_unit=True)
        t = original.astype('datetime64[ns]')
        if np.isnat(t):
            raise ValueError('查询时刻不能是NaT')
        if side not in ('left','right','exact'):
            raise ValueError('side只能是left/right/exact')
        if side == "exact":
            pos = int(np.searchsorted(self.days, t))
            # 日期形式的公司行动对应该日第一根 bar。
            is_day = original.dtype == np.dtype("datetime64[D]")
            if pos < len(self.days) and (self.days[pos] == t or is_day and self.days[pos].astype("datetime64[D]") == t.astype("datetime64[D]")):
                return pos
            raise KeyError(f"不存在时间戳：{value}")
        pos = int(np.searchsorted(self.days, t, side="right")-1) if side == "left" else int(np.searchsorted(self.days, t))
        if side not in ("left", "right") or pos < 0 or pos >= len(self.days):
            raise KeyError(value)
        return pos

    def slice_of(self, start=None, end=None):
        start = None if start is None else naive_datetime_scalar(start,'start')
        if start is not None and np.isnat(start):
            raise ValueError('start不能是NaT')
        a = 0 if start is None else int(np.searchsorted(self.days, np.datetime64(start, "ns")))
        if end is None:
            b = len(self.days)
        else:
            t = naive_datetime_scalar(end,'end',keep_unit=True)
            if np.isnat(t):
                raise ValueError('end不能是NaT')
            if t.dtype == np.dtype("datetime64[D]"):
                b = int(np.searchsorted(self.days, (t+np.timedelta64(1,"D")).astype("datetime64[ns]")))
            else:
                b = int(np.searchsorted(self.days, t.astype("datetime64[ns]"), side="right"))
        return a, max(a, b)


class BarDataset(Dataset):
    @classmethod
    def from_arrays(cls, timestamps, codes, previous_close=None, **panels):
        timestamps = naive_datetime_input(timestamps, 'timestamps').copy()
        if timestamps.ndim != 1 or len(timestamps) < 2 or np.isnat(timestamps).any() or (np.diff(timestamps) <= np.timedelta64(0,"ns")).any():
            raise ValueError("时间戳须严格递增且无 NaT")
        if not len(codes) or len(set(codes)) != len(codes) or set(panels) != set(FILES):
            raise ValueError("代码须唯一且提供全部七面板")
        obj = object.__new__(cls)
        obj.cache_dir, obj._mmap, obj.verbose = None, False, False
        obj.dates, obj.codes, obj.codes_arr = timestamps, list(codes), np.asarray(codes).copy()
        obj.T, obj.N = len(timestamps), len(codes)
        obj.calendar, obj._raw, obj._c = BarCalendar(timestamps), {}, {}
        obj.previous_close = (np.full(obj.N,np.nan) if previous_close is None else
                              np.broadcast_to(np.asarray(previous_close,float),(obj.N,)).copy())
        if np.isinf(obj.previous_close).any() or (obj.previous_close[np.isfinite(obj.previous_close)] <= 0).any():
            raise ValueError("previous_close 必须为正数或NaN")
        obj.intraday = True
        for name, values in panels.items():
            a = _numeric_panel(values, name)
            if a.shape != (obj.T, obj.N):
                raise ValueError(f"{name} 形状或数值无效")
            _validate_panel(name, a)
            obj._raw[name] = a.copy()
        return obj

    @property
    def reference_prices(self):
        return self.cached("session_reference",lambda d:d._reference_prices_from(d.close_ff))

    def _reference_prices_from(self, close_ff):
        days = self.dates.astype("datetime64[D]")
        starts = np.r_[0,np.flatnonzero(days[1:]!=days[:-1])+1]
        ends = np.r_[starts[1:],self.T]
        ref = np.full((self.T,self.N),np.nan)
        for a,b in zip(starts,ends):
            close = self.previous_close if a==0 else close_ff[a-1]
            factor = self.cumadj_ff[0] if a==0 else self.cumadj_ff[a-1]
            ref[a:b] = close*factor/self.cumadj_ff[a:b]
        return ref

    @property
    def list_age(self):
        # 日内行数不能当上市天数；按观测到的会话计数，仍只是报价年龄代理。
        def build(d):
            valid = np.isfinite(d.close) & (d.close > 0)
            session = np.r_[0, np.cumsum(np.diff(d.dates.astype("datetime64[D]")) != np.timedelta64(0,"D"))]
            first = np.argmax(valid, axis=0)
            age = session[:,None]-session[first][None,:]
            age[~np.maximum.accumulate(valid,axis=0)] = -1
            age[:,valid[0]] = 32767
            return age
        return self.cached("bar_age", build)

    def __reduce__(self):
        return (_restore_bars, (self.dates, self.codes, self.previous_close, self._raw))


def _restore_bars(dates, codes, previous_close, panels):
    return BarDataset.from_arrays(dates, codes, previous_close=previous_close, **panels)
