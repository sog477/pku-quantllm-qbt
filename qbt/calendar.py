"""交易日历。

回测里所有"时间"都必须走这里，不允许直接拿自然日做加减——A 股一年只有
约 242 个交易日，用自然日推 T+1 会错得离谱（春节、国庆一停就是一周多）。

日历本身就是数据里的 date 列，所以它和行情永远是一致的，不存在"日历和
数据对不上"这类问题。
"""

from __future__ import annotations

import numpy as np
from ._validation import naive_datetime_scalar

__all__ = ["Calendar"]


class Calendar:
    """一条不可变的交易日序列。"""

    __slots__ = ("days", "_pos")

    def __init__(self, days):
        d = np.asarray(days)
        if d.dtype.kind == "M":
            d = d.astype("datetime64[D]")
        elif d.dtype.kind in "iu":
            d = d.astype("datetime64[ns]").astype("datetime64[D]")
        else:
            raise TypeError(f"不认识的日期类型: {d.dtype}")
        if d.ndim != 1 or np.isnat(d).any() or (np.diff(d) <= np.timedelta64(0,'D')).any():
            raise ValueError("日历必须是一维、无NaT、严格递增且不重复的交易日序列")
        d.setflags(write=False)
        self.days = d
        self._pos = {int(x): i for i, x in enumerate(d.astype("int64"))}

    # -- 基本信息 --------------------------------------------------------
    def __len__(self):
        return len(self.days)

    def __repr__(self):
        if not len(self.days):
            return "Calendar(空)"
        return f"Calendar({self.days[0]} .. {self.days[-1]}, {len(self.days)} 个交易日)"

    @property
    def first(self):
        return self.days[0]

    @property
    def last(self):
        return self.days[-1]

    # -- 定位 ------------------------------------------------------------
    def index(self, day, side="left"):
        """把任意日期换算成下标。

        side="left"  —— 取 **不晚于** day 的最后一个交易日（默认）。
                        传周末/节假日会自然落到上一个交易日，这是回测里
                        最常用的语义（"到 2021-01-01 为止"）。day 本身是
                        交易日时返回它自己。
        side="right" —— 取 **不早于** day 的第一个交易日。
        side="exact" —— 必须是交易日，否则报错。

        注意这是 **闭区间** 语义：index("2021-01-04") 就是 2021-01-04 那一行
        （它确实是交易日的话），不是它前一天。差一位是回测里最隐蔽的 bug 之一
        ——净值曲线照样画得出来，只是每一天都错位一格。
        """
        d = naive_datetime_scalar(day,'查询日期').astype('datetime64[D]')
        if np.isnat(d):
            raise ValueError('查询日期不能是NaT')
        if side not in ('left','right','exact'):
            raise ValueError(f'side只能是left/right/exact，收到{side!r}')
        if not len(self.days):
            raise KeyError('空日历没有可定位的交易日')
        key = int(d.astype("int64"))
        if side == "exact":
            i = self._pos.get(key)
            if i is None:
                raise KeyError(f"{d} 不是交易日")
            return i
        if side == "left":                       # 最后一个 <= day
            i = np.searchsorted(self.days.astype("int64"), key, side="right") - 1
            if i < 0:
                raise KeyError(f"{d} 早于数据起点 {self.days[0]}")
        elif side == "right":                    # 第一个 >= day
            i = np.searchsorted(self.days.astype("int64"), key, side="left")
            if i >= len(self.days):
                raise KeyError(f"{d} 晚于数据终点 {self.days[-1]}")
        else:
            raise ValueError(f'side 只能是 left/right/exact，收到 {side!r}')
        return int(i)

    def clamp(self, day, default=None):
        """index() 的宽松版本：落在区间外就返回 None（或 default）。"""
        try:
            return self.index(day)
        except (KeyError, TypeError, ValueError):
            return default

    # -- 位移 ------------------------------------------------------------
    def shift(self, i, k):
        """从第 i 个交易日往前/后推 k 个交易日。越界返回 None。"""
        j = i + k
        return j if 0 <= j < len(self.days) else None

    def days_between(self, i, j):
        """第 i 到第 j 个交易日之间隔了几个交易日（j-i）。"""
        return j - i

    def year_frac(self, i, j):
        """两个交易日之间的年化因子，用 242 交易日/年 折算。"""
        return abs(j - i) / 242.0

    # -- 切片 ------------------------------------------------------------
    def slice_of(self, start=None, end=None):
        """把 (start, end) 变成半开区间 [a, b)，两端都是闭区间语义。

        `start`/`end` 可以落在非交易日上，会自动吸附到区间内的第一个/最后一个
        交易日；整个区间没有任何交易日时返回空区间 (a == b)。
        """
        a = 0
        if start is not None:
            try:
                a = self.index(start, side="right")
            except KeyError:
                a = len(self.days)
        b = len(self.days)
        if end is not None:
            try:
                b = self.index(end, side="left") + 1
            except KeyError:
                b = 0
        if b < a:
            b = a
        return a, b

    def is_month_end(self, *, final_month_complete=False):
        """观测到跨月的最后交易日；截断样本的尾行默认不猜月末。

        尾行恰为自然月末时可以确认；节假日前的最后交易日需要完整的
        交易所日历，调用者确认最后一个月完整后可传 final_month_complete=True。
        中间月份以输入交易日序列完整为前提。
        """
        m = self.days.astype("datetime64[M]")
        out = np.zeros(len(m), dtype=bool)
        if len(m) > 1:
            out[:-1] = m[:-1] != m[1:]
        if len(out):
            next_day = self.days[-1] + np.timedelta64(1, 'D')
            out[-1] = bool(final_month_complete) or next_day.astype('datetime64[M]') != m[-1]
        return out

    def month_labels(self):
        """月度标签，形如 ['2020-01', '2020-02', ...]。"""
        d = self.days.astype("datetime64[D]").astype(object)
        return [f"{x.year:04d}-{x.month:02d}" for x in d]

    def tolist(self):
        return [str(x) for x in self.days]
