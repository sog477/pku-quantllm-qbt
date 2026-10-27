"""Universe：选股域。

老师问过"什么因子类策略怎么选你的 universe？"——这一步做错的策略，
回测再好看也是假的。两个最常见的错法：

**错法一：把"选股"和"过滤"写成 `bottom(score, 100) & u`。**
这是在 5451 只里按分数选出 100 只，**然后**拿 universe 去筛，实际剩下的
可能只有 30~60 只，而且选中的是"分数够低**且**恰好可交易"的那批 ——
决定权落在可交易性上，不落在信号上。正确写法是先筛后选：
`u.bottom(score, 100)`。

**错法二：不看换手。**
等权目标价每天随价格漂移，如果每天照着重置目标权重，就是每天全仓换手，
佣金印花税直接把 alpha 吃光。所以要么用 `buffer()` 做进出缓冲，要么让
引擎按漂移带调仓（默认就是这么做的）。

所有运算都用 ≤ t 的行，天然无前视。
"""

from __future__ import annotations

import numpy as np
from ._validation import boolean_mask

__all__ = ["Universe", "select_by_rank"]


def _topk(valid, score, k, largest):
    """按日选出 score 最大/最小的 k 只；(T,N) bool。

    逐日做 argpartition（O(N)）而不是整体 argsort（O(N log N) 且要开一个
    (T,N) int64 的排序结果，71MB）。1643 天 × 5451 只大约 0.15s。

    先显式屏蔽无效位置：`np.argsort` 会把 NaN 排到最后，不屏蔽的话
    `top(score, 100)` 会认真地选出 100 个 NaN。
    """
    T, N = score.shape
    out = np.zeros((T, N), dtype=bool)
    if k <= 0:
        return out
    for t in range(T):
        idx = np.flatnonzero(valid[t])
        if idx.size == 0:
            continue
        if idx.size <= k:
            out[t, idx] = True
            continue
        v = score[t, idx]
        part = np.argpartition(-v if largest else v, k - 1)[:k]
        out[t, idx[part]] = True
    return out


def _topk_per_day(valid, s, keep_n, largest=True):
    """每日保留数量不同的版本（keep_n 是 (T,1) int）。"""
    T, N = s.shape
    out = np.zeros((T, N), dtype=bool)
    for t in range(T):
        idx = np.flatnonzero(valid[t])
        n = int(keep_n[t, 0])
        if idx.size == 0 or n <= 0:
            continue
        if idx.size <= n:
            out[t, idx] = True
            continue
        v = s[t, idx]
        out[t, idx[np.argpartition(-v if largest else v, n - 1)[:n]]] = True
    return out


def _keep_top_n(valid, s, keep_n):
    """每日保留分数最高的 keep_n[t] 只（keep_n 是 (T,1)）。"""
    return _topk_per_day(valid, s, keep_n, largest=True)


def select_by_rank(score, mask, k, largest=True):
    """在 mask 内按 score 取前 k 名。"""
    s = np.asarray(score, dtype=np.float64)
    valid = boolean_mask(mask) & np.isfinite(s)
    return _topk(valid, s, k, largest)


class Universe:
    """逐日选股结果。`mask` 是想持有的，`keep` 是允许继续持有的。

    `keep` 只有用了 `buffer()` 才和 `mask` 不同。
    """

    __slots__ = ("mask", "keep", "note", "ctx")

    def __init__(self, mask, keep=None, note="", ctx=None):
        m = boolean_mask(mask)
        self.mask = m
        self.keep = m if keep is None else boolean_mask(keep, 'keep')
        self.note = note
        self.ctx = ctx

    # -- 基本信息 --------------------------------------------------------
    @property
    def shape(self):
        return self.mask.shape

    def __len__(self):
        return int(self.mask.sum())

    def __repr__(self):
        per_day = self.mask.sum(axis=1)
        return (f"Universe({self.mask.shape[0]} 日 × {self.mask.shape[1]} 只, "
                f"日均持有 {per_day.mean():.0f} 只"
                + (f", 最小 {per_day.min()} 最大 {per_day.max()}" if per_day.size else "")
                + (f", {self.note}" if self.note else "") + ")")

    def __and__(self, other):
        o = other.mask if isinstance(other, Universe) else boolean_mask(other)
        return Universe(self.mask & o, self.keep & o, ctx=self.ctx)

    def __or__(self, other):
        o = other.mask if isinstance(other, Universe) else boolean_mask(other)
        return Universe(self.mask | o, self.keep | o, ctx=self.ctx)

    def __invert__(self):
        return Universe(~self.mask, ~self.keep, ctx=self.ctx)

    # -- 相对选股（在 universe 内部选）------------------------------------
    def top(self, score, k, note=None):
        """在**当前 universe 内部**按 score 取前 k 名（score 越大越好）。"""
        m = select_by_rank(score, self.mask, k, largest=True)
        return Universe(m, ctx=self.ctx, note=note or f"top({k})")

    def bottom(self, score, k, note=None):
        """在**当前 universe 内部**按 score 取后 k 名（score 越小越好）。

        动量/反转类因子的标准用法：`u.bottom(ctx.ret(5), 100)`。
        """
        m = select_by_rank(score, self.mask, k, largest=False)
        return Universe(m, ctx=self.ctx, note=note or f"bottom({k})")

    def where(self, cond, note=None):
        """再用一个布尔条件过滤。"""
        cond = boolean_mask(cond, 'cond')
        return Universe(self.mask & cond,
                        self.keep & cond,
                        ctx=self.ctx, note=note or self.note)

    # -- 进出缓冲 --------------------------------------------------------
    def buffer(self, score, add_k, drop_k=None, note=None):
        """进出缓冲：把"进出场门槛"分开，压换手。

        `add_k`  排名进到前 add_k 才**买**
        `drop_k` 排名掉出前 drop_k 才**卖**（默认 = add_k 的 1.5 倍）

        留一个带（add_k < drop_k）意味着：已经在手里的票，排名掉到
        add_k 之外也不急着卖，要掉出 drop_k 才卖。这样每天的调仓量从
        "全部重置"降到"只换掉真正掉队的几只"，佣金和印花税能省一个量级。

        引擎拿 `mask`（=可以买）和 `keep`（=可以留）跟实际持仓求交并，
        所以这个缓冲是"有记忆"的 —— 光看当天的分数算不出来。
        """
        if drop_k is None:
            drop_k = int(round(add_k * 1.5))
        if drop_k < add_k:
            raise ValueError(f"drop_k({drop_k}) 必须 >= add_k({add_k})")
        add = select_by_rank(score, self.mask, add_k, largest=True)
        keep = select_by_rank(score, self.mask, drop_k, largest=True)
        return Universe(add, keep, ctx=self.ctx,
                        note=note or f"buffer(add={add_k}, drop={drop_k})")

    def drop_small(self, score, k, note=None):
        """去掉分数最差的 k 只。

        注意这和 `top(score, n-k)` 不等价：`top` 的 k 是常数，而这里要把
        当天的有效只数 n 算出来再减 k —— 停牌/新股会让 n 每天都不一样。
        """
        s = np.asarray(score, dtype=np.float64)
        valid = self.mask & np.isfinite(s)
        keep_n = np.maximum(valid.sum(axis=1, keepdims=True) - k, 0)
        return Universe(_keep_top_n(valid, s, keep_n), ctx=self.ctx,
                        note=note or f"drop_small({k})")

    # -- 权重（供报告/诊断用）--------------------------------------------
    def count(self):
        """每日持仓只数，(T,)。"""
        return self.mask.sum(axis=1)

    def turnover_proxy(self):
        """相邻两日选股集合的对称差比例，粗略反映换手压力。"""
        m = self.mask.astype(np.int8)
        if m.shape[0] < 2:
            return np.zeros(0)
        chg = np.abs(m[1:] - m[:-1]).sum(axis=1)
        denom = np.maximum(m[1:].sum(axis=1) + m[:-1].sum(axis=1), 1)
        return chg / denom
