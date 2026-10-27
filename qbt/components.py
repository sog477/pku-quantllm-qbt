"""可替换的组合分配与风险组件。每次 run 创建独立实例。"""
from __future__ import annotations

import numpy as np


class WeightAllocation:
    """返回成员内权重；自定义组件遵循同一 callable 协议。"""

    def __call__(self, members, *, weight="equal", score=None):
        n = int(np.count_nonzero(members))
        if not n:
            return np.empty(0)
        if weight == "equal":
            return np.full(n, 1.0 / n)
        if weight not in ("score", "value") or score is None:
            raise ValueError(f"不支持的分配方式或缺少 score：{weight!r}")
        s = np.asarray(score, float)[members]
        s = np.where(np.isfinite(s) & (s > 0), s, 0.)
        return s / s.sum() if s.sum() else np.full(n, 1. / n)


class RiskControls:
    """开盘评估止损/止盈，收盘后更新追踪高水位。"""

    def __init__(self, n_stocks, stops=None):
        self.peak = np.full(n_stocks, np.nan)
        self.rebase_seen=np.zeros(n_stocks,float)
        self.generation_seen=np.zeros(n_stocks,np.int64)
        self.configure(stops)

    def configure(self, stops):
        self.stops = dict({} if stops is None else stops)
        for k, v in self.stops.items():
            if k not in ("hard", "trailing", "take_profit"):
                raise ValueError(f"未知风控配置：{k}")
            if not np.isscalar(v) or not np.isfinite(v) or v <= 0:
                raise ValueError(f"{k} 必须是正的有限标量")

    def exits(self, account, adj_open):
        self._rebase(account)
        held = account.shares > 0
        ret = account.unrealized(adj_open)
        hit = np.zeros(held.size, bool)
        if "hard" in self.stops:
            hit |= ret <= -self.stops["hard"]
        if "take_profit" in self.stops:
            hit |= ret >= self.stops["take_profit"]
        if "trailing" in self.stops:
            with np.errstate(invalid="ignore", divide="ignore"):
                hit |= account.risk_price(adj_open) / self.peak - 1 <= -self.stops["trailing"]
        return held & hit

    def mark_close(self, account, adj_close):
        self._rebase(account)
        held = account.shares > 0
        effective = account.risk_price(adj_close)
        self.peak[held] = np.fmax(self.peak[held], effective[held])
        self.peak[~held] = np.nan

    def _rebase(self, account):
        """Adding capital must not turn dilution of earned dividends into drawdown."""
        changed=account.position_generation!=self.generation_seen
        self.peak[changed]=np.nan
        delta=account.risk_rebases-self.rebase_seen
        self.peak+=delta
        self.rebase_seen=account.risk_rebases.copy()
        self.generation_seen=account.position_generation.copy()
