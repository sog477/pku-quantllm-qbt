"""策略契约：研究员写的东西长什么样。

核心是 `Target` —— 一份"我想持有什么"的声明，**不含任何交易细节**。
买不买得到、够不够钱、要不要整手、涨跌停挡不挡，全是引擎的事。
这样研究员写策略时不需要（也不应该）考虑这些，就不会漏。

    def 五日反转(ctx):
        u = ctx.universe(min_amt=5e7, min_listed=60, not_st=True)
        return ctx.long(u.buffer(-ctx.ret(5, lag=1), add_k=100, drop_k=150),
                        gross=0.98)

面板是 (T, N) 的：一次算完整条时间线，而不是逐日回调，所以没有
5451 × 1643 次 Python 调用的开销。想做事件驱动的策略，也可以返回
(N,) 的 Target，引擎会广播。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Target", "HOLD", "FLAT", "hold", "flat"]


@dataclass(frozen=True)
class Target:
    """目标持仓。

    sel     (T,N) 或 (N,) bool —— 想持有的股票
    keep    (T,N) 或 (N,) bool —— 允许继续持有的（进出缓冲的"留"那一档）；
            不给就等于 sel
    gross   目标总仓位（0~1），引擎把权重缩放到这个数，余额留在现金
    weight  "equal" 等权 / "score" 或 "value" 按显式score加权
            成交额加权须传 score=ctx.amt()，不会自动猜字段
    score   配合 weight="score"
    stops   止损设置，形如 {"hard": 0.10, "trailing": 0.20, "take_profit": 0.30}，次日开盘评估
    """

    sel: np.ndarray | None = None
    keep: np.ndarray | None = None
    gross: float = 1.0
    weight: str = "equal"
    score: np.ndarray | None = None
    stops: dict | None = None
    note: str = ""

    # 空仓
    @staticmethod
    def flat():
        return Target(sel=np.zeros(1, dtype=bool), gross=0.0)

    @property
    def keep_or_sel(self):
        return self.sel if self.keep is None else self.keep


class _Hold:
    def __repr__(self):
        return "qbt.HOLD"

    def __reduce__(self):
        return (_get_hold, ())


def _get_hold():
    return HOLD


#: 事件模式维持上一个目标；初始目标为空仓。None 始终是错误。
HOLD = _Hold()
#: 返回 FLAT 表示清仓
FLAT = Target.flat()


def hold(sel, gross=1.0, **kw):
    """等权持有 sel 里的股票。"""
    return Target(sel=sel, gross=gross, **kw)


def flat():
    return FLAT
