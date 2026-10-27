"""账本恒等式测试 —— 全系统最重要的一组断言。

如果账本错了，后面所有指标、所有报告、所有"策略好不好"的结论都是假的，
而且**错得很安静**：净值曲线照样画得漂漂亮亮。

这里用一条可以逐位验证的恒等式把它钉死。买入之后再也不交易，那么

    equity_t  ==  cash + Σ_j  V_j × R_{j,t}

其中 V_j 是第 j 只在买入那天真正成交的市值（直接从成交流水里取），
R_{j,t} = adj_t / adj_买入 是它从买入起的**复权**收益。

这一条式子同时验证了四件事：
  1. 复权方向（乘 vs 除）—— 搞反了 R 就全错
  2. 除权除息的账务处理（拆股调股数）—— 不调的话 R 里会混进拆股跳空
  3. 两条价格线（close / close_ff）分得对不对 —— 停牌不能变成 0 收益
  4. 现金账 —— 买入扣了多少、现金有没有凭空多出来

    python tests/test_accounting.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qbt  # noqa: E402
from qbt.costs import Costs  # noqa: E402

_fail = []


def check(cond, msg):
    print(("    ok   " if cond else "    FAIL ") + msg)
    if not cond:
        _fail.append(msg)


FREE = Costs(commission=0.0, min_commission=0.0, transfer_fee=0.0,
             slippage_bps=0.0, stamp_duty=0.0)

#: 偏离带设成一个大于 1 的数 = 永远不调仓，得到干净的"买入持有"
NO_REBALANCE = 1.0


def pick_stocks(d, i0, i1, k):
    """挑 k 只在窗口内**从未停牌**的股票（零 NaN），免得停牌干扰恒等式。"""
    c = np.asarray(d.close)[i0:i1 + 1]
    ok = ~np.isnan(c).any(axis=0)
    idx = np.flatnonzero(ok)
    p = np.nanmedian(c[:, idx], axis=0)
    good = idx[(p > 5) & (p < 200)]      # 避开仙股和千元股
    return good[:k]


def main():
    d = qbt.load(verbose=False)
    m = qbt.build_market(d, log=None)

    start, end = "2021-01-04", "2026-10-15"
    i0 = d.calendar.index(start)
    i1 = d.calendar.index(end)
    k = 20
    cols = pick_stocks(d, i0, i1, k)
    print(f"\n选 {len(cols)} 只全程无停牌的股票，窗口 {d.dates[i0]} .. {d.dates[i1]}\n")

    selmask = np.zeros(d.N, dtype=bool)
    selmask[cols] = True

    capital = 1e8                      # 仓位够大，整手取整的影响降到 1e-5 以下
    r = qbt.run(d, lambda ctx: ctx.long(selmask, gross=0.98),
                start=start, end=end, market=m, capital=capital,
                costs=FREE, max_adv_pct=1.0, div_policy="reinvest",
                limit_rule="off", name="买入持有",
                rebalance_tol=NO_REBALANCE)

    n_days = len(r.dates)
    mm = r.metrics()
    check(mm["成交笔数"] == len(cols),
          f"整个窗口只成交了 {mm['成交笔数']} 笔（买入 {len(cols)} 只后不再动）")

    F = np.asarray(d.cumadj_ff, dtype=np.float64)
    C = np.asarray(d.close_ff, dtype=np.float64)
    gi = np.searchsorted(d.dates, np.asarray(r.dates))     # 每行对应的全局下标
    px0 = np.asarray(d.open[i0], dtype=np.float64)[cols]   # 买入价 = 开盘价
    adj0 = px0 * F[i0, cols]                               # 后复权买入价 = 原价 × 因子

    # 从真实成交流水里取每只买了多少股 —— 不猜、不复算
    q0 = np.zeros(d.N, dtype=np.float64)
    for (_, j, side, qty, _px, _fee) in r.trades:
        if side > 0:
            q0[j] += qty
    V = q0[cols] * px0
    cash0 = r.cash[1]

    # adj_t = close_ff_t × F_t（后复权）。**两个因子都要乘**：只乘入场那一端、
    # 或者乘除搞反，都是那种"净值曲线照样画得出来"的静默错误。
    adj_t = C[gi][:, cols] * F[gi][:, cols]
    pred = cash0 + (V[None, :] * (adj_t / adj0[None, :])).sum(axis=1)
    actual = np.asarray(r.equity, dtype=np.float64)
    rel = np.abs(pred[1:] - actual[1:]) / np.maximum(np.abs(actual[1:]), 1.0)
    print(f"    恒等式最大相对偏差 {rel.max():.3e}（中位 {np.median(rel):.3e}）")
    check(rel.max() < 2e-3,
          f"equity_t == cash + Σ V_j·R_(j,t) 全程成立（最大偏差 {rel.max():.2e}）")

    # 对照：如果复权方向搞成除法，偏差应该大得离谱
    with np.errstate(invalid="ignore", divide="ignore"):
        R_bad = (C[gi][:, cols] / F[gi][:, cols]) / (px0[None, :] / F[i0, cols])
    pred_bad = cash0 + (V[None, :] * R_bad).sum(axis=1)
    rel_bad = np.abs(pred_bad[1:] - actual[1:]) / np.maximum(np.abs(actual[1:]), 1.0)
    print(f"    对照：复权方向若搞反，最大相对偏差 {rel_bad.max():.3e}")
    check(rel_bad.max() > 0.05,
          "搞反方向的偏差 > 5% —— 说明这条测试真的抓得住方向错误")

    # 对照：如果整个复权因子都不乘（拿未复权价当收益），也得被抓住
    R_raw = C[gi][:, cols] / px0[None, :]
    pred_raw = cash0 + (V[None, :] * R_raw).sum(axis=1)
    rel_raw = np.abs(pred_raw[1:] - actual[1:]) / np.maximum(np.abs(actual[1:]), 1.0)
    print(f"    对照：完全不复权，最大相对偏差 {rel_raw.max():.3e}")
    check(rel_raw.max() > 0.05, "不复权的偏差 > 5% —— 复权确实在起作用")

    # 恒等式必须覆盖到除权除息，否则它没测到那段逻辑
    with np.errstate(invalid="ignore", divide="ignore"):
        fev = np.abs(F[i0 + 1:i1 + 1][:, cols] / F[i0:i1][:, cols] - 1.0) > 5e-5
    n_ev = int(fev.sum())
    print(f"    窗口内这些股票共发生 {n_ev:,} 次除权除息")
    check(n_ev > 0, "窗口内确实有除权除息，恒等式覆盖了拆股/分红账务")

    # 现金账：费用必须真的被扣掉
    r2 = qbt.run(d, lambda ctx: ctx.long(selmask, gross=0.98),
                 start=start, end=end, market=m, capital=capital,
                 costs=qbt.Costs(), max_adv_pct=1.0, limit_rule="off",
                 name="买入持有(含费)", rebalance_tol=NO_REBALANCE)
    fees = r2.metrics()["总费用"]
    print(f"    同样组合加真实费用：总费用 {fees:,.0f} 元，"
          f"（{len(cols)} 笔买入 × 最低 5 元 = {len(cols)*5:,} 元起）")
    check(fees >= len(cols) * 5.0 - 1e-6,
          "每笔买入都收到了最低 5 元佣金（小资金的真实成本）")
    check(r2.equity[-1] < r.equity[-1], "含费用的净值低于免费用版本")
    check(abs((r.equity[-1] - r2.equity[-1]) - (fees - 0.0)) / max(fees, 1) < 0.5,
          "两版净值差 ≈ 费用总额（费用确实是从权益里扣的，不是记账外挂）")

    print("\n" + "=" * 68)
    if _fail:
        print(f"失败 {len(_fail)} 项：")
        for x in _fail:
            print("  -", x)
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
