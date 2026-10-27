"""绩效指标 + 两次回测的对比。

指标本身没什么玄机，但有两个口径问题必须钉死，否则报告会骗人：

1. 年化使用结果中时钟提供的 periods_per_year，默认日频为 242。
2. 波动率由每个估值区间的收益计算并年化；夏普沿用几何年化收益减 rf
   后除以年化波动率的口径。

`compare(纸面, 真实)` 是老师那道题的答案所在，单独说：

    被挡住的资金 = Σ over 被涨停挡掉的买单 (成交前想买的股数 × 开盘价)
    被挡个股的当日收益 = 这些票当天开盘到收盘的涨幅（成交额加权）

两者的乘积就是"纸面净值比真实净值多出来的那部分"，可以直接和两次回测的
净值差对上。对得上，说明引擎里关于涨停的处理是自洽的；对不上，说明还有
别的漏损（跌停卖不掉、容量、现金不够）没被算进去。
"""

from __future__ import annotations

import numpy as np

__all__ = ["metrics", "compare", "monthly_table", "yearly_table", "drawdown",
           "TRADING_DAYS"]

TRADING_DAYS = 242


def _safe(x):
    return float(x) if np.isfinite(x) else float("nan")


def drawdown(eq):
    """回撤序列（负数）与最大回撤。"""
    eq = np.asarray(eq, dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = eq / np.maximum(peak, 1e-12) - 1.0
    return dd, _safe(dd.min() if dd.size else np.nan)


def _ann_factor(n):
    return TRADING_DAYS / max(n, 1)


def equity_returns(equity):
    """零权益保持零计0收益；从零恢复的百分比无定义，不能用epsilon伪造巨额收益。"""
    eq = np.asarray(equity, dtype=np.float64)
    ret = np.full(len(eq)-1, np.nan)
    np.divide(np.diff(eq), eq[:-1], out=ret, where=eq[:-1] > 0)
    ret[(eq[:-1] == 0) & (eq[1:] == 0)] = 0.
    return ret


def perf(eq, n_days, rf=0.0, periods_per_year=TRADING_DAYS):
    """给定净值序列，算出全套指标。"""
    eq = np.asarray(eq, dtype=np.float64)
    if eq.size < 2 or eq[0] <= 0:
        return {}
    ret = equity_returns(eq)
    n = len(ret)
    total = eq[-1] / eq[0] - 1.0
    years = n / periods_per_year
    ann = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 and total >= -1 else float("nan")
    vol = float(np.std(ret, ddof=1) * np.sqrt(periods_per_year)) if n > 1 else float("nan")
    sharpe = (ann - rf) / vol if vol and np.isfinite(vol) and vol > 0 else float("nan")
    target = (1.0 + rf) ** (1.0 / periods_per_year) - 1.0
    shortfall = np.minimum(ret-target, 0.)
    dvol = float(np.sqrt(np.mean(shortfall**2) * periods_per_year))
    sortino = (ann - rf) / dvol if dvol and np.isfinite(dvol) and dvol > 0 else float("nan")
    _, mdd = drawdown(eq)
    calmar = ann / abs(mdd) if mdd and np.isfinite(mdd) and mdd < 0 else float("nan")
    return {
        "初始资金": float(eq[0]),
        "期末资产": float(eq[-1]),
        "累计收益": _safe(total),
        "年化收益": _safe(ann),
        "年化波动": _safe(vol),
        "夏普": _safe(sharpe),
        "索提诺": _safe(sortino),
        "最大回撤": _safe(mdd),
        "卡玛": _safe(calmar),
        "日胜率": _safe(float((ret[np.isfinite(ret)] > 0).mean()) if np.isfinite(ret).any() else np.nan),
        "收益未定义区间数": int((~np.isfinite(ret)).sum()),
        "交易日数": int(n),
        "年数": _safe(years),
    }


# --------------------------------------------------------------------------
def metrics(r):
    """`Result` -> 指标字典（报告会按这个顺序排）。"""
    eq = np.asarray(r.equity, dtype=np.float64)
    if eq.ndim != 1 or eq.size < 2 or np.asarray(r.dates).shape != eq.shape:
        raise ValueError('权益与日期须为等长的一维序列，且包含初始基值')
    for name, size in [('ret',len(eq)-1),('bench_ret',len(eq)-1),('bench_nav',len(eq))]:
        if np.asarray(getattr(r,name)).shape != (size,):
            raise ValueError(f'{name} 长度/形状未与权益日期对齐，不能截短后计算指标')
    frequency = getattr(r, "periods_per_year", TRADING_DAYS)
    m = perf(eq, len(eq) - 1, periods_per_year=frequency)
    m["估值区间数"] = len(eq)-1
    m["年化区间数"] = frequency

    ben = np.asarray(r.bench_nav, dtype=np.float64)
    b = perf(ben, len(ben) - 1, periods_per_year=frequency)
    bench_total = b.get("累计收益", float("nan"))
    bench_ann = b.get("年化收益", float("nan"))
    m["基准累计"] = bench_total
    m["基准年化"] = bench_ann
    m["超额累计"] = m["累计收益"] - bench_total
    m["超额年化"] = m["年化收益"] - bench_ann

    # 相对基准
    rr = np.asarray(r.ret, dtype=np.float64)
    br = np.asarray(r.bench_ret, dtype=np.float64)
    k = rr.size
    if k > 1:
        act = rr[:k] - br[:k]
        sd = float(np.std(act, ddof=1))
        m["跟踪误差"] = _safe(sd * np.sqrt(frequency))
        m["信息比率"] = _safe(float(np.mean(act)) / sd * np.sqrt(frequency)
                              if sd > 0 else np.nan)
        vb = float(np.var(br[:k], ddof=1))
        m["Beta"] = _safe(float(np.cov(rr[:k], br[:k])[0, 1]) / vb if vb > 0 else np.nan)
        m["Alpha"] = _safe((m["年化收益"] - m.get("Beta", 0.0) * bench_ann)
                           if np.isfinite(m.get("Beta", np.nan)) else np.nan)
        m["月度胜率"] = _monthly_winrate(r)

    # 交易与成本
    to = float(np.asarray(r.turnover_cum)[-1])
    avg_eq = float(np.mean(eq))
    years = max(m.get("年数", 0.0), 1e-9)
    m["换手率(年)"] = _safe(to / max(avg_eq, 1e-9) / years)
    m["总费用"] = float(np.asarray(r.fees_cum)[-1])
    m["费用占初始"] = _safe(m["总费用"] / max(float(eq[0]), 1e-9))
    m["费用占期末"] = _safe(m["总费用"] / max(float(eq[-1]), 1e-9))
    m["期末资产"] = float(eq[-1])
    m["初始资金"] = float(eq[0])
    m["平均持仓数"] = _safe(float(np.mean(r.n_positions[1:])))
    m["平均仓位"] = _safe(float(np.mean(r.gross[1:])))
    if r.trades is not None:
        m["成交笔数"] = int(len(r.trades))
    acct = r.account
    m["买入笔数"] = int(acct.n_buy)
    m["卖出笔数"] = int(acct.n_sell)
    m["除权事件"] = int(acct.n_div_events)
    m["退市强平笔数"] = int(acct.n_force_close)
    settlements=getattr(r,'settlements',())
    m["清算回收现金"] = float(sum(s.proceeds for s in settlements if s.pay_date is None or s.pay_date<=r.dates[-1]))
    m['清算待支付权益']=float(sum(s.proceeds for s in settlements if s.pay_date is not None and s.pay_date>r.dates[-1]))
    m['权益税款']=float(getattr(acct,'taxes',0.))
    m['应付税款']=float(getattr(acct,'tax_payables',np.zeros(1)).sum())
    m["被挡买单笔数"] = int(r.blocked_buy_n.sum())
    m["被挡卖单笔数"] = int(r.blocked_sell_n.sum())
    return m


def _monthly_winrate(r):
    """月度胜率 = 跑赢基准的月份占比。"""
    a = monthly_table(r, as_dict=True)
    b = monthly_table(r, as_dict=True, use_bench=True)
    common = [k for k in a if k in b and np.isfinite(a[k]) and np.isfinite(b[k])]
    if not common:
        return float("nan")
    return _safe(sum(1 for k in common if a[k] > b[k]) / len(common))


# --------------------------------------------------------------------------
def _month_key(d):
    d = np.datetime64(d, "D")
    y = d.astype("datetime64[Y]").astype(int) + 1970
    m = d.astype("datetime64[M]").astype(int) % 12 + 1
    return int(y), int(m)


def monthly_table(r, as_dict=False, use_bench=False):
    """月度收益表。返回 {datetime64[M]: 收益}，或者按年分组便于画热力图。"""
    eq = np.asarray(r.bench_nav if use_bench else r.equity, dtype=np.float64)
    dates = np.asarray(r.dates)
    if eq.size < 2:
        return {} if as_dict else {}
    ret = equity_returns(eq)
    dd = dates[1:]
    keys = [_month_key(x) for x in dd]
    out = {}
    for (y, mo), x in zip(keys, ret):
        out[(y, mo)] = (1.0 + out.get((y, mo), 0.0)) * (1.0 + float(x)) - 1.0
    if as_dict:
        return {np.datetime64(f"{y:04d}-{mo:02d}", "M"): v for (y, mo), v in out.items()}
    return out


def yearly_table(r, use_bench=False):
    eq = np.asarray(r.bench_nav if use_bench else r.equity, dtype=np.float64)
    dates = np.asarray(r.dates)
    if eq.size < 2:
        return {}
    ret = equity_returns(eq)
    years = np.array([_month_key(x)[0] for x in dates[1:]])
    out = {}
    for y in np.unique(years):
        sel = years == y
        out[int(y)] = float(np.prod(1.0 + ret[sel]) - 1.0)
    return out


# --------------------------------------------------------------------------
def compare(a, b, label_a=None, label_b=None):
    """对比两次回测 —— **老师那道题的答案就在这里**。

    典型用法：

        a = qbt.run(d, 策略, limit_rule="off")       # 纸面
        b = qbt.run(d, 策略)                         # 真实
        print(a.compare(b, "纸面（不看涨停）", "真实（开盘封板买不到）"))

    返回一个可打印的对象，`repr` 就是一张表。
    """
    if getattr(a,"periods_per_year",TRADING_DAYS) != getattr(b,"periods_per_year",TRADING_DAYS):
        raise ValueError("两次回测估值频率不同，请先统一时钟再比较")
    la = label_a or f"{a.name}[{a.limit_rule}]"
    lb = label_b or f"{b.name}[{b.limit_rule}]"

    # 按日期对齐（两次回测区间可能不同）
    da, db = np.asarray(a.dates), np.asarray(b.dates)
    common = np.intersect1d(da, db)
    if common.size < 2:
        raise ValueError("两次回测没有重叠的交易日，无法比较")
    for dates in (da,db):
        overlap = dates[(dates >= common[0]) & (dates <= common[-1])]
        if not np.array_equal(overlap,common):
            raise ValueError("重叠区间的估值时钟不一致，按日期交集计算会扭曲年化；请先统一时钟")

    def _slice(r, want):
        d = np.asarray(r.dates)
        idx = np.searchsorted(d, want)
        eq = np.asarray(r.equity, dtype=np.float64)[idx]
        if not np.isfinite(eq[0]) or eq[0] <= 0:
            raise ValueError('重叠区间起始权益为零或无效，无法归一化比较')
        return eq / eq[0]

    na, nb = _slice(a, common), _slice(b, common)

    ma = perf(na * 1.0, len(na) - 1, periods_per_year=getattr(a, "periods_per_year", TRADING_DAYS))
    mb = perf(nb * 1.0, len(nb) - 1, periods_per_year=getattr(b, "periods_per_year", TRADING_DAYS))

    # 被挡明细只能取自**真实口径**的那一次。`limit_rule="off"` 那一次里
    # 「被挡买单」按定义恒为 0（off 不拦任何涨跌停），如果这里写死读 `a`，
    # 而调用方恰好把纸面那次传成 `a`，表里就会印出
    # 「被挡个股当日平均涨幅 +0.0000% / 本来能赚却没赚到 0 元」——
    # 一张看起来正常、其实什么都没说的表。所以按数额挑，并把出处标出来。
    def _intervals(r):
        # 初始共同快照之前/当时的成交不属于随后这段净值变化。
        dates = np.asarray(r.dates)[1:]
        return (dates > common[0]) & (dates <= common[-1])

    def _blk(r):
        mask = _intervals(r)
        return int(np.asarray(r.blocked_buy_n)[mask].sum()) + \
            int(np.asarray(r.blocked_sell_n)[mask].sum())

    src, src_label = (a, la) if _blk(a) >= _blk(b) else (b, lb)

    mask = _intervals(src)
    w = np.asarray(src.blocked_buy_amt, dtype=np.float64)[mask]
    bounce = np.asarray(src.blocked_buy_bounce, dtype=np.float64)[mask]
    blk_amt = float(w.sum())
    blk_pnl = float((w * bounce).sum())
    blk_n = int(np.asarray(src.blocked_buy_n)[mask].sum())
    blk_days = int((np.asarray(src.blocked_buy_n)[mask] > 0).sum())
    wavg = float((w * bounce).sum() / w.sum()) if w.sum() > 0 else float("nan")

    sblk_amt = float(np.asarray(src.blocked_sell_amt)[mask].sum())
    sblk_n = int(np.asarray(src.blocked_sell_n)[mask].sum())

    # 换手和费用**不在净值曲线上**：一个亿的账户和一千万的账户可以有
    # 一模一样的净值形状。所以 `perf`（只看曲线）算不出这两行，早先这张表
    # 就把它们印成了 "-"，看着像引擎算崩了 —— 其实是取错了来源。
    # 现在从各自的完整回测里取，并在表头注明是**全区间**数字。
    for _m, _r in ((ma, a), (mb, b)):
        _full = metrics(_r)
        for _k in ("换手率(年)", "总费用"):
            _m[_k] = _full.get(_k)
    aligned = (np.asarray(a.dates).size == common.size
               and np.asarray(b.dates).size == common.size)

    return _Compare(la, lb, common, na, nb, ma, mb,
                    blk_n, blk_days, blk_amt, blk_pnl, wavg, sblk_n, sblk_amt,
                    capital=float(np.asarray(src.equity, dtype=np.float64)[0]),
                    blk_src=src_label,
                    badjust=src is b,
                    aligned=aligned)


class _Compare:
    def __init__(self, la, lb, common, na, nb, ma, mb, blk_n, blk_days,
                 blk_amt, blk_pnl, wavg, sblk_n, sblk_amt, capital=1.0,
                 blk_src="", badjust=False, aligned=True):
        self.capital = float(capital)
        self.label_a, self.label_b = la, lb
        self.dates = common
        self.nav_a, self.nav_b = na, nb
        self.metrics_a, self.metrics_b = ma, mb
        self.n_blocked_buy = blk_n
        self.n_days_with_block = blk_days
        self.blocked_amount = blk_amt
        self.blocked_pnl = blk_pnl
        self.blocked_avg_bounce = wavg
        self.n_blocked_sell = sblk_n
        self.blocked_sell_amount = sblk_amt
        self.blocked_source = blk_src
        self._badjust = badjust
        self._aligned = aligned

    @property
    def nav_gap(self):
        """两次回测的累计净值差（A - B）。"""
        return float(self.nav_a[-1] - self.nav_b[-1])

    def __repr__(self):
        ma, mb = self.metrics_a, self.metrics_b
        w = 26
        lines = [
            "=" * 76,
            f"{'':{w}}{self.label_a:>22}{self.label_b:>22}",
            "-" * 76,
        ]
        # 行名要跟实际口径对上：表里比的是**对齐区间**的相对净值，
        # 所以「期末资产」那一行其实是净值倍数，不是钱。印成「资产」
        # 会让人以为两次回测的钱不一样多。
        labels = {"期末资产": "期末净值（起点=1）",
                  "累计收益": "累计收益（对齐区间）"}
        for k in ("累计收益", "年化收益", "年化波动", "夏普", "最大回撤",
                  "换手率(年)", "总费用", "期末资产"):
            va, vb = ma.get(k), mb.get(k)
            lines.append(f"{labels.get(k, k):<{w}}{_fmt(va):>22}{_fmt(vb):>22}")
        lines.append("-" * 76)
        lines.append("  注：上表除「换手率」「总费用」外都是**对齐区间**上的口径；")
        lines.append("      那两行取自各自的完整回测（净值曲线本身算不出换手和费用）。")
        if not self._aligned:
            lines.append("      ⚠ 两次回测区间不同，已按共同交易日对齐后比较。")
        lines.append(f"{'累计净值差 (A - B)':<{w}}{self.nav_gap:>+22.4f}")
        lines.append("")
        lines.append("【涨停/跌停挡掉了什么】—— 老师那道题的答案")
        if self.n_blocked_buy:
            lines.append(f"  （被挡明细取自 {self.blocked_source}，"
                         "统计共同区间；金额比例使用该回测的初始本金）")
        lines.append(f"  被买单挡下的次数        {self.n_blocked_buy:>10,} 笔"
                     f"（发生在 {self.n_days_with_block:,} 个交易日）")
        if self.n_blocked_buy:
            lines.append(f"  被挡下的资金合计        {self.blocked_amount:>14,.0f} 元")
            lines.append(f"  被挡个股**当日**平均涨幅 "
                         f"{self.blocked_avg_bounce:>+13.4%}")
            if self.blocked_pnl >= 0:
                lines.append(f"  => 本来能赚却没赚到的   {self.blocked_pnl:>14,.0f} 元"
                             f"  （占初始资金 {self.blocked_pnl / self.capital:+.4%}）")
            else:
                # 负的机会成本不是笔误：涨停次日开盘买进去，当天通常是绿的。
                # 这时候"买不到"是**帮你躲开了**，不是损失。把它说成损失
                # 就是把符号当成了结论。
                lines.append(f"  => 没买到反而躲掉亏损   {-self.blocked_pnl:>14,.0f} 元"
                             f"  （占初始资金 {-self.blocked_pnl / self.capital:+.4%}）")
                lines.append("     注意符号：被挡的这批票当天平均是**跌**的 —— "
                             "「涨停买不到」在这里")
                lines.append("     不是损失，是保护。涨停次日的开盘价通常就是"
                             "别人出货给你的价。")
            lines.append(f"  实际累计净值差           {self.nav_gap:>+14.4f}")
            lines.append("    注：净值差不等于上面的金额。被挡的钱没消失，它按"
                         "『这个票当天涨多少』的机会成本记，")
            lines.append("        之后还会重新投出去，两条路径的现金/持仓从此就分岔了。"
                         "上表给的是**量级和方向**，")
            lines.append("        不是恒等式；要恒等式请看 tests/test_accounting.py。")
        else:
            lines.append("  本次共同区间没有记录涨停买入阻断；其他时段和参数仍可能受约束。")
        lines.append(f"  被卖单挡下的次数        {self.n_blocked_sell:>10,} 笔")
        if self.n_blocked_sell:
            lines.append(f"  被套住卖不掉的市值      {self.blocked_sell_amount:>14,.0f} 元")
            lines.append("    这些卖出意图被阻断；这项统计不能单独证明最大回撤的成因。")
        lines.append("=" * 76)
        return "\n".join(lines)

    def plot_data(self):
        return {"dates": self.dates, "a": self.nav_a, "b": self.nav_b,
                "label_a": self.label_a, "label_b": self.label_b}


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if not np.isfinite(v):
        return "-"
    if abs(v) < 10:
        return f"{v:+.4f}" if abs(v) < 1 else f"{v:+.3f}"
    return f"{v:,.0f}"
