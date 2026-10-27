"""面板策略体检：比较抽查点整个前缀内的选股、留仓、仓位、有效权重和风控配置。

同时试跑账本并比较成本/容量敏感性。有限探针不能证明不存在所有前视，
也不能代替数据源和成交假设审查。事件策略请检查逐日回调和真实成交流水。
"""

from __future__ import annotations

import numpy as np

from .context import Context
from .engine import run, _resolve_target, _as_panel, _as_panel_score, _final_membership
from .tradability import build_market

__all__ = ["health", "HealthReport"]


def _eq_bool(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    return a.shape == b.shape and bool((a == b).all())


def _eq_num(a, b, tol=0.0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not (fa == fb).all() or not np.all((a[~fa] == b[~fb]) | (np.isnan(a[~fa]) & np.isnan(b[~fb]))):
        return False
    if not fa.any():
        return True
    if tol <= 0:
        return bool((a[fa] == b[fa]).all())
    return bool(np.allclose(a[fa], b[fa], rtol=0, atol=tol, equal_nan=True))


class _Lines:
    def __init__(self):
        self.rows = []
        self.failed = []

    def check(self, ok, msg):
        self.rows.append(("ok  " if ok else "FAIL", msg))
        if not ok:
            self.failed.append(msg)

    def info(self, msg):
        self.rows.append(("    ", msg))

    def head(self, msg):
        self.rows.append(("##  ", msg))


class HealthReport:
    def __init__(self, lines, summary):
        self.lines = lines
        self.summary = summary

    @property
    def ok(self):
        return not self.summary["失败项"]

    def __repr__(self):
        w = max((len(m) for _s, m in self.lines if _s == "##  "), default=0)
        out = ["=" * 78, "回测体检", "=" * 78]
        for s, m in self.lines:
            if s == "##  ":
                out.append("")
                out.append(f"【{m}】".ljust(w + 4, "─"))
            elif s == "    ":
                out.append(f"       {m}")
            else:
                out.append(f"  {s}  {m}")
        out.append("")
        out.append("=" * 78)
        if self.ok:
            out.append("已覆盖检查通过；仍须核查输入数据与成交假设。")
        else:
            out.append(f"有 {len(self.summary['失败项'])} 项没过 —— "
                       f"请先核对失败项：")
            for x in self.summary["失败项"]:
                out.append(f"  - {x}")
        out.append("=" * 78)
        return "\n".join(out)

    def _repr_html_(self):
        return "<pre style='font:12px/1.5 ui-monospace,Consolas,monospace'>" + \
               repr(self).replace("<", "&lt;") + "</pre>"


def _target_at(tgt, row, T):
    """取目标面板第 row 行；标量 gross 广播。返回 (sel, keep, gross, stops, note)。"""
    def row_of(a):
        a = np.asarray(a)
        if a.ndim == 0:
            return a
        if a.ndim == 1:
            return a[0] if a.size == 1 else a
        return a[row]
    return (row_of(tgt.sel), row_of(tgt.keep) if tgt.keep is not None else None,
            row_of(np.asarray(tgt.gross, dtype=np.float64)), tgt.stops, tgt.note)


def _execution_signature(tgt, T, N):
    sel = _as_panel(tgt.sel, T, N, "Target.sel")
    keep = _as_panel(tgt.keep if tgt.keep is not None else sel, T, N, "Target.keep")
    gross = np.broadcast_to(np.asarray(tgt.gross, float), (T,))
    score = _as_panel_score(tgt.score, T, N, tgt.weight)
    effective = _final_membership(sel, keep)
    if tgt.weight == "equal":
        weights = None
    else:
        s = np.where(effective & np.isfinite(score) & (score > 0), score, 0.)
        total = s.sum(axis=1, keepdims=True)
        equal = effective / np.maximum(effective.sum(axis=1, keepdims=True), 1)
        weights = np.divide(s, total, out=equal.astype(float), where=total > 0)
    # 策略可能复用同一个numpy工作区。后续前缀调用不能改写全样本对照证据。
    return sel.copy(), keep.copy(), gross.copy(), weights, tgt.weight, dict({} if tgt.stops is None else tgt.stops)


def _same_signature(full, part, t):
    for k in (0, 1):
        if not _eq_bool(full[k][:t], part[k]):
            return False, ("sel" if k == 0 else "keep")
    if not _eq_num(full[2][:t], part[2], tol=1e-12):
        return False, "gross"
    if full[4] != part[4]:
        return False, "weight"
    if (full[3] is None) != (part[3] is None):
        return False, "score"
    if full[3] is not None and not _eq_num(full[3][:t], part[3], tol=1e-12):
        return False, "score/个股权重"
    if full[5].keys() != part[5].keys():
        return False, "stops"
    for key in full[5]:
        if not _eq_num(full[5][key], part[5][key], tol=1e-12):
            return False, f"stops.{key}"
    return True, ""


def health(ds, strategy, params=None, market=None, start=None, end=None,
           prefix_probes=8, market_kw=None, quick=False, capital=1e6,
           **run_options):
    """给一个策略做体检。返回一个可打印的 `HealthReport`。

    ds / strategy / params   同 `qbt.run`
    prefix_probes            前缀测试抽查多少个截断点（默认 8，均匀分布）
    capital                  试跑用的初始资金。**和正式回测保持一致** ——
                             佣金有"单笔最低 5 元"，100 万和 1000 万的账户
                             费率能差一倍，用不同的资金体检等于换了个口径。
    quick=True               跳过成本/容量敏感性那几组（每次都要真跑一遍回测）
    """
    L = _Lines()
    if 'market_profile' in run_options or 'reference_data' in run_options or isinstance(market,str):
        from .markets import prepare_market_options
        ds,run_options,_,_,_=prepare_market_options(ds,dict(run_options,market=market,market_kw=market_kw))
        market=run_options.pop('market')
        market_kw=None
    m = market if market is not None else build_market(ds, log=None, **({} if market_kw is None else market_kw))
    T, N = ds.T, ds.N
    summary = {}

    # ------------------------------------------------------------------
    L.head("A. 前视偏差（前缀测试）—— 最重要的一项")
    full_ctx = Context(ds, m, params)
    if run_options.get("mode", "panel") != "panel":
        raise ValueError("health 前缀检查用于面板策略；事件策略请用 run_events 并检查逐日回调")
    frequency = getattr(run_options.get("clock"),"periods_per_year",242.)
    tgt_full, _ = _resolve_target(ds, m, strategy, params, T, N, frequency)
    if tgt_full is None:
        raise TypeError("策略返回了 None；想清仓请返回 qbt.FLAT")
    if tgt_full.sel is None:
        raise ValueError("Target.sel 不能是 None")
    sel_full = _as_panel(tgt_full.sel, T, N, "Target.sel")
    sig_full = _execution_signature(tgt_full, T, N)
    sel_full = sig_full[0]

    # 截断点要绕开最前面一段：策略的 universe 常常要求 20~60 个交易日的预热，
    # 在那之前"两边都算不出来"是正常的，不构成前视，报出来只会淹掉真信号。
    lo = min(70, max(1, T // 2))
    hi = T - 1
    probes = np.unique(np.linspace(lo, hi, int(prefix_probes)).astype(int)) \
        if hi > lo else np.array([], dtype=int)

    bad = []
    for t in probes:
        try:
            ctx_t = Context(ds, m, params).sub(0, t)
            ctx_t.periods_per_year = frequency
            from .strategy import Target, HOLD
            tgt_t = strategy(ctx_t) if callable(strategy) else strategy
            if tgt_t is HOLD or tgt_t is None and strategy is None:
                tgt_t = Target(sel=np.zeros((t, N), bool), gross=0.)
            if not isinstance(tgt_t, Target):
                raise TypeError("策略必须返回 Target/HOLD/FLAT")
            sig_t = _execution_signature(tgt_t, t, N)
        except Exception as e:                        # noqa: BLE001
            bad.append((int(t), f"{type(e).__name__}: {e}"))
            continue
        if tgt_t is None:
            bad.append((int(t), "截断后返回了 None"))
            continue
        same, field = _same_signature(sig_full, sig_t, t)
        if not same:
            bad.append((int(t), f"前缀内 {field} 改变"))

    summary["前视探针数"] = int(probes.size)
    summary["前视失败点"] = [int(t) for t, _ in bad]
    L.info(f"抽查 {probes.size} 个截断点："
           + (", ".join(str(x) for x in probes[:12]) + ("..." if probes.size > 12 else "")))
    L.check(not bad, "抽查前缀内 sel/keep/gross/有效权重/风控配置一致（不能证明不存在所有前视）")
    for t, why in bad[:6]:
        L.info(f"  t={t}（第 {t-1} 行）：{why}")
    if bad:
        L.info("  查这三处：全样本统计（mean/std/rank 没截断）、"
               "ctx 的某个属性取了全历史、以及依赖『跑了多远』的分支。")

    # ------------------------------------------------------------------
    L.head("B. 选股面板的形状与进出缓冲")
    keep_full = sig_full[1]

    # 解进出缓冲：final[t] = (final[t-1] & keep[t]) | sel[t]（和引擎里同一套逻辑）
    fin = np.zeros((T, N), dtype=bool)
    hold = np.zeros(N, dtype=bool)
    for t in range(T):
        hold = (hold & keep_full[t]) | sel_full[t]
        fin[t] = hold
    per_day = fin.sum(axis=1)
    nz = per_day[per_day > 0]
    L.info(f"每日选股只数：中位 {int(np.median(nz)) if nz.size else 0}，"
           f"最少 {int(per_day.min())}，最多 {int(per_day.max())}")
    L.check(int(per_day.max()) <= N, "每日选股只数不超过股票池")
    L.check(bool(np.isfinite(np.asarray(tgt_full.gross, dtype=np.float64)).all()),
            "Target.gross 全部有限")

    # 换手不能看绝对值 —— 一个 5 天反转因子在 3000 只票上选 top100，排名本身每天
    # 就翻掉一半，那是**信号的噪声**，不是 bug。能说明问题的是**相对**的：
    # 加了进出缓冲（keep 比 sel 宽）之后，换手不该**高于**裸选。
    #
    # 注意这里只能查"不高于"，不能查"明显低于"：如果策略自己就把 sel 面板按调仓
    # 周期 forward-fill 了（presets.reversal 的 rebal_every 就是这么做的），那
    # sel 本身已经不是"每天重选"，两个数会一样大 —— 那是**正确**行为，按老规则
    # 报 FAIL 就是拿一条不适用的规则去冤枉它（体检表里的假警报比不报还坏：
    # 报几次假 FAIL，真 FAIL 也就没人看了）。
    raw = np.abs(np.diff(sel_full.astype(np.int8), axis=0)).sum(axis=1) / \
        np.maximum(sel_full[1:].sum(axis=1), 1)
    buf = np.abs(np.diff(fin.astype(np.int8), axis=0)).sum(axis=1) / \
        np.maximum(per_day[1:], 1)
    c_raw = float(np.median(raw)) if raw.size else 0.0
    c_buf = float(np.median(buf)) if buf.size else 0.0
    summary["选股域日换手(裸)"] = c_raw
    summary["选股域日换手(带缓冲)"] = c_buf
    L.info(f"选股域日换手（中位）：裸选 {c_raw:.1%} -> 带缓冲 {c_buf:.1%}"
           f"  → 折算年换手约 {c_buf * frequency:.0f} 倍（交易费用的大头就在这里）")
    if np.array_equal(keep_full, sel_full):
        L.info("  keep 与 sel 相同 —— 没设进出缓冲。等权目标每天随价格漂移，"
               "会有一批换手纯粹来自再平衡而不是信号。")
    else:
        L.check(c_buf <= c_raw + 1e-9,
                f"进出缓冲没有把换手弄高（裸选 {c_raw:.1%} vs 带缓冲 {c_buf:.1%}）")
        if c_raw <= 1e-9:
            L.info("  两行都是 0：策略自己在面板层就按调仓周期 forward-fill 了"
                   "（例如 presets.reversal 的 rebal_every），这是对的。"
                   "想看缓冲到底压掉多少换手，把它设成 1 再跑一遍。")

    # ------------------------------------------------------------------
    L.head("C. 真实跑一遍：账本与拒单")
    kw = dict(run_options, start=start, end=end, params=params, market=m, capital=capital, name="体检")
    r = run(ds, strategy, **kw)
    acct = r.account
    mm = r.metrics()

    L.check(acct.cash >= -1e-6, f"现金全程非负（期末 {acct.cash:,.0f} 元）")
    n_ord = acct.n_buy + acct.n_sell
    L.info(f"成交 {n_ord:,} 笔（买 {acct.n_buy:,} / 卖 {acct.n_sell:,}），"
           f"累计费用 {acct.fees:,.0f} 元 = 初始资金的 "
           f"{acct.fees / max(r.capital, 1):.2%}")
    L.info(f"期末持仓 {int((acct.shares > 0).sum())} 只，"
           f"仓位 {float(r.gross[-1]):.2%}，"
           f"平均仓位 {mm.get('平均仓位', float('nan')):.2%}")

    blk_b = int(np.asarray(r.blocked_buy_n).sum())
    blk_s = int(np.asarray(r.blocked_sell_n).sum())
    summary["被挡买单"] = blk_b
    summary["被挡卖单"] = blk_s
    L.info(f"被涨停挡住的买单 {blk_b:,} 笔（{mm.get('被挡买单笔数', 0):,}），"
           f"被跌停挡住的卖单 {blk_s:,} 笔")
    if blk_b or blk_s:
        L.info("  这不是 bug —— 是 A 股的真实约束。跑一次 limit_rule=\"off\" "
               "对比一下就老师那道题的答案。")

    dq = acct.n_div_events
    L.info(f"除权除息处理 {dq:,} 次；零事件只表示该回测未覆盖公司行动，不代表策略有错")
    for assumption in r.assumptions:
        L.info(assumption)
    L.info(f"期末应收分红 {r.receivables[-1]:,.2f} 元；长期无成交旧价市值 {r.stale_value[-1]:,.2f} 元")
    L.info(f"退市/长期停牌强制了结 {acct.n_force_close:,} 笔，"
           f"损益 {acct.pnl_delisted:,.0f} 元"
           f"（占已实现 {acct.pnl_delisted / acct.realized:.1%}）"
           if acct.realized else
           f"退市/长期停牌强制了结 {acct.n_force_close:,} 笔")

    npos = np.asarray(acct.shares, dtype=np.int64)
    summary["期末持仓只数"] = int((npos > 0).sum())

    # ------------------------------------------------------------------
    L.head("D. 口径敏感性 —— 收益是不是来自一个假设？")
    #
    # 下面这几个 run() 都**只**能靠 `kw` 传 market —— 别再写一遍 `market=m`。
    # `kw` 里已经有 market（第 C 节要跟正式回测同口径），再来一个就是
    # `TypeError: got multiple values for keyword argument 'market'`。
    # 这个坑之所以难发现：上面那个 try/except 会把异常吞成一句「跑不动」，
    # 只有底下这两行没包 try，才会真的炸出来 —— 于是 quick=False 的体检
    # 一路跑到 D 节末尾才报错，前面全是好话。
    if quick:
        L.info("（quick=True，已跳过）")
    else:
        base_ann = mm.get("年化收益", float("nan"))
        for label, kwa in (
            ("容量上限 5%  (宽松)", dict(max_adv_pct=0.05)),
            ("容量上限 0.5%(严格)", dict(max_adv_pct=0.005)),
            ("单边滑点 10bps", dict()),
        ):
            try:
                if "滑点" in label:
                    from .costs import Costs, SlippageOverlay
                    base_cost = kw.get("costs")
                    c = SlippageOverlay(Costs() if base_cost is None else base_cost, 10.)
                    rr = run(ds, strategy, **dict(kw, costs=c, name="体检"))
                else:
                    rr = run(ds, strategy, **dict(kw, **kwa, name="体检"))
                a = rr.metrics().get("年化收益", float("nan"))
                L.info(f"{label:<20} 年化 {a:+.2%}   （基准口径 {base_ann:+.2%}）")
            except Exception as e:                    # noqa: BLE001
                # 跑不动就记 FAIL，不记 info。这条敏感性检查原本把异常吞成
                # 一行淡淡的「跑不动」，结果是：一个 `TypeError`（参数传重了）
                # 在表上看起来跟「这个口径算不出来」一模一样，体检照报全部通过。
                # 会被吞掉的异常必须变成红的，否则这个 try 就是给 bug 打掩护。
                L.check(False, f"{label} 跑不动：{type(e).__name__}: {e}")

        L.info("注：这一档是**小资金**下的敏感性。它们之间差个零点几个百分点"
               "大多是成交路径分岔带来的噪声")
        L.info("    （严格容量反而更高一点，是因为少成交了几笔，此后的组合就不一样了）"
               "—— 看方向，别看小数点。")
        L.info("    大资金容量敏感性见 run_demo.py 第5节，数值按本次模型重新计算。")

        # 涨跌停口径：老师那道题
        a_off = run(ds, strategy, **dict(kw, name="纸面", limit_rule="off"))
        a_real = run(ds, strategy, **dict(kw, name="开盘模型", limit_rule="open_limit"))
        gap = float(np.asarray(a_off.equity)[-1] / np.asarray(a_real.equity)[-1] - 1.0)
        summary["纸面vs真实净值差"] = gap
        L.info(f"limit_rule=off 比 open_limit 高 {gap:+.2%} —— "
               f"这就是『涨停买不到』的价钱")
        if abs(gap) > 1e-9:
            L.check(True, "涨跌停口径对结果有影响（说明这条约束真的生效了）")
        else:
            L.info("  两次结果完全一样：策略大概率从不买涨停板，"
                   "或者选股域太窄，这个口径对它没约束力。")

    # ------------------------------------------------------------------
    L.head("E. 市场状态推断的可靠性")
    mk = m
    s = getattr(mk, "stats", {}) or {}
    band_px, band_n = np.unique(np.asarray(mk.band), return_counts=True)
    names = {0: "无限制", 5: "ST 5%", 10: "主板/创业板 10%",
             20: "科创板/创业板 20%", 30: "北交所 30%"}
    L.info("板幅分布（按股按日）：" + "，".join(
        f"{names.get(int(b), b)} {n / mk.band.size:.1%}"
        for b, n in zip(band_px, band_n) if n / mk.band.size > 1e-4))
    L.info(f"ST 推断（回看式，只用了过去）：{s.get('st_stocks', 0):,} 只股票被标过 ST，"
           f"占 {s.get('st_stock_days', 0):,} 个股票日，"
           f"翻转 {s.get('st_flips', 0):,} 次")
    L.info("  这份数据没有 ST 标记，状态是按『5% 板封住、且回看窗口内从未有过 "
           ">5.5% 的波动』反推的（只回看不前视）。翻转本身正常（戴帽/摘帽），"
           "推断用于下一开盘，仍有不确定性，并会影响限价与成交判定。")
    L.check(s.get("st_stocks", 0) <= 0.5 * N or s.get("st_source") == "explicit",
            f"被标 ST 的股票数（{s.get('st_stocks', 0):,}）不超过全市场一半"
            f"（噪声判别器的症状就是接近全市场）")
    L.info(f"无涨跌幅限制 {int(mk.no_limit.sum()):,} 格"
           f"（需要开盘前已知元数据；不会由当日涨幅反推）")
    L.info(f"停牌或无成交 {int(mk.suspended.sum()):,} 格"
           f"（NaN 价格与 amt==0 两种表示都算）")
    L.info(f"开盘就封涨停 {int(mk.open_at_up.sum()):,} 格 / "
           f"开盘就封跌停 {int(mk.open_at_down.sum()):,} 格 —— "
           f"这两组数就是『买不到』和『卖不掉』的直接来源")

    summary["失败项"] = L.failed
    return HealthReport(L.rows, summary)
