"""A 股日频引擎：前一收盘目标，次日先处理公司行动，再估值、解冻、风控和撮合。

run 面板模式一次计算信号；run_events 每日传入截至前一收盘的 Context。
账户/撮合/风险工厂、分配器、成本、市场与公司行动均可替换。
停牌减记只影响估值，不创造现金。显式清算需要真实事件输入。
"""

from __future__ import annotations

import numpy as np

from .account import Account
from .broker import Broker
from .context import Context
from .costs import Costs
from .strategy import Target, HOLD
from .tradability import build_market
from .components import RiskControls
from .clocks import DailyClock, clock_events, benchmark_at_steps, row_event
from .rules import AStockRules, CashAccountRules
from .provenance import callable_provenance
from ._validation import boolean_mask
from .metrics import equity_returns

__all__ = ["run", "run_events", "Result"]


# ==========================================================================
# 结果
# ==========================================================================
class Result:
    """一次回测的产出。所有序列都以 `dates` 对齐、长度 n。

    报告层（`report.py`）和体检层（`health.py`）只消费这个对象，不再碰引擎，
    所以它可以随手 pickle 到磁盘或者丢进多进程扫参。
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        m = self.metrics()
        return (f"Result({self.name!r} {self.dates[0]}..{self.dates[-1]}, "
                f"累计 {m['累计收益']:+.2%}, 年化 {m['年化收益']:+.2%}, "
                f"最大回撤 {m['最大回撤']:.2%}, 夏普 {m['夏普']:.2f}, "
                f"口径={self.limit_rule})")

    # -- 常用数列 --------------------------------------------------------
    @property
    def nav(self):
        """净值（起点 = 1）。"""
        return self.equity / self.equity[0]

    @property
    def n(self):
        return len(self.dates)

    def metrics(self):
        from .metrics import metrics
        return metrics(self)

    # -- 与另一次回测对比 ------------------------------------------------
    def compare(self, other, label_a="A", label_b="B"):
        from .metrics import compare
        return compare(self, other, label_a, label_b)


# ==========================================================================
# 目标面板的归一化
# ==========================================================================
def _as_panel(a, T, N, what):
    """把 (T,N) / (N,) / 0 长 / None 统一成 (T,N) 的 bool。

    允许 0 长和 1 长的数组（`qbt.FLAT` 用 0 长表示"什么都不选"），因为
    "空仓"这个意图用 `np.zeros(0)` 写出来最省事，引擎应该接住而不是报错。
    """
    if a is None:
        return None
    a = boolean_mask(a, what)
    if a.ndim == 1:
        if a.size == 0:
            return np.zeros((T, N), dtype=bool)
        if a.size == 1:
            return np.broadcast_to(a.reshape(1, 1), (T, N))
        if a.size != N:
            raise ValueError(f"{what} 是 (N,) 但长度 {a.size} != {N}")
        return np.broadcast_to(a[None, :], (T, N))
    if a.ndim == 2:
        if a.size == 0:
            return np.zeros((T, N), dtype=bool)
        if a.shape != (T, N):
            raise ValueError(f"{what} 是 {a.shape}，期望 ({T}, {N})")
        return a
    raise ValueError(f"{what} 的维度不对：{a.shape}")


def _resolve_target(ds, market, strategy, params, T, N, periods_per_year=242.):
    """跑一次策略，拿回一个干净的 (T,N) 目标。

    策略可以是：
      * 一个函数 `f(ctx) -> Target`
      * 一个 `Target`（常量目标）
      * None（纯 benchmark 空跑）
    """
    ctx = Context(ds, market, params)
    ctx.periods_per_year = periods_per_year
    if strategy is None:
        return Target(sel=np.zeros((T, N), dtype=bool), gross=0.0), ctx
    if isinstance(strategy, Target):
        tgt = strategy
    else:
        tgt = strategy(ctx)
    if tgt is HOLD:
        return Target(sel=np.zeros((T, N), dtype=bool), gross=0.), ctx
    if tgt is None:
        raise TypeError(
            "策略返回了 None。想维持昨天的目标请返回 qbt.HOLD，"
            "想清仓请返回 qbt.FLAT 或 ctx.cash()。")
    if not isinstance(tgt, Target):
        raise TypeError(f"策略必须返回 Target，收到 {type(tgt).__name__}")
    return tgt, ctx


def _apply_rebalance_band(broker, acct, want, px_open, equity_open, members,
                          gross_t, weight, score_t):
    """权重偏离带：老仓偏离不够大就不动，剩下的钱只在需要动的票之间重新分。

    为什么必须有这道带子：等权目标**每天随价格漂移**。今天涨得多的票权重
    自动变大，明天按等权重置就得卖掉一点——每天都在全仓换手。佣金万 2.5
    （最低 5 元）加印花税千 1（2023-08-28 后千 0.5），一年 240 个交易日，
    光成本就能吃掉十几个点。这是"回测很好、一上实盘很糟"的经典来源之一。

    三条容易写错、写错了会**静默**产生巨额换手的规则，都在这里钉死：

    1. **`want == 0` 不等于"卖"**。手里有、且**今天还在选股域里**的票，
       如果算出来的目标是 0，那是*算不出来*，不是*想清仓*：可能是当天
       `amt == 0`（停牌，容量上限就是 0），也可能是整手取整到 0。这两种
       情况都应该"不动"。若按 `want == 0` 就卖，第二天它又算出来了再买回来
       ——每天一个来回，佣金印花税照付，而净值曲线看不出任何异常。
    2. **只有真的掉出选股域的票才置 0**。`want` 在 `members` 之外本来就是 0
       （`target_qty` 只给成员定量），所以"卖出"由 `want` 自己表达，这道带子
       不需要也不应该额外去清零任何东西。
    3. **冻结之后必须重新分配预算**，不然被冻的老仓还占着钱，新买的票又按原
       目标全额下单，总仓位会超过 100%，买单被现金配给砍一刀，变成系统性欠配。
    """
    tol = broker.rebalance_tol
    if tol <= 0 or equity_open <= 0 or not members.any():
        return want
    held = acct.shares > 0
    both = held & members                     # 手里有、且还在选股域里
    if not both.any():
        return want
    with np.errstate(invalid="ignore", divide="ignore"):
        w_cur = acct.shares * px_open / equity_open
        w_tgt = want * px_open / equity_open
    # 价格缺失（停牌）会让两个权重都变成 NaN，比较结果为 False -> 误判成"要调仓"。
    # 算不出来的就当作"不动"，与规则 1 同一个道理。
    computable = np.isfinite(w_cur) & np.isfinite(w_tgt)
    close_enough = computable & (np.abs(w_tgt - w_cur) <= tol)
    frozen = both & (close_enough | ~computable | (want <= 0))
    if not frozen.any():
        return want

    out = np.where(frozen, acct.shares, want)
    free = members & ~frozen & (want > 0)     # 真的要新建/加仓的票
    if not free.any():
        return out.astype(np.int64)           # 老仓全冻，掉出选股域的由 want=0 清掉
    held_val = float(np.nansum((acct.shares * px_open)[frozen]))
    budget = gross_t * equity_open - held_val
    if budget <= 0:
        return out.astype(np.int64)           # 仓位已经满了，不新买
    q = broker.target_qty(free, equity_open, px_open,
                          gross=budget / equity_open, weight=weight,
                          score=score_t)
    out[free] = q[free]
    return out.astype(np.int64)


def _final_membership(sel, keep):
    """解进出缓冲：`final[t] = (final[t-1] & keep[t]) | sel[t]`。

    这一步**必须顺序地做**，因为它依赖的是"你手里现在有什么"，光看当天的
    分数算不出来。这是 `Universe.buffer()` 能压换手的原因。
    """
    T, N = sel.shape
    final = np.empty((T, N), dtype=bool)
    prev = np.zeros(N, dtype=bool)
    for t in range(T):
        prev = (prev & keep[t]) | sel[t]
        final[t] = prev
    return final


# ==========================================================================
# 基准
# ==========================================================================
def ew_market_returns(ds, close_adj=None):
    """全市场等权日收益。

    **这不是沪深300** —— 这份数据里没有指数。等权全市场是最容易算、也最
    难被策略"作弊"赢过的基准（它自动含了小盘股的风险溢价）。要跟指数比，
    得自己把指数数据准备好传进来。
    """
    a = np.asarray(ds.close_adj if close_adj is None else close_adj, dtype=np.float64)
    out = np.full(a.shape[0], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = a[1:] / a[:-1] - 1.0
    valid = np.isfinite(r)
    n = valid.sum(axis=1)
    s = np.where(valid, r, 0.0).sum(axis=1)
    out[1:] = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    return out


# ==========================================================================
# 主循环
# ==========================================================================
from .markets import market_entry


@market_entry
def run(ds, strategy=None, start=None, end=None, capital=1_000_000.0,
        params=None, limit_rule="open_limit", costs=None, max_adv_pct=0.02,
        rebalance_tol=0.02, div_policy=None, max_stale_days=60,
        name=None, verbose=False, market=None, record_trades=True,
        market_kw=None, account_factory=Account, broker_factory=Broker,
        risk_factory=RiskControls, allocation=None, corporate_actions=None,
        stale_policy="write_down", capacity_mode="lagged_adv", adv_window=20,
        mode="panel", clock=None, asset_rules=None, account_rules=None, **target_kw):
    """跑一次回测。

    ds            `qbt.load()` 的返回值
    strategy      `f(ctx) -> Target`；None 表示空仓（拿来看基准）
    start / end   "2021-01-01" 这样的日期字符串，闭区间；落在非交易日会吸附
    capital       初始资金（元）
    limit_rule    涨跌停口径：off / sealed / broken_board_close / open_limit
    costs         `qbt.Costs(...)`
    max_adv_pct   单票单日最多吃掉当日成交额的百分之几
    rebalance_tol 已持有且仍在选股域内的票，权重偏离超过这个值才调整。
                  设成 1.0 就是"买入后不再调仓"，做账务核对时很有用。
    div_policy    events（显式事件）/ reinvest（缺事件时默认近似）/ ignore
    max_stale_days 无成交超过阈值时减记估值或披露旧价，不生成成交和现金

    最常用的两种跑法，用来回答老师那道题：

        a = qbt.run(d, 策略, limit_rule="off")     # 纸面收益
        b = qbt.run(d, 策略)                       # 真实（默认 open_limit）
        a.compare(b)                               # 差在哪、差多少
    """
    if target_kw:
        # 拼错一个参数名、或者参数被**静默吃掉**，是回测里最危险的一类 bug：
        # 回测照跑、曲线照画，结果却是另一套口径，而且没有任何提示。
        # 所以这里直接报错，并把最可能的那个拼写错误点出来。
        bad = ", ".join(sorted(target_kw))
        hint = ""
        if "rebalance" in bad or "tol" in bad:
            hint = "（调仓偏离带是 rebalance_tol=）"
        raise TypeError(f"run() 收到不认识的参数：{bad}{hint}")

    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("capital 必须是正的有限数")
    clock = DailyClock() if clock is None else clock
    account_rules = CashAccountRules() if account_rules is None else account_rules
    if callable(getattr(account_rules,'reset',None)):
        account_rules.reset()
    asset_rules = AStockRules() if asset_rules is None else asset_rules
    if mode not in ("panel", "events"):
        raise ValueError("mode 只能是 panel / events")
    if stale_policy not in ("hold", "write_down"):
        raise ValueError("stale_policy 只能是 hold / write_down，不支持无成交自动变现")
    if max_stale_days < 0:
        raise ValueError("max_stale_days 必须 >= 0")
    if div_policy is None:
        div_policy = "events" if corporate_actions is not None else "reinvest"
    if div_policy not in ("events", "reinvest", "ignore"):
        raise ValueError("div_policy 只能是 events / reinvest / ignore")
    if corporate_actions is not None and div_policy != "events":
        raise ValueError("提供 corporate_actions 时请使用 events，避免事件被忽略或重复入账")
    if div_policy == "events" and corporate_actions is None:
        raise ValueError("events 口径需要 corporate_actions")
    m = market if market is not None else build_market(
        ds, log=(print if verbose else None), **({} if market_kw is None else market_kw))
    if m.ds is not ds:
        raise ValueError('market绑定的是另一份Dataset；请为当前数据重新构建market，不能复用旧行情对象')
    T, N = ds.T, ds.N
    if mode == "events" and callable(getattr(strategy, "reset", None)):
        strategy.reset()

    provenance=dict(strategy=callable_provenance(strategy),engine=callable_provenance(getattr(run,'__wrapped__',run)))
    tgt, ctx = _resolve_target(ds, m, strategy if mode == "panel" else None, params, T, N, clock.periods_per_year)
    sel = _as_panel(tgt.sel, T, N, "Target.sel")
    if sel is None:
        raise ValueError("Target.sel 不能是 None；空仓请用 qbt.FLAT 或 ctx.cash()")
    keep = _as_panel(tgt.keep if tgt.keep is not None else sel, T, N, "Target.keep")
    score = _as_panel_score(tgt.score, T, N, tgt.weight)
    final = _final_membership(sel, keep)

    # 目标总仓位：(T,) —— 允许策略逐日调仓位
    gross = np.broadcast_to(np.asarray(tgt.gross, dtype=np.float64),
                            (T,)).astype(np.float64)
    account_rules.validate_gross(gross)

    # -- 时间范围 --------------------------------------------------------
    events, indices = clock_events(clock, ds, start, end)
    i0, i1 = int(indices[0]), int(indices[-1])

    # -- 组件 ------------------------------------------------------------
    acct = account_factory(capital, N)
    broker = broker_factory(m, costs=costs if costs is not None else Costs(),
                    limit_rule=limit_rule, max_adv_pct=max_adv_pct,
                    rebalance_tol=rebalance_tol, record_trades=record_trades,
                    allocation=allocation, capacity_mode=capacity_mode, adv_window=adv_window,
                    asset_rules=asset_rules, account_rules=account_rules)
    risk = risk_factory(N, tgt.stops)
    action_book = corporate_actions.prepare(ds) if corporate_actions is not None else None
    # 止损出局之后**不能第二天就买回来**：`final` 是纯面板、不含状态，不加
    # 这个禁令的话策略会因为"还选着它"而当天卖、次日买，来回摩擦。禁令一直
    # 有效到该股离开选股域为止。
    banned = np.zeros(N, dtype=bool)

    n_days = len(events)
    eq = np.empty(n_days + 1)
    cash = np.empty(n_days + 1)
    npos = np.zeros(n_days + 1, dtype=np.int32)
    gross_expo = np.zeros(n_days + 1)
    fees_cum = np.zeros(n_days + 1)
    turnover_cum = np.zeros(n_days + 1)
    receivables = np.zeros(n_days + 1)
    stale_value = np.zeros(n_days + 1)
    # 被涨停/跌停挡掉的资金，以及它们当天本来能赚多少 —— 老师那道题的答案
    blk_buy_amt = np.zeros(n_days)
    blk_buy_bounce = np.zeros(n_days)
    blk_sell_amt = np.zeros(n_days)
    blk_sell_bounce = np.zeros(n_days)
    blk_buy_n = np.zeros(n_days, dtype=np.int32)
    blk_sell_n = np.zeros(n_days, dtype=np.int32)
    trades = [] if record_trades else None

    eq[0] = capital
    cash[0] = capital
    days = ds.dates[np.r_[i0-1, indices]]

    cumadj = ds.cumadj_ff
    event_members = np.zeros(N, bool)
    event_gross, event_score, event_weight = 0., None, "equal"

    def mark_prices(prices):
        px = acct.price_at(prices).copy()
        if stale_policy == "write_down" and max_stale_days:
            px[(acct.stale > max_stale_days) & (acct.shares > 0)] = 0.
        return px

    previous_action_i = i0-1
    if action_book is not None:
        action_book.start_at(previous_action_i)
    for k, event in enumerate(events):
        i = event.index
        if i > previous_action_i + 1:
            # 不在这些行调用策略或撮合，但分红、估值、冻结与风险高水位照常推进。
            for row in range(previous_action_i+1,i):
                idle = row_event(clock,ds,row)
                if action_book is not None:
                    action_book.advance(acct,ds,previous_action_i,row)
                else:
                    acct.apply_corporate_actions(cumadj[previous_action_i],cumadj[row],ds.open[row],policy=div_policy)
                account_rules.start_bar(acct,idle,broker.asset_rules)
                prices = np.where(m.suspended[row],np.nan,ds.close[row])
                acct.mark(prices,advance=idle.new_session)
                if action_book is not None:
                    action_book.record_close(acct,row)
                acct._valuation = mark_prices(prices).copy()
                risk.mark_close(acct,np.where(m.suspended[row],np.nan,
                    np.asarray(ds.close_ff[row],float)*cumadj[row]))
                previous_action_i = row
            event = row_event(clock,ds,i)
        if mode == "events":
            event_ctx = ctx.sub(0, i)
            event_ctx.portfolio = dict(acct.snapshot(), shares=acct.shares.copy())
            event_ctx.now, event_ctx.execution_time = event.previous_timestamp, event.timestamp
            decision = strategy(event_ctx) if strategy is not None else HOLD
            if decision is not HOLD:
                if not isinstance(decision, Target):
                    raise TypeError("事件策略须返回 Target / HOLD / FLAT，不能返回 None")
                es = _as_panel(decision.sel, i, N, "Target.sel")
                if es is None:
                    raise ValueError("Target.sel 不能为 None")
                ek = _as_panel(decision.keep if decision.keep is not None else es, i, N, "Target.keep")
                event_members = (event_members & ek[-1]) | es[-1]
                eg = np.asarray(decision.gross, float)
                if eg.ndim > 1 or (eg.ndim == 1 and eg.size != i):
                    raise ValueError("事件 gross 应为标量或与前缀长度一致")
                event_gross = float(eg[-1] if eg.ndim else eg)
                account_rules.validate_gross(event_gross)
                ep = _as_panel_score(decision.score, i, N, decision.weight)
                event_score = None if ep is None else ep[-1]
                event_weight = decision.weight
                risk.configure(decision.stops)
        # -- 1) 开盘估值 ---------------------------------------------------
        # 公司行动必须先入账，再决定开盘权益和下单预算。
        if action_book is not None:
            action_book.advance(acct, ds, previous_action_i, i)
        else:
            acct.apply_corporate_actions(cumadj[previous_action_i], cumadj[i], ds.open[i], policy=div_policy)
        previous_action_i = i
        # 被禁止作为成交证据的旧/未来报价，也不能抬高已有持仓并放大其他买单。
        px_open = mark_prices(np.where(m.quote_suspended[i],np.nan,ds.open[i]))
        equity_open = account_rules.equity(acct, px_open)

        # -- 3) 解冻（T+1）-------------------------------------------------
        account_rules.start_bar(acct, event, broker.asset_rules)

        # -- 4) 止损（按开盘价评估）------------------------------------------
        adj_open = np.where(m.quote_suspended[i],np.nan,
                            np.asarray(ds.open_ff[i], dtype=np.float64) * cumadj[i])
        banned |= boolean_mask(risk.exits(acct, adj_open), 'risk.exits')

        # -- 5) 目标（策略在 i-1 收盘后的决定）--------------------------------
        t_sig = i - 1
        members = event_members if mode == "events" else final[t_sig]
        gross_t = event_gross if mode == "events" else gross[t_sig]
        weight_t = event_weight if mode == "events" else tgt.weight
        score_t = event_score if mode == "events" else (None if score is None else score[t_sig])
        sel_t = members & ~banned
        if action_book is not None:
            sel_t &= ~action_book.closed
        # 已经离开选股域的股票解除禁令，允许以后再买回来
        banned &= members
        if gross_t > 0 and sel_t.any():
            q = broker.target_qty(sel_t, equity_open, px_open,
                                  gross=gross_t, weight=weight_t, score=score_t)
        else:
            q = np.zeros(N, dtype=np.int64)

        want = _apply_rebalance_band(
            broker, acct, q, px_open, equity_open, sel_t, gross_t, weight_t, score_t)
        if gross_t <= 0:
            want[:] = 0

        # -- 6) 先卖后买 -----------------------------------------------------
        info = broker.execute(acct, i, want, ds.dates[i])
        if limit_rule == "broken_board_close":
            close_info = broker.execute(acct, i, want, ds.dates[i], phase="close")
            if info["fills"] is not None:
                info["fills"].extend(close_info["fills"] or [])

        # -- 记录被挡住的收益（老师那道题的答案）------------------------------
        # 注意用 broker 返回的**成交前**数组：账本已经被改过了。
        bm = info["buy_mask"]
        if bm.any():
            w_qty = info["want_buy_arr"][bm]
            w = px_open[bm] * w_qty
            bounce = ctx_bounce(ds, m, i)
            tot_w = float(np.nansum(w))
            bb = float(np.nansum(w * bounce[bm]) / tot_w) if tot_w > 0 else 0.0
            blk_buy_amt[k] = float(np.nansum(w))
            blk_buy_bounce[k] = bb if np.isfinite(bb) else 0.0
            blk_buy_n[k] = int(bm.sum())
        sm = info["sell_mask"]
        if sm.any():
            w_s = info["want_sell_arr"][sm]
            amt_s = px_open[sm] * w_s
            tot_s = float(np.nansum(amt_s))
            amt = float(np.nansum(amt_s))
            blk_sell_amt[k] = amt
            blk_sell_n[k] = int(sm.sum())
            # 卖不掉的代价：本想按开盘价出手，结果只能扛到收盘。
            # 负数 = 又跌了（被套得更深），正数 = 反而涨了（少赚）。
            sb = ctx_bounce(ds, m, i, up=False)
            ss = float(np.nansum(amt_s * sb[sm]) / tot_s) if tot_s > 0 else 0.0
            blk_sell_bounce[k] = ss if np.isfinite(ss) else 0.0
        if trades is not None and info["fills"]:
            trades.extend(info["fills"])

        # -- 7) 收盘打标 -----------------------------------------------------
        acct.mark(np.where(m.suspended[i], np.nan, ds.close[i]), advance=event.new_session)
        if action_book is not None:
            action_book.record_close(acct,i)
        px_close = mark_prices(np.where(m.suspended[i], np.nan, ds.close[i]))
        acct._valuation = px_close.copy()
        risk.mark_close(acct, np.where(m.suspended[i],np.nan,
                        np.asarray(ds.close_ff[i], dtype=np.float64) * cumadj[i]))
        if max_stale_days:
            stale = (acct.shares > 0) & (acct.stale > max_stale_days)
            stale_value[k + 1] = float(np.nansum((acct.shares * acct.last_px)[stale]))

        e = account_rules.equity(acct, px_close)
        eq[k + 1] = e
        cash[k + 1] = acct.cash
        npos[k + 1] = int((acct.shares > 0).sum())
        gross_expo[k + 1] = 0.0 if e <= 0 else acct.position_value(px_close) / e
        fees_cum[k + 1] = acct.fees
        turnover_cum[k + 1] = acct.turnover
        receivables[k + 1] = float(acct.receivables.sum())
        account_rules.check(acct)

        if verbose and (k % 250 == 0 or k == n_days - 1):
            print(f"    [{k+1:4d}/{n_days}] {ds.dates[i]}  净值 {e/capital:.4f}  "
                  f"持仓 {npos[k+1]:4d}  费用 {acct.fees:,.0f}")

    # -- 基准 --------------------------------------------------------------
    bench_ret, bench_nav = benchmark_at_steps(ds, indices, m)

    return Result(
        dates=days, equity=eq, cash=cash, n_positions=npos, gross=gross_expo,
        fees_cum=fees_cum, turnover_cum=turnover_cum,
        ret=equity_returns(eq),
        bench_ret=bench_ret, bench_nav=bench_nav * eq[0],
        blocked_buy_amt=blk_buy_amt, blocked_buy_bounce=blk_buy_bounce,
        blocked_sell_amt=blk_sell_amt, blocked_sell_bounce=blk_sell_bounce,
        blocked_buy_n=blk_buy_n, blocked_sell_n=blk_sell_n,
        trades=trades, account=acct, market=m, ds=ds,
        limit_rule=limit_rule, capital=float(capital),
        div_policy=div_policy, max_adv_pct=float(max_adv_pct),
        stale_policy=stale_policy, max_stale_days=max_stale_days,
        stale_value=stale_value, receivables=receivables, final_prices=px_close,
        capacity_mode=capacity_mode, adv_window=adv_window, mode=mode,
        periods_per_year=clock.periods_per_year, step_indices=indices,
        clock_name=type(clock).__name__, asset_rule_name=type(asset_rules).__name__,
        account_rule_name=type(account_rules).__name__,
        cost_description=repr(broker.costs),
        execution_provenance=provenance,
        settlements=tuple(action_book.settlement_log) if action_book is not None else (),
        assumptions=[
            "公司行动：显式分红/送转事件" if div_policy == "events" else
            ("公司行动：复权因子再投资近似，非真实现金分红" if div_policy == "reinvest" else "公司行动：忽略（诊断口径）"),
            "ST：" + m.stats.get("st_source", "自定义市场"),
            "报价源时间：" + m.stats.get("quote_time_source", "未提供"),
            f"容量：{capacity_mode}，买卖对称 {max_adv_pct:.1%}；当日额只作成交上界",
            f"长期无成交：{stale_policy}，阈值 {max_stale_days} 会话；不会自动变现",
            f"时钟 {type(clock).__name__}，年化 {clock.periods_per_year:g} 个估值区间；资产 {type(asset_rules).__name__}；账户 {type(account_rules).__name__}",
            "sealed 为事后乐观对照" if limit_rule == "sealed" else
            ("开板后收盘成交：收盘资金不能用于当日开盘买入" if limit_rule == "broken_board_close" else "默认开盘成交模型；未模拟队列或日内冲击"),
        ],
        name=name or getattr(strategy, "__name__", None) or "策略",
        note=tgt.note, window=(i0, i1),
        stats={"n_blocked_buy": int(blk_buy_n.sum()),
               "n_blocked_sell": int(blk_sell_n.sum()),
               "blocked_buy_amt": float(blk_buy_amt.sum()),
               "blocked_buy_pnl": float((blk_buy_amt * blk_buy_bounce).sum())},
    )


def run_events(ds, strategy, **kwargs):
    """逐日回调：只传入截至前一收盘的 Context；HOLD 沿用上次目标。"""
    return run(ds, strategy, mode="events", **kwargs)


def _as_panel_score(s, T, N, weight):
    if s is None:
        if weight in ("score", "value"):
            raise ValueError(f'weight="{weight}" 需要同时给 score= 面板')
        return None
    s = np.asarray(s, dtype=np.float64)
    if s.ndim == 1:
        if s.size != N:
            raise ValueError(f"score 是 (N,) 但长度 {s.size} != {N}")
        return np.broadcast_to(s[None, :], (T, N))
    if s.shape != (T, N):
        raise ValueError(f"score 是 {s.shape}，期望 ({T}, {N})")
    return s


def ctx_bounce(ds, market, i, up=True):
    """i 日"开盘就封涨停"（up=False 则是跌停）的票，当天开盘到收盘的涨幅。

    这是老师那道题的第二个数：涨停买不到，那**等到能买**的时候（也就是次日
    开盘）进来，收益还剩多少。榜单上被挡住的这批票当天还在继续涨，所以
    「本来能赚却没赚到」= 被挡金额 × 这里的涨幅。
    """
    o = np.asarray(ds.open_ff[i], dtype=np.float64)
    c = np.asarray(ds.close_ff[i], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = c / o - 1.0
    mask = market.open_at_up[i] if up else market.open_at_down[i]
    return np.where(mask, r, np.nan)
