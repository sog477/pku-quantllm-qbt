"""目标持仓转成实际成交。先卖后买，股票 T+1，资金不足按比例缩单。

买卖两侧都受同一容量比例约束；默认过去 ADV 决定预算，当日额仅作成交上界。
容量截断与现金配给后重新检查申报单位，沪深普通股买入100股整数倍，科创板200股起、北交所100股起，后二者1股递增。
开板后收盘成交分为 open/close 两阶段，收盘卖出款不能资助当日开盘买入。
sealed 是事后乐观对照。没有盘口数据时不能保证队列成交。
"""

from __future__ import annotations

import numpy as np

from .costs import Costs, evaluate_cost
from .tradability import LIMIT_RULES
from .components import WeightAllocation
from .rules import AStockRules, CashAccountRules
from ._validation import boolean_mask

__all__ = ["Broker"]

_STAR_PREFIX = ("688", "689")


def _cap_qty(cap_amt, px):
    """由"金额上限"和成交价算出"股数上限"。停牌价为 0 -> 上限 0。"""
    safe = np.where(np.asarray(px, dtype=np.float64) > 0, px, np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.floor(np.asarray(cap_amt, dtype=np.float64) / safe)
    return np.nan_to_num(q, nan=0.0, posinf=0.0).astype(np.int64)


class Broker:
    """成交撮合 + 各种约束。

    limit_rule   涨跌停口径，见模块 docstring
    max_adv_pct  单票单日成交量上限，占当日成交额的比例。默认 2%。
                 这是个**容量约束**，不做的话大资金策略的回测是纯幻觉。
    lot          买入最小单位（股）。主板/创业板 100
    star_lot     科创板 688/689 起买 200 股
    rebalance_tol 已持有且仍在选股域内的票，权重偏离超过这个值才调整。
                 不设阈值的话，等权目标每天随价格漂移 -> 每天全仓换手 ->
                 佣金印花税把 alpha 吃光。
    """

    __slots__ = ("m", "costs", "limit_rule", "max_adv_pct", "lots",
                 "rebalance_tol", "record_trades", "stats", "minimums", "allocation", "adv_window", "capacity_mode", "asset_rules", "account_rules", "_capacity_index", "_capacity_used")

    def __init__(self, market, costs=None, limit_rule="open_limit",
                 max_adv_pct=0.02, lot=100, star_lot=200,
                 rebalance_tol=0.02, record_trades=True, allocation=None,
                 adv_window=20, capacity_mode="lagged_adv", asset_rules=None, account_rules=None):
        if limit_rule not in LIMIT_RULES:
            raise ValueError(f"limit_rule 只能是 {LIMIT_RULES}，收到 {limit_rule!r}")
        self.m = market
        self.costs = costs if costs is not None else Costs()
        self.limit_rule = limit_rule
        self.max_adv_pct = float(max_adv_pct)
        if not np.isfinite(self.max_adv_pct) or not 0 < self.max_adv_pct <= 1:
            raise ValueError("max_adv_pct 必须位于 (0, 1]")
        if capacity_mode not in ("lagged_adv", "realized_day"):
            raise ValueError("capacity_mode 只能是 lagged_adv / realized_day")
        self.capacity_mode = capacity_mode
        self.adv_window = int(adv_window)
        if self.adv_window < 1:
            raise ValueError("adv_window 必须 >= 1")
        self.allocation = allocation if allocation is not None else WeightAllocation()
        self.rebalance_tol = float(rebalance_tol)
        if not np.isfinite(self.rebalance_tol) or self.rebalance_tol < 0:
            raise ValueError("rebalance_tol 必须是非负有限数")
        self.record_trades = bool(record_trades)

        self.asset_rules = (AStockRules(lot, star_lot) if asset_rules is None else asset_rules).bind(market.ds)
        self.account_rules = CashAccountRules() if account_rules is None else account_rules
        self.lots, self.minimums = self.asset_rules.steps, self.asset_rules.minimums
        self.stats = {}
        self._capacity_index = None
        self._capacity_used = np.zeros(market.ds.N)

    def __repr__(self):
        return (f"Broker(口径={self.limit_rule}, 容量={self.max_adv_pct:.1%}成交额, "
                f"{self.costs})")

    # ------------------------------------------------------------------
    # 目标股数
    # ------------------------------------------------------------------
    def target_qty(self, members, equity, px, gross=1.0, weight="equal",
                   score=None, base=None):
        """把"想持有哪些票"变成"想持有多少股"。

        members  (N,) bool —— 今天的目标持仓集合（进出缓冲已经解完）
        equity   执行时点的总权益
        px       执行价（(N,) 数组）
        weight   "equal" / "score" / "value"
        base     已持有的股票（`weight="keep"` 时用来保持现有权重，见下）

        返回 (N,) int64 目标股数。整手向下取整——**取整只会让你买得更少，
        不会让你超支**，这个方向很重要。
        """
        n = int(members.sum())
        out = np.zeros(len(members), dtype=np.int64)
        if n == 0 or gross <= 0:
            return out

        budget = gross * float(equity)
        w = np.asarray(self.allocation(members, weight=weight, score=score), float)
        if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1):
            raise ValueError("分配组件必须返回成员内非负且和为 1 的权重")

        p = np.asarray(px, dtype=np.float64)[members]
        money = w * budget
        with np.errstate(invalid="ignore", divide="ignore"):
            q = money / p
        q = np.where(np.isfinite(q) & (p > 0), q, 0.0)
        lots = self.lots[members]
        out[members] = (np.floor(q / lots) * lots).astype(np.int64)
        out[out < self.minimums] = 0
        return out

    def _round_buy(self, q):
        return self.asset_rules.round_buy(q)

    def _round_sell(self, q, held):
        return self.asset_rules.round_sell(q, held)

    # ------------------------------------------------------------------
    # 撮合
    # ------------------------------------------------------------------
    def capacity_amount(self, i):
        ds = self.m.ds
        amt_i = np.nan_to_num(np.asarray(ds.amt[i], dtype=np.float64), nan=0.0)
        if self.capacity_mode == "lagged_adv":
            def build_adv(d):
                a = np.nan_to_num(np.asarray(d.amt, float), nan=0., posinf=0., neginf=0.)
                sums = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])
                row = np.arange(a.shape[0])
                start = np.maximum(row - self.adv_window, 0)
                return (sums[row] - sums[start]) / np.maximum(row - start, 1)[:, None]
            if hasattr(ds, "cached"):
                budget_amt = ds.cached(("broker_adv", self.adv_window), build_adv)[i]
            else:
                budget_amt = np.mean(ds.amt[max(0, i-self.adv_window):i], axis=0) if i else np.zeros_like(amt_i)
        else:
            budget_amt = amt_i
        # 过去 ADV 决定订单预算；当日实际成交额仅作事后成交可行性的上界。
        cap_b_amt = self.max_adv_pct * np.minimum(np.maximum(budget_amt, 0), amt_i)
        cap_s_amt = cap_b_amt

        return cap_b_amt

    def begin_capacity(self, i):
        """同一行情行共用已成交金额；也可由直接订单组件共享此预算状态。"""
        if self._capacity_index != i:
            self._capacity_used.fill(0.)
            self._capacity_index = i
        return self._capacity_used

    def execute(self, acct, i, want_qty, day, phase="open"):
        """执行日 i：把持仓推向 `want_qty`。返回当天的成交与拦截明细。"""
        m = self.m
        ds = m.ds
        lots = self.lots
        px_buy = np.nan_to_num(np.asarray(m.fill_price(i, "buy", self.limit_rule),
                                          dtype=np.float64), nan=0.0)
        px_sell = np.nan_to_num(np.asarray(m.fill_price(i, "sell", self.limit_rule),
                                           dtype=np.float64), nan=0.0)
        buyable = boolean_mask(self.asset_rules.can_buy(m, i, self.limit_rule),'can_buy',(ds.N,)).copy()
        sellable = boolean_mask(self.asset_rules.can_sell(m, i, self.limit_rule),'can_sell',(ds.N,)).copy()
        limit_buyable, limit_sellable = buyable.copy(), sellable.copy()
        if self.limit_rule == "broken_board_close":
            limited = self.asset_rules.limit_flags
            opening = np.nan_to_num(np.asarray(ds.open[i],float),nan=0.)
            px_buy = np.where(limited,px_buy,opening)
            px_sell = np.where(limited,px_sell,opening)
            delayed_b = limited & m.open_at_up[i] & ~m.close_at_up[i]
            delayed_s = limited & m.open_at_down[i] & ~m.close_at_down[i]
            buyable &= delayed_b if phase == "close" else ~delayed_b
            sellable &= delayed_s if phase == "close" else ~delayed_s
        elif phase != "open":
            raise ValueError("只有 broken_board_close 支持收盘撮合阶段")
        # 同一证券的买卖、开收盘阶段及重复execute都不能重新领一份预算。
        used = self.begin_capacity(i)
        cap_b_amt = cap_s_amt = np.maximum(self.capacity_amount(i)-used,0.)

        held = acct.shares
        avail = self.account_rules.sellable(acct)
        want = np.asarray(want_qty, dtype=np.int64)
        want_sell = np.maximum(held - want, 0)
        want_buy = np.maximum(want - held, 0)

        # -- 卖出（必须先于买入：A股资金 T+0，卖出的钱当天能拿来买）----------
        # 申报取整交给资产规则；主板完整零股尾数可以单独或随整手一起卖。
        sq = want_sell.copy()
        sq = np.minimum(sq, avail)
        sq = np.minimum(sq, _cap_qty(cap_s_amt, px_sell))
        sq = self._round_sell(sq, held)
        sq = np.minimum(sq, avail)
        sq = self._round_sell(sq, held)
        sq = np.where(sellable, sq, 0).astype(np.int64)
        sq = np.maximum(sq, 0)

        sj = np.flatnonzero(sq > 0)
        sold_amt = 0.0
        fills = [] if self.record_trades else None
        if sj.size:
            amt_s = sq[sj] * px_sell[sj]
            fee_s = evaluate_cost(self.costs.sell, amt_s, day, codes=np.asarray(ds.codes)[sj],quantity=sq[sj])
            for k, j in enumerate(sj):
                if float(self.account_rules.buying_power(acct)) + amt_s[k] < fee_s[k]-1e-9:
                    sq[j] = 0
                    continue
                self.account_rules.sell(acct, int(j), int(sq[j]), float(px_sell[j]), float(fee_s[k]),
                          float(ds.cumadj_ff[i, j]), self.asset_rules)
                used[j] += float(amt_s[k])
                sold_amt += float(amt_s[k])
                if fills is not None:
                    fills.append((i, int(j), -1, int(sq[j]),
                                  float(px_sell[j]), float(fee_s[k])))

        # -- 买入 -----------------------------------------------------------
        bq = (want_buy // lots) * lots
        bq = np.minimum(bq, _cap_qty(cap_b_amt, px_buy))
        bq = self._round_buy(bq)
        bq = np.where(buyable, bq, 0).astype(np.int64)
        bq = self._ration(np.maximum(bq, 0), px_buy, self.account_rules.buying_power(acct), day)

        bj = np.flatnonzero(bq > 0)
        bought_amt = 0.0
        if bj.size:
            amt_b = bq[bj] * px_buy[bj]
            fee_b = evaluate_cost(self.costs.buy, amt_b, day, codes=np.asarray(ds.codes)[bj],quantity=bq[bj])
            for k, j in enumerate(bj):
                self.account_rules.buy(acct, int(j), int(bq[j]), float(px_buy[j]), float(fee_b[k]),
                         float(ds.cumadj_ff[i, j]), self.asset_rules)
                used[j] += float(amt_b[k])
                if fills is not None:
                    fills.append((i, int(j), 1, int(bq[j]),
                                  float(px_buy[j]), float(fee_b[k])))
            bought_amt = float(amt_b.sum())

        # -- 拦截明细（给报告和老师那道题用）--------------------------------
        #
        # 「被挡住」必须拆成两种，混在一起会把老师那道题答错：
        #
        #   停牌挡住  —— 今天根本没法交易（那格 amt==0 / 价格 NaN）。跟涨跌停
        #               没关系，而且**任何** limit_rule 都挡它，包括 "off"。
        #   涨跌停挡住 —— 只在真实口径下发生。这才是「涨停买不到 / 跌停卖不掉」，
        #               也是「纸面 vs 真实」差异的唯一来源。
        #
        # 判据是拿最宽松的 "off" 档做基准：`off` 都买不了的 → 停牌；`off` 能买
        # 而本档买不了的 → 涨跌停。这样 limit_rule="off" 的那次回测里，
        # blocked_* 恒为 0，报告上的「被挡个股涨幅」不会再被一堆停牌票的 NaN
        # 稀释成 0.0000%。
        buyable_off = boolean_mask(self.asset_rules.can_buy(m, i, "off"),'can_buy(off)',(ds.N,))
        sellable_off = boolean_mask(self.asset_rules.can_sell(m, i, "off"),'can_sell(off)',(ds.N,))
        invalid_open = getattr(m, 'invalid_limit_open', None)
        bad_open = invalid_open(i) if callable(invalid_open) else np.zeros(ds.N,bool)
        susp_b = (want_buy > 0) & ~buyable_off
        susp_s = (want_sell > 0) & ~sellable_off
        blocked_b = (want_buy > 0) & buyable_off & ~limit_buyable & ~bad_open
        blocked_s = (want_sell > 0) & sellable_off & ~limit_sellable & ~bad_open
        info = {
            "bought_amt": bought_amt,
            "sold_amt": sold_amt,
            "n_buy": int(bj.size),
            "n_sell": int(np.count_nonzero(sq)),
            "want_buy": int((want_buy > 0).sum()),
            "want_sell": int((want_sell > 0).sum()),
            # 被涨跌停挡住的（老师那道题的主角）
            "blocked_buy_limit": int(blocked_b.sum()),
            "blocked_sell_limit": int(blocked_s.sum()),
            "blocked_buy_qty": int(want_buy[blocked_b].sum()),
            "blocked_sell_qty": int(want_sell[blocked_s].sum()),
            # 被停牌挡住的（另一回事，报告里单独一行）
            "suspended_buy": int(susp_b.sum()),
            "suspended_sell": int(susp_s.sum()),
            "invalid_open_buy": int(((want_buy > 0) & ~buyable & bad_open).sum()),
            "invalid_open_sell": int(((want_sell > 0) & ~sellable & bad_open).sum()),
            # 想买却一股没买到（含容量和现金不够）
            "n_buy_zero": int(((want_buy > 0) & (bq == 0)).sum()),
            "n_buy_partial": int(((want_buy > 0) & (bq > 0) & (bq < want_buy)).sum()),
            "buy_mask": blocked_b,
            "sell_mask": blocked_s,
            # 成交前的"想要多少股"，减仓/加仓金额由引擎拿去算被挡住的收益
            "want_buy_arr": want_buy,
            "want_sell_arr": want_sell,
            "px_buy": px_buy,
            "px_sell": px_sell,
            "fills": fills,
        }
        return info

    # ------------------------------------------------------------------
    def _ration(self, bq, px, cash, day):
        """共同缩放所有买单；费用/最小申报导致不连续时，用有界二分求可负担数量。"""
        if not np.isfinite(cash) or cash < -1e-6:
            raise AssertionError(f"买入前可用现金无效：{cash}")
        cash = max(float(cash), 0.)
        q = self._round_buy(bq)
        if not q.any() or cash == 0:
            return np.zeros_like(q)

        def total(candidate):
            amount = candidate * px
            fee = evaluate_cost(self.costs.buy, amount, day, codes=self.m.ds.codes,quantity=candidate)
            return float(amount.sum() + fee.sum())

        needed = total(q)
        if needed <= cash + 1e-9:
            return q
        candidate = self._round_buy(np.floor(q * (cash / needed) * .999).astype(np.int64))
        if total(candidate) <= cash + 1e-9:
            return candidate
        lo, hi = 0., cash / needed
        best = np.zeros_like(q)
        for _ in range(60):
            mid = (lo + hi) / 2.
            candidate = self._round_buy(np.floor(q * mid).astype(np.int64))
            if total(candidate) <= cash + 1e-9:
                best, lo = candidate, mid
            else:
                hi = mid
        return best
