"""原始价格、整数股、现金、应收分红和冻结股的账户。

显式公司行动由 actions.py 处理：现金分红先计应收，再按支付日期进入现金；
送转调整股数并保留可交易日期约束。cost_raw 用于货币损益，cost_adj 用于
复权收益口径的止损。reinvest 是缺少事件数据时的近似，不代表真实现金流水。

长期无报价不会自动产生清算现金；force_close 仅用于用户提供的已知清算事件。
"""

from __future__ import annotations

import numpy as np
from ._validation import integer as _integer

__all__ = ["Account", "DIV_POLICIES"]

DIV_POLICIES = ("reinvest", "ignore")
_FACTOR_TOL = 1e-4          # 因子只有 4 位小数，小于这个的跳变是舍入噪声
_MAX_SHARES = int(np.iinfo(np.int64).max)


class Account:
    """一个组合的全部状态。

    账本里的每个量都是"当前时刻"的，重置一次就是一次新的回测。
    """

    __slots__ = ("cash", "shares", "frozen", "cost_adj", "cost_raw", "receivables", "restricted", "last_px", "stale",
                 "realized", "fees", "turnover", "n_buy", "n_sell",
                 "n_div_events", "shares_issued", "pnl", "n_force_close",
                 "pnl_delisted", "N", "init_cash", "_valuation", "income_adj", "taxes", "tax_payables", "risk_rebases", "position_generation")

    def __init__(self, cash, n_stocks):
        self.N = _integer(n_stocks,1,_MAX_SHARES,'n_stocks')
        if not np.isfinite(cash) or cash < 0 or self.N <= 0:
            raise ValueError("账户现金须为非负有限数，证券数须为正")
        self.init_cash = float(cash)
        self._valuation = None
        self.income_adj = np.zeros(self.N, dtype=np.float64)
        self.taxes = 0.
        self.tax_payables = np.zeros(self.N,dtype=float)
        self.risk_rebases=np.zeros(self.N,dtype=float)
        self.position_generation=np.zeros(self.N,dtype=np.int64)
        self.cash = float(cash)
        self.shares = np.zeros(self.N, dtype=np.int64)
        self.frozen = np.zeros(self.N, dtype=np.int64)       # T+1：当日买入，次日才可卖
        self.restricted = np.zeros(self.N, dtype=np.int64)   # 尚未上市流通的送股
        self.receivables = np.zeros(self.N, dtype=np.float64)
        self.cost_raw = np.full(self.N, np.nan, dtype=np.float64)
        self.cost_adj = np.full(self.N, np.nan, dtype=np.float64)
        self.last_px = np.full(self.N, np.nan, dtype=np.float64)   # 最后有效成交价
        self.stale = np.zeros(self.N, dtype=np.int32)        # 连续多少个交易日没有报价
        self.realized = 0.0                                  # 已实现盈亏（元）
        self.fees = 0.0
        self.turnover = 0.0                                  # 累计成交额
        self.n_buy = 0
        self.n_sell = 0
        self.n_div_events = 0
        self.shares_issued = 0                               # 除权除息新增的股数
        self.pnl = np.zeros(self.N, dtype=np.float64)        # 逐股已实现盈亏（元）
        self.n_force_close = 0                               # 退市/长期停牌强制了结的笔数
        self.pnl_delisted = 0.0                              # 这部分贡献的损益

    def __repr__(self):
        return (f"Account(现金 {self.cash:,.0f}, 持仓 {int((self.shares > 0).sum())} 只, "
                f"市值 {self.position_value(self.last_px):,.0f}, "
                f"累计费用 {self.fees:,.0f}, 成交 {self.n_buy + self.n_sell} 笔)")

    # ------------------------------------------------------------------
    # 估值
    # ------------------------------------------------------------------
    def price_at(self, px_row):
        """给一行可能带 NaN 的价格，替换成"最后有效价"。

        停牌当天没有报价，持仓不能让净值变成 NaN，也不能当成 0——只能沿用
        最后成交价。这是 `close` / `close_ff` 两条价格线存在的理由。
        """
        px = np.asarray(px_row, dtype=np.float64)
        return np.where(np.isfinite(px) & (px > 0), px, self.last_px)

    def position_value(self, px=None):
        """持仓市值。"""
        p = (self._valuation if self._valuation is not None else self.last_px) if px is None else np.asarray(px, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            v = self.shares * p
        return float(np.nansum(v))

    def equity(self, px=None):
        return self.cash + self.position_value(px) + float(self.receivables.sum()) - float(self.tax_payables.sum())

    def settle_taxes(self):
        """Known assessments reduce NAV immediately; cash payment cannot overdraw."""
        for j in np.flatnonzero(self.tax_payables>0):
            payment=min(self.cash,float(self.tax_payables[j]))
            self.cash-=payment
            self.tax_payables[j]-=payment

    def weights(self, px=None):
        """当前权重（按市值）。空仓时返回全 0。"""
        p = (self._valuation if self._valuation is not None else self.last_px) if px is None else np.asarray(px, dtype=np.float64)
        e = self.equity(p)
        if e <= 0:
            return np.zeros(self.N, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            w = self.shares * p
        return np.nan_to_num(w / e, nan=0.0)

    def holdings(self, px=None):
        """有持仓的股票下标。"""
        return np.flatnonzero(self.shares > 0)

    def unrealized(self, adj_price):
        """逐股的浮动收益率（复权口径，含分红）。没持仓的是 NaN。"""
        with np.errstate(invalid="ignore", divide="ignore"):
            r = self.risk_price(adj_price) / self.cost_adj - 1.0
        return np.where(self.shares > 0, r, np.nan)

    def risk_price(self, adj_price):
        """Split-adjusted mark plus distributions earned by the current position."""
        return np.asarray(adj_price, float) + self.income_adj

    # ------------------------------------------------------------------
    # 逐日
    # ------------------------------------------------------------------
    def mark(self, raw_close_row, advance=True):
        """收盘打标：更新最后有效价和"多久没有报价"的计数。

        退市/长期停牌的票只能靠 `stale` 认出来——前向填充价会一路填下去，
        永远不告诉你这只票其实已经没了。
        """
        px = np.asarray(raw_close_row, dtype=np.float64)
        valid = np.isfinite(px) & (px > 0)
        self.last_px[valid] = px[valid]
        self.stale[valid] = 0
        if advance:
            self.stale[~valid] += 1
        return self.last_px

    def unfreeze(self):
        """新的一天：昨天买的今天可以卖了（T+1 按自然交易日算，和当天是否
        能成交无关）。"""
        self.frozen[:] = 0

    # ------------------------------------------------------------------
    # 成交
    # ------------------------------------------------------------------
    def buy(self, j, qty, price, fee, f_now):
        """买入 qty 股。`f_now` 是当日的累计复权因子，用来记复权成本。

        返回实际从现金里扣掉的钱。
        """
        j = _integer(j,0,self.N-1,'证券下标')
        qty = _integer(qty,0,_MAX_SHARES,'成交数量')
        if qty == 0:
            return 0.0
        if not np.isfinite(price) or price <= 0 or not np.isfinite(fee) or fee < 0 or not np.isfinite(f_now) or f_now <= 0:
            raise ValueError("成交价格/因子必须为正数，费用必须为非负有限数")
        amount = qty * float(price)
        outlay = amount + float(fee)
        old_q = int(self.shares[j])
        add_adj = float(price) * float(f_now)
        if old_q+qty > _MAX_SHARES or not np.isfinite(outlay) or not np.isfinite(add_adj):
            raise ValueError('买入派生金额/复权成本或累计股数超出账本可表示范围')
        if outlay > self.cash - self.tax_payables.sum() + 1e-6:
            raise AssertionError(
                f"现金不够（需要 {outlay:,.2f}，只有 {self.cash:,.2f}）——"
                f"下单前必须先按可用现金缩减")
        self.cash -= outlay
        previous_income=self.income_adj[j]
        self.income_adj[j] *= old_q / (old_q + qty)
        self.risk_rebases[j]+=self.income_adj[j]-previous_income
        if old_q==0: self.position_generation[j]+=1
        self.fees += float(fee)
        self.turnover += amount
        self.n_buy += 1

        self.cost_raw[j] = (float(price) if old_q <= 0 else
                            (self.cost_raw[j] * old_q + float(price) * qty) / (old_q + qty))
        if old_q <= 0:
            self.cost_adj[j] = add_adj
        else:
            # 加权平均，复权口径
            self.cost_adj[j] = (self.cost_adj[j] * old_q + add_adj * qty) / (old_q + qty)
        self.shares[j] = old_q + qty
        self.last_px[j] = float(price)
        self.frozen[j] += qty
        return outlay

    def sell(self, j, qty, price, fee, f_now):
        """卖出 qty 股。返回收到的净现金。"""
        j = _integer(j,0,self.N-1,'证券下标')
        qty = _integer(qty,0,_MAX_SHARES,'成交数量')
        if qty == 0:
            return 0.0
        if not np.isfinite(price) or price <= 0 or not np.isfinite(fee) or fee < 0 or not np.isfinite(f_now) or f_now <= 0:
            raise ValueError("成交价格/因子必须为正数，费用必须为非负有限数")
        if qty > int(self.shares[j] - self.frozen[j] - self.restricted[j]):
            raise AssertionError("不能卖出当日买入或尚未可交易的股份")
        if qty > int(self.shares[j]):
            raise AssertionError(f"卖出 {qty} 股但只持有 {int(self.shares[j])} 股")
        amount = qty * float(price)
        net = amount - float(fee)
        if not np.isfinite(net) or not np.isfinite(self.cash+net):
            raise ValueError('卖出派生金额超出现金账本可表示范围')
        if self.cash + net < -1e-6:
            raise AssertionError("卖出所得与可用现金不足以支付费用，不能透支现金账户")
        self.cash += net
        self.fees += float(fee)
        self.turnover += amount
        self.n_sell += 1

        # 已实现盈亏：买入价按今日因子还原到今日的每股口径
        entry = self.cost_raw[j]
        gain = qty * (float(price) - entry)
        self.realized += gain
        self.pnl[j] += gain

        left = int(self.shares[j]) - qty
        self.shares[j] = left
        if left == 0:
            self.income_adj[j] = 0.
            self.cost_adj[j] = np.nan
            self.cost_raw[j] = np.nan
        return net

    def force_close(self, j, price, f_now):
        """按已知清算价格结算；引擎不会因为停牌时长自动调用。"""
        j = _integer(j,0,self.N-1,'证券下标')
        price = float(price)
        if not np.isfinite(price) or price < 0:
            raise ValueError('已知清算价格必须为非负有限数，缺失不能代替零回收')
        qty = int(self.shares[j])
        if qty <= 0:
            return 0.0
        proceeds = qty * price
        if not np.isfinite(proceeds) or not np.isfinite(self.cash+proceeds):
            raise ValueError('清算派生金额超出现金账本可表示范围')
        self.cash += proceeds
        entry = self.cost_raw[j]
        gain = qty * (price - entry)
        self.realized += gain
        self.pnl[j] += gain
        self.pnl_delisted += gain
        self.n_force_close += 1
        self.shares[j] = 0
        self.income_adj[j] = 0.
        self.cost_adj[j] = np.nan
        self.cost_raw[j] = np.nan
        self.frozen[j] = 0
        self.restricted[j] = 0
        return proceeds

    # ------------------------------------------------------------------
    # 除权除息
    # ------------------------------------------------------------------
    def apply_corporate_actions(self, f_prev, f_now, raw_px, policy="reinvest"):
        """处理 t 日的除权除息。返回 (事件数, 新增股数, 记入现金的零头)。

        拆股调股数、分红调股数——都走"按因子比例调股数"一条路。见模块
        docstring 里的理由。`raw_px` 是当日原始价（用来给不足一股的零头估值）。
        """
        if policy not in DIV_POLICIES:
            raise ValueError(f"div_policy 只能是 {DIV_POLICIES}，收到 {policy!r}")
        if policy == "ignore":
            return 0, 0, 0.0

        held = self.shares > 0
        if not held.any():
            return 0, 0, 0.0

        fp = np.asarray(f_prev, dtype=np.float64)
        fn = np.asarray(f_now, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = fn / fp
        ev = held & np.isfinite(ratio) & (np.abs(ratio - 1.0) > _FACTOR_TOL)
        if not ev.any():
            return 0, 0, 0.0

        j = np.flatnonzero(ev)
        px = np.asarray(raw_px, dtype=np.float64)[j]
        px = np.where(np.isfinite(px) & (px > 0), px, self.last_px[j]/ratio[j])
        px = np.where(np.isfinite(px) & (px >= 0), px, 0.0)

        q = self.shares[j].astype(np.float64)
        target = q * ratio[j]
        tolerance = np.maximum(1e-9, 8*np.spacing(np.abs(target)))
        q_new = np.floor(target+tolerance).astype(np.int64)
        remainder = np.maximum(target-q_new,0.)              # 浮点噪声不能把整股变成现金
        cash_in = float(np.sum(remainder * px))
        self.cash += cash_in
        residual_gain = remainder * (px - self.cost_raw[j] / ratio[j])
        self.realized += float(np.sum(residual_gain))
        self.pnl[j] += residual_gain
        issued = int((q_new - q).sum())
        self.shares[j] = q_new
        self.restricted[j] = np.minimum(q_new, np.ceil(self.restricted[j]*ratio[j]-1e-9)).astype(np.int64)
        self.frozen[j] = np.minimum(q_new-self.restricted[j], np.ceil(self.frozen[j]*ratio[j]-1e-9)).astype(np.int64)
        self.cost_raw[j] /= ratio[j]
        # 调整估值参考价不等于得到新报价：stale不会因此归零。
        self.last_px[j] /= ratio[j]
        if self._valuation is not None:
            self._valuation[j] /= ratio[j]
        self.shares_issued += issued
        self.n_div_events += int(j.size)
        return int(j.size), issued, cash_in

    @staticmethod
    def implied_dividend(f_prev, f_now, raw_px):
        """**只用于报告/校验，不进账本。**

        `D = raw_t × (F_t/F_{t-1} - 1)`，即复权因子隐含的每股现金分红。
        对纯现金分红是精确的（002555.SZ 2020-05-11 实例：D=0.2956，代回
        `(raw_t+D)/raw_{t-1}-1` 与复权收益逐位吻合）；对送转会把拆股也
        算成一笔现金。所以它是个**诊断量**，不是账务口径。
        """
        fp = np.asarray(f_prev, dtype=np.float64)
        fn = np.asarray(f_now, dtype=np.float64)
        px = np.asarray(raw_px, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return px * (fn / fp - 1.0)

    # ------------------------------------------------------------------
    # 自检
    # ------------------------------------------------------------------
    def check_cash(self, tol=1e-6):
        """现金永不为负。任何"忘记按可用现金缩减下单"的 bug 都会在这里炸。"""
        if not np.isfinite(self.cash) or self.cash < -tol:
            raise AssertionError(f"现金为负：{self.cash:.6f} 元")

    def snapshot(self):
        return {
            "cash": self.cash,
            "应收分红": float(self.receivables.sum()),
            "equity": self.equity(),
            "n_positions": int((self.shares > 0).sum()),
            "fees": self.fees,
            "turnover": self.turnover,
            "realized": self.realized,
            "n_buy": self.n_buy,
            "n_sell": self.n_sell,
            "n_div_events": self.n_div_events,
            "shares_issued": self.shares_issued,
        }
