"""交易成本：佣金、印花税、过户费、滑点。

A 股的成本结构和美股完全不同，而且**印花税在样本期内改过**：

    印花税  卖出单边   0.1%   ->  0.05%（2023-08-28 起）
    佣金    双边       万 2.5，**单笔最低 5 元**
    过户费  双边       按历史日期与市场；2022-04-29起统一0.001%

"最低 5 元"这条对小资金策略的伤害很大：1 万元买 5 只票，每只 2000 元，
按万 2.5 算佣金才 0.5 元，实收 5 元 —— 实际费率是万 25，十倍。
很多回测框架忽略这一条，于是小资金高频策略的回测收益被系统性高估。
"""

from __future__ import annotations

import numpy as np

__all__ = ["Costs", "DEFAULT"]

# 印花税调整：2023-08-28 从 0.1% 降到 0.05%（活跃资本市场的一揽子政策之一）
_STAMP_SCHEDULE = (
    (np.datetime64("1900-01-01"), 0.001),
    (np.datetime64("2023-08-28"), 0.0005),
)


def transfer_fee_on(day=None, codes=None):
    """样本期A股过户费。无日期按较新费率；无代码按沪深普通A股。

    中国结算2022-04-28通知：次日起沪深0.02‰、北京0.025‰统一至0.01‰。
    2015-08-01前的计费基础不同，本函数不冒充支持那段制度。
    """
    if day is None:
        return 1e-5
    day=np.datetime64(day,'D')
    if np.isnat(day) or day < np.datetime64('2015-08-01'):
        raise ValueError('自动过户费表支持2015-08-01起；更早日期请显式指定transfer_fee或成本组件')
    if day >= np.datetime64('2022-04-29'):
        return 1e-5
    if codes is None:
        return 2e-5
    code=np.asarray(codes,dtype=str)
    beijing=np.char.endswith(code,'.BJ')
    return np.where(beijing,2.5e-5,2e-5)


def stamp_duty_on(day):
    """给定日期，返回卖出印花税率。"""
    d = np.datetime64(day, "D")
    rate = _STAMP_SCHEDULE[0][1]
    for start, r in _STAMP_SCHEDULE:
        if d >= start:
            rate = r
    return rate


def stamp_duty_panel(dates):
    """整条日历的印花税率，(T,)。"""
    out = np.full(len(dates), _STAMP_SCHEDULE[0][1], dtype=np.float64)
    for start, r in _STAMP_SCHEDULE:
        out[np.asarray(dates) >= start] = r
    return out


class Costs:
    """一笔成交要交多少钱。

    commission     佣金费率（双边），默认万 2.5
    min_commission 单笔最低佣金，默认 5 元 —— 别关掉，关了回测就假了
    transfer_fee   过户费（双边），None按历史表；显式数值固定覆盖
    slippage_bps   滑点，按成交额的万分之几算，默认 0
    """

    __slots__ = ("commission", "min_commission", "transfer_fee", "slippage_bps",
                 "stamp_duty_override")

    def __init__(self, commission=2.5e-4, min_commission=5.0,
                 transfer_fee=None, slippage_bps=0.0, stamp_duty=None):
        self.commission = float(commission)
        self.min_commission = float(min_commission)
        self.transfer_fee = None if transfer_fee is None else float(transfer_fee)
        self.slippage_bps = float(slippage_bps)
        self.stamp_duty_override = stamp_duty
        for key in ("commission", "min_commission", "transfer_fee", "slippage_bps"):
            value = getattr(self, key)
            if key == 'transfer_fee' and value is None:
                continue
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} 必须是非负有限数")
        if stamp_duty is not None and (not np.isfinite(stamp_duty) or stamp_duty < 0):
            raise ValueError("stamp_duty 必须是非负有限数")

    # -- 费用 ------------------------------------------------------------
    # 下面两个方法既接受标量也接受数组：订单是一批一批算的，逐笔调 Python
    # 函数会给每天增加几百次解释器开销，回测循环里这很贵。
    def buy(self, amount, day=None, *, codes=None):
        """买入 amount 元（成交额）的费用。标量进标量出，数组进数组出。"""
        amount = np.abs(np.asarray(amount, dtype=np.float64))
        fee = np.maximum(amount * self.commission, self.min_commission)
        transfer = transfer_fee_on(day,codes) if self.transfer_fee is None else self.transfer_fee
        fee = fee + amount * transfer
        fee = fee + amount * self.slippage_bps * 1e-4
        fee = np.where(amount > 0, fee, 0.0)      # 没成交就没有最低佣金
        return fee if fee.ndim else float(fee)

    def sell(self, amount, day=None, *, codes=None):
        """卖出 amount 元（成交额）的费用，含印花税。"""
        amount = np.abs(np.asarray(amount, dtype=np.float64))
        stamp = self.stamp_duty_override
        if stamp is None:
            stamp = stamp_duty_on(day) if day is not None else _STAMP_SCHEDULE[0][1]
        fee = np.maximum(amount * self.commission, self.min_commission)
        transfer = transfer_fee_on(day,codes) if self.transfer_fee is None else self.transfer_fee
        fee = fee + amount * transfer
        fee = fee + amount * float(stamp)
        fee = fee + amount * self.slippage_bps * 1e-4
        fee = np.where(amount > 0, fee, 0.0)
        return fee if fee.ndim else float(fee)

    def __repr__(self):
        transfer = '按历史日期/市场' if self.transfer_fee is None else f'{self.transfer_fee*1e4:.2f}bp'
        stamp = '按历史日期' if self.stamp_duty_override is None else f'{self.stamp_duty_override*1e4:g}bp'
        return (f"Costs(佣金 {self.commission*1e4:.1f}bp 最低 {self.min_commission:.0f} 元, "
                f"过户 {transfer}, 印花税 {stamp}, 滑点 {self.slippage_bps:.1f}bp)")

    buy.supports_codes = True
    sell.supports_codes = True


DEFAULT = Costs()


def evaluate_cost(method, amount, day=None, *, codes=None, quantity=None):
    """把每笔费用广播到成交额形状；零成交不收费，禁止少算标量固定费。"""
    amounts = np.asarray(amount, dtype=float)
    kw={}
    if getattr(method,'supports_codes',False): kw['codes']=codes
    if getattr(method,'supports_quantity',False): kw['quantity']=quantity
    value=method(amounts,day,**kw)
    fees = np.asarray(value, dtype=float)
    try:
        fees = np.broadcast_to(fees, amounts.shape)
    except ValueError as error:
        raise ValueError("成本组件返回的形状不能与逐笔成交额对齐") from error
    fees = np.where(amounts > 0, fees, 0.)
    if not np.isfinite(fees).all() or (fees < 0).any():
        raise ValueError("成本组件返回非有限或负费用")
    return fees


class SlippageOverlay:
    """给任意兼容 buy/sell 的成本组件叠加滑点，不要求它有内部费率属性。"""
    def __init__(self, base, bps):
        self.base, self.bps = base, bps

    def buy(self, amount, day=None, *, codes=None, quantity=None):
        return evaluate_cost(self.base.buy,amount,day,codes=codes,quantity=quantity) + np.abs(np.asarray(amount)) * self.bps * 1e-4

    def sell(self, amount, day=None, *, codes=None, quantity=None):
        return evaluate_cost(self.base.sell,amount,day,codes=codes,quantity=quantity) + np.abs(np.asarray(amount)) * self.bps * 1e-4

    buy.supports_codes = True
    sell.supports_codes = True
    buy.supports_quantity = True
    sell.supports_quantity = True

    def __repr__(self):
        return f"SlippageOverlay({self.base!r}, +{self.bps}bps)"
