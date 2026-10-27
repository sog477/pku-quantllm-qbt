"""资产申报/结算规则和账户购买力/仓位规则，均可被独立替换。"""
import copy
import numpy as np
from ._validation import integer


def _integer_panel(value, size, lower, name):
    raw = np.broadcast_to(np.asarray(value), (size,))
    return np.array([integer(v, lower, int(np.iinfo(np.int64).max), name) for v in raw], dtype=np.int64)


class CashAssetRules:
    def __init__(self, minimum=1, step=1, settlement_days=0, price_limits=False, price_tick=None):
        self.minimum, self.step, self.settlement_days = minimum, step, settlement_days
        self.price_limits = price_limits
        self.price_tick = price_tick

    def bind(self, ds):
        obj = copy.copy(self)
        obj.minimums = _integer_panel(self.minimum, ds.N, 1, 'minimum')
        obj.steps = _integer_panel(self.step, ds.N, 1, 'step')
        obj.delays = _integer_panel(self.settlement_days, ds.N, 0, 'settlement_days')
        obj.price_ticks = None
        if self.price_tick is not None:
            ticks = np.broadcast_to(np.asarray(self.price_tick, float), (ds.N,))
            if not np.isfinite(ticks).all() or (ticks <= 0).any():
                raise ValueError('price_tick 必须为正的有限数；不限制精度请使用None')
            obj.price_ticks = ticks.copy()
        flags=np.asarray(self.price_limits)
        if flags.dtype.kind not in 'buif' or not np.isin(flags,[0,1]).all():
            raise ValueError('price_limits 必须为bool或0/1，不能把缺失/字符串/未知值转成交易规则')
        obj.limit_flags = np.broadcast_to(flags.astype(bool), (ds.N,)).copy()
        if (obj.minimums <= 0).any() or (obj.steps <= 0).any() or not np.isin(obj.delays, [0, 1]).all():
            raise ValueError("minimum/step 必须为正，内置结算支持会话 T+0/T+1")
        return obj

    def valid_order_price(self, j, price):
        if self.price_ticks is None:
            return True
        ticks = float(price) / self.price_ticks[j]
        return bool(np.isfinite(ticks) and np.isclose(ticks, np.rint(ticks), rtol=0, atol=1e-7))

    def valid_order_quantity(self, j, quantity, side, held):
        normal = quantity >= self.minimums[j] and quantity % self.steps[j] == 0
        return bool(normal or side == 'sell' and quantity == held)

    def round_buy(self, qty):
        q = np.maximum(np.asarray(qty, np.int64), 0)
        q = (q // self.steps) * self.steps
        return np.where(q >= self.minimums, q, 0)

    def round_sell(self, qty, held):
        q = np.maximum(np.asarray(qty, np.int64), 0)
        full = q >= held
        q = (q // self.steps) * self.steps
        return np.where(full, held, np.where(q >= self.minimums, q, 0))

    def can_buy(self, market, i, rule):
        if bool(self.limit_flags.all()):
            return market.can_buy(i,rule)
        return np.where(self.limit_flags,market.can_buy(i,rule),~market.quote_suspended[i])

    def can_sell(self, market, i, rule):
        if bool(self.limit_flags.all()):
            return market.can_sell(i,rule)
        return np.where(self.limit_flags,market.can_sell(i,rule),~market.quote_suspended[i])


class AStockRules(CashAssetRules):
    def __init__(self, lot=100, star_minimum=200, bse_minimum=100):
        super().__init__(minimum=lot, step=lot, settlement_days=1, price_limits=True, price_tick=.01)
        self.lot, self.star_minimum = lot, star_minimum
        self.bse_minimum = bse_minimum

    def bind(self, ds):
        obj = super().bind(ds)
        minimums = _integer_panel(self.star_minimum, ds.N, 1, 'star_minimum')
        bse_minimums = _integer_panel(self.bse_minimum, ds.N, 1, 'bse_minimum')
        star = np.array([c[:3] in ("688", "689") for c in ds.codes])
        beijing = np.array([c.endswith('.BJ') or c[:2] in ('43','83','87','88','92') for c in ds.codes])
        obj.steps[star], obj.minimums[star] = 1, minimums[star]
        obj.steps[beijing], obj.minimums[beijing] = 1, bse_minimums[beijing]
        return obj

    def valid_order_quantity(self, j, quantity, side, held):
        if super().valid_order_quantity(j, quantity, side, held):
            return True
        step = self.steps[j]
        # 主板零股余额须整块卖出，可以单独卖尾数，保留整手持仓。
        return bool(side == 'sell' and step > 1 and 0 < quantity <= held and
                    held % step > 0 and quantity % step == held % step)

    def round_sell(self, qty, held):
        q = np.minimum(np.maximum(np.asarray(qty,np.int64),0),held)
        base = super().round_sell(q, held)
        residual = held % self.steps
        candidate = ((q-residual)//self.steps)*self.steps+residual
        legal = (self.steps > 1) & (residual > 0) & (q >= residual)
        return np.maximum(base,np.where(legal,candidate,0))


class CashAccountRules:
    name = "long_only_cash"

    def reset(self):
        pass

    def validate_gross(self, gross):
        g = np.asarray(gross, float)
        if not np.isfinite(g).all() or (g < 0).any() or (g > 1).any():
            raise ValueError("现金账户规则：gross 必须位于 [0,1]")

    def start_bar(self, account, event, asset_rules):
        account.settle_taxes()
        if event.new_session:
            account.unfreeze()

    def buying_power(self, account):
        return max(0.,account.cash-account.tax_payables.sum())

    def sellable(self, account):
        return np.maximum(account.shares-account.frozen-account.restricted, 0)

    def buy(self, account, j, qty, price, fee, factor, asset_rules):
        account.buy(j, qty, price, fee, factor)
        if asset_rules.delays[j] == 0:
            account.frozen[j] -= qty

    def sell(self, account, j, qty, price, fee, factor, asset_rules):
        account.sell(j, qty, price, fee, factor)

    def equity(self, account, prices):
        return account.equity(prices)

    def check(self, account):
        account.check_cash()
        if ((account.shares < 0).any() or (account.frozen < 0).any() or
                (account.restricted < 0).any() or
                (account.frozen + account.restricted > account.shares).any()):
            raise AssertionError("现金账户库存不合法：股份/冻结/限制股须非负，锁定总量不能超过持仓")
        if not np.isfinite(account.receivables).all() or (account.receivables < -1e-6).any():
            raise AssertionError("现金账户应收权益须为非负有限数")
        if not np.isfinite(account.tax_payables).all() or (account.tax_payables < -1e-6).any():
            raise AssertionError('tax liabilities must be finite and nonnegative')
