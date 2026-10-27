"""直接委托：持久订单簿、独立执行模型、可查询的状态与成交事件。"""
from dataclasses import dataclass, replace
from .costs import evaluate_cost
from ._validation import integer, boolean_mask, naive_datetime_scalar
import numpy as np

ACTIVE = ("OPEN", "PARTIAL")


@dataclass(frozen=True)
class OrderRequest:
    code: str
    quantity: int
    side: str = "buy"
    limit_price: float | None = None
    tif: str = "DAY"
    valid_until: object = None
    tag: str = ""

    def __post_init__(self):
        if self.side not in ("buy", "sell") or self.tif not in ("DAY", "GTC", "GTD"):
            raise ValueError("side 为 buy/sell；tif 为 DAY/GTC/GTD")
        quantity = integer(self.quantity,1,int(np.iinfo(np.int64).max),'quantity')
        object.__setattr__(self, 'quantity', quantity)
        if self.limit_price is not None:
            if np.ndim(self.limit_price) != 0 or np.asarray(self.limit_price).dtype.kind not in 'uif':
                raise ValueError("limit_price 必须是正的有限数")
            price = float(self.limit_price)
            if not np.isfinite(price) or price <= 0:
                raise ValueError("limit_price 必须是正的有限数")
            object.__setattr__(self, 'limit_price', price)
        if self.tif == "GTD" and self.valid_until is None:
            raise ValueError("GTD 必须指定 valid_until")
        if self.valid_until is not None:
            deadline = naive_datetime_scalar(self.valid_until,'valid_until',keep_unit=True)
            if np.isnat(deadline):
                raise ValueError("valid_until 不能是 NaT")
            object.__setattr__(self,'valid_until',deadline)


@dataclass
class Order:
    id: int
    request: OrderRequest
    submitted_at: object
    active_from: int
    session: object
    status: str = "OPEN"
    filled: int = 0
    amount: float = 0.
    fee: float = 0.
    reason: str = ""
    fee_day: object = None
    day_amount: float = 0.
    day_quantity: int = 0

    @property
    def remaining(self):
        return self.request.quantity-self.filled

    @property
    def average_price(self):
        return self.amount/self.filled if self.filled else np.nan


@dataclass(frozen=True)
class OrderEvent:
    order_id: int
    timestamp: object
    status: str
    filled: int
    remaining: int
    reason: str


@dataclass(frozen=True)
class OrderFill:
    order_id: int
    index: int
    timestamp: object
    phase: str
    code: str
    side: str
    quantity: int
    price: float
    fee: float
    source_timestamp: object = None
    evidence_age_ns: int | None = None


@dataclass(frozen=True)
class Cancel:
    order_id: int


class OpenExecution:
    """下一可见开盘撮合；没有开盘价格优势就等待，不猜 bar 内触价顺序。"""
    description = "下一bar开盘；限价须在开盘可成交；FIFO，无盘口排队模型"
    phases = ("open",)

    def price(self, request, ds, i, phase):
        if phase != "open":
            return None
        price = float(ds.open[i, ds.col(request.code)])
        if not np.isfinite(price) or price <= 0:
            return None
        limit = request.limit_price
        if limit is None or (price <= limit if request.side == "buy" else price >= limit):
            return price
        return None


class BarTouchExecution(OpenExecution):
    """显式 OHLC 乐观假设：未在开盘成交的限价单，在bar结束后按触价成交。"""
    description = "开盘优先；OHLC触价在bar结束记账，FIFO假设，不能证明真实触价/排队顺序"
    phases = ("open", "close")

    def price(self, request, ds, i, phase):
        if phase == "open":
            return super().price(request, ds, i, phase)
        if request.limit_price is None:
            return None
        j = ds.col(request.code)
        price = request.limit_price
        o,c,h,l=(float(getattr(ds,name)[i,j]) for name in ('open','close','high','low'))
        usable=lambda x:np.isfinite(x) and x>0
        extreme=l if request.side=='buy' else h
        if not usable(extreme):
            return None  # 0/负值缺失哨兵不是触价证据。
        known=[x for x in (o,c,h,l) if usable(x)]
        tolerance=max(1e-9,max(known)*2*np.finfo(np.float32).eps)
        if usable(h) and usable(l) and l>h+tolerance:
            return None
        for value in (o,c):
            if usable(value) and ((usable(l) and value<l-tolerance) or
                                  (usable(h) and value>h+tolerance)):
                return None
        touched = l <= price if request.side == "buy" else h >= price
        return float(price) if touched else None


def _deadline(value):
    if value is None:
        return None
    t = np.datetime64(value)
    if t.dtype == np.dtype("datetime64[D]"):
        return (t+np.timedelta64(1,"D")).astype("datetime64[ns]")-np.timedelta64(1,"ns")
    return t.astype("datetime64[ns]")


class OrderBook:
    def __init__(self, ds, broker, account, execution):
        self.ds, self.broker, self.account, self.execution = ds, broker, account, execution
        self.records, self.events, self.fills = {}, [], []
        self.codes = {c:j for j,c in enumerate(ds.codes)}
        self.event = None
        self.used = None
        self.blocked = set()

    def _record(self, order, timestamp, reason=""):
        order.reason = reason
        self.events.append(OrderEvent(order.id,timestamp,order.status,order.filled,order.remaining,reason))

    def submit(self, request):
        if not isinstance(request, OrderRequest):
            raise TypeError("submit 需要 OrderRequest")
        e = self.event
        order = Order(len(self.records)+1, request, e.previous_timestamp, e.index,
                      e.session if e.session is not None else e.timestamp.astype("datetime64[D]"))
        self.records[order.id] = order
        j = self.codes.get(request.code)
        invalid = j is None
        if j is not None:
            rules = self.broker.asset_rules
            validator = getattr(rules, 'valid_order_quantity', None)
            if callable(validator):
                invalid = not validator(j,request.quantity,request.side,self.account.shares[j])
            else:
                odd_exit = request.side == "sell" and request.quantity == self.account.shares[j]
                invalid = (not odd_exit and (request.quantity < rules.minimums[j] or request.quantity % rules.steps[j] != 0))
            valid_price = getattr(rules, 'valid_order_price', None)
            if request.limit_price is not None and callable(valid_price):
                invalid = invalid or not valid_price(j, request.limit_price)
        if invalid:
            order.status = "REJECTED"
        self._record(order,e.previous_timestamp,"unknown_instrument_or_invalid_quantity_or_price" if invalid else "submitted")
        deadline = _deadline(request.valid_until)
        if not invalid and deadline is not None and e.timestamp > deadline:
            order.status = "EXPIRED"
            self._record(order,e.timestamp,"valid_until")
        return order.id

    def cancel(self, order_id, reason="user_cancel", *, timestamp=None):
        order = self.records.get(order_id)
        if order is None or order.status not in ACTIVE:
            return False
        order.status = "CANCELED"
        self._record(order,self.event.previous_timestamp if timestamp is None else timestamp,reason)
        return True

    def get(self, order_id):
        return replace(self.records[order_id])

    def orders(self, active_only=False):
        return tuple(replace(o) for o in self.records.values() if not active_only or o.status in ACTIVE)

    def begin(self, event):
        self.event = event
        if hasattr(self.broker.asset_rules,'day'):
            self.broker.asset_rules.day=event.timestamp.astype('datetime64[D]')
        # 同一历史成交额上界由所有方向/执行阶段共用，不能买卖各拿一份。
        begin_capacity = getattr(self.broker, 'begin_capacity', None)
        self.used = begin_capacity(event.index) if callable(begin_capacity) else np.zeros(self.ds.N)
        self.blocked = set()
        for order in self.records.values():
            deadline = _deadline(order.request.valid_until)
            session = event.session if event.session is not None else event.timestamp.astype("datetime64[D]")
            if order.status in ACTIVE and ((deadline is not None and event.timestamp > deadline) or
                                         (order.request.tif == "DAY" and order.session != session)):
                order.status = "EXPIRED"
                self._record(order,event.timestamp,"valid_until")

    def cancel_instruments(self, indices, reason):
        indices = set(map(int,indices))
        for o in self.records.values():
            if self.codes.get(o.request.code) in indices:
                self.cancel(o.id,reason,timestamp=self.event.timestamp)

    def match(self, phase):
        b, ds, a, e = self.broker, self.ds, self.account, self.event
        cap = b.capacity_amount(e.index)
        masks = {"buy":boolean_mask(b.asset_rules.can_buy(b.m,e.index,b.limit_rule),'can_buy',(ds.N,)),
                 "sell":boolean_mask(b.asset_rules.can_sell(b.m,e.index,b.limit_rule),'can_sell',(ds.N,))}
        for order in self.records.values():
            if order.status not in ACTIVE or order.active_from > e.index:
                continue
            req, j = order.request, self.codes[order.request.code]
            validator = getattr(b.asset_rules, 'valid_order_quantity', None)
            if (req.side == 'sell' and order.filled == 0 and callable(validator) and
                    not validator(j,req.quantity,req.side,a.shares[j])):
                self.cancel(order.id,'quantity_no_longer_valid_for_position',timestamp=e.timestamp)
                continue
            price = self.execution.price(req,ds,e.index,phase)
            reason = ""
            if not masks[req.side][j]:
                reason = "market_blocked"
                invalid_open = getattr(b.m, 'invalid_limit_open', None)
                bad_open = bool(invalid_open(e.index)[j]) if callable(invalid_open) else False
                if bad_open:
                    reason = 'outside_assumed_price_band'
                baseline = b.asset_rules.can_buy if req.side == "buy" else b.asset_rules.can_sell
                if not bad_open and boolean_mask(baseline(b.m,e.index,"off"),'tradability(off)',(ds.N,))[j]:
                    self.blocked.add(order.id)
            elif price is None:
                reason = "price_not_reached"
            if reason:
                if order.reason != reason:
                    self._record(order,e.timestamp,reason)
                continue
            price = float(price)
            if not np.isfinite(price) or price <= 0 or req.limit_price is not None and (price > req.limit_price if req.side == "buy" else price < req.limit_price):
                raise ValueError("执行模型返回无效或违反限价的价格")
            available = max(cap[j]-self.used[j],0.)
            qty = min(order.remaining,int(np.floor(available/price+1e-10)))
            if req.side == "sell":
                qty = min(qty,int(b.account_rules.sellable(a)[j]))
            day = e.timestamp.astype("datetime64[D]")
            fee_session = e.session if e.session is not None else day
            previous_amount = order.day_amount if order.fee_day is not None and order.fee_day == fee_session else 0.
            previous_quantity = order.day_quantity if order.fee_day is not None and order.fee_day == fee_session else 0
            cost = b.costs.buy if req.side == "buy" else b.costs.sell
            def fee_for(n):
                fee = float(evaluate_cost(cost,previous_amount+n*price,day,codes=req.code,quantity=previous_quantity+n)-evaluate_cost(cost,previous_amount,day,codes=req.code,quantity=previous_quantity))
                if not np.isfinite(fee) or fee < -1e-8:
                    raise ValueError("成本组件必须返回非负有限的增量费用")
                return max(fee,0.)
            if req.side == "buy":
                power = self._buying_power()
                lo,hi = 0,qty
                while lo < hi:
                    mid = (lo+hi+1)//2
                    if mid*price+fee_for(mid) <= power+1e-9:
                        lo = mid
                    else:
                        hi = mid-1
                qty = lo
            if qty <= 0:
                reason = "cash_or_capacity" if req.side == "buy" else "settlement_position_or_capacity"
                if order.reason != reason:
                    self._record(order,e.timestamp,reason)
                continue
            fee = fee_for(qty)
            if req.side == "sell" and self._buying_power() + qty*price < fee-1e-9:
                if order.reason != "sell_fee_exceeds_buying_power":
                    self._record(order,e.timestamp,"sell_fee_exceeds_buying_power")
                continue
            fill_method = b.account_rules.buy if req.side == "buy" else b.account_rules.sell
            fill_method(a,j,qty,price,fee,ds.cumadj_ff[e.index,j],b.asset_rules)
            order.filled += qty
            order.amount += qty*price
            order.fee += fee
            order.fee_day, order.day_amount = fee_session,previous_amount+qty*price
            order.day_quantity=previous_quantity+qty
            self.used[j] += qty*price
            order.status = "FILLED" if order.remaining == 0 else "PARTIAL"
            source_time = b.m.quote_time[e.index,j]
            evidence_age = int((e.timestamp.astype('datetime64[ns]')-source_time).astype('timedelta64[ns]').astype(np.int64))
            self.fills.append(OrderFill(order.id,e.index,e.timestamp,phase,req.code,req.side,qty,price,fee,source_time,evidence_age))
            self._record(order,e.timestamp,"filled")

    def _buying_power(self):
        power = float(self.broker.account_rules.buying_power(self.account))
        if not np.isfinite(power) or power < -1e-6:
            raise ValueError(f'账户组件返回未知/无效购买力：{power}')
        return max(power,0.)

    def end(self, last_in_session):
        for order in self.records.values():
            if order.status not in ACTIVE:
                continue
            deadline = _deadline(order.request.valid_until)
            if order.request.tif == "DAY" and last_in_session or deadline is not None and self.event.timestamp >= deadline:
                order.status = "EXPIRED"
                self._record(order,self.event.timestamp,"time_in_force")


class OrderAPI:
    def __init__(self, book):
        self.book = book

    def order(self, code, quantity, side="buy", **kwargs):
        return self.book.submit(OrderRequest(code,quantity,side,**kwargs))

    def cancel(self, order_id):
        return self.book.cancel(order_id)

    def get_order(self, order_id):
        return self.book.get(order_id)

    def orders(self, active_only=False):
        return self.book.orders(active_only)
