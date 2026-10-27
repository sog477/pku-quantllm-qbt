"""显式公司行动。输入必须是当时已知的每旧股分红、送转及实际到账日期。"""
from __future__ import annotations

from dataclasses import dataclass, replace
import csv
import numpy as np
from ._validation import naive_datetime_scalar


@dataclass(frozen=True)
class CorporateAction:
    date: str
    code: str
    cash_dividend: float = 0.
    share_ratio: float = 1.
    pay_date: str | None = None
    share_date: str | None = None
    cash_in_lieu_price: float | None = None
    record_date: str | None = None
    known_at: str | None = None
    withholding_rate: float | None = None
    entitlement_basis: str = 'ex_previous_close'


@dataclass(frozen=True)
class Settlement:
    """已知的实际清算事件；不是凭停牌时长猜出的成交。"""
    date: str
    code: str
    price: float
    pay_date: str | None = None
    known_at: str | None = None
    withholding_rate: float | None = None


@dataclass(frozen=True)
class SettlementRecord:
    index: int
    timestamp: object
    code: str
    quantity: int
    price: float
    proceeds: float
    pay_date: object = None


@dataclass(frozen=True)
class TaxAssessment:
    """Actual later withholding/assessment, e.g. sale-triggered dividend tax."""
    date: str
    code: str
    amount: float
    known_at: str | None = None
    reason: str = 'actual tax assessment'


class CorporateActions:
    def __init__(self, events=(), settlements=(), taxes=()):
        self.events = tuple(events)
        self.settlements = tuple(settlements)
        self.taxes = tuple(taxes)

    @classmethod
    def from_csv(cls, path):
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        events = []
        for row in rows:
            kw = {k: v for k, v in row.items() if v != ""}
            for k in ("cash_dividend", "share_ratio", "cash_in_lieu_price"):
                if k in kw:
                    kw[k] = float(kw[k])
            events.append(CorporateAction(**kw))
        return cls(events)

    def prepare(self, ds):
        return _ActionBook(self, ds)


class _ActionBook:
    def __init__(self, source, ds):
        self.ex = {}
        self.pay = {}
        self.release = {}
        self.settlements = {}
        self.covered = set()
        codes = {c: j for j, c in enumerate(ds.codes)}
        self.ds = ds
        self.closed = np.zeros(ds.N, bool)
        self.last_applied = -1
        self.settlement_log = []
        self.entitlements = {}
        self.entitlement_log = []
        self.record_rows = {}
        self.assessments = {}
        seen = set()
        for ev in source.events:
            if ev.withholding_rate is None:
                ev=replace(ev,withholding_rate=0.)  # legacy gross mode; strict ReferenceData requires explicit tax.
            if ev.code not in codes:
                raise ValueError(f"公司行动代码不在数据中：{ev.code}")
            i = ds.calendar.index(ev.date, "exact")
            j = codes[ev.code]
            if not np.isfinite(ev.withholding_rate) or not 0 <= ev.withholding_rate <= 1:
                raise ValueError('withholding_rate must be in [0,1]')
            if ev.entitlement_basis not in ('ex_previous_close','record_close'):
                raise ValueError('unknown entitlement basis')
            if ev.known_at is not None and naive_datetime_scalar(ev.known_at,'known_at') > ds.dates[i]:
                raise ValueError('corporate action was not known at its effective timestamp')
            if ev.record_date is not None:
                record = naive_datetime_scalar(ev.record_date,'record_date',keep_unit=True)
                if np.isnat(record):
                    raise ValueError('record_date cannot be NaT')
            if ev.entitlement_basis == 'record_close':
                if ev.record_date is None or ev.share_ratio != 1.:
                    raise ValueError('record_close requires record_date and cash-only distributions')
                record_i=ds.calendar.index(ev.record_date,'exact')
                if np.datetime64(ev.record_date).dtype==np.dtype('datetime64[D]'):
                    same_day=np.flatnonzero(ds.dates.astype('datetime64[D]')==np.datetime64(ev.record_date,'D'))
                    record_i=int(same_day[-1])
                if record_i >= i:
                    raise ValueError('record_close must precede ex date; use ex_previous_close for normal ex-date rights')
                self.record_rows.setdefault(record_i,[]).append((i,j))
            if (i, j) in seen:
                raise ValueError("同日同股公司行动须先合并为一条")
            seen.add((i, j))
            if (not np.isfinite(ev.share_ratio) or ev.share_ratio <= 0 or
                    not np.isfinite(ev.cash_dividend) or ev.cash_dividend < 0):
                raise ValueError("share_ratio 须为正，cash_dividend 须非负")
            if ev.cash_dividend and ev.pay_date is None:
                raise ValueError("现金分红必须提供实际 pay_date，不能把除息日当到账日")
            pay = naive_datetime_scalar(ev.date if ev.pay_date is None else ev.pay_date,'pay_date',keep_unit=True)
            release = naive_datetime_scalar(ev.date if ev.share_date is None else ev.share_date,'share_date',keep_unit=True)
            for value in (pay, release):
                if np.isnat(value):
                    raise ValueError("到账/可交易日期不能是 NaT")
                boundary = ds.dates[i].astype('datetime64[D]') if value.dtype == np.dtype('datetime64[D]') else ds.dates[i]
                if value < boundary:
                    raise ValueError("到账/可交易日期不能早于除权除息日")
            self.ex.setdefault(i, []).append((j, ev, pay, release))
            self.covered.add((i, j))
        for ev in source.settlements:
            if ev.withholding_rate is None:
                ev=replace(ev,withholding_rate=0.)
            if ev.code not in codes or not np.isfinite(ev.price) or ev.price < 0:
                raise ValueError("清算事件的代码或价格无效")
            i = ds.calendar.index(ev.date, "exact")
            key = (i, codes[ev.code])
            if key in self.settlements:
                raise ValueError("重复清算事件")
            if not np.isfinite(ev.withholding_rate) or not 0<=ev.withholding_rate<=1:
                raise ValueError('invalid terminal distribution withholding rate')
            if ev.known_at is not None and naive_datetime_scalar(ev.known_at,'known_at') > ds.dates[i]:
                raise ValueError('terminal event was not known at its effective timestamp')
            pay=naive_datetime_scalar(ev.date if ev.pay_date is None else ev.pay_date,'pay_date',keep_unit=True)
            if np.isnat(pay) or pay < ds.dates[i].astype('datetime64[D]'):
                raise ValueError('terminal payment cannot precede termination')
            self.settlements[key] = (ev.price,pay,ev.withholding_rate)
            self.covered.add(key)
        for ev in source.taxes:
            if ev.code not in codes or not np.isfinite(ev.amount) or ev.amount<0:
                raise ValueError('invalid tax assessment')
            i=ds.calendar.index(ev.date,'exact')
            if ev.known_at is not None and naive_datetime_scalar(ev.known_at,'known_at')>ds.dates[i]:
                raise ValueError('tax assessment was not yet known')
            self.assessments.setdefault(i,[]).append((codes[ev.code],ev))

    def start_at(self, previous_i):
        """空账户从区间中途开始，也必须继承此前已终止的证券状态。"""
        for (row, j) in self.settlements:
            if row <= previous_i:
                self.closed[j] = True
        self.last_applied = previous_i

    def advance(self, account, ds, previous_i, i):
        """按底层行情时间推进权益事件；稀疏策略时钟不能跳过分红或清算。"""
        affected = set()
        for row in range(previous_i + 1, i + 1):
            self.require_coverage(account, ds.cumadj_ff[row-1], ds.cumadj_ff[row], row)
            affected.update(j for j, *_ in self.ex.get(row, ()))
            affected.update(j for (event_row, j) in self.settlements if event_row == row)
            self.apply(account, ds, row)
        return affected

    def record_close(self, account, i):
        for target in self.record_rows.get(i,()):
            self.entitlements[target]=int(account.shares[target[1]])

    def _resize_locks(self, account, j, ratio, new_shares):
        """送转即时可用/合股时，把旧股份的不可卖状态映射到新份额。"""
        restricted = min(new_shares, int(np.ceil(account.restricted[j] * ratio - 1e-9)))
        frozen = min(new_shares-restricted, int(np.ceil(account.frozen[j] * ratio - 1e-9)))
        # 按到期顺序累计取整，避免多个解禁批次分别取整后超过总锁定份额。
        cumulative, allocated = 0, 0
        for date in sorted(self.release):
            updated = []
            for code, qty in self.release[date]:
                if code == j:
                    cumulative += qty
                    target = min(restricted, int(np.ceil(cumulative * ratio - 1e-9)))
                    updated.append((code, target-allocated))
                    allocated = target
                else:
                    updated.append((code, qty))
            self.release[date] = updated
        account.restricted[j], account.frozen[j] = restricted, frozen

    def apply(self, account, ds, i):
        if i == self.last_applied:
            return  # 同一事件回放幂等，避免分红/送股重复兑现。
        if i < self.last_applied:
            raise ValueError("公司行动不能逆序回放；新回测须重新 prepare")
        day = ds.dates[i]
        for j,ev in self.assessments.get(i,()):
            if not np.isfinite(account.tax_payables[j]+ev.amount):
                raise ValueError('tax liability overflow')
        # 先检查所有持仓的零股结算资料，失败时不得先给任何证券入账分红。
        for j, ev, pay, release in self.ex.get(i, ()):
            target = int(account.shares[j]) * ev.share_ratio
            dividend = int(account.shares[j]) * ev.cash_dividend
            if (not np.isfinite(target) or target >= 2.**63 or
                    not np.isfinite(dividend) or not np.isfinite(account.receivables[j]+dividend)):
                raise ValueError('公司行动派生股数或分红超出账本可表示范围，未入账')
            fraction = target - int(np.floor(target + 1e-9))
            if fraction > 1e-8:
                if ev.cash_in_lieu_price is None or not np.isfinite(ev.cash_in_lieu_price) or ev.cash_in_lieu_price < 0:
                    raise ValueError("存在不足一股的公司行动权益，须提供 cash_in_lieu_price")
                if ev.pay_date is None:
                    raise ValueError("不足一股的现金结算也须提供实际 pay_date")
        for j, ev, pay, release in self.ex.get(i, ()):
            old = int(account.shares[j])
            eligible = self.entitlements.get((i,j),0) if ev.entitlement_basis=='record_close' else old
            gross = eligible * ev.cash_dividend
            tax = gross * ev.withholding_rate
            dividend = gross-tax
            account.taxes += tax
            account.receivables[j] += dividend
            account.realized += dividend
            account.pnl[j] += dividend
            self.pay.setdefault(pay, []).append((j, dividend))
            self.entitlement_log.append(dict(index=i,code=ds.codes[j],eligible_shares=eligible,
                record_date=ev.record_date,basis=ev.entitlement_basis,gross=gross,tax=tax,net=dividend,pay_date=str(pay)))
            if not old:
                continue
            if getattr(ds,'explicit_split_basis',False):
                account.income_adj[j] += dividend/old*ds.cumadj_ff[max(i-1,0),j]
            target = old * ev.share_ratio
            new = int(np.floor(target + 1e-9))
            fraction = target - new
            if fraction > 1e-8:
                if ev.cash_in_lieu_price is None or not np.isfinite(ev.cash_in_lieu_price) or ev.cash_in_lieu_price < 0:
                    raise ValueError("存在不足一股的公司行动权益，须提供 cash_in_lieu_price")
                if ev.pay_date is None:
                    raise ValueError("不足一股的现金结算也须提供实际 pay_date")
                amount = fraction * ev.cash_in_lieu_price
                account.receivables[j] += amount
                self.pay.setdefault(pay, []).append((j, amount))
                gain = fraction * (ev.cash_in_lieu_price - account.cost_raw[j] / ev.share_ratio)
                account.realized += gain
                account.pnl[j] += gain
            account.cost_raw[j] /= ev.share_ratio
            account.last_px[j] = max(account.last_px[j]-ev.cash_dividend,0.)/ev.share_ratio
            if account._valuation is not None:
                account._valuation[j] = max(account._valuation[j]-ev.cash_dividend,0.)/ev.share_ratio
            account.shares[j] = new
            issued = max(new - old, 0)
            if ev.share_ratio < 1 or release <= day:
                self._resize_locks(account, j, ev.share_ratio, new)
            if release > day and issued:
                account.restricted[j] += issued
                self.release.setdefault(release, []).append((j, issued))
            account.shares_issued += new - old
            account.n_div_events += 1
        for date in list(self.pay):
            if date <= day:
                for j, amount in self.pay.pop(date):
                    account.receivables[j] -= amount
                    account.cash += amount
        for date in list(self.release):
            if date <= day:
                for j, qty in self.release.pop(date):
                    if not self.closed[j]:
                        account.restricted[j] -= qty
        for (date_i, j), (price,pay,tax_rate) in self.settlements.items():
            if date_i == i:
                quantity = int(account.shares[j])
                proceeds = account.force_close(j, price*(1-tax_rate), ds.cumadj_ff[i, j])
                account.taxes += quantity*price*tax_rate
                if pay > day:
                    account.cash -= proceeds
                    account.receivables[j] += proceeds
                    self.pay.setdefault(pay,[]).append((j,proceeds))
                if quantity:
                    self.settlement_log.append(SettlementRecord(i,ds.dates[i],ds.codes[j],quantity,float(price),proceeds,pay))
                self.closed[j] = True
        for j,ev in self.assessments.get(i,()):
            account.tax_payables[j]+=ev.amount
            account.taxes+=ev.amount
            account.realized-=ev.amount
            account.pnl[j]-=ev.amount
            if account.shares[j] and getattr(ds,'explicit_split_basis',False):
                account.income_adj[j]-=ev.amount/account.shares[j]*ds.cumadj_ff[i,j]
        account.settle_taxes()
        self.last_applied = i

    def require_coverage(self, account, factors_prev, factors_now, i):
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = factors_now / factors_prev
        changed = np.flatnonzero((account.shares > 0) & np.isfinite(ratio) & (np.abs(ratio - 1) > 1e-4))
        missing = [self.ds.codes[j] for j in changed if (i, int(j)) not in self.covered]
        if missing:
            raise ValueError(f"{self.ds.dates[i]} 持仓公司行动缺少显式事件：{missing[:5]}；请补数据或明确用 reinvest 近似")
