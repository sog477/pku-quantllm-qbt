"""独立Decimal总成本账本；不使用qbt均价重建参考成本。"""
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import qbt

D=lambda x:Decimal(str(x))


class ReferenceLedger:
    def __init__(self,n):
        self.cash=D(1000000);self.quantity=[0]*n;self.basis=[D(0) for _ in range(n)]
        self.realized=D(0);self.fees=D(0);self.pending=[]

    @property
    def receivable(self):return sum((amount for _,amount in self.pending),D(0))

    def buy(self,j,q,p,fee):
        self.cash-=q*p+fee;self.quantity[j]+=q;self.basis[j]+=q*p;self.fees+=fee

    def sell(self,j,q,p,fee):
        basis=self.basis[j]*q/self.quantity[j]
        self.cash+=q*p-fee;self.quantity[j]-=q;self.basis[j]-=basis
        self.realized+=q*p-basis;self.fees+=fee

    def action(self,j,ratio,dividend,lieu,pay):
        old=self.quantity[j]
        if not old:return
        target=old*ratio;new=int(target.to_integral_value(rounding=ROUND_FLOOR))
        fractional=target-new
        remaining=self.basis[j]*new/target
        dividend=old*dividend;fractional_cash=fractional*lieu
        self.realized+=dividend+fractional_cash-(self.basis[j]-remaining)
        self.pending.append((pay,dividend+fractional_cash))
        self.quantity[j]=new;self.basis[j]=remaining

    def pay(self,i):
        due=[x for x in self.pending if x[0]<=i]
        self.pending=[x for x in self.pending if x[0]>i]
        self.cash+=sum((v for _,v in due),D(0))

    def settle(self,j,p):
        self.cash+=self.quantity[j]*p
        self.realized+=self.quantity[j]*p-self.basis[j]
        self.quantity[j]=0;self.basis[j]=D(0)


class DecimalLedgerAudit(unittest.TestCase):
    def test_mixed_partial_trades_dividends_splits_and_terminal_cash(self):
        for seed in (19,47,103,1009):
            with self.subTest(seed=seed):
                rng=np.random.default_rng(seed);T,N=64,2
                dates=np.busday_offset(np.datetime64('2024-01-02'),np.arange(T))
                price=rng.integers(500,10000,size=(T,N))/100.
                ds=qbt.Dataset.from_arrays(dates,['600001.SH','600002.SH'],open=price,close=price,
                    high=price,low=price,amt=np.full_like(price,1e8),num_trades=np.ones_like(price),cumadj=np.ones_like(price))
                events=[];schedule={}
                for i in (6,14,23,35,46,58):
                    j=int(rng.integers(N));ratio=D(str(rng.choice([.5,1.5,2.])))
                    dividend=D(int(rng.integers(0,5)))/10;lieu=D('23.47');pay=i+3
                    schedule[i]=(j,ratio,dividend,lieu,pay)
                    events.append(qbt.CorporateAction(str(dates[i]),ds.codes[j],float(dividend),float(ratio),
                        str(dates[pay]),str(dates[i+2]),float(lieu)))
                settlement_i=60;settlement_price=D('15.13')
                book=qbt.CorporateActions(events,[qbt.Settlement(str(dates[settlement_i]),ds.codes[0],float(settlement_price))]).prepare(ds)
                a=qbt.Account(1000000,N);ref=ReferenceLedger(N);rules=qbt.CashAccountRules()
                for i in range(T):
                    if i in schedule:ref.action(*schedule[i])
                    ref.pay(i)
                    if i==settlement_i:ref.settle(0,settlement_price)
                    book.apply(a,ds,i);a.unfreeze()
                    for j in range(N):
                        if j==0 and i>=settlement_i:continue
                        p=D(price[i,j]);fee=D(5)
                        available=int(a.shares[j]-a.restricted[j])
                        if available and rng.random()<.55:
                            q=int(rng.integers(1,available+1))
                            a.sell(j,q,float(p),float(fee),1.);ref.sell(j,q,p,fee)
                        else:
                            affordable=max(0,int((ref.cash-fee)/p))
                            q=min(int(rng.integers(1,501)),affordable)
                            if q:a.buy(j,q,float(p),float(fee),1.);ref.buy(j,q,p,fee)
                    a.mark(price[i]);rules.check(a)
                    np.testing.assert_array_equal(a.shares,ref.quantity)
                    for got,want,label in ((a.cash,ref.cash,'cash'),(a.receivables.sum(),ref.receivable,'receivable'),
                                           (a.realized,ref.realized,'realized'),(a.fees,ref.fees,'fees')):
                        self.assertAlmostEqual(got,float(want),delta=1e-6,msg=f'{seed}/{i}/{label}')
                    for j in range(N):
                        basis=0. if not a.shares[j] else a.shares[j]*a.cost_raw[j]
                        self.assertAlmostEqual(basis,float(ref.basis[j]),delta=1e-6,msg=f'{seed}/{i}/basis/{j}')
                    value=sum((ref.quantity[j]*D(price[i,j]) for j in range(N)),D(0))
                    expected=ref.cash+ref.receivable+value
                    self.assertAlmostEqual(a.equity(),float(expected),delta=1e-6,msg=f'{seed}/{i}/equity')
                    # 独立经济恒等式：净资产=初始资金+毛已实现+未实现-全部费用。
                    identity=D(1000000)+ref.realized+value-sum(ref.basis)-ref.fees
                    self.assertAlmostEqual(float(identity),a.equity(),delta=1e-6)


if __name__=='__main__':unittest.main(verbosity=2)
