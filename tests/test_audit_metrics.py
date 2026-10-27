"""网页中的报表/持仓损益/风险指标案例在本系统的独立核查。"""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import qbt
from qbt.metrics import perf,drawdown
from qbt._report import _contrib_section


def data():
    p=np.full((6,1),10.)
    return qbt.Dataset.from_arrays(np.busday_offset(np.datetime64('2024-01-02'),np.arange(6)),['600001.SH'],
        open=p,close=p,high=p,low=p,amt=np.full((6,1),1e8),num_trades=np.full((6,1),100.),cumadj=np.ones((6,1)))


class MetricsAuditTests(unittest.TestCase):
    def test_comparison_opportunity_percentage_uses_its_source_capital(self):
        d=data();d.open[1,0]=11.;d.high[1,0]=11.
        a=qbt.run_orders(d,lambda c:None,capital=1000)
        def strategy(c):
            if c.T==1:c.order(d.codes[0],100)
        b=qbt.run_orders(d,strategy,capital=10000)
        compared=a.compare(b)
        self.assertEqual(compared.n_blocked_buy,1)
        self.assertEqual(compared.capital,10000)

    def test_metrics_reject_misaligned_return_arrays_instead_of_truncating(self):
        r=qbt.run_orders(data(),lambda c:None)
        r.bench_ret=r.bench_ret[:-1]
        with self.assertRaisesRegex(ValueError,'对齐|长度|形状'):r.metrics()

    def test_compare_blocked_trades_are_from_the_same_overlapping_intervals(self):
        d=data();d.open[[1,4],0]=11.;d.high[[1,4],0]=11.
        def strategy(c):
            if c.T in (1,4):c.order(d.codes[0],100)
        a=qbt.run_orders(d,strategy)
        b=qbt.run_orders(d,strategy,start=d.dates[3])
        self.assertEqual(a.blocked_buy_n.sum(),2)
        comparison=a.compare(b)
        self.assertEqual(comparison.n_blocked_buy,1)
        self.assertEqual(comparison.blocked_amount,1100.)

    def test_compare_cannot_normalize_from_zero_overlap_equity(self):
        d=data();a=qbt.run_orders(d,lambda c:None)
        b=qbt.run_orders(d,lambda c:None,start=d.dates[3])
        a.equity[2:]=0.
        with self.assertRaisesRegex(ValueError,'起|零|归一化'):a.compare(b)

    def test_complete_loss_stays_minus_one_hundred_percent_after_zero_rows(self):
        result=perf(np.array([1000.,0.,0.,0.]),3)
        self.assertEqual(result['累计收益'],-1.)
        self.assertEqual(result['年化收益'],-1.)
        self.assertTrue(np.isfinite(result['年化波动']))

    def test_recovery_from_zero_valuation_does_not_invent_a_huge_return(self):
        d=data()
        for name in ('open','close','high','low'):getattr(d,name)[2:5]=np.nan
        d.amt[2:5]=0.;d.num_trades[2:5]=0.
        free=qbt.Costs(commission=0,min_commission=0,transfer_fee=0,stamp_duty=0)
        def strategy(c):
            if c.T==1:c.order(d.codes[0],100,tif='GTC')
        results=[qbt.run(d,lambda c:c.long(np.ones(c.N,bool)),capital=1000,costs=free,max_stale_days=1),
                 qbt.run_orders(d,strategy,capital=1000,costs=free,max_stale_days=1)]
        for r in results:
            self.assertEqual(r.equity[-2],0.)
            self.assertEqual(r.equity[-1],1000.)
            self.assertTrue(np.isnan(r.ret[-1]))
            self.assertEqual(r.metrics()['收益未定义区间数'],1)

    def test_first_day_loss_is_in_max_drawdown(self):
        _,value=drawdown([1.,.9,.95,1.])
        self.assertAlmostEqual(value,-.1)

    def test_realized_pnl_uses_entry_basis_not_previous_mark(self):
        a=qbt.Account(10000,1);a.buy(0,100,10.,0.,1.);a.unfreeze()
        a.mark(np.array([11.]));a.mark(np.array([12.]));a.mark(np.array([13.]))
        a.sell(0,100,13.,0.,1.)
        self.assertEqual(a.realized,300.)
        self.assertEqual(a.cash,10300.)

    def test_sortino_uses_downside_deviation_not_centered_negative_std(self):
        r=perf(np.array([1.,.99,.9801]),2)
        expected=(.99**242-1)/(.01*np.sqrt(242))
        self.assertTrue(np.isfinite(r['索提诺']))
        self.assertAlmostEqual(r['索提诺'],expected,places=10)

    def test_sparse_overlap_cannot_silently_double_annualization(self):
        d=data()
        a=qbt.run_orders(d,lambda c:None)
        b=qbt.run_orders(d,lambda c:None,clock=qbt.BarClock(242,stride=2))
        with self.assertRaisesRegex(ValueError,'时钟'):
            a.compare(b)

    def test_stock_contribution_includes_fee_only_loss(self):
        d=data()
        def strategy(ctx):
            if ctx.T==1:ctx.order(d.codes[0],100,tif='GTC')
        r=qbt.run_orders(d,strategy,costs=qbt.Costs(commission=0,min_commission=5,transfer_fee=0,stamp_duty=0))
        self.assertAlmostEqual(r.equity[-1]-r.equity[0],-5.)
        html=_contrib_section(r,d)
        self.assertIn('600001.SH',html)
        self.assertIn('费用',html)
        self.assertIn('净合计',html)


if __name__=='__main__':unittest.main(verbosity=2)
