import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import qbt
from qbt.research import exit_schedule,run_traced,residual_inventory,data_quality


def dataset():
    x=np.full((10,2),10.)
    return qbt.Dataset.from_arrays(np.arange('2025-01-01','2025-01-11',dtype='datetime64[D]'),['600001.SH','600002.SH'],
        open=x,high=x+1,low=x-1,close=x,amt=np.full_like(x,1e8),num_trades=np.ones_like(x),cumadj=np.ones_like(x))


class ResearchTests(unittest.TestCase):
    def test_schedule_time_and_no_mutation(self):
        d=dataset();t=qbt.Target(sel=np.ones((10,2),bool),gross=.9)
        new,plan=exit_schedule(t,d.dates,'2025-01-10',3)
        np.testing.assert_allclose(new.gross[6:9],[.6,.3,0])
        self.assertEqual(t.gross,.9);self.assertEqual(plan['first_execution'],'2025-01-08')

    def test_invalid_schedule(self):
        d=dataset();t=qbt.Target(sel=np.ones((10,2),bool))
        with self.assertRaises(ValueError):exit_schedule(t,d.dates,'2025-02-01')
        with self.assertRaises(ValueError):exit_schedule(t,d.dates,'2025-01-10',True)

    def test_exit_ignores_later_target_changes(self):
        d=dataset();sel=np.ones((10,2),bool)
        original,_=exit_schedule(qbt.Target(sel=sel,gross=.9),d.dates,'2025-01-10',3)
        sel[7:]=False
        changed,_=exit_schedule(qbt.Target(sel=sel,gross=.9),d.dates,'2025-01-10',3)
        np.testing.assert_array_equal(original.sel,changed.sel)
        np.testing.assert_array_equal(original.gross,changed.gross)

    def test_trace_does_not_change_execution(self):
        d=dataset();t=qbt.Target(sel=np.ones((10,2),bool),gross=.98)
        a=qbt.run(d,t,capital=1e5);b,trace=run_traced(d,t,capital=1e5)
        np.testing.assert_allclose(a.equity,b.equity,rtol=0,atol=0)
        self.assertTrue(all(v['requested']==v['filled']+v['unfilled'] for v in trace))
        report=qbt.report(b,plotly=False)
        self.assertIn('退出与未完成目标',report.html)

    def test_capacity_residual_and_no_cash_sale(self):
        d=dataset();d._raw['amt'][:]=10000
        sel=np.ones((10,2),bool);sel[5:]=False
        r,trace=run_traced(d,qbt.Target(sel=sel,gross=.98),capital=1e5,max_adv_pct=.5)
        self.assertTrue(any('capacity_below_requested' in v['constraints'] for v in trace))
        self.assertTrue(all(v['actual_cash_proceeds']==0 for v in residual_inventory(r)))

    def test_no_buy_after_exit_start(self):
        d=dataset();t=qbt.Target(sel=np.ones((10,2),bool),gross=.98)
        r,_=run_traced(d,t,capital=1e5,stop_buys_from='2025-01-02')
        self.assertEqual(len(r.trades),0)
        self.assertGreater(r.scheduled_suppressed_buy_requests,0)

    def test_data_quality_finds_bad_bar(self):
        d=dataset();d._raw['high'][3,0]=5
        report=data_quality(d)
        self.assertEqual(report['inconsistent_ohlc'],1)

    def test_liquidation_suspension_keeps_stock(self):
        d=dataset();d._raw['amt'][7:]=0
        t,plan=exit_schedule(qbt.Target(sel=np.ones((10,2),bool),gross=.9),d.dates,'2025-01-10',3)
        r,trace=run_traced(d,t,capital=1e5,stop_buys_from=plan['first_execution'])
        self.assertEqual(r.n_positions[-1],2)
        self.assertEqual(len(residual_inventory(r)),2)
        self.assertTrue(any('no_traded_amount' in v['constraints'] for v in trace))


if __name__=='__main__':unittest.main()
