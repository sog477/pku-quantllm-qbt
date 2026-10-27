"""来源#9899/#9929等线索：把成交、到期、公司行动放在同一时刻核验。"""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import qbt

FREE=qbt.Costs(commission=0,min_commission=0,transfer_fee=0,stamp_duty=0)


def bars(times):
    p=np.full((len(times),1),10.)
    return qbt.BarDataset.from_arrays(np.array(times,dtype='datetime64[ns]'),['600001.SH'],
        previous_close=10.,open=p,close=p,high=p,low=p,amt=np.full_like(p,1e8),
        num_trades=np.full_like(p,100.),cumadj=np.ones_like(p))


class BoundaryAudit(unittest.TestCase):
    def test_gtd_before_at_after_execution_boundary_for_both_phases(self):
        for phase in ('open','close'):
            for offset,expected in ((-1,False),(0,True),(1,True)):
                with self.subTest(phase=phase,offset_ns=offset):
                    d=bars(['2024-01-02T09:59','2024-01-02T10:00',
                            '2024-01-02T10:05','2024-01-02T10:06'])
                    if phase=='open':
                        for name in ('open','close','high','low'):getattr(d,name)[2:]=9.
                    else:d.low[2:,0]=9.;d.close[2:,0]=9.5
                    deadline=d.dates[2]+np.timedelta64(offset,'ns')
                    def strategy(c):
                        if c.T==1:c.order(d.codes[0],100,limit_price=9.5,tif='GTD',valid_until=deadline)
                    r=qbt.run_orders(d,strategy,clock=qbt.BarClock(242*240),costs=FREE,
                        execution=qbt.BarTouchExecution())
                    self.assertEqual(bool(r.order_fills),expected)
                    if expected:
                        self.assertEqual([(f.index,f.phase) for f in r.order_fills],[(2,phase)])
                        self.assertEqual(r.orders[0].status,'FILLED')
                    else:self.assertEqual(r.orders[0].status,'EXPIRED')

    def test_gtd_does_not_fill_when_only_the_following_bar_reaches_limit(self):
        d=bars(['2024-01-02T09:59','2024-01-02T10:00','2024-01-02T10:05','2024-01-02T10:06'])
        for name in ('open','close','high','low'):getattr(d,name)[3]=9.
        def strategy(c):
            if c.T==1:c.order(d.codes[0],100,limit_price=9.5,tif='GTD',valid_until=d.dates[2])
        r=qbt.run_orders(d,strategy,clock=qbt.BarClock(242*240),costs=FREE,execution=qbt.BarTouchExecution())
        self.assertFalse(r.order_fills)
        self.assertEqual(r.orders[0].status,'EXPIRED')
        expiry=[e for e in r.order_events if e.status=='EXPIRED']
        self.assertEqual(expiry[0].timestamp,d.dates[2])

    def test_day_order_can_fill_in_final_bar_close_before_expiry(self):
        d=bars(['2024-01-02T14:59','2024-01-02T15:00'])
        d.low[1,0]=9.;d.close[1,0]=9.5
        def strategy(c):c.order(d.codes[0],100,limit_price=9.5,tif='DAY')
        r=qbt.run_orders(d,strategy,clock=qbt.BarClock(242*240),costs=FREE,execution=qbt.BarTouchExecution())
        self.assertEqual([(f.index,f.phase) for f in r.order_fills],[(1,'close')])
        self.assertEqual(r.orders[0].status,'FILLED')
        self.assertFalse([e for e in r.order_events if e.status=='EXPIRED'])

    def test_corporate_action_cancels_pending_order_at_its_execution_deadline(self):
        d=bars(['2024-01-02T09:59','2024-01-02T10:00','2024-01-02T10:05'])
        for name in ('open','close','high','low'):getattr(d,name)[2]=5.
        d.cumadj[2]=2.
        actions=qbt.CorporateActions([qbt.CorporateAction(str(d.dates[2]),d.codes[0],share_ratio=2.)])
        def strategy(c):
            if c.T==1:c.order(d.codes[0],100,limit_price=5.,tif='GTD',valid_until=d.dates[2])
        r=qbt.run_orders(d,strategy,clock=qbt.BarClock(242*240),costs=FREE,corporate_actions=actions)
        self.assertFalse(r.order_fills)
        self.assertEqual(r.orders[0].status,'CANCELED')
        self.assertEqual(r.orders[0].reason,'corporate_action')

    def test_dividend_split_and_terminal_settlement_at_one_label_conserve_claim(self):
        d=bars(['2024-01-02T09:59','2024-01-02T10:00','2024-01-03T10:00'])
        for name in ('open','close','high','low'):getattr(d,name)[2]=4.5
        d.cumadj[2]=2.
        actions=qbt.CorporateActions([
            qbt.CorporateAction(str(d.dates[2]),d.codes[0],cash_dividend=1.,share_ratio=2.,pay_date=str(d.dates[2]))],
            settlements=[qbt.Settlement(str(d.dates[2]),d.codes[0],4.5)])
        def strategy(c):
            if c.T==1:c.order(d.codes[0],100)
        r=qbt.run_orders(d,strategy,capital=1000,clock=qbt.BarClock(242*240),costs=FREE,
            corporate_actions=actions)
        self.assertEqual(r.account.cash,1000.)
        self.assertEqual(r.account.shares[0],0)
        self.assertEqual(r.account.receivables[0],0.)
        self.assertEqual(len(r.settlements),1)
        self.assertEqual(r.settlements[0].proceeds,900.)
        self.assertEqual(len(r.order_fills),1)


if __name__=='__main__':unittest.main(verbosity=2)
