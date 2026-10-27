"""停牌跨除权时，旧每股价格不能直接用于新的股份数量。"""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import qbt
FREE=qbt.Costs(commission=0,min_commission=0,transfer_fee=0,stamp_duty=0)


def data(ratio):
    p=np.full((5,1),10.);p[2:]=np.nan
    f=np.ones_like(p);f[2:]=ratio
    amt=np.full_like(p,1e8);amt[2:]=0.
    return qbt.Dataset.from_arrays(np.busday_offset(np.datetime64('2024-01-02'),np.arange(5)),['600001.SH'],
        open=p,close=p,high=p,low=p,amt=amt,num_trades=np.ones_like(p),cumadj=f)


class HaltedActionsAudit(unittest.TestCase):
    def test_ffill_adjusts_per_share_price_without_inventing_a_quote(self):
        d=data(2.)
        self.assertEqual(d.close_ff[-1,0],5.)
        self.assertEqual(d.open_ff[-1,0],5.)
        self.assertTrue(np.isnan(d.close[-1,0]))
        np.testing.assert_allclose(d.close_adj,10.)
        self.assertEqual(qbt.Context(d,qbt.build_market(d,log=None)).ret(1)[-1,0],0.)

    def test_halted_split_does_not_double_equity_in_either_entry(self):
        for explicit in (False,True):
            for orders in (False,True):
                with self.subTest(explicit=explicit,orders=orders):
                    d=data(2.)
                    actions=qbt.CorporateActions([qbt.CorporateAction(str(d.dates[2]),d.codes[0],share_ratio=2.)]) if explicit else None
                    opts=dict(capital=10000,costs=FREE,corporate_actions=actions,max_adv_pct=1.,stale_policy='hold',asset_rules=qbt.CashAssetRules())
                    if orders:
                        def strategy(c):
                            if c.T==1:c.order(d.codes[0],100,tif='GTC')
                        r=qbt.run_orders(d,strategy,**opts)
                    else:
                        def strategy(c):
                            return c.long(np.ones(c.N,bool),gross=.1) if c.T==1 else qbt.HOLD
                        r=qbt.run_events(d,strategy,rebalance_tol=1.,**opts)
                    np.testing.assert_allclose(r.equity,10000.)
                    self.assertEqual(r.account.last_px[0],5.)
                    self.assertEqual(r.account.shares[0],200)
                    self.assertEqual(r.account.stale[0],3)

    def test_dividend_receivable_and_stale_mark_do_not_double_company_value(self):
        d=data(1.)
        actions=qbt.CorporateActions([qbt.CorporateAction(str(d.dates[2]),d.codes[0],cash_dividend=1.,pay_date=str(d.dates[4]))])
        def strategy(c):
            if c.T==1:c.order(d.codes[0],100,tif='GTC')
        r=qbt.run_orders(d,strategy,capital=10000,costs=FREE,corporate_actions=actions,max_adv_pct=1.,stale_policy='hold')
        np.testing.assert_allclose(r.equity,10000.)
        self.assertEqual(r.account.cash,9100.)
        self.assertEqual(r.account.last_px[0],9.)


if __name__=='__main__':unittest.main(verbosity=2)
