"""跳过策略/成交回调，不能跳过市场状态与账户权益的时间推进。"""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import qbt

FREE=qbt.Costs(commission=0,min_commission=0,transfer_fee=0,stamp_duty=0)


def data():
    p=np.full((7,1),10.)
    return qbt.Dataset.from_arrays(np.busday_offset(np.datetime64('2024-01-02'),np.arange(7)),['600001.SH'],
        open=p,close=p,high=p,low=p,amt=np.full_like(p,1e8),num_trades=np.full_like(p,100),cumadj=np.ones_like(p))


def run(ds,orders=False,**kwargs):
    opts=dict(clock=qbt.BarClock(242/3,stride=3),costs=FREE,capital=10000,max_adv_pct=1.,asset_rules=qbt.CashAssetRules())
    opts.update(kwargs)
    if orders:
        def strategy(c):
            if c.T==1:c.order(ds.codes[0],100,tif='GTC')
        return qbt.run_orders(ds,strategy,**opts)
    def strategy(c):
        if c.T==1:return c.long(np.ones(c.N,bool),gross=.1)
        return qbt.HOLD
    return qbt.run_events(ds,strategy,rebalance_tol=1.,**opts)


class SparseClockAudit(unittest.TestCase):
    def test_missing_selected_quote_still_uses_last_completed_market_row(self):
        for orders in (False,True):
            d=data();d.close[2:4]=12.;d.close[4:]=np.nan;d.open[4:]=np.nan
            r=run(d,orders,stale_policy='hold')
            self.assertAlmostEqual(r.equity[-1],10200.)
            self.assertEqual(r.account.last_px[0],12.)

    def test_stale_age_counts_market_sessions_not_strategy_callbacks(self):
        for orders in (False,True):
            d=data();d.close[2:]=np.nan;d.open[2:]=np.nan;d.amt[2:]=0.
            r=run(d,orders,max_stale_days=2)
            self.assertEqual(r.account.stale[0],3)
            self.assertAlmostEqual(r.equity[-1],9000.)

    def test_intermediate_factor_changes_cannot_cancel_before_share_rounding(self):
        for orders in (False,True):
            d=data();d.cumadj[2:4]=1.005
            r=run(d,orders,stale_policy='hold')
            # 100 -> 100 shares + 5 cash -> 99 shares + 5.024875... cash.
            self.assertEqual(r.account.shares[0],99)
            self.assertAlmostEqual(r.account.cash,9010.02487562189)

    def test_callback_snapshot_includes_events_before_its_visible_timestamp(self):
        for orders in (False,True):
            d=data();seen=[]
            actions=qbt.CorporateActions([qbt.CorporateAction(str(d.dates[2]),d.codes[0],cash_dividend=1.,pay_date=str(d.dates[3]))])
            def strategy(c):
                seen.append(c.portfolio['cash'])
                if c.T==1:
                    if orders:c.order(d.codes[0],100,tif='GTC')
                    else:return c.long(np.ones(c.N,bool),gross=.1)
                return None if orders else qbt.HOLD
            opts=dict(clock=qbt.BarClock(242/3,stride=3),costs=FREE,capital=10000,max_adv_pct=1.,corporate_actions=actions,asset_rules=qbt.CashAssetRules())
            (qbt.run_orders if orders else qbt.run_events)(d,strategy,**opts)
            self.assertAlmostEqual(seen[1],9100.)


if __name__=='__main__':unittest.main(verbosity=2)
