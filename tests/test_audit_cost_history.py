"""历史费用必须按成交日期与证券市场计算，不能把今天的费率套回2020年。"""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import qbt
from qbt.costs import evaluate_cost, SlippageOverlay


class CostHistoryAudit(unittest.TestCase):
    def test_default_transfer_schedule(self):
        c=qbt.Costs(commission=0,min_commission=0,stamp_duty=0)
        self.assertAlmostEqual(c.buy(1e6,'2022-04-28'),20.)
        self.assertAlmostEqual(c.sell(1e6,'2022-04-28'),20.)
        self.assertAlmostEqual(c.buy(1e6,'2022-04-29'),10.)

    def test_codes_distinguish_old_beijing_transfer_rate(self):
        c=qbt.Costs(commission=0,min_commission=0,stamp_duty=0)
        np.testing.assert_allclose(evaluate_cost(c.buy,[1e6,1e6],'2022-04-28',codes=['600001.SH','830001.BJ']),[20.,25.])
        np.testing.assert_allclose(evaluate_cost(c.buy,[1e6,1e6],'2022-04-29',codes=['600001.SH','830001.BJ']),[10.,10.])

    def test_explicit_transfer_override_is_preserved(self):
        c=qbt.Costs(commission=0,min_commission=0,stamp_duty=0,transfer_fee=0)
        self.assertEqual(c.buy(1e6,'2020-01-02'),0.)

    def test_slippage_overlay_preserves_code_context(self):
        c=SlippageOverlay(qbt.Costs(commission=0,min_commission=0,stamp_duty=0),10.)
        np.testing.assert_allclose(evaluate_cost(c.buy,[1e6,1e6],'2022-04-28',codes=['600001.SH','830001.BJ']),[1020.,1025.])

    def test_actual_execution_and_budget_use_market_specific_costs(self):
        dates=np.array(['2022-04-27','2022-04-28'],dtype='datetime64[D]');p=np.full((2,2),10.)
        codes=['600001.SH','830001.BJ']
        d=qbt.Dataset.from_arrays(dates,codes,open=p,close=p,high=p,low=p,amt=np.full_like(p,1e9),num_trades=np.ones_like(p),cumadj=np.ones_like(p))
        c=qbt.Costs(commission=0,min_commission=0,stamp_duty=0)
        opts=dict(capital=2000000.,costs=c,max_adv_pct=1.,asset_rules=qbt.CashAssetRules())
        r=qbt.run(d,lambda c:c.long(np.ones(c.N,bool),gross=.9),**opts)
        self.assertAlmostEqual(r.fees_cum[-1],40.5)
        def strategy(ctx):
            for code in codes:ctx.order(code,90000)
        r=qbt.run_orders(d,strategy,**opts)
        self.assertAlmostEqual(r.fees_cum[-1],40.5)


if __name__=='__main__':unittest.main(verbosity=2)
