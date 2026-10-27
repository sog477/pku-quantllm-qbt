"""独立小行情核验5DR的窗口、方向、复权、时序及代码文件入口。"""
from pathlib import Path
import tempfile
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import qbt
from run_strategy import load_strategy, parse_params
from strategies.five_day_reversal import five_day_reversal

PARAMS=dict(k=1,min_amt=0,min_listed=0,gross=.5)
FREE=qbt.Costs(commission=0,min_commission=0,transfer_fee=0,stamp_duty=0)


def data(prefix=0):
    prices=np.full((9,3),10.)
    prices[5]=[8.,9.,11.]
    prices[6:]=[12.,9.,11.]
    prices=np.vstack([np.full((prefix,3),10.),prices])
    count=len(prices)
    opens=np.vstack([prices[0],prices[:-1]])
    return qbt.Dataset.from_arrays(np.busday_offset(np.datetime64("2024-01-02"),np.arange(count)),
        ["600001.SH","600002.SH","600003.SH"],open=opens,close=prices,
        high=np.maximum(opens,prices),low=np.minimum(opens,prices),amt=np.full((count,3),1e8),
        num_trades=np.full((count,3),100.),cumadj=np.ones((count,3)))


def context(d,params=None):
    return qbt.Context(d,qbt.build_market(d,log=None),params or PARAMS)


class FiveDRTests(unittest.TestCase):
    def test_window_and_direction(self):
        ctx=context(data())
        np.testing.assert_allclose(ctx.ret(5)[5],[-.2,-.1,.1],atol=1e-14)
        target=five_day_reversal(ctx)
        self.assertFalse(target.sel[:5].any())
        self.assertEqual(np.flatnonzero(target.sel[5]).tolist(),[0])
        self.assertEqual(np.flatnonzero(target.sel[6]).tolist(),[1])

    def test_first_fill_is_next_open_and_selection_updates_daily(self):
        d=data()
        r=qbt.run(d,five_day_reversal,params=PARAMS,capital=10000,costs=FREE,
                  asset_rules=qbt.CashAssetRules(),max_adv_pct=1.,rebalance_tol=0)
        self.assertEqual(r.trades[0][:3],(6,0,1))
        self.assertEqual(r.trades[0][4],8.)
        self.assertTrue(any(i==7 and j==0 and side==-1 for i,j,side,*_ in r.trades))
        self.assertTrue(any(i==7 and j==1 and side==1 for i,j,side,*_ in r.trades))

    def test_split_is_not_false_negative_return(self):
        d=data()
        d.close[5:,0]=5.
        d.cumadj[5:,0]=2.
        ctx=context(d)
        self.assertAlmostEqual(ctx.ret(5)[5,0],0.)
        self.assertEqual(np.flatnonzero(five_day_reversal(ctx).sel[5]).tolist(),[1])

    def test_filter_before_ranking_and_use_parameters(self):
        d=data(prefix=20)
        d.amt[:,0]=0.
        target=five_day_reversal(context(d,dict(PARAMS,min_amt=1)))
        self.assertEqual(np.flatnonzero(target.sel[25]).tolist(),[1])
        target=five_day_reversal(context(data(),dict(PARAMS,k=2,gross=.7)))
        self.assertEqual(target.sel[5].sum(),2)
        self.assertEqual(float(target.gross),.7)

    def test_future_data_cannot_change_past_selection(self):
        d=data()
        first=five_day_reversal(context(d)).sel.copy()
        d.close[7:]=[1.,100.,1000.]
        second=five_day_reversal(context(d)).sel
        np.testing.assert_array_equal(first[:7],second[:7])

    def test_prefix_equivalence(self):
        ctx=context(data())
        full=five_day_reversal(ctx)
        for end in (3,6,8):
            np.testing.assert_array_equal(full.sel[:end],five_day_reversal(ctx.sub(0,end)).sel)

    def test_bad_parameters_fail(self):
        for changes in (dict(k=0),dict(k=1.5),dict(min_listed=-1),dict(min_amt=-1),dict(gross=float("nan")),dict(kk=10)):
            with self.assertRaises(ValueError):
                five_day_reversal(context(data(),dict(PARAMS,**changes)))

    def test_load_file_and_module(self):
        fn,path=load_strategy(str(ROOT/"strategies/five_day_reversal.py")+":five_day_reversal")
        np.testing.assert_array_equal(fn(context(data())).sel,five_day_reversal(context(data())).sel)
        self.assertEqual(path.name,"five_day_reversal.py")
        self.assertTrue(callable(load_strategy("strategies.five_day_reversal:five_day_reversal")[0]))
        with tempfile.TemporaryDirectory() as folder:
            custom=Path(folder)/"自定义.py"
            custom.write_text("def my_strategy(ctx):\n    return ctx.cash()\n",encoding="utf-8")
            fn,_=load_strategy(str(custom)+":my_strategy")
            self.assertEqual(fn(context(data())).gross,0.)

    def test_parse_cli_parameters(self):
        self.assertEqual(parse_params(["k=20","gross=0.5"]),dict(k=20,gross=.5))
        with self.assertRaises(ValueError):parse_params(["k"])
        with self.assertRaises(ValueError):load_strategy("missing_function_name")


if __name__=="__main__":
    unittest.main(verbosity=2)
