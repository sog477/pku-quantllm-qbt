"""网页线索的独立行为核验；无错误时保留未复现证据。"""
import sys
from pathlib import Path
import tempfile
import unittest
import subprocess
import os
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import qbt
from qbt.factor_evaluation import daily_diagnostics

FREE=qbt.Costs(commission=0,min_commission=0,transfer_fee=0,stamp_duty=0)


def data(n=1):
    p=np.full((6,n),10.)
    return qbt.Dataset.from_arrays(np.busday_offset(np.datetime64('2024-01-02'),np.arange(6)),
        [f'60000{j+1}.SH' for j in range(n)],open=p,close=p,high=p,low=p,amt=np.full_like(p,1e8),num_trades=np.ones_like(p),cumadj=np.ones_like(p))


class SourceBehaviorAudit(unittest.TestCase):
    def test_fixed_orders_are_reproducible_across_python_hash_seeds(self):
        code='''import json,runpy,qbt,numpy as np
fixture=runpy.run_path('tests/test_audit_source_behaviors.py')
d=fixture['data'](2)
def strategy(c):
    if c.T==1:
        for symbol in d.codes:c.order(symbol,100,tif='GTC')
r=qbt.run_orders(d,strategy,costs=fixture['FREE'],capital=1000,max_adv_pct=1.,asset_rules=qbt.CashAssetRules())
print(json.dumps(dict(equity=r.equity.tolist(),shares=r.account.shares.tolist(),fills=[(f.code,f.quantity,f.price) for f in r.order_fills])))
'''
        output=[]
        for seed in ('0','41','777'):
            env=dict(os.environ,PYTHONHASHSEED=seed,PYTHONIOENCODING='utf-8')
            result=subprocess.check_output([sys.executable,'-B','-c',code],cwd=Path(__file__).resolve().parents[1],env=env)
            output.append(json.loads(result))
        self.assertTrue(all(value==output[0] for value in output))

    def test_history_negative_index_is_latest_completed_bar_in_every_prefix(self):
        d=data();d.close[:]=np.arange(10.,16.)[:,None]
        c=qbt.Context(d,qbt.build_market(d,log=None))
        for n in (2,3,6):
            self.assertEqual(c.sub(0,n).close()[-1,0],9+n)
            self.assertEqual(c.sub(0,n).close()[-2,0],8+n)

    def test_short_window_and_repeat_long_window_do_not_lose_fixed_scheduled_order(self):
        d=data()
        def strategy(c):
            if c.now==d.dates[2]:c.order(d.codes[0],100,tif='GTC')
        opts=dict(costs=FREE,max_adv_pct=1.)
        a=qbt.run_orders(d,strategy,**opts)
        short=qbt.run_orders(d,strategy,start=d.dates[3],**opts)
        b=qbt.run_orders(d,strategy,**opts)
        observed=lambda r:[(f.index,f.code,f.quantity,f.price) for f in r.order_fills]
        self.assertEqual(observed(a),observed(short))
        self.assertEqual(observed(a),observed(b))
        self.assertEqual(observed(a),[(3,d.codes[0],100,10.)])
        np.testing.assert_array_equal(a.equity,b.equity)

    def test_duplicate_symbols_are_rejected_before_any_backtest(self):
        d=data(2)
        panels={name:getattr(d,name) for name in ('open','high','low','close','amt','num_trades','cumadj')}
        with self.assertRaises(ValueError):qbt.Dataset.from_arrays(d.dates,[d.codes[0]]*2,**panels)
        clone=qbt.Dataset.from_arrays(d.dates,np.asarray(d.codes),**panels)
        self.assertEqual(clone.codes,d.codes)

    def test_unrelated_instrument_does_not_change_fixed_order_fill(self):
        outputs=[]
        for n in (1,2):
            d=data(n)
            def strategy(c):
                if c.T==1:c.order('600001.SH',100,tif='GTC')
            r=qbt.run_orders(d,strategy,costs=FREE,max_adv_pct=1.)
            outputs.append([(f.index,f.code,f.quantity,f.price) for f in r.order_fills])
        self.assertEqual(*outputs)

    def test_queries_and_report_cannot_change_result_or_subsequent_fills(self):
        d=data()
        def strategy(c):
            if c.T==1:c.order(d.codes[0],100,tif='GTC')
            c.orders();c.portfolio.copy();c.close();c.adj()
        r=qbt.run_orders(d,strategy,costs=FREE,max_adv_pct=1.)
        equity=r.equity.copy();cash=r.cash.copy();fills=r.order_fills
        for plotly in (False,True):
            qbt.report(r,plotly=plotly)
            r.metrics()
        np.testing.assert_array_equal(r.equity,equity)
        np.testing.assert_array_equal(r.cash,cash)
        self.assertEqual(r.order_fills,fills)

    def test_trade_recording_option_does_not_change_execution(self):
        d=data()
        strategy=lambda c:c.long(np.ones(c.N,bool),gross=.5)
        a=qbt.run(d,strategy,costs=FREE,record_trades=True)
        b=qbt.run(d,strategy,costs=FREE,record_trades=False)
        np.testing.assert_array_equal(a.equity,b.equity)
        self.assertIsNone(b.trades)
        self.assertTrue(a.trades)
        self.assertEqual(a.account.n_buy,b.account.n_buy)

    def test_dynamic_zero_risk_actually_exits(self):
        d=data()
        r=qbt.run(d,lambda c:c.long(np.ones(c.N,bool),gross=np.array([.5,0,0,0,0,0])),costs=FREE,max_adv_pct=1.)
        self.assertGreater(r.n_positions[1],0)
        self.assertEqual(r.n_positions[-1],0)
        self.assertEqual(r.account.n_sell,1)

    def test_five_groups_include_nondivisible_stock_count(self):
        score=np.arange(103.)[None,:]
        r=daily_diagnostics(score,score,np.ones_like(score,bool),min_stocks=10)
        self.assertEqual(r.signal_n[0],103)
        self.assertEqual(r.q1[0],10.)
        self.assertEqual(r.q5[0],92.5)

    def test_partial_closes_do_not_round_away_subcent_pnl(self):
        a=qbt.Account(10000,1);a.buy(0,1000,1.,.003,1.);a.unfreeze()
        for _ in range(1000):a.sell(0,1,1.000001,.0000002,1.)
        self.assertAlmostEqual(a.realized,.001,places=9)
        self.assertAlmostEqual(a.fees,.0032,places=9)
        self.assertAlmostEqual(a.cash,9999.9978,places=8)
        self.assertEqual(a.shares[0],0)

    def test_gtd_exact_deadline_fills_but_later_price_does_not(self):
        for last_touch,filled in ((2,True),(3,False)):
            times=np.array(['2024-01-02T09:59','2024-01-02T10:04','2024-01-02T10:05','2024-01-02T10:06'],dtype='datetime64[ns]')
            p=np.full((4,1),11.);p[last_touch]=10.
            d=qbt.BarDataset.from_arrays(times,['600001.SH'],open=p,close=p,high=p,low=p,amt=np.full_like(p,1e8),num_trades=np.ones_like(p),cumadj=np.ones_like(p))
            def strategy(c):
                if c.T==1:c.order(d.codes[0],100,limit_price=10.,tif='GTD',valid_until='2024-01-02T10:05')
            r=qbt.run_orders(d,strategy,costs=FREE,clock=qbt.BarClock(242),asset_rules=qbt.CashAssetRules())
            self.assertEqual(bool(r.order_fills),filled)
            self.assertEqual(r.orders[0].status,'FILLED' if filled else 'EXPIRED')

    def test_day_expiry_does_not_preempt_final_bar_close_phase(self):
        d=data();d.low[1,0]=9.
        def strategy(c):
            if c.T==1:c.order(d.codes[0],100,limit_price=9.)
        r=qbt.run_orders(d,strategy,costs=FREE,execution=qbt.BarTouchExecution(),max_adv_pct=1.)
        self.assertEqual(r.orders[0].status,'FILLED')
        self.assertEqual(r.order_fills[0].phase,'close')

    def test_changing_fee_configuration_changes_fills_and_recorded_cost(self):
        d=data()
        strategy=lambda c:c.long(np.ones(c.N,bool),gross=.9)
        a=qbt.run(d,strategy,costs=FREE)
        b=qbt.run(d,strategy,costs=qbt.Costs(min_commission=100))
        self.assertGreater(b.account.fees,a.account.fees)
        self.assertNotEqual(a.cost_description,b.cost_description)
        self.assertLess(b.equity[-1],a.equity[-1])

    def test_directory_reports_preserve_distinct_experiments_with_same_name(self):
        d=data();r=qbt.run_orders(d,lambda c:None,name='same_name')
        with tempfile.TemporaryDirectory() as folder:
            first=qbt.report(r,path=folder,plotly=False,title='first experiment')
            original=Path(first.path).read_bytes()
            second=qbt.report(r,path=folder,plotly=False,title='second experiment')
            self.assertNotEqual(first.path,second.path)
            self.assertEqual(Path(first.path).read_bytes(),original)

    def test_parallel_experiments_and_reports_get_independent_paths(self):
        from concurrent.futures import ThreadPoolExecutor
        from run_strategy import make_run_directory
        d=data();r=qbt.run_orders(d,lambda c:None,name='parallel')
        with tempfile.TemporaryDirectory() as folder:
            with ThreadPoolExecutor(4) as pool:
                runs=list(pool.map(lambda _:make_run_directory(folder),range(8)))
                reports=list(pool.map(lambda _:qbt.report(r,path=folder,plotly=False).path,range(8)))
            self.assertEqual(len(set(runs)),8)
            self.assertEqual(len(set(reports)),8)
            self.assertTrue(all(Path(p).is_file() for p in reports))

    def test_gap_order_uses_new_price_and_never_fabricates_cash(self):
        d=data();
        for name in ('open','close','high','low'):getattr(d,name)[1:]=20.
        def strategy(c):
            if c.T==1:c.order(d.codes[0],100,tif='GTC')
        opts=dict(capital=1500,costs=FREE,asset_rules=qbt.CashAssetRules(),max_adv_pct=1.)
        r=qbt.run_orders(d,strategy,**opts)
        self.assertEqual(r.account.shares[0],75)
        self.assertEqual(r.cash[-1],0.)
        self.assertEqual(r.orders[0].remaining,25)
        a=qbt.run(d,lambda c:c.long(np.ones(c.N,bool)),**opts)
        self.assertEqual(a.account.shares[0],75)
        np.testing.assert_allclose(a.equity,1500.)


if __name__=='__main__':unittest.main(verbosity=2)
