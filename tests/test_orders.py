"""订单生命周期、时间戳数据和可替换规则的行为验收。"""
from pathlib import Path
import sys
import pickle
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import qbt

FREE=qbt.Costs(commission=0,min_commission=0,transfer_fee=0,stamp_duty=0)


def data(intraday=False, count=6, n=1):
    times = (np.array(["2024-01-02T09:30","2024-01-02T09:31","2024-01-02T09:32",
                       "2024-01-03T09:30","2024-01-03T09:31","2024-01-03T09:32"],dtype="datetime64[ns]")
             if intraday else np.busday_offset(np.datetime64("2024-01-02"),np.arange(count)))
    prices=np.full((len(times),n),10.)
    cls=qbt.BarDataset if intraday else qbt.Dataset
    return cls.from_arrays(times,[f"60000{j+1}.SH" for j in range(n)],**(dict(previous_close=np.full(n,10.)) if intraday else {}),open=prices,close=prices,
        high=prices,low=prices,amt=np.full(prices.shape,1e5),num_trades=np.full(prices.shape,100.),cumadj=np.ones(prices.shape))


def run(d,strategy,**kwargs):
    options=dict(capital=10000.,costs=FREE,max_adv_pct=1.,asset_rules=qbt.CashAssetRules())
    options.update(kwargs)
    return qbt.run_orders(d,strategy,**options)


class OrderTests(unittest.TestCase):
    def test_market_order_next_open_and_history(self):
        d=data()
        visible=[]
        def strategy(ctx):
            visible.append((ctx.T,ctx.now,ctx.dates[-1]))
            if ctx.T==1: ctx.order(d.codes[0],100,tif="GTC")
        r=run(d,strategy)
        self.assertEqual(r.order_fills[0].index,1)
        self.assertEqual(r.orders[0].status,"FILLED")
        self.assertEqual(r.account.cash,9000.)
        self.assertEqual(visible[0],(1,d.dates[0],d.dates[0]))

    def test_return_request_and_cancel_command(self):
        d=data()
        def strategy(ctx):
            if ctx.T==1: return qbt.OrderRequest(d.codes[0],100,limit_price=9.,tif="GTC")
            if ctx.T==2: return qbt.Cancel(1)
        r=run(d,strategy)
        self.assertEqual(r.orders[0].status,"CANCELED")
        self.assertEqual(r.account.cash,10000.)

    def test_limit_waits_and_fills_at_better_open(self):
        d=data()
        d.open[3,0]=8.
        def strategy(ctx):
            if ctx.T==1: ctx.order(d.codes[0],100,limit_price=9.,tif="GTC")
        r=run(d,strategy)
        self.assertEqual(r.order_fills[0].index,3)
        self.assertEqual(r.order_fills[0].price,8.)

    def test_day_order_expires(self):
        d=data()
        def strategy(ctx):
            if ctx.T==1: ctx.order(d.codes[0],100,limit_price=9.)
        r=run(d,strategy)
        self.assertEqual(r.orders[0].status,"EXPIRED")
        self.assertFalse(r.order_fills)

    def test_gtc_remains_open_at_end(self):
        d=data()
        def strategy(ctx):
            if ctx.T==1: ctx.order(d.codes[0],100,limit_price=9.,tif="GTC")
        r=run(d,strategy)
        self.assertEqual(r.orders[0].status,"OPEN")

    def test_gtd_date_and_exact_timestamp(self):
        for intraday in (False,True):
            d=data(intraday)
            def strategy(ctx):
                if ctx.T==1:
                    ctx.order(d.codes[0],100,limit_price=9.,tif="GTD",valid_until=d.dates[1])
            r=run(d,strategy,**(dict(clock=qbt.BarClock(726)) if intraday else {}))
            self.assertEqual(r.orders[0].status,"EXPIRED")

    def test_cancel_after_partial_fill_keeps_trade(self):
        d=data()
        def strategy(ctx):
            if ctx.T==1: ctx.order(d.codes[0],300,tif="GTC")
            if ctx.T==2: self.assertTrue(ctx.cancel(1))
        r=run(d,strategy,max_adv_pct=.01)
        self.assertEqual((r.orders[0].status,r.orders[0].filled,r.orders[0].remaining),("CANCELED",100,200))
        self.assertEqual(r.account.shares[0],100)

    def test_new_order_already_past_deadline_never_fills(self):
        d=data()
        def strategy(ctx):
            if ctx.T==1:
                ctx.order(d.codes[0],100,tif="GTD",valid_until=d.dates[0])
        r=run(d,strategy)
        self.assertEqual(r.orders[0].status,"EXPIRED")
        self.assertFalse(r.order_fills)

    def test_custom_clock_invalid_frequency_rejected(self):
        clock=qbt.BarClock(242)
        clock.periods_per_year=float("nan")
        with self.assertRaises(ValueError): run(data(),lambda ctx: None,clock=clock)

    def test_capacity_shared_by_multiple_orders(self):
        d=data(count=2)
        def strategy(ctx):
            ctx.order(d.codes[0],100,tif="GTC")
            ctx.order(d.codes[0],100,tif="GTC")
        r=run(d,strategy,max_adv_pct=.01)
        self.assertEqual(sum(f.quantity for f in r.order_fills),100)
        self.assertEqual([o.status for o in r.orders],["FILLED","OPEN"])

    def test_legal_order_can_have_odd_partial_trade(self):
        d=data(count=3)
        d.amt[:]=55000.
        def strategy(ctx):
            if ctx.T==1: ctx.order(d.codes[0],200,tif="GTC")
        r=run(d,strategy,asset_rules=qbt.AStockRules(),max_adv_pct=.02)
        self.assertEqual([f.quantity for f in r.order_fills],[110,90])
        self.assertEqual(r.orders[0].status,"FILLED")

    def test_invalid_main_board_order_rejected(self):
        d=data()
        def strategy(ctx):
            if ctx.T==1: ctx.order(d.codes[0],110,tif="GTC")
        r=run(d,strategy,asset_rules=qbt.AStockRules())
        self.assertEqual(r.orders[0].status,"REJECTED")

    def test_commission_minimum_once_per_order_session(self):
        d=data(True)
        # 三次部分成交跨两个会话，最低佣金分别在两个会话计收。
        def strategy(ctx):
            if ctx.T==1: ctx.order(d.codes[0],300,tif="GTC")
        cost=qbt.Costs(commission=.00025,min_commission=5,transfer_fee=0,stamp_duty=0)
        r=run(d,strategy,clock=qbt.BarClock(726),max_adv_pct=.01,costs=cost)
        self.assertEqual([f.fee for f in r.order_fills],[5.,0.,5.])
        self.assertEqual(r.orders[0].fee,10.)

    def test_fifo_cash_does_not_overdraw(self):
        d=data()
        def strategy(ctx):
            if ctx.T==1:
                ctx.order(d.codes[0],100,tif="GTC")
                ctx.order(d.codes[0],100,tif="GTC")
        r=run(d,strategy,capital=1000.)
        self.assertEqual(r.account.cash,0.)
        self.assertEqual([o.status for o in r.orders],["FILLED","OPEN"])

    def test_intraday_t0_and_t1_use_sessions(self):
        for delay,sell_index in [(0,2),(1,3)]:
            d=data(True)
            def strategy(ctx):
                if ctx.T==1: ctx.order(d.codes[0],100,tif="GTC")
                if ctx.T==2: ctx.order(d.codes[0],100,side="sell",tif="GTC")
            r=run(d,strategy,clock=qbt.BarClock(726),asset_rules=qbt.CashAssetRules(settlement_days=delay))
            self.assertEqual([f.index for f in r.order_fills],[1,sell_index])
            self.assertEqual(r.account.cash,10000.)

    def test_mixed_asset_quantities_and_settlement(self):
        d=data(True,n=2)
        def strategy(ctx):
            if ctx.T==1:
                ctx.order(d.codes[0],100,tif="GTC")
                ctx.order(d.codes[1],1,tif="GTC")
            if ctx.T==2:
                ctx.order(d.codes[0],100,side="sell",tif="GTC")
                ctx.order(d.codes[1],1,side="sell",tif="GTC")
        assets=qbt.CashAssetRules(minimum=[100,1],step=[100,1],settlement_days=[1,0],price_limits=[True,False])
        r=run(d,strategy,clock=qbt.BarClock(726),asset_rules=assets)
        sells={f.code:f.index for f in r.order_fills if f.side=="sell"}
        self.assertEqual(sells,{d.codes[0]:3,d.codes[1]:2})

    def test_bar_touch_is_explicit_close_phase(self):
        d=data(count=2)
        d.low[1,0]=8.
        def strategy(ctx): ctx.order(d.codes[0],100,limit_price=9.)
        plain=run(d,strategy)
        touched=run(d,strategy,execution=qbt.BarTouchExecution())
        self.assertFalse(plain.order_fills)
        self.assertEqual((touched.order_fills[0].phase,touched.order_fills[0].price),("close",9.))

    def test_touch_sell_cannot_fund_open_buy(self):
        d=data(count=4,n=2)
        d.high[2,0]=11.
        def strategy(ctx):
            if ctx.T==1: ctx.order(d.codes[0],100,tif="GTC")
            if ctx.T==2:
                ctx.order(d.codes[0],100,side="sell",limit_price=11.,tif="GTC")
                ctx.order(d.codes[1],100,tif="GTC")
        r=run(d,strategy,capital=1000.,execution=qbt.BarTouchExecution())
        buy_second=[f for f in r.order_fills if f.code==d.codes[1]]
        self.assertEqual(buy_second[0].index,3)

    def test_unknown_instrument_is_rejected(self):
        r=run(data(),lambda ctx: qbt.OrderRequest("unknown",100))
        self.assertTrue(all(o.status=="REJECTED" for o in r.orders))

    def test_order_snapshots_cannot_mutate_book(self):
        d=data()
        def strategy(ctx):
            if ctx.T==1:
                oid=ctx.order(d.codes[0],100,tif="GTC")
                ctx.get_order(oid).filled=9999
        r=run(d,strategy)
        self.assertEqual(r.orders[0].filled,100)

    def test_bar_dataset_pickle_and_day_range(self):
        d=pickle.loads(pickle.dumps(data(True)))
        np.testing.assert_array_equal(d.dates,data(True).dates)
        self.assertEqual(d.calendar.slice_of("2024-01-03","2024-01-03"),(3,6))
        self.assertEqual(d.calendar.index("2024-01-03","exact"),3)

    def test_intraday_requires_explicit_clock(self):
        with self.assertRaises(ValueError): run(data(True),lambda ctx: None)

    def test_custom_clock_drives_both_entries(self):
        d=data()
        clock=qbt.BarClock(121,stride=2)
        target=qbt.run(d,lambda ctx: ctx.long(np.ones(ctx.N,bool),gross=.5),clock=clock,costs=FREE)
        manual=run(d,lambda ctx: None,clock=clock)
        np.testing.assert_array_equal(target.step_indices,[1,3,5])
        np.testing.assert_array_equal(target.dates,manual.dates)
        self.assertEqual(manual.metrics()["年化区间数"],121)

    def test_custom_account_rules_called(self):
        d=data()
        class Policy(qbt.CashAccountRules):
            calls=0
            def buying_power(self,account):
                type(self).calls+=1
                return min(account.cash,100.)
        def strategy(ctx):
            if ctx.T==1: ctx.order(d.codes[0],100,tif="GTC")
        r=run(d,strategy,account_rules=Policy())
        self.assertGreater(Policy.calls,0)
        self.assertEqual(r.order_fills[0].quantity,10)

    def test_target_gross_validation_delegates_to_policy(self):
        class Half(qbt.CashAccountRules):
            def validate_gross(self,gross):
                if (np.asarray(gross)>.5).any(): raise ValueError("half")
        d=data()
        with self.assertRaisesRegex(ValueError,"half"):
            qbt.run(d,lambda ctx:ctx.long(np.ones(ctx.N,bool),gross=.8),account_rules=Half())

    def test_target_intraday_t1_compatibility(self):
        d=data(True)
        def strategy(ctx):
            self.assertEqual(ctx.now,d.dates[ctx.T-1])
            self.assertEqual(ctx.execution_time,d.dates[ctx.T])
            return ctx.long(np.ones(ctx.N,bool),gross=.5) if ctx.T==1 else qbt.FLAT
        r=qbt.run_events(d,strategy,clock=qbt.BarClock(726),costs=FREE,max_adv_pct=1.)
        self.assertEqual([f[0] for f in r.trades],[1,3])

    def test_dynamic_gross_shortcut(self):
        d=data()
        r=qbt.run(d,lambda ctx:ctx.long(np.ones(ctx.N,bool),gross=np.full(ctx.T,.5)),costs=FREE)
        self.assertGreater(r.account.n_buy,0)

    def test_invalid_order_inputs(self):
        for kw in (dict(quantity=0),dict(quantity=1.5),dict(quantity=1,tif="GTD"),dict(quantity=1,limit_price=-1)):
            with self.assertRaises(ValueError): qbt.OrderRequest("x",**kw)

    def test_intraday_price_limit_reference_is_previous_session_close(self):
        d=data(True)
        d.close[1,0]=10.5
        d.open[2,0]=11.
        m=qbt.build_market(d,log=None)
        np.testing.assert_array_equal(m.reference[:3,0],[10.,10.,10.])
        self.assertFalse(m.can_buy(2)[0])
        self.assertEqual(m.stats["st_source"],"intraday_requires_metadata")

    def test_custom_session_can_cross_midnight(self):
        d=data(True)
        times=np.array(["2024-01-02T23:58","2024-01-02T23:59","2024-01-03T00:00","2024-01-03T08:00"],dtype="datetime64[ns]")
        d=qbt.BarDataset.from_arrays(times,d.codes,**{name:getattr(d,name)[:4] for name in ["open","high","low","close","amt","num_trades","cumadj"]})
        class Overnight(qbt.BarClock):
            def session_key(self,timestamp):
                return (timestamp-np.timedelta64(6,"h")).astype("datetime64[D]")
        def strategy(ctx):
            if ctx.T==1:ctx.order(d.codes[0],100,tif="GTC")
            if ctx.T==2:ctx.order(d.codes[0],100,side="sell",tif="GTC")
        r=run(d,strategy,clock=Overnight(726),asset_rules=qbt.CashAssetRules(settlement_days=1))
        self.assertEqual([f.index for f in r.order_fills],[1,3])

    def test_partial_expiry_and_fill_events_reconcile(self):
        d=data(count=2)
        def strategy(ctx):ctx.order(d.codes[0],200)
        r=run(d,strategy,max_adv_pct=.01)
        self.assertEqual(r.orders[0].status,"EXPIRED")
        self.assertEqual(r.orders[0].filled,100)
        self.assertIn("PARTIAL",[e.status for e in r.order_events])
        self.assertEqual(sum(f.quantity*f.price+f.fee for f in r.order_fills),10000.-r.account.cash)

    def test_intraday_report_filename_and_order_table(self):
        import tempfile
        d=data(True)
        r=run(d,lambda ctx:None,clock=qbt.BarClock(726))
        with tempfile.TemporaryDirectory(prefix="qbt_report_") as folder:
            report=qbt.report(r,path=folder,plotly=False)
            self.assertTrue(Path(report.path).exists())
            self.assertNotIn(":",Path(report.path).name)
            self.assertIn("订单生命周期",report.html)

    def test_compare_rejects_different_annualization_frequencies(self):
        d=data()
        a=run(d,lambda ctx:None)
        b=run(d,lambda ctx:None,clock=qbt.BarClock(121))
        with self.assertRaises(ValueError): a.compare(b)


if __name__=="__main__":
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):stream.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
