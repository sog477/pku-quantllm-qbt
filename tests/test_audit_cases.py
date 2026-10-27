"""共享聊天案例迁移：个人系统的公司行动、策略意图与陈旧成交反例。"""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import qbt

FREE=qbt.Costs(commission=0,min_commission=0,transfer_fee=0,stamp_duty=0)


def data(intraday=False):
    times=(np.array(['2024-01-02T09:30','2024-01-02T09:31','2024-01-03T09:30','2024-01-03T09:31','2024-01-04T09:30','2024-01-04T09:31'],dtype='datetime64[ns]')
           if intraday else np.busday_offset(np.datetime64('2024-01-02'),np.arange(6)))
    px=np.full((6,1),10.)
    cls=qbt.BarDataset if intraday else qbt.Dataset
    return cls.from_arrays(times,['600001.SH'],**(dict(previous_close=[10.]) if intraday else {}),open=px,close=px,high=px,low=px,
        amt=np.full((6,1),1e8),num_trades=np.full((6,1),100.),cumadj=np.ones((6,1)))


class AuditCases(unittest.TestCase):
    def test_same_action_replay_cannot_double_credit(self):
        d=data(); a=qbt.Account(10000,1); a.buy(0,100,10.,0.,1.)
        book=qbt.CorporateActions([qbt.CorporateAction(str(d.dates[2]),d.codes[0],cash_dividend=1.,pay_date=str(d.dates[2]))]).prepare(d)
        book.apply(a,d,2); cash=a.cash
        book.apply(a,d,2)
        self.assertEqual(a.cash,cash)
        self.assertEqual(a.n_div_events,1)

    def test_same_day_intraday_dividend_is_valid(self):
        d=data(True)
        source=qbt.CorporateActions([qbt.CorporateAction('2024-01-03',d.codes[0],cash_dividend=1.,pay_date='2024-01-03')])
        book=source.prepare(d)
        a=qbt.Account(10000,1);a.buy(0,100,10.,0.,1.)
        book.apply(a,d,2)
        self.assertEqual(a.cash,9100.)

    def test_invalid_payment_nat_rejected(self):
        d=data()
        with self.assertRaises(ValueError):
            qbt.CorporateActions([qbt.CorporateAction(str(d.dates[2]),d.codes[0],cash_dividend=1.,pay_date='NaT')]).prepare(d)

    def test_settlement_before_restricted_release_keeps_ledger_nonnegative(self):
        d=data();a=qbt.Account(10000,1);a.buy(0,100,10.,0.,1.)
        source=qbt.CorporateActions([qbt.CorporateAction(str(d.dates[2]),d.codes[0],share_ratio=2.,share_date=str(d.dates[4]))],
            [qbt.Settlement(str(d.dates[3]),d.codes[0],5.)])
        book=source.prepare(d)
        for i in range(2,6):book.apply(a,d,i)
        self.assertEqual(a.restricted[0],0)
        self.assertEqual(a.shares[0],0)
        self.assertEqual(a.cash,10000.)

    def test_orders_never_expand_when_position_increases(self):
        d=data()
        def strategy(ctx):
            if ctx.T==1:ctx.order(d.codes[0],100,tif='GTC')
            if ctx.T==2:
                ctx.order(d.codes[0],100,side='sell',limit_price=11.,tif='GTC')
                ctx.order(d.codes[0],200,tif='GTC')
        r=qbt.run_orders(d,strategy,costs=FREE,max_adv_pct=1.)
        self.assertEqual(r.orders[1].request.quantity,100)
        self.assertLessEqual(r.orders[1].filled,100)
        self.assertEqual(r.account.shares[0],300)

    def test_gtc_cannot_use_filled_forward_open_when_current_quote_missing(self):
        d=data();d.open[2:,0]=np.nan;d.close[2:,0]=np.nan
        def strategy(ctx):
            if ctx.T==1:ctx.order(d.codes[0],100,limit_price=9.,tif='GTC')
        r=qbt.run_orders(d,strategy,costs=FREE,max_adv_pct=1.)
        self.assertFalse(r.order_fills)
        self.assertEqual(r.account.cash,1e6)

    def test_sparse_clock_preserves_dividend_and_settlement(self):
        d=data()
        actions=qbt.CorporateActions([qbt.CorporateAction(str(d.dates[2]),d.codes[0],cash_dividend=1.,pay_date=str(d.dates[2]))],
            [qbt.Settlement(str(d.dates[4]),d.codes[0],10.)])
        for orders in (False,True):
            def strategy(ctx):
                if orders:
                    if ctx.T==1:ctx.order(d.codes[0],100,tif='GTC')
                else:
                    return ctx.long(np.ones(ctx.N,bool),gross=.1)
            runner=qbt.run_orders if orders else qbt.run
            r=runner(d,strategy,capital=10000,costs=FREE,max_adv_pct=1.,corporate_actions=actions,
                     clock=qbt.BarClock(121,stride=2),**({} if orders else dict(rebalance_tol=1.)))
            self.assertEqual(r.account.shares[0],0)
            self.assertEqual(r.account.cash,10100.)

    def test_explicit_dividend_cancels_pending_order_without_factor_jump(self):
        d=data()
        actions=qbt.CorporateActions([qbt.CorporateAction(str(d.dates[2]),d.codes[0],cash_dividend=1.,pay_date=str(d.dates[2]))])
        def strategy(ctx):
            if ctx.T==1:ctx.order(d.codes[0],100,limit_price=9.,tif='GTC')
        r=qbt.run_orders(d,strategy,costs=FREE,corporate_actions=actions)
        self.assertEqual(r.orders[0].status,'CANCELED')

    def test_invalid_fractional_action_does_not_partially_credit_cash(self):
        d=data();a=qbt.Account(10000,1);a.buy(0,100,10.,0.,1.)
        book=qbt.CorporateActions([qbt.CorporateAction(str(d.dates[2]),d.codes[0],cash_dividend=1.,pay_date=str(d.dates[2]),share_ratio=1.005)]).prepare(d)
        for attempt in range(2):
            with self.assertRaises(ValueError):book.apply(a,d,2)
            self.assertEqual(a.receivables[0],0.)
            self.assertEqual(a.cash,9000.)
            self.assertEqual(a.realized,0.)

    def test_sell_fee_cannot_leave_negative_cash_or_partially_mutated_ledger(self):
        a=qbt.Account(1.,1);a.buy(0,1,1.,0.,1.);a.unfreeze()
        with self.assertRaises(AssertionError):a.sell(0,1,1.,5.,1.)
        self.assertEqual(a.cash,0.)
        self.assertEqual(a.shares[0],1)

    def test_settlement_not_double_counted_when_replayed(self):
        d=data();a=qbt.Account(10000,1);a.buy(0,100,10.,0.,1.)
        book=qbt.CorporateActions(settlements=[qbt.Settlement(str(d.dates[2]),d.codes[0],10.)]).prepare(d)
        book.apply(a,d,2);book.apply(a,d,2)
        self.assertEqual(a.cash,10000.)
        self.assertEqual(a.shares[0],0)
        self.assertEqual(a.n_force_close,1)

    def test_sell_fee_shortfall_waits_in_both_engines(self):
        d=data();d.open[:]=1.;d.close[:]=1.;d.high[:]=1.;d.low[:]=1.
        class Fee:
            def buy(self,amount,day=None):return np.asarray(amount)*0.
            def sell(self,amount,day=None):return np.where(np.asarray(amount)>0,5.,0.)
        for orders in (False,True):
            def strategy(ctx):
                if orders:
                    if ctx.T==1:ctx.order(d.codes[0],1,tif='GTC')
                    if ctx.T==2:ctx.order(d.codes[0],1,side='sell',tif='GTC')
                else:
                    return ctx.long(np.arange(ctx.T)[:,None]==0,gross=1.)
            run=qbt.run_orders if orders else qbt.run
            r=run(d,strategy,capital=1.,costs=Fee(),asset_rules=qbt.CashAssetRules(),max_adv_pct=1.)
            self.assertEqual(r.account.shares[0],1)
            self.assertEqual(r.account.cash,0.)
            self.assertFalse(any(side==-1 for _,_,side,*_ in r.trades))

    def test_intraday_explicit_payment_timestamp_is_not_truncated(self):
        d=data(True);a=qbt.Account(10000,1);a.buy(0,100,10.,0.,1.)
        book=qbt.CorporateActions([qbt.CorporateAction('2024-01-03',d.codes[0],cash_dividend=1.,pay_date='2024-01-03T09:31')]).prepare(d)
        book.apply(a,d,2)
        self.assertEqual(a.cash,9000.)
        self.assertEqual(a.receivables[0],100.)
        book.apply(a,d,3)
        self.assertEqual(a.cash,9100.)
        self.assertEqual(a.receivables[0],0.)

    def test_intrasession_split_does_not_make_t1_inventory_sellable(self):
        d=data(True);a=qbt.Account(10000,1);a.buy(0,100,10.,0.,1.)
        book=qbt.CorporateActions([qbt.CorporateAction(str(d.dates[1]),d.codes[0],share_ratio=2.)]).prepare(d)
        book.apply(a,d,1)
        self.assertEqual(a.shares[0],200)
        self.assertEqual(qbt.CashAccountRules().sellable(a)[0],0)

    def test_reverse_split_scales_restricted_inventory_and_release(self):
        d=data();a=qbt.Account(10000,1);a.buy(0,100,10.,0.,1.);a.unfreeze()
        source=qbt.CorporateActions([
            qbt.CorporateAction(str(d.dates[1]),d.codes[0],share_ratio=2.,share_date=str(d.dates[4])),
            qbt.CorporateAction(str(d.dates[2]),d.codes[0],share_ratio=.5)])
        book=source.prepare(d)
        book.apply(a,d,1);book.apply(a,d,2)
        self.assertEqual(a.shares[0],100)
        self.assertEqual(a.restricted[0],50)
        self.assertEqual(qbt.CashAccountRules().sellable(a)[0],50)
        book.apply(a,d,4)
        self.assertEqual(a.restricted[0],0)

    def test_inferred_reinvestment_also_preserves_t1_lock(self):
        a=qbt.Account(10000,1);a.buy(0,100,10.,0.,1.)
        a.apply_corporate_actions(np.array([1.]),np.array([2.]),np.array([5.]))
        self.assertEqual(a.shares[0],200)
        self.assertEqual(qbt.CashAccountRules().sellable(a)[0],0)

    def test_cash_policy_catches_invalid_inventory_not_just_negative_cash(self):
        a=qbt.Account(10000,1);a.restricted[0]=-1
        with self.assertRaises(AssertionError):qbt.CashAccountRules().check(a)
        a.restricted[0]=0;a.cash=float('nan')
        with self.assertRaises(AssertionError):qbt.CashAccountRules().check(a)

    def test_resting_order_checks_quote_age_at_each_fill_attempt(self):
        d=data(True)
        for field in ('open','close','high','low'):getattr(d,field)[2:,0]=8.
        times=np.broadcast_to(d.dates[:,None],(d.T,d.N)).copy()
        times[2,0]=d.dates[1]
        market=qbt.build_market(d,log=None,metadata=qbt.MarketMetadata(quote_time=times,max_quote_age=np.timedelta64(1,'m')))
        def strategy(ctx):
            if ctx.T==1:ctx.order(d.codes[0],100,limit_price=9.,tif='GTC')
        r=qbt.run_orders(d,strategy,market=market,costs=FREE,max_adv_pct=1.,clock=qbt.BarClock(726),asset_rules=qbt.CashAssetRules())
        self.assertEqual([f.index for f in r.order_fills],[3])
        self.assertEqual(r.order_fills[0].evidence_age_ns,0)
        self.assertEqual(r.order_fills[0].source_timestamp,d.dates[3])

    def test_future_and_nat_quote_time_never_trade_even_with_limits_off(self):
        d=data(True);times=np.broadcast_to(d.dates[:,None],(d.T,d.N)).copy()
        times[1,0]=d.dates[1]+np.timedelta64(1,'m');times[2:,0]=np.datetime64('NaT')
        m=qbt.build_market(d,log=None,metadata=qbt.MarketMetadata(quote_time=times))
        def strategy(ctx):
            if ctx.T==1:ctx.order(d.codes[0],100,tif='GTC')
        r=qbt.run_orders(d,strategy,market=m,costs=FREE,clock=qbt.BarClock(726),asset_rules=qbt.CashAssetRules(),limit_rule='off')
        self.assertFalse(r.order_fills)

    def test_quote_age_guard_is_shared_by_target_engine(self):
        d=data(True);times=np.full((d.T,d.N),d.dates[0],dtype='datetime64[ns]')
        m=qbt.build_market(d,log=None,metadata=qbt.MarketMetadata(quote_time=times))
        r=qbt.run(d,lambda c:c.long(np.ones(c.N,bool)),market=m,costs=FREE,clock=qbt.BarClock(726),asset_rules=qbt.CashAssetRules())
        self.assertFalse(r.trades)

    def test_scalar_cost_model_does_not_undercharge_multiorder_buy_budget(self):
        prices=np.full((3,2),10.)
        d=qbt.Dataset.from_arrays(np.busday_offset(np.datetime64('2024-01-02'),np.arange(3)),['600001.SH','600002.SH'],
            open=prices,close=prices,high=prices,low=prices,amt=np.full((3,2),1e8),num_trades=np.full((3,2),100.),cumadj=np.ones((3,2)))
        class ScalarFee:
            def buy(self,amount,day=None):return 5.
            def sell(self,amount,day=None):return 5.
        class VectorFee:
            def buy(self,amount,day=None):return np.full_like(np.asarray(amount,float),5.)
            def sell(self,amount,day=None):return np.full_like(np.asarray(amount,float),5.)
        options=dict(capital=27.,asset_rules=qbt.CashAssetRules(),max_adv_pct=1.)
        a=qbt.run(d,lambda c:c.long(np.ones(c.N,bool)),costs=ScalarFee(),**options)
        b=qbt.run(d,lambda c:c.long(np.ones(c.N,bool)),costs=VectorFee(),**options)
        np.testing.assert_array_equal(a.equity,b.equity)
        self.assertGreaterEqual(a.cash.min(),0.)

    def test_zero_quote_cannot_create_a_fake_total_loss(self):
        d=data();d.close[2,0]=0.
        def strategy(ctx):
            if ctx.T==1:ctx.order(d.codes[0],100,tif='GTC')
        r=qbt.run_orders(d,strategy,costs=FREE,max_adv_pct=1.)
        np.testing.assert_allclose(r.equity,1e6)
        self.assertTrue(r.market.suspended[2,0])
        self.assertEqual(d.close_ff[2,0],10.)

    def test_bad_zero_factor_fails_instead_of_erasing_shares(self):
        d=data();d.cumadj[2:,0]=0.
        def strategy(ctx):
            if ctx.T==1:ctx.order(d.codes[0],100,tif='GTC')
        with self.assertRaisesRegex(ValueError,'复权因子'):
            qbt.run_orders(d,strategy,costs=FREE,max_adv_pct=1.)

    def test_fractional_lot_metadata_is_not_silently_truncated(self):
        d=data()
        for kwargs in (dict(step=2.5),dict(minimum=2.5),dict(settlement_days=1.8)):
            with self.assertRaises(ValueError):qbt.CashAssetRules(**kwargs).bind(d)

    def test_invalid_market_band_does_not_truncate_into_a_valid_band(self):
        d=data()
        for value in (10.9,266.):
            with self.assertRaises(ValueError):qbt.build_market(d,log=None,band_override=np.full((d.T,d.N),value))

    def test_asset_limit_flags_reject_unknown_values(self):
        d=data()
        for value in (np.nan,2,'False'):
            with self.assertRaises(ValueError):qbt.CashAssetRules(price_limits=value).bind(d)

    def test_partial_fills_accumulate_instead_of_replacing_inventory(self):
        d=data();d.amt[0:2]=4550.;d.amt[2:]=250.
        def strategy(ctx):
            if ctx.T==1:ctx.order(d.codes[0],480,tif='GTC')
        r=qbt.run_orders(d,strategy,costs=FREE,capital=10000.,max_adv_pct=1.,asset_rules=qbt.CashAssetRules())
        self.assertEqual([f.quantity for f in r.order_fills],[455,25])
        self.assertEqual(r.account.shares[0],480)
        self.assertEqual(r.orders[0].filled,480)

    def test_reverse_split_float_error_cannot_cash_out_a_whole_share(self):
        for quantity in (8,16):
            for explicit in (False,True):
                a=qbt.Account(10000,1);a.buy(0,quantity,10.,0.,1.)
                d=data()
                ratio=.125-1e-14
                if explicit:
                    book=qbt.CorporateActions([qbt.CorporateAction(str(d.dates[2]),d.codes[0],share_ratio=ratio)]).prepare(d)
                    book.apply(a,d,2)
                else:
                    a.apply_corporate_actions(np.array([1.]),np.array([ratio]),np.array([80.]))
                self.assertEqual(a.shares[0],quantity//8)
                self.assertAlmostEqual(a.cash,10000-quantity*10.)

    def test_known_settlement_is_not_fabricated_market_turnover(self):
        d=data()
        actions=qbt.CorporateActions(settlements=[qbt.Settlement(str(d.dates[3]),d.codes[0],10.)])
        def strategy(ctx):
            if ctx.T==1:ctx.order(d.codes[0],100,tif='GTC')
        r=qbt.run_orders(d,strategy,capital=10000,costs=FREE,max_adv_pct=1.,corporate_actions=actions)
        self.assertEqual(r.account.n_force_close,1)
        self.assertEqual(r.account.n_sell,0)
        self.assertEqual(r.turnover_cum[-1],1000.)
        self.assertEqual(len(r.trades),1)
        self.assertEqual(r.settlements[0].proceeds,1000.)
        self.assertEqual(r.settlements[0].quantity,100)

    def test_settlement_before_start_prevents_ghost_reentry(self):
        d=data()
        for settlement_i,start_i in ((0,1),(2,3)):
            actions=qbt.CorporateActions(settlements=[qbt.Settlement(str(d.dates[settlement_i]),d.codes[0],10.)])
            opts=dict(start=d.dates[start_i],corporate_actions=actions,costs=FREE,max_adv_pct=1.)
            a=qbt.run(d,lambda c:c.long(np.ones(c.N,bool)),**opts)
            b=qbt.run_orders(d,lambda c:qbt.OrderRequest(d.codes[0],100,tif='GTC'),**opts)
            self.assertFalse(a.trades)
            self.assertFalse(b.trades)
            self.assertEqual(a.account.shares[0],0)
            self.assertEqual(b.account.shares[0],0)

    def test_cached_market_from_another_dataset_is_not_silently_used(self):
        d=data();other=data();other.open[:]=20.;other.close[:]=20.
        market=qbt.build_market(d,log=None)
        with self.assertRaisesRegex(ValueError,'market'):
            qbt.run(other,lambda c:c.long(np.ones(c.N,bool)),market=market,costs=FREE)
        with self.assertRaisesRegex(ValueError,'market'):
            qbt.run_orders(other,lambda c:None,market=market,costs=FREE)


if __name__=='__main__':unittest.main(verbosity=2)
