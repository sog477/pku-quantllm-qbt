"""05:20续查：跨组件组合的资金、数量和市场容量约束。"""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import qbt

FREE=qbt.Costs(commission=0,min_commission=0,transfer_fee=0,stamp_duty=0)


def data(count=4,n=1):
    p=np.full((count,n),10.)
    return qbt.Dataset.from_arrays(np.busday_offset(np.datetime64('2024-01-02'),np.arange(count)),
        [f'60000{j+1}.SH' for j in range(n)],open=p,close=p,high=p,low=p,
        amt=np.full_like(p,100000.),num_trades=np.full_like(p,100.),cumadj=np.ones_like(p))


class ParticipationAudit(unittest.TestCase):
    def test_repeated_broker_execute_cannot_reuse_the_same_bar_capacity(self):
        d=data(3);b=qbt.Broker(qbt.build_market(d,log=None),costs=FREE,max_adv_pct=.01)
        a=qbt.Account(10000,1)
        for _ in range(2):b.execute(a,1,np.array([200]),d.dates[1])
        self.assertEqual(a.shares[0],100)
        b.execute(a,2,np.array([200]),d.dates[2])
        self.assertEqual(a.shares[0],200)

    def test_default_broker_and_order_book_share_their_capacity_state(self):
        from qbt.orders import OrderBook
        from qbt.clocks import ClockEvent
        for book_first in (True,False):
            with self.subTest(book_first=book_first):
                d=data(3);a=qbt.Account(10000,1)
                b=qbt.Broker(qbt.build_market(d,log=None),costs=FREE,max_adv_pct=.01)
                book=OrderBook(d,b,a,qbt.OpenExecution())
                book.begin(ClockEvent(1,d.dates[1],d.dates[0],True,d.dates[1]))
                book.submit(qbt.OrderRequest(d.codes[0],100,tif='GTC'))
                if book_first:book.match('open')
                b.execute(a,1,np.array([200]),d.dates[1])
                if not book_first:book.match('open')
                self.assertEqual(a.shares[0],100)

    def test_opposite_orders_share_one_instrument_market_budget(self):
        d=data(3)
        def strategy(c):
            if c.T==1:c.order(d.codes[0],100,tif='GTC')
            if c.T==2:
                c.order(d.codes[0],100,side='sell')
                c.order(d.codes[0],100,side='buy')
        r=qbt.run_orders(d,strategy,costs=FREE,max_adv_pct=.01,capital=10000)
        for i in (1,2):
            gross=sum(f.price*f.quantity for f in r.order_fills if f.index==i)
            self.assertLessEqual(gross,1000.)
        self.assertEqual(r.account.shares[0],0)
        self.assertEqual(r.orders[-1].filled,0)

    def test_open_buy_and_close_sell_do_not_reuse_the_same_budget(self):
        d=data(2);d.amt[:]=110000.;d.high[1,0]=11.
        def strategy(c):
            c.order(d.codes[0],100,tif='GTC')
            c.order(d.codes[0],100,side='sell',limit_price=11.,tif='GTC')
        r=qbt.run_orders(d,strategy,costs=FREE,max_adv_pct=.01,capital=10000,
            asset_rules=qbt.CashAssetRules(),execution=qbt.BarTouchExecution())
        gross=sum(f.price*f.quantity for f in r.order_fills)
        self.assertLessEqual(gross,1100.)
        self.assertEqual([(f.side,f.quantity,f.phase) for f in r.order_fills],
                         [('buy',100,'open'),('sell',9,'close')])


class ExplicitComponentAudit(unittest.TestCase):
    def test_no_limit_asset_does_not_inherit_stock_close_execution_from_another_asset(self):
        d=data(3,n=2)
        d.open[1]=11.;d.close[1]=10.5;d.high[1]=11.;d.low[1]=10.5
        assets=qbt.CashAssetRules(minimum=[100,1],step=[100,1],price_limits=[True,False])
        b=qbt.Broker(qbt.build_market(d,log=None),costs=FREE,asset_rules=assets,
                     limit_rule='broken_board_close',max_adv_pct=1.)
        a=qbt.Account(10000,2)
        open_result=b.execute(a,1,np.array([100,100]),d.dates[1])
        self.assertEqual([(f[1],f[4]) for f in open_result['fills']],[(1,11.)])
        close_result=b.execute(a,1,np.array([100,100]),d.dates[1],phase='close')
        self.assertEqual([(f[1],f[4]) for f in close_result['fills']],[(0,10.5)])

    def test_broker_does_not_mutate_a_custom_components_reusable_mask(self):
        class Assets(qbt.CashAssetRules):
            def __init__(self):
                super().__init__(price_limits=True)
                self.mask=np.ones(1,bool)
            def can_buy(self,*a):return self.mask
        d=data(3);d.open[1]=11.;d.close[1]=10.5;d.high[1]=11.;d.low[1]=10.5
        assets=Assets()
        b=qbt.Broker(qbt.build_market(d,log=None),costs=FREE,asset_rules=assets,
                     limit_rule='broken_board_close',max_adv_pct=1.)
        a=qbt.Account(10000,1)
        b.execute(a,1,np.array([100]),d.dates[1])
        self.assertTrue(assets.mask[0])
        b.execute(a,1,np.array([100]),d.dates[1],phase='close')
        self.assertEqual(a.shares[0],100)

    def test_unknown_risk_mask_is_not_silently_an_exit_signal(self):
        from qbt.components import RiskControls
        for value in (np.nan,.5,-1.,'False'):
            class Risk(RiskControls):
                def exits(self,*a):return np.array([value])
            with self.subTest(value=str(value)):
                with self.assertRaises(ValueError):
                    qbt.run(data(),lambda c:c.long(np.ones(c.N,bool)),risk_factory=Risk,costs=FREE)

    def test_unknown_custom_tradability_cannot_become_permission_to_fill(self):
        class Assets(qbt.CashAssetRules):
            def can_buy(self,*a):return np.array([np.nan])
        d=data()
        def strategy(c):
            if c.T==1:c.order(d.codes[0],100)
        with self.assertRaises(ValueError):qbt.run_orders(d,strategy,asset_rules=Assets(),costs=FREE)

    def test_unknown_buying_power_is_an_error_instead_of_a_silent_empty_result(self):
        class Policy(qbt.CashAccountRules):
            def buying_power(self,a):return np.nan
        d=data()
        def strategy(c):
            if c.T==1:c.order(d.codes[0],100)
        with self.assertRaises((ValueError,AssertionError)):
            qbt.run_orders(d,strategy,account_rules=Policy(),costs=FREE)

    def test_false_metadata_and_keyword_mapping_cannot_drop_suspensions(self):
        class Metadata(qbt.MarketMetadata):
            def __bool__(self):return False
        class Mapping(dict):
            def __bool__(self):return False
        d=data();metadata=Metadata(suspended=np.ones((d.T,d.N),bool))
        market=qbt.build_market(d,metadata=metadata,log=None)
        self.assertFalse(market.can_buy(1).any())
        r=qbt.run(d,lambda c:c.long(np.ones(c.N,bool)),costs=FREE,
                  market_kw=Mapping(metadata=metadata))
        self.assertEqual(r.account.n_buy,0)

    def test_false_nonempty_stop_mapping_cannot_disable_loss_protection(self):
        class Mapping(dict):
            def __bool__(self):return False
        d=data()
        for name in ('open','close','high','low'):getattr(d,name)[2:]=8.
        r=qbt.run(d,lambda c:c.long(np.ones(c.N,bool),gross=.5,stops=Mapping(hard=.1)),costs=FREE)
        self.assertEqual(r.account.shares[0],0)

    def test_false_parameter_mapping_keeps_its_values(self):
        class Mapping(dict):
            def __bool__(self):return False
        d=data();params=Mapping(gross=.5)
        seen=[]
        def strategy(c):
            seen.append(c.p.get('gross'))
            return qbt.FLAT
        qbt.run(d,strategy,params=params,costs=FREE)
        qbt.run_orders(d,lambda c:seen.append(c.p.get('gross')),params=params,costs=FREE)
        self.assertTrue(seen)
        self.assertTrue(all(value==.5 for value in seen))

    def test_false_execution_model_is_not_replaced_by_open_execution(self):
        class Execution(qbt.OpenExecution):
            def __bool__(self):return False
            def price(self,*a):return None
        d=data()
        def strategy(c):
            if c.T==1:c.order(d.codes[0],100,tif='GTC')
        r=qbt.run_orders(d,strategy,costs=FREE,execution=Execution())
        self.assertFalse(r.order_fills)
        self.assertEqual(r.execution_name,'Execution')

    def test_supplied_false_cost_model_is_used_by_both_entries(self):
        class ZeroCost:
            def __bool__(self):return False
            def buy(self,amount,day=None):return np.zeros_like(np.asarray(amount,float))
            def sell(self,amount,day=None):return np.zeros_like(np.asarray(amount,float))
        d=data()
        def orders(c):
            if c.T==1:c.order(d.codes[0],100,tif='GTC')
        a=qbt.run(d,lambda c:c.long(np.ones(c.N,bool),gross=.5),costs=ZeroCost())
        b=qbt.run_orders(d,orders,costs=ZeroCost())
        self.assertEqual(a.fees_cum[-1],0.)
        self.assertEqual(b.fees_cum[-1],0.)

    def test_false_clock_preserves_frequency_and_schedule(self):
        class Clock(qbt.BarClock):
            def __bool__(self):return False
        d=data();clock=Clock(121,stride=2)
        a=qbt.run(d,None,clock=clock)
        b=qbt.run_orders(d,lambda c:None,clock=clock)
        for r in (a,b):
            self.assertEqual(r.periods_per_year,121)
            np.testing.assert_array_equal(r.step_indices,[1,3])

    def test_false_asset_rules_are_not_replaced_by_default_stock_rules(self):
        class Assets(qbt.CashAssetRules):
            def __bool__(self):return False
        d=data()
        def strategy(c):
            if c.T==1:c.order(d.codes[0],1,tif='GTC')
        r=qbt.run_orders(d,strategy,costs=FREE,asset_rules=Assets())
        self.assertEqual(r.orders[0].status,'FILLED')
        self.assertEqual(r.account.shares[0],1)

    def test_false_account_policy_still_limits_purchasing_power(self):
        class Policy(qbt.CashAccountRules):
            def __bool__(self):return False
            def buying_power(self,a):return min(a.cash,10.)
        d=data()
        def strategy(c):
            if c.T==1:c.order(d.codes[0],100,tif='GTC')
        r=qbt.run_orders(d,strategy,costs=FREE,asset_rules=qbt.CashAssetRules(),account_rules=Policy())
        self.assertEqual(r.order_fills[0].quantity,1)


class ListingEvidenceAudit(unittest.TestCase):
    def test_zero_and_negative_quotes_do_not_make_a_stock_preexisting(self):
        d=data(6)
        for name in ('open','close','high','low'):
            getattr(d,name)[:3,0]=[0.,-1.,np.nan]
        np.testing.assert_array_equal(d.list_age[:,0],[-1,-1,-1,0,1,2])
        c=qbt.Context(d,qbt.build_market(d,log=None))
        np.testing.assert_array_equal(c.universe(min_listed=2,not_st=False).mask[:,0],
                                     [False,False,False,False,False,True])

    def test_intraday_listing_proxy_uses_positive_quote_sessions(self):
        times=np.array(['2024-01-02T09:30','2024-01-02T09:31','2024-01-03T09:30',
                        '2024-01-03T09:31','2024-01-04T09:30','2024-01-04T09:31'],dtype='datetime64[ns]')
        p=np.array([0.,-1.,10.,10.,10.,10.])[:,None]
        d=qbt.BarDataset.from_arrays(times,['600001.SH'],open=p,close=p,high=p,low=p,
            amt=np.full_like(p,100000.),num_trades=np.ones_like(p),cumadj=np.ones_like(p))
        np.testing.assert_array_equal(d.list_age[:,0],[-1,-1,0,0,1,1])


class LedgerInputAudit(unittest.TestCase):
    def test_out_of_range_buy_index_does_not_charge_cash_before_it_fails(self):
        a=qbt.Account(10000,1)
        before=a.snapshot()
        with self.assertRaises((IndexError,ValueError)):a.buy(1,100,10.,5.,1.)
        self.assertEqual(a.snapshot(),before)
        self.assertEqual(a.shares[0],0)

    def test_fractional_and_boolean_quantities_are_not_silently_converted(self):
        for method in ('buy','sell'):
            for quantity in (1.5,True,np.bool_(True),-1):
                with self.subTest(method=method,quantity=str(quantity)):
                    a=qbt.Account(10000,1);a.buy(0,100,10.,0.,1.);a.unfreeze()
                    before=a.snapshot()
                    with self.assertRaises(ValueError):getattr(a,method)(0,quantity,10.,0.,1.)
                    self.assertEqual(a.snapshot(),before)
                    self.assertEqual(a.shares[0],100)

    def test_negative_index_cannot_alias_last_instrument(self):
        a=qbt.Account(10000,1)
        with self.assertRaises((ValueError,IndexError)):a.buy(-1,100,10.,0.,1.)
        self.assertEqual(a.cash,10000)

    def test_invalid_settlement_price_cannot_turn_into_zero_recovery(self):
        for price in (np.nan,np.inf,-1.):
            a=qbt.Account(10000,1);a.buy(0,100,10.,0.,1.)
            before=a.snapshot()
            with self.assertRaises(ValueError):a.force_close(0,price,1.)
            self.assertEqual(a.snapshot(),before)
            self.assertEqual(a.shares[0],100)

    def test_oversized_later_split_cannot_leave_earlier_dividend_half_applied(self):
        d=data(n=2);a=qbt.Account(10000,2)
        for j in (0,1):a.buy(j,100,10.,0.,1.)
        source=qbt.CorporateActions([
            qbt.CorporateAction(str(d.dates[2]),d.codes[0],cash_dividend=1.,pay_date=str(d.dates[3])),
            qbt.CorporateAction(str(d.dates[2]),d.codes[1],share_ratio=1e20)])
        book=source.prepare(d);before=a.snapshot()
        with self.assertRaises((ValueError,OverflowError)):book.apply(a,d,2)
        self.assertEqual(a.snapshot(),before)
        np.testing.assert_array_equal(a.shares,[100,100])
        np.testing.assert_array_equal(a.receivables,[0,0])


class CalendarOwnershipAudit(unittest.TestCase):
    def test_unknown_clock_flag_cannot_unfreeze_intraday_shares(self):
        from types import SimpleNamespace
        class Clock(qbt.BarClock):
            def events(self,*args,**kwargs):
                for e in super().events(*args,**kwargs):
                    yield SimpleNamespace(index=e.index,timestamp=e.timestamp,
                        previous_timestamp=e.previous_timestamp,new_session='False',session=e.session)
        source=data(3)
        d=qbt.BarDataset.from_arrays(np.array(['2024-01-02T09:30','2024-01-02T09:31','2024-01-02T09:32'],dtype='datetime64[ns]'),
            source.codes,previous_close=10.,**{name:getattr(source,name) for name in ('open','close','high','low','amt','num_trades','cumadj')})
        def strategy(c):
            if c.T==1:c.order(d.codes[0],100)
            if c.T==2:c.order(d.codes[0],100,side='sell')
        with self.assertRaisesRegex(ValueError,'new_session|会话|bool'):
            qbt.run_orders(d,strategy,clock=Clock(242*240),costs=FREE)

    def test_frozen_clock_event_does_not_follow_its_callers_flag_array(self):
        from qbt.clocks import ClockEvent
        flag=np.array(False)
        e=ClockEvent(1,np.datetime64('2024-01-03'),np.datetime64('2024-01-02'),flag)
        flag[...]=True
        self.assertFalse(e.new_session)

    def test_queries_do_not_guess_numeric_units_or_strip_timezone_offsets(self):
        from qbt.bars import BarCalendar
        calendars=[qbt.Calendar(np.array(['2024-01-02','2024-01-03'],dtype='datetime64[D]')),
                   BarCalendar(np.array(['2024-01-02T09:30','2024-01-03T09:30'],dtype='datetime64[ns]'))]
        for cal in calendars:
            with self.subTest(calendar=type(cal).__name__):
                with self.assertRaises(ValueError):cal.index(20240103)
                with self.assertRaisesRegex(ValueError,'时区'):cal.index('2024-01-03T00:00:00+08:00')
                with self.assertRaisesRegex(ValueError,'时区'):cal.slice_of(end='2024-01-03T00:00:00+08:00')

    def test_numeric_dates_require_an_explicit_epoch_unit(self):
        for cls,labels in ((qbt.Dataset,[20240102,20240103]),
                           (qbt.BarDataset,[1704153600000,1704240000000])):
            p=np.ones((2,1))
            with self.subTest(dataset=cls.__name__):
                with self.assertRaisesRegex(ValueError,'整数|单位|日期|时间'):
                    cls.from_arrays(labels,['600001.SH'],open=p,close=p,high=p,low=p,amt=p,
                                    num_trades=p,cumadj=p)

    def test_aware_datetime_and_offset_text_require_explicit_market_conversion(self):
        from datetime import datetime,timezone,timedelta
        aware=[datetime(2024,1,2,tzinfo=timezone(timedelta(hours=8))),
               datetime(2024,1,3,tzinfo=timezone(timedelta(hours=8)))]
        text=['2024-01-02T00:00:00+08:00','2024-01-03T00:00:00+08:00']
        for cls in (qbt.Dataset,qbt.BarDataset):
            for labels in (aware,text):
                with self.subTest(dataset=cls.__name__,type=type(labels[0]).__name__):
                    p=np.ones((2,1))
                    with self.assertRaisesRegex(ValueError,'时区'):
                        cls.from_arrays(labels,['600001.SH'],open=p,close=p,high=p,low=p,amt=p,
                                        num_trades=p,cumadj=p)

    def test_intraday_constructor_owns_timestamp_and_code_arrays(self):
        times=np.array(['2024-01-02T09:30','2024-01-02T09:31'],dtype='datetime64[ns]')
        codes=np.array(['600001.SH']);expected=times.copy();p=np.full((2,1),10.)
        d=qbt.BarDataset.from_arrays(times,codes,open=p,close=p,high=p,low=p,amt=p,num_trades=p,cumadj=np.ones_like(p))
        times[0]=np.datetime64('2030-01-01');codes[0]='600002.SH'
        np.testing.assert_array_equal(d.dates,expected)
        np.testing.assert_array_equal(d.codes_arr,['600001.SH'])
        np.testing.assert_array_equal(d.calendar.days,expected)

    def test_intraday_calendar_days_cannot_rewrite_event_time(self):
        from qbt.bars import BarCalendar
        calendar=BarCalendar(np.array(['2024-01-02T09:30','2024-01-02T09:31'],dtype='datetime64[ns]'))
        with self.assertRaises(ValueError):calendar.days[0]=np.datetime64('2030-01-01')

    def test_nat_query_is_not_silently_mapped_to_start_or_end(self):
        from qbt.bars import BarCalendar
        calendars=[qbt.Calendar(np.array(['2024-01-02','2024-01-03'],dtype='datetime64[D]')),
                   BarCalendar(np.array(['2024-01-02T09:30','2024-01-02T09:31'],dtype='datetime64[ns]'))]
        for cal in calendars:
            for side in ('left','right','exact'):
                with self.subTest(calendar=type(cal).__name__,side=side):
                    with self.assertRaises((ValueError,KeyError)):cal.index('NaT',side)
            with self.assertRaises(ValueError):cal.slice_of(start='NaT')
            with self.assertRaises(ValueError):cal.slice_of(end='NaT')


class TouchEvidenceAudit(unittest.TestCase):
    def test_out_of_band_bad_quote_is_not_counted_as_limit_queue_opportunity(self):
        d=data(3);d.open[1,0]=8.
        selected=np.ones((d.T,d.N),bool)
        r=qbt.run(d,qbt.Target(sel=selected,gross=.5),costs=FREE)
        self.assertEqual(r.blocked_buy_n[0],0)
        def strategy(c):
            if c.T==1:c.order(d.codes[0],100)
        orders=qbt.run_orders(d,strategy,costs=FREE)
        self.assertEqual(orders.blocked_buy_n[0],0)
        self.assertEqual(orders.orders[0].reason,'time_in_force')
        self.assertIn('outside_assumed_price_band',[e.reason for e in orders.order_events])

    def test_unknown_out_of_band_open_is_not_executable_in_a_limited_market(self):
        for opening in (8.,12.):
            d=data(3);d.open[2,0]=opening
            m=qbt.build_market(d,log=None)
            for rule in ('open_limit','sealed','broken_board_close'):
                with self.subTest(opening=opening,rule=rule):
                    self.assertFalse(m.can_buy(2,rule)[0])
                    self.assertFalse(m.can_sell(2,rule)[0])
            self.assertTrue(m.can_buy(2,'off')[0])
            self.assertTrue(m.can_sell(2,'off')[0])
            no_limit=np.ones((d.T,d.N),bool)
            official=qbt.build_market(d,metadata=qbt.MarketMetadata(no_limit=no_limit),log=None)
            self.assertTrue(official.can_buy(2)[0])
            self.assertTrue(official.can_sell(2)[0])

    def test_broken_board_close_does_not_turn_missing_close_into_a_fill(self):
        for side in ('buy','sell'):
            for close in (np.nan,0.,-1.):
                with self.subTest(side=side,close=close):
                    d=data(4);d.open[2,0]=11. if side=='buy' else 9.
                    d.high[2,0]=11.;d.low[2,0]=9.;d.close[2,0]=close
                    selected=np.ones((d.T,d.N),bool)
                    selected[0]=side=='sell';selected[1:]=side=='buy'
                    r=qbt.run(d,qbt.Target(sel=selected,gross=.5),costs=FREE,
                              limit_rule='broken_board_close',max_adv_pct=1.)
                    self.assertFalse([f for f in r.trades if f[0]==2])

    def test_broken_board_requires_a_close_inside_its_assumed_price_band(self):
        for side,opening,close in (('buy',11.,12.),('sell',9.,8.)):
            d=data(4);d.open[2,0]=opening;d.close[2,0]=close
            d.high[2,0]=max(opening,close);d.low[2,0]=min(opening,close)
            selected=np.ones((d.T,d.N),bool);selected[0]=side=='sell';selected[1:]=side=='buy'
            r=qbt.run(d,qbt.Target(sel=selected,gross=.5),costs=FREE,
                      limit_rule='broken_board_close',max_adv_pct=1.)
            self.assertFalse([f for f in r.trades if f[0]==2])

    def test_zero_or_negative_low_is_not_proof_a_buy_limit_was_touched(self):
        for low in (0.,-1.,np.nan):
            d=data(2);d.low[1,0]=low
            def strategy(c):c.order(d.codes[0],100,limit_price=9.)
            r=qbt.run_orders(d,strategy,costs=FREE,execution=qbt.BarTouchExecution())
            self.assertFalse(r.order_fills)

    def test_reversed_extremes_do_not_fabricate_a_close_phase_fill(self):
        for side in ('buy','sell'):
            d=data(3)
            if side=='buy':d.low[2,0],d.high[2,0]=9.,8.
            else:d.low[2,0],d.high[2,0]=12.,11.
            def strategy(c):
                if c.T==1 and side=='sell':c.order(d.codes[0],100,tif='GTC')
                if c.T==2:c.order(d.codes[0],100,side=side,limit_price=9.5 if side=='buy' else 11.)
            r=qbt.run_orders(d,strategy,costs=FREE,execution=qbt.BarTouchExecution())
            self.assertFalse([f for f in r.order_fills if f.index==2])


class SelectionContractAudit(unittest.TestCase):
    def test_context_cash_is_valid_in_panel_and_event_modes(self):
        d=data(n=2)
        for run in (qbt.run,qbt.run_events):
            r=run(d,lambda c:c.cash(),costs=FREE)
            self.assertEqual(r.account.n_buy,0)
            np.testing.assert_array_equal(r.equity,np.full(d.T,1e6))

    def test_missing_or_nonbinary_mask_cannot_silently_select_stocks(self):
        from qbt.universe import Universe,select_by_rank
        d=data(n=2);c=qbt.Context(d,qbt.build_market(d,log=None))
        for bad in (np.nan,np.inf,-1.,.5,'False'):
            values=np.full((d.T,d.N),bad)
            entries={
                'Universe':lambda:Universe(values),
                'long':lambda:c.long(values),
                'target':lambda:qbt.run(d,qbt.Target(sel=values),costs=FREE),
                'keep':lambda:qbt.run(d,qbt.Target(sel=np.ones((d.T,d.N),bool),keep=values),costs=FREE),
                'extra':lambda:c.universe(not_st=False,extra=values),
                'where':lambda:c.all_stocks().where(values),
                'and':lambda:c.all_stocks() & values,
                'or':lambda:c.all_stocks() | values,
                'rank':lambda:select_by_rank(np.ones_like(values,dtype=float),values,1),
                'factor_rank':lambda:c.rank(np.ones_like(values,dtype=float),mask=values),
            }
            for name,entry in entries.items():
                with self.subTest(value=str(bad),entry=name):
                    with self.assertRaises(ValueError):entry()

    def test_binary_numeric_masks_remain_supported(self):
        d=data(n=2)
        for dtype in (bool,int,float):
            mask=np.tile(np.array([1,0],dtype=dtype),(d.T,1))
            r=qbt.run(d,lambda c:c.long(mask),costs=FREE)
            self.assertGreater(r.account.shares[0],0)
            self.assertEqual(r.account.shares[1],0)


class OrderMetadataAudit(unittest.TestCase):
    def test_deadline_does_not_silently_use_a_different_clock(self):
        with self.assertRaisesRegex(ValueError,'时区'):
            qbt.OrderRequest('600001.SH',100,tif='GTD',valid_until='2024-01-03T10:05:00+08:00')
        with self.assertRaises(ValueError):
            qbt.OrderRequest('600001.SH',100,tif='GTD',valid_until=20240103)

    def test_empty_payment_date_cannot_mean_immediate_cash(self):
        d=data()
        for date in ('','2024-01-03T00:00:00+08:00'):
            with self.subTest(pay_date=date):
                source=qbt.CorporateActions([qbt.CorporateAction(str(d.dates[1]),d.codes[0],
                    cash_dividend=1.,pay_date=date)])
                with self.assertRaises(ValueError):source.prepare(d)

    def test_main_board_can_sell_the_whole_odd_residual_without_closing_every_share(self):
        d=data(3);a=qbt.Account(10000,1);a.buy(0,250,10.,0.,1.);a.unfreeze()
        b=qbt.Broker(qbt.build_market(d,log=None),costs=FREE,max_adv_pct=1.)
        b.execute(a,1,np.array([200]),d.dates[1])
        self.assertEqual(a.shares[0],200)
        self.assertEqual(a.cash,8000.)

    def test_odd_residual_order_is_accepted_but_cannot_be_claimed_twice(self):
        for repeat in (False,True):
            with self.subTest(repeat=repeat):
                d=data(5)
                for name in ('open','close','high','low'):getattr(d,name)[2:]=4.
                d.cumadj[2:]=2.5
                def strategy(c):
                    if c.T==1:c.order(d.codes[0],100)
                    if c.T==3:
                        c.order(d.codes[0],50,side='sell',tif='GTC')
                        if repeat:c.order(d.codes[0],50,side='sell',tif='GTC')
                r=qbt.run_orders(d,strategy,costs=FREE)
                self.assertEqual(r.orders[1].status,'FILLED')
                self.assertEqual(r.account.shares[0],200)
                if repeat:
                    self.assertEqual(r.orders[2].status,'CANCELED')
                    self.assertEqual(r.orders[2].reason,'quantity_no_longer_valid_for_position')

    def test_beijing_stock_allows_151_share_order_after_100_share_minimum(self):
        d=data(2);d.codes=['920001.BJ']
        def strategy(c):c.order(d.codes[0],151)
        r=qbt.run_orders(d,strategy,costs=FREE)
        self.assertEqual(r.orders[0].status,'FILLED')
        self.assertEqual(r.account.shares[0],151)

    def test_beijing_portfolio_rounding_keeps_one_share_increment(self):
        d=data(2);d.codes=['920001.BJ']
        b=qbt.Broker(qbt.build_market(d,log=None),costs=FREE)
        np.testing.assert_array_equal(b.target_qty(np.array([True]),1510.,np.array([10.])),[151])
        np.testing.assert_array_equal(b.target_qty(np.array([True]),990.,np.array([10.])),[0])

    def test_request_owns_scalar_quantity_and_limit_price(self):
        quantity=np.array(100);price=np.array(10.)
        request=qbt.OrderRequest('600001.SH',quantity,limit_price=price)
        quantity[...]=200;price[...]=20.
        self.assertEqual(request.quantity,100)
        self.assertEqual(request.limit_price,10.)

    def test_stock_limit_price_must_be_an_actual_cent_tick(self):
        d=data(2)
        def strategy(c):c.order(d.codes[0],100,limit_price=10.005)
        r=qbt.run_orders(d,strategy,costs=FREE)
        self.assertEqual(r.orders[0].status,'REJECTED')
        self.assertFalse(r.order_fills)

    def test_non_stock_asset_rules_preserve_configurable_price_precision(self):
        d=data(2)
        def strategy(c):c.order(d.codes[0],1,limit_price=10.005)
        r=qbt.run_orders(d,strategy,costs=FREE,asset_rules=qbt.CashAssetRules())
        self.assertEqual(r.orders[0].status,'FILLED')

    def test_invalid_star_minimum_cannot_be_truncated_or_bypass_minimum(self):
        d=data(2);d.codes=['688001.SH']
        for minimum in (200.5,0,-1,True):
            with self.subTest(minimum=minimum):
                with self.assertRaises(ValueError):qbt.AStockRules(star_minimum=minimum).bind(d)

    def test_order_quantity_cannot_be_boolean_or_exceed_ledger_integer_range(self):
        for quantity in (np.bool_(True),np.array(True),2**63):
            with self.subTest(quantity=str(quantity)):
                with self.assertRaises(ValueError):qbt.OrderRequest('600001.SH',quantity)


class HealthEvidenceAudit(unittest.TestCase):
    def test_reused_strategy_buffer_cannot_rewrite_the_full_run_reference(self):
        d=data(100)
        shared=np.zeros((d.T,d.N),bool)
        def strategy(c):
            shared[:c.T]=c.T < d.T  # 反例：输出依赖完整样本尾部，却复用工作数组。
            return qbt.Target(sel=shared[:c.T],gross=.5)
        report=qbt.health(d,strategy,quick=True,prefix_probes=3,costs=FREE)
        self.assertEqual(len(report.summary['前视失败点']),3)
        self.assertFalse(report.ok)


class ValuationEvidenceAudit(unittest.TestCase):
    def test_invalid_first_quote_cannot_make_an_instrument_a_preexisting_old_stock(self):
        d=data(4)
        quote=d.dates[:,None].copy().astype('datetime64[ns]');quote[0,0]=d.dates[1]
        m=qbt.build_market(d,metadata=qbt.MarketMetadata(quote_time=quote),log=None)
        c=qbt.Context(d,m)
        np.testing.assert_array_equal(c.age()[:,0],[-1,0,1,2])
        np.testing.assert_array_equal(c.universe(min_listed=2,not_st=False).mask[:,0],
                                     [False,False,False,True])

    def test_builtin_price_features_do_not_use_invalid_quote_evidence(self):
        d=data(5);d.close[2,0]=100.;d.open[2,0]=1.
        quote=d.dates[:,None].copy().astype('datetime64[ns]');quote[2,0]=d.dates[3]
        m=qbt.build_market(d,metadata=qbt.MarketMetadata(quote_time=quote),log=None)
        c=qbt.Context(d,m)
        np.testing.assert_array_equal(c.close()[:,0],np.full(d.T,10.))
        np.testing.assert_array_equal(c.open()[:,0],np.full(d.T,10.))
        np.testing.assert_array_equal(c.ret(1)[1:,0],np.zeros(d.T-1))
        self.assertTrue(np.isnan(c.close_raw()[2,0]))
        self.assertEqual(d.close[2,0],100.)  # 底层原始诊断输入仍保留。

    def test_benchmark_does_not_trade_a_future_quote_round_trip(self):
        d=data(5);d.close[2,0]=100.
        quote=d.dates[:,None].copy().astype('datetime64[ns]');quote[2,0]=d.dates[3]
        m=qbt.build_market(d,metadata=qbt.MarketMetadata(quote_time=quote),log=None)
        for run,strategy in ((qbt.run,None),(qbt.run_orders,lambda c:None)):
            with self.subTest(entry=run.__name__):
                r=run(d,strategy,market=m,costs=FREE)
                np.testing.assert_array_equal(r.bench_ret,np.zeros(d.T-1))
                np.testing.assert_array_equal(r.bench_nav,np.full(d.T,1e6))

    def test_future_bar_cannot_change_next_sessions_reference_price(self):
        for intraday in (False,True):
            for bad_close in (10.,100.):
                with self.subTest(intraday=intraday,bad_close=bad_close):
                    d=data(6)
                    row,next_row=2,3
                    if intraday:
                        times=np.array(['2024-01-02T09:30','2024-01-02T15:00','2024-01-03T09:30',
                            '2024-01-03T15:00','2024-01-04T09:30','2024-01-04T15:00'],dtype='datetime64[ns]')
                        d=qbt.BarDataset.from_arrays(times,d.codes,previous_close=10.,
                            **{name:getattr(d,name) for name in ('open','close','high','low','amt','num_trades','cumadj')})
                        row,next_row=3,4
                    d.close[row,0]=bad_close
                    quote=d.dates[:,None].copy().astype('datetime64[ns]');quote[row,0]=d.dates[next_row]
                    m=qbt.build_market(d,metadata=qbt.MarketMetadata(quote_time=quote),log=None)
                    self.assertEqual(m.reference[next_row,0],10.)
                    self.assertTrue(m.can_sell(next_row)[0])

    def test_future_sealed_quote_cannot_create_a_later_st_classification(self):
        d=data(5)
        for name in ('open','close','high','low'):getattr(d,name)[2,0]=10.5
        quote=d.dates[:,None].copy().astype('datetime64[ns]');quote[2,0]=d.dates[3]
        m=qbt.build_market(d,metadata=qbt.MarketMetadata(quote_time=quote),log=None)
        self.assertFalse(m.st[3,0])
        self.assertEqual(m.band[3,0],10)

    def test_future_quote_on_held_stock_cannot_resize_another_stock_purchase(self):
        observed=[]
        for bad_open in (10.,100.):
            d=data(4,n=2);d.open[2,0]=bad_open
            quote=np.broadcast_to(d.dates[:,None],(d.T,d.N)).copy().astype('datetime64[ns]')
            quote[2,0]=d.dates[3]
            market=qbt.build_market(d,metadata=qbt.MarketMetadata(quote_time=quote),log=None)
            selected=np.ones((d.T,d.N),bool);selected[0,1]=False
            r=qbt.run(d,qbt.Target(sel=selected,gross=.5),capital=10000,costs=FREE,
                      market=market,max_adv_pct=1.)
            observed.append([(f[1],f[3],f[4]) for f in r.trades if f[0]==2])
        self.assertEqual(*observed)
        self.assertEqual(observed[0],[(1,200,10.)])

    def test_future_quote_cannot_trigger_a_latched_stop_or_trailing_peak(self):
        for field,value,stops in (('open',1.,{'hard':.1}),('close',100.,{'trailing':.1})):
            with self.subTest(field=field):
                d=data(5);getattr(d,field)[2,0]=value
                quote=d.dates[:,None].copy().astype('datetime64[ns]');quote[2,0]=d.dates[3]
                m=qbt.build_market(d,metadata=qbt.MarketMetadata(quote_time=quote),log=None)
                r=qbt.run(d,lambda c:c.long(np.ones(c.N,bool),gross=.5,stops=stops),market=m,costs=FREE)
                self.assertEqual(r.account.n_sell,0)
                self.assertGreater(r.account.shares[0],0)


if __name__=='__main__':unittest.main(verbosity=2)
