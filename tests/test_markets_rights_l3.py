"""Economic invariants and independently specified execution examples."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import qbt
from qbt.components import RiskControls

FREE=qbt.Costs(commission=0,min_commission=0,transfer_fee=0,stamp_duty=0)


def sessions(profile='US',days=None):
    p=qbt.market_profile(profile)
    days=days or ['2024-03-08','2024-03-11','2024-03-12','2024-03-13','2024-03-14','2024-03-15']
    hours={'US':('09:30','16:00'),'CN':('09:30','15:00'),'KR':('09:00','15:30')}
    o,c=hours[p.code]
    return qbt.SessionCalendar([dict(date=d,open=d+'T'+o,close=d+'T'+c,
        **(dict(break_start=d+'T11:30',break_end=d+'T13:00') if p.code=='CN' else {})) for d in days],p,'test fixture')


def data(profile='US',prices=None,calendar=None):
    cal=calendar or sessions(profile)
    days=cal.days
    prices=np.full((len(days),1),10.) if prices is None else np.array(prices,float).reshape(-1,1)
    code={'US':'ABC','CN':'600001.SH','KR':'005930.KS'}[profile]
    return qbt.Dataset.from_arrays(days,[code],open=prices,high=prices,low=prices,close=prices,
        amt=np.full(prices.shape,1e7),num_trades=np.ones(prices.shape)*100,cumadj=np.ones(prices.shape))


def reference(ds,profile='US',calendar=None,**kw):
    cal=calendar or sessions(profile)
    records=[]
    for code in ds.codes:
        for field in ('suspended','st'):
            records.append(dict(code=code,field=field,value=False,effective_from='2000-01-01',known_at='1999-01-01'))
        if profile=='KR':
            for i,day in enumerate(ds.dates.astype('datetime64[D]')):
                for field,value in [('limit_up',float(ds.close[i,0]*1.3)),('limit_down',float(ds.close[i,0]*.7))]:
                    records.append(dict(code=code,field=field,value=value,effective_from=str(day),effective_until=str(day),known_at='1999-01-01'))
    options=dict(records=records,fees=[dict(effective_from='2020-01-01',effective_until='2030-01-01',commission=0,minimum=0)],
        securities=[dict(code=c,listing_date='2000-01-01',known_at='1999-01-01',listed_sessions_at_start=5000) for c in ds.codes])
    options.update(kw)
    return qbt.ReferenceData(profile,cal,**options)


class MarketTests(unittest.TestCase):
    def test_same_code_runs_three_markets_both_entries(self):
        for profile,quantity in [('CN',100),('US',1),('KR',1)]:
            ds=data(profile); ref=reference(ds,profile)
            def strategy(ctx):
                if ctx.T==1: ctx.order(ds.codes[0],quantity)
            r=qbt.run_orders(ds,strategy,market=profile,reference_data=ref,capital=10000,max_adv_pct=1)
            self.assertEqual(r.account.shares[0],quantity)
            self.assertEqual(r.currency,{'CN':'CNY','US':'USD','KR':'KRW'}[profile])
            r2=qbt.run(ds,qbt.Target(sel=np.ones(ds.N,bool)),market_profile=profile,reference_data=ref,capital=10000,max_adv_pct=1)
            self.assertGreater(r2.account.shares[0],0)

    def test_market_requires_real_reference_inputs(self):
        with self.assertRaises(ValueError): qbt.run(data(),None,market='US')

    def test_explicit_none_keeps_selected_market_defaults(self):
        ds=data()
        def strategy(ctx):
            if ctx.T==1: ctx.order('ABC',1)
        r=qbt.run_orders(ds,strategy,market='US',reference_data=reference(ds),capital=10,max_adv_pct=1,
            clock=None,asset_rules=None,account_rules=None,costs=None)
        self.assertEqual(r.account.shares[0],1)
        self.assertEqual(r.account.fees,0.)

    def test_custom_cost_does_not_require_unused_default_fee_schedule(self):
        ds=data()
        r=qbt.run(ds,lambda c:c.long(np.ones(c.N,bool)),market='US',reference_data=reference(ds,fees=[]),
            costs=FREE,capital=100,max_adv_pct=1)
        self.assertEqual(r.account.shares[0],10)

    def test_profile_mismatch_rejected(self):
        ds=data()
        with self.assertRaises(ValueError): qbt.run(ds,None,market='KR',reference_data=reference(ds))

    def test_dst_utc_close(self):
        c=sessions()
        self.assertEqual(c.utc_close('2024-03-08'),np.datetime64('2024-03-08T21:00','ns'))
        self.assertEqual(c.utc_close('2024-03-11'),np.datetime64('2024-03-11T20:00','ns'))

    def test_aware_inputs_convert_to_local(self):
        p=np.full((2,1),10.)
        ds=qbt.from_market_arrays('US',['2024-03-08T14:30:00Z','2024-03-11T13:30:00Z'],['ABC'],intraday=True,
             open=p,close=p,high=p,low=p,amt=p,num_trades=p,cumadj=p/10)
        self.assertEqual(ds.dates[0],np.datetime64('2024-03-08T09:30','ns'))
        self.assertEqual(ds.dates[1],np.datetime64('2024-03-11T09:30','ns'))

    def test_missing_calendar_day_fails(self):
        ds=data(); ds.dates=ds.dates[[0,1,3,4,5]]
        with self.assertRaises(ValueError): sessions().validate(ds)

    def test_cn_lunch_rejected(self):
        from types import SimpleNamespace
        with self.assertRaises(ValueError):
            sessions('CN').validate(SimpleNamespace(dates=np.array(['2024-03-08T12:00'],dtype='datetime64[ns]'),intraday=True))

    def test_us_cash_uses_t2_then_t1(self):
        p=qbt.market_profile('US')
        self.assertEqual(p.settlement_delay('2024-05-24'),2)
        self.assertEqual(p.settlement_delay('2024-05-28'),1)

    def test_kr_unsettled_sale_cannot_finance_purchase(self):
        ds=data('KR'); ref=reference(ds,'KR')
        def strat(ctx):
            if ctx.T==1: ctx.order(ds.codes[0],100)
            if ctx.T==2:
                ctx.order(ds.codes[0],100,'sell')
                ctx.order(ds.codes[0],100,tif='GTC')
        r=qbt.run_orders(ds,strat,capital=1000,market='KR',reference_data=ref,max_adv_pct=1)
        self.assertEqual([f.index for f in r.order_fills],[1,2,4])

    def test_cn_sale_proceeds_can_be_reused_same_session(self):
        ds=data('CN'); ref=reference(ds,'CN')
        def strat(ctx):
            if ctx.T==1: ctx.order(ds.codes[0],100)
            if ctx.T==2:
                ctx.order(ds.codes[0],100,'sell'); ctx.order(ds.codes[0],100)
        r=qbt.run_orders(ds,strat,capital=1000,market='CN',reference_data=ref,max_adv_pct=1)
        self.assertEqual([f.index for f in r.order_fills],[1,2,2])

    def test_kr_ticks_and_us_subdollar(self):
        kr=qbt.EquityRules('KR').bind(data('KR'))
        for price,tick in [(1999,1),(2000,5),(5000,10),(20000,50),(50000,100),(200000,500),(500000,1000)]:
            self.assertEqual(kr.tick(0,price),tick)
        us=qbt.EquityRules('US').bind(data())
        self.assertTrue(us.valid_order_price(0,.1234)); self.assertFalse(us.valid_order_price(0,1.001))

    def test_fee_schedule_cannot_silently_extend(self):
        c=qbt.ScheduledCosts([dict(effective_from='2024-01-01',effective_until='2024-12-31',sell_tax=.002)])
        self.assertEqual(float(c.sell(10000,'2024-03-01')),20.)
        with self.assertRaises(ValueError): c.sell(10000,'2025-01-01')

    def test_separate_bank_calendar_skips_bank_only_holiday(self):
        c=sessions()
        c.settlement_days=np.array(['2024-03-08','2024-03-12','2024-03-13'],dtype='datetime64[D]')
        self.assertEqual(c.due('2024-03-08',1),np.datetime64('2024-03-12'))

    def test_st_band_changes_on_2026_reform_date(self):
        c=sessions('CN',['2026-07-03','2026-07-06','2026-07-07'])
        ds=data('CN',calendar=c); ref=reference(ds,'CN',calendar=c)
        ref.records+=(dict(code=ds.codes[0],field='st',value=True,effective_from='2020-01-01',known_at='2019-01-01'),)
        _,m,_,_=ref.prepare(ds)
        np.testing.assert_array_equal(m.band[:,0],[5,10,10])

    def test_per_share_fees_have_execution_quantity(self):
        ds=data()
        rows=[dict(effective_from='2020-01-01',effective_until='2030-01-01',sell_per_share=.1,sell_per_share_cap=3.)]
        def s(ctx):
            if ctx.T==1: ctx.order('ABC',100)
            if ctx.T==2: ctx.order('ABC',100,'sell')
        r=qbt.run_orders(ds,s,market='US',reference_data=reference(ds,fees=rows),capital=1000,max_adv_pct=1)
        self.assertEqual(r.order_fills[-1].fee,3.)

    def test_health_uses_market_reference_and_costs(self):
        ds=data()
        h=qbt.health(ds,lambda c:c.long(np.ones(c.N,bool)),market='US',reference_data=reference(ds),quick=True)
        self.assertTrue(h.ok,str(h))

    def test_intraday_state_change_uses_publication_instant(self):
        times=np.array(['2024-03-08T09:30','2024-03-08T09:31','2024-03-08T09:32'],dtype='datetime64[ns]')
        a=np.full((3,1),10.)
        ds=qbt.BarDataset.from_arrays(times,['ABC'],open=a,high=a,low=a,close=a,amt=a,num_trades=a,cumadj=a/10)
        ref=reference(ds,periods_per_year=252*390)
        ref.records+=(dict(code='ABC',field='suspended',value=True,effective_from='2024-03-08T09:31',known_at='2024-03-08T09:31'),)
        _,m,_,_=ref.prepare(ds)
        np.testing.assert_array_equal(m.quote_suspended[:,0],[False,True,True])

    def test_market_csv_preserves_currency_units_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bars.csv'
            head='timestamp,code,currency,open,high,low,close,amount\n'
            first='2024-03-08,ABC,USD,10,10,10,10,1000\n'
            second='2024-03-11,ABC,USD,11,11,11,11,2000\n'
            p.write_text(head+first+second,encoding='utf-8')
            ds=qbt.load_market_csv(p,'US')
            self.assertEqual(ds.amt[1,0],2000)
            self.assertTrue(np.isnan(ds.num_trades).all())
            p.write_text(head+first+first+second,encoding='utf-8')
            with self.assertRaises(ValueError): qbt.load_market_csv(p,'US')
            p.write_text((head+first+second).replace('USD','CNY'),encoding='utf-8')
            with self.assertRaises(ValueError): qbt.load_market_csv(p,'US')

    def test_kr_missing_official_bounds_rejected(self):
        ds=data('KR'); ref=reference(ds,'KR')
        ref.records=tuple(r for r in ref.records if r['field'] not in ('limit_up','limit_down'))
        with self.assertRaises(ValueError): ref.prepare(ds)

    def test_invalid_strict_flag_not_silently_coerced(self):
        with self.assertRaises(ValueError): reference(data(),strict='False')

    def test_reference_future_revision_does_not_rewrite_history(self):
        ds=data('CN'); ref=reference(ds,'CN')
        ref.records+= (dict(code=ds.codes[0],field='st',value=True,effective_from='2024-03-08',known_at='2024-03-12T10:00'),)
        _,m,_,_=ref.prepare(ds)
        self.assertFalse(m.st[2,0]); self.assertTrue(m.st[3,0])

    def test_missing_reference_state_fails_strict(self):
        ds=data()
        with self.assertRaises(ValueError): reference(ds,records=[]).prepare(ds)


class RightsTests(unittest.TestCase):
    def test_net_dividend_paydate_risk_and_signal_conservation(self):
        ds=data(prices=[10,10,9,9,9,9])
        ev=qbt.CorporateAction('2024-03-12','ABC',cash_dividend=1,pay_date='2024-03-14',
            known_at='2024-03-08T09:00',record_date='2024-03-13',withholding_rate=.1)
        ref=reference(ds,actions=[ev])
        r=qbt.run(ds,qbt.Target(sel=np.ones(1,bool),stops={'hard':.05}),market='US',reference_data=ref,
                  capital=1000,max_adv_pct=1,rebalance_tol=1.)
        self.assertEqual(r.account.shares[0],100)
        np.testing.assert_allclose(r.equity,[1000,1000,990,990,990,990])
        self.assertEqual(r.account.taxes,10.)
        self.assertEqual(r.account.cash,90.)
        prepared,m,_,_=ref.prepare(ds)
        self.assertAlmostEqual(qbt.Context(prepared,m).ret(1)[2,0],-.01)
        self.assertEqual(prepared.cumadj_ff[2,0],1.)
        np.testing.assert_array_equal(ds.cumadj_ff,np.ones((6,1)))

    def test_split_price_basis_has_no_false_stop(self):
        ds=data(prices=[10,10,5,5,5,5])
        ev=qbt.CorporateAction('2024-03-12','ABC',share_ratio=2,share_date='2024-03-12',known_at='2024-03-08')
        ref=reference(ds,actions=[ev])
        r=qbt.run(ds,qbt.Target(sel=np.ones(1,bool),stops={'hard':.01,'trailing':.01}),market='US',reference_data=ref,
                  capital=1000,max_adv_pct=1,rebalance_tol=1.)
        np.testing.assert_allclose(r.equity,1000.)
        self.assertEqual(r.account.shares[0],200)

    def test_factor_disagreement_detected(self):
        ds=data()
        ev=qbt.CorporateAction('2024-03-12','ABC',share_ratio=2,share_date='2024-03-12',known_at='2024-03-08')
        with self.assertRaises(ValueError): reference(ds,actions=[ev],factor_mode='validate_split').prepare(ds)

    def test_record_close_entitlement_survives_sale(self):
        ds=data()
        ev=qbt.CorporateAction('2024-03-13','ABC',cash_dividend=1,pay_date='2024-03-14',record_date='2024-03-11',
            known_at='2024-03-08',entitlement_basis='record_close',withholding_rate=0.)
        def strat(ctx):
            if ctx.T==1: ctx.order('ABC',100)
            if ctx.T==2: ctx.order('ABC',100,'sell')
        r=qbt.run_orders(ds,strat,capital=1000,market='US',reference_data=reference(ds,actions=[ev]),max_adv_pct=1)
        self.assertEqual(r.account.cash,1100.)
        self.assertEqual(r.account.shares[0],0)

    def test_terminal_claim_does_not_become_immediate_cash(self):
        ds=data()
        ev=qbt.Settlement('2024-03-12','ABC',8.,pay_date='2024-03-14',known_at='2024-03-08',withholding_rate=.1)
        def strat(ctx):
            if ctx.T==1: ctx.order('ABC',100)
        r=qbt.run_orders(ds,strat,capital=1000,market='US',reference_data=reference(ds,settlements=[ev]),max_adv_pct=1)
        self.assertEqual(r.cash[2],0.)
        self.assertEqual(r.receivables[2],720.)
        self.assertEqual(r.cash[4],720.)
        self.assertEqual(r.account.taxes,80.)
        self.assertAlmostEqual(r.bench_ret[1],-.28)
        self.assertTrue(r.market.quote_suspended[2:,0].all())

    def test_late_rights_are_not_accepted(self):
        ds=data()
        ev=qbt.CorporateAction('2024-03-12','ABC',share_ratio=2,known_at='2024-03-12T10:00')
        with self.assertRaises(ValueError): reference(ds,actions=[ev]).prepare(ds)

    def test_unexplained_source_factor_cannot_be_erased(self):
        ds=data(); ds.cumadj[2:]=2.
        with self.assertRaises(ValueError): reference(ds).prepare(ds)

    def test_later_tax_assessment_has_liability_without_negative_cash(self):
        ds=data()
        tax=qbt.TaxAssessment('2024-03-12','ABC',50.,known_at='2024-03-08',reason='actual broker deferred dividend tax')
        r=qbt.run(ds,lambda c:c.long(np.ones(c.N,bool)),market='US',reference_data=reference(ds,taxes=[tax]),
            capital=1000,max_adv_pct=1,rebalance_tol=1.)
        self.assertEqual(r.account.cash,0.)
        self.assertEqual(r.account.tax_payables[0],50.)
        self.assertEqual(r.equity[-1],950.)
        self.assertEqual(r.account.taxes,50.)

    def test_dividend_dilution_after_adding_shares_is_not_market_drawdown(self):
        a=qbt.Account(10000,1); risk=RiskControls(1,{'trailing':.01})
        a.buy(0,100,10,0,1); a.income_adj[0]=1.
        risk.mark_close(a,np.array([9.]))
        a.buy(0,100,9,0,1)
        self.assertFalse(risk.exits(a,np.array([9.]))[0])

    def test_future_quote_cannot_pollute_total_return_signal(self):
        ds=data(prices=[10,10,100,10,10,10]); qt=np.broadcast_to(ds.dates[:,None],(6,1)).copy()
        qt[2,0]=ds.dates[3]
        prepared,m,_,_=reference(ds,quote_time=qt).prepare(ds)
        self.assertEqual(qbt.Context(prepared,m).ret(1)[2,0],0.)
        self.assertTrue(m.quote_suspended[2,0])

    def test_explicit_quote_time_keeps_dividends_in_benchmark(self):
        ds=data(prices=[10,10,9,9,9,9]); qt=np.broadcast_to(ds.dates[:,None],(6,1)).copy()
        ev=qbt.CorporateAction('2024-03-12','ABC',cash_dividend=1,pay_date='2024-03-14',known_at='2024-03-08',withholding_rate=0.)
        r=qbt.run(ds,None,market='US',reference_data=reference(ds,actions=[ev],quote_time=qt))
        np.testing.assert_allclose(r.bench_ret,0.,atol=1e-12)

    def test_missing_dividend_day_price_forward_fill_does_not_double_count(self):
        ds=data(prices=[10,10,100,9,9,9]); qt=np.broadcast_to(ds.dates[:,None],(6,1)).copy();qt[2,0]=ds.dates[3]
        ev=qbt.CorporateAction('2024-03-12','ABC',cash_dividend=1,pay_date='2024-03-14',known_at='2024-03-08',withholding_rate=0.)
        prepared,m,_,_=reference(ds,actions=[ev],quote_time=qt).prepare(ds)
        self.assertEqual(qbt.Context(prepared,m).close()[2,0],9.)

    def test_unknown_dividend_tax_and_share_delivery_cannot_be_assumed(self):
        ds=data()
        cash=qbt.CorporateAction('2024-03-12','ABC',cash_dividend=1,pay_date='2024-03-14',known_at='2024-03-08')
        shares=qbt.CorporateAction('2024-03-12','ABC',share_ratio=2,known_at='2024-03-08')
        for ev in (cash,shares):
            with self.assertRaises(ValueError): reference(ds,actions=[ev]).prepare(ds)


class FXTests(unittest.TestCase):
    def test_future_and_late_published_fx_not_used(self):
        f=qbt.FXRates('CNY',[qbt.FXQuote('USD','2024-03-08T20:00Z','2024-03-08T20:01Z',7.1),
             qbt.FXQuote('USD','2024-03-08T20:00Z','2024-03-08T22:00Z',8.)])
        self.assertEqual(f.at('USD',np.datetime64('2024-03-08T21:00')),7.1)
        with self.assertRaises(ValueError): f.at('USD',np.datetime64('2024-03-08T19:00'))

    def test_fx_staleness(self):
        f=qbt.FXRates('CNY',[qbt.FXQuote('USD','2024-03-08T20:00Z','2024-03-08T20:01Z',7.1)],np.timedelta64(1,'h'))
        with self.assertRaises(ValueError): f.at('USD',np.datetime64('2024-03-09T20:00'))


def l3(kind,oid,qty,seq,second,side='buy',price=10.):
    return qbt.L3Event(f'2024-03-08T09:30:{second:02}',seq,kind,'ABC',oid,side,price,qty)


class L3Tests(unittest.TestCase):
    def run_replay(self,events,strategy,**kw):
        return qbt.run_l3(events,strategy,profile='US',calendar=sessions(),codes=['ABC'],capital=10000,costs=FREE,**kw)

    def test_queue_ahead_must_be_consumed_before_own_fill(self):
        feed=[l3('add','ahead',100,1,0),l3('add','behind',100,2,1),
              l3('execute','ahead',100,3,2),l3('execute','behind',40,4,3)]
        def strat(ctx):
            if str(ctx.now).endswith('00.000000000'): ctx.order(qbt.OrderRequest('ABC',100,limit_price=10.))
        r=self.run_replay(feed,strat)
        self.assertEqual([f['quantity'] for f in r.fills],[40])
        self.assertIn('4/behind',r.fills[0]['evidence'])

    def test_two_own_orders_share_trade_volume(self):
        feed=[l3('add','ahead',100,1,0),l3('add','behind',100,2,1),
              l3('cancel','ahead',100,3,2),l3('execute','behind',60,4,3)]
        def strat(ctx):
            if len(ctx.book)==1 and 'ahead' in ctx.book:
                ctx.order(qbt.OrderRequest('ABC',40,limit_price=10.));ctx.order(qbt.OrderRequest('ABC',40,limit_price=10.))
        r=self.run_replay(feed,strat)
        self.assertEqual([f['quantity'] for f in r.fills],[40,20])

    def test_latency_loses_queue_priority(self):
        feed=[l3('add','ahead',100,1,0),l3('add','behind',100,2,1),
              l3('cancel','ahead',100,3,2),l3('execute','behind',60,4,3)]
        called=[]
        def strat(ctx):
            if not called: called.append(ctx.order(qbt.OrderRequest('ABC',100,limit_price=10.)))
        r=self.run_replay(feed,strat,latency=np.timedelta64(2,'s'))
        self.assertEqual(len(r.fills),0)

    def test_sequence_gap_aborts(self):
        with self.assertRaises(ValueError): self.run_replay([l3('add','a',10,1,0),l3('cancel','a',10,3,1)],lambda ctx:None)

    def test_non_fifo_public_execution_aborts(self):
        with self.assertRaises(ValueError): self.run_replay([l3('add','a',10,1,0),l3('add','b',10,2,1),l3('execute','b',10,3,2)],lambda ctx:None)

    def test_aggressive_orders_cannot_reuse_displayed_liquidity(self):
        def strat(ctx):
            ctx.order(qbt.OrderRequest('ABC',100)); ctx.order(qbt.OrderRequest('ABC',100))
        r=self.run_replay([l3('add','ask',150,1,0,side='sell')],strat)
        self.assertEqual(sum(f['quantity'] for f in r.fills),150)

    def test_snapshot_warmup_does_not_expose_partial_depth_to_strategy(self):
        seen=[]
        self.run_replay([l3('add','a',10,1,0),l3('add','b',10,2,0),l3('execute','a',5,3,1)],
                        lambda ctx:seen.append(len(ctx.book)),warmup_events=2)
        self.assertEqual(seen,[2])

    def test_ended_evidence_does_not_claim_exchange_expiry(self):
        called=[]
        def s(ctx):
            if not called: called.append(ctx.order(qbt.OrderRequest('ABC',10,limit_price=10.,tif='GTC')))
        r=self.run_replay([l3('add','a',10,1,0)],s)
        self.assertEqual(r.orders[0]['status'],'EVIDENCE_ENDED')

    def test_all_exchange_messages_win_zero_latency_timestamp_tie(self):
        called=[]
        def s(ctx):
            if not called: called.append(ctx.order(qbt.OrderRequest('ABC',10,limit_price=10.)))
        r=self.run_replay([l3('add','a',10,1,0),l3('add','b',10,2,0),
                           l3('cancel','a',10,3,1),l3('execute','b',10,4,2)],s)
        # b was already at the exchange at arrival, so its execution is ahead.
        self.assertEqual(len(r.fills),0)

    def test_cancel_arrival_after_trade_does_not_erase_fill(self):
        feed=[l3('add','a',10,1,0),l3('cancel','a',10,2,1),l3('add','b',10,3,2),l3('execute','b',10,4,3)]
        state=[]
        def strat(ctx):
            if not state: state.append(ctx.order(qbt.OrderRequest('ABC',10,limit_price=10.)))
            if str(ctx.now).startswith('2024-03-08T09:30:02'): ctx.cancel(state[0])
        r=self.run_replay(feed,strat,latency=np.timedelta64(1,'s'))
        # Exchange execution wins the same-timestamp cancel-arrival tie.
        self.assertEqual(sum(f['quantity'] for f in r.fills),10)


if __name__=='__main__': unittest.main()
