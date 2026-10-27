"""行为回归：使用独立的小市场验证撮合、账务、时序和可替换组件。"""
from __future__ import annotations

import os
import pickle
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qbt
from qbt._health import _execution_signature, _same_signature
from qbt._report import to_html


class MemoryData(SimpleNamespace):
    def cached(self, key, builder):
        if key not in self.cache:
            self.cache[key] = builder(self)
        return self.cache[key]


def sample(T=100, N=2):
    dates = np.busday_offset(np.datetime64("2024-01-02"), np.arange(T))
    close = np.full((T, N), 10.)
    return MemoryData(T=T, N=N, dates=dates, codes=[f"60000{j+1}.SH" for j in range(N)],
        calendar=qbt.Calendar(dates), close=close, close_ff=close, close_adj=close,
        open=close.copy(), open_ff=close.copy(), high=close.copy(), low=close.copy(),
        amt=np.full((T, N), 1e8), cumadj_ff=np.ones((T, N)), list_age=np.full((T, N), 200), cache={})


FREE = qbt.Costs(commission=0, min_commission=0, transfer_fee=0, stamp_duty=0)


def run(d, strategy, **kw):
    return qbt.run(d, strategy, **dict(costs=FREE, max_adv_pct=1., rebalance_tol=1., **kw))


def long_all(ctx):
    return ctx.long(np.ones(ctx.N, bool), gross=.5)


class RegressionTests(unittest.TestCase):
    def test_open_classification_ignores_current_close_and_extremes(self):
        states = []
        for sealed in (True, False):
            d = sample()
            d.open[80, 0] = 10.5
            d.close[80, 0] = 10.5 if sealed else 10.4
            d.high[80, 0] = 10.5 if sealed else 10.6
            d.low[80, 0] = 10.1
            m = qbt.build_market(d, log=None)
            states.append((bool(m.st[80, 0]), bool(m.can_buy(80)[0])))
        self.assertEqual(states, [(False, True), (False, True)])

    def test_current_wild_return_does_not_remove_open_limits(self):
        d = sample()
        d.open[80, 0] = 11.
        d.close[80, 0] = 15.
        m = qbt.build_market(d, log=None)
        self.assertFalse(m.no_limit[80, 0])
        self.assertFalse(m.can_buy(80)[0])

    def test_explicit_metadata_and_cache_are_separate(self):
        d = sample()
        plain = qbt.build_market(d, log=None)
        st = np.zeros((d.T, d.N), bool)
        st[80, 0] = True
        official = qbt.build_market(d, metadata=qbt.MarketMetadata(st=st), log=None)
        self.assertEqual(official.band[80, 0], 5)
        self.assertEqual(plain.band[80, 0], 10)
        band = np.full((d.T, d.N), 20, np.int8)
        self.assertEqual(qbt.build_market(d, band_override=band, log=None).band[80, 0], 20)
        self.assertIs(qbt.build_market(d, log=None), plain)

    def test_metadata_shape_is_validated(self):
        with self.assertRaises(ValueError):
            qbt.build_market(sample(), metadata=qbt.MarketMetadata(st=np.zeros((2, 2), bool)), log=None)

    def test_explicit_no_limit_allows_ipo_without_previous_quote(self):
        d = sample()
        no_limit = np.zeros((d.T, d.N), bool)
        no_limit[0] = True
        m = qbt.build_market(d, metadata=qbt.MarketMetadata(no_limit=no_limit), log=None)
        self.assertTrue(m.can_buy(0).all())

    def test_sealed_requires_all_day_seal(self):
        d = sample()
        d.open[5, 0] = d.close[5, 0] = d.high[5, 0] = 11.
        d.low[5, 0] = 10.5
        m = qbt.build_market(d, log=None)
        self.assertTrue(m.can_buy(5, "sealed")[0])
        self.assertFalse(m.can_buy(5, "open_limit")[0])

    def test_buy_capacity_keeps_main_board_lots(self):
        d = sample(N=1)
        d.amt[:] = 55000.
        m = qbt.build_market(d, log=None)
        a = qbt.Account(10000., 1)
        qbt.Broker(m, costs=FREE).execute(a, 1, np.array([200]), d.dates[1])
        self.assertEqual(a.shares[0], 100)

    def test_sell_capacity_symmetric_and_odd_residual_exit(self):
        d = sample(N=1)
        d.amt[:] = 55000.
        b = qbt.Broker(qbt.build_market(d, log=None), costs=FREE)
        a = qbt.Account(10000., 1)
        a.buy(0, 175, 10., 0., 1.)
        a.unfreeze()
        b.execute(a, 1, np.array([0]), d.dates[1])
        self.assertEqual(a.shares[0], 75)
        b.execute(a, 2, np.array([0]), d.dates[2])
        self.assertEqual(a.shares[0], 0)

    def test_star_minimum_and_one_share_increment(self):
        d = sample(N=1)
        d.codes = ["688001.SH"]
        b = qbt.Broker(qbt.build_market(d, log=None), costs=FREE, max_adv_pct=1.)
        self.assertEqual(b.target_qty(np.array([True]), 2010., np.array([10.]))[0], 201)
        self.assertEqual(b.target_qty(np.array([True]), 1990., np.array([10.]))[0], 0)
        a = qbt.Account(2005., 1)
        b.execute(a, 1, np.array([201]), d.dates[1])
        self.assertTrue(a.shares[0] == 0 or a.shares[0] >= 200)
        self.assertGreaterEqual(a.cash, 0.)

    def test_capacity_uses_past_adv_for_budget(self):
        results = []
        for future_amt in (1e8, 1e10):
            d = sample(N=1)
            d.amt[:] = 1e5
            d.amt[1:] = future_amt
            b = qbt.Broker(qbt.build_market(d, log=None), costs=FREE)
            a = qbt.Account(1e6, 1)
            b.execute(a, 1, np.array([10000]), d.dates[1])
            results.append(int(a.shares[0]))
        self.assertEqual(results, [200, 200])

    def test_no_volume_cannot_fill(self):
        d = sample(N=1)
        d.amt[1] = 0
        r = run(d, long_all, end=d.dates[1])
        self.assertEqual(r.account.n_buy, 0)

    def test_same_day_buys_cannot_be_sold(self):
        d = sample(N=1)
        b = qbt.Broker(qbt.build_market(d, log=None), costs=FREE)
        a = qbt.Account(10000., 1)
        b.execute(a, 1, np.array([100]), d.dates[1])
        b.execute(a, 1, np.array([0]), d.dates[1])
        self.assertEqual(a.shares[0], 100)
        a.unfreeze()
        b.execute(a, 2, np.array([0]), d.dates[2])
        self.assertEqual(a.shares[0], 0)

    def test_stale_position_never_creates_cash(self):
        d = sample(T=8, N=1)
        d.amt[2:] = 0
        r = run(d, long_all, max_stale_days=2)
        self.assertEqual(r.account.shares[0], 50000)
        self.assertEqual(r.account.cash, 500000.)
        self.assertEqual(r.account.n_force_close, 0)
        self.assertEqual(r.equity[-1], r.account.cash)
        self.assertEqual(r.stale_value[-1], 500000.)

    def test_stale_recovery_restores_valuation_without_cash(self):
        d = sample(T=8, N=1)
        d.amt[2:7] = 0
        r = run(d, long_all, max_stale_days=2)
        self.assertEqual(r.equity[-2], 500000.)
        self.assertEqual(r.equity[-1], 1000000.)
        self.assertEqual(r.account.cash, 500000.)

    def dividend_data(self):
        d = sample(T=8, N=1)
        d.open[2:] = d.high[2:] = d.low[2:] = d.close[2:] = 9.
        d.open_ff = d.open
        d.cumadj_ff[2:] = 10. / 9.
        d.close_adj = d.close * d.cumadj_ff
        actions = qbt.CorporateActions([qbt.CorporateAction(str(d.dates[2]), d.codes[0],
            cash_dividend=1., pay_date=str(d.dates[4]))])
        return d, actions

    def test_dividend_is_receivable_until_actual_pay_date(self):
        d, actions = self.dividend_data()
        r = run(d, long_all, corporate_actions=actions, end=d.dates[3], capital=10000.)
        self.assertEqual(r.account.shares[0], 500)
        self.assertEqual(r.account.cash, 5000.)
        self.assertEqual(r.account.receivables[0], 500.)
        self.assertEqual(r.equity[-1], 10000.)

    def test_dividend_payment_and_repeat_run_are_independent(self):
        d, actions = self.dividend_data()
        a = run(d, long_all, corporate_actions=actions, capital=10000.)
        b = run(d, long_all, corporate_actions=actions, capital=10000.)
        self.assertEqual(a.account.cash, 5500.)
        self.assertEqual(a.account.receivables[0], 0.)
        np.testing.assert_array_equal(a.equity, b.equity)

    def test_split_adjusts_shares_and_waits_for_listing(self):
        d = sample(T=6, N=1)
        d.open[2:] = d.close[2:] = d.high[2:] = d.low[2:] = 5.
        d.open_ff = d.open
        d.cumadj_ff[2:] = 2.
        d.close_adj = d.close * d.cumadj_ff
        actions = qbt.CorporateActions([qbt.CorporateAction(str(d.dates[2]), d.codes[0],
            share_ratio=2., share_date=str(d.dates[4]))])
        r = run(d, long_all, corporate_actions=actions, end=d.dates[3], capital=10000.)
        self.assertEqual(r.account.shares[0], 1000)
        self.assertEqual(r.account.restricted[0], 500)
        self.assertEqual(r.account.cash, 5000.)
        self.assertEqual(r.equity[-1], 10000.)
        r = run(d, long_all, corporate_actions=actions, capital=10000.)
        self.assertEqual(r.account.restricted[0], 0)

    def test_explicit_mode_rejects_missing_action_coverage(self):
        d, _ = self.dividend_data()
        with self.assertRaises(ValueError):
            run(d, long_all, corporate_actions=qbt.CorporateActions())

    def test_cash_dividend_requires_payment_date(self):
        d = sample()
        with self.assertRaises(ValueError):
            qbt.CorporateActions([qbt.CorporateAction(str(d.dates[2]), d.codes[0], cash_dividend=1.)]).prepare(d)

    def test_known_zero_settlement_writes_off_without_reentry(self):
        d = sample(T=6, N=1)
        actions = qbt.CorporateActions(settlements=[qbt.Settlement(str(d.dates[3]), d.codes[0], 0.)])
        r = run(d, long_all, corporate_actions=actions, capital=10000.)
        self.assertEqual(r.account.cash, 5000.)
        self.assertEqual(r.account.shares[0], 0)
        self.assertEqual(r.account.n_buy, 1)

    def test_factories_and_allocation_are_used(self):
        d = sample(T=5)
        calls = []
        class A(qbt.Account):
            def __init__(self, *args):
                calls.append("account")
                super().__init__(*args)
        class B(qbt.Broker):
            def __init__(self, *args, **kwargs):
                calls.append("broker")
                super().__init__(*args, **kwargs)
        class R(qbt.RiskControls):
            def exits(self, account, adj_open):
                calls.append("risk")
                return super().exits(account, adj_open)
        def allocation(members, **kw):
            calls.append("allocation")
            return np.array([0., 1.])
        r = run(d, long_all, account_factory=A, broker_factory=B, risk_factory=R, allocation=allocation)
        self.assertTrue(set(["account", "broker", "risk", "allocation"]) <= set(calls))
        self.assertEqual(r.account.shares[0], 0)
        self.assertGreater(r.account.shares[1], 0)

    def test_take_profit_is_separate_risk_component(self):
        d = sample(T=5, N=1)
        d.open[2:] = d.close[2:] = 12.
        d.open_ff = d.open
        def strategy(ctx):
            return ctx.long(np.array([True]), gross=.5, stops={"take_profit": .1})
        r = run(d, strategy, limit_rule="off")
        self.assertEqual(r.account.shares[0], 0)
        self.assertEqual(r.account.n_sell, 1)

    def test_hold_sentinel_and_none_error(self):
        d = sample(T=5)
        self.assertIs(pickle.loads(pickle.dumps(qbt.HOLD)), qbt.HOLD)
        self.assertEqual(run(d, lambda ctx: qbt.HOLD).account.n_buy, 0)
        with self.assertRaises(TypeError):
            run(d, lambda ctx: None)

    def test_event_hold_and_flat(self):
        d = sample(T=6, N=1)
        visible = []
        def strategy(ctx):
            visible.append(ctx.T)
            if ctx.T == 1:
                return ctx.long(np.array([True]), gross=.5)
            return qbt.FLAT if ctx.T >= 4 else qbt.HOLD
        r = run(d, strategy, mode="events")
        self.assertEqual(visible, [1, 2, 3, 4, 5])
        self.assertEqual(r.account.n_buy, 1)
        self.assertEqual(r.account.n_sell, 1)
        self.assertEqual(r.account.shares[0], 0)

    def test_prefix_check_catches_future_weights(self):
        d = sample()
        d.close[:, 0] = np.linspace(10., 12., d.T)
        def cheating(ctx):
            return qbt.Target(sel=np.ones((ctx.T, ctx.N), bool), weight="score",
                score=np.tile(ctx.close_raw()[-1], (ctx.T, 1)), gross=.5)
        h = qbt.health(d, cheating, quick=True, prefix_probes=4)
        self.assertTrue(h.summary["前视失败点"])
        self.assertFalse(h.ok)

    def test_prefix_check_catches_keep_weight_mode_and_risk(self):
        d = sample()
        for field in ("keep", "weight", "stops"):
            def strategy(ctx):
                a = np.ones((ctx.T, ctx.N), bool)
                kw = dict(sel=a, keep=a.copy(), gross=.5)
                if field == "keep":
                    kw["keep"][:, 0] = ctx.T == d.T
                elif field == "weight":
                    kw.update(weight="score" if ctx.T == d.T else "equal", score=np.ones(a.shape))
                else:
                    kw["stops"] = {"hard": ctx.T / 1000.}
                return qbt.Target(**kw)
            h = qbt.health(d, strategy, quick=True, prefix_probes=3)
            self.assertTrue(h.summary["前视失败点"], field)

    def test_valid_strategy_and_flat_pass_health(self):
        d = sample()
        for strategy in (long_all, lambda ctx: qbt.FLAT):
            self.assertTrue(qbt.health(d, strategy, quick=True).ok)

    def test_unknown_health_option_is_not_silently_ignored(self):
        with self.assertRaises(TypeError):
            qbt.health(sample(), long_all, quick=True, misspelled_option=1)

    def test_close_sale_proceeds_cannot_finance_open_buy(self):
        d = sample(T=4)
        d.open[2, 0] = 9.
        d.close[2, 0] = 9.5
        d.high[2, 0], d.low[2, 0] = 10., 9.
        d.open_ff = d.open
        def strategy(ctx):
            return ctx.long(np.array([True, False]) if ctx.T == 1 else np.array([False, True]), gross=1.)
        r = run(d, strategy, mode="events", limit_rule="broken_board_close", end=d.dates[2], capital=10000.)
        self.assertEqual(r.account.shares[1], 0)
        self.assertEqual(r.account.cash, 9500.)

    def test_report_discloses_assumptions_and_receivables(self):
        d, events = self.dividend_data()
        r = run(d, long_all, corporate_actions=events)
        html = to_html(r, plotly=False)
        self.assertIn("假设与数据口径", html)
        self.assertIn("应收分红", html)
        self.assertIn("<svg", html)

    def test_single_day_interval_and_weekend_start(self):
        d = sample(T=10)
        r = run(d, long_all, start=d.dates[3], end=d.dates[3])
        self.assertEqual(r.n, 2)
        saturday = np.datetime64("2024-01-06")
        r = run(d, long_all, start=saturday, end="2024-01-09")
        self.assertEqual(d.dates[r.window[0]], np.datetime64("2024-01-08"))

    def test_wait_strategy_keeps_unfilled_buy_until_available(self):
        d = sample(T=6, N=1)
        d.close[1, 0], d.high[1, 0] = 11., 11.
        d.open[2, 0], d.close[2, 0], d.high[2, 0], d.low[2, 0] = 12.1, 12., 12.1, 11.9
        d.open[3:, 0] = d.close[3:, 0] = d.high[3:, 0] = d.low[3:, 0] = 12.
        d.open_ff = d.open
        strategy = qbt.presets.limit_wait(k=1, min_amt=0, min_listed=0, holding_days=1)
        r = run(d, strategy, mode="events", start=d.dates[2])
        self.assertEqual(r.trades[0][0], 3)
        self.assertEqual(r.trades[1][0], 4)
        self.assertEqual(r.account.n_buy, 1)
        self.assertEqual(r.account.n_sell, 1)

    def test_array_dataset_alignment_and_pickle(self):
        d = sample(T=5)
        arrays = dict(open=d.open, high=d.high, low=d.low, close=d.close,
                      amt=d.amt, num_trades=np.full((d.T,d.N), 100.), cumadj=d.cumadj_ff)
        data = qbt.Dataset.from_arrays(d.dates, d.codes, **arrays)
        restored = pickle.loads(pickle.dumps(data))
        np.testing.assert_array_equal(restored.close, d.close)
        self.assertEqual(run(data, long_all).account.n_buy, 2)

    def test_dividend_sell_pnl_reconciles_to_equity(self):
        d, actions = self.dividend_data()
        def strategy(ctx):
            return ctx.long(np.array([True]), gross=.5) if ctx.T == 1 else qbt.FLAT
        r = run(d, strategy, corporate_actions=actions, mode="events", capital=10000.)
        self.assertEqual(r.account.shares[0], 0)
        self.assertEqual(r.account.cash, 10000.)
        self.assertAlmostEqual(r.account.realized, r.equity[-1]-10000.)

    def test_reinvestment_fraction_does_not_overdraw_zero_cash(self):
        a = qbt.Account(1000., 1)
        a.buy(0, 100, 10., 0., 1.)
        a.apply_corporate_actions(np.array([1.]), np.array([1.005]), np.array([10.]))
        self.assertGreaterEqual(a.cash, 0.)
        self.assertLessEqual(a.shares[0], 100.5)
        a.check_cash()

    def test_ration_returns_zero_when_common_scale_cannot_buy_minimum(self):
        d = sample()
        b = qbt.Broker(qbt.build_market(d, log=None))
        q = b._ration(np.array([1000000, 1000000]), np.array([10., 10.]), 1., d.dates[1])
        np.testing.assert_array_equal(q, [0, 0])

    def test_health_accepts_custom_cost_protocol(self):
        class Cost:
            def buy(self, amount, day=None):
                return np.abs(np.asarray(amount)) * .001
            def sell(self, amount, day=None):
                return np.abs(np.asarray(amount)) * .001
        self.assertTrue(qbt.health(sample(), long_all, costs=Cost(), quick=False, prefix_probes=2).ok)

    def test_split_precedes_open_allocation(self):
        d = sample(T=5, N=1)
        d.open[2:] = d.close[2:] = d.high[2:] = d.low[2:] = 5.
        d.open_ff = d.open
        d.cumadj_ff[2:] = 2.
        d.close_adj = d.close * d.cumadj_ff
        events = qbt.CorporateActions([qbt.CorporateAction(str(d.dates[2]), d.codes[0], share_ratio=2.)])
        r = qbt.run(d, long_all, corporate_actions=events, capital=10000., costs=FREE, rebalance_tol=0.)
        self.assertEqual(r.account.shares[0], 1000)
        self.assertEqual(r.account.n_buy+r.account.n_sell, 1)

    def test_event_reset_runs_for_each_interval(self):
        class Strategy:
            def reset(self):
                self.first = True
            def __call__(self, ctx):
                if self.first:
                    self.first = False
                    return ctx.long(np.ones(ctx.N, bool), gross=.5)
                return qbt.HOLD
        d = sample(T=8, N=1)
        strategy = Strategy()
        for start, end in ((d.dates[1], d.dates[2]), (d.dates[5], d.dates[6])):
            self.assertEqual(run(d, strategy, mode="events", start=start, end=end).account.n_buy, 1)

    def test_fractional_rights_wait_for_payment(self):
        d = sample(T=6, N=1)
        d.cumadj_ff[2:] = 1.005
        events = qbt.CorporateActions([qbt.CorporateAction(str(d.dates[2]), d.codes[0],
            share_ratio=1.005, cash_in_lieu_price=10., pay_date=str(d.dates[4]))])
        r = run(d, long_all, corporate_actions=events, capital=2000., end=d.dates[3])
        self.assertEqual(r.account.cash, 1000.)
        self.assertAlmostEqual(r.account.receivables[0], 5.)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
