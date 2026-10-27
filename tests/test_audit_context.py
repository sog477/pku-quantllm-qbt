"""策略意外原地写入特征/行情时，不得污染全局数据和后续回测。"""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import qbt


def context():
    prices=np.full((30,1),10.)
    d=qbt.Dataset.from_arrays(np.busday_offset(np.datetime64('2024-01-02'),np.arange(30)),['600001.SH'],
        open=prices,close=prices,high=prices,low=prices,amt=np.full((30,1),1e8),
        num_trades=np.full((30,1),100.),cumadj=np.ones((30,1)))
    m=qbt.build_market(d,log=None)
    return qbt.Context(d,m)


class ContextAuditTests(unittest.TestCase):
    def test_accidental_raw_feature_write_cannot_change_dataset(self):
        ctx=context();before=ctx.d.close.copy()
        with self.assertRaises(ValueError):ctx.close_raw()[-1,0]=999.
        np.testing.assert_array_equal(ctx.d.close,before)

    def test_forward_filled_price_write_cannot_change_later_backtests(self):
        ctx=context();before=ctx.d.close_ff.copy()
        with self.assertRaises(ValueError):ctx.close()[:]=999.
        np.testing.assert_array_equal(ctx.d.close_ff,before)

    def test_market_feature_write_cannot_disable_trade_constraints(self):
        ctx=context();before=ctx.m.suspended.copy()
        with self.assertRaises(ValueError):ctx.suspended()[:]=True
        np.testing.assert_array_equal(ctx.m.suspended,before)

    def test_cached_adjusted_price_cannot_change_future_factor_reads(self):
        ctx=context();before=ctx.ret(5)
        with self.assertRaises(ValueError):ctx.adj()[0]=1.
        np.testing.assert_array_equal(ctx.ret(5),before)

    def test_explicit_feature_copy_can_be_edited(self):
        ctx=context();feature=ctx.close_raw().copy();feature[0]=999.
        self.assertEqual(ctx.d.close[0,0],10.)

    def test_strategy_cannot_change_shared_dates_by_writing_view(self):
        ctx=context();before=ctx.d.dates.copy()
        with self.assertRaises(ValueError):ctx.dates[0]=np.datetime64('2030-01-01')
        np.testing.assert_array_equal(ctx.d.dates,before)

    def test_strategy_symbol_list_edits_do_not_rename_dataset(self):
        ctx=context();codes=ctx.codes;codes[0]='WRONG_SYMBOL'
        self.assertEqual(ctx.d.codes[0],'600001.SH')

    def test_account_snapshot_does_not_follow_later_cash_changes(self):
        account=qbt.Account(10000,1);snap=account.snapshot()
        account.buy(0,100,10.,0.,1.)
        self.assertEqual(snap['cash'],10000.)


if __name__=='__main__':unittest.main(verbosity=2)
