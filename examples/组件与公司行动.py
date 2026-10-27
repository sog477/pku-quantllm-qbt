"""可直接运行的合成市场示例；不冒充课程股票的真实分红数据。"""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import qbt


class LoggedRisk(qbt.RiskControls):
    calls = 0

    def exits(self, account, adj_open):
        type(self).calls += 1
        return super().exits(account, adj_open)


def main():
    dates = np.busday_offset(np.datetime64("2024-01-02"), np.arange(6))
    prices = np.full((6, 2), 10.)
    prices[2:, 0], prices[2:, 1] = 9., 5.
    factors = np.ones((6, 2))
    factors[2:, 0], factors[2:, 1] = 10./9., 2.
    d = qbt.Dataset.from_arrays(dates, ["600001.SH", "600002.SH"],
        open=prices, high=prices, low=prices, close=prices, amt=np.full((6, 2), 1e8),
        num_trades=np.full((6, 2), 1000.), cumadj=factors)
    events = qbt.CorporateActions([
        qbt.CorporateAction(str(dates[2]), d.codes[0], cash_dividend=1., pay_date=str(dates[4])),
        qbt.CorporateAction(str(dates[2]), d.codes[1], share_ratio=2., share_date=str(dates[4])),
    ])
    meta = qbt.MarketMetadata(st=np.zeros((6, 2), bool), no_limit=np.zeros((6, 2), bool))
    market = qbt.build_market(d, metadata=meta, log=None)
    def strategy(ctx):
        return ctx.long(np.array([True, True]), gross=.8)
    r = qbt.run(d, strategy, market=market, corporate_actions=events,
        capital=10000., rebalance_tol=1., risk_factory=LoggedRisk,
        allocation=qbt.WeightAllocation(),
        costs=qbt.Costs(commission=0, min_commission=0, transfer_fee=0, stamp_duty=0),
        name="显式公司行动示例")
    print("合成市场：每股分红1元，另一股1拆2；现金与新增股份在第4日到账/可卖。")
    for day, cash, receivable, equity in zip(r.dates, r.cash, r.receivables, r.equity):
        print(f"{day} 现金={cash:.2f} 应收={receivable:.2f} 总资产={equity:.2f}")
    assert r.account.shares.tolist() == [400, 800]
    assert r.account.cash == 2400.
    assert np.allclose(r.equity, 10000.)
    assert not r.account.restricted.any()
    assert LoggedRisk.calls > 0
    path = qbt.report(r, path=str(ROOT / "reports"), plotly=False).path
    print("报告：", path)
    return r


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
