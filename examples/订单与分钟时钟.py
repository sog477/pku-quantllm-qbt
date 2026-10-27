"""合成分钟行情：两种现金资产、部分成交、撤单、T+0和跨会话T+1。"""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import qbt


class Strategy:
    def reset(self):
        self.waiting=None

    def __call__(self,ctx):
        if ctx.T==1:
            ctx.order("600001.SH",200,tif="GTC",tag="分批成交")
            ctx.order("CASH_ASSET",5,tif="GTC")
            self.waiting=ctx.order("600001.SH",100,limit_price=9.,tif="GTC",tag="等待后撤销")
        if ctx.T==2:
            ctx.order("600001.SH",200,side="sell",tif="GTC")
            ctx.order("CASH_ASSET",5,side="sell",tif="GTC")
        if ctx.T==3:
            ctx.cancel(self.waiting)


def main():
    times=np.array(["2024-01-02T09:30","2024-01-02T09:31","2024-01-02T09:32",
                    "2024-01-03T09:30","2024-01-03T09:31","2024-01-03T09:32"],dtype="datetime64[ns]")
    prices=np.full((6,2),10.)
    d=qbt.BarDataset.from_arrays(times,["600001.SH","CASH_ASSET"],previous_close=[10.,10.],
        open=prices,high=prices,low=prices,close=prices,amt=np.full((6,2),1e5),
        num_trades=np.full((6,2),100.),cumadj=np.ones((6,2)))
    assets=qbt.CashAssetRules(minimum=[100,1],step=[100,1],settlement_days=[1,0],price_limits=[True,False])
    r=qbt.run_orders(d,Strategy(),capital=10000.,clock=qbt.BarClock(periods_per_year=242*3),
        asset_rules=assets,max_adv_pct=.01,
        costs=qbt.Costs(commission=0,min_commission=0,transfer_fee=0,stamp_duty=0),
        name="订单与分钟时钟示例")
    print("这是合成数据；每会话3根bar，年化参数也显式设为242*3。")
    for o in r.orders:
        print(o.id,o.request.code,o.request.side,o.status,"成交",o.filled,"剩余",o.remaining)
    for fill in r.order_fills:
        print(fill.timestamp,fill.code,fill.side,fill.quantity,fill.price,fill.phase)
    assert r.account.cash==10000. and not r.account.shares.any()
    assert r.orders[2].status=="CANCELED"
    stock_sells=[f.index for f in r.order_fills if f.code=="600001.SH" and f.side=="sell"]
    assert stock_sells==[3,4]
    print(qbt.report(r,path=str(ROOT/"reports"),plotly=False).path)
    return r


if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
    main()
