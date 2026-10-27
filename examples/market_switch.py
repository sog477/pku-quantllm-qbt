"""Run all three markets offline. Fixtures demonstrate rules, not real market data.

From project root: python examples/market_switch.py
For your own data: python run_strategy.py --market US --reference my_reference.json --data bars.csv ...
"""
from pathlib import Path
import json
import sys
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import qbt

ROOT=Path(__file__).resolve().parents[1]


def buy_and_hold(ctx):
    return ctx.long(np.ones(ctx.N,dtype=bool),stops={'hard':.05})


def fixture(profile):
    p=qbt.market_profile(profile)
    days=['2024-03-08','2024-03-11','2024-03-12','2024-03-13','2024-03-14','2024-03-15']
    code={'CN':'600001.SH','US':'DEMO','KR':'005930.KS'}[profile]
    scale=7000 if profile=='KR' else 1
    price=np.array([10,10,9,9,9,9],float)[:,None]*scale
    d=qbt.from_market_arrays(p,days,[code],open=price,close=price,high=price,low=price,
        amt=np.full(price.shape,1e9*scale),num_trades=np.ones(price.shape)*100,cumadj=np.ones(price.shape))
    o,c={'CN':('09:30','15:00'),'US':('09:30','16:00'),'KR':('09:00','15:30')}[profile]
    schedule=[dict(date=day,open=day+'T'+o,close=day+'T'+c,
        **(dict(break_start=day+'T11:30',break_end=day+'T13:00') if profile=='CN' else {})) for day in days]
    manifest=dict(profile=profile,sessions=schedule,settlement_sessions=days,
        securities=[dict(code=code,currency=p.currency,listing_date='2000-01-01',known_at='1999-01-01',listed_sessions_at_start=5000)],
        records=[dict(code=code,field=field,value=False,effective_from='2000-01-01',known_at='1999-01-01') for field in ('st','suspended')],
        actions=[dict(date='2024-03-12',code=code,cash_dividend=float(scale),pay_date='2024-03-14',
                      record_date='2024-03-13',known_at='2024-03-08T09:00',withholding_rate=.1)],
        fees=[dict(effective_from='2024-01-01',effective_until='2024-12-31',commission=0,minimum=0,buy_tax=0,sell_tax=0)],
        source='SYNTHETIC EDUCATIONAL FIXTURE; zero fees and 10% withholding are example parameters, not statutory rates')
    if profile=='KR':
        for i,day in enumerate(days):
            for field,value in [('limit_up',float(price[i,0]*1.3)),('limit_down',float(price[i,0]*.7))]:
                manifest['records'].append(dict(code=code,field=field,value=value,
                    effective_from=day,effective_until=day,known_at=day+'T08:00'))
    return d,manifest


def main():
    base=ROOT/'reports'/'market_upgrade'/'demo';base.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='run_',dir=base))
    results={}; summary={}
    for profile in ('CN','US','KR'):
        ds,manifest=fixture(profile)
        path=out/(profile+'_reference.json')
        path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        ref=qbt.ReferenceData.from_json(path)
        capital=10_000*(7000 if profile=='KR' else 1)
        r=qbt.run(ds,buy_and_hold,capital=capital,market=profile,reference_data=ref,rebalance_tol=1,max_adv_pct=1,
                  name=profile+' synthetic cash-dividend example')
        qbt.report(r,path=str(out/(profile+'.html')),plotly=False)
        results[profile]=r
        summary[profile]=dict(currency=r.currency,nav=r.nav.tolist(),cash=r.cash.tolist(),
             receivables=r.receivables.tolist(),taxes=r.account.taxes,report=str(out/(profile+'.html')))
    # Example FX only, observed/published before every selected valuation.
    quotes=[]
    for day in ['2024-03-08','2024-03-11','2024-03-12','2024-03-13','2024-03-14','2024-03-15']:
        for currency,rate in [('USD',7.),('KRW',.005)]:
            quotes.append(qbt.FXQuote(currency,day+'T00:00Z',day+'T00:01Z',rate))
    fx=qbt.FXRates('CNY',quotes,np.timedelta64(4,'D'))
    us=results['US']
    portfolio=qbt.aggregate_results(results,fx,[us.session_calendar.utc_close(t) for t in us.dates])
    summary['portfolio_CNY']=portfolio['equity'].tolist()
    # A true order-by-order input path, separate from the OHLC examples above.
    feed=[qbt.L3Event('2024-03-08T09:30:00',1,'add','DEMO','ahead','buy',10.,100),
          qbt.L3Event('2024-03-08T09:30:01',2,'add','DEMO','behind','buy',10.,100),
          qbt.L3Event('2024-03-08T09:30:02',3,'cancel','DEMO','ahead','buy',10.,100),
          qbt.L3Event('2024-03-08T09:30:03',4,'execute','DEMO','behind','buy',10.,40)]
    submitted=[]
    def maker(ctx):
        if not submitted: submitted.append(ctx.order(qbt.OrderRequest('DEMO',100,limit_price=10.)))
    replay=qbt.run_l3(feed,maker,profile='US',calendar=results['US'].session_calendar,codes=['DEMO'],capital=10000,
                      costs=qbt.Costs(0,0,0,stamp_duty=0))
    summary['L3_filled_quantity']=sum(f['quantity'] for f in replay.fills)
    replay.save_json(out/'l3_replay.json')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(out)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
