"""Reproduce the course baseline and reusable factor evaluation on local qbt data."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
import qbt
from qbt.factor_evaluation import (forward_open_returns, split_masks, daily_diagnostics,
    residual_rank, hac_mean, bh_adjust, market_regression, corr)
from strategies.five_day_reversal import five_day_reversal

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'reports/factor_homework'


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    return x


def save_json(path, obj):
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv(frame, name):
    frame.to_csv(OUT/name, index=False, encoding='utf-8-sig')


def run():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    d = qbt.load()
    m = qbt.build_market(d)
    ctx = qbt.Context(d, m)
    u = ctx.universe(min_amt=5e7, min_listed=60, not_st=True).mask
    adv, vol, mom = ctx.adv(20), ctx.vol(20), ctx.ret(20)
    scores = {'5DR': -ctx.ret(5), 'momentum20': mom, 'lowvol20': -vol,
              'liquidity20': np.log(np.maximum(adv, 1))}
    # Economic directions are fixed before evaluation, never flipped on test data.
    scores['5DR_residual'] = residual_rank(scores['5DR'], [adv, vol, mom], u)
    raw_open_adj = np.asarray(d.open, float)*np.asarray(d.cumadj_ff, float)
    summary, all_daily, offsets = [], [], []
    diagnostics = {}
    for name, score in scores.items():
        for h in [1, 5, 10, 20]:
            frame = daily_diagnostics(score, forward_open_returns(raw_open_adj, h), u)
            frame['date'] = d.dates.astype(str)
            frame['factor'], frame['horizon'] = name, h
            exit_idx = np.arange(d.T)+1+h
            frame['realized_date'] = [str(d.dates[i]) if i < d.T else '' for i in exit_idx]
            masks = split_masks(d.dates, h)
            for split, mask in masks.items():
                f = frame.loc[mask]
                stat = hac_mean(f.rank_ic, max(h-1, 5))
                summary.append(dict(factor=name, horizon=h, split=split, **stat,
                    ic=float(f.ic.mean()), rank_icir=float(f.rank_ic.mean()/f.rank_ic.std()),
                    positive_fraction=float((f.rank_ic.dropna()>0).mean()),
                    coverage=float(f.coverage.mean()), spread=float(f.spread.mean()),
                    monotonicity=float(f.monotonicity.mean()),
                    **{f'q{k}':float(f[f'q{k}'].mean()) for k in range(1,6)}))
                if h == 5:
                    for offset in range(5):
                        ff = frame.loc[mask & (np.arange(d.T)%5 == offset)]
                        offsets.append(dict(factor=name,split=split,offset=offset,
                            n=int(ff.rank_ic.count()), rank_ic=float(ff.rank_ic.mean()), spread=float(ff.spread.mean())))
            if h == 5:
                diagnostics[name] = frame
            all_daily.append(frame)
            print('Evaluated', name, h, flush=True)
    summary = pd.DataFrame(summary)
    summary['q_bh'] = np.nan
    for split in summary.split.unique():
        mask = summary.split.eq(split)
        summary.loc[mask,'q_bh'] = bh_adjust(summary.loc[mask,'p'])
    csv(summary, 'factor_scorecard.csv')
    csv(pd.concat(all_daily), 'daily_factor_diagnostics.csv')
    csv(pd.DataFrame(offsets), 'nonoverlap_offsets.csv')

    # Publish rolling values only on the exit/realization date, never signal date.
    dynamic, annual = [], []
    for name, f in diagnostics.items():
        valid = f[f.realized_date.ne('')].copy()
        valid.index = pd.to_datetime(valid.realized_date)
        for w in [21,63,126,252]:
            roll = valid.rank_ic.rolling(w, min_periods=w).mean()
            dynamic.extend(dict(factor=name,date=str(date.date()),window=w,rank_ic=value)
                           for date,value in roll.items() if date >= pd.Timestamp('2025-01-01'))
        for year in [2025,2026]:
            ff = f.loc[(f.date >= f'{year}-01-01') & (f.realized_date <= f'{year}-12-31') & f.realized_date.ne('')]
            annual.append(dict(factor=name,year=year,**hac_mean(ff.rank_ic,5),spread=float(ff.spread.mean())))
    csv(pd.DataFrame(dynamic),'dynamic_rankic.csv')
    csv(pd.DataFrame(annual),'yearly_factor.csv')
    # Same-date rank correlation, averaged across the requested display interval.
    names = list(scores)
    pairs = []
    for i,a in enumerate(names):
        for b in names[i+1:]:
            cs=[]
            for t in np.flatnonzero(d.dates >= np.datetime64('2025-01-01')):
                ok=u[t]&np.isfinite(scores[a][t])&np.isfinite(scores[b][t])
                if ok.sum() >= 100:
                    cs.append(corr(pd.Series(scores[a][t,ok]).rank(),pd.Series(scores[b][t,ok]).rank()))
            pairs.append(dict(factor_a=a,factor_b=b,rank_corr=float(np.nanmean(cs))))
    csv(pd.DataFrame(pairs),'factor_correlations.csv')

    # Market conditions known at the signal close. Threshold for high vol uses train only.
    market_daily = pd.Series(np.nanmean(ctx.ret(1),axis=1))
    trend = (1+market_daily).rolling(20).apply(np.prod,raw=True).to_numpy()-1
    market_vol = market_daily.rolling(20).std().to_numpy()
    threshold = float(np.nanmedian(market_vol[d.dates < np.datetime64('2024-01-01')]))
    conditions=[]
    f=diagnostics['5DR']; test=split_masks(d.dates,5)['test']
    for name,mask in [('past20_up',trend>=0),('past20_down',trend<0),
                      ('high_vol',market_vol>=threshold),('low_vol',market_vol<threshold)]:
        ff=f.loc[test&mask]
        conditions.append(dict(condition=name,**hac_mean(ff.rank_ic,5),spread=float(ff.spread.mean())))
    csv(pd.DataFrame(conditions),'conditions.csv')

    # All scenarios declared before inspecting the newly calculated results.
    specs = [
        dict(id='baseline'), dict(id='zero_cost',zero=True),
        dict(id='slippage10',slip=10),dict(id='k20',k=20),dict(id='k50',k=50),
        dict(id='liquid100m',min_amt=1e8),dict(id='main_board',boards='main'),
        dict(id='weekly',every=5),dict(id='buffer150',buffer=150),
        dict(id='delay1',delay=1),dict(id='capacity005',capacity=.005),
        dict(id='capital100m',capital=1e8),dict(id='lowvol20',factor='lowvol20'),
        dict(id='blend50',factor='blend50')]
    save_json(OUT/'experiment_plan.json',specs)
    results={}; records=[]; navs=None; regressions=[]; yearly=[]
    for spec in specs:
        ident=spec['id']; destination=OUT/ident; destination.mkdir(exist_ok=True)
        k=spec.get('k',100)
        uu=ctx.universe(min_amt=spec.get('min_amt',5e7),min_listed=60,
                        boards=spec.get('boards'),not_st=True)
        score=scores[spec.get('factor','5DR')] if spec.get('factor')!='blend50' else (
            pd.DataFrame(np.where(u,scores['5DR'],np.nan)).rank(axis=1,pct=True).to_numpy() +
            pd.DataFrame(np.where(u,scores['lowvol20'],np.nan)).rank(axis=1,pct=True).to_numpy())*.5
        if spec.get('buffer'):
            target=ctx.long(uu.buffer(score,add_k=k,drop_k=spec['buffer']),gross=.98)
        else:
            sel=uu.top(score,k).mask
            if spec.get('every'):
                anchor=int(np.searchsorted(d.dates,np.datetime64('2025-01-02')))-1
                # Select using only last scheduled signal, absolute calendar schedule.
                for t in range(anchor,d.T):
                    if (t-anchor)%spec['every']:
                        sel[t]=sel[t-1]
            if spec.get('delay'):
                delay=spec['delay']; sel=np.vstack([np.zeros((delay,d.N),bool),sel[:-delay]])
            target=qbt.Target(sel=sel,gross=.98)
        opts=dict(start='2025-01-02',end=str(d.dates[-1]),capital=spec.get('capital',1e7),
                  market=m,max_adv_pct=spec.get('capacity',.02),name=ident)
        if spec.get('zero'):
            opts['costs']=qbt.Costs(commission=0,min_commission=0,transfer_fee=0,stamp_duty=0)
        elif spec.get('slip'):
            opts['costs']=qbt.Costs(slippage_bps=spec['slip'])
        r=qbt.run(d,target,**opts)
        results[ident]=r
        metrics=r.metrics(); records.append(dict(scenario=ident,**metrics))
        save_json(destination/'summary.json',dict(config=spec,metrics=metrics,assumptions=r.assumptions))
        if navs is None: navs=pd.DataFrame({'date':r.dates.astype(str),'market_equal':r.bench_nav/r.bench_nav[0]})
        navs[ident]=r.nav
        pd.DataFrame(r.trades,columns=['t','stock_index','side','qty','price','fee']).assign(
            date=[str(d.dates[t]) for t,j,s,q,p,fee in r.trades],
            code=[d.codes[j] for t,j,s,q,p,fee in r.trades]).to_csv(destination/'trades.csv',index=False,encoding='utf-8-sig')
        regressions.append(dict(scenario=ident,**market_regression(r.ret,r.bench_ret)))
        for year in [2025,2026]:
            mask=np.array([str(day).startswith(str(year)) for day in r.dates[1:]])
            rr=r.ret[mask]; bb=r.bench_ret[mask]
            eq=np.r_[1,np.cumprod(1+rr)]
            yearly.append(dict(scenario=ident,year=year,days=len(rr),
                total=float(eq[-1]-1),benchmark=float(np.prod(1+bb)-1),
                max_drawdown=float((eq/np.maximum.accumulate(eq)-1).min()),
                mean_daily_excess=float(np.mean(rr-bb))))
        if ident=='baseline':
            reference=qbt.run(d,five_day_reversal,**{**opts,'record_trades':False})
            np.testing.assert_allclose(r.equity,reference.equity,rtol=1e-12)
            # 作业构建器引用这个固定页面；显式路径表示更新该次作业的当前产物。
            qbt.report(r,path=str(destination/f"{ident}_{str(r.dates[-1]).replace('-','')}.html"))
        print('Scenario',ident,metrics['累计收益'],flush=True)
    csv(pd.DataFrame(records),'scenario_summary.csv'); csv(navs,'nav.csv')
    csv(pd.DataFrame(regressions),'market_attribution.csv'); csv(pd.DataFrame(yearly),'yearly_account.csv')
    # Compare independently traded signals; blend50 combines ranks before trading.
    increments=[]
    base=results['baseline']
    for name in ['lowvol20','blend50']:
        r=results[name]; delta=r.ret-base.ret
        increments.append(dict(scenario=name,return_corr=corr(r.ret,base.ret),
            total_delta=r.metrics()['累计收益']-base.metrics()['累计收益'],
            drawdown_delta=r.metrics()['最大回撤']-base.metrics()['最大回撤'],
            **hac_mean(delta,5)))
    csv(pd.DataFrame(increments),'portfolio_increment.csv')
    # Retain causal diagnostics rather than selling them as forward selection.
    manifest=dict(data_start=str(d.dates[0]),data_end=str(d.dates[-1]),days=d.T,stocks=d.N,
        display_start='2025-01-02',annualization=242,
        split={'train':'2020-2023','validation':'2024','test':'2025-2026-10-16'},
        fixed_directions={'5DR':'negative past5 return','momentum20':'positive past20 return',
                          'lowvol20':'negative past20 volatility','liquidity20':'positive log ADV20'},
        fdr_family='5 scores including residual x 4 horizons = 20 tests per split',
        historical_holdout_not_blind=True, market_vol_threshold_train=threshold,
        scenario_count=len(specs),baseline_matches_original_strategy=True,
        seconds=round(time.monotonic()-started,2),
        data_sha256={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in (ROOT/'data').glob('*.feather')},
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'qbt/factor_evaluation.py',ROOT/'strategies/five_day_reversal.py']})
    save_json(OUT/'manifest.json',manifest)
    print('Finished',manifest['seconds'],flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
