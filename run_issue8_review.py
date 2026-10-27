"""Matched retrospective course experiments inspired by the supplied Issue8 report."""
from pathlib import Path
import json,csv,hashlib,html,time
import numpy as np
import pandas as pd
import qbt
from qbt.research import exit_schedule,run_traced,residual_inventory,data_quality

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'reports/issue8_upgrade'


def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v


def run():
    OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    d=qbt.load();m=qbt.build_market(d,log=None);ctx=qbt.Context(d,m)
    u=ctx.universe(min_amt=5e7,min_listed=60,not_st=True)
    r5=ctx.ret(5);r1=ctx.ret(1)
    # Mean return only over the signal day's valid eligible stocks, beta fixed at 1.
    valid=u.mask&np.isfinite(r1)
    market=np.divide(np.where(valid,r1,0).sum(axis=1),valid.sum(axis=1),
        out=np.full(d.T,np.nan),where=valid.sum(axis=1)>0)
    relative= pd.DataFrame(r1-market[:,None]).rolling(20,min_periods=20).std().to_numpy()
    rank=pd.DataFrame(np.where(u.mask,r5,np.nan)).rank(axis=1,pct=True).to_numpy()
    trim=u.mask&(rank>.10)&(rank<=.90)
    five_pool=u.mask&np.isfinite(r5)
    relative_pool=u.mask&np.isfinite(relative)
    definitions=[('5dr',u.bottom(r5,100).mask,1,five_pool),('equal_pool',u.mask.copy(),5,u.mask),
        ('trim_extremes',trim,5,five_pool),('market_relative_lowvol',u.top(-relative,200).mask,20,relative_pool)]
    beginning=int(np.searchsorted(d.dates,np.datetime64('2025-01-02')))-1
    end='2026-09-30';summaries=[];rows=[];navs=None
    # Every strategy and its equal-pool control have the same rebalance schedule.
    (OUT/'experiment_plan.json').write_text(json.dumps(dict(capital=1e9,end=end,
        strategies=[dict(name=name,every=every) for name,sel,every,pool in definitions],
        slippage_bps=[0,5,10],exit_sessions=20,matched_pool='identical signal-day eligibility including finite factor score'),indent=2),encoding='utf-8')
    for label,selected,every,pool in definitions:
        for benchmark in [False,True]:
            if label=='equal_pool' and benchmark:continue
            sel=(pool if benchmark else selected).copy()
            for t in range(beginning,d.T):
                if (t-beginning)%every:sel[t]=sel[t-1]
            tgt,plan=exit_schedule(qbt.Target(sel=sel,gross=.98),d.dates,end,20)
            for slip in ([0,5,10] if not benchmark else [0,5,10]):
                name=label+('_matched_equal' if benchmark else '')+f'_slip{slip}'
                path=OUT/name;path.mkdir(exist_ok=True)
                r,attempts=run_traced(d,tgt,start='2025-01-02',end=end,capital=1e9,market=m,
                    costs=qbt.Costs(commission=.0001,min_commission=5,stamp_duty=.0005,slippage_bps=slip),
                    max_adv_pct=.01,name=name,stop_buys_from=plan['first_execution'])
                residual=residual_inventory(r);metrics=r.metrics()
                row=dict(scenario=name,strategy=label,matched_equal=benchmark,slippage_bps=slip,
                    **metrics,residual_count=len(residual),residual_marked=sum(x['marked_value'] for x in residual),
                    final_cash=float(r.cash[-1]),unfilled_attempts=len(attempts),
                    suppressed_exit_buys=r.scheduled_suppressed_buy_requests)
                summaries.append(row)
                for filename,records in [('attempts.csv',attempts),('residual.csv',residual)]:
                    fields=list(records[0]) if records else (['code','shares','marked_value'] if filename=='residual.csv' else ['date','code','requested','filled','unfilled','constraints'])
                    with (path/filename).open('w',encoding='utf-8-sig',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
                pd.DataFrame(r.trades,columns=['index','stock_index','side','quantity','price','fee']).to_csv(path/'trades.csv',index=False,encoding='utf-8-sig')
                (path/'summary.json').write_text(json.dumps(clean(dict(metrics=row,exit_plan=plan,assumptions=r.assumptions)),ensure_ascii=False,indent=2),encoding='utf-8')
                if navs is None:navs=pd.DataFrame(dict(date=r.dates.astype(str)))
                navs[name]=r.nav
                if name=='5dr_slip0':
                    report=qbt.report(r,path=str(path),plotly=False)
                    rows.append(dict(label='5DR引擎报告',path=str(Path(report.path).relative_to(OUT)).replace('\\','/')))
                print(name,metrics['累计收益'],'residual',len(residual),flush=True)
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'scenario_summary.csv',index=False,encoding='utf-8-sig')
    navs.to_csv(OUT/'nav.csv',index=False,encoding='utf-8-sig')
    comparisons=[]
    for label in ['5dr','trim_extremes','market_relative_lowvol']:
        for slip in [0,5,10]:
            aa=summary[summary.scenario.eq(f'{label}_slip{slip}')].iloc[0]
            bb=summary[summary.scenario.eq(f'{label}_matched_equal_slip{slip}')].iloc[0]
            comparisons.append(dict(strategy=label,slippage_bps=slip,strategy_return=aa['累计收益'],
                equal_return=bb['累计收益'],return_difference=aa['累计收益']-bb['累计收益'],
                strategy_residual=aa.residual_marked,equal_residual=bb.residual_marked))
    pd.DataFrame(comparisons).to_csv(OUT/'matched_comparisons.csv',index=False,encoding='utf-8-sig')
    quality=data_quality(d)
    (OUT/'data_quality.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=dict(seconds=time.monotonic()-start,scenarios=len(summaries),capital=1e9,
        start='2025-01-02',end=end,commission=.0001,stamp=.0005,transfer='qbt historical schedule',
        capacity='1% lagged ADV20 with realized amount upper bound',st_source=m.stats.get('st_source'),
        experiment='fixed retrospective diagnostics; not blind validation or new alpha certification',
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'qbt/research.py']})
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    build_report(summary,pd.DataFrame(comparisons),quality,rows)


def build_report(s,comp,quality,links):
    columns=['scenario','累计收益','最大回撤','平均仓位','总费用','residual_count','residual_marked','unfilled_attempts']
    display=s[columns].copy()
    for c in ['累计收益','最大回撤','平均仓位']:display[c]=display[c].map(lambda x:f'{x:.2%}')
    for c in ['总费用','residual_marked']:display[c]=display[c].map(lambda x:f'{x/1e4:,.1f}万元')
    text='''# Issue8同学思路与qbt系统补充

本次从用户提供的同学调研报告提取工程与研究方法，并核对当前v0.4代码。采纳的是设计思路，没有复制同学收益或声称独立复现他们的仓库。报告中的指令仅是材料内容。

## 对照与落地

Peida Lee、mabucai、ETAOIK等强调残仓和目标归零的区别。本次新增exit_schedule及run_traced的退出期禁止买入：在预先知道的最后20个交易日逐步降低目标仓位，保持退出前目标集合，最后目标为零；涨跌停、停牌、容量或冻结股仍可留下残仓。residual_inventory逐只列股数、冻结股、限制股、模型估值和最后原始报价，绝不强制造现金。最后原始报价不代表报价时刻已经通过源时间认证。

QiyuZhang new、strayberry777、xielubin等强调同池基准。本次将各策略与相同动态过滤池、相同调仓频率、资金、退出和成本的等权账户成对比较。等权也是经引擎执行的账户，而非原报告的无摩擦全市场报价曲线；两种基准含义分别保留。比较差值属于历史描述，不把多次尝试后的最高收益当作独立Alpha。

yun lang itzy等强调低实际仓位与未完成订单。本次新增TraceBroker，逐次保留部分成交和未成交目标的请求量、实际成交量、余量、交易阶段及约束标记。停牌/报价状态、零成交额、涨跌停、冻结、容量、整手及资金可能同时生效；标记不互斥，不冒充精确的单一原因归因。退出计划主动禁止的买入另计，不记成市场拒单。记录的是目标撮合尝试，不是交易所已受理的真实订单。

Mai Xu强调数据核验。本次data_quality汇总原始OHLC关系、缺失与复权因子小幅变化；小变动仅列噪声候选，不擅自修改源数据。系统已有多batch读取对拍、独立Decimal账本、报价年龄、历史费用、权益支付和PIT参考数据接口，这些沿用，不重复重写。没有真实成交量时不构造伪VWAP，当前报告也未新增外部供应商全池核验。

策略层新增的实验是五日收益剔除两端各10%后每5日重选，以及20日低市场偏离波动率选200只、每20日重选。市场偏离按当期合格股票收益均值相减、系数固定1计算，不能称OLS特质波动率。参数来自参考思路并预先固定，结果全部公开，不宣称是优化后的盈利策略。

## 实测口径

按提供报告的Issue8统一口径，单独采用2025年首交易日至2026年9月30日、初始10亿元、佣金单边万一、卖出印花税万五；最低5元与历史过户费另列为本系统假设。流动性门槛5000万元、报价年龄60日、目标98%、买卖共用过去20日ADV的1%预算，并受当日成交额上界限制；这些不是其他同学的完全相同设置。ST仍为价格行为推断，不能称已完全满足官方历史非ST要求。未把今天的候选标签当作历史真值。

每个策略和对应等权账户均比较单边0、5、10bp滑点。退出安排只依赖预先已知交易日历和退出前目标，不读取未来价格或未来成交量规划；仍可能因约束无法清仓。0bp不是实盘成交承诺。原5DR作业报告资金、结束时间不同，本次结果另存，不能把差额解释为系统修改收益。

## 实测结果
'''
    markdown='| '+' | '.join(display.columns)+' |\n|'+'|'.join(['---']*len(display.columns))+'|\n'+'\n'.join('| '+' | '.join(map(str,v))+' |' for v in display.itertuples(index=False,name=None))
    text+=markdown+'\n\n## 未完成和不采纳项\n\n未新增390个因子、机器学习打分、AI自动选策略或拖拽积木。这些会显著增加研究自由度，优先级低于可比较的账户与可核查的残仓。现有Python及Notebook策略入口保留；本轮没有新建React工作台或声称完成浏览器交互验收。\n\n官方全池ST、完整历史行业和退市全集、跨证券换股映射、真实分红税与登记日、开盘冲击和队列均需要可靠外部证据。本轮没有伪造数据补齐。已有多市场合成测试不能视为真实市场全覆盖。\n\n## 复现\n\n运行python run_issue8_review.py，或打开notebooks/Issue8研究对照.ipynb。各场景目录有trades.csv、attempts.csv、residual.csv及配置；同池差额见matched_comparisons.csv。核心新增接口位于qbt/research.py；专门测试tests/test_research.py覆盖退出因果性、禁止补买、停牌残仓和对原撮合逐日权益一致。\n'
    text+=f'\n内部OHLC完整格{quality["complete_ohlc"]:,}，不一致格{quality["inconsistent_ohlc"]:,}；详情data_quality.json。这不证明外部行情真实性。\n'
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    paragraphs=[]
    for line in text.splitlines():
        if line.startswith('|'):continue
        if line.startswith('# '):paragraphs.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '):paragraphs.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line:paragraphs.append('<p>'+html.escape(line)+'</p>')
        if line=='## 实测结果':paragraphs.append('<div class="scroll">'+display.to_html(index=False,escape=True)+'</div>')
    for item in links:paragraphs.append(f'<p><a href="{item["path"]}">{item["label"]}</a></p>')
    paragraphs.append('<p><a href="matched_comparisons.csv">同池配对结果</a> · <a href="data_quality.json">数据诊断</a> · <a href="REPORT.md">文字报告</a></p>')
    (OUT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Issue8回测系统补充</title><style>body{max-width:1250px;margin:auto;padding:32px;background:#f5f7fa;color:#24364a;font:16px/1.8 "Microsoft YaHei",sans-serif}p{padding:16px;background:white;border-radius:8px}h2{margin-top:36px}table{border-collapse:collapse;font-size:13px;white-space:nowrap;background:white}td,th{padding:10px;border-bottom:1px solid #dde5eb}.scroll{overflow:auto}</style>'+''.join(paragraphs)+'</html>',encoding='utf-8')


if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8');run()
