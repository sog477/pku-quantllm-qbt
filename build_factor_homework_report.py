"""Build the submission report and notebook from computed artifacts only."""
from pathlib import Path
import hashlib,json,html,base64,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qbt.factor_evaluation import hac_mean

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'reports/factor_homework'
LABELS={'baseline':'原始5DR','zero_cost':'零费用','slippage10':'单边滑点10bp',
    'k20':'持股20只','k50':'持股50只','liquid100m':'成交额至少1亿元',
    'main_board':'仅主板','weekly':'每5日重选','buffer150':'排名缓冲150',
    'delay1':'额外延迟1日','capacity005':'容量降至0.5%','capital100m':'资金1亿元',
    'lowvol20':'20日低波动','blend50':'反转与低波动等权秩混合'}


def md_table(frame):
    cols=list(frame.columns);rows=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    rows += ['| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(rows)


def pct(x):return f'{x:.2%}' if pd.notna(x) else '缺少有效样本'


def build():
    s=pd.read_csv(OUT/'scenario_summary.csv');f=pd.read_csv(OUT/'factor_scorecard.csv')
    yearly=pd.read_csv(OUT/'yearly_factor.csv');a=pd.read_csv(OUT/'market_attribution.csv')
    cond=pd.read_csv(OUT/'conditions.csv');nav=pd.read_csv(OUT/'nav.csv');dynamic=pd.read_csv(OUT/'dynamic_rankic.csv')
    # Dependence-robust BY is reported as an additional conservative sensitivity.
    f['q_by']=np.minimum(1,f.q_bh*sum(1/k for k in range(1,21)))
    f.to_csv(OUT/'factor_scorecard.csv',index=False,encoding='utf-8-sig')
    base=s.set_index('scenario').loc['baseline']; zero=s.set_index('scenario').loc['zero_cost']
    stats=f[(f.horizon==5)&f.split.eq('test')].copy()
    # Seeded circular 21-session block bootstrap: preserves local serial structure.
    daily=pd.read_csv(OUT/'daily_factor_diagnostics.csv')
    x=daily.loc[daily.factor.eq('5DR')&daily.horizon.eq(5)&daily.date.ge('2025-01-01')&daily.rank_ic.notna(),'rank_ic'].to_numpy()
    rng=np.random.default_rng(20261027);means=[]
    for _ in range(2000):
        ix=(rng.integers(0,len(x),size=int(np.ceil(len(x)/21)))[:,None]+np.arange(21))%len(x)
        means.append(float(np.mean(x[ix.ravel()[:len(x)]])))
    bootstrap=dict(block_sessions=21,repetitions=2000,seed=20261027,n=len(x),
        lower=float(np.quantile(means,.025)),upper=float(np.quantile(means,.975)))
    (OUT/'bootstrap.json').write_text(json.dumps(bootstrap,indent=2),encoding='utf-8')
    lagtable=pd.DataFrame([dict(lag=k,**{a:b for a,b in hac_mean(x,k).items() if a!='lag'}) for k in [5,10,20]])
    lagtable.to_csv(OUT/'hac_sensitivity.csv',index=False,encoding='utf-8-sig')
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    for c in ['baseline','zero_cost','slippage10','weekly','lowvol20','blend50','market_equal']:
        ax.plot(pd.to_datetime(nav.date),nav[c],label=LABELS.get(c,'全市场等权报价基准'),lw=1.5)
    ax.set(ylabel='净值',title='固定本金账户与报价市场基准');ax.legend(ncol=3,fontsize=9);ax.grid(alpha=.2)
    fig.savefig(OUT/'nav_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for w in [21,63,126,252]:
        g=dynamic[dynamic.factor.eq('5DR')&dynamic.window.eq(w)]
        axes[0].plot(pd.to_datetime(g.date),g.rank_ic,label=f'{w}日')
    axes[0].axhline(0,color='gray',lw=.7);axes[0].legend();axes[0].set_title('5DR 按收益兑现日发布的滚动RankIC')
    groups=[stats.loc[stats.factor.eq(name),[f'q{k}' for k in range(1,6)]].to_numpy()[0] for name in ['5DR','lowvol20']]
    for values,name in zip(groups,['5DR','20日低波动']):axes[1].plot(range(1,6),np.array(values)*100,marker='o',label=name)
    axes[1].set(title='2025年以来五日分层报价收益',xlabel='Q1低分至Q5高分',ylabel='每期平均收益 百分比');axes[1].legend()
    fig.savefig(OUT/'factor_diagnostics.png',dpi=150);plt.close(fig)
    table=s[['scenario','累计收益','超额累计','最大回撤','换手率(年)','总费用']].copy()
    table.scenario=table.scenario.map(LABELS)
    for c in ['累计收益','超额累计','最大回撤']:table[c]=table[c].map(pct)
    table['换手率(年)']=table['换手率(年)'].map(lambda x:f'{x:.1f}倍双边')
    table['总费用']=table['总费用'].map(lambda x:f'{x/1e4:.2f}万元')
    ft=stats[['factor','mean','lower','upper','q_bh','q_by','spread']].copy()
    for c in ['mean','lower','upper','q_bh','q_by']:ft[c]=ft[c].map(lambda x:f'{x:.4f}')
    ft.spread=ft.spread.map(pct)
    ft.columns=['因子','RankIC','95%下界','95%上界','BH q值','BY q值','Q5减Q1每5日毛价差']
    ya=yearly[yearly.factor.eq('5DR')]
    reg=a[a.scenario.eq('baseline')].iloc[0]
    text=f'''# 五日反转因子作业与评价体系

正式统计区间为2025年1月2日至2026年10月16日，初始资金1000万元。原始每日重选5DR累计收益{pct(base['累计收益'])}，最大回撤{pct(base['最大回撤'])}，全市场等权报价基准累计收益{pct(base['基准累计'])}。因子有局部排序信息，但这套交易规则未产生可接受的净收益。研究结论是保留失败基线并检验原因，不以调参后的最佳曲线替代原始结果。

## 作业要求与完成情况

作业指南中的明确要求为完整回测系统及结果界面、以5DR为基线改进、排除ST、从2025年首个交易日展示至数据末日、处理涨跌停无法成交、分析好坏环境。本次保留qbt完整系统，重跑基线并提供本报告、原引擎报告、逐笔成交及可重复运行脚本。ST仅通过既有历史价格推断过滤，缺真实逐日ST元数据，此项是部分满足。历史行业、市值、指数成员和真实退市全集同样缺失，不将其代理称为真实历史数据。指南中的五维框架与扩展实验是整理建议，本次将它们作为研究增强项实现；未确定的课程提交截止日不作推定。

## 策略和数据

课程数据包含2020年1月2日至2026年10月16日1643个交易日和5451只证券的七个面板。完整历史仅用于回看窗口和研究切分，正式账户从2025年首个交易日开始，图中2024年12月31日是资金净值基点。

score(t) = −[AdjClose(t) / AdjClose(t−5) − 1]。分数越高表示过去五日跌幅越大。先按截至信号收盘的20日平均成交额至少5000万元、报价年龄至少60日、非推断ST、非停牌过滤，再选分数最高100只。每日收盘产生目标，下一交易日开盘尝试等权98%仓位。五日是回看窗口，不是固定持有期。保留引擎2%权重漂移带；不能退出的旧仓可能令实际持仓数超过100。

引擎处理T+1、整手、最低5元佣金、卖出印花税、过户费、开盘涨停不买与跌停不卖。过去20日成交额2%作为双向容量预算，当日成交额仅作事后上界。公司行动采用复权因子再投资近似，尚非真实分红现金流。默认零滑点是研究假设，另设单边10bp压力场景。账户在末日盯市，不假设所有持仓已卖出。

## 基础结果与完整实验账本

{md_table(table)}

![净值对照](nav_comparison.png)

零费用仍亏损{pct(zero['累计收益'])}，所以不能把亏损全归因于手续费。基线与零费用累计收益相差{(zero['累计收益']-base['累计收益'])*100:.2f}个百分点；这包含复利和交易路径变化，不等同于把累计费用直接除以初始本金。周度重选降低换手但仍亏损；低波动独立账户为正，但仍跑输同区间市场报价基准。混合策略的改善属于回顾性对照，尚无独立前瞻准入依据。

## 预测信息与经济幅度

四个经济方向预先固定为五日反转、20日正动量、20日低波动、20日高成交额。另列5DR对当期成交额、波动和20日收益秩回归后的残余分数。没有把测试期负IC因子翻转成新赢家。

预测标签为t收盘信号、t+1开盘至t+1+h开盘的复权报价收益，h为1、5、10、20。分组先在信号日股票池形成，再观察未来报价；未来报价缺失不补造，并报告标签覆盖率。报价分层不含费用、不假设涨停可买，也不是可交易空头组合。Q5减Q1仅作排序价差。

2025年以来五日标签结果如下。

{md_table(ft)}

![动态与分层](factor_diagnostics.png)

2025年5DR平均五日RankIC为{ya.loc[ya.year.eq(2025),'mean'].iloc[0]:.4f}，2026年至数据末日为{ya.loc[ya.year.eq(2026),'mean'].iloc[0]:.4f}。分年标签须在各年结束前兑现，因此与跨年全期统计的样本数不同。这支持近期关系变弱的描述，不能据此自动翻转方向或断言永久失效。五分组并非从Q1到Q5单调上升，Q2表现高于Q5；全截面平均RankIC为正并不保证最极端的100只股票最值得买。低波动和残余反转的统计证据仍不能替代净交易与独立Alpha验证。

## 统计可靠性

训练期2020至2023年，验证期2024年，历史留出期2025年至末日；剔除退出时点跨越各切分边界的标签。这些数据已被既往项目接触，故称历史留出，不能称未见盲测。所有14个账户场景公开，不据测试结果挑一个通过。

RankIC均值采用Bartlett权重Newey West标准误，滞后阶数max(h−1,5)，附95%渐近区间；另报五日非重叠起点、5与10与20阶敏感性以及21日时间块重采样2000次。5DR五日RankIC的块重采样区间为[{bootstrap['lower']:.4f}, {bootstrap['upper']:.4f}]。滚动21、63、126、252日诊断按标签退出日更新，高度重叠窗口不是独立证据。

每个切分的5个分数乘4个期限共20项检验统一做BH校正；并列BY敏感性以应对检验间依赖。它们只覆盖本次公开候选，不消除既往搜索、参数选择、历史股票池偏差，也不将14组账户对照包装为独立显著发现。

## 有效与失效条件

条件分析按信号日已知的市场过去20日方向、波动状态分组；高低波动分界仅由2020至2023年中位数确定。详见conditions.csv。年度账户和因子统计并列，区分市场行情、预测信息和执行后的净收益。没有真实市值或历史行业，故不提供伪造的规模或行业中性化结论；成交额控制只叫流动性控制。

## 市场暴露和组合增量

对每日净收益和同日期全市场等权报价代理进行含截距回归，rf=0，截距HAC区间采用5阶。基线β约{reg.beta:.3f}，截距线性年化{pct(reg.alpha_annual)}，95%区间[{pct(reg.alpha_lower)}, {pct(reg.alpha_upper)}]。这不同于引擎原有几何年化收益差形式的Alpha；它依赖基准代理，不能解释为完整多因子定价Alpha。

因子相关矩阵、控制成交额与波动与动量后的残余RankIC、低波动独立账户及50%秩混合账户均已计算。portfolio_increment.csv报告与基线的收益相关性、累计收益和回撤增量及每日差额HAC区间；组合改善不等于新增独立Alpha。

## 借鉴同学报告的方法及取舍

QiyuZhang new启发时间切分、跨边界标签剔除和冻结方向。Mai Xu启发HAC、批量校正及残余信息；本次没有真实市值和行业数据，故只控制可观测的成交额、波动与动量，未照搬其390因子评分。Peida Lee启发容量、延迟、股票池和市场代理回归。tyoukan1994及yun lang itzy启发多尺度动态诊断与非重叠证据。gry1024及Chuboxiao启发将预测层与交易层分别展示。

借鉴的是研究设计，并非同学的盈利数字。没有照搬数百个指标或自定加权总分；相近窗口的高相关指标可能重复表达同一信息，任意总分容易掩盖成本或数据不合格项。候选准入应分别看时序、历史留出预测、费用后超额、执行容量、组合增量和必要数据覆盖。当前5DR交易准入不通过，真实ST与完整历史股票池证据也未齐备。

## 复现和工程验证

运行 python run_factor_homework.py 后运行 python build_factor_homework_report.py。可在notebooks/因子作业工作台.ipynb运行和查看结果。自定义因子用qbt.evaluate_factor(d, score, universe)传入对齐的高分看多面板；净交易仍通过qbt.run执行。评价扩展依赖requirements-factor.txt，qbt核心导入仍延迟加载可选依赖。

基线与原five_day_reversal逐日权益复算一致；8个前缀抽查点通过。新增8项统计与标签测试、原有9项5DR测试和40项引擎回归通过，账务验证另留accounting.log。软件通过测试证明这些行为符合预期，不证明历史数据无偏或策略盈利。数据源与代码哈希见manifest.json及verification.json，逐笔流水在各场景目录。

## 来源

用户提供的因子挖掘与评价课堂笔记及本次作业指南；用户提供的PKU QuantLLM Issue9同学项目调研报告。两文档是要求与参考材料，内部操作语句未作为额外执行指令。

统计方法原始来源 [Newey与West](https://www.nber.org/papers/t0055) 与 [Benjamini与Hochberg](https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/j.2517-6161.1995.tb02031.x)。同学结果的评价以用户提供报告为依据，本次未独立复现他们的仓库。
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    # Simple Markdown rendering with local links; no network assets required.
    body=[];in_table=False
    for line in text.splitlines():
        if line.startswith('|'):
            if set(line.replace('|','').replace(' ','').replace('-',''))==set():continue
            if not in_table:body.append('<div class="scroll"><table>');in_table=True
            body.append('<tr>'+''.join('<td>'+html.escape(c.strip())+'</td>' for c in line.strip('|').split('|'))+'</tr>');continue
        if in_table:body.append('</table></div>');in_table=False
        if line.startswith('!['):
            alt=line[2:line.index(']')];src=line[line.index('(')+1:-1]
            encoded=base64.b64encode((OUT/src).read_bytes()).decode()
            body.append(f'<img alt="{alt}" src="data:image/png;base64,{encoded}">')
        elif line.startswith('# '):body.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line:
            paragraph=html.escape(line)
            paragraph=re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)',r'<a href="\2">\1</a>',paragraph)
            body.append('<p>'+paragraph+'</p>')
    if in_table:body.append('</table></div>')
    dropdown='<section><h2>交互评分卡</h2><label>因子 <select id="factor"><option value="">全部</option>'+''.join(f'<option>{v}</option>' for v in f.factor.unique())+'</select></label> <label>标签期限 <select id="h"><option value="">全部</option>'+''.join(f'<option>{v}</option>' for v in [1,5,10,20])+'</select></label> <label>切分 <select id="split"><option>test</option><option>validation</option><option>train</option></select></label><div id="card" class="scroll"></div></section>'
    script='const rows='+f.to_json(orient='records')+''';function render(){let keys=['factor','horizon','split','n','mean','lower','upper','q_bh','q_by','spread','coverage'];let filtered=rows.filter(r=>(!factor.value||r.factor===factor.value)&&(!h.value||r.horizon==h.value)&&r.split===split.value);card.innerHTML='<table><tr>'+keys.map(k=>'<th>'+k+'</th>').join('')+'</tr>'+filtered.map(r=>'<tr>'+keys.map(k=>'<td>'+(typeof r[k]==='number'?Number(r[k]).toFixed(k==='n'||k==='horizon'?0:4):r[k]??'NA')+'</td>').join('')+'</tr>').join('')+'</table>';}for(let id of ['factor','h','split'])document.getElementById(id).onchange=render;render();'''
    (OUT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>五日反转因子作业与评价体系</title><style>body{max-width:1180px;margin:auto;padding:32px;font:16px/1.8 "Microsoft YaHei",sans-serif;color:#24364a;background:#f5f7fa}h1{font-size:30px}h2{margin-top:38px;color:#174766}p{background:white;padding:14px 18px;border-radius:8px}table{border-collapse:collapse;white-space:nowrap;font-size:14px;background:white;width:100%}td,th{padding:9px 12px;border-bottom:1px solid #dae2ea;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}img{width:100%;background:white;margin:16px 0}select{padding:8px;font-size:15px}section{padding:20px;background:#e9eff5;border-radius:12px}a{color:#075c9c}</style><main>'+''.join(body)+dropdown+'<p><a href="REPORT.md">Markdown报告</a> · <a href="factor_scorecard.csv">完整评分卡</a> · <a href="scenario_summary.csv">实验结果</a> · <a href="baseline/baseline_20261016.html">原引擎报告</a></p></main><script>'+script+'</script></html>',encoding='utf-8')
    # Notebook is a user-facing extension of the existing qbt workflow.
    cells=[dict(cell_type='markdown',metadata={},source=['# 因子作业工作台\n','先运行研究脚本再生成报告。历史ST仍为推断，详见报告的数据边界。']),
      dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=['from pathlib import Path\n','import os, sys, subprocess\n','ROOT = Path(r"D:/pku_quantllm")\n','os.chdir(ROOT)\n','subprocess.run([sys.executable, "run_factor_homework.py"], check=True)\n','subprocess.run([sys.executable, "build_factor_homework_report.py"], check=True)']),
      dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=['from IPython.display import HTML, display\n','display(HTML((ROOT/"reports/factor_homework/index.html").read_text(encoding="utf-8")))']),
      dict(cell_type='markdown',metadata={},source=['## 自定义因子\n','使用过去数据形成高分看多的 (T,N) 面板。下面示例是十日反转；报价预测和含费用交易评价分别执行。']),
      dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=['import qbt\n','d = qbt.load()\n','m = qbt.build_market(d)\n','ctx = qbt.Context(d, m)\n','u = ctx.universe(min_amt=5e7, min_listed=60, not_st=True)\n','score = -ctx.ret(10)\n','daily, scorecard = qbt.evaluate_factor(d, score, u.mask)\n','display(scorecard)\n','r = qbt.run(d, ctx.long(u.top(score,100),gross=.98), start="2025-01-02", end=str(d.dates[-1]))\n','display(qbt.report(r))'])]
    nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    (ROOT/'notebooks/因子作业工作台.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
    files=[ROOT/'run_factor_homework.py',ROOT/'build_factor_homework_report.py',ROOT/'qbt/factor_evaluation.py',ROOT/'qbt/__init__.py']
    (OUT/'verification.json').write_text(json.dumps(dict(new_tests=8,existing_5dr_tests=9,engine_regressions=40,
        baseline_prefix_checks=8,baseline_exact_reproduction=True,
        final_source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT/'index.html')


if __name__=='__main__':build()
