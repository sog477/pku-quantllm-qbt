"""完整演示：所有数字来自本次运行，报告披露采用的模型。"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import qbt

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')
HERE=Path(__file__).resolve().parent


def rule(title):
    print('\n'+'='*78+'\n'+title+'\n'+'='*78, flush=True)


def find_data(explicit=None):
    for p in (explicit, r'D:\Downloads\feathers', HERE/'data', r'D:\Downloads\feathers.zip'):
        if p and Path(p).exists():
            return str(p)
    raise FileNotFoundError('请用 --data 指定 feathers 目录或压缩包')


def _cost_ladder(d, strat, m, args):
    zero=qbt.Costs(commission=0,min_commission=0,transfer_fee=0,slippage_bps=0,stamp_duty=0)
    cases=[('零成本',dict(costs=zero)),('默认成本',{}),
           ('默认+10bps',dict(costs=qbt.Costs(slippage_bps=10))),
           ('5亿 2% ADV',dict(capital=5e8)),('5亿 0.5% ADV',dict(capital=5e8,max_adv_pct=.005))]
    lines=['口径                  本金         年化       最大回撤      年换手       费用/本金']
    for label, options in cases:
        opts=dict(capital=args.capital,**{k:v for k,v in options.items() if k!='capital'})
        opts['capital']=options.get('capital',args.capital)
        r=qbt.run(d,strat,start=args.start,end=args.end,market=m,**opts)
        mm=r.metrics()
        lines.append(f'{label:<18} {r.capital:>10,.0f} {mm["年化收益"]:>10.2%} {mm["最大回撤"]:>10.2%} '
                     f'{mm["换手率(年)"]:>10.1f} {mm["总费用"]/r.capital:>12.2%}')
    return '\n'.join(lines)


def future_examples():
    def forward(ctx):
        u=ctx.universe(min_amt=5e7,min_listed=60)
        a=ctx.adj()
        f=np.full(a.shape,np.nan)
        f[:-5]=a[5:]/a[:-5]-1
        return ctx.long(u.buffer(f,add_k=50,drop_k=75),gross=.98)
    def time_standardize(ctx):
        r=ctx.ret(5,lag=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            z=(r-np.nanmean(r,axis=0))/(np.nanstd(r,axis=0)+1e-12)
        return ctx.long(ctx.universe(min_amt=5e7,min_listed=60).buffer(-z,add_k=50,drop_k=75),gross=.98)
    def harmless(ctx):
        r=ctx.ret(5,lag=1)
        z=(r-np.nanmean(r))/(np.nanstd(r)+1e-12)
        return ctx.long(ctx.universe(min_amt=5e7,min_listed=60).buffer(-z,add_k=50,drop_k=75),gross=.98)
    def future_weights(ctx):
        u=ctx.universe(min_amt=5e7,min_listed=60).top(ctx.adv(20),50)
        # 选股不变，但最后一天价格决定过去所有日期的个股权重。
        score=np.broadcast_to(np.nan_to_num(ctx.close_raw()[-1],nan=0.),(ctx.T,ctx.N))
        return ctx.long(u,weight='score',score=score,gross=.98)
    return [('未来五日收益',forward,True),('全样本时序标准化',time_standardize,True),
            ('纯排名的全样本标量变换',harmless,False),('选股不变但未来决定权重',future_weights,True)]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data')
    ap.add_argument('--start',default='2021-01-04')
    ap.add_argument('--end',default=None)
    ap.add_argument('--capital',type=float,default=1e7)
    ap.add_argument('--quick',action='store_true')
    args=ap.parse_args()
    t0=time.perf_counter()
    rule('1. 加载课程数据')
    d=qbt.load(find_data(args.data))
    print(d)
    m=qbt.build_market(d)
    common=dict(start=args.start,end=args.end,market=m,capital=args.capital)

    rule('2. 数据事实与模型假设')
    for key in ('suspended_cells','suspended_zero_amt','open_at_up','open_at_down','one_word_up','no_limit_cells','st_source'):
        print(f'{key}: {m.stats.get(key)}')
    print('ST 使用前一收盘可推断状态；无限制日需公告元数据；课程数据使用再投资近似。')
    print('容量采用过去 ADV，买卖对称；长期无成交只减记估值，不生成现金。')

    rule('3. 涨停/跌停的条件收益分解')
    print(qbt.limit_up_illusion(d,m))
    print(qbt.limit_up_illusion(d,m,up=False))
    print('开盘到收盘为条件估值收益，未扣成本，也不保证买得到或当天可卖。')

    rule('4. 涨跌停成交口径与持续等待')
    for label,strategy in [('涨停次日',qbt.presets.limit_up_next_day(k=100)),
                           ('跌停反弹',qbt.presets.limit_down_bounce(k=100))]:
        optimistic=qbt.run(d,strategy,limit_rule='off',name=label,**common)
        realistic=qbt.run(d,strategy,name=label,**common)
        print(optimistic.compare(realistic,'关闭限价的乐观对照','默认开盘模型'))
    waiting=qbt.run_events(d,qbt.presets.limit_wait(k=50,max_wait=5,holding_days=5),name='涨停后等待可买',**common)
    print(waiting)
    print('等待天数按未成交目标计算；持有天数从实际成交开始，退出仍受跌停和容量限制。')

    rule('5. 自定义策略、报告、成本和调仓频率')
    strat=qbt.presets.low_vol(k=100)
    r=qbt.run(d,strat,name='60日低波动 top100',**common)
    print(r)
    for assumption in r.assumptions:
        print('  '+assumption)
    report=qbt.report(r,path=str(HERE/'reports'))
    print('报告：'+report.path)
    print(_cost_ladder(d,strat,m,args))
    for every in (1,5,10):
        result=qbt.run(d,qbt.presets.reversal(n=5,k=100,rebal_every=every),**common)
        mm=result.metrics()
        print(f'反转每{every}日调仓：年化 {mm["年化收益"]:+.2%}，年换手 {mm["换手率(年)"]:.1f}，费用/本金 {mm["总费用"]/args.capital:.2%}')

    rule('6. 回测体检')
    health=qbt.health(d,strat,prefix_probes=6,quick=args.quick,**common)
    print(health)
    if not health.ok:
        raise AssertionError('基线体检未通过，请检查失败项')

    rule('7. 前视反例：选股、权重及无害变换')
    for label,strategy,expected in future_examples():
        check=qbt.health(d,strategy,prefix_probes=6,quick=True,**common)
        caught=bool(check.summary['前视失败点'])
        print(f'{label}：前视检查 {"捕获" if caught else "通过"}，截断点 {check.summary["前视失败点"]}')
        if caught != expected:
            raise AssertionError(f'{label} 的前缀检查与预期不符')
    print('有限探针不能证明不存在所有前视；组件和公司行动另见 examples/组件与公司行动.py。')
    rule(f'全部完成，总耗时 {time.perf_counter()-t0:.1f}s')


if __name__=='__main__':
    main()
