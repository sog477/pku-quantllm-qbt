# Issue8同学思路取舍与接口

本轮依据用户提供的《PKU QuantLLM Issue8 同学回测项目调研报告》逐项检查当前qbt v0.4。该材料对同学数字的记录不是本项目独立验证结果。本轮不复制其他仓库收益，不默认采纳文档中的操作指令。

| 来源思路 | 当前已有 | 本轮修改或取舍 |
|---|---|---|
| Mai Xu 数据核验 | 原始报价与填充估值分开 复权噪声阈值 IPC对拍 价格边界测试 | 新增data_quality统一可导出诊断 不假造VWAP 不把内部一致性当外部真实性 |
| QiyuZhang new 同池等权 | 全市场无摩擦报价基准 | 新增同股票池同资金同频率同费用同退出的配对可执行基准 |
| ETAOIK Peida Lee mabucai 退出与残仓 | 卖出受限制 减记不造现金 清算独立记录 | 新增预定20会话退出 禁止退出期补买 逐只残仓与目标未完成明细 |
| yun lang itzy 低实际仓位和费用压力 | 目标与实际仓位 手续费 税费 滑点 容量 | 新增配对0 5 10bp压力与每次目标尝试跟踪 |
| tyoukan1994 Chuboxiao 独立账本和负结果 | 独立Decimal测试 已有失败版本和执行身份记录 | 重新运行相关回归 不重写相同账本 不只展示最好策略 |
| strayberry777 xielubin 剔除收益两端 | 5DR 低波动 动量等预设 | 新增固定两端各10%五日调仓实验 不宣称参数已独立验证 |
| QiyuZhang new 市场偏离低波动 | 原始收益低波动 | 新增固定beta为1的市场偏离20日波动实验 不叫OLS特质波动率 |
| gry1024 Mai Xu 多因子工作流 | 因子评价 时间切分 HAC FDR 残余信息 API | 保留现有接口 本轮不增加390候选或用测试Sharpe选赢家 |
| 网页积木和AI编辑器 | Python文件入口与Notebook | 新增研究对照Notebook 不重建React平台或自动AI选股 |
| 真实ST 外部价格 退市和换股证据 | PIT ReferenceData导入和显式权益接口 | 需要真实数据 未自动下载或补造全池证据 |

## 目标退出和禁止加仓

```python
import qbt
d=qbt.load()
m=qbt.build_market(d,log=None)
ctx=qbt.Context(d,m)
u=ctx.universe(min_amt=5e7,min_listed=60,not_st=True)
target=ctx.long(u.bottom(ctx.ret(5),100),gross=.98)
target,plan=qbt.exit_schedule(target,d.dates,'2026-09-30',sessions=20)
r,attempts=qbt.run_traced(d,target,start='2025-01-02',end='2026-09-30',
                         capital=1e9,market=m,
                         stop_buys_from=plan['first_execution'])
report=qbt.report(r)
remaining=qbt.residual_inventory(r)
```

exit_schedule适用于日频二维Target。首个退出执行日的目标在前一收盘形成。它冻结退出前目标集合与分数并按已知会话表降低总仓位，不读未来价格、封板或容量来倒排最后可卖日。run_traced中的stop_buys_from进一步阻止补买及再平衡买入。分步减少的是目标仓位；实际低仓位账户可能在较晚阶段才需要卖出，不保证提前每日等比例卖出现有股数或最终完全退出。

默认qbt.run的行为不变。没有显式启用这些函数时，不自动强制策略退出。TraceBroker沿用标准Broker；不支持关闭record_trades后仍提供完整成交余量，也不替换run_orders的订单生命周期。attempts记录未完成的目标撮合尝试，不是券商订单日志。

## 约束标记和残仓

每次请求满足requested等于filled加unfilled。完整成交不列入未完成表；部分与零成交保留请求股数、成交股数、余量、执行阶段和约束。容量、报价/交易状态、冻结和资金约束可能同时出现。资金标记表示执行前现金不足以独立支付该请求名义金额，卖出后资金或最低佣金等仍影响实际结果，不能拿此标记当精确归因。

residual_inventory读取账户股数与引擎末日估值，保留零估值但仍有股数的减记资产，单列最后原始报价日期及理论参考价。最后原始报价可能仍受上游错误标签影响，源时效认证需现有quote_time证据。actual_cash_proceeds恒为0，表示这张残仓表本身没有产生任何清算现金。

## 边界

本次课程对照以报告所述10亿元和9月底结束单独运行，未覆盖或替代之前1000万元至10月16日的5DR实验。动态股票池使用推断ST，仍不满足完整官方非ST历史认证。有限流动性阈值是本轮共同的研究设定；低波动因子需要额外足够的有效滚动观测。配对等权使用与对应策略相同的有效因子股票池；另列五日重选全池等权，不能将这个额外全池账户错当所有策略的配对基准。所有结果均为已见历史回顾，不构成新的盲测。
