# 5DR 五日反转策略：代码输入与回测

策略源代码在 `strategies/five_day_reversal.py`，可直接编辑。`run_strategy.py` 接受外部 Python 文件及函数名，加载后传给 qbt.run；也支持导入模块。这里的“代码输入”是运行本地 Python 策略文件，没有额外的代码编辑网页。

## 策略定义

在交易日 t 收盘后计算每只股票的五日累计复权收益：

`return_5d[t] = adjusted_close[t] / adjusted_close[t-5] - 1`

其中 `adjusted_close = close_ff * cumadj_ff`。在先过滤的股票池内按收益从低到高，选择最多 k 只股票，等权只做多。每天重新计算选股目标，t+1 开盘尝试执行，实际成交继续受到涨跌停、停牌、T+1、整手、费用和容量限制。

五日指信号的回看窗口；本示例没有“买入后固定持有五日”的条件，也不强制隔五日才重选。当所有股票五日收益都为正时，选的是上涨最少的股票；不会额外要求收益为负。

默认参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| k | 100 | 目标最多选100只；因无法退出的旧持仓等原因，实际持仓可能不同 |
| min_amt | 50000000 | 截至信号日的20个交易日平均成交额最低5000万元；至少10个有效观测 |
| min_listed | 60 | 最低报价年龄60个交易日；这是数据中的上市年龄代理 |
| gross | 0.98 | 目标总仓位98%，留部分现金用于费用与取整 |

股票池同时排除信号日已知的ST与停牌状态。课程七面板没有完整真实ST和上市日期，继续使用引擎披露的推断/报价代理；没有指数数据，报告基准是全市场等权，不是沪深300。

本示例使用 `ctx.ret(5, lag=0)`：引擎已经在下一日执行，不再多滞后一天。没有使用持仓缓冲；已有票的股数再平衡沿用引擎默认2%的权重漂移带。资金/容量不足或不能退出时，不会保证实际等权，也不保证下一日完全完成目标。

## 命令行输入策略

在 PowerShell 中运行：

```powershell
cd .
python run_strategy.py --strategy strategies/five_day_reversal.py:five_day_reversal --data D:\Downloads\feathers --start 2022-01-04 --end 2025-06-30 --param k=100 --param gross=0.98 --health
```

也可直接 `python run_strategy.py`：默认加载此5DR文件和项目data。参数使用可重复的 `--param 名称=JSON值`；例如 `--param k=50 --param min_amt=100000000`。这四个参数之外的拼写会报错。

每次输出在 `reports/5dr/run_.../` 独立子目录：HTML报告、`trades.csv`实际成交流水、`summary.json`参数/策略源码和已加载代码SHA256/指标及假设。终端打印本次准确路径，同秒并行实验也不共用产物。带 `--health` 时另存 `health.txt`；体检不通过返回非零退出码。有限前缀体检不能证明策略不存在所有前视。

## 自己编写或替换策略

```python
import qbt
from strategies.five_day_reversal import five_day_reversal

d = qbt.load(r"D:\Downloads\feathers")
r = qbt.run(d, five_day_reversal,
            params={"k": 100, "gross": .98},
            start="2022-01-04", end="2025-06-30", capital=1e7)
qbt.report(r, path="reports/5dr")
```

自己新建 `strategies/my_strategy.py`，定义 `def my_strategy(ctx): ...` 并返回 `ctx.long(...)` 或其他 Target，然后运行：

```powershell
python run_strategy.py --strategy strategies/my_strategy.py:my_strategy
```

文件路径也可为绝对路径；模块形式是 `--strategy strategies.five_day_reversal:five_day_reversal`。这个命令行入口调用面板式目标策略；直接订单策略可使用上一版的 `qbt.run_orders` API。

## 与原有 reversal 预设的区别

原来的 `qbt.presets.reversal(n=5)` 使用 `lag=1`，并默认带排名缓冲和五日重选。本次新增独立的作业基准，不改原预设，从而保留已有实验的含义。以后比较策略结果时，需要把信号延迟、股票池、重选频率、持仓缓冲与费用口径一并说明。

验收命令：`python tests/test_5dr.py`。测试通过独立小行情核对收益方向、五日窗口、下一日开盘成交、每日换股、复权、先过滤后排名和前缀一致性；另检查从用户Python文件和模块加载策略。

## 课程数据的实测结果

2026-10-19 使用上述默认参数、初始资金1000万元，对2022-01-04至2025-06-30进行回测（净值基点是前一交易日2021-12-31）：累计收益 **-62.01%**，年化 **-24.26%**，最大回撤 **-74.46%**。实际成交83,078笔，总费用约283.93万元，占初始资金28.39%；报告的年化双边换手约235.91倍。

这是一份不带排名缓冲的每日反转基准，回测亏损不应通过更改既定策略含义来掩盖。高换手和费用是实测现象；尚未作零费用等归因对照，不能把全部亏损归因于费用。此结果用于作业实现验证，不是对盈利能力的承诺。

专项测试共9项通过；真实行情的8个抽查前缀、目标形状与现金检查通过，体检使用 quick=True，未运行额外成本/容量敏感性回测。完整报告见 `reports/5dr/five_day_reversal_20250630.html`，实际参数、指标与源码哈希在同目录 summary.json；验证日志保存在 `reports/validation_5dr_20261019/`。
