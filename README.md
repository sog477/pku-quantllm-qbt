# qbt 课程回测与因子研究

本仓库分别对应 PKU QuantLLM 第四讲回测 Issue 8 与第五讲因子评价 Issue 9。代码和结果固定为本次提交快照；历史研究结果不代表真实市场盈利能力。

## 提交入口

- [第四讲 回测系统和21组同池执行对照](reports/issue8_upgrade/REPORT.md)
- [第五讲 五日反转与因子评价](reports/factor_homework/REPORT.md)
- [本次运行测试记录](reports/publication/tests.json)
- [研究对照接口及取舍](docs/Issue8思路取舍与接口.md)

Issue 8使用10亿元、2025-01-02至2026-09-30、单边佣金万一和卖出税万五，另列最低佣金、过户费、滑点与容量。Issue 9使用1000万元、2025-01-02至2026-10-16的独立因子和账户研究。两组结果口径不同，不能直接比较收益。

## 安装和课程数据

Python 3.11及以上。核心引擎仅需numpy；评价与研究报告用pandas、scipy和matplotlib。

```bash
python -m pip install -r requirements-factor.txt
```

课程原始行情不重复上传。从[课程Release](https://github.com/aslan9/pku_quantllm/releases/tag/data-backtest-20261016)下载feathers.zip，解压后将七个feather文件放在本项目data目录。压缩包SHA256为5766950f2a105244867b842709cf4bad890e4e11df0eebeeaa4fec42dc05803d。各文件实际哈希随运行记录保留。

```bash
python run_issue8_review.py
python run_factor_homework.py
python build_factor_homework_report.py
```

结果写入reports。HTML可在下载仓库后本地浏览，GitHub页面直接阅读REPORT.md及PNG结果图。Notebook入口为notebooks/Issue8研究对照.ipynb和notebooks/因子作业工作台.ipynb。完整CSV含逐笔交易、未完成目标和残仓，可下载本仓库快照后检查。

```python
import qbt
from strategies.five_day_reversal import five_day_reversal

d = qbt.load()
r = qbt.run(d, five_day_reversal, start="2025-01-02", end="2026-09-30", capital=1e9)
qbt.report(r, path="reports/my_run")
```

## 研究与执行设计

- 收盘信息形成目标，下一交易日开盘尝试成交，T+1、整手、现金、费用、涨跌停和双向容量共同限制。
- 配对基准也通过同一账户和撮合引擎；相同信号日过滤池、调仓频率、资金、成本与预定退出规则。
- 退出计划只读已知交易日历，退出期禁止补买；卖不掉的股票保留，估值和清算现金分别记录。
- 预测层比较IC、RankIC、分层、跨边界标签剔除、HAC与FDR、动态衰减及残余信息；交易层单独核算净收益、换手和增量。
- 原始报价与理论估值分开；IPC读取、权益支付、报价时效及Decimal账本等由专门测试覆盖。

## 已知限制

课程七个面板缺完整官方历史ST、历史行业、退市全集及公司行为明细。默认ST为价格行为推断，不能宣称完全满足真实非ST历史认证。复权再投资仍是近似，日线开盘撮合、容量和滑点不能证明盘口可成交或实盘冲击可控。统计留出已被历史研究接触，属于回顾性验证。残仓估值不等于已变现现金，因子统计显著不等于可交易Alpha。

本次学习同学提交中的数据核验、同池比较、残仓披露和负结果记录方法；没有把同学报告中的收益当作自己的结果。多市场和逐笔模块的额外样例见docs及examples，合成测试不等于真实市场数据全覆盖。
