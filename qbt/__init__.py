"""qbt —— 默认面向 A 股日频、支持可替换时钟与规则的回测系统。

给研究员用的入口只有几个：

    import qbt
    d = qbt.load()                       # 一次，之后走缓存
    r = qbt.run(d, 我的策略, start="2021-01-01")
    r = qbt.run_orders(d, 订单策略)          # 直接提交/撤销持久订单
    qbt.report(r)                        # 生成自包含 HTML
    qbt.health(d, 我的策略)               # 回测体检（查前视等低级 bug）

设计取向是"引擎兜底、研究员少写代码"：A 股特有的坑（复权、涨跌停买不到、
停牌、T+1、整手、最低佣金、印花税变动、停牌与退市）都在引擎里处理掉，
目标策略声明想持有什么，订单策略可自行安排委托与退出。

老师那道题，两行就能回答：

    a = qbt.run(d, 策略, limit_rule="off")   # 纸面：假装涨停也买得到
    b = qbt.run(d, 策略)                     # 真实：开盘封板就排不到队
    print(a.compare(b))
"""

from __future__ import annotations

__version__ = "0.4.0"

from .account import Account                          # noqa: F401
from .arrow import ArrowError, read_feather           # noqa: F401
from .broker import Broker                            # noqa: F401
from .calendar import Calendar                        # noqa: F401
from .context import Context                          # noqa: F401
from .costs import Costs                              # noqa: F401
from .data import Dataset, load                       # noqa: F401
from .bars import BarDataset
from .clocks import DailyClock, BarClock, ClockEvent
from .rules import AStockRules, CashAssetRules, CashAccountRules
from .orders import OrderRequest, Order, OrderEvent, OrderFill, Cancel, OpenExecution, BarTouchExecution
from .order_engine import run_orders
from .engine import Result, run, run_events           # noqa: F401
from .actions import CorporateAction, CorporateActions, Settlement, SettlementRecord
from .components import RiskControls, WeightAllocation
from .tradability import MarketMetadata
from .metrics import compare, metrics                 # noqa: F401
from . import presets                                 # noqa: F401
from .strategy import FLAT, HOLD, Target, flat, hold  # noqa: F401
from .tradability import build_market, limit_up_illusion  # noqa: F401
from .universe import Universe                       # noqa: F401
from .markets import MarketProfile, market_profile, SessionCalendar, SessionClock, SettledCashRules, EquityRules, ScheduledCosts, from_market_arrays
from .reference import ReferenceData
from .actions import TaxAssessment
from .fx import FXQuote, FXRates, aggregate_results
from .microstructure import L3Event, L3Result, load_l3_csv, run_l3
from .market_io import load_market_csv
from .research import exit_schedule, run_traced, residual_inventory, data_quality

__all__ = [
    # 数据
    "load", "Dataset", "Calendar", "read_feather", "ArrowError",
    # 回测
    "run", "Result", "compare", "metrics",
    # 策略契约
    "Target", "HOLD", "FLAT", "hold", "flat", "Context", "Universe", "presets",
    # 组件（想自己拼一套的时候用）
    "Costs", "Broker", "Account", "build_market", "limit_up_illusion",
    "__version__",
    "run_events", "MarketMetadata", "CorporateAction", "CorporateActions", "Settlement", "SettlementRecord",
    "RiskControls", "WeightAllocation",
    "run_orders", "BarDataset", "DailyClock", "BarClock", "ClockEvent",
    "AStockRules", "CashAssetRules", "CashAccountRules", "OrderRequest", "Order",
    "OrderEvent", "OrderFill", "Cancel", "OpenExecution", "BarTouchExecution",
    "MarketProfile", "market_profile", "SessionCalendar", "SessionClock", "SettledCashRules",
    "EquityRules", "ScheduledCosts", "from_market_arrays", "ReferenceData",
    "FXQuote", "FXRates", "aggregate_results", "L3Event", "L3Result", "load_l3_csv", "run_l3",
    "TaxAssessment",
    "load_market_csv",
    "exit_schedule", "run_traced", "residual_inventory", "data_quality",
]


# 报告和体检是可选依赖（pandas / plotly 有则更好），所以延迟导入，
# 免得 `import qbt` 因为缺一个画图库就整个挂掉。
#
# 这两个模块**故意**叫 `_report.py` / `_health.py`（带下划线），不叫
# `report.py` / `health.py`。原因是个很容易踩的 Python 坑：`from .report
# import report` 会把子模块对象 `qbt.report` 绑成包的属性、**覆盖掉**同名的
# 函数。于是第一次调用还正常，第二次就 `TypeError: 'module' object is not
# callable`。文件名错开就没有这个冲突了。
def report(*a, **kw):                                 # pragma: no cover
    """生成回测报告。见 `qbt._report`。"""
    from ._report import report as _r
    return _r(*a, **kw)


def health(*a, **kw):                                 # pragma: no cover
    """回测体检（前缀测试查前视、现金非负、被拒单率……）。"""
    from ._health import health as _h
    return _h(*a, **kw)


def evaluate_factor(*a, **kw):
    """自定义 (T,N) 因子的预测评价，需要 pandas 与 scipy；交易评价继续用 run。"""
    from .factor_evaluation import evaluate_factor as _evaluate
    return _evaluate(*a, **kw)
