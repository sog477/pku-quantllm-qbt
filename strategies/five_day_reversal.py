"""作业用 5DR：五日复权收益最低的 k 只，等权只做多，每天重新选股。"""
import numpy as np


def five_day_reversal(ctx):
    """t 收盘计算 P_adj[t]/P_adj[t-5]-1，目标在 t+1 开盘尝试执行。

    参数来自 qbt.run(..., params={...})：k、min_amt、min_listed、gross。
    k 为最多持有数量；五日是回看窗口，不是固定持有五日。
    """
    p = ctx.p
    unknown = set(p) - {"k", "min_listed", "min_amt", "gross"}
    if unknown:
        raise ValueError(f"5DR 不认识的参数：{', '.join(sorted(unknown))}")
    k = p.get("k", 100)
    min_listed = p.get("min_listed", 60)
    min_amt = float(p.get("min_amt", 5e7))
    gross = float(p.get("gross", .98))
    for name, value, lower in (("k", k, 1), ("min_listed", min_listed, 0)):
        if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or int(value) != value or value < lower:
            raise ValueError(f"{name} 必须是 >= {lower} 的整数")
    if not np.isfinite(min_amt) or min_amt < 0:
        raise ValueError("min_amt 必须是非负有限数")
    if not np.isfinite(gross) or gross < 0:
        raise ValueError("gross 必须是非负有限数；仓位上限由账户规则检查")
    days = ctx.dates.astype("datetime64[D]")
    if np.unique(days).size != days.size:
        raise ValueError("5DR 示例按交易日计算，请提供日频行情")

    # 1. 先过滤股票池。使用截至当天已知的信息，交易时的限制由撮合器检查。
    universe = ctx.universe(min_amt=min_amt, min_listed=int(min_listed),
                            not_st=True, exclude_suspended=True)
    # 2. 五个交易日累计收益；lag=0，因为引擎已经延迟到下一日执行。
    return_5d = ctx.ret(5, lag=0)
    # 3. 低收益优先。无足够历史或无有效信号的股票不会入选。
    selected = universe.bottom(return_5d, int(k))
    # 4. 等权目标；不添加持仓缓冲、不强制持有五日。
    return ctx.long(selected, gross=gross, weight="equal",
                    note=f"5DR：五日复权收益最低的{int(k)}只，每日重选，次日开盘尝试成交")
