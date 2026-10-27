"""现成的策略，拿来就能跑。

这些不是"推荐策略"，是**对照组**。写自己的策略时，先跑一个 preset 当基线：
如果自己的策略赢不过一个 20 行的小市值反转，那多半不是策略的问题，是别处
出了问题（前视、成本、宇宙）。

其中 `limit_up_next_day` 是老师那道题的正主：

    「好策略：第一天涨停的股票第二天买，收益率好，but 你买不到；
      if wait till you can buy 收益率负」

它的写法刻意只用了 ≤ t 的信息（`hit_limit_up(lag=0)` 是"t 日收盘时能看到的
t 日涨停"），所以可以直接拿去做 `limit_rule="off"` vs 默认口径的对比。

每个 preset 都是 `f(ctx) -> Target`，参数用闭包或者 `functools.partial` 传：

    s = qbt.presets.momentum(n=20, k=100)
    r = qbt.run(d, s, start="2021-01-01")
"""

from __future__ import annotations

import numpy as np

from .strategy import Target
from .universe import Universe

__all__ = [
    "momentum", "reversal", "small_cap", "low_vol", "high_turnover",
    "limit_up_next_day", "buy_and_hold", "equal_weight_all",
    "limit_down_bounce", "amihud_illiquidity",
    "limit_wait",
]


def _univ(ctx, min_amt=5e7, min_listed=60, **kw):
    return ctx.universe(min_amt=min_amt, min_listed=min_listed, not_st=True,
                        exclude_suspended=True, **kw)


def _rebal(ctx, sel, keep, every):
    """把"每天重选"改成"每 `every` 个交易日重选一次"（`every<=1` 原样返回）。

    这是**降换手最有效的一招**，而且完全因果：调仓日由**绝对行号**
    （`全局第几个交易日 % every`）决定，在完整 Context 和 `ctx.sub(0, t)` 里
    第 t 行永远是第 t 行，前缀测试照样过；不需要看"跑了多远"。
    （注意不能用"本 Context 的第几行"—— 那个在完整跑和截断跑里不一样，
    一用就变成前视。`ctx.rows()[0] + t` 才是不变量。）

    非调仓日沿用上一次的 sel/keep 一直 forward-fill 到下一次调仓日。效果是
    持仓被强制至少持有 `every` 天 —— 一个 5 日反转信号本来也没有理由每天换。

    为什么这条重要：不做这一步，3176 只票上选 top100 的 5 日反转，**光排名
    噪声**就让选股域每天进出 80%，年化换手 200 倍。费用是资金的 20%+，任何
    alpha 都被吃干净。这不是 bug，是短周期信号的固有成本 —— 报告里要写成
    一行，让人看到。
    """
    every = int(every)
    if every <= 1:
        return sel, keep
    a, b = ctx.rows()
    rebal = (np.arange(a, b) % every) == 0        # (T,) bool，绝对行号取模
    T = rebal.size
    idx = np.where(rebal, np.arange(T), -1)
    np.maximum.accumulate(idx, out=idx)          # 每个 t 对应的最近一次调仓日
    ok = idx >= 0                                # 开头还没遇到调仓日的那几天
    idx = np.where(ok, idx, 0)
    sel_out = np.asarray(sel, dtype=bool)[idx]
    sel_out[~ok] = False
    keep_out = np.asarray(keep, dtype=bool)[idx]
    keep_out[~ok] = False
    return sel_out, keep_out


def momentum(n=20, k=100, skip=0, min_amt=5e7, buffer=3.0, gross=0.98,
             rebal_every=1):
    """动量：过去 n 日复权收益最高的一批。

    `skip` 是"跳过最近几天的收益"（经典的 12-1 动量里的那个 1，短周期上也有
    意义：最近几天的收益包含太多买卖价差和反转噪声）。`buffer` 是进出缓冲的
    倍数：`add_k=k`、`drop_k=k×buffer`。`rebal_every` 见 `_rebal` 的说明。
    """
    add_k = int(k)
    drop_k = int(round(k * buffer))

    def _s(ctx):
        u = _univ(ctx, min_amt=min_amt)
        r = ctx.ret(n, lag=1 + skip) - (ctx.ret(skip, lag=1) if skip else 0.0)
        ub = u.buffer(r, add_k=add_k, drop_k=drop_k)
        sel, keep = _rebal(ctx, ub.mask, ub.keep, rebal_every)
        return ctx.long(Universe(sel, keep, ctx=ctx), gross=gross,
                        note=f"{n}日动量 top{add_k}（缓冲到 {drop_k}"
                             + (f"，{rebal_every} 日调仓" if rebal_every > 1 else "")
                             + "）")

    _s.__name__ = f"momentum{n}_{add_k}"
    return _s


def reversal(n=5, k=100, min_amt=5e7, buffer=3.0, gross=0.98, rebal_every=5):
    """短期反转：过去 n 日跌得最多的一批。A 股日频上通常比动量有效。

    默认 `rebal_every=5`：信号本身是 5 日的，每天重选只是把排名噪声换成一堆
    手续费。想看"每天重选"有多惨，传 `rebal_every=1` 对比一下，费用会翻好几倍。
    """
    add_k = int(k)
    drop_k = int(round(k * buffer))

    def _s(ctx):
        u = _univ(ctx, min_amt=min_amt)
        r = ctx.ret(n, lag=1)
        ub = u.buffer(-r, add_k=add_k, drop_k=drop_k)
        sel, keep = _rebal(ctx, ub.mask, ub.keep, rebal_every)
        return ctx.long(Universe(sel, keep, ctx=ctx), gross=gross,
                        note=f"{n}日反转（跌得多的）top{add_k}"
                             + (f"，{rebal_every} 日调仓" if rebal_every > 1 else ""))

    _s.__name__ = f"reversal{n}_{add_k}"
    return _s


def small_cap(k=100, min_amt=5e7, buffer=3.0, gross=0.98):
    """小市值。注意：这份数据里**没有市值**，用成交额做代理，所以严格说是
    "成交额最小的一批"——小市值和高换手/低流动性高度重合，但不等价。
    报告里要写清楚是代理，不能含糊过去。
    """
    add_k = int(k)
    drop_k = int(round(k * buffer))

    def _s(ctx):
        u = _univ(ctx, min_amt=min_amt)
        adv = ctx.adv(20, lag=1)                # 20 日均成交额，单位元
        return ctx.long(u.buffer(-adv, add_k=add_k, drop_k=drop_k), gross=gross,
                        note=f"低成交额（小市值代理）top{add_k}")

    _s.__name__ = f"smallcap_{add_k}"
    return _s


def low_vol(k=100, min_amt=5e7, buffer=3.0, gross=0.98):
    """低波动：过去 n 日复权收益标准差最低的一批。"""
    add_k = int(k)
    drop_k = int(round(k * buffer))

    def _s(ctx):
        u = _univ(ctx, min_amt=min_amt)
        v = ctx.vol(60, lag=1)
        return ctx.long(u.buffer(-v, add_k=add_k, drop_k=drop_k), gross=gross,
                        note=f"60日低波动 top{add_k}")

    _s.__name__ = f"lowvol_{add_k}"
    return _s


def amihud_illiquidity(k=100, min_amt=1e8, buffer=3.0, gross=0.98):
    """Amihud 非流动性：|日收益| / 日成交额，越高越不流动。选最不流动的一批。"""
    add_k = int(k)
    drop_k = int(round(k * buffer))

    def _s(ctx):
        u = _univ(ctx, min_amt=min_amt)
        r = np.abs(ctx.dret(lag=1))
        a = ctx.amt(lag=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            illiq = np.where(a > 0, r / a, np.nan)
        return ctx.long(u.buffer(illiq, add_k=add_k, drop_k=drop_k), gross=gross,
                        note=f"Amihud 非流动性 top{add_k}")

    _s.__name__ = f"amihud_{add_k}"
    return _s


def high_turnover(k=100, min_amt=5e7, win=20, buffer=3.0, gross=0.98):
    """高换手（用成交笔数做代理——这份数据没有流通股本）。

    这是个**故意做坏**的对照组：它选的票是市场最热的，涨停概率最高，
    所以"纸面 vs 真实"的差距最大。拿它去演示老师那道题最有说服力。
    """
    add_k = int(k)
    drop_k = int(round(k * buffer))

    def _s(ctx):
        u = _univ(ctx, min_amt=min_amt)
        t = ctx.trades(win, lag=1)
        return ctx.long(u.buffer(t, add_k=add_k, drop_k=drop_k), gross=gross,
                        note=f"高成交笔数（换手代理）top{add_k}")

    _s.__name__ = f"highturn_{add_k}"
    return _s


def limit_up_next_day(k=50, min_amt=5e7, lookback=1, gross=0.98):
    """**老师那道题**：昨天涨停的，今天买。

    信号只用了 ≤ t 的数据（`hit_limit_up(lag=0)` = t 日收盘就能看到的 t 日
    涨停状态），引擎在 t+1 开盘执行 —— 因果上是干净的。

    然后这样跑两遍：

        a = qbt.run(d, s, limit_rule="off")       # 纸面：假装排得到队
        b = qbt.run(d, s)                         # 真实：开盘封板就买不到
        print(a.compare(b, "纸面", "真实"))

    `k` 是每天最多买几只（涨停的票可能一天几百只，不设上限的话持仓会失控）。
    `lookback=1` 表示看"昨天涨停"，=2 就是"前天涨停"。
    """
    kk = int(k)

    def _s(ctx):
        u = _univ(ctx, min_amt=min_amt)
        hit = ctx.hit_limit_up(lag=lookback - 1)     # t 日（或 t-1 日）的涨停状态
        # 涨停的票每天可能很多，按成交额取最小的 k 只（成交额小 = 更容易被
        # 资金推动，也是这类"策略"看起来最赚的原因）
        sc = -ctx.adv(20, lag=1)
        return ctx.long(u.where(hit).top(sc, kk), gross=gross,
                        note=f"{lookback}日前涨停，次日买 top{kk}")

    _s.__name__ = f"limit_up_next_day_{kk}"
    return _s


def limit_down_bounce(k=50, min_amt=5e7, gross=0.98):
    """**老师那道题的反面**：昨天跌停的，今天买（抢反弹）。

    跌停的反面不是"买不到"，是"卖不掉"——所以这个策略的真正看点在于它持有
    的票在次日**能不能卖掉**。看报告里【被挡下的卖单】那一块。

    同时它还演示了跌停的另一个性质：跌停当天你**买得到**（封在跌停板上的
    卖单排着队等你接），这一点和涨停正好相反。
    """
    kk = int(k)

    def _s(ctx):
        u = _univ(ctx, min_amt=min_amt)
        hit = ctx.hit_limit_down(lag=0)
        sc = -ctx.adv(20, lag=1)
        return ctx.long(u.where(hit).top(sc, kk), gross=gross,
                        note=f"昨日跌停抢反弹 top{kk}")

    _s.__name__ = f"limit_down_bounce_{kk}"
    return _s


def buy_and_hold(codes, gross=0.98):
    """等权买入并持有指定代码。做账务核对、跑基准时用。"""
    idx = np.asarray(codes, dtype=np.int64)

    def _s(ctx):
        m = np.zeros(ctx.d.N, dtype=bool)
        m[idx] = True
        return ctx.long(m, gross=gross, note=f"等权持有 {idx.size} 只")

    _s.__name__ = "buy_and_hold"
    return _s


def equal_weight_all(min_amt=5e7, min_listed=60, gross=0.98):
    """全市场等权。最朴素也最难赢过的对照组。"""
    def _s(ctx):
        u = _univ(ctx, min_amt=min_amt, min_listed=min_listed)
        return ctx.long(u, gross=gross, note="全市场等权")

    _s.__name__ = "equal_weight_all"
    return _s


class _LimitWait:
    """事件策略：未成交目标持续等待；持仓计时以真实成交后的账户为准。"""

    def __init__(self, k, max_wait, holding_days, min_amt, min_listed, gross, up):
        if min(k, max_wait, holding_days) < 1:
            raise ValueError("k/max_wait/holding_days 必须 >= 1")
        self.k, self.max_wait, self.holding_days = int(k), int(max_wait), int(holding_days)
        self.min_amt, self.min_listed, self.gross, self.up = min_amt, min_listed, gross, up
        self.last_t = -1
        self.__name__ = "涨停等待买入" if up else "跌停反弹等待买入"

    def __call__(self, ctx):
        if not hasattr(ctx, "portfolio"):
            raise ValueError("limit_wait 是事件策略，请使用 qbt.run_events")
        t = ctx.rows()[1]
        if self.last_t < 0 or t <= self.last_t:
            self.pending = np.full(ctx.N, -1, int)
            self.entered = np.full(ctx.N, -1, int)
            self.previous_held = np.zeros(ctx.N, bool)
        held = ctx.portfolio["shares"] > 0
        newly_held = held & ~self.previous_held
        self.entered[newly_held] = t - 1
        self.entered[~held] = -1
        self.pending[held] = -1
        self.pending[(self.pending >= 0) & (t - self.pending >= self.max_wait)] = -1
        signal = (ctx.hit_limit_up() if self.up else ctx.hit_limit_down())[-1]
        recent = ctx._raw("amt")[-20:]
        amt = np.nansum(recent, axis=0) / np.maximum(np.isfinite(recent).sum(axis=0), 1)
        eligible = (amt >= self.min_amt) & (ctx.age()[-1] >= self.min_listed) & ~ctx.st()[-1]
        eligible &= ~ctx.suspended()[-1]
        new = signal & eligible & ~held & (self.pending < 0)
        self.pending[new] = t
        live = held & ((t - self.entered) < self.holding_days)
        waiting = np.flatnonzero((self.pending >= 0) & eligible)
        slots = max(self.k - int(live.sum()), 0)
        chosen = waiting[np.argsort(-amt[waiting], kind="stable")[:slots]]
        sel = live.copy()
        sel[chosen] = True
        self.last_t, self.previous_held = t, held.copy()
        return Target(sel=sel, gross=self.gross,
                      note=f"{'涨停' if self.up else '跌停'}信号：最多等待 {self.max_wait} 日，实际买入后持有 {self.holding_days} 日")

    def reset(self):
        self.last_t = -1


def limit_wait(k=50, max_wait=5, holding_days=5, min_amt=5e7,
               min_listed=60, gross=.98, up=True):
    """配合 run_events 使用。涨停买不到会继续排队，超时撤销；跌停退出受同样约束。"""
    return _LimitWait(k, max_wait, holding_days, min_amt, min_listed, gross, up)
