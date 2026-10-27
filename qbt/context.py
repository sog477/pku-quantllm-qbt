"""Context：策略函数拿到的那个对象。

研究员的全部工作就是"从 `ctx` 拿面板，拼一个 `Target` 出来"。所以这里的
设计目标只有一个：**常用因子的写法要短到不需要查文档**。

    def 五日反转(ctx):
        u = ctx.universe(min_amt=5e7, min_listed=60, not_st=True)
        return ctx.long(u.buffer(ctx.ret(5), add_k=100, drop_k=150))

关于 lag，一句话说清时点关系：

    引擎在第 t 行（收盘后）调策略，信号在第 t+1 行**开盘**成交。

所以只用 `<= t` 的数据就是安全的，`lag=0` 就是"用今天收盘算"。`lag=1` 是
额外再退一天，用来演示"更保守也不会改变结论"。

**前视安全是靠结构保证的，不是靠自觉。** `ctx.sub(a, b)` 返回一个只看得见
`[a, b)` 行的 Context，它内部的每一个派生面板都在这个切片上从头算。于是
"前缀等价"是可以断言的：`ctx.sub(0, t)` 的第 i 行（i < t）必须和完整
Context 的第 i 行**逐位相同**。`health()` 就是拿这个当体检项。

反过来说，任何绕过 `ctx` 直接读 `ctx.d.close`（全长面板）的策略都会在前缀
测试里当场失败——这正是我们想要的。
"""

from __future__ import annotations

import numpy as np

from .strategy import Target
from .tradability import is_main_board
from .universe import Universe
from ._validation import boolean_mask

__all__ = ["Context"]

NEW_DAYS = 60          # `not_new=True` 时要求的最少上市交易日
_GEM_PREFIX = ("300", "301", "302")
_STAR_PREFIX = ("688", "689")
_BSE_PREFIX = ("43", "83", "87", "88", "92")


# --------------------------------------------------------------------------
# 面板小工具
# --------------------------------------------------------------------------
def _shift(a, k):
    """把面板往下挪 k 行：`out[t] = a[t-k]`，上面 k 行置空。

    这是整个系统里唯一能让前视溜进来的地方，所以只有这一个实现、只有这一个
    语义：**只往过去看**。绝不做环绕填充（那是最经典的前视 bug）。

    `k < 0` 直接报错，不返回原值。这一条是踩出来的：早先写成 `if k <= 0:
    return a`，于是 `ctx.ret(5, lag=-5)` —— 一个想"往前多退 5 天"的人很容易
    写出来的东西 —— **静默地变成了 `lag=0`**：跑得通、有结果、没有任何提示，
    只是口径换成了另一个。符号写错就换一套口径，是回测里最贵的一类静默 bug。
    """
    if k < 0:
        raise ValueError(
            f"lag={k} 是在看未来，本引擎不支持（_shift 只往过去看）。"
            f"想做前视对照组请直接索引数据面板，别走 ctx 的因子函数。")
    if k == 0:
        result = np.asarray(a).view()
        result.setflags(write=False)
        return result
    out = np.empty_like(a)
    if a.dtype == np.bool_:
        out[:k] = False
    elif a.dtype.kind == "f":
        out[:k] = np.nan
    else:
        out[:k] = -1
    out[k:] = a[:-k]
    return out


def _roll_sum(a, w):
    """按列的 w 日滚动和 + 窗口内有效观测数。用 cumsum，O(T·N)。"""
    a = np.asarray(a, dtype=np.float64)
    ok = np.isfinite(a)
    x = np.where(ok, a, 0.0)
    T, N = a.shape
    z = np.zeros((1, N))
    cs = np.concatenate([z, np.cumsum(x, axis=0)], axis=0)
    cn = np.concatenate([z, np.cumsum(ok, axis=0)], axis=0)
    out_s = np.full((T, N), np.nan)
    out_n = np.zeros((T, N))
    out_s[w - 1:] = cs[w:] - cs[:-w]
    out_n[w - 1:] = cn[w:] - cn[:-w]
    return out_s, out_n


def _roll_mean(a, w, min_obs=None):
    """w 日滚动均值。窗口里有效观测太少就出 NaN，不拿 0 去充数。"""
    if min_obs is None:
        min_obs = max(1, w // 2)
    s, n = _roll_sum(a, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= min_obs, s / np.maximum(n, 1.0), np.nan)


def _roll_std(a, w, min_obs=None):
    """w 日滚动标准差（总体口径，E[x²]-E[x]²）。"""
    if min_obs is None:
        min_obs = max(2, w // 2)
    s, n = _roll_sum(a, w)
    s2, _ = _roll_sum(a * a, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / np.maximum(n, 1.0)
        var = np.maximum(s2 / np.maximum(n, 1.0) - m * m, 0.0)
        return np.where(n >= min_obs, np.sqrt(var), np.nan)


# --------------------------------------------------------------------------
class Context:
    """策略函数唯一的输入。

    `ctx.p` 是参数字典——同一个策略函数可以扫参，不用改代码。
    `ctx.d` / `ctx.m` 分别是原始数据和市场状态，需要更细的东西时直接用。
    """

    def __init__(self, ds, market, params=None, base=0, n=None):
        self.d = ds
        self.m = market
        self.p = dict({} if params is None else params)
        self._base = int(base)                       # 本 Context 第 0 行在全局面板里的行号
        self._n = ds.T - self._base if n is None else int(n)
        self._cache = {}
        self.periods_per_year = 242.

    # -- 自省 ------------------------------------------------------------
    @property
    def T(self):
        return self._n

    @property
    def N(self):
        return self.d.N

    @property
    def dates(self):
        dates = self.d.dates[self._base:self._base + self._n].view()
        dates.setflags(write=False)
        return dates

    @property
    def codes(self):
        return list(self.d.codes)

    @property
    def calendar(self):
        return self.d.calendar

    def day(self, t):
        """本 Context 的第 t 行对应的交易日。"""
        return self.d.dates[self._base + t]

    def rows(self):
        """本 Context 覆盖的全局行号范围 `(base, base+n)`。"""
        return self._base, self._base + self._n

    def __repr__(self):
        return (f"Context({self.T} 日 × {self.N} 只, "
                f"{self.dates[0]} .. {self.dates[-1]}, "
                f"参数={self.p if self.p else '{}'})")

    # -- 行切片（前缀测试用）---------------------------------------------
    def sub(self, a, b=None):
        """返回只看得见 `[a, b)` 行的 Context（相对本 Context 的行号）。

        这个 Context 里的面板**全部在这个切片上从头算**，所以它绝对看不见
        切片之前的数据。`ctx.sub(0, t)` 的第 i 行（i < t）与完整 Context 的
        第 i 行逐位相同——这就是前缀等价，也是查前视的唯一可靠手段。
        """
        b = self.T if b is None else int(b)
        a = int(a)
        if not (0 <= a <= b <= self.T):
            raise ValueError(f"sub({a}, {b}) 越界（本 Context 共 {self.T} 行）")
        result = Context(self.d, self.m, self.p, base=self._base + a, n=b - a)
        result.periods_per_year = self.periods_per_year
        return result

    def _raw(self, name):
        """本切片范围内的原始面板。"""
        observed = getattr(self.m,'observed_price',None)
        if name in ('close','open','close_ff','open_ff') and callable(observed):
            v = observed(name.removesuffix('_ff'),name.endswith('_ff'))
        elif name == 'list_age' and callable(getattr(self.m,'observed_age',None)):
            v = self.m.observed_age()
        else:
            v = getattr(self.d, name)
        if self._base == 0 and self._n == v.shape[0]:
            return v
        return v[self._base:self._base + self._n]

    def _p(self, key, build):
        v = self._cache.get(key)
        if v is None:
            v = build()
            self._cache[key] = v
        return v

    # -- 价格 ------------------------------------------------------------
    def _signal_adj(self):
        if hasattr(self.d,'total_return_close'):
            return np.asarray(self._raw('total_return_close'),dtype=float)
        return np.asarray(self._raw('close_ff'),dtype=float)*np.asarray(self._raw('cumadj_ff'),dtype=float)

    def adj(self, lag=0):
        """后复权收盘价 `close_ff × cumadj_ff`。

        复权方向是 **乘**，`adj = raw × cumadj`，全量数据验证过：除权日
        |收益| 中位数从不复权的 2.53% 降到 1.29%，改用除法反而升到 3.94%。
        做反了是个**静默的灾难性 bug**：回测照跑，结果全错。
        """
        return _shift(self._p("adj", self._signal_adj), lag)

    def close_raw(self, lag=0):
        """未复权收盘价，停牌处是 NaN。判涨跌停、看绝对价格用它。"""
        return _shift(self._raw("close"), lag)

    def close(self, lag=0):
        """前向填充收盘价。算持仓市值用它——停牌不能把净值变成 NaN。"""
        return _shift(self._raw("close_ff"), lag)

    def open(self, lag=0):
        return _shift(self._raw("open_ff"), lag)

    # -- 常用因子 --------------------------------------------------------
    def ret(self, n=1, lag=0):
        """滞后 lag 日的 n 日收益率（复权价）。

            ctx.ret(5)      最近 5 个交易日收益
            ctx.ret(5, 1)   再退一天（更保守的写法）
            ctx.ret(1)      日收益

        第 t 行是 `adj[t-lag] / adj[t-lag-n] - 1`。
        """
        if n < 1:
            raise ValueError("n 要 >= 1；日频数据里没有日内收益")
        a = self._p("adj", self._signal_adj)
        out = np.full(a.shape, np.nan, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[n:] = a[n:] / a[:-n] - 1.0
        return _shift(out, lag)

    def amt(self, lag=0):
        """当日成交额（元）。"""
        return _shift(np.asarray(self._raw("amt"), dtype=np.float64), lag)

    def adv(self, win=20, lag=0):
        """win 日平均成交额（元）——流动性过滤的标准口径。

        用滚动均值而不是当日值：单日成交额噪声大，一天放量就能把一只烂票
        放进 universe，容量约束就形同虚设。
        """
        a = self._p(f"adv{win}", lambda: _roll_mean(self._raw("amt"), win))
        return _shift(a, lag)

    def vol(self, win=20, lag=0, annualize=True):
        """win 日已实现波动率（默认年化，242 个交易日/年）。"""
        a = self._p(f"vol{win}", lambda: _roll_std(self._dret(), win))
        if annualize:
            a = a * np.sqrt(self.periods_per_year)
        return _shift(a, lag)

    def rtrades(self, lag=0):
        """当日成交笔数。"""
        return _shift(np.asarray(self._raw("num_trades"), dtype=np.float64), lag)

    def trades(self, win=20, lag=0):
        """win 日平均成交笔数 —— **换手率的代理**。

        这份数据里没有流通股本，算不出真正的换手率。成交笔数的绝对水平是可比
        的（订单流的一些结论是"笔数多 = 散户多 = 情绪票"），所以用它做代理，
        报告里必须写明这是代理而不是换手率本身。
        """
        a = self._p(f"ntr{win}", lambda: _roll_mean(self._raw("num_trades"), win))
        return _shift(a, lag)

    def _dret(self):
        def build():
            a = self._p("adj", self._signal_adj)
            out = np.full(a.shape, np.nan, dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[1:] = a[1:] / a[:-1] - 1.0
            return out
        return self._p("dret", build)

    def dret(self, lag=0):
        """日收益率（复权）。"""
        return _shift(self._dret(), lag)

    def age(self, lag=0):
        """上市交易日数；-1 = 还没上市，32767 = 数据开始前就是老股。"""
        return _shift(self._raw("list_age"), lag)

    # -- 市场状态（老师的那些坑，全部现成）--------------------------------
    def st(self, lag=0):
        """ST 状态（推断值，见 `tradability.infer_st` 里的诚实说明）。"""
        return _shift(self.m.st[self._base:self._base + self._n], lag)

    def suspended(self, lag=0):
        """停牌：没有报价，或者全天零成交。两种表示都算了。"""
        return _shift(self.m.suspended[self._base:self._base + self._n], lag)

    def _mpanel(self, name, lag):
        return _shift(getattr(self.m, name)[self._base:self._base + self._n], lag)

    def limit_up_price(self, lag=0):
        """当日涨停价。会随除权除息调整（`ref = 昨收×昨因子/今因子`）。"""
        return self._mpanel("limit_up", lag)

    def limit_down_price(self, lag=0):
        return self._mpanel("limit_down", lag)

    def band(self, lag=0):
        """当日板幅（百分点）：0/5/10/20/30。**必须按股按日看**——创业板
        2020-08-24 注册制改革当天从 10% 抬到 20%，ST 是 5%，新股前 5 天
        没有限制。"""
        return self._mpanel("band", lag)

    def hit_limit_up(self, lag=0):
        """收盘涨停。`lag=1` 就是老师那句"第一天涨停"。"""
        return self._mpanel("close_at_up", lag)

    def hit_limit_down(self, lag=0):
        return self._mpanel("close_at_down", lag)

    def open_at_limit_up(self, lag=0):
        """开盘就涨停——你排不到队的那批。"""
        return self._mpanel("open_at_up", lag)

    def open_at_limit_down(self, lag=0):
        return self._mpanel("open_at_down", lag)

    def no_limit(self, lag=0):
        """当天没有涨跌幅限制（复牌首日 / 注册制新股前 5 日 / 退市整理期）。"""
        return self._mpanel("no_limit", lag)

    def one_word(self, lag=0):
        """一字板（开=高=低=收=涨停价）。数据集里 8,171 格，
        **没有一格 amt==0**——封板当天流动性看着很足，你照样买不到。"""
        s = slice(self._base, self._base + self._n)
        m = self.m
        lu = np.round(m.limit_up[s].astype(np.float64) * 100)
        ow = (m.open_at_up[s] & m.close_at_up[s]
              & (np.abs(np.round(m.ds.high[s].astype(np.float64) * 100) - lu) < 0.5)
              & (np.abs(np.round(m.ds.low[s].astype(np.float64) * 100) - lu) < 0.5))
        return _shift(ow, lag)

    def blocked_bounce(self, lag=1, side="buy"):
        """**诊断用（全系统最有说服力的一个数）**。

        t 日开盘就封板、你根本排不到队的那些票，从 t 日**开盘价**到 t 日收盘
        到底涨了多少。老师那句"第一天涨停的股票第二天买，收益率好，but 你
        买不到"，买不到的那部分收益就在这个面板里。

        `qbt.compare(纸面, 真实)` 会把它按笔加权平均出来，直接给出"纸面净值
        与真实净值的差，有多少是被涨停挡掉的"。
        """
        r = self.bounce_next_day()
        return _shift(r, lag)

    def bounce_next_day(self):
        """开盘到收盘的当日收益，只在开盘封板处有值。"""
        o = np.asarray(self._raw("open_ff"), dtype=np.float64)
        c = np.asarray(self._raw("close_ff"), dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = c / o - 1.0
        flag = self.m.open_at_up[self._base:self._base + self._n]
        return np.where(flag, r, np.nan)

    def bounce_at_limit_down(self):
        """开盘跌停、卖不掉的票，当日开盘到收盘的收益（应该更负）。"""
        o = np.asarray(self._raw("open_ff"), dtype=np.float64)
        c = np.asarray(self._raw("close_ff"), dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = c / o - 1.0
        flag = self.m.open_at_down[self._base:self._base + self._n]
        return np.where(flag, r, np.nan)

    # -- 横截面变换 ------------------------------------------------------
    def rank(self, x, mask=None):
        """逐日横截面排名，归一化到 (0, 1]，1 = 最大。"""
        a = np.asarray(x, dtype=np.float64)
        T, N = a.shape
        out = np.full((T, N), np.nan, dtype=np.float64)
        m_all = None if mask is None else boolean_mask(mask)
        for t in range(T):
            row = a[t]
            m = np.isfinite(row)
            if m_all is not None:
                m = m & m_all[t]
            idx = np.flatnonzero(m)
            n = idx.size
            if n == 0:
                continue
            order = np.argsort(row[idx], kind="stable")
            r = np.empty(n, dtype=np.float64)
            r[order] = np.arange(1, n + 1, dtype=np.float64)
            out[t, idx] = r / n
        return out

    def zscore(self, x, mask=None):
        """逐日标准化。无效值不参与统计，也不会被算成 0。"""
        a = np.asarray(x, dtype=np.float64)
        if mask is not None:
            a = np.where(boolean_mask(mask), a, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.nanmean(a, axis=1, keepdims=True)
            sd = np.nanstd(a, axis=1, keepdims=True)
            return (a - mu) / np.where(sd > 0, sd, np.nan)

    def winsor(self, x, p=0.01, mask=None):
        """逐日按分位数截尾（默认上下 1%）。"""
        a = np.asarray(x, dtype=np.float64)
        if mask is not None:
            a = np.where(boolean_mask(mask), a, np.nan)
        with np.errstate(invalid="ignore"):
            lo = np.nanquantile(a, p, axis=1, keepdims=True)
            hi = np.nanquantile(a, 1.0 - p, axis=1, keepdims=True)
        return np.clip(a, lo, hi)

    def demean(self, x, mask=None):
        """逐日去均值。"""
        a = np.asarray(x, dtype=np.float64)
        if mask is not None:
            a = np.where(boolean_mask(mask), a, np.nan)
        with np.errstate(invalid="ignore"):
            return a - np.nanmean(a, axis=1, keepdims=True)

    # -- universe --------------------------------------------------------
    def universe(self, min_amt=0.0, amt_win=20, min_listed=0, not_st=True,
                 not_new=False, new_days=NEW_DAYS, exclude_suspended=True,
                 min_price=None, max_price=None, boards=None, extra=None,
                 lag=0, note=""):
        """筛出可投资的股票域。

        每个参数对应一个真实的坑：

        min_amt        日均成交额下限（元）。不做容量约束的策略回测基本是假的。
        min_listed     上市满多少个交易日。新股前 5 天没有涨跌幅限制，而且
                       次新股的价格行为和老股完全不是一回事。
        not_st         排除 ST。ST 是 5% 板，混进动量因子会污染排序。
        not_new        等价于 min_listed = max(min_listed, 60)。
        min_price      剔除仙股（价格太低时一个 tick 就是 1%+，成本模型失真）。
        boards         "main" / "gem" / "star" / "bse"，或它们的列表。
        extra          额外的 (T,N) bool 掩码，放任何自定义条件。

        **先筛后选**：拿到 universe 再用 `u.bottom(score, 100)`。不要写成
        `bottom(score, 100) & u`——那是在 5451 只里选完再过筛，实际只剩
        30~60 只，决定权落在掩码上而不是信号上（详见 `universe.py`）。
        """
        T, N = self.T, self.N
        ok = np.ones((T, N), dtype=bool)

        if min_listed or not_new:
            need = max(int(min_listed), new_days if not_new else 0)
            ok &= self.age(lag) >= need
        if not_st:
            ok &= ~self.st(lag)
        if exclude_suspended:
            ok &= ~self.suspended(lag)
        if min_amt:
            ok &= np.nan_to_num(self.adv(amt_win, lag), nan=0.0) >= float(min_amt)
        if min_price is not None:
            ok &= np.nan_to_num(self.close(lag), nan=0.0) >= float(min_price)
        if max_price is not None:
            c = self.close(lag)
            ok &= np.isfinite(c) & (c <= float(max_price))
        if boards is not None:
            ok &= self._board_mask(boards)
        if extra is not None:
            ok &= boolean_mask(extra, 'extra')

        note = note or f"min_amt={min_amt:g} min_listed={min_listed}"
        return Universe(ok, ctx=self, note=note)

    def _board_mask(self, boards):
        if isinstance(boards, str):
            boards = [boards]
        if getattr(self.d,'market_code','CN') in ('US','KR'):
            allowed={'NYSE','NASDAQ','AMEX','UNKNOWN'} if self.d.market_code=='US' else {'KOSPI','KOSDAQ'}
            wanted={str(b).upper() for b in boards}
            if not wanted<=allowed: raise ValueError('board filter does not belong to selected market')
            return np.broadcast_to(np.isin(np.char.upper(self.d.market_boards),list(wanted))[None,:],(self.T,self.N))
        want = {str(b).lower() for b in boards}
        codes = np.asarray(self.d.codes)
        pre = np.array([c[:3] for c in codes])
        main = is_main_board(codes)
        m = np.zeros(len(codes), dtype=bool)
        for b in want:
            if b in ("main", "主板", "sh", "sz"):
                m |= main
            elif b in ("gem", "创业板", "300"):
                m |= np.isin(pre, _GEM_PREFIX)
            elif b in ("star", "科创板", "688"):
                m |= np.isin(pre, _STAR_PREFIX)
            elif b in ("bse", "北交所"):
                m |= np.array([c[:2] in _BSE_PREFIX for c in codes])
            else:
                raise ValueError(f"不认识的 boards={b!r}")
        return np.broadcast_to(m[None, :], (self.T, self.N))

    def all_stocks(self, note="全市场"):
        """不做任何筛选的全市场 universe（对照用）。"""
        return Universe(np.ones((self.T, self.N), dtype=bool), ctx=self, note=note)

    # -- 目标持仓 --------------------------------------------------------
    def long(self, u, gross=1.0, weight="equal", score=None, stops=None, note=""):
        """持有 `u` 里的股票。

        gross   目标总仓位（0~1），其余留现金。用 0.98 而不是 1.0 是留一点
                缓冲，免得整手取整后现金变负。
        weight  "equal" 等权 / "score" 或 "value" 按显式score加权
                例如 score=ctx.amt() 按已完成信号行的成交额加权
        stops   {"hard": 0.10, "trailing": 0.20}，次日开盘评估
        """
        if isinstance(u, Target):
            return u
        if not isinstance(u, Universe):
            u = Universe(u, ctx=self)
        return Target(sel=u.mask, keep=u.keep, gross=np.asarray(gross, dtype=float),
                      weight=weight, score=score, stops=stops,
                      note=note or u.note)

    def cash(self, gross=0.0):
        """空仓（或指定仓位）。"""
        return Target(sel=np.zeros(self.N, dtype=bool), gross=float(gross))
