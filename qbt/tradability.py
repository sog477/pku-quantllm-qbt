"""市场状态与可交易性。默认开盘板幅仅使用开盘前已知的元数据或前一收盘推断。

当日高低收只用于收盘信号、统计与标记为事后对照的撮合模式。
无限制日不根据当日大涨反推，准确历史状态可通过 MarketMetadata 注入。
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

__all__ = ["Market", "build_market", "limit_up_illusion", "infer_st",
           "round_to_cent", "reference_price", "limit_prices", "base_band",
           "is_main_board", "LIMIT_RULES", "TICK"]

TICK = 0.01
_EPS = 1e-9          # 四舍五入的偏置：把 1116.4999999999998 这类表示误差推回半分位

# 板幅档位（百分点）
BAND_NONE, BAND_ST, BAND_MAIN, BAND_GEM, BAND_BSE = 0, 5, 10, 20, 30

# 涨跌停判定用的四种口径（老师那道题的开关）
LIMIT_RULES = ("off", "sealed", "broken_board_close", "open_limit")


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def round_to_cent(x):
    """四舍五入到分。交易所是"四舍五入"，numpy 默认是"四舍六入五成双"，不一样。"""
    return np.floor(np.asarray(x, dtype=np.float64) * 100.0 + 0.5 + _EPS) / 100.0


def _cents(x):
    """换算成整数"分"，避开 0.01 的二进制表示误差。NaN 保留为 NaN。"""
    return np.round(np.asarray(x, dtype=np.float64) * 100.0)


def _eq_price(a, b):
    """两个价格是否相等（到分为止）。NaN 一律 False。"""
    return np.abs(_cents(a) - _cents(b)) < 0.5


# --------------------------------------------------------------------------
# 板幅
# --------------------------------------------------------------------------
_GEM_REFORM = np.datetime64("2020-08-24")
_GEM_PREFIX = ("300", "301", "302")
_STAR_PREFIX = ("688", "689")
_BSE_PREFIX = ("43", "83", "87", "88", "92")


def base_band(codes, dates):
    """基准板幅（不含 ST），(T, N) int8。"""
    T, N = len(dates), len(codes)
    band = np.full((T, N), BAND_MAIN, dtype=np.int8)
    gem_day = dates >= _GEM_REFORM
    for j, c in enumerate(codes):
        p = c[:3]
        if p in _GEM_PREFIX:
            band[:, j] = np.where(gem_day, BAND_GEM, BAND_MAIN)
        elif p in _STAR_PREFIX:
            band[:, j] = BAND_GEM
        elif c[:2] in _BSE_PREFIX:
            band[:, j] = BAND_BSE
    return band


def is_main_board(codes):
    """主板（ST 会降到 5% 的那种）。创业板/科创板 ST 依然是 20%。"""
    out = np.ones(len(codes), dtype=bool)
    for j, c in enumerate(codes):
        p = c[:3]
        if p in _GEM_PREFIX or p in _STAR_PREFIX or c[:2] in _BSE_PREFIX:
            out[j] = False
    return out


# --------------------------------------------------------------------------
# 限价
# --------------------------------------------------------------------------
def reference_price(close, cumadj):
    """除权除息参考价：`prev_close_raw * cumadj_prev / cumadj_today`。

    返回 (T, N)，第 0 行是 NaN（没有前收）。这个公式已用全量数据验证：
    20% 档命中 10325 次**全部**落在创业板/科创板。
    """
    T = close.shape[0]
    ref = np.full_like(close, np.nan, dtype=np.float64)
    ref[1:] = (close[:-1] * cumadj[:-1]) / cumadj[1:]
    return ref


def limit_prices(ref, band):
    """由参考价和板幅算出涨停/跌停价。band==0（无限制）时给 ±inf。"""
    b = band.astype(np.float64) / 100.0
    up = round_to_cent(ref * (1.0 + b))
    dn = round_to_cent(ref * (1.0 - b))
    nolim = band == BAND_NONE
    up[nolim] = np.inf
    dn[nolim] = -np.inf
    return up, dn


# --------------------------------------------------------------------------
# ST 推断
# --------------------------------------------------------------------------
ST_WINDOW = 120     # 回看多少个交易日（约半年）


def _trailing_any(a, w):
    """过去 w 个交易日（含当日）里是否出现过 True。纯 numpy，无时间循环。"""
    c = np.cumsum(a.astype(np.int32), axis=0)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out > 0


def infer_st(close, high, low, cumadj, main_board, log=None, window=ST_WINDOW):
    """推断 ST（主板跌到 5% 涨跌幅）状态，返回 (st 面板, 统计)。

    为什么不能用"价格撞上限价"来判
    ------------------------------
    第一版就是这么写的，结果 5451 只里有 3287 只被判成过 ST —— 完全是噪声。
    实测：收盘价恰好等于 5% 限价的格子有 6.4 万个，分布在 4819 只股票上
    （中位数每只 4 天）。因为价格是分位的，任何一只正常波动的股票迟早会有
    一天恰好收在 +5.00% 那个价位上。**价格撞上某个价位，和价格被这个价位
    挡住，是两件完全不同的事。**

    真正有区分度的是"能不能越过去"
    ------------------------------
    5% 板意味着 |日收益| 永远 <= 5%。所以：

      * 强证据（sealed）：收在 5% 限价上，且当日最高/最低就是那个价 —— 封板
      * 反证（big）：当日 |复权收益| > 5.5% —— 板幅肯定不是 5%

    实测全样本只有 4 只股票从未有过 >5.5% 的日收益，所以单看"从未"太严；
    改成**回看 120 个交易日**的滚动窗口：窗口内出现过封板、且窗口内从没有
    过 >5.5% 的波动 -> 判为 ST。

    正常股票在 120 天里几乎必然有过一次 >5.5% 的波动（日波动 3% 时单日
    概率约 6.7%，120 天里一次都没有的概率约等于 0），所以这个"与"条件
    把噪声清得很干净。

    只回看不前视 —— 这是回测能用的前提。
    """
    T, N = close.shape
    ref = reference_price(close, cumadj.astype(np.float64, copy=False))
    lu5, ld5 = limit_prices(ref, np.full((T, N), BAND_ST, np.int8))

    # 封在 5% 板上：收在限价，且当日极值就是限价
    sealed = (_eq_price(close, lu5) & _eq_price(high, lu5)) | \
             (_eq_price(close, ld5) & _eq_price(low, ld5))
    sealed &= main_board[None, :]

    # 反证：单日波动超过 5.5%，说明当天板幅不是 5%
    c_adj = close * cumadj
    prev = np.vstack([np.full((1, N), np.nan), c_adj[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        big = np.nan_to_num(np.abs(c_adj / prev - 1.0)) > 0.055

    has_seal = _trailing_any(sealed, window)
    has_big = _trailing_any(big, window)
    st = has_seal & ~has_big

    changes = np.zeros(T, dtype=np.int64)
    if T > 1:
        changes[1:] = (st[1:] != st[:-1]).sum(axis=1)
    stats = {
        "st_stocks": int(st.any(axis=0).sum()),
        "st_stock_days": int(st.sum()),
        "st_flips": int(changes.sum()),
        "st_window": window,
        "sealed_5pct_cells": int(sealed.sum()),
        "big_move_cells": int(big.sum()),
    }
    if log:
        log(f"ST 推断（回看 {window} 日）：{stats['st_stocks']:,} 只股票被标过 ST，"
            f"占 {stats['st_stock_days']:,} 个股票日，状态翻转 {stats['st_flips']:,} 次")
    return st, stats


# --------------------------------------------------------------------------
# Market
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketMetadata:
    """开盘前已公布的 (T,N) 面板；必须按 Dataset 日期和代码对齐。

    不提供时 ST 用前一日收盘可推断状态；无限制日不靠当日涨跌幅猜测。
    reference 可填交易所除权参考价，缺失格用累计因子推导的参考价。
    """
    st: object = None
    no_limit: object = None
    suspended: object = None
    reference: object = None
    quote_time: object = None
    max_quote_age: object = None
    limit_up: object = None
    limit_down: object = None


class Market:
    """按日按股的市场状态面板集合。所有数组都是 (T, N)。"""

    __slots__ = ("band", "limit_up", "limit_down", "st", "suspended", "no_limit",
                 "name", "open_at_up", "open_at_down", "close_at_up", "close_at_down",
                 "touched_up", "touched_down", "stats", "ds", "open_suspended",
                 "sealed_up", "sealed_down", "reference", "quote_suspended", "quote_time", "quote_fresh", "_known_prices")

    def __init__(self, **kw):
        self._known_prices = {}
        for k, v in kw.items():
            setattr(self, k, v)

    def observed_price(self, name, forward_fill=False):
        """显式来源约束下，内置价格因子不能重新使用已判无效的历史报价。"""
        if self.stats.get('quote_time_source') != 'explicit':
            return getattr(self.ds,name+'_ff' if forward_fill else name)
        key = (name,forward_fill)
        if key not in self._known_prices:
            raw = np.where(self.quote_fresh,getattr(self.ds,name),np.nan)
            self._known_prices[key] = self.ds._filled_price(raw) if forward_fill else raw
        return self._known_prices[key]

    def observed_age(self):
        if hasattr(self.ds,'_listed_age'):
            return self.ds._listed_age
        if self.stats.get('quote_time_source') != 'explicit':
            return self.ds.list_age
        key = ('list_age',False)
        if key not in self._known_prices:
            d = self.ds
            close = self.observed_price('close')
            valid = np.isfinite(close) & (close > 0)
            steps = (np.r_[0,np.cumsum(np.diff(d.dates.astype('datetime64[D]')) != np.timedelta64(0,'D'))]
                     if getattr(d,'intraday',False) else np.arange(d.T))
            first = np.argmax(valid,axis=0)
            age = steps[:,None]-steps[first][None,:]
            age[~np.maximum.accumulate(valid,axis=0)] = -1
            age[:,valid[0]] = 32767
            if not getattr(d,'intraday',False):
                age = np.clip(age,-1,32767).astype(np.int16)
            self._known_prices[key] = age
        return self._known_prices[key]

    def __repr__(self):
        s = self.stats
        return (f"Market(板幅 {sorted(set(self.band.ravel().tolist()))}%, "
                f"停牌 {s['suspended_cells']:,} 格, "
                f"开盘涨停 {s['open_at_up']:,} 格, 开盘跌停 {s['open_at_down']:,} 格)")

    # -- 可交易判定（引擎的四个口径）---------------------------------------
    def invalid_limit_open(self, t):
        """报价不在当前假定制度内；这与合法涨跌停处排不到队不同。"""
        opening = self.ds.open[t]
        return (np.isfinite(opening) & (opening > 0) &
                ((opening < self.limit_down[t]-0.005) | (opening > self.limit_up[t]+0.005)))

    def can_buy(self, t, rule="open_limit"):
        """t 日开盘时能不能买入。返回 (N,) 的 bool。"""
        if rule == "off":
            return ~self.open_suspended[t]
        ok = ~self.open_suspended[t]
        # 未提供无限制日元数据时，越过已假定板幅的报价不能在另一方向照常成交。
        ok &= ((self.ds.open[t] >= self.limit_down[t]-0.005) &
               (self.ds.open[t] <= self.limit_up[t]+0.005))
        if rule == "open_limit":
            ok &= ~(self.ds.open[t] >= self.limit_up[t] - 0.005)
        elif rule == "sealed":
            ok &= ~self.sealed_up[t]
        elif rule == "broken_board_close":
            close = self.ds.close[t]
            executable_close = (np.isfinite(close) & (close > 0) &
                                (close < self.limit_up[t]-0.005) &
                                (close >= self.limit_down[t]-0.005))
            ok &= ~self.open_at_up[t] | executable_close
        else:
            raise ValueError(f"不认识的 limit_rule={rule!r}，可选 {LIMIT_RULES}")
        return ok

    def can_sell(self, t, rule="open_limit"):
        """t 日开盘时能不能卖出。"""
        if rule == "off":
            return ~self.open_suspended[t]
        ok = ~self.open_suspended[t]
        ok &= ((self.ds.open[t] >= self.limit_down[t]-0.005) &
               (self.ds.open[t] <= self.limit_up[t]+0.005))
        if rule == "open_limit":
            ok &= ~(self.ds.open[t] <= self.limit_down[t] + 0.005)
        elif rule == "sealed":
            ok &= ~self.sealed_down[t]
        elif rule == "broken_board_close":
            close = self.ds.close[t]
            executable_close = (np.isfinite(close) & (close > 0) &
                                (close > self.limit_down[t]+0.005) &
                                (close <= self.limit_up[t]+0.005))
            ok &= ~self.open_at_down[t] | executable_close
        else:
            raise ValueError(f"不认识的 limit_rule={rule!r}，可选 {LIMIT_RULES}")
        return ok

    def fill_price(self, t, side, rule="open_limit"):
        """成交价。

        默认按开盘价成交。`broken_board_close` 这一档对"开盘封板但当天开板"的
        票改用收盘价成交——那是耐心挂单的人更接近的真实结果。
        """
        o = self.ds.open_ff[t]
        if rule != "broken_board_close":
            return o
        opened = self.open_at_up if side == "buy" else self.open_at_down
        closed = self.close_at_up if side == "buy" else self.close_at_down
        return np.where(opened[t] & ~closed[t], self.ds.close[t], o)

    # -- 诊断 --------------------------------------------------------------
    def blocked_report(self, t, side, rule):
        """某一天被挡住的股票数与名单（报告中"被套住的卖出"要用）。"""
        ok = self.can_buy(t, rule) if side == "buy" else self.can_sell(t, rule)
        return ~ok


def _quote_evidence(d, meta, stats):
    """本行OHLC的来源时刻约束；先验证，再允许任何市场状态使用报价。"""
    event_time = d.dates.astype('datetime64[ns]')[:,None]
    if meta.quote_time is None:
        if meta.max_quote_age is not None:
            raise ValueError("设置 max_quote_age 时必须同时提供源 quote_time")
        stats['quote_time_source'] = 'bar_timestamp_assumed'
        stats['stale_quote_cells'] = stats['future_quote_cells'] = stats['missing_quote_time_cells'] = 0
        return np.broadcast_to(event_time, (d.T,d.N)), True
    quote_time = np.asarray(meta.quote_time)
    if quote_time.shape != (d.T,d.N) or quote_time.dtype.kind != 'M':
        raise ValueError("quote_time 须为与行情对齐的 datetime64 (T,N) 源报价时刻")
    quote_time = quote_time.astype('datetime64[ns]').copy()
    tolerance = np.asarray(meta.max_quote_age if meta.max_quote_age is not None else np.timedelta64(0,'ns'))
    if tolerance.ndim or tolerance.dtype.kind != 'm' or np.isnat(tolerance) or tolerance < np.timedelta64(0,'ns'):
        raise ValueError("max_quote_age 须为非负有限 timedelta64")
    age = event_time - quote_time
    fresh = ~np.isnat(quote_time) & (age >= np.timedelta64(0,'ns')) & (age <= tolerance)
    stats['quote_time_source'] = 'explicit'
    stats['stale_quote_cells'] = int((age > tolerance).sum())
    stats['future_quote_cells'] = int((age < np.timedelta64(0,'ns')).sum())
    stats['missing_quote_time_cells'] = int(np.isnat(quote_time).sum())
    return quote_time, fresh


def build_market(ds, band_override=None, st=True, log=print, metadata=None):
    """算全套市场状态面板。结果会被缓存在 ds 上，重复调用不重算。"""
    def build(d):
        close = d.close
        high, low = d.high, d.low
        amt = d.amt
        T, N = close.shape
        dates = d.dates
        main = is_main_board(d.codes)
        meta = MarketMetadata() if metadata is None else metadata

        def bool_panel(value, name):
            a = np.asarray(value)
            if a.shape != (T, N) or a.dtype != np.bool_:
                raise ValueError(f"{name} 必须是对齐的 ({T},{N}) bool 面板")
            return a.copy()

        band0 = base_band(d.codes, dates)
        stats = {}
        quote_time, fresh = _quote_evidence(d, meta, stats)
        if meta.quote_time is not None:
            close, high, low = (np.where(fresh,a,np.nan) for a in (close,high,low))

        # -- ST ------------------------------------------------------------
        if st and not getattr(d,"intraday",False):
            st_panel, st_stats = infer_st(close, high, low, d.cumadj_ff, main, log)
            stats.update(st_stats)
            st_panel = np.vstack([np.zeros((1, N), bool), st_panel[:-1]])
        else:
            st_panel = np.zeros((T, N), dtype=bool)
            stats.update(st_flips=0, st_stocks=0, st_stock_days=0)
        if meta.st is not None:
            st_panel = bool_panel(meta.st, "metadata.st")
        stats["st_source"] = "explicit" if meta.st is not None else ("lagged_inference" if st else "disabled")
        if getattr(d,"intraday",False) and meta.st is None:
            stats["st_source"] = "intraday_requires_metadata"
        stats["st_stocks"] = int(st_panel.any(axis=0).sum())
        stats["st_stock_days"] = int(st_panel.sum())
        stats["st_flips"] = int((st_panel[1:] != st_panel[:-1]).sum())

        # 主板 ST 降到 5%；创业板/科创板 ST 维持 20%（ST 不等于 5%，只对主板成立）
        st_band=np.where(dates.astype('datetime64[D]')[:,None]>=np.datetime64('2026-07-06'),BAND_MAIN,BAND_ST)
        band = np.where(st_panel & main[None, :], st_band, band0).astype(np.int8)

        # -- 停牌 ----------------------------------------------------------
        # 两种表示：价格 NaN（无报价），以及 amt==0（有价但全天无成交，
        # 价格沿用昨收——实测 23182/23229 格如此，最长连续 589 天）。
        # 后者最容易被误当成"一字涨停"，其实完全相反：是没人交易。
        no_quote = ~np.isfinite(close) | (close <= 0)
        zero_amt = (amt == 0)
        suspended = no_quote | zero_amt
        known_suspended = (bool_panel(meta.suspended, "metadata.suspended")
                           if meta.suspended is not None else np.zeros((T, N), bool))
        suspended |= known_suspended
        stats["suspended_cells"] = int(suspended.sum())
        stats["suspended_nan"] = int(no_quote.sum())
        stats["suspended_zero_amt"] = int(zero_amt.sum())

        # -- 无涨跌幅限制日 -------------------------------------------------
        # 三类：复牌首日（前一日无有效价 → 参考价算不出来）、注册制新股前 5 日、
        # 复权收益超过板幅（退市整理期首日、重组复牌都落在这里）。
        # 只影响能否买入；持仓股永远还得能卖。不传播到 t+1 —— 逐日算，天然满足。
        close_ff = d.close_ff if meta.quote_time is None else d._filled_price(close)
        ref = ((d.reference_prices.copy() if meta.quote_time is None else d._reference_prices_from(close_ff))
               if getattr(d,"intraday",False) else reference_price(close_ff, d.cumadj_ff))
        if meta.reference is not None:
            official_ref = np.asarray(meta.reference, float)
            if official_ref.shape != (T, N) or np.isinf(official_ref).any() or (official_ref[np.isfinite(official_ref)] <= 0).any():
                raise ValueError("metadata.reference 必须是对齐的正数/NaN 面板")
            ref = np.where(np.isfinite(official_ref), official_ref, ref)
        c_adj = close * d.cumadj_ff
        prev = np.vstack([np.full((1, N), np.nan), c_adj[:-1]])
        with np.errstate(invalid="ignore", divide="ignore"):
            adj_ret = np.abs(c_adj / prev - 1.0)

        # 当日收盘大涨不能反过来改变当日开盘的板幅。缺少上市/复牌公告时，
        # 保守沿用板块限制；准确的无限制日由开盘前元数据显式提供。
        no_limit = (bool_panel(meta.no_limit, "metadata.no_limit") if meta.no_limit is not None
                    else np.zeros((T, N), bool))
        with np.errstate(invalid="ignore"):
            wild = adj_ret > (band.astype(np.float64) / 100.0 + 0.02)
        stats["no_limit_cells"] = int(no_limit.sum())
        stats["wild_move_cells"] = int(np.nan_to_num(wild, nan=False).sum())

        band_eff = np.where(no_limit, BAND_NONE, band).astype(np.int8)
        if band_override is not None:
            override = np.asarray(band_override)
            if override.shape != (T, N) or not np.isin(override, [0, 5, 10, 20, 30]).all():
                raise ValueError("band_override 必须是对齐的 0/5/10/20/30 板幅面板")
            band_eff = override.astype(np.int8)
            no_limit = band_eff == 0

        # -- 限价 ----------------------------------------------------------
        # 存 float32：判定用的是"到分为止的相等"，float32 在 2700 元处的误差
        # 是 1.5e-5 元，比一个 tick(0.01) 小三个数量级，绰绰有余。
        lu, ld = limit_prices(ref, band_eff)
        for key,target in [('limit_up',lu),('limit_down',ld)]:
            value=getattr(meta,key)
            if value is not None:
                values=np.asarray(value,float)
                if values.shape!=(T,N) or np.isinf(values).any() or (values[np.isfinite(values)]<=0).any():
                    raise ValueError('official limit prices must be aligned positive values/NaN')
                target[:]=np.where(np.isfinite(values),values,target)
        if (ld>lu).any():
            raise ValueError('official limit_down exceeds limit_up')
        stats['limit_price_source']='explicit_or_fallback' if meta.limit_up is not None or meta.limit_down is not None else 'model_reference_and_band'
        lu = lu.astype(np.float32)
        ld = ld.astype(np.float32)

        o = d.open.astype(np.float64)
        if meta.quote_time is not None:
            o = np.where(fresh,o,np.nan)
        open_at_up = _eq_price(o, lu)
        open_at_down = _eq_price(o, ld)
        close_at_up = _eq_price(close, lu)
        close_at_down = _eq_price(close, ld)
        touched_up = _eq_price(high, lu)
        touched_down = _eq_price(low, ld)
        sealed_up = open_at_up & close_at_up & _eq_price(high, lu) & _eq_price(low, lu)
        sealed_down = open_at_down & close_at_down & _eq_price(high, ld) & _eq_price(low, ld)
        quote_suspended = ~np.isfinite(o) | (o <= 0) | known_suspended | ~np.asarray(fresh)
        open_suspended = (quote_suspended |
                          (~np.isfinite(ref) & ~no_limit))
        stats['invalid_limit_open_cells'] = int((np.isfinite(o) & (o > 0) &
            ((o < ld-0.005) | (o > lu+0.005))).sum())

        stats.update(
            open_at_up=int(open_at_up.sum()), open_at_down=int(open_at_down.sum()),
            close_at_up=int(close_at_up.sum()), close_at_down=int(close_at_down.sum()),
            touched_up=int(touched_up.sum()), touched_down=int(touched_down.sum()),
            sealed_up=int(sealed_up.sum()),
            sealed_down=int(sealed_down.sum()),
        )
        # 一字板（开=高=低=收=涨停）
        one_word = open_at_up & close_at_up & _eq_price(high, lu) & _eq_price(low, lu)
        stats["one_word_up"] = int(one_word.sum())

        if log:
            log(f"板幅分布 {dict(zip(*np.unique(band_eff, return_counts=True)))}")
            log(f"停牌 {stats['suspended_cells']:,} 格 "
                f"(NaN {stats['suspended_nan']:,} / amt==0 {stats['suspended_zero_amt']:,})；"
                f"无涨跌幅限制 {stats['no_limit_cells']:,} 格")
            log(f"开盘涨停 {stats['open_at_up']:,} 格，其中一字板 {stats['one_word_up']:,}；"
                f"开盘跌停 {stats['open_at_down']:,} 格")

        market = Market(band=band_eff, limit_up=lu, limit_down=ld, st=st_panel,
                      suspended=suspended, no_limit=no_limit, ds=d,
                      open_suspended=open_suspended, reference=ref,
                      quote_suspended=quote_suspended,
                      quote_time=quote_time,
                      quote_fresh=fresh,
                      sealed_up=sealed_up, sealed_down=sealed_down,
                      open_at_up=open_at_up, open_at_down=open_at_down,
                      close_at_up=close_at_up, close_at_down=close_at_down,
                      touched_up=touched_up, touched_down=touched_down,
                      stats=stats, name="market")
        if meta.quote_time is not None:
            market._known_prices[('close',True)] = close_ff
        return market

    if metadata is not None or band_override is not None:
        return build(ds)
    return ds.cached(("market_v2", bool(st)), build)


# ==========================================================================
# 老师那道题：把「涨停次日」的收益拆开看
# ==========================================================================
class _Illusion:
    """收益路径分解：不将条件收益冒充可成交收益。"""
    def __init__(self, up=True, mask_desc="", rows=(), n=0, resid=0.):
        self.up, self.mask_desc, self.rows, self.n, self.resid = up, mask_desc, rows, n, resid
    def __repr__(self):
        side = "涨停" if self.up else "跌停"
        lines = [f"{side}事件：{self.n:,} 个样本 {self.mask_desc}",
                 "价格路径（未扣成本、未保证成交）                    均值       中位数"]
        for name, reach, mean, median in self.rows:
            lines.append(f"{name:<28} {mean:>+10.2%} {median:>+10.2%}")
        lines += [f"逐格 (1+跳空)*(1+日内)=(1+总收益)，最大残差 {self.resid:.2e}",
                  "日内为开盘到收盘估值收益；次日仍可能封板，买入当天受 T+1 限制。",
                  "实际成交、成本、等待和退出请看 run / run_events 的结果。"]
        return "\n".join(lines)


def limit_up_illusion(ds, market=None, mask=None, up=True, mask_desc=""):
    """把「涨停次日」的收益按**成交价口径**拆开 —— 老师那道题的答案。

    对每一个「t 日收盘涨停」的样本，分别算三条路径上的收益：

        纸面      买在 t 日收盘  -> t+1 收盘      做不到（当天已经封板）
        跳空      t 日收盘      -> t+1 开盘      ——你为此多付的钱
        条件估值  t+1 开盘  -> t+1 收盘（不保证成交或当日可卖）

    三行之间有一条**逐格精确**的恒等式：

        (1 + 跳空) × (1 + 真实) = (1 + 纸面)

    因为 `(O_{t+1}/C_t) × (C_{t+1}/O_{t+1}) = C_{t+1}/C_t`。这不是巧合，是
    同一段收益的两种切法：一截是隔夜跳空，一截是次日的日内走势。表里三行的
    **均值**不满足乘性恒等式（算术平均 ≠ 几何平均），但每一格都精确成立 ——
    所以这个拆解本身没有近似，也没有可以调的余地。

    `mask` 给一个 (T,N) 的布尔面板（通常是策略的 universe），只统计域内的
    样本。不给就用全部涨停样本。

    `up=False` 看跌停的镜像版本。
    """
    m = market if market is not None else build_market(ds, log=None)
    ev = np.asarray(m.close_at_up if up else m.close_at_down, dtype=bool)
    if mask is not None:
        ev = ev & np.asarray(mask, dtype=bool)
    F = np.asarray(ds.cumadj_ff, dtype=np.float64)
    O = np.asarray(ds.open_ff, dtype=np.float64) * F
    C = np.asarray(ds.close_ff, dtype=np.float64) * F
    nxt_o = np.vstack([O[1:], np.full((1, O.shape[1]), np.nan)])
    nxt_c = np.vstack([C[1:], np.full((1, C.shape[1]), np.nan)])

    def band(sel, label, reach):
        v = sel[ev]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return (label, reach, float("nan"), float("nan"))
        return (label, reach, float(np.mean(v)), float(np.median(v)))

    # 分解恒等式逐格验证 —— 不是"应该成立"，是把残差算出来印在表里
    with np.errstate(invalid="ignore", divide="ignore"):
        g = nxt_o[ev] / C[ev] - 1.0          # 跳空
        d = nxt_c[ev] / nxt_o[ev] - 1.0      # 次日日内
        p = nxt_c[ev] / C[ev] - 1.0          # 纸面
    ok = np.isfinite(g) & np.isfinite(d) & np.isfinite(p)
    resid = (float(np.max(np.abs((1.0 + g[ok]) * (1.0 + d[ok]) - (1.0 + p[ok]))))
             if ok.any() else 0.0)

    if up:
        rows = [
            band(nxt_c / C - 1.0, "买在涨停日收盘 → 次日收盘", "做不到（已封板）"),
            band(nxt_o / C - 1.0, "涨停日收盘 → 次日开盘（跳空）", "——"),
            band(nxt_c / nxt_o - 1.0, "买在次日开盘 → 次日收盘", "条件估值收益"),
        ]
    else:
        rows = [
            band(nxt_c / C - 1.0, "卖在跌停日收盘 → 次日收盘", "做不到（已封死）"),
            band(nxt_o / C - 1.0, "跌停日收盘 → 次日开盘（跳空）", "——"),
            band(nxt_c / nxt_o - 1.0, "卖在次日开盘 → 次日收盘", "条件估值收益"),
        ]
    return _Illusion(up=up, mask_desc=mask_desc, rows=rows, n=int(ev.sum()),
                     resid=resid)
