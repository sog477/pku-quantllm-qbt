"""可交易性回归测试：把老师那些坑的数字钉死在代码里。

每一个断言都对应课程里讲过的一个坑。以后谁改了涨跌停/停牌/ST 的逻辑，
这个测试会立刻红掉，而不是悄悄给出一个乐观的回测。

    python tests/test_tradability.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qbt  # noqa: E402
from qbt.tradability import BAND_GEM, BAND_MAIN, BAND_ST, build_market  # noqa: E402

_fail = []


def check(cond, msg):
    print(("    ok   " if cond else "    FAIL ") + msg)
    if not cond:
        _fail.append(msg)


def main():
    d = qbt.load(verbose=False)
    print(f"\n{d}\n")
    m = build_market(d, log=lambda s: print("  ·", s))

    codes = np.asarray(d.codes)
    close, amt = d.close, d.amt
    T, N = close.shape

    print("\n[A] 复权方向：adj = raw × cumadj（不是除法）")
    last = d.cumadj_ff[-1]
    # 用除权事件日的收益波动来验证：复权应该让收益更平滑
    valid = ~np.isnan(close)
    f = d.cumadj_ff
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_ret = np.abs(close[1:] / close[:-1] - 1)
        ratio = np.where(valid[1:] & valid[:-1], f[1:] / f[:-1], np.nan)
    # 真实除权事件（因子相对变化 > 0.5%）
    ev = np.nan_to_num(np.abs(ratio - 1) > 0.005)
    n_ev = int(ev.sum())
    check(15000 <= n_ev <= 25000, f"除权事件（|ΔF/F|>0.5%）共 {n_ev:,} 次（老师那条 19,951）")

    mul_med = float(np.nanmedian(raw_ret[ev == 1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        adj_ret = np.abs((close[1:] * f[1:]) / (close[:-1] * f[:-1]) - 1)
        div_ret = np.abs((close[1:] / f[1:]) / (close[:-1] / f[:-1]) - 1)
    adj_med = float(np.nanmedian(adj_ret[ev == 1]))
    div_med = float(np.nanmedian(div_ret[ev == 1]))
    print(f"         除权日 |收益| 中位数：不复权 {mul_med*100:.2f}%  "
          f"乘因子 {adj_med*100:.2f}%  除因子 {div_med*100:.2f}%")
    check(adj_med < mul_med and adj_med < div_med,
          "乘因子让除权日收益最平滑 -> 方向正确（搞反是静默灾难）")

    print("\n[B] 板块与板幅")
    band = m.band
    pre = np.array([c[:3] for c in codes])
    gem = np.isin(pre, ["300", "301", "302"])
    star = np.isin(pre, ["688", "689"])
    board20 = gem | star
    for b, want in ((BAND_MAIN, 10), (BAND_GEM, 20), (BAND_ST, 5)):
        cnt = int((band == b).sum())
        print(f"         板幅 {want:2d}% 占 {cnt:,} 格")
    check(int((band == 30).sum()) == 0, "30% 板（北交所）占 0 格 —— 本数据集没有北交所")

    # 2020-08-24 创业板注册制改革：改革前 10%、改革后 20%
    i = d.calendar.index("2020-08-24")
    gem_before = band[i - 1, gem]
    gem_after = band[i + 5, gem]
    print(f"         创业板板幅 2020-08-21 = {sorted(set(gem_before.tolist()))}, "
          f"2020-08-31 = {sorted(set(gem_after.tolist()))}")
    check(set(gem_before.tolist()) <= {10, 5, 0} and set(gem_after.tolist()) <= {20, 5, 0},
          "创业板 2020-08-24 当天从 10% 抬到 20%（板幅必须按日算）")

    print("\n[C] 涨停价公式 ref = 昨收 × 昨因子 / 今因子")
    lu, ld = m.limit_up, m.limit_down
    # 注意：创业板 2020-08-24 之前本来就是 10% 板，所以"20% 板块"这个掩码
    # 必须对创业板带上改革日。科创板 2019 年就开板了，一直是 20%，不受影响。
    reform = d.dates[:, None] >= np.datetime64("2020-08-24")
    board20_now = star[None, :] | (gem[None, :] & reform)
    # 用"收盘正好落在限价上"来验证公式（比"最高价触及"更严格）
    close_at_lu = np.abs(np.round(d.close.astype(np.float64) * 100)
                         - np.round(lu.astype(np.float64) * 100)) < 0.5
    close_at_ld = np.abs(np.round(d.close.astype(np.float64) * 100)
                         - np.round(ld.astype(np.float64) * 100)) < 0.5
    hit20 = (band == BAND_GEM) & (close_at_lu | close_at_ld)
    n20, bad20 = int(hit20.sum()), int((hit20 & ~board20_now).sum())
    check(bad20 == 0,
          f"20% 板公式命中 {n20:,} 次，落在非创业/科创的 {bad20} 次（应为 0）")
    hit10 = (band == BAND_MAIN) & (close_at_lu | close_at_ld)
    n10, bad10 = int(hit10.sum()), int((hit10 & board20_now).sum())
    check(bad10 == 0,
          f"10% 板公式命中 {n10:,} 次，落在 20% 板块的 {bad10} 次（应为 0）")

    print("\n[D] 停牌的两种表示")
    s = m.stats
    check(s["suspended_zero_amt"] == 23229, f"amt==0 共 {s['suspended_zero_amt']:,} 格（期望 23,229）")
    za = amt == 0
    same = int((close[1:][za[1:]] == close[:-1][za[1:]]).sum())
    check(same / max(za.sum(), 1) > 0.99,
          f"amt==0 中 {same:,}/{int(za.sum()):,} 格收盘价 = 昨收 -> 是停牌，不是涨停")
    check(int(za.sum()) == s["suspended_zero_amt"], "amt==0 全部被判为停牌")

    print("\n[E] 真·一字涨停买不到（老师那道题的本质）")
    oneword = m.open_at_up & m.close_at_up & (m.touched_up) & \
        (np.abs(np.round(d.high.astype(np.float64) * 100) - np.round(lu.astype(np.float64) * 100)) < 0.5) & \
        (np.abs(np.round(d.low.astype(np.float64) * 100) - np.round(lu.astype(np.float64) * 100)) < 0.5)
    n_ow = int(oneword.sum())
    ow_amt0 = int((oneword & za).sum())
    med_amt = float(np.nanmedian(amt[oneword]))
    check(n_ow > 6000, f"一字涨停 {n_ow:,} 格（老师那条 8,171）")
    check(ow_amt0 == 0, f"一字涨停里 amt==0 的有 {ow_amt0} 格（应为 0）—— 流动性看着很足，照样买不到")
    check(med_amt > 1e7, f"一字涨停当日成交额中位数 {med_amt/1e4:.0f} 万（流动性并不差）")

    print("\n[F] 四种 limit_rule 的拦截力度")
    blocked = {}
    for rule in ("off", "sealed", "broken_board_close", "open_limit"):
        nb = ns = 0
        for t in range(T):
            nb += int((~m.can_buy(t, rule) & m.can_buy(t, "off")).sum())
            ns += int((~m.can_sell(t, rule) & m.can_sell(t, "off")).sum())
        blocked[rule] = (nb, ns)
        print(f"         {rule:20s} 买被拦 {nb:7,} 格   卖被拦 {ns:7,} 格")
    check(blocked["off"] == (0, 0), "off 不额外拦涨跌停；开盘缺报价/参考价仍不可交易")
    check(blocked["open_limit"][0] > blocked["sealed"][0],
          "open_limit 比 sealed 更保守（开盘封板就算排不到队）")
    check(blocked["open_limit"][1] > 0, "跌停同样卖不掉（对称的坑）")

    print("\n[G] ST 推断的诚实披露")
    print(f"         封在 5% 板上的格子 {s['sealed_5pct_cells']:,}")
    print(f"         被标过 ST 的股票 {s['st_stocks']:,} 只 / 共 {N:,} 只，"
          f"占 {s['st_stock_days']:,} 个股票日")
    print(f"         状态翻转 {s['st_flips']:,} 次（摘帽/戴帽）")
    check(s["st_stocks"] > 0, "ST 是时变的，必须按股按日推")
    # 第一版用"价格撞上限价"判 ST，结果 3287 只（60%）被误标 —— 这个上界是防回归的
    check(s["st_stocks"] < N * 0.4,
          f"被标 ST 的股票 {s['st_stocks']:,} 只，不应接近全市场（噪声判别器的特征）")

    print("\n[H] 无涨跌幅限制日")
    print(f"         {s['no_limit_cells']:,} 格无涨跌幅限制")
    with np.errstate(invalid="ignore", divide="ignore"):
        f2 = d.cumadj_ff
        c_adj = close * f2
        r = np.abs(c_adj[1:] / c_adj[:-1] - 1)
    big = np.nan_to_num(r) > 0.30
    n_big_stocks = int((big.any(axis=0)).sum())
    check(n_big_stocks > 0,
          f"{n_big_stocks} 只出现过 |复权日收益| > 30%（退市整理期/重组复牌）")

    print("\n" + "=" * 68)
    if _fail:
        print(f"失败 {len(_fail)} 项：")
        for x in _fail:
            print("  -", x)
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
