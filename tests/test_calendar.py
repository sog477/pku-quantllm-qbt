"""日历的**闭区间**语义 —— 专门防一个已经把净值曲线画歪过一次的 off-by-one。

背景：`index(day)` 最早写的是"返回 day 之前最后一个交易日"，于是

    index("2021-01-04") -> 242 -> dates[242] == "2020-12-31"

整整差一格。这种 bug 的可怕之处在于它**不影响任何断言**：净值照样算得出来、
图照样画得出来、夏普照样有值，只是每一天都错位一天，而且错得足以让
"用 t 日收盘信号在 t+1 开盘成交"变成"用 t-1 日的信号"。

所以这里逐个把语义钉死：

    index(d, "left")   最后一个 <= d 的交易日    （回测常用：区间左端）
    index(d, "right")  第一个  >= d 的交易日    （回测常用：区间右端）
    index(d, "exact")  d 本身，不是交易日就报错   （对表用）

跑法：  python tests/test_calendar.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qbt  # noqa: E402

_FAILED = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        _FAILED.append(msg)


def main():
    d = qbt.load()
    cal = d.calendar
    days = cal.days
    print(f"日历：{len(days)} 个交易日，{days[0]} .. {days[-1]}")

    print("\n1) exact：交易日就是它自己")
    for i in (0, 1, 242, 500, len(days) - 1):
        dd = days[i]
        check(cal.index(dd, "exact") == i,
              f"index({dd}, exact) == {i}（拿到 {cal.index(dd, 'exact')}）")
        check(cal.index(dd, "left") == i,
              f"index({dd}, left) == {i}（拿到 {cal.index(dd, 'left')}）")
        check(cal.index(dd, "right") == i,
              f"index({dd}, right) == {i}（拿到 {cal.index(dd, 'right')}）")

    print("\n2) 非交易日：left 往前、right 往后、exact 报错")
    # 挑几段真正的休市（周末/长假），而不是相邻的两个交易日
    probes = 0
    for i in range(1, len(days) - 1):
        a, b = np.datetime64(days[i]), np.datetime64(days[i + 1])
        gap = int((b - a) / np.timedelta64(1, "D"))
        if gap < 3:
            continue
        mid = a + np.timedelta64(gap // 2, "D")
        check(cal.index(mid, "left") == i,
              f"index({mid}, left) == {i}（{days[i]}，往前找）")
        check(cal.index(mid, "right") == i + 1,
              f"index({mid}, right) == {i+1}（{days[i+1]}，往后找）")
        try:
            cal.index(mid, "exact")
            check(False, f"index({mid}, exact) 应当报错")
        except KeyError:
            check(True, f"index({mid}, exact) 报错（它确实不是交易日）")
        probes += 1
        if probes >= 3:
            break
    check(probes > 0, "日历里确实存在休市日（否则第 2 组什么都没测到）")

    print("\n3) 区间外：left 太早 / right 太晚 报错，而不返回 0 或 len-1")
    early = np.datetime64(days[0]) - np.timedelta64(400, "D")
    late = np.datetime64(days[-1]) + np.timedelta64(400, "D")
    for day, side in ((early, "left"), (late, "right")):
        try:
            cal.index(day, side)
            check(False, f"index({day}, {side}) 应当报错")
        except KeyError:
            check(True, f"index({day}, {side}) 报错（超出行程，不静默截断）")
    # 另一端是**合法**的：left 晚于终点 = 最后一个交易日，right 早于起点 = 第 0 个
    check(cal.index(late, "left") == len(days) - 1, "index(远未来, left) == 最后一天")
    check(cal.index(early, "right") == 0, "index(远古, right) == 第 0 天")

    print("\n4) slice_of：两端都是闭区间，且不抛异常（夹取语义）")
    s, e = days[100], days[104]
    a, b = cal.slice_of(s, e)
    check((a, b) == (100, 105), f"slice_of({s}, {e}) == (100, 105)（拿到 {(a, b)}）")
    check(b - a == 5, f"闭区间含 5 个交易日（拿到 {b - a}）")
    check(all(cal.days[a + k] == days[100 + k] for k in range(5)),
          "切片内容与 days[100:105] 逐位相同")
    a, b = cal.slice_of(s, None)
    check(a == 100 and b == len(days), f"slice_of(s, None) == (100, {len(days)})")
    a, b = cal.slice_of(None, e)
    check(a == 0 and b == 105, "slice_of(None, e) == (0, 105)")
    # 起止反了 / 全在区间外 → 空区间，而不是抛错或倒着切
    a, b = cal.slice_of(days[104], days[100])
    check(a == b, f"起止颠倒 → 空区间（拿到 {(a, b)}）")
    a, b = cal.slice_of(early, early + np.timedelta64(1, "D"))
    check(a == 0 and b == 0, f"整段早于数据起点 → (0, 0)（拿到 {(a, b)}）")
    a, b = cal.slice_of(late, late + np.timedelta64(1, "D"))
    check(a == len(days) and b == len(days),
          f"整段晚于数据终点 → ({len(days)}, {len(days)})（拿到 {(a, b)}）")

    print("\n5) 和引擎实际用的那一行对得上（端到端对账）")
    # engine.run 用 i1 = cal.index(end, "left")，days 覆盖 dates[i0-1 : i1+1]。
    # 这里直接验证：回测最后一行的日期 == 请求的 end。
    for end in (days[300], days[len(days) // 2], days[-1]):
        r = qbt.run(d, qbt.presets.equal_weight_all(min_amt=1e9),
                    start=days[200], end=end, market=None)
        check(np.datetime64(r.dates[-1]) == np.datetime64(end),
              f"end={end} 时回测最后一行就是 {end}（拿到 {r.dates[-1]}）")
        break   # 跑一次够了：equal_weight_all 只是拿来当一根"不挑票"的探针
    for end in (days[301], days[len(days) // 2 + 1]):
        a, b = cal.slice_of(days[200], end)
        check(cal.days[b - 1] == end, f"slice_of(_, {end}) 的最后一天就是 {end}")

    print()
    if _FAILED:
        print(f"有 {len(_FAILED)} 项没过：")
        for x in _FAILED:
            print("  - " + x)
        return 1
    print("全部通过 —— 日历是闭区间语义，没有 off-by-one。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
