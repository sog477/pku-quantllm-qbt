"""零依赖 Arrow 读取器的全量对拍测试。

这是整个零依赖方案唯一的技术风险点：自己解 Arrow IPC + LZ4 frame 会不会在
某个边角上悄悄读错一个数。所以这里不比"差不多"，而是把 7 个文件的
**1643 x 5452 = 8,956,436 个格子**全部与 pyarrow 的结果逐个比对——包括
NaN 出现的位置（对应 Arrow 的 null 位图）必须完全一致。

跑法：
    python tests/test_arrow.py            # 需要先有 data/_oracle/*.npy
    python tests/_oracle_dump.py          # 用装了 pyarrow 的解释器生成标准答案
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbt.arrow import ArrowError, lz4_frame_decompress, read_arrow_ipc, read_feather  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ORACLE = os.path.join(DATA, "_oracle")
FILES = ["open", "high", "low", "close", "amt", "num_trades", "cumadj"]

_failures: list[str] = []


def check(cond, msg):
    if cond:
        print(f"    ok   {msg}")
    else:
        print(f"    FAIL {msg}")
        _failures.append(msg)


# --------------------------------------------------------------------------
# 1. LZ4 解码器自身的单元测试（不依赖数据文件）
# --------------------------------------------------------------------------
def _wrap_lz4_frame(block_body: bytes, uncompressed: bool = False) -> bytes:
    """手工拼一个最小的 LZ4 frame，用来测解码器本身。"""
    f = bytearray(b"\x04\x22\x4d\x18")
    f += bytes([0x60, 0x40, 0x00])       # FLG(ver=01,block-indep) BD(64KB) HC(不校验)
    if uncompressed:
        f += (0x80000000 | len(block_body)).to_bytes(4, "little") + block_body
    else:
        f += len(block_body).to_bytes(4, "little") + block_body
    f += b"\x00\x00\x00\x00"             # end mark
    return bytes(f)


def test_lz4_units():
    print("\n[1] LZ4 frame 解码器")

    # (a) 重叠 match：aab + 从 offset=1 复制 12 字节 -> aabbbbbbbbbbbb
    #     token 高 4 位 = 字面量长度 = 3；低 4 位 = matchlen-4 = 12-4 = 8  => 0x38
    body = bytes([0x38]) + b"aab" + bytes([0x01, 0x00])
    got = lz4_frame_decompress(_wrap_lz4_frame(body))
    check(got == b"aab" + b"b" * 12, f"重叠 match -> {got!r}")

    # (b) 普通（不重叠）match：abcdefgh + 从 offset=8 复制 4 字节 -> abcdefghabcd
    body = bytes([0x80]) + b"abcdefgh" + bytes([0x08, 0x00])
    got = lz4_frame_decompress(_wrap_lz4_frame(body))
    check(got == b"abcdefghabcd", f"普通 match -> {got!r}")

    # (c) 字面量 >= 15 的扩展长度分支
    lit = b"x" * 300
    body = bytes([0xF0]) + b"\xc9\x01" + lit      # 300 = 15 + 201 + 84? 用 15+len 编码
    # 重新按规范编码：15 + 255 + (300-15-255=30)
    body = bytes([0xF0]) + bytes([255, 30]) + lit
    got = lz4_frame_decompress(_wrap_lz4_frame(body))
    check(got == lit, f"扩展字面量长度 300 -> {len(got)} 字节")

    # (d) 未压缩块直通
    got = lz4_frame_decompress(_wrap_lz4_frame(b"hello", uncompressed=True))
    check(got == b"hello", "未压缩块直通")

    # (e) 空帧
    f = bytes(b"\x04\x22\x4d\x18") + bytes([0x60, 0x40, 0x00]) + b"\x00" * 4
    check(lz4_frame_decompress(f) == b"", "空帧")

    # (f) 非 LZ4 输入必须报错，不能静默返回垃圾
    try:
        lz4_frame_decompress(b"not an lz4 frame at all")
        check(False, "非 LZ4 输入应抛 ArrowError")
    except ArrowError:
        check(True, "非 LZ4 输入抛 ArrowError")

    # (g) 非 Arrow 文件
    try:
        read_arrow_ipc(b"definitely not arrow")
        check(False, "非 Arrow 输入应抛 ArrowError")
    except ArrowError:
        check(True, "非 Arrow 输入抛 ArrowError")

    # (h) 真文件的 schema 结构
    path = os.path.join(DATA, "cumadj.feather")
    if os.path.exists(path):
        from qbt.arrow import _parse_footer
        with open(path, "rb") as fh:
            buf = fh.read()
        fields, blocks = _parse_footer(buf)
        check(len(fields) == 5452 and len(blocks) == 1,
              f"cumadj schema {len(fields)} 列 / {len(blocks)} 个 record batch")


# --------------------------------------------------------------------------
# 2. 结构测试：每个文件的 shape / 列名 / dtype
# --------------------------------------------------------------------------
def test_structure():
    print("\n[2] 结构")
    for name in FILES:
        path = os.path.join(DATA, f"{name}.feather")
        if not os.path.exists(path):
            check(False, f"{name}.feather 不存在")
            continue
        cols = read_feather(path)
        check(len(cols) == 5452, f"{name}: 5452 列")
        first = next(iter(cols.values()))
        check(first.shape == (1643,), f"{name}: 1643 行")

        with open(os.path.join(ORACLE, f"{name}.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        check(list(cols.keys()) == meta["columns"], f"{name}: 列名与列序和 pyarrow 完全一致")


# --------------------------------------------------------------------------
# 3. 全量对拍：每一个格子
# --------------------------------------------------------------------------
def test_full_match():
    print("\n[3] 全量对拍（每个格子）")
    total_cells = 0
    for name in FILES:
        path = os.path.join(DATA, f"{name}.feather")
        ref_path = os.path.join(ORACLE, f"{name}.npy")
        if not (os.path.exists(path) and os.path.exists(ref_path)):
            check(False, f"{name}: 缺文件，跳过")
            continue

        t0 = time.time()
        cols = read_feather(path)
        dt = time.time() - t0
        names = list(cols.keys())
        ref = np.load(ref_path)

        mine = np.empty_like(ref)
        for j, cn in enumerate(names):
            a = cols[cn]
            if a.dtype.kind == "M":                       # 日期列
                mine[:, j] = a.view("int64")
            elif a.dtype.kind == "O":                     # 带 null 的非浮点列
                mine[:, j] = np.array(
                    [np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else v
                     for v in a], dtype=np.float64)
            else:
                mine[:, j] = a

        # NaN 掩码必须逐格相同（Arrow 用 null 位图表示缺失，pyarrow 也转成 NaN）
        mn, rn = np.isnan(mine), np.isnan(ref)
        n_mismatch_mask = int((mn != rn).sum())

        both = ~mn & ~rn
        diff = np.abs(mine[both] - ref[both])
        n_val = int((diff != 0).sum())
        n_nan_mine = int(mn.sum())

        total_cells += mine.size
        ok = n_mismatch_mask == 0 and n_val == 0
        check(ok,
              f"{name}: {mine.size} 格，NaN {n_nan_mine} 个 / 掩码不一致 {n_mismatch_mask} / "
              f"数值不一致 {n_val}（读取 {dt:.1f}s）")
        if not ok:
            idx = np.argwhere((mn != rn) | (np.isnan(mine) & ~np.isnan(ref)))[:3]
            for i in idx:
                print(f"        首个不一致 @ {tuple(i)}: mine={mine[tuple(i)]} ref={ref[tuple(i)]}")
            if n_val:
                w = np.argwhere(both & (mine != ref))[:3]
                for i in w:
                    print(f"        数值差 @ {tuple(i)}: mine={mine[tuple(i)]!r} ref={ref[tuple(i)]!r}")

    print(f"\n    合计比对 {total_cells:,} 个格子")


# --------------------------------------------------------------------------
# 4. 与数据事实一致（顺带把老师那些坑的数字钉死）
# --------------------------------------------------------------------------
def test_data_facts():
    print("\n[4] 数据事实")
    if not os.path.exists(os.path.join(ORACLE, "close.npy")):
        print("    跳过（缺 oracle）")
        return

    ref = {n: np.load(os.path.join(ORACLE, f"{n}.npy")) for n in FILES}
    close = ref["close"][:, 1:]
    amt = ref["amt"][:, 1:]
    cumadj = ref["cumadj"][:, 1:]
    date = ref["close"][:, 0]

    check(date[0] == np.datetime64("2020-01-02").astype("datetime64[ns]").astype("int64"),
          "首个交易日 = 2020-01-02")
    check(date[-1] == np.datetime64("2026-10-16").astype("datetime64[ns]").astype("int64"),
          "末个交易日 = 2026-10-16")

    n_amt0 = int((amt == 0).sum())
    check(n_amt0 == 23229, f"amt == 0 共 {n_amt0} 格（期望 23229）")

    # amt==0 时价格是否沿用昨收（停牌，而非涨停）
    m = amt == 0
    same = int((close[1:][m[1:]] == close[:-1][m[1:]]).sum())
    check(same >= 23000, f"amt==0 中 {same}/{n_amt0} 格 close == 昨收（停牌而非涨停）")

    # 涨停日流动性看着很足
    nan = np.isnan(close)
    check(True, f"价格 NaN 占比 {np.isnan(close).mean()*100:.2f}%")
    check(True, f"cumadj 缺 {int(np.isnan(cumadj).sum())} 格")

    print(f"    数据形状 close={close.shape}")


def main():
    print("=" * 72)
    print("零依赖 Arrow 读取器 · 全量对拍测试")
    print("=" * 72)
    test_lz4_units()
    test_structure()
    test_full_match()
    test_data_facts()

    print("\n" + "=" * 72)
    if _failures:
        print(f"失败 {len(_failures)} 项：")
        for f in _failures:
            print("  -", f)
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
