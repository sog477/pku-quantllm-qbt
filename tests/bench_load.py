"""测量解码各阶段的耗时，并验证多进程解码在真实脚本里能正常工作。

    python tests/bench_load.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbt.arrow import read_arrow_ipc, read_feather  # noqa: E402
from qbt.data import _decode_one, load              # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SRC = os.path.join(DATA, "close.feather")


def phases(path):
    t = {}
    t0 = time.perf_counter()
    with open(path, "rb") as fh:
        buf = fh.read()
    t["read file"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    fields, cols = read_arrow_ipc(buf)
    t["decode arrow+lz4"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    names = [f["name"] for f in fields][1:]
    mat = np.empty((1643, len(names)), dtype=np.float64)
    for j, c in enumerate(names):
        mat[:, j] = cols[c]
    t["pack matrix"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ff = np.isnan(mat)
    idx = np.where(~ff, np.arange(mat.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = mat[idx, np.arange(mat.shape[1])[None, :]]
    t["ffill"] = time.perf_counter() - t0
    return t


def main():
    print(f"close.feather  {os.path.getsize(SRC)/1e6:.1f} MB")

    # 冷启动各阶段
    t = phases(SRC)
    for k, v in t.items():
        print(f"  {k:20s} {v:6.2f}s")
    print(f"  {'合计':20s} {sum(t.values()):6.2f}s")

    # 热缓存再跑一次（排除磁盘）
    t = phases(SRC)
    print("  热缓存:")
    for k, v in t.items():
        print(f"  {k:20s} {v:6.2f}s")

    # 单文件 _decode_one，和上面的 read_feather 路径对比
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        t0 = time.perf_counter()
        r = _decode_one(("close", SRC, td))
        print(f"  _decode_one         {time.perf_counter()-t0:6.2f}s  (它自己报 {r[4]:.2f}s)")

    # 多进程整体
    t0 = time.perf_counter()
    d = load(verbose=False)
    print(f"\nload() 总耗时 {time.perf_counter()-t0:.2f}s")
    print(" ", d)
    print(f"  close      {d.close.dtype}  nan={int(np.isnan(d.close).sum()):,}")
    print(f"  close_ff   {d.close_ff.dtype}  nan={int(np.isnan(d.close_ff).sum()):,}")
    nan = np.isnan(d.close)
    leading = np.isnan(d.close_ff) & nan
    print(f"  其中前导(未上市) NaN = {int(leading.sum()):,}，"
          f"停牌被填上的 = {int(nan.sum()-leading.sum()):,}")


if __name__ == "__main__":
    main()
