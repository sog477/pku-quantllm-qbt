"""用 pyarrow 把 7 个 feather 全量转成 .npy，作为对拍的"标准答案"。

这个脚本必须用**装了 pyarrow 的解释器**跑，例如：
    C:\\Users\\15504\\scoop\\apps\\python312\\current\\python.exe tests\\_oracle_dump.py

它只做一件事：把每个文件读成 (T, N) 的 float64 矩阵落盘。列的先后顺序按
pyarrow 给的 schema 来，同时把列名写成 .json，这样对拍时列顺序不会串。
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pyarrow.feather as feather

FILES = ["open", "high", "low", "close", "amt", "num_trades", "cumadj"]

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ORACLE = os.path.join(DATA, "_oracle")


def main():
    os.makedirs(ORACLE, exist_ok=True)
    for name in FILES:
        src = os.path.join(DATA, f"{name}.feather")
        dst = os.path.join(ORACLE, f"{name}.npy")
        jst = os.path.join(ORACLE, f"{name}.json")
        if os.path.exists(dst) and os.path.exists(jst):
            print(f"  {name:12s} 已有，跳过")
            continue

        t0 = time.time()
        tbl = feather.read_table(src)
        cols = tbl.column_names
        mat = np.empty((tbl.num_rows, len(cols)), dtype=np.float64)
        for j, cn in enumerate(cols):
            arr = tbl.column(cn).combine_chunks()
            if j == 0 and "date" in cn:
                # 日期列存成 int64（ns since epoch），避免时区/单位差异混进对拍
                mat[:, j] = arr.cast("int64").to_numpy(zero_copy_only=False)
            else:
                mat[:, j] = arr.to_numpy(zero_copy_only=False)

        np.save(dst, mat)
        with open(jst, "w", encoding="utf-8") as fh:
            json.dump({"columns": cols, "rows": tbl.num_rows,
                       "types": [str(tbl.schema.field(c).type) for c in cols]},
                      fh, ensure_ascii=False)
        print(f"  {name:12s} {mat.shape}  {time.time()-t0:.1f}s  -> {os.path.basename(dst)}")

    print("oracle 完成:", ORACLE)


if __name__ == "__main__":
    main()
