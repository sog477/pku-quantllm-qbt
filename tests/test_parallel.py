"""临时目录验证单进程、多进程与缓存一致；不删除项目数据或现有缓存。"""
from pathlib import Path
import os
import sys
import tempfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import qbt
from qbt.data import FILES


def main():
    with tempfile.TemporaryDirectory(prefix="qbt_parallel_") as temp:
        root = Path(temp).resolve()
        if not root.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            raise RuntimeError("临时验证目录超出预期范围")
        t = time.perf_counter()
        single = qbt.load(ROOT / "data", cache_dir=str(root / "single"), workers=1, verbose=False)
        print(f"单进程 {time.perf_counter()-t:.2f}s", flush=True)
        t = time.perf_counter()
        parallel = qbt.load(ROOT / "data", cache_dir=str(root / "parallel"), workers=2, verbose=False)
        print(f"多进程 {time.perf_counter()-t:.2f}s", flush=True)
        t = time.perf_counter()
        cached = qbt.load(ROOT / "data", cache_dir=str(root / "parallel"), verbose=False)
        print(f"缓存 {time.perf_counter()-t:.2f}s", flush=True)
        for name in FILES:
            np.testing.assert_array_equal(getattr(single, name), getattr(parallel, name))
            np.testing.assert_array_equal(getattr(single, name), getattr(cached, name))
        assert single.codes == parallel.codes == cached.codes
        np.testing.assert_array_equal(single.dates, parallel.dates)
        print("七矩阵、代码和日期一致；项目原缓存保留。", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
