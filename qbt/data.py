"""数据层：把 7 个 feather 变成一个可以直接算的面板集合。

研究员看到的就是 `qbt.load()` 返回的一个 `Dataset`：

    d = qbt.load()                     # 首次解码 ~15s，之后走 .npy 缓存 1~2s
    d.close                            # (1643, 5451) 未复权收盘价，停牌处是 NaN
    d.close_ff                         # 前向填充版，用来算净值（不能让停牌把净值变 NaN）
    d.close_adj                        # 后复权价格 = close_ff * cumadj，用来算收益
    d.codes[:5], d.dates[0], d.calendar

几个刻意的设计：

* **两条价格线**。`close` 保留 NaN（信号、涨跌停判定必须用真的），`close_ff`
  前向填充（Mark-to-Market 必须用连续的）。混用会一边造假收益、一边把净值
  搞成 NaN。这是这个数据集最容易踩的坑。
* **OHLC和复权因子用float64**。活动量面板存float32。即使float32误差小于一分，
  也会让真实10.01开盘变成10.01000023，错误拒绝10.01的合法买入限价单。
  交易所的涨跌停价是"四舍五入到分"，float32 在 11.5 附近的误差
  (1e-6) 理论上足以让一个正好卡在半分位的限价翻边，进而是错的可交易性、
  错的回测。不能为节省缓存而改变合法限价单的成交边界。
* **缓存带指纹**。缓存目录名 = 源文件的 (名字, 大小, mtime, 内容哈希) 指纹。换数据、
  改读取器版本，指纹就变，自动重建，不会读到过期缓存。
* **锁文件**。Jupyter 里多开几个 kernel 同时 load() 是常态，没有锁就会两个
  进程同时写同一个 .npy，读出半个文件。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
import zlib
import tempfile

import numpy as np

from .arrow import read_arrow_ipc
from ._validation import naive_datetime_input

__all__ = ["load", "Dataset", "FILES", "READER_VERSION"]

FILES = ("open", "high", "low", "close", "amt", "num_trades", "cumadj")
READER_VERSION = 7                       # v7保留所有OHLC源精度，避免float32跨越真实限价边界
_PRECISE = frozenset({"open", "high", "low", "close", "cumadj"})
_ORIGIN = "2020-01-02"                     # 缓存指纹里带上，防止换起点不重建

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
DEFAULT_DATA_DIR = os.path.join(_PROJECT, "data")
DEFAULT_CACHE_DIR = os.path.join(DEFAULT_DATA_DIR, "_cache")


# ==========================================================================
# 源文件解析
# ==========================================================================
def _resolve_source(source) -> str:
    """把 source 变成一个装着 7 个 .feather 的目录。"""
    if source is None:
        source = DEFAULT_DATA_DIR
    source = os.fspath(source)

    if os.path.isdir(source):
        for cand in (source, os.path.join(source, "feathers")):
            if all(os.path.exists(os.path.join(cand, f + ".feather")) for f in FILES):
                return cand
        raise FileNotFoundError(
            f"{source} 里找不到 {', '.join(f + '.feather' for f in FILES)}；"
            f"请指向解压后的目录，或者直接把 .zip 传进来"
        )

    if os.path.isfile(source) and source.lower().endswith(".zip"):
        out = os.path.join(os.path.dirname(source), "feathers")
        os.makedirs(out, exist_ok=True)
        with zipfile.ZipFile(source) as z, tempfile.TemporaryDirectory(prefix='.qbt_extract_',dir=out) as staging:
            members = {}
            expected = {name+'.feather' for name in FILES}
            for name in z.namelist():
                base = os.path.basename(name)
                if base.startswith('._') or base.lower() not in expected:
                    continue
                key = base.lower()
                if key in members:
                    raise ValueError(f'压缩包含多个同名面板{key}，不能混用数据版本；请指定唯一一套解压目录')
                members[key] = name
            replacements = []
            for f in FILES:
                m = members.get(f + ".feather")
                if m is None:
                    raise FileNotFoundError(f"压缩包里没有 {f}.feather")
                dst = os.path.join(out, f + ".feather")
                if os.path.exists(dst) and os.path.getsize(dst) == z.getinfo(m).file_size:
                    crc = 0
                    with open(dst, "rb") as existing:
                        for chunk in iter(lambda: existing.read(1 << 22), b""):
                            crc = zlib.crc32(chunk, crc)
                    if crc == z.getinfo(m).CRC:
                        continue
                staged = os.path.join(staging, f + '.feather')
                with z.open(m) as src, open(staged, "wb") as fh:
                    shutil.copyfileobj(src, fh, 1 << 22)
                replacements.append((staged,dst))
            # 七个成员全部解压并通过ZIP的CRC检查后，才替换现有源文件。
            for staged,dst in replacements:
                os.replace(staged,dst)
        return out

    if os.path.isfile(source):
        # 直接给了一个 .feather：用它的目录
        d = os.path.dirname(os.path.abspath(source))
        if all(os.path.exists(os.path.join(d, f + ".feather")) for f in FILES):
            return d
    raise FileNotFoundError(f"认不出这个 source: {source!r}")


def fingerprint(feather_dir: str) -> str:
    """包含实际内容，防同尺寸、保留时间戳的更新静默复用旧缓存。"""
    h = hashlib.sha256(f"v{READER_VERSION}|{_ORIGIN}|content-v1|".encode())
    for f in FILES:
        p = os.path.join(feather_dir, f + ".feather")
        st = os.stat(p)
        h.update(f"{f}:{st.st_size}:{st.st_mtime_ns}|".encode())
        with open(p, "rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 22), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


# ==========================================================================
# 解码（可以在子进程里跑）
# ==========================================================================
def _validate_panel(name, values):
    if np.isinf(values).any():
        raise ValueError(f'{name} 包含无穷值，不能发布为可回测面板')
    if name in ('amt', 'num_trades') and (values < 0).any():
        raise ValueError(f'{name} 必须非负或NaN，负值会污染后续流动性窗口')


def _numeric_panel(values, name):
    if np.asarray(values).dtype.kind == 'b':
        raise ValueError(f'{name}须为数值面板，布尔特征不能当价格/活动量')
    return np.asarray(values,float)


def _decode_one(task):
    """读一个 feather，把股票列按 (T, N) float32/float64 矩阵存成 .npy。

    取日期列的活由第一个文件干，这里返回它让主进程存。
    """
    name, src, out_dir = task
    t0 = time.time()
    with open(src, "rb") as fh:
        buf = fh.read()
    fields, cols = read_arrow_ipc(buf)
    del buf

    names = [f["name"] for f in fields]
    date_col = fields[0]["type"] in (8, 10)      # Timestamp / Date32
    if date_col and fields[0].get('timezone'):
        raise ValueError('feather日期带时区；请先按目标市场显式转换交易日/时间标签，不能直接按UTC日历截断')
    codes = names[1:] if date_col else names
    n_col, T = len(codes), len(cols[names[0]])

    dt = np.float64 if name in _PRECISE else np.float32
    mat = np.empty((T, n_col), dtype=dt)
    for j, c in enumerate(codes):
        if fields[j+int(date_col)]['type'] == 6:
            raise ValueError(f'{name}列{c}是bool，不是价格或活动量')
        a = cols.pop(c)
        if a.dtype.kind == "O":                  # 带 null 的非浮点列
            a = a.astype(np.float64)
        _validate_panel(name, a)
        if (np.abs(a) > np.finfo(dt).max).any():
            raise ValueError(f'{name} 超过缓存{np.dtype(dt)}范围，不能将有限源值存成无穷')
        mat[:, j] = a
        del a

    dates = None
    if date_col:
        d = cols[names[0]]
        dates = d.astype('datetime64[ns]').view("i8").copy() if d.dtype.kind == "M" else np.asarray(d, dtype="i8")

    np.save(os.path.join(out_dir, f"{name}.npy"), mat)
    return name, codes, mat.shape, dates, time.time() - t0


def _acquire_lock(path, timeout=600.0):
    """拿一个排它锁；已经有人在做同样的解码就等它做完。"""
    t0 = time.time()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"等锁 {path} 超时；如果确认没有别的进程在跑，删掉它")
            time.sleep(0.25)


def _wait_for_meta(meta_path, lock_path, timeout=900.0):
    """别人正在解码 → 等 meta.json 出现。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(meta_path):
            return True
        if not os.path.exists(lock_path):
            return False          # 那个人挂了，锁没了也没写出 meta
        time.sleep(0.25)
    return False


def _valid_cache_meta(cache_dir, fp):
    """元数据存在不代表缓存完整；检查全部NPY头和长度，不扫描行情负载。"""
    try:
        with open(os.path.join(cache_dir,'meta.json'),encoding='utf-8') as stream:
            meta=json.load(stream)
        if (meta['fingerprint'] != fp or meta['reader_version'] != READER_VERSION or
                meta['files'] != list(FILES) or len(meta['codes']) != meta['N'] or
                len(set(meta['codes'])) != meta['N'] or meta['T'] < 2 or meta['N'] < 1):
            return None
        for name in (*FILES,'dates'):
            path=os.path.join(cache_dir,name+'.npy')
            with open(path,'rb') as stream:
                version=np.lib.format.read_magic(stream)
                reader = {(1,0):np.lib.format.read_array_header_1_0,
                          (2,0):np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    return None
                shape,fortran,dtype=reader(stream)
                expected=(meta['T'],) if name=='dates' else (meta['T'],meta['N'])
                expected_dtype=np.dtype('int64' if name=='dates' else 'float64' if name in _PRECISE else 'float32')
                if shape != expected or dtype != expected_dtype or fortran:
                    return None
                if os.fstat(stream.fileno()).st_size != stream.tell()+int(np.prod(shape))*dtype.itemsize:
                    return None
        return meta
    except (OSError,ValueError,KeyError,TypeError,EOFError):
        return None


# ==========================================================================
# Dataset
# ==========================================================================
class Dataset:
    """7 个原始面板 + 一堆按需派生的面板。

    派生面板第一次访问时才算，算完就缓存（`self._c`），所以来回用不会重复算。
    """

    def __init__(self, cache_dir, dates, codes, mmap=False, verbose=False):
        from .calendar import Calendar

        self.cache_dir = cache_dir
        self._mmap = mmap
        self.codes = list(codes)
        self.codes_arr = np.asarray(self.codes)
        self.dates = np.asarray(dates).astype("datetime64[ns]").astype("datetime64[D]")
        self.calendar = Calendar(self.dates)
        self.T = len(self.dates)
        self.N = len(self.codes)
        self._raw = {}
        self._c = {}
        self.verbose = verbose

    @classmethod
    def from_arrays(cls, dates, codes, **panels):
        """从对齐的七个 numpy 面板构造数据集，适用于自有数据和小型实验。"""
        dates = naive_datetime_input(dates, 'dates').astype('datetime64[D]')
        if dates.ndim != 1 or len(dates) < 2 or np.isnat(dates).any() or (np.diff(dates) <= np.timedelta64(0, "D")).any():
            raise ValueError("dates 必须是至少两天、严格递增且无 NaT 的日历")
        if not len(codes) or len(set(codes)) != len(codes):
            raise ValueError("codes 不能为空或重复")
        if set(panels) != set(FILES):
            raise ValueError(f"需要且仅支持七个面板：{FILES}")
        d = cls(None, dates, codes)
        for name, values in panels.items():
            a = _numeric_panel(values, name)
            if a.shape != (d.T, d.N):
                raise ValueError(f"{name} 必须是 ({d.T},{d.N}) 有限数/NaN 面板")
            _validate_panel(name, a)
            d._raw[name] = a.copy()
        return d

    # -- 原始面板 --------------------------------------------------------
    def _panel(self, name):
        a = self._raw.get(name)
        if a is None:
            p = os.path.join(self.cache_dir, f"{name}.npy")
            if self._mmap:
                a = np.load(p, mmap_mode="r")
            else:
                a = np.load(p)
            self._raw[name] = a
        return a

    def __getattr__(self, k):
        # 只有 7 个原始面板名会走到这里；其余一律 AttributeError，
        # 这样打错字会立刻报错，而不是莫名其妙变成"找不到 .npy"
        if k in FILES:
            a = Dataset._panel(self, k)
            object.__setattr__(self, k, a)      # 缓存成实例属性，下次直接命中
            return a
        raise AttributeError(
            f"Dataset 没有 {k!r}；原始面板只有 {', '.join(FILES)}"
        )

    # -- 派生面板的缓存入口 ----------------------------------------------
    def cached(self, key, builder):
        if key in self._c:
            return self._c[key]
        v = builder(self)
        self._c[key] = v
        return v

    # -- 复权 / 填充 -----------------------------------------------------
    @property
    def cumadj_ff(self):
        """复权因子前向填充。

        末两格（2026-10-16 的 601059.SH / 601198.SH）有价无因子。因为累计
        因子是单调的、只在除权日跳变，ffill 不是近似而是唯一合理的取值。
        """
        def build(d):
            factors = d.cumadj.astype(np.float64, copy=False)
            if np.isinf(factors).any() or (factors[np.isfinite(factors)] <= 0).any():
                raise ValueError("复权因子必须为正数或NaN，不能用零/负值抹掉持仓")
            return _ffill(factors)
        return self.cached("cumadj_ff", build)

    @property
    def close_ff(self):
        """估值参考价；缺价跨除权时先填复权价再还原，避免新股数乘旧每股价。"""
        return self.cached("close_ff", lambda d: d._filled_price(d.close))

    @property
    def open_ff(self):
        """前向填充的开盘价，在执行时点给整个组合估值用。"""
        return self.cached("open_ff", lambda d: d._filled_price(d.open))

    def _filled_price(self, raw):
        if getattr(self,'explicit_split_basis',False):
            # A missing post-dividend quote must not carry the old pre-dividend
            # price while the ledger separately owns the distribution.
            out=np.empty_like(raw,dtype=float); last=np.full(self.N,np.nan)
            for i in range(self.T):
                last=np.maximum(last-self._cash_dividends[i],0.)/self._share_ratios[i]
                valid=np.isfinite(raw[i]) & (raw[i]>0)
                last=np.where(valid,raw[i],last)
                out[i]=last
            return out
        valid=np.isfinite(raw) & (raw > 0)
        factor=self.cumadj_ff
        adjusted=_ffill(np.where(valid,raw*factor,np.nan))
        # 真实格保留原始精度，只有缺失格使用除权后的理论参考价；不制造成交报价。
        return np.where(valid,raw,adjusted/factor)

    @property
    def close_adj(self):
        """后复权收盘价 = 原始价 × 累计复权因子。

        方向已实测确认（不是除以因子）：除权日 |收益| 中位数从 2.53% 降到
        1.29%；用除法得 3.94%，比不复权还差。方向搞反是个静默的灾难性 bug。
        """
        def build(d):
            f = d.cumadj_ff
            return (d.close_ff.astype(np.float64) * f).astype(np.float64)
        return self.cached("close_adj", build)

    @property
    def open_adj(self):
        def build(d):
            return (d.open_ff.astype(np.float64) * d.cumadj_ff).astype(np.float64)
        return self.cached("open_adj", build)

    # -- 上市天数 --------------------------------------------------------
    @property
    def list_age(self):
        """自该股**首个有效报价**起的交易日数，int16，-1 = 还没上市。

        数据首日（2020-01-02）就有报价的股票记 32767（"很老"）。不这么干的话
        2020 年头 60 个交易日 `min_listed=60` 会把整个 universe 清空。
        """
        def build(d):
            valid = np.isfinite(d.close) & (d.close > 0)
            seen = np.maximum.accumulate(valid, axis=0)   # 到 t 为止是否已出现过
            first = np.argmax(valid, axis=0).astype(np.int32)
            age = np.clip(np.arange(d.T, dtype=np.int32)[:, None] - first[None, :],
                          0, 32767).astype(np.int16)
            age[~seen] = -1                                # 还没上市 / 从未有报价
            age[:, valid[0]] = 32767                      # 样本开始前已上市，整个样本均视为老股
            return age
        return self.cached("list_age", build)

    # -- 工具 ------------------------------------------------------------
    def col(self, code):
        """取一只股票的所有面板下标。"""
        return self.codes.index(code)

    def row_date(self, i):
        return self.dates[i]

    def __repr__(self):
        return (f"Dataset({self.T} 个交易日 × {self.N} 只股票, "
                f"{self.dates[0]} .. {self.dates[-1]}, 缓存={os.path.basename(self.cache_dir) if self.cache_dir else 'memory'})")

    # -- 多进程扫参时要能 pickle ------------------------------------------
    def __reduce__(self):
        if self.cache_dir is None:
            return (_from_arrays, (self.dates, self.codes, self._raw))
        return (_reopen, (self.cache_dir, self.dates, self.codes, self._mmap))


def _reopen(cache_dir, dates, codes, mmap):
    return Dataset(cache_dir, dates, codes, mmap=mmap)


def _from_arrays(dates, codes, panels):
    return Dataset.from_arrays(dates, codes, **panels)


def _ffill(a):
    """按列前向填充 NaN。前导 NaN 保持 NaN。

    做法：算出每个位置"最近一个有效观测"的行号（累加最大值），然后按行号
    取数。前导 NaN 的位置行号停留在 0，而 a[0] 本身是 NaN，所以天然保持
    NaN —— 不需要额外打补丁。
    """
    a = np.asarray(a)
    T, N = a.shape
    nan = np.isnan(a)
    if not nan.any():
        return a
    idx = np.where(~nan, np.arange(T, dtype=np.int64)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return a[idx, np.arange(N)[None, :]]


# ==========================================================================
# load
# ==========================================================================
def load(source=None, cache_dir=None, workers=None, mmap=False, rebuild=False,
         verbose=True):
    """加载课程数据，返回 `Dataset`。

    source    目录 / feathers.zip / 某个 .feather 文件；None 表示用项目里的 data/
    cache_dir .npy 缓存放哪；默认 data/_cache
    workers   并行解码的进程数；默认 min(4, CPU-1)
    mmap      True 时不把面板读进内存，用 memmap（省内存，但随机访问慢）
    rebuild   忽略已有缓存强制重建
    """
    t0 = time.time()
    feather_dir = _resolve_source(source)
    fp = fingerprint(feather_dir)
    cache_dir = os.path.join(cache_dir or DEFAULT_CACHE_DIR, fp)
    meta_path = os.path.join(cache_dir, "meta.json")
    lock_path = os.path.join(cache_dir, ".lock")

    meta = None if rebuild else _valid_cache_meta(cache_dir,fp)

    if meta is None:
        os.makedirs(cache_dir, exist_ok=True)
        # rebuild=True 时不要被"别人已经建好的 meta"骗过去
        waited = False if rebuild else _wait_for_meta(meta_path, lock_path, timeout=0.5)
        if waited:
            meta = _valid_cache_meta(cache_dir,fp)
        if meta is None:
            _acquire_lock(lock_path)          # 拿不到会一直等到超时再抛
            try:
                # 排队期间别人可能已经写完了，再确认一次
                meta = None if rebuild else _valid_cache_meta(cache_dir,fp)
                if meta is None:
                    # 重建时先撤销完成标记，不能让其他加载者把半成品当完成缓存。
                    if os.path.exists(meta_path):
                        os.remove(meta_path)
                    if verbose:
                        print(f"[qbt] 首次解码 {len(FILES)} 个 feather 文件 -> {cache_dir}")
                    meta = _build_cache(feather_dir, cache_dir, fp, workers, verbose)
            finally:
                if os.path.exists(lock_path):
                    os.remove(lock_path)

    dates = np.load(os.path.join(cache_dir, "dates.npy"))
    if verbose:
        print(f"[qbt] {meta['T']} 个交易日 × {meta['N']} 只股票，"
              f"{time.time()-t0:.2f}s 就绪")
    return Dataset(cache_dir, dates, meta["codes"], mmap=mmap, verbose=verbose)


def _build_cache(feather_dir, cache_dir, fp, workers, verbose):
    from concurrent.futures import ProcessPoolExecutor

    if workers is None:
        workers = min(len(FILES), max(2, (os.cpu_count() or 4) - 1), 6)

    tasks = [(f, os.path.join(feather_dir, f + ".feather"), cache_dir) for f in FILES]
    results = {}
    if workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for r in ex.map(_decode_one, tasks):
                    results[r[0]] = r
                    if verbose:
                        print(f"       {r[0]:11s} {r[2]}  {r[4]:.1f}s")
        except Exception as exc:                 # 子进程出问题就退回单进程
            if verbose:
                print(f"[qbt] 并行解码不可用（{exc}），改用单进程")
            results = {}
    if not results:
        for t in tasks:
            r = _decode_one(t)
            results[r[0]] = r
            if verbose:
                print(f"       {r[0]:11s} {r[2]}  {r[4]:.1f}s")

    codes = results[FILES[0]][1]
    dates = results[FILES[0]][3]
    for f in FILES:
        if results[f][1] != codes:
            raise RuntimeError(f"{f}.feather 的股票列顺序和其他文件不一致，数据有问题")
        if dates is None or results[f][3] is None or not np.array_equal(results[f][3], dates):
            raise RuntimeError(f"{f}.feather 的日期列或顺序与其他面板不一致")
    if dates is None:
        raise RuntimeError("读不到日期列")

    np.save(os.path.join(cache_dir, "dates.npy"), dates)

    # 边角校验：面板形状必须一致
    shapes = {f: results[f][2] for f in FILES}
    if len(set(shapes.values())) != 1:
        raise RuntimeError(f"7 个面板形状不一致: {shapes}")
    if np.ndim(dates) != 1 or len(dates) != shapes[FILES[0]][0] or (np.diff(dates) <= 0).any():
        raise RuntimeError("日期列必须与行情等长、严格递增且不重复")
    # 纳秒整数递增还不够：NaT的整数溢出可通过diff，同一天两根bar也不能发布为日频。
    from .calendar import Calendar
    try:
        Calendar(dates)
    except (ValueError, TypeError) as error:
        raise RuntimeError('日期列不能构成合法日频交易日历，未发布缓存') from error
    if min(shapes[FILES[0]]) < 1 or len(dates) < 2:
        raise RuntimeError('日频缓存至少需要两天及一个证券，未发布缓存')
    if fingerprint(feather_dir) != fp:
        raise RuntimeError("解码过程中源数据发生变化，未发布缓存；请在数据导出完成后重试")

    meta = {
        "fingerprint": fp, "reader_version": READER_VERSION,
        "T": int(shapes[FILES[0]][0]), "N": int(shapes[FILES[0]][1]),
        "codes": codes, "files": list(FILES),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": feather_dir,
    }
    tmp = os.path.join(cache_dir, "meta.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)
    os.replace(tmp, os.path.join(cache_dir, "meta.json"))
    return meta
