"""从 Python 文件或模块输入一个策略，并输出回测报告与成交流水。"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import sys
import time
import tempfile

import qbt

ROOT = Path(__file__).resolve().parent


def load_strategy(spec):
    """支持 path/to/file.py:function，也支持 package.module:function。"""
    source, sep, name = spec.rpartition(":")
    if not sep or not source or not name:
        raise ValueError("策略格式为 文件.py:函数名 或 模块名:函数名")
    if source.endswith(".py"):
        path = Path(source).resolve()
        module_spec = importlib.util.spec_from_file_location("qbt_user_strategy", path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        # 显式输入文件必须执行本次字节，不能被同秒/同尺寸旧pyc缓存替换。
        exec(compile(path.read_bytes(),str(path),'exec',dont_inherit=True),module.__dict__)
    else:
        module = importlib.import_module(source)
        path = Path(module.__file__).resolve()
    strategy = getattr(module, name)
    if not callable(strategy):
        raise TypeError(f"{spec} 不是可调用的策略函数")
    return strategy, path


def parse_params(items):
    params = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError("参数格式为 --param 名称=JSON值，例如 --param k=100")
        params[key] = json.loads(value)
    return params


def make_run_directory(base):
    """每次实验独立目录；随机排他创建避免同秒并行覆盖CSV和summary。"""
    base=Path(base).resolve()
    base.mkdir(parents=True,exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='run_',dir=base))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default=str(ROOT / "strategies/five_day_reversal.py") + ":five_day_reversal")
    parser.add_argument("--data", default=None, help="feathers目录或zip；默认使用项目data")
    parser.add_argument('--market',choices=['CN','US','KR'],help='显式市场配置；需同时提供 --reference')
    parser.add_argument('--reference',help='PIT证券资料、交易/交割日历、权益和费用 JSON 清单')
    parser.add_argument("--start", default="2022-01-04")
    parser.add_argument("--end", default="2025-06-30")
    parser.add_argument("--capital", type=float, default=1e7)
    parser.add_argument("--param", action="append", default=[], help="可重复，例如 --param k=100")
    parser.add_argument("--output", default=str(ROOT / "reports/5dr"))
    parser.add_argument("--health", action="store_true", help="额外运行前缀及账本体检")
    args = parser.parse_args(argv)
    strategy, source = load_strategy(args.strategy)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    params = parse_params(args.param)
    out = make_run_directory(args.output)
    started = time.monotonic()
    if bool(args.market)!=bool(args.reference):
        raise ValueError('--market 和 --reference 必须同时提供')
    d = (qbt.load_market_csv(args.data,args.market) if args.market and args.data and Path(args.data).suffix.lower()=='.csv'
         else qbt.load(args.data))
    options = dict(start=args.start, end=args.end, capital=args.capital, params=params)
    if args.market:
        options.update(market_profile=args.market,reference_data=qbt.ReferenceData.from_json(args.reference))
    result = qbt.run(d, strategy, **options)
    report = qbt.report(result, path=str(out))
    with (out / "trades.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["日期", "证券代码", "方向", "数量", "价格", "费用"])
        for i, j, side, qty, price, fee in result.trades:
            writer.writerow([str(d.dates[i]), d.codes[j], "买入" if side > 0 else "卖出", qty, price, fee])
    summary = dict(strategy=args.strategy, source=str(source),
        source_sha256=source_sha256,
        execution_provenance=result.execution_provenance,
        params=params, start=args.start, end=args.end, capital=args.capital,
        metrics={k: None if isinstance(v, float) and not math.isfinite(v) else v for k,v in result.metrics().items()},
        assumptions=result.assumptions, report=str(report.path))
    if args.market:
        summary.update(market=result.market_profile,currency=result.currency,reference=result.reference_provenance,
                       unsettled_cash=result.unsettled_cash,withdrawable_cash=result.withdrawable_cash)
    if args.health:
        health = qbt.health(d, strategy, quick=True, **options)
        (out / "health.txt").write_text(str(health), encoding="utf-8")
        summary["health_ok"] = health.ok
        print(health)
    summary["seconds"] = round(time.monotonic()-started, 2)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(result)
    print("策略源文件：", source)
    print("报告：", report.path)
    print("成交流水：", out / "trades.csv")
    return 1 if summary.get("health_ok") is False else 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    sys.exit(main())
