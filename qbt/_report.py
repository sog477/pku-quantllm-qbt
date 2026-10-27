"""回测报告：一个自包含 HTML，双击就能看，发给别人也不会缺图。

设计上的取舍
------------
* **自包含**。plotly 的 JS 直接 inline 进去（约 3MB）。换来的是"发给老师/
  同学，双击就能打开"，不用起 http server、不用管相对路径。对一份作业报告，
  这个交换是划算的。
* **有 plotly 就用 plotly，没有就降级到内置 SVG**。硬依赖只有 numpy，
  画图库是加分项——不能让 `import qbt` 因为缺一个画图库就整个挂掉。
* **报告只读 `Result`**，不碰引擎。所以它可以随手喂一个从硬盘 pickle 回来
  的结果，也可以多进程里生成。

报告里有两块是**这个系统特有**的，不是通用绩效报告：

1. 【涨停/跌停挡掉了什么】—— 老师那道题的答案。纸面回测（`limit_rule="off"`）
   和真实口径的净值差，要能被"被挡住的买单 × 那些票当天的涨幅"解释掉。
2. 【被套住的卖出】—— 跌停卖不掉比涨停买不到更伤，最大回撤通常来自这里。
"""

from __future__ import annotations

import html as _html
import os

import numpy as np

from .metrics import metrics as _metrics
from .metrics import monthly_table, yearly_table, drawdown, TRADING_DAYS

__all__ = ["report", "to_html", "Report"]


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def _pct(x, nd=2):
    return "-" if x is None or not np.isfinite(x) else f"{x * 100:.{nd}f}%"


def _num(x, nd=2):
    return "-" if x is None or not np.isfinite(x) else f"{x:,.{nd}f}"


def _money(x):
    return "-" if x is None or not np.isfinite(x) else f"{x:,.0f}"


def _cls(x, invert=False):
    """涨绿跌红：A 股习惯（和欧美相反）。"""
    if x is None or not np.isfinite(x):
        return ""
    good = x < 0 if invert else x > 0
    return "up" if good else ("dn" if x != 0 else "")


def _table(rows, cls="grid"):
    out = [f'<table class="{cls}">']
    for r in rows:
        if r is None:
            out.append('<tr class="sep"><td colspan="99"></td></tr>')
            continue
        head, body = r
        cells = "".join(f"<th>{c}</th>" for c in head)
        out.append(f"<thead><tr>{cells}</tr></thead><tbody>")
        for row in body:
            tds = []
            for c in row:
                if isinstance(c, tuple):
                    txt, klass = c
                    tds.append(f'<td class="{klass}">{txt}</td>')
                else:
                    tds.append(f"<td>{c}</td>")
            out.append("<tr>" + "".join(tds) + "</tr>")
        out.append("</tbody>")
    out.append("</table>")
    return "".join(out)


def _fig(fig, first):
    """把一个 plotly figure 变成嵌入用的 div。第一个负责 inline 整个 JS。"""
    import plotly.io as pio
    return pio.to_html(fig, include_plotlyjs=("inline" if first else False),
                       full_html=False,
                       config={"displayModeBar": False, "responsive": True})


def _svg_lines(series, labels, width=980, height=260, colors=None):
    """降级用的折线图：纯 inline SVG，不依赖任何库。"""
    colors = colors or ["#2b6cb0", "#a0aec0", "#e53e3e", "#38a169"]
    arrs = [np.asarray(s, dtype=np.float64) for s in series]
    n = max(len(a) for a in arrs)
    lo = min(float(np.nanmin(a)) for a in arrs)
    hi = max(float(np.nanmax(a)) for a in arrs)
    if not np.isfinite(lo) or hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad

    def xy(a):
        if len(a) < 2:
            return ""
        xs = np.linspace(0, width - 60, len(a)) + 50
        ys = height - 24 - (a - lo) / (hi - lo) * (height - 48)
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    parts = [f'<svg class="fallback" viewBox="0 0 {width} {height}" '
             f'preserveAspectRatio="none">',
             f'<rect x="50" y="10" width="{width - 60}" height="{height - 34}" '
             f'fill="none" stroke="#e2e8f0"/>']
    for a, lab, c in zip(arrs, labels, colors):
        parts.append(f'<polyline fill="none" stroke="{c}" stroke-width="1.6" '
                     f'points="{xy(a)}"/>')
    parts.append(f'<text x="8" y="18" font-size="11" fill="#718096">{hi:,.3g}</text>')
    parts.append(f'<text x="8" y="{height - 16}" font-size="11" fill="#718096">{lo:,.3g}</text>')
    for i, (lab, c) in enumerate(zip(labels, colors)):
        parts.append(f'<text x="{60 + i * 150}" y="{height - 6}" font-size="12" '
                     f'fill="{c}">■ {_html.escape(lab)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# 各板块
# --------------------------------------------------------------------------
def _metric_rows(m, bench_label="基准(全市场等权)"):
    """老师赞赏的那种"绩效指标矩阵"。"""

    def g(k):
        return m.get(k, float("nan"))

    rows = [
        ("收益", [
            ("累计收益", _pct(g("累计收益")), _cls(g("累计收益"))),
            ("年化收益", _pct(g("年化收益")), _cls(g("年化收益"))),
            (bench_label, _pct(g("基准累计")), _cls(g("基准累计"))),
            ("超额收益(累计)", _pct(g("超额累计")), _cls(g("超额累计"))),
            ("超额收益(年化)", _pct(g("超额年化")), _cls(g("超额年化"))),
        ]),
        ("风险", [
            ("年化波动", _pct(g("年化波动")), ""),
            ("最大回撤", _pct(g("最大回撤")), "dn" if np.isfinite(g("最大回撤")) else ""),
            ("Beta", _num(g("Beta"), 3), ""),
            ("跟踪误差", _pct(g("跟踪误差")), ""),
        ]),
        ("风险调整", [
            ("夏普比率", _num(g("夏普"), 3), _cls(g("夏普"))),
            ("索提诺比率", _num(g("索提诺"), 3), _cls(g("索提诺"))),
            ("卡玛比率", _num(g("卡玛"), 3), _cls(g("卡玛"))),
            ("信息比率", _num(g("信息比率"), 3), _cls(g("信息比率"))),
        ]),
        ("交易", [
            ("日胜率", _pct(g("日胜率"), 1), ""),
            ("月度胜率", _pct(g("月度胜率"), 1), ""),
            ("收益未定义区间数", _num(g("收益未定义区间数"), 0), ""),
            ("换手率(年化)", _num(g("换手率(年)"), 2), ""),
            ("成交笔数", f"{int(g('成交笔数')):,}" if np.isfinite(g("成交笔数")) else "-", ""),
            ("平均持仓数", _num(g("平均持仓数"), 1), ""),
            ("平均仓位", _pct(g("平均仓位"), 1), ""),
        ]),
        ("资金与成本", [
            ("初始资金", _money(g("初始资金")), ""),
            ("期末资产", _money(g("期末资产")), _cls(g("期末资产") - g("初始资金"))),
            ("总费用", _money(g("总费用")), ""),
            ("权益税款（另列）", _money(g('权益税款')), ''),
            ("应付税款", _money(g('应付税款')), ''),
            ("费用占初始", _pct(g("费用占初始")), ""),
            ("费用占期末", _pct(g("费用占期末")), ""),
        ]),
    ]
    out = []
    for title, items in rows:
        out.append(([title, "数值"], [(k, v) for k, v, _c in items]))
    return out


def _monthly_heatmap(r):
    tab = monthly_table(r, as_dict=True)
    if not tab:
        return None
    keys = sorted(tab)
    years = sorted({k.astype("datetime64[Y]").astype(int) + 1970 for k in keys})
    z, text = [], []
    for y in years:
        zr, tr = [], []
        for mo in range(1, 13):
            k = np.datetime64(f"{y:04d}-{mo:02d}", "M")
            v = tab.get(k, np.nan)
            zr.append(v if np.isfinite(v) else None)
            tr.append("" if not np.isfinite(v) else f"{v * 100:.1f}")
        z.append(zr)
        text.append(tr)
    import plotly.graph_objects as go
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{m}月" for m in range(1, 13)], y=[str(y) for y in years],
        text=text, texttemplate="%{text}", textfont={"size": 11},
        colorscale=[[0.0, "#1a7a3c"], [0.5, "#f7fafc"], [1.0, "#c53030"]],
        zmid=0, showscale=False, hoverongaps=False))
    fig.update_layout(title="月度收益热力图 (%)", height=110 + 34 * len(years),
                      margin=dict(l=60, r=20, t=50, b=40),
                      yaxis=dict(autorange="reversed"))
    return fig


def _nav_fig(r):
    import plotly.graph_objects as go
    nav = np.asarray(r.nav, dtype=np.float64)
    ben = np.asarray(r.bench_nav, dtype=np.float64)
    ben = ben / ben[0] if ben.size and ben[0] else ben
    k = min(nav.size, ben.size)
    ex = nav[:k] / np.maximum(ben[:k], 1e-12)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.dates, y=nav, name="策略", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=r.dates[:k], y=ben[:k], name="基准(全市场等权)",
                             line=dict(width=1.4, dash="dot")))
    fig.add_trace(go.Scatter(x=r.dates[:k], y=ex, name="超额(策略/基准)",
                             line=dict(width=1.2)))
    fig.update_layout(title="净值曲线", height=430, hovermode="x unified",
                      margin=dict(l=60, r=20, t=50, b=40),
                      legend=dict(orientation="h", y=1.08))
    return fig


def _dd_fig(r):
    import plotly.graph_objects as go
    dd, _ = drawdown(r.equity)
    fig = go.Figure(go.Scatter(x=r.dates, y=dd, name="回撤",
                               fill="tozeroy", line=dict(width=1, color="#c53030")))
    fig.update_layout(title="回撤", height=250, margin=dict(l=60, r=20, t=50, b=40),
                      yaxis=dict(tickformat=".0%"), showlegend=False)
    return fig


def _expo_fig(r):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.dates, y=np.asarray(r.gross, float), name="仓位",
                             line=dict(width=1.4), yaxis="y"))
    fig.add_trace(go.Scatter(x=r.dates, y=np.asarray(r.n_positions, float),
                             name="持仓只数", line=dict(width=1.2, dash="dot"),
                             yaxis="y2"))
    fig.update_layout(title="仓位与持仓只数", height=300,
                      margin=dict(l=60, r=60, t=50, b=40),
                      legend=dict(orientation="h", y=1.1),
                      yaxis=dict(tickformat=".0%"),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False))
    return fig


def _blocked_section(r):
    """老师那道题的答案面板。"""
    ba = np.asarray(r.blocked_buy_amt, dtype=np.float64)
    bn = np.asarray(r.blocked_buy_n)
    bb = np.asarray(r.blocked_buy_bounce, dtype=np.float64)
    sa = np.asarray(r.blocked_sell_amt, dtype=np.float64)
    sn = np.asarray(r.blocked_sell_n)

    tot_n, tot_amt = int(bn.sum()), float(ba.sum())
    days = int((bn > 0).sum())
    pnl = float(np.nansum(ba * bb))
    wavg = float(np.nansum(ba * bb) / tot_amt) if tot_amt > 0 else float("nan")
    cap = float(np.asarray(r.equity, dtype=np.float64)[0])

    # 符号的诚实处理，和 `metrics.compare` 一个道理：被挡住的这批票当天
    # 平均可能是**跌**的。这时候「买不到」是帮你躲开了，不是损失 ——
    # 印成「本来能赚却没赚到 -94 万」等于把一个正的收益写成了负的。
    if pnl >= 0:
        hit_label, hit_val, hit_cls = "=> 本来能赚却没赚到", pnl, "up"
        hit_note = None
    else:
        hit_label, hit_val, hit_cls = "=> 没买到反而躲掉亏损", -pnl, "up"
        hit_note = ("符号说明：被挡的这批票当天平均是<b>跌</b>的 —— "
                    "「涨停买不到」在这里不是损失，是保护。涨停次日的"
                    "开盘价通常就是别人出货给你的价。")

    rows = [("被挡下的买单", [
        ("笔数", f"{tot_n:,} 笔  （分布在 {days:,} 个交易日）", ""),
        ("被挡资金合计", _money(tot_amt), ""),
        ("被挡资金占初始", _pct(tot_amt / cap), ""),
        ("被挡个股当日平均涨幅", _pct(wavg), _cls(wavg)),
        (hit_label, _money(hit_val), hit_cls),
        ("=> 占初始资金", _pct(hit_val / cap), hit_cls),
    ])]
    if hit_note:
        rows.append(("", [("", hit_note, "")]))
    s_n, s_amt = int(sn.sum()), float(sa.sum())
    rows.append(("被挡下的卖单（被套住）", [
        ("笔数", f"{s_n:,} 笔", ""),
        ("被套住的市值合计", _money(s_amt), ""),
        ("被套住市值占初始", _pct(s_amt / cap), "dn"),
    ]))
    return _table(rows)


def _exit_section(r):
    parts=[]
    if hasattr(r, 'target_attempts'):
        from .research import residual_inventory
        remainder = residual_inventory(r)
        residual_value = sum(row['marked_value'] for row in remainder)
        incomplete = r.target_attempts
        parts.append("<h2>退出与未完成目标</h2><div class='card'>")
        parts.append("<p>末日现金 " + _money(r.cash[-1]) + "；剩余库存 " +
            str(len(remainder)) + " 只，模型估值 " + _money(residual_value) +
            "。目标归零不等于卖出成交，剩余估值未计为清仓现金。</p>")
        parts.append("<p>未完成目标尝试 " + str(len(incomplete)) +
            " 次；退出安排主动禁止的买入请求 " +
            str(getattr(r,'scheduled_suppressed_buy_requests',0)) +
            " 次。约束标记可能同时成立，不是互斥的原因归因；目标尝试也不是交易所受理记录。</p>")
        if remainder:
            parts.append(_table([(["证券", "剩余股数", "模型估值", "冻结股", "限制股"], [
                (_html.escape(row['code']), str(row['shares']), _money(row['marked_value']),
                 str(row['frozen']), str(row['restricted'])) for row in remainder[:20]])]))
            if len(remainder)>20:parts.append('<p>这里只显示前20只，完整库存见导出的residual.csv。</p>')
        parts.append('</div>')

    return ''.join(parts)


def _contrib_section(r, ds, top=15):
    acct = r.account
    n = ds.N
    pnl = np.asarray(acct.pnl, dtype=np.float64).copy()
    # 未实现部分：按最后一个交易日的收盘价估
    i1 = r.window[1]
    last = np.asarray(getattr(r, "final_prices", ds.close_ff[i1]), dtype=np.float64)
    entry = acct.cost_raw
    unreal = acct.shares * (last - entry)
    total = pnl + np.nan_to_num(unreal, nan=0.0)
    fees = None
    if r.trades is not None:
        fees = np.zeros(n)
        for trade in r.trades:
            fees[int(trade[1])] += float(trade[5])
        total -= fees
    idx = np.argsort(-np.abs(total))[:top]
    eq0 = float(np.asarray(r.equity, float)[0])
    body = []
    for j in idx:
        if not np.isfinite(total[j]) or total[j] == 0:
            continue
        body.append([
            _html.escape(str(ds.codes[j])),
            _money(pnl[j]),
            _money(unreal[j] if np.isfinite(unreal[j]) else 0.0),
            _money(fees[j]) if fees is not None else "-",
            (_money(total[j]), _cls(total[j])),
            (_pct(total[j] / eq0), _cls(total[j])),
        ])
    if not body:
        return "<p class='muted'>没有可归因的个股损益。</p>"
    note = ("" if fees is not None else "<p class='muted'>未记录逐笔交易，无法分配费用；下表显示毛损益。</p>")
    return note + _table([(["代码", "已实现（毛）", "未实现（毛）", "费用", "净合计" if fees is not None else "毛合计", "占初始资金"], body)])


def _yearly_section(r):
    y = yearly_table(r)
    yb = yearly_table(r, use_bench=True)
    if not y:
        return "<p class='muted'>区间太短，没有年度数据。</p>"
    body = []
    for k in sorted(y):
        a, b = y[k], yb.get(k, float("nan"))
        body.append([str(k), (_pct(a), _cls(a)), _pct(b),
                     (_pct(a - b), _cls(a - b))])
    return _table([(["年份", "策略", "基准", "超额"], body)])


def _header(r, mm):
    bits = [
        f"区间 {r.dates[0]} ~ {r.dates[-1]}",
        f"{mm.get('交易日数', 0):,} 个估值区间",
        f"初始资金 {_money(mm.get('初始资金'))}",
        f"涨跌停口径 <code>{_html.escape(str(r.limit_rule))}</code>",
        f"成交模型 <code>{_html.escape(getattr(r,'execution_name','open/close 两阶段' if r.limit_rule == 'broken_board_close' else 'open'))}</code>",
        f"买卖容量 {_html.escape(getattr(r, 'capacity_mode', 'realized_day'))} {r.max_adv_pct:.1%}",
        f"除权口径 <code>{_html.escape(str(r.div_policy))}</code>",
    ]
    if r.note:
        bits.append(_html.escape(str(r.note)))
    return " · ".join(bits)


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
class Report:
    """`to_html()` 的返回值，`repr` 直接在 notebook 里渲染。"""

    def __init__(self, html, path=None):
        self.html = html
        self.path = path

    def _repr_html_(self):
        return self.html

    def __repr__(self):
        return (f"Report({self.path})" if self.path else
                f"Report({len(self.html):,} 字节 HTML；在 Jupyter 里直接显示，"
                f"或 report(...) 传 path= 存盘)")

    def save(self, path, *, overwrite=True):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w" if overwrite else "x", encoding="utf-8") as f:
            f.write(self.html)
        self.path = path
        return path


_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 32px 64px; background: #f7fafc; color: #1a202c;
       font: 14px/1.6 -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 34px 0 10px; padding-bottom: 6px;
     border-bottom: 2px solid #e2e8f0; }
.meta { color: #718096; font-size: 13px; margin-bottom: 8px; }
code { background: #edf2f7; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
.muted { color: #a0aec0; }
.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 16px 20px; margin-bottom: 16px; }
table.grid { border-collapse: collapse; width: 100%; background: #fff;
             margin-bottom: 8px; }
table.grid th, table.grid td { border-bottom: 1px solid #edf2f7; padding: 7px 10px;
                               text-align: right; }
table.grid th { background: #f7fafc; color: #4a5568; font-weight: 600;
                font-size: 12.5px; letter-spacing: .3px; }
table.grid td:first-child, table.grid th:first-child { text-align: left; }
table.grid thead:not(:first-child) th { border-top: 1px solid #e2e8f0; }
.up { color: #c53030; font-variant-numeric: tabular-nums; }   /* A股：红涨 */
.dn { color: #1a7a3c; font-variant-numeric: tabular-nums; }   /* A股：绿跌 */
table.grid td { font-variant-numeric: tabular-nums; }
.cols { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
.cols > * { flex: 1 1 420px; min-width: 340px; }
svg.fallback { width: 100%; height: 260px; }
footer { color: #a0aec0; font-size: 12px; margin-top: 40px; }
"""


def to_html(r, title=None, plotly=True):
    """把 `Result` 变成自包含 HTML 字符串。"""
    mm = _metrics(r)
    ds = r.ds
    title = title or f"{r.name} · 回测报告"
    parts = [
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{_html.escape(title)}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>{_html.escape(str(r.name))}</h1>",
        f"<div class='meta'>{_header(r, mm)}</div>",
        "<h2>绩效指标</h2><div class='card'>",
        _table(_metric_rows(mm)),
        "</div>",
    ]

    if hasattr(r,'currency'):
        parts.append("<div class='card'><h2>市场与账户</h2><p>"+_html.escape(
            f"{r.market_profile} · {r.currency} · {r.timezone}；本报告所有金额均以 {r.currency} 计价。")+
            f"</p><p>未交割卖出款：{_money(getattr(r,'unsettled_cash',None))}；可提取现金：{_money(getattr(r,'withdrawable_cash',None))}；"
            f"累计权益税款：{_money(r.account.taxes)}；应付税款：{_money(r.account.tax_payables.sum())}。</p></div>")
    provenance=getattr(r,'execution_provenance',{})
    if provenance:
        parts.append('<details class="card"><summary>实际执行代码身份</summary><ul>')
        for role,info in provenance.items():
            label={'strategy':'策略','engine':'引擎入口','market_adapter':'市场适配器'}.get(role,role)
            loaded=info.get('loaded_code_sha256') or '无Python code object'
            status={True:'入口代码体与磁盘一致',False:'磁盘源码与已加载代码不一致',None:'无文件或动态定义，无法比对'}[info.get('source_match')]
            parts.append('<li>'+_html.escape(f'{label}：{loaded}；{status}')+'</li>')
            if info.get('source_match') is False:
                parts.append('<p class="dn">'+_html.escape(f'{label}磁盘源码与已加载代码不一致；本报告保留的是实际运行对象，请确认是否需要重载模块或重启kernel。')+'</p>')
        parts.append('</ul><p>这里只核对入口代码对象，不能证明外部数据、闭包状态和全部依赖版本一致。</p></details>')

    parts.append("<div class='card'><h2>假设与数据口径</h2><ul>" + "".join(
        "<li>" + _html.escape(str(x)) + "</li>" for x in getattr(r, "assumptions", ())) + "</ul>" +
        f"<p>期末应收分红及其他权益：{_money(float(r.account.receivables.sum()))}；"
        f"长期无成交持仓旧价市值：{_money(float(getattr(r, 'stale_value', [0])[-1]))}。"
        "历史测试通过表示已覆盖案例符合预期，不构成无前视或可成交的完整证明。</p></div>")

    have_plotly = False
    if hasattr(r,"orders"):
        rows = [[o.id,_html.escape(o.request.code),_html.escape(o.request.side),o.request.quantity,
                 o.filled,o.remaining,_html.escape(o.status),_html.escape(o.reason),_money(o.fee)]
                for o in r.orders[:100]]
        parts.append("<div class='card'><h2>订单生命周期</h2>" +
            f"<p>共 {len(r.orders)} 笔订单，展示前100笔；完整记录见 Result.orders / order_events / order_fills。</p>" +
            _table([(["ID","代码","方向","申报","成交","剩余","状态","原因","费用"],rows)]) + "</div>")
    if plotly:
        try:
            import plotly  # noqa: F401
            have_plotly = True
        except ImportError:
            have_plotly = False

    if have_plotly:
        first = True
        for maker in (_nav_fig, _dd_fig):
            parts.append(f"<div class='card'>{_fig(maker(r), first)}</div>")
            first = False
        # 月度热力图 + 仓位图并排
        heat = _monthly_heatmap(r)
        parts.append("<div class='cols'>")
        if heat is not None:
            parts.append(f"<div class='card'>{_fig(heat, False)}</div>")
        parts.append(f"<div class='card'>{_fig(_expo_fig(r), False)}</div>")
        parts.append("</div>")
    else:
        nav = np.asarray(r.nav, float)
        ben = np.asarray(r.bench_nav, float)
        ben = ben / ben[0] if ben.size and ben[0] else ben
        parts.append("<div class='card'><h2 style='border:0;margin-top:0'>净值曲线"
                     "（降级模式：未装 plotly）</h2>"
                     + _svg_lines([nav, ben], ["策略", "基准(全市场等权)"])
                     + "</div>")
        dd, _ = drawdown(r.equity)
        parts.append("<div class='card'><h2 style='border:0;margin-top:0'>回撤</h2>"
                     + _svg_lines([dd], ["回撤"], colors=["#c53030"]) + "</div>")
        tab = monthly_table(r, as_dict=True)
        if tab:
            keys = sorted(tab)
            years = sorted({k.astype("datetime64[Y]").astype(int) + 1970 for k in keys})
            body = []
            for y in years:
                row = [str(y)]
                for mo in range(1, 13):
                    v = tab.get(np.datetime64(f"{y:04d}-{mo:02d}", "M"), np.nan)
                    row.append((_pct(v, 1), _cls(v)))
                body.append(row)
            parts.append("<div class='card'><h2 style='border:0;margin-top:0'>"
                         "月度收益</h2>"
                         + _table([(["年"] + [f"{m}月" for m in range(1, 13)], body)])
                         + "</div>")
        # 降级模式也得有仓位/持仓数。少了这块，报告会比 plotly 版**静默地**
        # 少一节 —— 看图的人不会知道少了什么，只会以为这个策略没有仓位图。
        # 两个量纲差两个数量级（0.98 vs 160 只），塞进一张共享纵轴的图里
        # 持仓数会把仓位压成一条直线，所以分成两张。
        gross = np.asarray(r.gross, dtype=np.float64)
        npos = np.asarray(r.n_positions, dtype=np.float64)
        parts.append("<div class='cols'>")
        parts.append("<div class='card'><h2 style='border:0;margin-top:0'>仓位</h2>"
                     + _svg_lines([gross], ["仓位"], height=220)
                     + f"<p class='muted'>平均 {_pct(np.nanmean(gross[1:]))}"
                       f"（总额 / 权益）</p></div>")
        parts.append("<div class='card'><h2 style='border:0;margin-top:0'>持仓只数</h2>"
                     + _svg_lines([npos], ["持仓只数"], height=220,
                                  colors=["#38a169"])
                     + f"<p class='muted'>平均 {_num(np.nanmean(npos[1:]), 1)} 只</p>"
                       "</div>")
        parts.append("</div>")

    parts += [
        "<h2>涨停 / 跌停挡掉了什么 —— 老师那道题</h2>",
        "<div class='card'>",
        "<p class='muted'>纸面口径（<code>limit_rule=\"off\"</code>）假装涨停也买得到。"
        "把真实口径重跑一遍，两版的净值差应当能被下面这张表<b>解释个大概</b>："
        "<b>被挡资金 × 那些票当天的涨幅</b>就是这个价差的机会成本部分。"
        "注意两点：第一，它不是恒等式 —— 被挡的钱没消失，它会重新投出去，"
        "两条路径的现金/持仓从此分岔，精确恒等式在 "
        "<code>tests/test_accounting.py</code>；第二，这个乘积<b>可能是负的</b>，"
        "那说明「买不到」当时是在保护你，不是损失。</p>",
        _blocked_section(r),
        "</div>",
        "<h2>年度收益</h2><div class='card'>", _yearly_section(r), "</div>",
        "<h2>个股损益贡献（前 15）</h2><div class='card'>",
        _contrib_section(r, ds), "</div>", _exit_section(r),
    ]

    acct = r.account
    snap = acct.snapshot()
    parts.append("<h2>账本自检</h2><div class='card'>")
    parts.append(_table([(["项目", "数值"], [
        ("现金", (_money(snap["cash"]), "")),
        ("期末总资产", _money(snap["equity"])),
        ("持仓只数", f"{snap['n_positions']:,}"),
        ("已实现盈亏（未扣交易费用）", _money(snap["realized"])),
        ("累计成交额", _money(snap["turnover"])),
        ("累计费用", _money(snap["fees"])),
        ("买入 / 卖出笔数", f"{snap['n_buy']:,} / {snap['n_sell']:,}"),
        ("除权除息事件", f"{snap['n_div_events']:,} 次（新增 {snap['shares_issued']:,} 股）"),
        ("已知清算事件", f"{acct.n_force_close:,} 笔，损益 {_money(acct.pnl_delisted)}"),
        ("应收分红（未到可用现金）", _money(snap["应收分红"])),
        ("现金非负", "是" if np.all(np.asarray(r.cash) >= -1e-7) else "否"),
        ("账户检查规则", _html.escape(getattr(r,"account_rule_name","CashAccountRules"))),
    ])]))
    parts.append("</div>")

    parts.append("<footer>qbt —— 可扩展回测系统 · 本次成本组件：" +
                 _html.escape(getattr(r, "cost_description", "自定义成本")) + "</footer>")
    parts.append("</body></html>")
    return "".join(parts)


def report(r, path=None, title=None, plotly=True, open_browser=False):
    """生成回测报告。

    r       `qbt.run()` 的返回值
    path    存盘路径（.html）。给目录则自动命名。为 None 则只在内存里返回，
            Jupyter 里直接当表达式求值就能渲染。
    """
    html = to_html(r, title=title, plotly=plotly)
    rep = Report(html)
    if path:
        directory = os.path.isdir(path) or not os.path.splitext(path)[1]
        if directory:
            os.makedirs(path, exist_ok=True)
            last = r.dates[-1]
            stamp_text = str(last) if np.datetime_data(np.asarray(last).dtype)[0] == "D" else np.datetime_as_string(last,unit="s")
            stamp = "".join(c for c in stamp_text if c.isalnum())
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(r.name))
            stem = os.path.join(path, f"{safe}_{stamp}")
            serial = 1
            while True:
                path = stem + (f"_{serial}" if serial > 1 else "") + '.html'
                try:
                    rep.save(path,overwrite=False)  # 原子排他创建，多个实验不能互相覆盖。
                    break
                except FileExistsError:
                    serial += 1
        else:
            rep.save(path)
        if open_browser:
            import webbrowser
            webbrowser.open("file://" + os.path.abspath(path))
    return rep
