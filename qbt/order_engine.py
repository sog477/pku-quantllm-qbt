"""与目标组合入口并列的订单入口，共享账本、撮合容量、时钟及规则组件。"""
import numpy as np
from .account import Account
from .broker import Broker
from .context import Context
from .costs import Costs
from .engine import Result
from .clocks import DailyClock, clock_events, benchmark_at_steps, row_event
from .rules import AStockRules, CashAccountRules
from .orders import OrderRequest, Cancel, OrderAPI, OrderBook, OpenExecution
from .strategy import HOLD
from .tradability import build_market
from .provenance import callable_provenance
from .metrics import equity_returns


from .markets import market_entry


@market_entry
def run_orders(ds, strategy, start=None, end=None, capital=1e6, *, params=None,
               market=None, market_kw=None, clock=None, asset_rules=None, account_rules=None,
               execution=None, costs=None, limit_rule="open_limit", max_adv_pct=.02,
               capacity_mode="lagged_adv", adv_window=20, account_factory=Account,
               broker_factory=Broker, corporate_actions=None, div_policy=None,
               max_stale_days=60, stale_policy="write_down", name=None):
    if not np.isfinite(capital) or capital <= 0 or max_stale_days < 0:
        raise ValueError("capital 必须为正，max_stale_days 必须非负")
    if stale_policy not in ("hold", "write_down"):
        raise ValueError("stale_policy 为 hold / write_down")
    if limit_rule not in ("open_limit", "off"):
        raise ValueError("订单入口用 execution 选择撮合模型，limit_rule 使用 open_limit/off")
    clock = DailyClock() if clock is None else clock
    rules = CashAccountRules() if account_rules is None else account_rules
    if callable(getattr(rules,'reset',None)):
        rules.reset()
    assets = AStockRules() if asset_rules is None else asset_rules
    execution = OpenExecution() if execution is None else execution
    events, indices = clock_events(clock,ds,start,end)
    m = market if market is not None else build_market(ds,log=None,**({} if market_kw is None else market_kw))
    if m.ds is not ds:
        raise ValueError('market绑定的是另一份Dataset；请为当前数据重新构建market，不能复用旧行情对象')
    account = account_factory(capital,ds.N)
    broker = broker_factory(m,costs=Costs() if costs is None else costs,limit_rule=limit_rule,
        max_adv_pct=max_adv_pct,capacity_mode=capacity_mode,adv_window=adv_window,
        asset_rules=assets,account_rules=rules)
    book = OrderBook(ds,broker,account,execution)
    provenance=dict(strategy=callable_provenance(strategy),engine=callable_provenance(getattr(run_orders,'__wrapped__',run_orders)))
    api = OrderAPI(book)
    if div_policy is None:
        div_policy = "events" if corporate_actions is not None else "reinvest"
    if div_policy not in ("events","reinvest","ignore") or ((corporate_actions is not None) != (div_policy == "events")):
        raise ValueError("显式 corporate_actions 对应 events；否则使用 reinvest/ignore")
    actions = corporate_actions.prepare(ds) if corporate_actions is not None else None
    if callable(getattr(strategy,"reset",None)):
        strategy.reset()
    n = len(events)
    eq, cash = np.empty(n+1),np.empty(n+1)
    eq[0],cash[0] = capital,capital
    positions,exposure,fees,turnover,receivable,stale_value = [np.zeros(n+1) for _ in range(6)]
    blocked_b,blocked_s,buy_amount,sell_amount,buy_bounce,sell_bounce = [np.zeros(n) for _ in range(6)]
    previous_i = indices[0]-1
    if actions is not None:
        actions.start_at(previous_i)

    def valuation(prices):
        px = account.price_at(prices).copy()
        if stale_policy == "write_down" and max_stale_days:
            px[(account.stale > max_stale_days) & (account.shares > 0)] = 0.
        return px

    for k,event in enumerate(events):
        i = event.index
        if i > previous_i + 1:
            for row in range(previous_i+1,i):
                idle = row_event(clock,ds,row)
                book.begin(idle)
                changed = np.flatnonzero(np.abs(ds.cumadj_ff[row]/ds.cumadj_ff[previous_i]-1)>1e-4)
                book.cancel_instruments(changed,"corporate_action")
                if actions is not None:
                    affected=actions.advance(account,ds,previous_i,row)
                    book.cancel_instruments(affected,"corporate_action")
                else:
                    account.apply_corporate_actions(ds.cumadj_ff[previous_i],ds.cumadj_ff[row],ds.open[row],policy=div_policy)
                rules.start_bar(account,idle,broker.asset_rules)
                prices=np.where(m.suspended[row],np.nan,ds.close[row])
                account.mark(prices,advance=idle.new_session)
                if actions is not None:
                    actions.record_close(account,row)
                account._valuation=valuation(prices).copy()
                book.end(clock.is_session_end(ds,row))
                previous_i=row
            event=row_event(clock,ds,i)
        book.begin(event)
        ctx = Context(ds,m,params).sub(0,i)
        ctx.periods_per_year = clock.periods_per_year
        ctx.portfolio = dict(account.snapshot(),shares=account.shares.copy())
        ctx.now, ctx.execution_time = event.previous_timestamp,event.timestamp
        ctx.order,ctx.cancel,ctx.get_order,ctx.orders = api.order,api.cancel,api.get_order,api.orders
        decision = strategy(ctx)
        if decision is not None and decision is not HOLD:
            commands = decision if isinstance(decision,(list,tuple)) else [decision]
            for command in commands:
                if isinstance(command,OrderRequest):
                    book.submit(command)
                elif isinstance(command,Cancel):
                    book.cancel(command.order_id)
                else:
                    raise TypeError("订单策略返回 OrderRequest/Cancel/列表/None；目标组合使用 run/run_events")
        with np.errstate(invalid="ignore",divide="ignore"):
            changed = np.flatnonzero(np.abs(ds.cumadj_ff[i]/ds.cumadj_ff[previous_i]-1) > 1e-4)
        if changed.size:
            book.cancel_instruments(changed,"corporate_action")
        if actions is not None:
            affected = actions.advance(account,ds,previous_i,i)
            book.cancel_instruments(affected,"corporate_action")
            book.cancel_instruments(np.flatnonzero(actions.closed),"settled_instrument")
        else:
            account.apply_corporate_actions(ds.cumadj_ff[previous_i],ds.cumadj_ff[i],ds.open[i],policy=div_policy)
        previous_i = i
        rules.start_bar(account,event,broker.asset_rules)
        for phase in execution.phases:
            book.match(phase)
        for oid in book.blocked:
            order = book.records[oid]
            j = book.codes[order.request.code]
            amount = ds.open[i,j]*order.remaining
            bounce = ds.close[i,j]/ds.open[i,j]-1.
            if order.request.side == "buy":
                blocked_b[k] += 1
                buy_amount[k] += amount
                buy_bounce[k] += amount*bounce
            else:
                blocked_s[k] += 1
                sell_amount[k] += amount
                sell_bounce[k] += amount*bounce
        if buy_amount[k]: buy_bounce[k] /= buy_amount[k]
        if sell_amount[k]: sell_bounce[k] /= sell_amount[k]
        last_session_bar = clock.is_session_end(ds,i)
        book.end(last_session_bar)
        prices = np.where(m.suspended[i],np.nan,ds.close[i])
        account.mark(prices,advance=event.new_session)
        if actions is not None:
            actions.record_close(account,i)
        close = valuation(prices)
        account._valuation = close.copy()
        eq[k+1] = rules.equity(account,close)
        cash[k+1],positions[k+1] = account.cash,np.count_nonzero(account.shares)
        exposure[k+1] = account.position_value(close)/eq[k+1] if eq[k+1] else 0.
        fees[k+1],turnover[k+1],receivable[k+1] = account.fees,account.turnover,account.receivables.sum()
        if max_stale_days:
            mask = (account.shares > 0) & (account.stale > max_stale_days)
            stale_value[k+1] = np.nansum((account.shares*account.last_px)[mask])
        rules.check(account)
    bench_ret,bench_nav = benchmark_at_steps(ds,indices,m)
    trades = [(f.index,book.codes[f.code],1 if f.side=="buy" else -1,f.quantity,f.price,f.fee) for f in book.fills]
    return Result(dates=ds.dates[np.r_[indices[0]-1,indices]],equity=eq,cash=cash,
        n_positions=positions,gross=exposure,fees_cum=fees,turnover_cum=turnover,
        ret=equity_returns(eq),bench_ret=bench_ret,bench_nav=bench_nav*capital,
        blocked_buy_n=blocked_b.astype(int),blocked_sell_n=blocked_s.astype(int),
        blocked_buy_amt=buy_amount,blocked_sell_amt=sell_amount,
        blocked_buy_bounce=buy_bounce,blocked_sell_bounce=sell_bounce,
        trades=trades,account=account,market=m,ds=ds,limit_rule=limit_rule,capital=capital,
        div_policy=div_policy,max_adv_pct=max_adv_pct,stale_policy=stale_policy,
        max_stale_days=max_stale_days,stale_value=stale_value,receivables=receivable,
        final_prices=close,capacity_mode=capacity_mode,adv_window=adv_window,mode="orders",
        periods_per_year=clock.periods_per_year,step_indices=indices,
        clock_name=type(clock).__name__,asset_rule_name=type(assets).__name__,account_rule_name=type(rules).__name__,
        cost_description=repr(broker.costs),name=name or getattr(strategy,"__name__","订单策略"),
        execution_provenance=provenance,
        execution_name=type(execution).__name__,
        note="直接订单入口",window=(int(indices[0]),int(indices[-1])),
        orders=book.orders(),order_events=tuple(book.events),order_fills=tuple(book.fills),
        settlements=tuple(actions.settlement_log) if actions is not None else (),
        assumptions=[execution.description,"订单逐笔FIFO，无现金/股份预留；不足部分继续等待或到期",
            "同证券每行情行的买卖与open/close阶段，共用累计成交额容量预算",
            "报价源时间：" + m.stats.get("quote_time_source", "未提供"),
            f"{type(clock).__name__}：年化 {clock.periods_per_year:g} 个估值区间",
            f"资产规则 {type(assets).__name__}；账户规则 {type(rules).__name__}",
            "公司行动：显式事件" if actions else f"公司行动：{div_policy}（再投资为近似）",
            "公司行动发生时取消受影响的未成交委托，不猜价格/数量调整",
            f"容量 {capacity_mode}，{max_adv_pct:.1%}；当期成交额为事后上界，不保证开盘容量",
            f"长期无成交 {stale_policy}，阈值 {max_stale_days} 会话，不自动变现"],
        stats={"n_blocked_buy":int(blocked_b.sum()),"n_blocked_sell":int(blocked_s.sum()),
               "blocked_buy_amt":float(buy_amount.sum()),"blocked_buy_pnl":float(np.sum(buy_amount*buy_bounce))})
