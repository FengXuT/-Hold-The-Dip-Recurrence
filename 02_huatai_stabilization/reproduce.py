"""执行固定规则、参数稳健性、跨指数复现及两个因子扩展。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'outputs'
START, END = pd.Timestamp('2022-07-22'), pd.Timestamp('2026-08-31')
CAPITAL, BUDGET, FEE, ANNUAL_DAYS = 100_000_000, 12_000_000, .00003, 250
SETTINGS = {'IM': ('000852', -.0492, 200), 'IC': ('000905', -.0432, 200),
            'IF': ('000300', -.0345, 300), 'IH': ('000016', -.0349, 300)}


def load_data():
    calendar = pd.DatetimeIndex([row[0] for row in json.loads((RAW / 'calendar.json').read_text(encoding='utf-8'))['data']['data']]).sort_values()
    indices = {}
    for prefix, (code, _, _) in SETTINGS.items():
        frame = pd.DataFrame(json.loads((RAW / f'{code}.json').read_text(encoding='utf-8'))['data'])
        frame.index = pd.to_datetime(frame.tradeDate)
        frame = frame[['open', 'high', 'low', 'close']].astype(float).reindex(calendar)
        assert frame.notna().all().all() and not frame.index.has_duplicates, code
        indices[prefix] = frame
    futures, expiries = {}, {}
    for file in sorted((RAW / 'futures').glob('*.json')):
        frame = pd.DataFrame(json.loads(file.read_text(encoding='utf-8')))
        frame.index = pd.to_datetime(frame.d)
        frame = frame[['o', 'h', 'l', 'c']].astype(float).sort_index()
        assert not frame.index.has_duplicates and (frame > 0).all().all(), file.name
        futures[file.stem] = frame
        first = pd.Timestamp(2000 + int(file.stem[2:4]), int(file.stem[4:]), 1)
        expiry = first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)
        # 第三个周五遇休市，顺延到下一交易日；2026年12月到期日在样本日历之后。
        if expiry <= calendar.max():
            expiry = calendar[calendar.searchsorted(expiry)]
        expiries[file.stem] = expiry
    return indices, futures, expiries


def metrics(nav, trades):
    returns = nav.pct_change().fillna(0)
    annual = nav.iloc[-1] ** (ANNUAL_DAYS / len(nav)) - 1
    volatility = returns.std() * np.sqrt(ANNUAL_DAYS)
    drawdown = -(nav / nav.cummax() - 1).min()
    return dict(trades=len(trades), final_nav=float(nav.iloc[-1]), annual_return=float(annual),
                volatility=float(volatility), max_drawdown=float(drawdown),
                sharpe=float((annual - .02) / volatility), calmar=float(annual / drawdown))


def simulate(indices, futures, expiries, prefix='IM', wait=7, hold=20, extension=None):
    q = indices[prefix]
    calendar = q.index
    _, threshold, multiplier = SETTINGS[prefix]
    shock = q.close.pct_change(5)
    daily_return = q.close.pct_change()
    vol5 = daily_return.rolling(5).std()
    vol20 = daily_return.rolling(20).std()
    trades, free = [], -1
    for confirm, date in enumerate(calendar):
        trigger, entry = confirm - wait, confirm + 1
        exit_day = entry + hold
        if confirm < wait or not START <= date <= END or confirm < free:
            continue
        if not shock.iloc[trigger] < threshold or q.close.iloc[confirm] < q.close.iloc[trigger]:
            continue
        if exit_day >= len(calendar) or calendar[exit_day] > END:
            continue
        eligible = [name for name, frame in futures.items() if name.startswith(prefix)
                    and frame.index.min() <= calendar[entry] and expiries[name] > calendar[entry + 40]]
        name = min(eligible, key=lambda item: expiries[item])
        frame = futures[name]
        buy, sell = frame.loc[calendar[entry], 'o'], frame.loc[calendar[exit_day], 'o']
        margin = .12
        if calendar[entry] < pd.Timestamp('2023-05-05'):
            margin = {'IM': .15, 'IC': .14, 'IF': .12, 'IH': .12}[prefix]
        repair = q.close.iloc[confirm] / q.close.iloc[trigger] - 1
        contraction = vol5.iloc[confirm] / vol20.iloc[trigger]
        efficiency = repair / daily_return.iloc[trigger + 1:confirm + 1].abs().sum() if wait else 0.
        scale = 1.
        if extension == 'volatility':
            scale = float(np.clip(1 / contraction, .5, 1))
        elif extension == 'efficiency':
            scale = float(np.clip(efficiency, 0, 1))
        quantity = int(BUDGET * scale / (buy * multiplier * margin))
        if quantity == 0:
            continue
        fee = quantity * multiplier * (buy + sell) * FEE
        trades.append(dict(trigger=calendar[trigger], confirm=date, entry=calendar[entry], exit=calendar[exit_day],
                           contract=name, quantity=quantity, margin_rate=margin, entry_price=buy, exit_price=sell,
                           shock=shock.iloc[trigger], repair=repair, volatility_ratio=contraction,
                           repair_efficiency=efficiency, budget_scale=scale,
                           profit=quantity * multiplier * (sell - buy) - fee))
        # 持仓期间产生的信号可在平仓后完成确认；平仓当日收盘可重新评估。
        free = exit_day
    pnl = pd.Series(0., index=calendar[(calendar >= START) & (calendar <= END)])
    for trade in trades:
        dates = pnl.loc[trade['entry']:trade['exit']].index
        prices = futures[trade['contract']].loc[dates, 'c'].copy()
        prices.iloc[-1] = trade['exit_price']
        changes = prices.diff()
        changes.iloc[0] = prices.iloc[0] - trade['entry_price']
        profit = changes * trade['quantity'] * multiplier
        profit.iloc[0] -= trade['quantity'] * multiplier * trade['entry_price'] * FEE
        profit.iloc[-1] -= trade['quantity'] * multiplier * trade['exit_price'] * FEE
        pnl.loc[dates] += profit
    nav = 1 + pnl.cumsum() / CAPITAL
    table = pd.DataFrame(trades)
    assert abs((nav.iloc[-1] - 1) * CAPITAL - table.profit.sum()) < .01
    return nav, table, metrics(nav, table)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    indices, futures, expiries = load_data()
    summaries, curves = {}, {}
    for label, options in [('direct', dict(wait=0)), ('confirmed', {}),
                           ('volatility', dict(extension='volatility')), ('efficiency', dict(extension='efficiency'))]:
        nav, trades, summary = simulate(indices, futures, expiries, **options)
        curves[label], summaries[label] = nav, summary
        trades.to_csv(OUT / f'{label}_trades.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(curves).to_csv(OUT / 'nav.csv', index_label='date')
    assert summaries['direct']['trades'] == 17 and summaries['confirmed']['trades'] == 15
    direct = pd.read_csv(OUT / 'direct_trades.csv')
    confirmed = pd.read_csv(OUT / 'confirmed_trades.csv')
    a = direct.loc[direct.entry == '2024-01-09'].iloc[0]
    b = confirmed.loc[confirmed.entry == '2026-07-30'].iloc[0]
    assert a.contract == 'IM2403' and a.exit == '2024-02-06' and round(a.profit / CAPITAL * 100, 2) == -26.11
    assert b.contract == 'IM2612' and b.exit == '2026-08-27' and round(b.profit / CAPITAL * 100, 2) == 6.38
    for prefix in SETTINGS:
        nav, trades, summary = simulate(indices, futures, expiries, prefix=prefix)
        summaries[prefix] = summary
        trades.to_csv(OUT / f'{prefix}_trades.csv', index=False, encoding='utf-8-sig')
    grid = []
    for wait in [5, 6, 7, 8, 9]:
        for hold in [15, 18, 20, 22, 25]:
            _, _, summary = simulate(indices, futures, expiries, wait=wait, hold=hold)
            grid.append(dict(wait=wait, hold=hold, **summary))
    pd.DataFrame(grid).to_csv(OUT / 'parameter_grid.csv', index=False)
    (OUT / 'metrics.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(summaries).T.round(6).to_string())
    print('Anchor trades and cash reconciliation passed.')
