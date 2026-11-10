"""回测下行压力衰减、新低停止及两者组合，固定主参数后做对照。"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import reproduce as base

OUT = ROOT / 'outputs'
NAMES = {'baseline': '原企稳策略', 'pressure': '下行压力衰减', 'floor': '连续3日守底', 'combined': '双因子组合'}


def candidates(q, prefix='IM'):
    calendar = q.index
    returns = q.close.pct_change()
    downside_square = returns.clip(upper=0).pow(2)
    recent = downside_square.rolling(3, min_periods=3).mean().pow(.5)
    previous = downside_square.rolling(20, min_periods=20).mean().pow(.5).shift(1)
    shock = q.close.pct_change(5)
    records = []
    for c, date in enumerate(calendar):
        t, e, x = c - 7, c + 1, c + 21
        if c < 7 or not base.START <= date <= base.END or x >= len(q) or calendar[x] > base.END:
            continue
        if not shock.iloc[t] < base.SETTINGS[prefix][1] or q.close.iloc[c] < q.close.iloc[t]:
            continue
        assert previous.iloc[t] > 0
        lows = q.low.iloc[t:c + 1].to_numpy()
        # 并列最低价取最后一次，避免将反复触底计为连续守底。
        last_low = c - int(np.argmin(lows[::-1]))
        records.append(dict(trigger=calendar[t], confirm=date, entry=calendar[e], exit=calendar[x],
                            trigger_i=t, confirm_i=c, entry_i=e, exit_i=x,
                            downside_ratio=float(recent.iloc[c] / previous.iloc[t]),
                            days_since_low=c - last_low,
                            repair=float(q.close.iloc[c] / q.close.iloc[t] - 1)))
    return pd.DataFrame(records)


def backtest(events, futures, expiries, prefix='IM', method='baseline', ratio_limit=1., floor_days=3,
             fee_rate=base.FEE, slippage=0.):
    multiplier = base.SETTINGS[prefix][2]
    free, records = -1, []
    for z in events.itertuples():
        if z.confirm_i < free:
            continue
        pressure_pass, floor_pass = z.downside_ratio <= ratio_limit, z.days_since_low >= floor_days
        if method in ['pressure', 'combined'] and not pressure_pass:
            continue
        if method in ['floor', 'combined'] and not floor_pass:
            continue
        calendar = indices[prefix].index
        eligible = [name for name, quotes in futures.items() if name.startswith(prefix)
                    and quotes.index.min() <= z.entry and expiries[name] > calendar[z.entry_i + 40]]
        name = min(eligible, key=lambda item: expiries[item])
        buy = float(futures[name].loc[z.entry, 'o']) + slippage
        sell = float(futures[name].loc[z.exit, 'o']) - slippage
        margin = .12
        if z.entry < pd.Timestamp('2023-05-05'):
            margin = {'IM': .15, 'IC': .14, 'IF': .12, 'IH': .12}[prefix]
        quantity = int(base.BUDGET / (buy * multiplier * margin))
        commission = quantity * multiplier * (buy + sell) * fee_rate
        records.append(dict(trigger=z.trigger, confirm=z.confirm, entry=z.entry, exit=z.exit,
                            contract=name, quantity=quantity, margin_rate=margin, multiplier=multiplier,
                            entry_price=buy, exit_price=sell, fee_rate=fee_rate,
                            downside_ratio=z.downside_ratio, days_since_low=z.days_since_low,
                            repair=z.repair, profit=quantity * multiplier * (sell-buy) - commission))
        free = z.exit_i
    return pd.DataFrame(records)


def daily_results(trades, futures, calendar):
    dates = calendar[(calendar >= base.START) & (calendar <= base.END)]
    pnl, exposure = pd.Series(0., index=dates), pd.Series(0., index=dates)
    for z in trades.itertuples():
        held = pnl.loc[z.entry:z.exit].index
        prices = futures[z.contract].loc[held, 'c'].copy()
        prices.iloc[-1] = z.exit_price
        changes = prices.diff()
        changes.iloc[0] = prices.iloc[0] - z.entry_price
        amount = changes * z.quantity * z.multiplier
        amount.iloc[0] -= z.quantity * z.multiplier * z.entry_price * z.fee_rate
        amount.iloc[-1] -= z.quantity * z.multiplier * z.exit_price * z.fee_rate
        pnl.loc[held] += amount
        exposure.loc[held[:-1]] += z.quantity * z.multiplier * z.entry_price / base.CAPITAL
    nav = 1 + pnl.cumsum() / base.CAPITAL
    assert abs(pnl.sum() - trades.profit.sum()) < .01
    summary = base.metrics(nav, trades)
    summary.update(mean_exposure=float(exposure.mean()), invested_days=int((exposure > 0).sum()),
                   win_rate=float(trades.profit.gt(0).mean()), total_profit=float(pnl.sum()),
                   top3_profit_share=float(trades.profit.nlargest(3).sum() / pnl.sum()))
    return nav, exposure, summary


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    indices, futures, expiries = base.load_data()
    event_sets = {prefix: candidates(q, prefix) for prefix, q in indices.items()}
    events = event_sets['IM']
    events.to_csv(OUT / 'all_candidates.csv', index=False, encoding='utf-8-sig')
    navs, exposures, summaries, ledgers = {}, {}, {}, {}
    for method in NAMES:
        trades = backtest(events, futures, expiries, method=method)
        nav, exposure, summary = daily_results(trades, futures, indices['IM'].index)
        navs[method], exposures[method], summaries[method], ledgers[method] = nav, exposure, summary, trades
        trades.to_csv(OUT / f'{method}_trades.csv', index=False, encoding='utf-8-sig')
    parent_nav, parent_trades, _ = base.simulate(indices, futures, expiries)
    pd.testing.assert_series_equal(navs['baseline'], parent_nav)
    pd.testing.assert_frame_equal(ledgers['baseline'][['trigger','confirm','entry','exit','contract','quantity','profit']],
                                  parent_trades[['trigger','confirm','entry','exit','contract','quantity','profit']])
    pd.DataFrame(navs).to_csv(OUT / 'nav.csv', index_label='date')
    pd.DataFrame(exposures).to_csv(OUT / 'exposure.csv', index_label='date')
    (OUT / 'metrics.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')

    cohort = ledgers['baseline'][['trigger','confirm','entry','exit','profit','downside_ratio','days_since_low']].copy()
    cohort['pressure_pass'] = cohort.downside_ratio <= 1
    cohort['floor_pass'] = cohort.days_since_low >= 3
    cohort['combined_pass'] = cohort.pressure_pass & cohort.floor_pass
    cohort.to_csv(OUT / 'baseline_event_diagnostics.csv', index=False, encoding='utf-8-sig')
    frozen = ledgers['baseline'].loc[cohort.combined_pass].reset_index(drop=True)
    _, _, frozen_summary = daily_results(frozen, futures, indices['IM'].index)
    (OUT / 'fixed_event_filter.json').write_text(json.dumps(frozen_summary, indent=2), encoding='utf-8')

    january_old = ledgers['baseline'].loc[ledgers['baseline'].entry == pd.Timestamp('2025-01-15')].iloc[0]
    january_new = ledgers['combined'].loc[ledgers['combined'].entry == pd.Timestamp('2025-01-17')].iloc[0]
    old_contract = futures[january_old.contract]
    buy = float(old_contract.loc[january_new.entry, 'o'])
    sell = float(old_contract.loc[january_new.exit, 'o'])
    quantity = int(base.BUDGET / (buy * 200 * .12))
    same_contract_profit = quantity * 200 * (sell-buy) - quantity * 200 * (buy+sell) * base.FEE
    decomposition = dict(same_contract_profit=same_contract_profit,
                         timing_and_size_gain=same_contract_profit-float(january_old.profit),
                         contract_gain=float(january_new.profit)-same_contract_profit)
    (OUT / 'january_decomposition.json').write_text(json.dumps(decomposition, indent=2), encoding='utf-8')

    matched = []
    for method in ['pressure','floor','combined']:
        scale = summaries[method]['mean_exposure'] / summaries['baseline']['mean_exposure']
        resized = 1 + (navs['baseline']-1) * scale
        matched.append(dict(method=method, baseline_scale=scale, **base.metrics(resized, ledgers['baseline'])))
    pd.DataFrame(matched).to_csv(OUT / 'matched_exposure.csv', index=False)

    annual = []
    for method, nav in navs.items():
        previous_nav = 1.
        for year, values in nav.groupby(nav.index.year):
            annual.append(dict(method=method, year=int(year), nav_return=float(values.iloc[-1]/previous_nav-1),
                               profit=float((values.iloc[-1]-previous_nav)*base.CAPITAL)))
            previous_nav = values.iloc[-1]
    pd.DataFrame(annual).to_csv(OUT / 'annual.csv', index=False)

    cross = []
    for prefix in base.SETTINGS:
        for method in NAMES:
            trades = backtest(event_sets[prefix], futures, expiries, prefix=prefix, method=method)
            _, _, summary = daily_results(trades, futures, indices[prefix].index)
            cross.append(dict(prefix=prefix, method=method, **summary))
    pd.DataFrame(cross).to_csv(OUT / 'cross_index.csv', index=False)

    sensitivity = []
    for ratio_limit in [.75,1.,1.25]:
        for floor_days in [2,3,4]:
            trades = backtest(events, futures, expiries, method='combined', ratio_limit=ratio_limit, floor_days=floor_days)
            _, _, summary = daily_results(trades, futures, indices['IM'].index)
            sensitivity.append(dict(ratio_limit=ratio_limit, floor_days=floor_days, **summary))
    pd.DataFrame(sensitivity).to_csv(OUT / 'sensitivity.csv', index=False)

    stress = []
    for method in NAMES:
        trades = backtest(events, futures, expiries, method=method, fee_rate=base.FEE*2, slippage=1.)
        _, _, summary = daily_results(trades, futures, indices['IM'].index)
        stress.append(dict(method=method, **summary))
    pd.DataFrame(stress).to_csv(OUT / 'cost_stress.csv', index=False)

    # 移除截止日之后的价格，验证历史因子与已完成交易的决定保持一致。
    saved_end = base.END
    for cutoff in [pd.Timestamp('2024-06-28'), pd.Timestamp('2025-06-30')]:
        base.END = cutoff
        q = indices['IM'].loc[:cutoff].copy()
        historical_events = candidates(q)
        expected_events = events.loc[events.exit <= cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(historical_events, expected_events)
        historical_futures = {name: frame.loc[:cutoff] for name, frame in futures.items() if frame.index.min() <= cutoff}
        for method in NAMES:
            historical = backtest(historical_events, historical_futures, expiries, method=method)
            expected = ledgers[method].loc[ledgers[method].exit <= cutoff].reset_index(drop=True)
            pd.testing.assert_frame_equal(historical, expected)
    base.END = saved_end
    (OUT / 'validation.json').write_text(json.dumps(dict(baseline_matches=True, historical_cutoffs=2,
        methods_checked=4, cash_reconciliation=True, main_ratio_limit=1., main_floor_days=3),indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries).T.round(5).to_string())
    print('Matched exposure:\n',pd.DataFrame(matched).round(5).to_string(index=False))
    print('Baseline replay, cash reconciliation and historical truncation checks passed.')
