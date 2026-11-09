from pathlib import Path
import argparse
import io
import json
import platform
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent
DEPTHS = [0.05, 0.10, 0.15, 0.20]
LOOKBACKS = [5, 10, 15, 21, 63, 126, 252]
HOLDS = [21, 42, 63, 126, 252, 756, 1260]
START, END, FORWARD_END = '1965-01-01', '2025-09-30', '2026-09-30'


def load_data():
    raw = ROOT / 'data/raw'
    chart = json.loads((raw / 'gspc_daily.json').read_text())['chart']['result'][0]
    dates = pd.to_datetime(chart['timestamp'], unit='s', utc=True).tz_convert('America/New_York').normalize().tz_localize(None)
    prices = pd.DataFrame({'close': chart['indicators']['quote'][0]['close']}, index=dates)
    prices.index.name = 'date'
    with zipfile.ZipFile(raw / 'ff_daily.zip') as archive:
        text = archive.read(archive.namelist()[0]).decode('utf-8-sig')
    lines = [line for line in text.splitlines() if len(line.split(',')[0]) == 8 and line.split(',')[0].isdigit()]
    factors = pd.read_csv(io.StringIO('\n'.join(lines)), header=None, names=['date', 'mkt_rf', 'smb', 'hml', 'rf'])
    factors['date'] = pd.to_datetime(factors['date'].astype(str), format='%Y%m%d')
    rf = factors.set_index('date')['rf'] / 100.0
    if prices.index.has_duplicates or rf.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError('Dates must be unique and price dates increasing')
    if prices['close'].isna().any() or (prices['close'] <= 0).any():
        raise ValueError('Prices must be positive and complete')
    data = prices.loc['1963-01-01':FORWARD_END].join(rf, how='left')
    if data['rf'].isna().any():
        raise ValueError('Some price dates have no French RF; inspect dates before proceeding')
    if data.index[0] > pd.Timestamp('1963-01-10') or data.index[-1] < pd.Timestamp(FORWARD_END):
        raise ValueError('Input data does not cover the fixed research window')
    data['market_return'] = data['close'].pct_change(fill_method=None)
    data['market_excess'] = data['market_return'] - data['rf']
    data.iloc[1:].to_csv(ROOT / 'data/processed/market_daily.csv', float_format='%.12g')
    return data


def positions(signal, hold, mode='ignore', lag=1):
    weight = np.zeros(len(signal), dtype=np.int8)
    exit_close = -1
    for t in np.flatnonzero(signal):
        if mode == 'ignore' and t < exit_close:
            continue
        first_return = t + lag
        exit_close = first_return + hold - 1
        # A signal at the exit close may schedule the next holding cycle.
        weight[first_return:exit_close + 1] = 1
    return weight


def monthly_returns(returns, dates):
    return ((1 + pd.DataFrame(returns, index=dates)).resample('ME').prod() - 1).to_numpy()


def performance(returns, rf, dates):
    monthly = monthly_returns(np.column_stack([returns, rf]), dates)
    excess = monthly[:, 0] - monthly[:, 1]
    nav = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(np.r_[1.0, nav])[1:]
    volatility = np.std(excess, ddof=1) * np.sqrt(12)
    return {'cagr': float(nav[-1] ** (252 / len(nav)) - 1),
            'volatility': float(np.std(monthly[:, 0], ddof=1) * np.sqrt(12)),
            'sharpe': float(np.mean(excess) * 12 / volatility) if volatility > 1e-12 else None,
            'max_drawdown': float(np.min(nav / peak - 1)), 'terminal_nav': float(nav[-1])}


def regress(returns, rf, market, dates):
    monthly = monthly_returns(np.column_stack([returns, rf, market]), dates)
    excess = monthly[:, 0] - monthly[:, 1]
    market_excess = monthly[:, 2] - monthly[:, 1]
    fit = sm.OLS(excess, sm.add_constant(market_excess)).fit()
    hac = fit.get_robustcov_results(cov_type='HAC', maxlags=3, use_correction=True, use_t=False)
    return {'alpha_ann': float(fit.params[0] * 12), 'beta': float(fit.params[1]),
            't_iid': float(fit.tvalues[0]), 't_hac': float(hac.tvalues[0]),
            'p_hac': float(hac.pvalues[0])}


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check', action='store_true', help='Run small timing and accounting checks before research')
    args = parser.parse_args()
    out = ROOT / 'outputs'
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    (ROOT / 'data/processed').mkdir(parents=True, exist_ok=True)
    checks = []
    if args.self_check:
        np.testing.assert_array_equal(positions([True, False, True, False, False, False, False], 2), [0, 1, 1, 1, 1, 0, 0])
        np.testing.assert_array_equal(positions([True, True, False, False, False, False], 2), [0, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(positions([True, True, False, False, False, False], 2, 'extend'), [0, 1, 1, 1, 0, 0])
        np.testing.assert_array_equal(positions([False] * 8, 3), np.zeros(8))
        np.testing.assert_array_equal(positions([True, True, False, False, False, False], 2, lag=2), [0, 0, 1, 1, 0, 0])
        np.testing.assert_array_equal(positions([True, False, True, False, False, False, False], 2, lag=2), [0, 0, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(positions([True, False, False, True, False, False, False, False], 2, lag=2), [0, 0, 1, 1, 0, 1, 1, 0])
        np.testing.assert_allclose(monthly_returns([0.10, -0.05, 0.02], pd.to_datetime(['2000-01-28', '2000-01-31', '2000-02-01']))[:, 0], [0.045, 0.02])
        checks += ['close-to-next-close timing and exact holding length', 'delayed entry ignores signals until actual exit close', 'monthly compounding across month boundaries', 'ignore vs extend repeated signals', 'no-signal cash position']
    data = load_data()
    # The experiment begins flat; pre-1965 prices are used only for lookback calculation.
    raw_signals = {(l, d): ((data['close'] / data['close'].shift(l) - 1 <= -d) & (data.index >= START)).to_numpy()
                   for l in LOOKBACKS for d in DEPTHS}
    sample = (data.index >= START) & (data.index <= END)
    dates = data.index[sample]
    rf = data.loc[sample, 'rf'].to_numpy()
    market = data.loc[sample, 'market_return'].to_numpy()
    mx = market - rf
    bh = performance(market, rf, dates)
    records, cost_records, period_records = [], [], []
    all_positions, ids = [], []
    selected = {'L21_D10_H63', 'L63_D10_H252', 'L252_D20_H1260'}
    daily = pd.DataFrame(index=data.index[sample])
    daily['buy_hold'] = market
    daily['cash'] = rf
    mean_weight = np.zeros(len(data))
    for lookback in LOOKBACKS:
        for depth in DEPTHS:
            for hold in HOLDS:
                sid = f'L{lookback}_D{round(depth * 100)}_H{hold}'
                w_full = positions(raw_signals[lookback, depth], hold)
                w = w_full[sample]
                turnover = np.abs(np.diff(w_full.astype(float), prepend=0))[sample]
                returns = rf + w * mx
                record = {'strategy': sid, 'lookback_days': lookback, 'depth': depth, 'hold_days': hold,
                          'exposure': float(w.mean()), 'entries': int(np.sum(np.diff(w_full.astype(float), prepend=0)[sample] > 0)),
                          'annual_turnover': float(turnover.sum() * 252 / len(w)),
                          **performance(returns, rf, dates), **regress(returns, rf, market, dates)}
                record['sharpe_gap'] = record['sharpe'] - bh['sharpe']
                records.append(record)
                for bps in [0, 5, 10, 25]:
                    net = returns - turnover * bps / 10000
                    cost_records.append({'strategy': sid, 'one_way_bps': bps, **performance(net, rf, dates)})
                for label, first, last in [('1965-1989', '1965-01-01', '1989-12-31'),
                                           ('1990-2009', '1990-01-01', '2009-12-31'),
                                           ('2010-2025Q3', '2010-01-01', END),
                                           ('after_publication', '2025-12-02', FORWARD_END)]:
                    mask = (data.index >= first) & (data.index <= last)
                    sub_rf = data.loc[mask, 'rf'].to_numpy()
                    sub_m = data.loc[mask, 'market_return'].to_numpy()
                    sub_r = sub_rf + w_full[mask] * (sub_m - sub_rf)
                    p = performance(sub_r, sub_rf, data.index[mask])
                    period_records.append({'strategy': sid, 'period': label, 'days': int(mask.sum()),
                                           'exposure': float(w_full[mask].mean()), **p,
                                           'benchmark_sharpe': performance(sub_m, sub_rf, data.index[mask])['sharpe']})
                if sid in selected:
                    daily[sid] = returns
                mean_weight += w_full / 196
                all_positions.append(w)
                ids.append(sid)
    results = pd.DataFrame(records)
    results['p_holm'] = multipletests(results['p_hac'].fillna(1).to_numpy(), method='holm')[1]
    results.to_csv(out / 'grid_results.csv', index=False, float_format='%.12g')
    costs = pd.DataFrame(cost_records)
    costs.to_csv(out / 'cost_sensitivity.csv', index=False, float_format='%.12g')
    periods = pd.DataFrame(period_records)
    periods.to_csv(out / 'period_results.csv', index=False, float_format='%.12g')
    pd.DataFrame(np.array(all_positions).T, index=daily.index, columns=ids).to_csv(out / 'positions.csv.gz', compression='gzip')
    daily['equal_weight_196'] = rf + mean_weight[sample] * mx
    # This is a transparent project baseline, separate from AQR's SG Trend analysis.
    trend_signal = (data['close'] > data['close'].shift(252)).astype(int).shift(1, fill_value=0).to_numpy()
    daily['trend_252'] = rf + trend_signal[sample] * mx
    daily.to_csv(out / 'selected_daily_returns.csv', float_format='%.12g')
    nav = (1 + daily).cumprod()
    nav.to_csv(out / 'selected_nav.csv', float_format='%.12g')
    baselines = {name: performance(daily[name].to_numpy(), rf, dates) for name in daily.columns}
    sensitivity = []
    for mode, lag in [('ignore', 1), ('ignore', 2), ('extend', 1)]:
        gaps = []
        for l in LOOKBACKS:
            for d in DEPTHS:
                for h in HOLDS:
                    w = positions(raw_signals[l, d], h, mode, lag)[sample]
                    r = rf + w * mx
                    sr = performance(r, rf, dates)['sharpe']
                    gaps.append(sr - bh['sharpe'])
                    sensitivity.append({'mode': mode, 'lag': lag, 'lookback_days': l, 'depth': d, 'hold_days': h, 'sharpe': sr,
                                        'sharpe_gap': sr - bh['sharpe']})
    pd.DataFrame(sensitivity).to_csv(out / 'implementation_sensitivity.csv', index=False, float_format='%.12g')
    if args.self_check:
        # Reconstruct the selected strategy from its saved positions and check cash/cost accounting.
        selected_id = 'L21_D10_H63'
        w = all_positions[ids.index(selected_id)]
        np.testing.assert_allclose(daily[selected_id], rf + w * mx)
        costs_pivot = costs.pivot(index='strategy', columns='one_way_bps', values='cagr')
        assert (costs_pivot[25] <= costs_pivot[0] + 1e-12).all()
        # Appending future data must leave all historical positions unchanged.
        sig = raw_signals[21, 0.10]
        cut = np.where(sample)[0][-1] + 1
        np.testing.assert_array_equal(positions(sig[:cut], 63), positions(sig, 63)[:cut])
        checks += ['position/return accounting', 'cost monotonicity across all 196 strategies', 'prefix causality']
    summary = {'sample_start': str(daily.index[0].date()), 'sample_end': str(daily.index[-1].date()), 'days': len(daily),
               'strategy_count': len(results), 'benchmark': bh, 'statistics_frequency': 'monthly', 'signal_lag': 1, 'hac_month_lags': 3,
               'mean_sharpe': float(results.sharpe.mean()), 'mean_sharpe_gap': float(results.sharpe_gap.mean()),
               'underperform_share': float((results.sharpe_gap < 0).mean()),
               'mean_alpha_ann': float(results.alpha_ann.mean()),
               'iid_significant_count': int((results.t_iid.abs() > 1.96).sum()),
               'hac_significant_count': int((results.p_hac < 0.05).sum()),
               'holm_significant_count': int((results.p_holm < 0.05).sum()),
               'baselines': baselines, 'self_checks': checks,
               'data_rows': len(data), 'max_abs_daily_market_return': float(data.market_return.abs().max()),
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                            'matplotlib': matplotlib.__version__, 'statsmodels': statsmodels.__version__},
               'run_utc': datetime.now(timezone.utc).isoformat()}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    plt.rcParams.update({'figure.dpi': 160, 'savefig.dpi': 180, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(11, 5))
    matrix = results.groupby(['lookback_days', 'hold_days']).sharpe_gap.mean().unstack()
    lim = float(np.abs(matrix.to_numpy()).max())
    im = ax.imshow(matrix, cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto')
    ax.set_xticks(range(7), ['1m', '2m', '3m', '6m', '1y', '3y', '5y'])
    ax.set_yticks(range(7), ['1w', '2w', '3w', '1m', '3m', '6m', '1y'])
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f'{matrix.iloc[i,j]:+.2f}', ha='center', va='center', fontsize=9)
    ax.set(xlabel='Holding period', ylabel='Lookback period', title='BTD minus buy-and-hold Sharpe | mean across 4 dip depths')
    fig.colorbar(im, ax=ax, label='Sharpe difference'); fig.tight_layout(); fig.savefig(out / 'figures/01_sharpe_heatmap.png'); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    colors = plt.cm.viridis(np.tile(np.arange(7) / 6, 28))
    axes[0].bar(range(196), results.sharpe, color=colors, width=0.9)
    axes[0].axhline(bh['sharpe'], color='#b83c37', ls='--', label='Buy and hold'); axes[0].legend()
    axes[0].set(ylabel='Sharpe', title='All 196 pre-specified BTD rules | 1965-2025 Q3')
    axes[1].bar(range(196), results.alpha_ann * 100, color=colors, width=0.9)
    axes[1].axhline(0, color='black', lw=0.7); axes[1].set(ylabel='Annual alpha (%)', xlabel='Lookback, depth, then holding period (see grid_results.csv)')
    fig.tight_layout(); fig.savefig(out / 'figures/02_all_strategies.png'); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios':[2, 1]})
    labels = {'buy_hold':'Buy and hold', 'cash':'Cash', 'equal_weight_196':'Equal weight of 196 BTD rules', 'trend_252':'252-day long/cash trend'}
    for name, label in labels.items():
        axes[0].plot(nav.index, nav[name], label=label, lw=1.2)
        dd = nav[name] / np.maximum.accumulate(np.r_[1, nav[name].to_numpy()])[1:] - 1
        axes[1].plot(nav.index, dd * 100, lw=0.9)
    axes[0].set(yscale='log', ylabel='Wealth (log scale)', title='Gross price-return strategies, cash earns French RF'); axes[0].legend(fontsize=8)
    axes[1].set(ylabel='Drawdown (%)'); fig.tight_layout(); fig.savefig(out / 'figures/03_nav_drawdown.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(results.t_hac, bins=24, color='#417e95', edgecolor='white')
    for x in [-1.96, 1.96]: axes[0].axvline(x, ls='--', color='#b83c37')
    axes[0].set(xlabel='Alpha t-statistic (monthly HAC, 3 lags)', ylabel='Strategy count', title='Alpha uncertainty')
    sr = costs.groupby('one_way_bps').sharpe.agg(['mean', 'median'])
    axes[1].plot(sr.index, sr['mean'], marker='o', label='Mean BTD Sharpe'); axes[1].plot(sr.index, sr['median'], marker='s', label='Median BTD Sharpe')
    axes[1].axhline(bh['sharpe'], ls='--', color='#b83c37', label='Gross buy-and-hold reference')
    axes[1].set(xlabel='One-way cost (bps)', ylabel='Sharpe', title='Cost sensitivity'); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / 'figures/04_alpha_cost.png'); plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['baselines','versions']}, indent=2))


if __name__ == '__main__':
    run()

