from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pypdf import PdfReader
from run_research import positions

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
parser = argparse.ArgumentParser(description='Compare 196 rules with AQR PDF vector bars.')
parser.add_argument('--self-check', action='store_true')
args = parser.parse_args()
reader = PdfReader(ROOT / 'data/raw/aqr_original.pdf')
reference = pd.DataFrame()
# Printed tick marks calibrate these chart-derived approximations in PDF points.
for metric, page, zero, scale in [('sharpe', 5, 564.0, (676.70 - 541.44) / 0.6),
                                ('alpha_ann', 6, 366.24, (451.74 - 309.30) / 0.05),
                                ('t_iid', 7, 629.04, (696.07 - 561.93) / 6)]:
    bars, path = [], []
    for operands, operator in reader.pages[page].get_contents().operations:
        if operator == b'm':
            path = [tuple(float(v) for v in operands)]
        elif operator == b'l':
            path.append(tuple(float(v) for v in operands))
        elif operator == b'h' and len(path) == 4:
            width = max(v[0] for v in path) - min(v[0] for v in path)
            if 1 < width < 2:
                bars.append((sum(v[0] for v in path) / 4, (path[0][1] - zero) / scale))
    if len(bars) != 196:
        raise ValueError(f'{metric}: expected 196 PDF bars, received {len(bars)}')
    reference[metric] = [value for _, value in sorted(bars)]
reference.to_csv(OUT / 'paper_reference.csv', index=False)
current = pd.read_csv(OUT / 'grid_results.csv')
market = pd.read_csv(ROOT / 'data/processed/market_daily.csv', index_col=0, parse_dates=True)
mask = (market.index >= '1965-01-01') & (market.index <= '2025-09-30')
mx = market.loc[mask, 'market_excess'].to_numpy()
X = np.column_stack([np.ones(len(mx)), mx])
intercept_variance_scale = np.linalg.inv(X.T @ X)[0, 0]
legacy = {1: [], 2: []}
for row in current.itertuples():
    signal = ((market.close / market.close.shift(row.lookback_days) - 1 <= -row.depth)
              & (market.index >= '1965-01-01')).to_numpy()
    for lag in [1, 2]:
        excess = positions(signal, row.hold_days, lag=lag)[mask] * mx
        coefficients = np.linalg.lstsq(X, excess, rcond=None)[0]
        residual = excess - X @ coefficients
        se = np.sqrt((residual @ residual) / (len(mx) - 2) * intercept_variance_scale)
        legacy[lag].append({'sharpe': excess.mean() / excess.std(ddof=1) * np.sqrt(252),
                            'alpha_ann': coefficients[0] * 252, 't_iid': coefficients[0] / se})
cases = {'prior_daily_lag2': pd.DataFrame(legacy[2]),
         'daily_lag1': pd.DataFrame(legacy[1]), 'corrected_monthly_lag1': current}
comparison = []
for name, values in cases.items():
    for metric in ['sharpe', 'alpha_ann', 't_iid']:
        error = values[metric] - reference[metric]
        comparison.append({'case': name, 'metric': metric,
                           'correlation': float(values[metric].corr(reference[metric])),
                           'rmse': float(np.sqrt(np.mean(error ** 2))),
                           'mean_error': float(error.mean()),
                           'reproduced_mean': float(values[metric].mean()),
                           'paper_figure_mean': float(reference[metric].mean())})
comparison = pd.DataFrame(comparison)
comparison.to_csv(OUT / 'paper_comparison.csv', index=False)
checked = comparison.loc[comparison.case == 'corrected_monthly_lag1'].set_index('metric')
if args.self_check:
    assert (checked.correlation > 0.98).all(), 'Curve shape still differs materially'
    assert checked.loc['sharpe', 'rmse'] < 0.03
    assert checked.loc['alpha_ann', 'rmse'] < 0.001
    assert checked.loc['t_iid', 'rmse'] < 0.15
summary = {'metrics': comparison.to_dict(orient='records'),
           'paper_figure_benchmark_sharpe': (610.44 - 564.0) / ((676.70 - 541.44) / 0.6),
           'paper_figure_mean_sharpe_gap': float(reference.sharpe.mean() - (610.44 - 564.0) / ((676.70 - 541.44) / 0.6)),
           'paper_figure_iid_significant_count': int((reference.t_iid.abs() > 1.96).sum()),
           'self_check_passed': bool(args.self_check),
           'source': 'AQR Hold the Dip, PDF pages 6-8; vector bars calibrated to printed axes'}
(OUT / 'paper_comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for ax, metric, title, multiplier in zip(axes, ['sharpe', 'alpha_ann', 't_iid'],
                                        ['Sharpe ratio', 'Annual alpha (%)', 'Alpha OLS t-statistic'], [1, 100, 1]):
    ax.plot(np.arange(196), cases['prior_daily_lag2'][metric] * multiplier,
            color='#d79246', lw=0.9, alpha=0.6, label='Prior: daily statistics, lag 2')
    ax.plot(np.arange(196), reference[metric] * multiplier,
            color='#263c4a', lw=1.5, label='AQR figure: digitized vector bars')
    ax.plot(np.arange(196), current[metric] * multiplier,
            color='#168da7', lw=1, ls='--', label='Corrected: monthly statistics, lag 1')
    ax.set_ylabel(title)
    ax.grid(axis='y', alpha=0.15)
    ax.set_title(f"{title}: corrected correlation {checked.loc[metric, 'correlation']:.4f}", loc='left')
axes[0].legend(fontsize=8, ncol=3, loc='upper left', bbox_to_anchor=(0, 1.30))
axes[-1].set_xlabel('196 rules in AQR order: lookback, dip depth, holding period')
fig.tight_layout(h_pad=1.3)
fig.savefig(OUT / 'figures/05_paper_overlay.png', dpi=170, bbox_inches='tight')
plt.close(fig)
print(comparison.to_string(index=False))
print('PDF reference t-statistics exceeding 1.96:', summary['paper_figure_iid_significant_count'])
print('Curve checks:', 'passed' if args.self_check else 'not requested')
