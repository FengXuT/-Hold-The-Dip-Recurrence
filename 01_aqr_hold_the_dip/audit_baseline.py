from pathlib import Path
import io
import json
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
market = pd.read_csv(ROOT / 'data/processed/market_daily.csv', index_col=0, parse_dates=True)
sample = market.loc['1965-01-01':'2025-09-30']
monthly = (1 + sample[['market_return', 'rf_french', 'rf']]).resample('ME').prod() - 1
with zipfile.ZipFile(ROOT / 'data/raw/ff_monthly.zip') as archive:
    text = archive.read(archive.namelist()[0]).decode('utf-8-sig')
lines = [line for line in text.splitlines() if len(line.split(',')[0]) == 6 and line.split(',')[0].isdigit()]
factors = pd.read_csv(io.StringIO('\n'.join(lines)), header=None, names=['date', 'mkt', 'smb', 'hml', 'rf'])
factors.index = pd.to_datetime(factors.date.astype(str), format='%Y%m').dt.to_period('M')
precise_cash = factors.rf.reindex(monthly.index.to_period('M')).to_numpy() / 100
if np.isnan(precise_cash).any():
    raise ValueError('Official French monthly RF does not cover the research period')
reference = json.loads((OUT / 'paper_comparison.json').read_text(encoding='utf-8'))
figure_benchmark = reference['paper_figure_benchmark_sharpe']
rows = []
for name, cash in [('french_daily', monthly.rf_french.to_numpy()),
                   ('french_monthly', precise_cash), ('bill_3m_proxy', monthly.rf.to_numpy())]:
    excess = monthly.market_return.to_numpy() - cash
    volatility = excess.std(ddof=1) * np.sqrt(12)
    sharpe = excess.mean() * 12 / volatility
    rows.append({'method': name, 'annual_mean_cash_return': float(cash.mean() * 12),
                 'annual_mean_equity_return': float(monthly.market_return.mean() * 12),
                 'annual_excess_volatility': float(volatility), 'benchmark_sharpe': float(sharpe),
                 'difference_to_figure': float(sharpe - figure_benchmark)})
cash_table = pd.DataFrame(rows)
cash_table.to_csv(OUT / 'baseline_cash_audit.csv', index=False)

frequency_rows = []
for frequency, annualizer in [('D', 252), ('W-FRI', 52), ('ME', 12), ('QE', 4), ('YE', 1)]:
    daily = sample[['market_return', 'rf']]
    returns = daily if frequency == 'D' else (1 + daily).resample(frequency).prod() - 1
    if frequency != 'D':
        returns = returns.loc[returns.index <= sample.index[-1]]
    excess = returns.market_return - returns.rf
    frequency_rows.append({'frequency': frequency, 'observations': len(returns), 'last_period_end': str(returns.index[-1].date()),
                           'benchmark_sharpe': float(excess.mean() / excess.std() * np.sqrt(annualizer))})
frequency_table = pd.DataFrame(frequency_rows)
frequency_table.to_csv(OUT / 'baseline_frequency_audit.csv', index=False)
old = cash_table.loc[cash_table.method == 'french_daily'].iloc[0]
new = cash_table.loc[cash_table.method == 'bill_3m_proxy'].iloc[0]
annual_cash_for_figure = new.annual_mean_cash_return + new.difference_to_figure * new.annual_excess_volatility
cagr = (1 + sample[['market_return', 'rf']]).prod() ** (252 / len(sample)) - 1
geometric = (cagr.market_return - cagr.rf) / (monthly.market_return.std() * np.sqrt(12))
summary = {'cash_methods': rows, 'frequency_methods': frequency_rows,
           'previous_benchmark_sharpe': float(old.benchmark_sharpe),
           'corrected_benchmark_sharpe': float(new.benchmark_sharpe),
           'cash_term_explained_fraction': float((old.benchmark_sharpe - new.benchmark_sharpe) / old.difference_to_figure),
           'annual_cash_mean_needed_for_figure_at_fixed_volatility': float(annual_cash_for_figure),
           'geometric_return_over_monthly_volatility': float(geometric),
           'text_implied_benchmark_approximation': 0.04 / 0.16,
           'text_implied_benchmark_difference': float(new.benchmark_sharpe - 0.04 / 0.16),
           'figure_gap_vs_text_gap_difference': float(reference['paper_figure_mean_sharpe_gap'] - (-0.04)),
           'cash_proxy': {'quote': 'Yahoo Finance ^IRX, annual discount yield in percent',
                          'roll': 'previous trading month-end, maturity three calendar months later',
                          'pricing': 'P=1-discount_yield*remaining_calendar_days/360',
                          'curve': 'flat: same 3M quote used for remaining maturity',
                          'missing_quotes': 'carry forward the most recent available quote'},
           'sources': ['https://www.treasurydirect.gov/marketable-securities/understanding-pricing/',
                       'https://www.sec.gov/Archives/edgar/data/789281/000162363222000679/fist-form.htm',
                       'https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/Driving-with-the-Rear-View-Mirror.pdf']}
assert abs(cash_table.loc[cash_table.method == 'french_monthly', 'benchmark_sharpe'].iloc[0] - old.benchmark_sharpe) < 0.002
assert new.benchmark_sharpe < old.benchmark_sharpe
assert all(np.isfinite(frequency_table.benchmark_sharpe))
(OUT / 'baseline_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(cash_table.to_string(index=False))
print(frequency_table.to_string(index=False))
print('Cash-term fraction of original figure gap explained:', summary['cash_term_explained_fraction'])
print('Text-implied benchmark (rounded values):', summary['text_implied_benchmark_approximation'])
