"""Independent rebuild of Tables 1/2 and Figure 1 in Yadav (September 2026)."""
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / 'data' / 'raw'
OUT = BASE / 'outputs'
FACTORS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
COST = 0.0014
CAP = 1.5
BURN = 120
SEED = 20260928
DRAWS = 4999


def read_french(name, monthly):
    with zipfile.ZipFile(RAW / name) as archive:
        lines = archive.read(archive.namelist()[0]).decode('latin1').splitlines()
    columns = next([c.strip() for c in line.split(',')[1:] if c.strip()] for line in lines if line.startswith(','))
    width = 6 if monthly else 8
    rows = [line for line in lines if re.match(r'^\d{' + str(width) + r'},', line)]
    frame = pd.read_csv(io.StringIO('\n'.join(rows)), names=['date'] + columns, index_col=0)
    frame.index = pd.to_datetime(frame.index.astype(str), format='%Y%m' if monthly else '%Y%m%d')
    if monthly:
        frame.index += pd.offsets.MonthEnd(0)
    if name.startswith('mom'):
        frame.columns = ['Mom']
    assert frame.index.is_unique and frame.index.is_monotonic_increasing, name
    assert np.isfinite(frame.to_numpy()).all() and (frame.to_numpy() > -99).all(), name
    return frame / 100


def load_inputs():
    daily = read_french('ff5_daily.zip', False).join(read_french('mom_daily.zip', False))
    monthly = read_french('ff5_monthly.zip', True).join(read_french('mom_monthly.zip', True))
    daily = daily.loc['1963-07-01':'2026-08-31', FACTORS]
    monthly = monthly.loc['1963-07-01':'2026-08-31', FACTORS]
    expected = pd.date_range('1963-07-31', '2026-08-31', freq='ME')
    assert monthly.index.equals(expected) and len(monthly) == 758
    assert not daily.isna().any().any() and not monthly.isna().any().any()
    assert daily.index.to_period('M').unique().equals(monthly.index.to_period('M'))
    return daily, monthly


def managed(returns, signal, cap=CAP, power=1.0):
    values = returns.to_numpy()
    risk = signal.shift(1).to_numpy() ** power
    weights = np.full(len(values), np.nan)
    # The June 1973 weight seeds July's turnover, as in the paper's implementation.
    for month in range(BURN - 1, len(values)):
        past_returns = values[1:month]
        inverse_scaled = past_returns / risk[1:month]
        scale = past_returns.std(ddof=1) / inverse_scaled.std(ddof=1)
        weights[month] = min(cap, scale / risk[month])
    panel = pd.DataFrame({'original': values, 'weight': weights}, index=returns.index)
    panel['turnover'] = panel.weight.diff().abs()
    panel['gross'] = panel.weight * panel.original
    panel['net'] = panel.gross - COST * panel.turnover
    return panel.iloc[BURN:].copy()


def combination(panel):
    original = panel.original.to_numpy()
    net = panel.net.to_numpy()
    weight = panel.weight.to_numpy()
    exposure = np.full(len(panel), np.nan)
    for month in range(59, len(panel)):
        history = np.column_stack([original[:month], net[:month]])
        mix = np.linalg.solve(np.cov(history, rowvar=False), history.mean(axis=0))
        scale = original[:month].std(ddof=1) / (history @ mix).std(ddof=1)
        exposure[month] = np.clip(scale * (mix[0] + mix[1] * weight[month]), -CAP, CAP)
    returns = exposure * original - COST * np.abs(np.diff(exposure, prepend=np.nan))
    returns[:60] = np.nan
    return returns


def sharpe(values):
    x = np.asarray(values)
    x = x[np.isfinite(x)]
    return float(np.sqrt(12) * x.mean() / x.std(ddof=1))


def sr_moments(moments):
    mean1, mean2, second1, second2 = np.moveaxis(moments, -1, 0)
    variance1, variance2 = second1 - mean1**2, second2 - mean2**2
    difference = mean1 / np.sqrt(variance1) - mean2 / np.sqrt(variance2)
    gradient = np.stack([second1 / variance1**1.5, -second2 / variance2**1.5,
                         -mean1 / (2 * variance1**1.5), mean2 / (2 * variance2**1.5)], axis=-1)
    return difference, gradient


def bootstrap_p(first, second, block=12):
    keep = np.isfinite(first) & np.isfinite(second)
    a, b = np.asarray(first)[keep], np.asarray(second)[keep]
    sample = np.column_stack([a, b, a*a, b*b])
    count = len(sample)
    moments = sample.mean(axis=0)
    difference, gradient = sr_moments(moments)
    centered = sample - moments
    covariance = centered.T @ centered / count
    for lag in range(1, block + 1):
        cross = centered[lag:].T @ centered[:-lag] / count
        covariance += (1 - lag / (block + 1)) * (cross + cross.T)
    standard_error = np.sqrt(gradient @ covariance @ gradient / count)
    blocks = int(np.ceil(count / block))
    starts = np.random.default_rng(SEED).integers(0, count, size=(DRAWS, blocks))
    indices = (starts[:, :, None] + np.arange(block)) % count
    # Keep complete blocks (648 observations for the main 638-month sample).
    resampled = sample[indices]
    draw_moments = resampled.mean(axis=(1, 2))
    draw_difference, draw_gradient = sr_moments(draw_moments)
    block_deviations = (resampled.sum(axis=2) - block * draw_moments[:, None, :]) / np.sqrt(block)
    block_covariance = np.einsum('bki,bkj->bij', block_deviations, block_deviations) / blocks
    draw_error = np.sqrt(np.einsum('bi,bij,bj->b', draw_gradient, block_covariance, draw_gradient) / (blocks * block))
    statistics = np.abs(draw_difference - difference) / draw_error
    return float((1 + np.count_nonzero(statistics >= abs(difference) / standard_error)) / (DRAWS + 1))


def holm(values):
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(np.asarray(values)[order] * np.arange(len(values), 0, -1)))
    return adjusted


def summed_loss(values, reference):
    x = np.asarray(values)
    cumulative = np.cumsum(x * np.std(reference, ddof=1) / np.std(x, ddof=1))
    return float(100 * np.min(cumulative - np.maximum.accumulate(cumulative)))


def main():
    OUT.mkdir(exist_ok=True)
    daily, monthly = load_inputs()
    ewma = (daily**2).ewm(halflife=21, adjust=False).mean().resample('ME').last() * 21
    canonical = daily.resample('ME').apply(lambda x: ((x - x.mean())**2).sum())
    summaries, decompositions, terciles, panels = [], [], [], []
    for factor in FACTORS:
        panel = managed(monthly[factor], ewma[factor])
        panel['combo'] = combination(panel)
        canon = managed(monthly[factor], canonical[factor])
        f, net = panel.original.to_numpy(), panel.net.to_numpy()
        summaries.append(dict(factor=factor, months=len(panel), sr_orig=sharpe(f), sr_canon=sharpe(canon.net),
                              sr_net=sharpe(net), p=bootstrap_p(net, f), sr_orig_cw=sharpe(f[60:]),
                              sr_combo=sharpe(panel.combo), p_c=bootstrap_p(panel.combo.to_numpy(), f),
                              dd_orig=summed_loss(f, f), dd_net=summed_loss(net, f)))
        for period, part in [('pre', panel.loc[:'2016-12-31']), ('post', panel.loc['2017-01-01':])]:
            exposure = part.weight.mean()
            average = part.original.mean()
            covariance = ((part.weight - exposure) * (part.original - average)).mean()
            costs = COST * part.turnover.mean()
            constant = exposure * average
            row = dict(factor=factor, period=period, months=len(part), avg_w=exposure, mean_f=12*average,
                       ew_ef=12*constant, cov_wf=12*covariance, cost=12*costs, mean_net=12*part.net.mean(),
                       M=(constant + covariance - costs)/constant if average > 0 else np.nan,
                       R=exposure*part.original.std(ddof=1)/part.net.std(ddof=1))
            decompositions.append(row)
            assert abs(row['mean_net'] - row['ew_ef'] - row['cov_wf'] + row['cost']) < 1e-14
            groups = pd.qcut(part.weight.rank(method='first'), 3, labels=['turbulent', 'middle', 'calm'])
            means = part.groupby(groups, observed=True).original.mean() * 12
            terciles.append(dict(factor=factor, period=period, **means.to_dict()))
        panels.append(panel.assign(factor=factor))
        print(f'{factor:6s} SR {summaries[-1]["sr_orig"]:.6f} -> {summaries[-1]["sr_net"]:.6f}', flush=True)
    summary = pd.DataFrame(summaries).set_index('factor')
    summary['p_holm'] = holm(summary.p.to_numpy())
    summary['p_c_holm'] = holm(summary.p_c.to_numpy())
    summary.to_csv(OUT / 'summary.csv')
    pd.DataFrame(decompositions).set_index(['factor', 'period']).to_csv(OUT / 'decomposition.csv')
    pd.DataFrame(terciles).set_index(['factor', 'period']).to_csv(OUT / 'terciles.csv')
    pd.concat(panels).rename_axis('date').reset_index().to_csv(OUT / 'monthly_paths.csv', index=False)

    baseline = managed(monthly.Mom, ewma.Mom)
    cutoff = pd.Timestamp('2008-12-31')
    altered_returns = monthly.Mom.copy()
    altered_signal = ewma.Mom.copy()
    altered_returns.loc[cutoff:] = 0.50
    altered_signal.loc[cutoff:] *= 7
    altered = managed(altered_returns, altered_signal)
    prior = baseline.index <= cutoff
    assert np.array_equal(baseline.loc[prior, 'weight'].to_numpy(), altered.loc[prior, 'weight'].to_numpy())
    assert (baseline.weight > 0).all() and (baseline.weight <= CAP).all()
    assert np.allclose(baseline.net, baseline.weight*baseline.original - COST*baseline.turnover, rtol=0, atol=1e-15)
    fixture = pd.Series([1., 4., 9.]).ewm(halflife=21, adjust=False).mean().to_numpy()
    decay = 2**(-1/21)
    expected = [1., decay + (1-decay)*4]
    expected.append(decay*expected[-1]+(1-decay)*9)
    assert np.allclose(fixture, expected, rtol=0, atol=1e-14)
    assert np.allclose(holm([.004, .20, .03]), [.012, .20, .06])
    # Finite differences cross-check the bootstrap gradient.
    moment = np.array([.01, .005, .002, .001])
    difference, gradient = sr_moments(moment)
    numerical = []
    for i in range(4):
        shifted = moment.copy(); shifted[i] += 1e-8
        numerical.append((sr_moments(shifted)[0] - difference) / 1e-8)
    assert np.allclose(gradient, numerical, rtol=3e-5)
    run = dict(input_start=str(monthly.index[0].date()), input_end=str(monthly.index[-1].date()),
               months=638, start='1973-07-31', end='2026-08-31', daily_observations=len(daily),
               seed=SEED, draws=DRAWS, block_months=12, cost_bps=14, cap=CAP,
               checks=['monthly coverage', 'EWMA recurrence', 'lagged information', 'exposure cap',
                       'cost accounting', '12 exact mean decompositions', 'Holm correction', 'bootstrap gradient'],
               versions=dict(numpy=np.__version__, pandas=pd.__version__))
    (OUT / 'run.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    print(summary.to_string())
    print('8 verification categories passed')


if __name__ == '__main__':
    main()
