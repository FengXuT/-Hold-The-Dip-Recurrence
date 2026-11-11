"""核对来源表、逐笔交易及信息时间。"""
import json
import pandas as pd
import numpy as np
import reproduce as r

indices, futures, expiries = r.load_data()
nav, trades, result = r.simulate(indices, futures, expiries)
for date in ['20221223', '20230104', '20230130', '20230208']:
    official = pd.read_csv(r.RAW / 'cffex' / f'{date}.csv', encoding='gb18030')
    official.columns = official.columns.str.strip()
    row = official.loc[official['合约代码'].str.strip() == 'IM2303'].iloc[0]
    quote = futures['IM2303'].loc[pd.Timestamp(date)]
    assert abs(float(row['今开盘']) - quote.o) < 1e-6
    assert abs(float(row['今收盘']) - quote.c) < 1e-6

# 截断未来行情后，已完成的历史交易仍保持相同。
for cutoff in [pd.Timestamp('2024-06-28'), pd.Timestamp('2025-06-30')]:
    historical_indices = {key: frame.loc[:cutoff] for key, frame in indices.items()}
    historical_futures = {key: frame.loc[:cutoff] for key, frame in futures.items() if frame.index.min() <= cutoff}
    r.END = cutoff
    _, historical_trades, _ = r.simulate(historical_indices, historical_futures, expiries)
    expected = trades.loc[trades.exit <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(historical_trades, expected)

source_annual = np.array([[9.4,12.1,11.1,12.1,11.7],[14,13,13.4,14.8,14.4],
                         [14.6,13.3,15,16.2,14.7],[13.9,14.1,13.6,14.9,15.4],[13.7,14,15.2,13.2,14.2]])
source_drawdown = np.array([[23.8,24.7,24.8,24.9,25.5],[6.8,7,6.8,6.8,7.3],
                           [5.9,6.7,6.6,7.3,9.1],[4.9,6.4,6.8,7.6,8.4],[6.6,6.7,7.3,7.9,11.8]])
grid = pd.read_csv(r.OUT / 'parameter_grid.csv')
aerr = abs(grid.annual_return.to_numpy().reshape(5,5) * 100 - source_annual)
derr = abs(grid.max_drawdown.to_numpy().reshape(5,5) * 100 - source_drawdown)
assert aerr.max() < .06 and derr.max() < .06
cross = json.loads((r.OUT / 'metrics.json').read_text(encoding='utf-8'))
for prefix, count in [('IH',13), ('IF',13), ('IC',15), ('IM',15)]:
    assert cross[prefix]['trades'] == count
summary = dict(official_price_checks=8, historical_cutoffs_checked=2,
               grid_annual_error_pp=float(aerr.max()), grid_drawdown_error_pp=float(derr.max()))
(r.OUT / 'validation.json').write_text(json.dumps(summary, indent=2),encoding='utf-8')
print(json.dumps(summary, indent=2))
