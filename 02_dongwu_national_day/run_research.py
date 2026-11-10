"""Reproduce the event statistics in Figures 3 and 4 of the source report."""
import json
import numpy as np
import pandas as pd
from pypdf import PdfReader
from catalog import RAW, OUT, SERIES, WINDOWS, YEARS, END_DATE


def summarize(values):
    a = np.sort(np.asarray(values, dtype=float))
    a = a[np.isfinite(a)]
    return len(a), 100 * (a > 0).mean(), a[1:-1].mean()


def main():
    OUT.mkdir(exist_ok=True)
    prices = {}
    for source, code, name in SERIES:
        data = json.loads((RAW / f'{source}_{code}.json').read_text(encoding='utf-8'))['data']
        if source == 'cni':
            rows = [(row[0], row[5]) for row in data['data']]
        elif source == 'csi':
            rows = [(row['tradeDate'], row['close']) for row in data]
        elif source == 'citic':
            rows = [(row['tday'], row['close']) for row in data]
        else:
            rows = [(row['bargaindate'], row['closeindex']) for row in data]
        frame = pd.DataFrame(rows, columns=['date', 'close'])
        frame['date'] = pd.to_datetime(frame['date'])
        frame['close'] = pd.to_numeric(frame['close'])
        assert frame['date'].is_unique and frame['close'].gt(0).all(), name
        prices[name] = frame.set_index('date')['close'].sort_index().loc[:END_DATE]
    # CNI's actual sessions remove CSIndex's non-trading range-base observations.
    calendar = prices['大盘成长'].index
    panel = pd.DataFrame({name: series.reindex(calendar) for name, series in prices.items()})
    panel.to_csv(OUT / 'prices.csv', encoding='utf-8-sig', index_label='date')

    reference = []
    for line in PdfReader(RAW / 'paper.pdf').pages[3].extract_text().splitlines():
        cells = line.split()
        if len(cells) == 9 and cells[1].endswith('%'):
            for j, window in enumerate(WINDOWS):
                reference.append(dict(name=cells[0], window=window,
                                      win_rate=float(cells[j + 1].rstrip('%')),
                                      trimmed_mean=float(cells[j + 5].rstrip('%'))))
    reference = pd.DataFrame(reference)
    assert len(reference) == 48 * 4 and not reference.duplicated(['name', 'window']).any()
    reference.to_csv(OUT / 'reference.csv', index=False, encoding='utf-8-sig')

    events, paths, dates, coverage = [], [], [], []
    for year in [*YEARS, 2026]:
        t = calendar.searchsorted(pd.Timestamp(year, 10, 1)) - 1
        assert calendar[t].year == year and calendar[t].month == 9
        for window, (left, right) in WINDOWS.items():
            assert calendar[t + right].year == year
            dates.append(dict(year=year, window=window, anchor=str(calendar[t].date()),
                              start=str(calendar[t + left].date()), end=str(calendar[t + right].date()),
                              intervals=right-left))
            returns = (panel.iloc[t + right] / panel.iloc[t + left] - 1) * 100
            standard_left = {'T-10': -10, 'T-5': -5}.get(window, left)
            standard = (panel.iloc[t + right] / panel.iloc[t + standard_left] - 1) * 100
            for source, code, name in SERIES:
                events.append(dict(source=source, code=code, name=name, year=year, window=window,
                                   return_pct=returns[name], standard_return_pct=standard[name]))
        for offset in range(-15, 16):
            row = (panel.iloc[t + offset] / panel.iloc[t] - 1) * 100
            for name in ['沪深300', '中证1000', '国证2000']:
                paths.append(dict(name=name, year=year, offset=offset, return_pct=row[name]))
    events = pd.DataFrame(events)
    historical = events[events.year.isin(YEARS)]
    counts = historical.groupby('name').return_pct.count()
    selected = []
    for source, code, name in SERIES:
        s = prices[name]
        complete = counts[name] == 80
        included = source != 'sw' or complete
        if included:
            selected.append(name)
        coverage.append(dict(source=source, code=code, name=name, first_date=str(s.index.min().date()),
                             last_date=str(s.index.max().date()), historical_cells=int(counts[name]),
                             full_20_years=bool(complete), included=bool(included)))
    assert len(selected) == 32
    results, valid, standard = [], [], []
    for (name, window), group in historical[historical.name.isin(selected)].groupby(['name', 'window'], sort=False):
        a = group.return_pct.dropna().to_numpy()
        n, win, mean = summarize(a)
        valid.append(dict(name=name, window=window, n=n, win_rate=win, trimmed_mean=mean))
        # This source-table convention is exposed separately from valid-sample results.
        zeros = name == '国证2000' and window.startswith('T+')
        source_a = group.return_pct.fillna(0).to_numpy() if zeros else a
        sn, sw, sm = summarize(source_a)
        results.append(dict(name=name, window=window, n=sn, missing_as_zero=zeros,
                            win_rate=sw, trimmed_mean=sm))
        nn, ww, mm = summarize(group.standard_return_pct.dropna())
        standard.append(dict(name=name, window=window, n=nn, win_rate=ww, trimmed_mean=mm))
    for name, frame in [('source_stats', pd.DataFrame(results)), ('valid_stats', pd.DataFrame(valid)),
                        ('standard_window_stats', pd.DataFrame(standard)), ('event_returns', events),
                        ('event_dates', pd.DataFrame(dates)), ('coverage', pd.DataFrame(coverage)),
                        ('event_paths', pd.DataFrame(paths))]:
        frame.to_csv(OUT / f'{name}.csv', index=False, encoding='utf-8-sig')
    assert summarize([-100, 1, 2, 3, 100])[2] == 2
    assert pd.DataFrame(paths).query('offset == 0').return_pct.dropna().eq(0).all()
    print(f'{len(selected)} indices; {len(results)*2} original-table cells; event years 2006-2025')


if __name__ == '__main__':
    main()
