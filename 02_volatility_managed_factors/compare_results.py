"""Compare every reproduced Table 1/2 and Figure 1 value with pinned author outputs."""
import json
import re
import numpy as np
import pandas as pd
from run_research import RAW, OUT


def main():
    printed = {}
    table1_columns = ['sr_orig', 'sr_canon', 'sr_net', 'p_holm', 'sr_orig_cw', 'sr_combo', 'p_c_holm', 'dd_orig', 'dd_net']
    table2_columns = ['avg_w', 'mean_f', 'ew_ef', 'cov_wf', 'cost', 'mean_net', 'M', 'R']
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
    for table, filename, columns, skip in [('Table 1', 'reference_table1.tex', table1_columns, 1),
                                          ('Table 2', 'reference_table2.tex', table2_columns, 2)]:
        lines = [line for line in (RAW / filename).read_text().splitlines() if ' & ' in line]
        for number, line in enumerate(lines):
            cells = line.replace(r'\ensuremath{-}', '-').split(' & ')[skip:]
            key = factors[number] if table == 'Table 1' else (factors[number//2], ['pre', 'post'][number%2])
            for column, cell in zip(columns, cells):
                value = re.search(r'-?\d+(?:\.\d+)?', cell)
                if value:
                    printed[(table, key, column)] = float(value.group())
    rows = []
    specifications = [
        ('Table 1', 'summary.csv', 'reference_summary.csv', ['factor'],
         ['sr_orig', 'sr_canon', 'sr_net', 'sr_orig_cw', 'sr_combo', 'p_holm', 'p_c_holm', 'dd_orig', 'dd_net']),
        ('Table 2', 'decomposition.csv', 'reference_periods.csv', ['factor', 'period'],
         ['avg_w', 'mean_f', 'ew_ef', 'cov_wf', 'cost', 'mean_net', 'M', 'R']),
        ('Figure 1', 'terciles.csv', 'reference_terciles.csv', ['factor', 'period'],
         ['turbulent', 'middle', 'calm']),
    ]
    for label, own_name, reference_name, index, columns in specifications:
        own = pd.read_csv(OUT / own_name).set_index(index)
        reference = pd.read_csv(RAW / reference_name).set_index(index).loc[own.index]
        for key in own.index:
            for column in columns:
                a, b = own.loc[key, column], reference.loc[key, column]
                if pd.isna(a) and pd.isna(b):
                    continue
                assert np.isfinite(a) and np.isfinite(b)
                percentage = column in ['mean_f', 'ew_ef', 'cov_wf', 'cost', 'mean_net', 'turbulent', 'middle', 'calm']
                scale = 100 if percentage else 1
                error = abs(a - b) * scale
                tolerance = .02 if percentage or column.startswith('dd_') else .001
                if label == 'Table 1':
                    decimals = 0 if column.startswith('dd_') else 3 if 'holm' in column else 2
                    paper_reference = printed[(label, key, column)]
                    paper_tolerance = .5 * 10**(-decimals) + 1e-12
                    paper_error = abs(a*scale - paper_reference)
                elif label == 'Table 2':
                    decimals = 1 if percentage else 2
                    paper_reference = printed[(label, key, column)]
                    paper_tolerance = .5 * 10**(-decimals) + 1e-12
                    paper_error = abs(a*scale - paper_reference)
                else:
                    paper_reference = b*scale
                    paper_tolerance = .05
                    paper_error = error
                rows.append(dict(section=label, key=str(key), metric=column, reference=b*scale,
                                 reproduced=a*scale, abs_error=error, tolerance=tolerance, strict_passed=bool(error <= tolerance), paper_reference=paper_reference,
                                 paper_error=paper_error, paper_tolerance=paper_tolerance,
                                 passed=bool(paper_error <= paper_tolerance)))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / 'comparison.csv', index=False)
    groups = {}
    for section, group in comparison.groupby('section', sort=False):
        groups[section] = dict(compared=len(group), passed=int(group.passed.sum()))
    sr = comparison[comparison.metric.str.startswith('sr_')]
    mean_items = comparison[(comparison.section == 'Table 2') & comparison.metric.isin(['mean_f', 'ew_ef', 'cov_wf', 'cost', 'mean_net'])]
    figure = comparison[comparison.section == 'Figure 1']
    losses = comparison[comparison.metric.str.startswith('dd_')]
    result = dict(total=len(comparison), passed=int(comparison.passed.sum()), all_passed=bool(comparison.passed.all()),
                  sections=groups, strict_csv_passed=int(comparison.strict_passed.sum()),
                  max_sharpe_error=float(sr.abs_error.max()),
                  max_return_item_error_pp=float(mean_items.abs_error.max()),
                  max_tercile_error_pp=float(figure.abs_error.max()),
                  tercile_rmse_pp=float(np.sqrt(np.mean(figure.abs_error**2))),
                  max_loss_error_pp=float(losses.abs_error.max()),
                  max_adjusted_p_error=float(comparison[comparison.metric.str.contains('holm')].abs_error.max()))
    (OUT / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    assert result['all_passed'], 'Inspect outputs/comparison.csv for failed result comparisons'


if __name__ == '__main__':
    main()
