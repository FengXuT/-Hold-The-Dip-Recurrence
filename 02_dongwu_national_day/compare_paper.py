"""Compare every reproduced cell with the numbers extracted from the source PDF."""
import json
import numpy as np
import pandas as pd
from catalog import OUT


def main():
    own = pd.read_csv(OUT / 'source_stats.csv')
    ref = pd.read_csv(OUT / 'reference.csv')
    coverage = pd.read_csv(OUT / 'coverage.csv')
    table = own.merge(ref, on=['name', 'window'], suffixes=('_own', '_paper'), validate='one_to_one')
    table = table.merge(coverage[['name', 'source']], on='name', validate='many_to_one')
    rows = []
    for row in table.to_dict('records'):
        for metric in ['win_rate', 'trimmed_mean']:
            own_value, paper_value = row[metric + '_own'], row[metric + '_paper']
            tolerance = .50001 if metric == 'win_rate' else (.04 if row['source'] == 'citic' else .00501)
            error = own_value - paper_value
            rows.append(dict(name=row['name'], source=row['source'], window=row['window'], metric=metric,
                             own=own_value, paper=paper_value, error_pp=error, tolerance_pp=tolerance,
                             print_match=abs(error) <= (.50001 if metric == 'win_rate' else .00501),
                             pass_check=abs(error) <= tolerance))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / 'comparison.csv', index=False, encoding='utf-8-sig')
    summary = dict(indices=int(own.name.nunique()), cells=len(comparison),
                   print_matches=int(comparison.print_match.sum()),
                   wins_print_match=int(comparison.query("metric == 'win_rate'").print_match.sum()),
                   max_mean_error_pp=float(comparison.query("metric == 'trimmed_mean'").error_pp.abs().max()),
                   rmse_mean_pp=float(np.sqrt(np.mean(comparison.query("metric == 'trimmed_mean'").error_pp ** 2))),
                   all_pass=bool(comparison.pass_check.all()))
    (OUT / 'validation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    # Test the boundary inference jointly on all 11 complete CNI/CSI series.
    standard = pd.read_csv(OUT / 'standard_window_stats.csv')
    audit_rows = []
    full_names = coverage.loc[coverage.source.isin(['cni', 'csi']) & coverage.full_20_years, 'name']
    assert len(full_names) == 11
    for label, candidate in [('original_boundary', own), ('standard_N_intervals', standard)]:
        check = candidate[candidate.name.isin(full_names)].merge(ref, on=['name', 'window'], suffixes=('_own', '_paper'))
        for window, group in check[check.window.str.startswith('T-')].groupby('window'):
            mean_error = (group.trimmed_mean_own - group.trimmed_mean_paper).abs()
            win_error = (group.win_rate_own - group.win_rate_paper).abs()
            audit_rows.append(dict(boundary=label, window=window, indices=len(group),
                                   mean_print_matches=int(mean_error.le(.00501).sum()),
                                   win_print_matches=int(win_error.le(.50001).sum()),
                                   max_mean_error_pp=float(mean_error.max())))
    pd.DataFrame(audit_rows).to_csv(OUT / 'window_boundary_audit.csv', index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    assert comparison.pass_check.all(), comparison.loc[~comparison.pass_check].to_string(index=False)


if __name__ == '__main__':
    main()
