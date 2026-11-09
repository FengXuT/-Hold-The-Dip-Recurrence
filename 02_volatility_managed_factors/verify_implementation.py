"""Cross-check monthly paths against the author's pinned implementation on identical inputs."""
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from download_data import AUTHOR, AUTHOR_COMMIT
from run_research import FACTORS, OUT, load_inputs, managed, combination


def main():
    daily, monthly = load_inputs()
    rows = []
    with tempfile.TemporaryDirectory(prefix='factor_author_check_') as temporary:
        directory = Path(temporary)
        for name in ['realtime.py', 'vol_managed_factors.py']:
            request = urllib.request.Request(AUTHOR + 'src/' + name, headers={'User-Agent': 'QuantResearchReproduction/1.0'})
            (directory / name).write_bytes(urllib.request.urlopen(request, timeout=60).read())
        sys.path.insert(0, str(directory))
        import realtime as author
        for factor in FACTORS:
            reference_panel = author.monthly_panel(daily[factor], monthly[factor])
            reference = author.realtime_managed(reference_panel, 1.5)
            signal = (daily[factor]**2).ewm(halflife=21, adjust=False).mean().resample('ME').last() * 21
            own = managed(monthly[factor], signal)
            author_combo, _ = author.realtime_combination(reference, 1.5)
            series = [(own.weight, reference.w, 'weight'), (own.net, reference.net, 'net'),
                      (combination(own), author_combo, 'combination')]
            for first, second, label in series:
                assert np.array_equal(np.isfinite(first), np.isfinite(second))
                error = float(np.nanmax(np.abs(np.asarray(first) - np.asarray(second))))
                rows.append(dict(factor=factor, series=label, max_abs_error=error))
                assert error < 1e-12, (factor, label, error)
        sys.path.pop(0)
    result = dict(source_commit=AUTHOR_COMMIT, same_official_inputs=True, series=rows,
                  max_abs_error=max(row['max_abs_error'] for row in rows))
    (OUT / 'independent_crosscheck.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'18 monthly paths passed; maximum absolute error: {result["max_abs_error"]:.3g}')


if __name__ == '__main__':
    main()
