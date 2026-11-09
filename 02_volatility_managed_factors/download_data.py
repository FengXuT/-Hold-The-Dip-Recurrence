"""Download official factor inputs and a pinned paper/reference snapshot."""
import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAW = BASE / 'data' / 'raw'
AUTHOR_COMMIT = '66104218e702293b6b46d94c0c14ade4fb0cf0f6'
FRENCH = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
AUTHOR = f'https://raw.githubusercontent.com/tanishhky/Equity-Factor-Volatility-Analysis/{AUTHOR_COMMIT}/'
URLS = {
    'ff5_daily.zip': FRENCH + 'F-F_Research_Data_5_Factors_2x3_daily_CSV.zip',
    'ff5_monthly.zip': FRENCH + 'F-F_Research_Data_5_Factors_2x3_CSV.zip',
    'mom_daily.zip': FRENCH + 'F-F_Momentum_Factor_daily_CSV.zip',
    'mom_monthly.zip': FRENCH + 'F-F_Momentum_Factor_CSV.zip',
    'paper.pdf': AUTHOR + 'paper/letter.pdf',
    'reference_table1.tex': AUTHOR + 'paper/generated/table_full.tex',
    'reference_table2.tex': AUTHOR + 'paper/generated/table_decomp.tex',
    'reference_summary.csv': AUTHOR + 'output/realtime_summary.csv',
    'reference_periods.csv': AUTHOR + 'output/realtime_subperiods.csv',
    'reference_terciles.csv': AUTHOR + 'output/realtime_terciles.csv',
    'reference_terciles.png': AUTHOR + 'paper/images/fig_terciles.png',
    'reference_momentum.png': AUTHOR + 'paper/images/fig_momentum.png',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    for name, url in URLS.items():
        path = RAW / name
        if args.refresh or not path.exists():
            request = urllib.request.Request(url, headers={'User-Agent': 'QuantResearchReproduction/1.0'})
            content = urllib.request.urlopen(request, timeout=60).read()
            if name.endswith('.zip'):
                assert content.startswith(b'PK'), name
            elif name.endswith('.pdf'):
                assert content.startswith(b'%PDF'), name
            elif name.endswith('.png'):
                assert content.startswith(b'\x89PNG'), name
            elif name.endswith('.tex'):
                assert b' & ' in content, name
            else:
                assert content.startswith(b'factor,'), name
            path.write_bytes(content)
            manifest[name] = {'url': url, 'downloaded_utc': datetime.now(timezone.utc).isoformat()}
        entry = manifest.setdefault(name, {'url': url, 'snapshot_mtime_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()})
        entry.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print(f'{name}: {entry["bytes"]:,} bytes')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
