from pathlib import Path
from urllib.request import Request, urlopen
from datetime import datetime, timezone
import argparse
import json

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Download public daily S&P 500 prices and French RF.")
parser.add_argument('--refresh', action='store_true')
args = parser.parse_args()
start = int(datetime(1963, 1, 1, tzinfo=timezone.utc).timestamp())
end = int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp())
urls = {
    'gspc_daily.json': f'https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1={start}&period2={end}&interval=1d',
    'aqr_original.pdf': 'https://www.aqr.com/-/media/AQR/Documents/Alternative-Thinking/AQR-Alternative-Thinking---Hold-the-Dip.pdf?sc_lang=en',
    'ff_daily.zip': 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip',
}
raw = ROOT / 'data/raw'
raw.mkdir(parents=True, exist_ok=True)
manifest_path = raw / 'download_manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
for name, url in urls.items():
    target = raw / name
    if target.exists() and not args.refresh:
        print(f'Using cached {name}')
        continue
    request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(request, timeout=60) as response:
        content = response.read()
    if name.endswith('.json'):
        result = json.loads(content)['chart']
        if result.get('error') or not result.get('result'):
            raise ValueError('Yahoo returned no price data')
    elif name.endswith('.pdf'):
        if not content.startswith(b'%PDF-'):
            raise ValueError('AQR download is not a PDF file')
    elif not content.startswith(b'PK'):
        raise ValueError('French download is not a ZIP file')
    target.write_bytes(content)
    manifest[name] = {'url': url, 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                      'bytes': len(content), 'method': 'Python urllib'}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Downloaded {name}: {len(content):,} bytes')
