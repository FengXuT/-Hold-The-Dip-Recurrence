"""Download the original securities report and official index histories."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from urllib.parse import urlencode

import truststore
truststore.inject_into_ssl()
import urllib.request

from catalog import RAW, SERIES, PAPER_URL, END_DATE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    entries = [('paper.pdf', PAPER_URL, None)]
    for source, code, name in SERIES:
        body = None
        if source == 'cni':
            url = 'https://hq.cnindex.com.cn/market/market/getIndexDailyDataWithDataFormat?' + urlencode(dict(indexCode=code, startDate='2005-01-01', endDate=END_DATE, frequency='day'))
        elif source == 'csi':
            url = 'https://www.csindex.com.cn/csindex-home/perf/index-perf?' + urlencode(dict(indexCode=code, startDate='20050101', endDate=END_DATE.replace('-', '')))
        elif source == 'citic':
            url = 'https://research.citicsinfo.com/default/api/idx/common/hup/list'
            body = urlencode(dict(csId=code, startDate='2005-01-01', endDate=END_DATE)).encode()
        else:
            url = 'https://www.swsresearch.com/institute-sw/api/index_publish/trend/?' + urlencode(dict(swindexcode=code, period='DAY'))
        entries.append((f'{source}_{code}.json', url, body))
    for name, url, body in entries:
        path = RAW / name
        if args.refresh or not path.exists():
            request = urllib.request.Request(url, data=body, headers={'User-Agent': 'Mozilla/5.0', 'Content-Type': 'application/x-www-form-urlencoded'})
            content = urllib.request.urlopen(request, timeout=60).read()
            assert content.startswith(b'%PDF') if name.endswith('.pdf') else bool(json.loads(content)['data']), name
            path.write_bytes(content)
            manifest[name] = dict(url=url, method='POST' if body else 'GET', body=body.decode() if body else None,
                                  downloaded_utc=datetime.now(timezone.utc).isoformat())
        entry = manifest.setdefault(name, dict(url=url, method='POST' if body else 'GET', body=body.decode() if body else None,
                                              snapshot_mtime_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()))
        entry.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        print(name, entry['bytes'], flush=True)


if __name__ == '__main__':
    main()
