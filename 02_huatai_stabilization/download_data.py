"""下载华泰研报、官方指数日线及逐合约期货行情。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError
from time import sleep
import argparse
import json
import re
import truststore

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'raw'
REPORT_URL = 'https://www.htfc.com/wz_upload/png_upload/20260909/1788943656749408f5b.pdf'
truststore.inject_into_ssl()


def download(item):
    path, url, kind = item
    if path.exists() and not args.refresh:
        return
    request = Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.sina.com.cn/'})
    for attempt in range(3):
        try:
            body = urlopen(request, timeout=60).read()
            break
        except URLError:
            if attempt == 2:
                print('Download failed:', path.name, url)
                raise
            sleep(1)
    if kind == 'futures':
        match = re.search(r'\(\s*(\[.*\]|null)\s*\)', body.decode('utf-8'), re.S)
        rows = json.loads(match.group(1))
        assert rows and all({'d', 'o', 'h', 'l', 'c'} <= row.keys() for row in rows), path.name
        body = json.dumps(rows).encode('utf-8')
    elif kind == 'index':
        rows = json.loads(body)['data']
        assert rows and all({'tradeDate', 'open', 'high', 'low', 'close'} <= row.keys() for row in rows)
    elif kind == 'calendar':
        assert json.loads(body)['data']['data']
    elif kind == 'pdf':
        assert body.startswith(b'%PDF')
    elif kind == 'csv':
        assert '合约代码' in body.decode('gb18030')
    else:
        assert b'<html' in body.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    print('Downloaded', path.name, len(body))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', help='重新下载已有文件')
    args = parser.parse_args()
    items = [(RAW / 'report.pdf', REPORT_URL, 'pdf')]
    items.append((RAW / 'margin_change.html', 'https://www.bocifco.com/newsinfo.aspx?cid=3&id=48089', 'html'))
    for date in ['20221223', '20230104', '20230130', '20230208']:
        url = f'http://www.cffex.com.cn/sj/hqsj/rtj/{date[:6]}/{date[6:]}/{date}_1.csv'
        items.append((RAW / 'cffex' / f'{date}.csv', url, 'csv'))
    for code in ['000016', '000300', '000905', '000852']:
        url = 'https://www.csindex.com.cn/csindex-home/perf/index-perf?' + urlencode(dict(indexCode=code, startDate='20050101', endDate='20261106'))
        items.append((RAW / f'{code}.json', url, 'index'))
    url = 'https://hq.cnindex.com.cn/market/market/getIndexDailyDataWithDataFormat?' + urlencode(dict(indexCode='399372', startDate='2005-01-01', endDate='2026-11-06', frequency='day'))
    items.append((RAW / 'calendar.json', url, 'calendar'))
    for prefix in ['IM', 'IC', 'IF', 'IH']:
        for year in range(2022, 2027):
            for month in [3, 6, 9, 12]:
                if year == 2022 and month < 9:
                    continue
                name = f'{prefix}{year % 100:02d}{month:02d}'
                url = 'https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20x=/InnerFuturesNewService.getDailyKLine?symbol=' + name
                items.append((RAW / 'futures' / f'{name}.json', url, 'futures'))
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(download, items))
    (RAW / 'sources.json').write_text(json.dumps([dict(file=str(p.relative_to(RAW)), url=u) for p, u, _ in items], ensure_ascii=False, indent=2), encoding='utf-8')
