"""下载独立评估缓存，保留原研报复现的原始快照。"""
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlencode
from urllib.error import URLError
from time import sleep
import json,re,truststore

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'evaluation'
END='2026-11-09'
truststore.inject_into_ssl()

if __name__=='__main__':
    DATA.mkdir(parents=True,exist_ok=True)
    items=[('index.json','https://www.csindex.com.cn/csindex-home/perf/index-perf?'+urlencode(dict(indexCode='000852',startDate='20050101',endDate=END.replace('-','')))),
           ('calendar.json','https://hq.cnindex.com.cn/market/market/getIndexDailyDataWithDataFormat?'+urlencode(dict(indexCode='399372',startDate='2005-01-01',endDate=END,frequency='day')))]
    for name in ['IM2606','IM2609','IM2612','IM2703']:
        items.append((name+'.json','https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20x=/InnerFuturesNewService.getDailyKLine?symbol='+name))
    for name,url in items:
        request=Request(url,headers={'User-Agent':'Mozilla/5.0','Referer':'https://finance.sina.com.cn/'})
        for attempt in range(3):
            try:
                body=urlopen(request,timeout=60).read()
                break
            except URLError:
                if attempt==2:raise
                sleep(1)
        if name.startswith('IM'):
            body=json.dumps(json.loads(re.search(r'\(\s*(\[.*\])\s*\)',body.decode(),re.S).group(1))).encode()
        else:json.loads(body)
        (DATA/name).write_bytes(body)
        print(name,len(body))
    (DATA/'sources.json').write_text(json.dumps(dict(end=END,files=[dict(file=n,url=u) for n,u in items]),indent=2),encoding='utf-8')
