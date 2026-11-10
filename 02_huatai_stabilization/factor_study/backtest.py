"""只用确认日已知信息下单，次日开盘成交，期末持仓逐日估值。"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import reproduce as base
OUT=ROOT/'outputs'
NAMES={'baseline':'原企稳条件','pressure':'下行压力衰减','floor':'连续3日守底','combined':'双因子组合'}
START,END=base.START,base.END


def load_data(evaluation=False):
    indices,futures,expiries=base.load_data()
    if evaluation:
        folder=ROOT/'data'/'evaluation'
        dates=json.loads((folder/'calendar.json').read_text(encoding='utf-8'))['data']['data']
        calendar=pd.DatetimeIndex([row[0] for row in dates]).sort_values()
        q=pd.DataFrame(json.loads((folder/'index.json').read_text(encoding='utf-8'))['data'])
        q.index=pd.to_datetime(q.tradeDate)
        q=q[['open','high','low','close']].astype(float).reindex(calendar)
        assert q.notna().all().all() and not q.index.has_duplicates
        indices['IM']=q
        for file in folder.glob('IM*.json'):
            f=pd.DataFrame(json.loads(file.read_text(encoding='utf-8')))
            f.index=pd.to_datetime(f.d)
            futures[file.stem]=f[['o','h','l','c']].astype(float).sort_index()
            year,month=2000+int(file.stem[2:4]),int(file.stem[4:])
            first=pd.Timestamp(year,month,1)
            expiries[file.stem]=first+pd.Timedelta(days=(4-first.weekday())%7+14)
        # 2026-06-19为休市日，按交易所规则顺延至22日。
        expiries['IM2606']=pd.Timestamp('2026-06-22')
    calendar=indices['IM'].index
    # 当前最新行情为2026-11-09，选约所需的未来40日只延伸到2027-01-08。
    # 2027元旦安排已于2026-11-05公布；其余新增日期为普通工作日。
    extra=pd.bdate_range(calendar.max()+pd.Timedelta(days=1),'2027-01-08')
    extra=extra[extra!=pd.Timestamp('2027-01-01')]
    calendar=calendar.union(extra)
    return indices,futures,expiries,calendar


def features(q,threshold):
    r=q.close.pct_change(fill_method=None)
    down=r.clip(upper=0).pow(2)
    ratio=down.rolling(3).mean().pow(.5)/down.rolling(20).mean().pow(.5).shift(8)
    age=q.low.rolling(8).apply(lambda x:float(np.argmin(x[::-1])),raw=True)
    threshold_series=pd.Series(threshold,index=q.index) if np.isscalar(threshold) else threshold.reindex(q.index)
    return pd.DataFrame(dict(trigger=q.index.to_series().shift(7),shock=q.close.pct_change(5,fill_method=None).shift(7),
        threshold=threshold_series.shift(7),repair=q.close/q.close.shift(7)-1,downside_ratio=ratio,days_since_low=age))


def statistics(nav,trades,exposure):
    returns=nav.diff();returns.iloc[0]=nav.iloc[0]-1
    daily=returns/(nav.shift(1).fillna(1))
    annual=float(nav.iloc[-1]**(base.ANNUAL_DAYS/len(nav))-1)
    vol=float(daily.std()*np.sqrt(base.ANNUAL_DAYS))
    peak=nav.cummax().clip(lower=1)
    dd=float(-(nav/peak-1).min())
    closed=trades.loc[trades.status=='closed'] if len(trades) else trades
    return dict(trades=len(trades),closed_trades=len(closed),open_trades=len(trades)-len(closed),
        final_nav=float(nav.iloc[-1]),total_return=float(nav.iloc[-1]-1),annual_return=annual,
        volatility=vol,max_drawdown=dd,sharpe=(annual-.02)/vol if vol else None,
        mean_exposure=float(exposure.mean()),total_profit=float((nav.iloc[-1]-1)*base.CAPITAL),
        win_rate=float(closed.profit.gt(0).mean()) if len(closed) else None)


def run(q,futures,expiries,calendar,start=START,end=END,prefix='IM',method='combined',threshold=None,
        ratio_limit=1.,floor_days=3,fee_rate=base.FEE,slippage=0.,opening_limit=True):
    if threshold is None:threshold=base.SETTINGS[prefix][1]
    signal=features(q,threshold)
    dates=q.loc[start:end].index
    multiplier=base.SETTINGS[prefix][2]
    pnl=pd.Series(0.,index=dates);exposure=pnl.copy()
    orders,ledger,rejected=[],[],[]
    pending,position=None,None
    for date in dates:
        if position is not None and date==position['planned_exit']:
            price=float(futures[position['contract']].loc[date,'o'])-slippage
            amount=position['quantity']*multiplier*(price-position['last_mark'])
            amount-=position['quantity']*multiplier*price*fee_rate
            pnl.loc[date]+=amount;position['profit']+=amount
            position.update(exit=date,exit_price=price,status='closed',last_mark=price)
            position=None
        if pending is not None and date==pending['entry']:
            price=float(futures[pending['contract']].loc[date,'o'])+slippage
            if opening_limit and price>pending['limit_price']:
                rejected.append(dict(entry=date,contract=pending['contract'],quantity=pending['quantity'],
                                     opening_price=price,limit_price=pending['limit_price']))
            else:
                fee=pending['quantity']*multiplier*price*fee_rate
                position=dict(pending,entry_price=price,exit=pd.NaT,exit_price=np.nan,status='open',
                    last_mark=price,profit=-fee,entry_margin=pending['quantity']*multiplier*price*pending['margin_rate'])
                ledger.append(position);pnl.loc[date]-=fee
            pending=None
        if position is not None:
            price=float(futures[position['contract']].loc[date,'c'])
            amount=position['quantity']*multiplier*(price-position['last_mark'])
            pnl.loc[date]+=amount;position['profit']+=amount;position['last_mark']=price
            exposure.loc[date]=position['quantity']*multiplier*position['entry_price']/base.CAPITAL
        if position is not None or pending is not None:continue
        s=signal.loc[date]
        if not s.shock<s.threshold or not s.repair>=0:continue
        if method in ['pressure','combined'] and not s.downside_ratio<=ratio_limit:continue
        if method in ['floor','combined'] and not s.days_since_low>=floor_days:continue
        c=calendar.get_loc(date);entry=calendar[c+1];exit_day=calendar[c+21]
        eligible=[name for name,f in futures.items() if name.startswith(prefix)
            and f.index.min()<=date and expiries[name]>calendar[c+41]]
        name=min(eligible,key=lambda n:expiries[n])
        sizing_price=float(futures[name].loc[date,'c'])
        margin=.12 if entry>=pd.Timestamp('2023-05-05') else {'IM':.15,'IC':.14,'IF':.12,'IH':.12}[prefix]
        quantity=int(base.BUDGET/(sizing_price*multiplier*margin))
        limit_price=np.floor(base.BUDGET/(quantity*multiplier*margin)/.2)*.2
        pending=dict(limit_price=float(limit_price),trigger=s.trigger,confirm=date,entry=entry,planned_exit=exit_day,contract=name,
            quantity=quantity,margin_rate=margin,sizing_price=sizing_price,downside_ratio=float(s.downside_ratio),
            days_since_low=int(s.days_since_low),repair=float(s.repair),threshold=float(s.threshold))
        orders.append(pending.copy())
    trades=pd.DataFrame(ledger)
    if trades.empty:trades=pd.DataFrame(columns=['entry','exit','status','profit','entry_margin'])
    nav=1+pnl.cumsum()/base.CAPITAL
    assert abs(pnl.sum()-trades.profit.sum())<.01
    summary=statistics(nav,trades,exposure)
    summary['rejected_orders']=len(rejected)
    summary['max_entry_margin']=float(trades.entry_margin.max()/base.CAPITAL) if len(trades) else 0.
    if opening_limit:assert summary['max_entry_margin']<=.12+1e-12
    return dict(rejected=pd.DataFrame(rejected),nav=nav,pnl=pnl,exposure=exposure,trades=trades,orders=pd.DataFrame(orders),
                features=signal,metrics=summary)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    indices,futures,expiries,calendar=load_data()
    results={method:run(indices['IM'],futures,expiries,calendar,method=method) for method in NAMES}
    for method,result in results.items():
        result['trades'].to_csv(OUT/f'{method}_trades.csv',index=False,encoding='utf-8-sig')
        result['orders'].to_csv(OUT/f'{method}_orders.csv',index=False,encoding='utf-8-sig')
        result['rejected'].to_csv(OUT/f'{method}_rejected.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({m:r['nav'] for m,r in results.items()}).to_csv(OUT/'nav.csv',index_label='date')
    pd.DataFrame({m:r['exposure'] for m,r in results.items()}).to_csv(OUT/'exposure.csv',index_label='date')
    summaries={m:r['metrics'] for m,r in results.items()}
    (OUT/'metrics.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    grid=[]
    for p in [.75,1.,1.25]:
        for a in [2,3,4]:
            r=run(indices['IM'],futures,expiries,calendar,ratio_limit=p,floor_days=a)
            grid.append(dict(ratio_limit=p,floor_days=a,**r['metrics']))
    pd.DataFrame(grid).to_csv(OUT/'sensitivity.csv',index=False)
    cross=[]
    for prefix in base.SETTINGS:
        for method in ['baseline','pressure','combined']:
            r=run(indices[prefix],futures,expiries,calendar,prefix=prefix,method=method)
            cross.append(dict(prefix=prefix,method=method,**r['metrics']))
    pd.DataFrame(cross).to_csv(OUT/'cross_index.csv',index=False)
    stress=[]
    for method in NAMES:
        r=run(indices['IM'],futures,expiries,calendar,method=method,fee_rate=base.FEE*2,slippage=1.)
        stress.append(dict(method=method,**r['metrics']))
    pd.DataFrame(stress).to_csv(OUT/'cost_stress.csv',index=False)
    print(pd.DataFrame(summaries).T.round(6).to_string())
