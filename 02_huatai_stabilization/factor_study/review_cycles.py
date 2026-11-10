"""三轮独立复查：输入与信号、现金核算、运行输出与报告。"""
from decimal import Decimal, ROUND_FLOOR
from html.parser import HTMLParser
import argparse,json
import numpy as np
import pandas as pd
from backtest import ROOT,OUT,NAMES,load_data,features,run,base


class ReportReader(HTMLParser):
    def __init__(self):
        super().__init__();self.text=[];self.tables=0;self.charts=0
    def handle_starttag(self,tag,attrs):
        self.tables+=tag=='table';self.charts+=tag=='svg'
    def handle_data(self,data):
        self.text.append(data)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round',type=int,choices=[1,2,3],required=True)
    number=parser.parse_args().round
    indices,futures,expiries,calendar=load_data(evaluation=True)
    checks={}
    if number==1:
        base_indices,base_futures,_,_=load_data()
        assert len(base_futures)==72 and len(futures)==73
        assert '72个季度期货合约' in (ROOT.parent/'README.md').read_text(encoding='utf-8')
        count=0
        for name,q in list(indices.items())+list(futures.items()):
            assert q.index.is_monotonic_increasing and not q.index.has_duplicates,name
            cols=['open','high','low','close'] if 'close' in q else ['o','h','l','c']
            a=q[cols].to_numpy()
            assert np.isfinite(a).all() and (a>0).all(),name
            assert (a[:,1]+1e-8>=np.maximum(a[:,0],a[:,3])).all(),name
            assert (a[:,2]-1e-8<=np.minimum(a[:,0],a[:,3])).all(),name
            count+=len(a)
        pd.testing.assert_frame_equal(indices['IM'].loc[base_indices['IM'].index],base_indices['IM'])
        for name in ['IM2606','IM2609','IM2612']:
            pd.testing.assert_frame_equal(futures[name].loc[base_futures[name].index],base_futures[name])
        rows=0;orders_checked=0
        for prefix,q in indices.items():
            computed=features(q,base.SETTINGS[prefix][1])
            close=q.close.to_numpy();low=q.low.to_numpy()
            r=np.r_[np.nan,close[1:]/close[:-1]-1]
            for i in np.flatnonzero((q.index>=base.START)&(q.index<=base.END)):
                trigger=i-7
                pressure=np.sqrt(np.mean(np.minimum(r[i-2:i+1],0)**2))/np.sqrt(np.mean(np.minimum(r[trigger-20:trigger],0)**2))
                age=i-max(j for j in range(trigger,i+1) if low[j]==min(low[trigger:i+1]))
                expected=[close[trigger]/close[trigger-5]-1,close[i]/close[trigger]-1,pressure,age]
                actual=computed.iloc[i][['shock','repair','downside_ratio','days_since_low']].astype(float).to_numpy()
                np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-9)
                assert (actual[2]<=1)==(expected[2]<=1) and (actual[3]>=3)==(expected[3]>=3)
                rows+=1
            result=run(q,futures,expiries,calendar,prefix=prefix)
            for z in result['orders'].itertuples():
                e=calendar.get_loc(z.entry);c=calendar.get_loc(z.confirm)
                assert e==c+1 and calendar.get_loc(z.planned_exit)==e+20
                eligible=[n for n,f in futures.items() if n.startswith(prefix) and f.index.min()<=z.confirm
                          and ((calendar>z.entry)&(calendar<=expiries[n])).sum()>40]
                assert z.contract==min(eligible,key=lambda n:expiries[n])
                price=Decimal(str(futures[z.contract].loc[z.confirm,'c']))
                unit=Decimal(base.SETTINGS[prefix][2])*Decimal(str(z.margin_rate))
                quantity=int(Decimal('12000000')/(price*unit))
                limit=(Decimal('12000000')/(Decimal(quantity)*unit)/Decimal('.2')).to_integral_value(rounding=ROUND_FLOOR)*Decimal('.2')
                assert z.quantity==quantity and abs(z.limit_price-float(limit))<1e-8
                orders_checked+=1
        checks=dict(ohlc_rows=count,indices=len(indices),contracts=len(futures),independent_factor_rows=rows,
                    independently_checked_orders=orders_checked,overlapping_snapshots_equal=True)
    elif number==2:
        scenarios=[]
        for prefix,q in indices.items():
            for method in NAMES:
                scenarios.append((prefix,method,q,dict()))
        scenarios.extend([('IM','combined',indices['IM'],dict(start=pd.Timestamp('2026-09-01'),end=pd.Timestamp('2026-11-09'))),
                          ('IM','combined',indices['IM'],dict(fee_rate=base.FEE*2,slippage=1.))])
        checked_trades=0;checked_days=0
        for prefix,method,q,options in scenarios:
            result=run(q,futures,expiries,calendar,prefix=prefix,method=method,**options)
            nav=result['nav'];expected=pd.Series(0.,index=nav.index);exposure=expected.copy()
            fee=options.get('fee_rate',base.FEE);slip=options.get('slippage',0.)
            multiplier=base.SETTINGS[prefix][2]
            for z in result['trades'].itertuples():
                buy=float(futures[z.contract].loc[z.entry,'o'])+slip
                assert abs(buy-z.entry_price)<1e-9 and buy<=z.limit_price+1e-9
                after=nav.index[nav.index>=z.entry]
                marks=futures[z.contract].loc[after[after<=z.planned_exit],'c'].reindex(after).ffill()
                entry_fee=z.quantity*multiplier*buy*fee
                value=z.quantity*multiplier*(marks-buy)-entry_fee
                held=after
                if z.status=='closed':
                    sell=float(futures[z.contract].loc[z.planned_exit,'o'])-slip
                    assert z.exit==z.planned_exit and abs(sell-z.exit_price)<1e-9
                    terminal=z.quantity*multiplier*(sell-buy)-entry_fee-z.quantity*multiplier*sell*fee
                    value.loc[value.index>=z.exit]=terminal
                    held=after[after<z.exit]
                expected.loc[after]+=value
                exposure.loc[held]+=z.quantity*multiplier*buy/base.CAPITAL
                assert abs(value.iloc[-1]-z.profit)<1e-6
                checked_trades+=1
            np.testing.assert_allclose(nav,1+expected/base.CAPITAL,rtol=0,atol=1e-12)
            np.testing.assert_allclose(exposure,result['exposure'],rtol=0,atol=1e-12)
            values=nav.to_numpy();previous=np.r_[1.,values[:-1]];returns=values/previous-1
            annual=values[-1]**(250/len(values))-1;vol=np.std(returns,ddof=1)*np.sqrt(250)
            dd=1-values/np.maximum.accumulate(np.r_[1.,values])[1:]
            m=result['metrics']
            np.testing.assert_allclose([m['annual_return'],m['volatility'],m['max_drawdown'],m['sharpe']],
                [annual,vol,dd.max(),(annual-.02)/vol],rtol=1e-10,atol=1e-12)
            checked_days+=len(nav)
        checks=dict(scenarios=len(scenarios),trade_pnl_reconstructions=checked_trades,
                    independently_rebuilt_daily_values=checked_days,open_and_closed_positions_checked=True,
                    fees_slippage_and_statistics_checked=True)
    else:
        required=['metrics.json','audit.json','frozen_training_config.json','nav.csv','holdout.csv','holdout_nav.csv',
                  'training_grid.csv','walk_forward.csv','post_report_holdout.csv','post_report_rejected_orders.csv',
                  'sensitivity.csv','cross_index.csv','cost_stress.csv','private_recent_year.json']
        for name in required:assert (OUT/name).is_file(),name
        def bad_constant(value):
            raise AssertionError('Non-standard JSON number: '+value)
        json_count=0
        for path in OUT.glob('*.json'):
            json.loads(path.read_text(encoding='utf-8'),parse_constant=bad_constant);json_count+=1
        for path in OUT.glob('*.csv'):
            pd.read_csv(path)
        m=json.loads((OUT/'metrics.json').read_text())
        direct=run(indices['IM'],futures,expiries,calendar)['metrics']
        assert m['combined']==direct
        reader=ReportReader();reader.feed((ROOT/'REPORT.html').read_text(encoding='utf-8'))
        text=' '.join(reader.text)
        for value in ['15.55%','12.87%','18.36%','19.04%','91.17万元']:
            assert value in text,value
        assert reader.charts==2 and reader.tables>=12
        assert '2025-11-10' not in text and '26.03%' not in text and '21.10%' not in text
        checks=dict(required_outputs=len(required),valid_json_files=json_count,report_charts=reader.charts,
                    report_tables=reader.tables,pipeline_outputs_consistent=True,private_year_excluded=True)
    OUT.mkdir(exist_ok=True)
    evidence=dict(round=number,status='passed',scope=['行情/信号/选约','现金核算/成本/统计','完整运行/输出/报告'][number-1],checks=checks)
    if number>1:
        for prior in range(1,number):
            assert json.loads((OUT/f'review_round_{prior}.json').read_text(encoding='utf-8'))['status']=='passed'
    (OUT/f'review_round_{number}.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=False,indent=2))
