"""验证过去结果不受未来行情影响，并执行训练期封存后的时间留出检验。"""
import json
import pandas as pd
from backtest import OUT,load_data,run,features,base


def fit_training(q,futures,expiries,calendar,train_end):
    history=q.close.pct_change(5,fill_method=None).loc['2022-07-22':train_end].dropna()
    threshold=float(history.quantile(.05))
    rows=[]
    for p in [.75,1.,1.25]:
        for a in [2,3,4]:
            r=run(q,futures,expiries,calendar,end=train_end,threshold=threshold,ratio_limit=p,floor_days=a)
            rows.append(dict(ratio_limit=p,floor_days=a,**r['metrics']))
    eligible=[r for r in rows if r['closed_trades']>=3 and r['sharpe'] is not None]
    chosen=max(eligible,key=lambda r:(r['sharpe'],-abs(r['ratio_limit']-1)-abs(r['floor_days']-3))) if eligible else next(r for r in rows if r['ratio_limit']==1 and r['floor_days']==3)
    config=dict(train_start='2022-07-22',train_end=str(pd.Timestamp(train_end).date()),threshold=threshold,
        threshold_observations=len(history),ratio_limit=chosen['ratio_limit'],floor_days=int(chosen['floor_days']),
        training_closed_trades=chosen['closed_trades'],training_sharpe=chosen['sharpe'],selection='training Sharpe; at least 3 closed trades; else fixed 1/3')
    return config,pd.DataFrame(rows)


if __name__=='__main__':
    indices,futures,expiries,calendar=load_data(evaluation=True);q=indices['IM']
    full=run(q,futures,expiries,calendar)
    evidence={}
    snapshots=[]
    for cutoff in map(pd.Timestamp,['2023-01-03','2024-02-19','2024-02-20','2025-01-16','2026-08-31']):
        short_f={n:f.loc[:cutoff] for n,f in futures.items() if f.index.min()<=cutoff}
        short=run(q.loc[:cutoff],short_f,expiries,calendar,end=cutoff)
        expected=full['orders'].loc[full['orders'].confirm<=cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(short['orders'],expected)
        pd.testing.assert_series_equal(short['nav'],full['nav'].loc[:cutoff])
        snapshots.append(dict(cutoff=str(cutoff.date()),orders=len(short['orders']),open_trades=short['metrics']['open_trades'],nav=short['metrics']['final_nav']))
    evidence['prefix_snapshots']=snapshots
    cutoff=pd.Timestamp('2024-02-19')
    mutated_q=q.copy();mutated_q.loc[mutated_q.index>cutoff,['open','high','low','close']]*=1.7
    mutated_f={n:f.copy() for n,f in futures.items()}
    for f in mutated_f.values():f.loc[f.index>cutoff,['o','h','l','c']]*=1.9
    mutated=run(mutated_q,mutated_f,expiries,calendar)
    pd.testing.assert_series_equal(mutated['nav'].loc[:cutoff],full['nav'].loc[:cutoff])
    pd.testing.assert_frame_equal(mutated['orders'].loc[mutated['orders'].confirm<=cutoff].reset_index(drop=True),
                                  full['orders'].loc[full['orders'].confirm<=cutoff].reset_index(drop=True))
    # 在开盘前已经确定的手数，须独立于下一日的实际开盘价。
    changed_f={n:f.copy() for n,f in futures.items()}
    changed_f['IM2606'].loc[pd.Timestamp('2026-04-02'),'o']*=1.05
    altered=run(q,changed_f,expiries,calendar)
    a=full['orders'].loc[full['orders'].entry==pd.Timestamp('2026-04-02')].iloc[0]
    b=altered['orders'].loc[altered['orders'].entry==pd.Timestamp('2026-04-02')].iloc[0]
    assert a.quantity==b.quantity==66
    evidence.update(future_price_perturbation_passed=True,open_price_sizing_independence=True,
        order_20260402=dict(quantity=int(a.quantity),known_sizing_price=float(a.sizing_price)))

    # 修正前数字来自上次已提交报告683f5ea；旧筛选条件在2月19日会丢失该信号。
    prior=dict(final_nav=1.7814311057919996,annual_return=.15579523729831157,
               max_drawdown=.06557148575219596,sharpe=1.466400931692998)
    signal=features(q,-.0492).loc[cutoff]
    planned_exit=calendar[calendar.get_loc(cutoff)+21]
    assert signal.shock<-.0492 and signal.repair>=0 and signal.downside_ratio<=1 and signal.days_since_low>=3
    assert planned_exit>cutoff
    evidence['legacy_lost_signal']=dict(trigger=str(signal.trigger.date()),confirm=str(cutoff.date()),
        planned_exit=str(planned_exit.date()),new_pending_quantity=int(full['orders'].loc[full['orders'].confirm==cutoff].quantity.iloc[0]))
    evidence['before_metrics']=prior
    # 截断到已开仓的2月20日时，旧条件仍排除了这笔持仓。
    open_on_cutoff=next(row['open_trades'] for row in snapshots if row['cutoff']=='2024-02-20')
    assert open_on_cutoff==1
    evidence['terminal_position_example']=dict(cutoff='2024-02-20',
        old_position_included=False,new_open_trades=open_on_cutoff,
        first_day_net_profit=float(full['pnl'].loc[pd.Timestamp('2024-02-20')]))
    evidence['after_metrics']=full['metrics']
    (OUT/'audit.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')

    config,training_grid=fit_training(q,futures,expiries,calendar,'2023-11-30')
    training_grid.to_csv(OUT/'training_grid.csv',index=False)
    future_q=q.copy();future_q.loc[future_q.index>pd.Timestamp(config['train_end']),:]*=1.5
    future_f={n:f.copy() for n,f in futures.items()}
    for f in future_f.values():f.loc[f.index>pd.Timestamp(config['train_end']),:]*=1.6
    repeated_config,repeated_grid=fit_training(future_q,future_f,expiries,calendar,config['train_end'])
    assert repeated_config==config
    pd.testing.assert_frame_equal(repeated_grid,training_grid)
    evidence['training_future_perturbation_passed']=True
    (OUT/'audit.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    # 训练结束到测试开始之间保留至少20个交易日的间隔。
    embargo=calendar[(calendar>pd.Timestamp(config['train_end']))&(calendar<pd.Timestamp('2024-01-02'))]
    assert len(embargo)>=20
    config['embargo_sessions']=len(embargo)
    (OUT/'frozen_training_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    holdout=[];holdout_nav={}
    for label,method,p,a in [('baseline','baseline',1.,3),('fixed_combined','combined',1.,3),
                              ('training_selected','combined',config['ratio_limit'],config['floor_days'])]:
        r=run(q,futures,expiries,calendar,start=pd.Timestamp('2024-01-02'),end=base.END,method=method,
              threshold=config['threshold'],ratio_limit=p,floor_days=a)
        holdout.append(dict(method=label,**r['metrics']))
        holdout_nav[label]=r['nav'];r['trades'].to_csv(OUT/f'holdout_{label}_trades.csv',index=False)
    pd.DataFrame(holdout).to_csv(OUT/'holdout.csv',index=False)
    pd.DataFrame(holdout_nav).to_csv(OUT/'holdout_nav.csv',index_label='date')
    folds=[]
    for year,train_end in [(2024,'2023-11-30'),(2025,'2024-11-29'),(2026,'2025-11-28')]:
        fold_config,_=fit_training(q,futures,expiries,calendar,train_end)
        stop=min(base.END,pd.Timestamp(year,12,31))
        for label,method in [('baseline','baseline'),('selected','combined')]:
            r=run(q,futures,expiries,calendar,start=pd.Timestamp(year,1,1),end=stop,method=method,
                threshold=fold_config['threshold'],ratio_limit=fold_config['ratio_limit'],floor_days=fold_config['floor_days'])
            folds.append(dict(year=year,method=label,train_end=train_end,threshold=fold_config['threshold'],
                ratio_limit=fold_config['ratio_limit'],floor_days=fold_config['floor_days'],training_closed_trades=fold_config['training_closed_trades'],**r['metrics']))
    pd.DataFrame(folds).to_csv(OUT/'walk_forward.csv',index=False)

    # 新增区间参与了执行约束审查，归为回放对照。
    tail=[]
    for method in ['baseline','combined']:
        r=run(q,futures,expiries,calendar,start=pd.Timestamp('2026-09-01'),end=pd.Timestamp('2026-11-09'),method=method)
        tail.append(dict(method=method,**r['metrics']))
        r['trades'].to_csv(OUT/f'post_report_{method}_trades.csv',index=False)
    market=run(q,futures,expiries,calendar,start=pd.Timestamp('2026-09-01'),end=pd.Timestamp('2026-11-09'),opening_limit=False)
    tail.append(dict(method='market_target_budget',**market['metrics']))
    pd.DataFrame(tail).to_csv(OUT/'post_report_holdout.csv',index=False)
    strict=run(q,futures,expiries,calendar,start=pd.Timestamp('2026-09-01'),end=pd.Timestamp('2026-11-09'))
    strict['rejected'].to_csv(OUT/'post_report_rejected_orders.csv',index=False)
    print('Audit snapshots:',snapshots)
    print('Frozen configuration:',config)
    print('Holdout:',pd.DataFrame(holdout).round(5).to_string(index=False))
    print('Walk-forward:',pd.DataFrame(folds).round(5).to_string(index=False))
    print('Post-report holdout:',pd.DataFrame(tail).round(5).to_string(index=False))
