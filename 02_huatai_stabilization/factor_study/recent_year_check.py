"""单独记录最近一年的组合结果；此文件的输出不进入报告生成链。"""
import json
import pandas as pd
from backtest import ROOT,OUT,load_data,run

if __name__=='__main__':
    indices,futures,expiries,calendar=load_data(evaluation=True)
    q=indices['IM'];end=pd.Timestamp('2026-11-09');start=pd.Timestamp('2025-11-10')
    result=run(q,futures,expiries,calendar,start=start,end=end)
    market=run(q,futures,expiries,calendar,start=start,end=end,opening_limit=False)
    continuous=run(q,futures,expiries,calendar,end=end)
    beginning=float(continuous['nav'].loc[continuous['nav'].index<start].iloc[-1])
    record=dict(start=str(start.date()),end=str(end.date()),initial_capital=100_000_000,
        threshold=-.0492,ratio_limit=1.,floor_days=3,execution='confirmed-close quantity, opening limit for hard margin cap, end-close mark',
        trading_days=len(result['nav']),**result['metrics'],
        target_budget_market_return=market['metrics']['total_return'],
        target_budget_market_max_drawdown=market['metrics']['max_drawdown'],
        continuous_account_start_nav=beginning,
        continuous_account_window_return=float(continuous['nav'].iloc[-1]/beginning-1),
        last_positions=result['trades'].loc[result['trades'].status=='open'].astype(str).to_dict('records'))
    (OUT/'private_recent_year.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    result['trades'].to_csv(OUT/'private_recent_year_trades.csv',index=False,encoding='utf-8-sig')
    result['rejected'].to_csv(OUT/'private_recent_year_rejected.csv',index=False)
    result['nav'].to_csv(OUT/'private_recent_year_nav.csv',index_label='date')
    print(json.dumps(record,ensure_ascii=False,indent=2))
