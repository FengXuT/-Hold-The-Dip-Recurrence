from pathlib import Path
import json
import base64
import numpy as np
import pandas as pd
import markdown

ROOT = Path(__file__).resolve().parent
out = ROOT / 'outputs'
summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
grid = pd.read_csv(out / 'grid_results.csv')
cost = pd.read_csv(out / 'cost_sensitivity.csv')
implementation = pd.read_csv(out / 'implementation_sensitivity.csv')
periods = pd.read_csv(out / 'period_results.csv')
market = pd.read_csv(ROOT / 'data/processed/market_daily.csv', index_col=0, parse_dates=True)
positions = pd.read_csv(out / 'positions.csv.gz', index_col=0, parse_dates=True)
daily = pd.read_csv(out / 'selected_daily_returns.csv', index_col=0, parse_dates=True)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows])


baseline_rows = []
names = {'buy_hold':'买入持有', 'cash':'现金', 'L21_D10_H63':'回看1月/跌10%/持有3月',
         'L63_D10_H252':'回看3月/跌10%/持有1年', 'L252_D20_H1260':'回看1年/跌20%/持有5年',
         'equal_weight_196':'196规则每日等权组合', 'trend_252':'252日趋势基线'}
for name, values in summary['baselines'].items():
    sr = '—' if values['sharpe'] is None else f"{values['sharpe']:.3f}"
    baseline_rows.append([names[name], f"{values['cagr']:.2%}", f"{values['volatility']:.2%}", sr, f"{values['max_drawdown']:.2%}"])
base_table = table(['规则/基线', '年化收益', '年化波动', '夏普', '最大回撤'], baseline_rows)
implementation_rows = []
for (mode, lag), values in implementation.groupby(['mode', 'lag']):
    implementation_rows.append(['持仓展期' if mode == 'extend' else '忽略新信号',
                                '信号下一收盘成交' if lag == 2 else '信号当日收盘成交近似',
                                f'{values.sharpe_gap.mean():+.3f}', f'{(values.sharpe_gap < 0).mean():.1%}'])
impl_table = table(['重复信号处理', '执行方式', '平均夏普差', '落后比例'], implementation_rows)
cost_rows = []
for bps, values in cost.groupby('one_way_bps'):
    cost_rows.append([bps, f'{values.sharpe.mean():.3f}', f'{values.cagr.mean():.2%}', f'{values.max_drawdown.mean():.2%}'])
cost_table = table(['单边成本/bps', '196组平均夏普', '196组平均年化收益', '196组平均最大回撤'], cost_rows)
period_rows = []
for label in ['1965-1989', '1990-2009', '2010-2025Q3', 'after_publication']:
    values = periods.loc[periods.period == label]
    period_rows.append(['发布后观察' if label == 'after_publication' else label, int(values.days.iloc[0]),
                        f'{values.sharpe.mean():.3f}', f'{values.benchmark_sharpe.iloc[0]:.3f}',
                        f'{(values.sharpe < values.benchmark_sharpe).mean():.1%}', f'{values.exposure.mean():.1%}'])
period_table = table(['阶段', '交易日数', '196组平均夏普', '买入持有夏普', '落后比例', '平均股票仓位'], period_rows)

sid = 'L21_D10_H63'
trigger = (market['close'] / market['close'].shift(21) - 1 <= -0.10)
signal_indices = np.flatnonzero(trigger & (market.index >= '1965-01-01') & (market.index <= '2025-09-30'))
last_sample = market.index.get_loc(positions.index[-1])
next_allowed = 0
trades = []
rebuilt = np.zeros(len(market), dtype=int)
for signal in signal_indices:
    if signal < next_allowed:
        continue
    next_allowed = signal + 63
    first = signal + 2
    if first > last_sample:
        continue
    scheduled_last = first + 63 - 1
    last = min(scheduled_last, last_sample)
    rebuilt[first:last + 1] = 1
    sample_returns = daily.loc[market.index[first]:market.index[last], sid]
    trades.append({'strategy':sid, 'signal_date':str(market.index[signal].date()),
                   'buy_close_date':str(market.index[signal + 1].date()),
                   'first_stock_return_date':str(market.index[first].date()),
                   'last_stock_return_date':str(market.index[last].date()),
                   'sell_close_date':str(market.index[last].date()) if scheduled_last <= last_sample else '',
                   'holding_return_days':last - first + 1,
                   'gross_stock_trade_return':float((1 + sample_returns).prod() - 1),
                   'status':'closed' if scheduled_last <= last_sample else 'open_at_sample_end'})
np.testing.assert_array_equal(pd.Series(rebuilt, index=market.index).loc[positions.index], positions[sid])
trades = pd.DataFrame(trades)
trades.to_csv(out / 'example_trades.csv', index=False)
trade_preview = table(['信号日', '下一收盘买入', '首个股票收益日', '卖出收盘日', '持有交易日', '该段股票收益'],
                      [[r.signal_date,r.buy_close_date,r.first_stock_return_date,r.sell_close_date,r.holding_return_days,f'{r.gross_stock_trade_return:.2%}'] for r in trades.head(5).itertuples()])

report = f'''# 抄底有用吗？AQR《Hold the Dip》公开数据复现结果

更新日期：2026-11-09。主样本为 **{summary['sample_start']} 至 {summary['sample_end']}，共 {summary['days']:,} 个交易日**。

## 先看结论

使用公开的标普500价格与现金利率，本次196组抄底规则的平均夏普为 **{summary['mean_sharpe']:.3f}**，买入持有为 **{summary['benchmark']['sharpe']:.3f}**；**{int((grid.sharpe_gap < 0).sum())}/196组（{summary['underperform_share']:.1%}）**落后。平均年化alpha为 **{summary['mean_alpha_ann']:.3%}**。HAC口径下有{summary['hac_significant_count']}组通过5%单项检验，经Holm多重检验校正后为{summary['holm_significant_count']}组。

这组实验支持“抄底规则整体难以改善风险调整后表现”的方向性判断。完整网格、成本与统计检验用于解释各规则的表现及其差异。

## 1. 与原文逐项比较

原文是AQR于2025-12-01发布的[《Hold the Dip》](https://www.aqr.com/insights/research/alternative-thinking/hold-the-dip)。下表原文数值来自[PDF第5—8页](https://www.aqr.com/-/media/AQR/Documents/Alternative-Thinking/AQR-Alternative-Thinking---Hold-the-Dip.pdf?sc_lang=en)；右列由本项目输出计算。

| 指标 | 原文 | 本次实测 |
|---|---|---|
| 策略数量 | 196 | {len(grid)} |
| 平均夏普减买入持有夏普 | 约-0.04 | {summary['mean_sharpe_gap']:+.3f} |
| 夏普落后的规则比例 | 超过60% | {summary['underperform_share']:.1%} |
| 平均年化alpha | 约0.5% | {summary['mean_alpha_ann']:.3%} |
| 传统t值绝对值超过1.96 | 16组 | {summary['iid_significant_count']}组（本项目普通OLS） |
| HAC(21)的p值小于0.05 | 原文未明确此口径 | {summary['hac_significant_count']}组 |
| Holm校正后p值小于0.05 | 本项目新增评价 | {summary['holm_significant_count']}组 |

**复现程度：核心实验已完成，结论方向相符，数值仍有明显差异。** 本次用Yahoo价格和French 1M现金收益，原文用Bloomberg及3M现金基准；固定交易日换算、下一日收盘成交、忽略持仓中新信号也是本次具体设定。原文对部分实施细节描述较简略，现有证据可用于呈现差异，并为进一步对齐提供依据。

## 2. 收益、风险与仓位一起看

{base_table}

表内为主样本毛收益，股票采用价格收益，空仓资金获得French RF。三条示例规则覆盖短、中、长持有期，展示用途相同；全部196条规则在CSV中保留。每个年化收益按252交易日换算。

买入持有年化收益为{summary['benchmark']['cagr']:.2%}，最大回撤为{summary['benchmark']['max_drawdown']:.2%}。抄底规则的股票平均仓位在全网格上为{grid.exposure.mean():.1%}，长期空仓的现金收益构成了策略回报的一部分。196规则组合的夏普为{summary['baselines']['equal_weight_196']['sharpe']:.3f}，其计算方式是每天将资金等分给全部规则后取平均日收益。

趋势基线用“价格高于252日前价格则持股”的单一规则，夏普为{summary['baselines']['trend_252']['sharpe']:.3f}。这个基线帮助比较顺势与抄底的仓位路径；AQR原文的SG Trend部分使用另一组数据与策略范围。

![196组参数结果](figures/02_all_strategies.png)

图1：上图为196条规则的夏普与买入持有参照线，下图为市场回归年化alpha。排列顺序为回看期、跌幅、持有期，便于与grid_results.csv对应。

![净值与回撤](figures/03_nav_drawdown.png)

图2：对数净值和回撤。不同策略的股票暴露不同，同时观察收益增长和风险路径有助于理解这种差别。

## 3. 参数结构：短持有与长持有的差别

![夏普差热力图](figures/01_sharpe_heatmap.png)

图3：横轴是持有期，纵轴是回看期，每格对四种跌幅取平均；数值为抄底夏普减买入持有夏普。全网格平均值为{summary['mean_sharpe_gap']:+.3f}。

按持有期汇总，21日持有规则的平均夏普为{grid.loc[grid.hold_days == 21, 'sharpe'].mean():.3f}，1260日持有规则为{grid.loc[grid.hold_days == 1260, 'sharpe'].mean():.3f}；对应平均股票仓位从{grid.loc[grid.hold_days == 21, 'exposure'].mean():.1%}上升到{grid.loc[grid.hold_days == 1260, 'exposure'].mean():.1%}。延长持有期后，规则逐渐增加股票暴露，夏普也更接近买入持有。每格的细分参数和平均仓位可在grid_results.csv核对。全网格净开仓次数介于{int(grid.entries.min())}和{int(grid.entries.max())}次之间，评价具体参数时可将这些信息一起读。

## 4. 对实现方式敏感吗

{impl_table}

三种实现下，整组规则平均夏普均低于买入持有。落后幅度和比例随实施方式发生变化，说明在复现中记录重复信号处理和成交时点有实际意义。主实验使用“下一日收盘成交、持仓期间忽略新信号”，以已经观察到的信号安排下一次收盘交易。

## 5. 费用与统计检验

{cost_table}

表中“平均年化收益”和“平均回撤”为196条规则各自指标的算术平均。买入持有参照采用毛收益。单边成本是每次买入或卖出按资金比例扣除的简化费用，成本提高对高换手规则影响更明显。

![Alpha与成本敏感性](figures/04_alpha_cost.png)

图4：左侧是HAC(21)的alpha t值分布，右侧是四档单边成本下的平均及中位夏普。主实验平均年化alpha为{summary['mean_alpha_ann']:.3%}，个体显著结果经过196项Holm校正后，达到5%水平的规则数为{summary['holm_significant_count']}。

回归截距回答在这一样本中、控制市场超额收益后，策略还有多少剩余收益。HAC和多重检验把时间依赖与多参数比较纳入同一评价流程，结果解释据此保持在当前样本和规则范围内。

## 6. 不同历史阶段与发布后观察

{period_table}

历史分段沿用原有仓位。阶段差异说明择时规则受市场路径影响，适合结合特定时期的波动和持仓记录继续分析。

发布后窗口是**2025-12-02至2026-09-30**，共{int(periods.loc[periods.period == 'after_publication', 'days'].iloc[0])}个交易日。该窗口展示固定规则在文章发布后的表现，报告同时呈现仓位和完整网格分布。窗口中的规则表现优劣以当前观察为范围，后续可按相同设定持续更新。

## 7. 看一笔交易，核对时间线

示例规则：回看21个交易日、跌幅达到10%、持有63个股票收益区间。前五条交易如下，完整清单在example_trades.csv。

{trade_preview}

信号收盘之后，下一交易日收盘成交，再从随后的日收益开始计入股票敞口。逐日仓位文件可以核对每个买入、持有和退出区间。`holding_return_days`统计实际纳入的股票日收益数量；样本结束仍有仓位时，交易清单用open_at_sample_end标记。若到期时接续一个新持有周期，两个周期在同一次收盘衔接；股票净仓位保持不变，费用按净仓位变化计算。报告生成时，用逐个触发周期独立重建示例仓位，并与主程序输出逐日核对。

## 8. 已完成的复现与接下来可研究的问题

已完成公开数据下载和快照留存、196组主实验、收益风险评价、OLS与HAC回归、Holm校正、执行与成本敏感性、历史分段、发布后观察，以及6类针对时间与资金逻辑的自检。

进一步研究可优先考虑两项：取得原文相同的3M现金数据及更详细的持仓实现说明，量化各口径对差距的贡献；把同一研究问题扩展到含分红的可交易ETF，结合开盘成交数据重新评估费用和可执行性。每次扩展保留一个明确的问题和固定实验设定，方便解释新增证据。

完整方法、数据来源和运行命令见仓库README。运行脚本后，原始数值保存在本地outputs/grid_results.csv，实际运行版本及自检记录保存在outputs/summary.json。
'''
body = markdown.markdown(report, extensions=['tables', 'fenced_code'])
for figure in (out / 'figures').glob('*.png'):
    encoded = base64.b64encode(figure.read_bytes()).decode('ascii')
    body = body.replace(f'src="figures/{figure.name}"', f'src="data:image/png;base64,{encoded}"')
css = '''body{max-width:1080px;margin:44px auto;padding:0 24px 80px;background:#f7f9fb;color:#243344;font:16px/1.85 system-ui,"Microsoft YaHei",sans-serif}h1{font-size:32px;color:#153c50;line-height:1.4}h2{margin-top:42px;padding-top:16px;border-top:1px solid #d8e3ea;color:#153c50;font-size:24px}p{margin:16px 0}a{color:#126b91}table{border-collapse:collapse;width:100%;background:white;font-size:14px;line-height:1.6}th,td{border:1px solid #d9e2e8;padding:9px 12px;text-align:left}th{background:#e8f0f5}img{display:block;max-width:100%;height:auto;margin:24px auto;border-radius:8px;background:white}code{background:#e8eef2;padding:2px 5px;border-radius:4px}strong{color:#124e68}@media print{body{max-width:none;margin:0;background:white;font-size:11pt}img,table{break-inside:avoid}h2{break-after:avoid}}'''
(ROOT / 'REPORT.html').write_text(f'<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AQR抄底策略公开数据复现</title><style>{css}</style><body>{body}</body></html>', encoding='utf-8')
print('Generated REPORT.html with embedded figures and local example trade ledger.')
print(trades.head().to_string(index=False))



