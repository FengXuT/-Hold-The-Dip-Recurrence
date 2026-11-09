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
comparison = json.loads((out / 'paper_comparison.json').read_text(encoding='utf-8'))
curve_metrics = pd.DataFrame(comparison['metrics'])
aligned = curve_metrics.loc[curve_metrics.case == 'corrected_monthly_lag1'].set_index('metric')
prior = curve_metrics.loc[curve_metrics.case == 'prior_daily_lag2'].set_index('metric')
previous = curve_metrics.loc[curve_metrics.case == 'previous_monthly_french_lag1'].set_index('metric')
audit = json.loads((out / 'baseline_audit.json').read_text(encoding='utf-8'))
cost = pd.read_csv(out / 'cost_sensitivity.csv')
implementation = pd.read_csv(out / 'implementation_sensitivity.csv')
periods = pd.read_csv(out / 'period_results.csv')
market = pd.read_csv(ROOT / 'data/processed/market_daily.csv', index_col=0, parse_dates=True)
positions = pd.read_csv(out / 'positions.csv.gz', index_col=0, parse_dates=True)
daily = pd.read_csv(out / 'selected_daily_returns.csv', index_col=0, parse_dates=True)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows])


benchmark_rows = []
benchmark_names = {'benchmark_sharpe': '买入持有夏普', 'underperform_share': '落后规则比例',
                   'mean_sharpe_gap': '平均夏普差'}
for item in comparison['benchmark_metrics']:
    if item['metric'] == 'underperform_share':
        original, reproduced = f"约{item['paper']:.0%}", f"{item['reproduced']:.1%}"
        difference = f"{item['difference'] * 100:+.1f}个百分点"
    else:
        original, reproduced = f"{item['paper']:.3f}", f"{item['reproduced']:.3f}"
        difference = f"{item['difference']:+.3f}"
    benchmark_rows.append([benchmark_names[item['metric']], original, reproduced, difference])
benchmark_table = table(['相对表现指标', '原图提取值', '本次复现', '差值'], benchmark_rows)
alignment_status = '参数曲线及基准均达到对齐阈值' if comparison['full_replication_aligned'] else '参数曲线形态接近，基准及相对表现仍待对齐'

cash_names = {'french_daily': '原French日频现金', 'french_monthly': 'French官方月度现金',
              'bill_3m_proxy': '本次3个月月末滚动估计'}
cash_rows = [[cash_names[item['method']], f"{item['annual_mean_cash_return']:.3%}",
              f"{item['benchmark_sharpe']:.4f}", f"{item['difference_to_figure']:+.4f}"]
             for item in audit['cash_methods']]
cash_table = table(['现金口径', '年化平均现金收益', '买入持有夏普', '相对原图基准差值'], cash_rows)
frequency_names = {'D': '日频', 'W-FRI': '周频', 'ME': '月频', 'QE': '季频', 'YE': '年频'}
frequency_table = table(['统计频率', '观察数', '最后完整周期', '被动夏普'],
                        [[frequency_names[item['frequency']], item['observations'], item['last_period_end'], f"{item['benchmark_sharpe']:.4f}"]
                         for item in audit['frequency_methods']])

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
    valid = values.loc[values.sharpe.notna()]
    if valid.empty:
        mean_sharpe, underperform = '—', '—'
    else:
        losers = int((valid.sharpe < valid.benchmark_sharpe).sum())
        mean_sharpe = f'{valid.sharpe.mean():.3f}'
        underperform = f'{losers}/{len(valid)} ({losers / len(valid):.1%})'
    period_rows.append(['发布后观察' if label == 'after_publication' else label, int(values.days.iloc[0]),
                        f'{len(valid)}/{len(values)}', int((values.exposure == 0).sum()), mean_sharpe,
                        f'{values.benchmark_sharpe.iloc[0]:.3f}', underperform, f'{values.exposure.mean():.1%}'])
period_table = table(['阶段', '交易日数', '有效夏普/总规则', '全程现金组数', '有效规则平均夏普',
                      '买入持有夏普', '有效规则落后数/比例', '全196组平均仓位'], period_rows)
published = periods.loc[periods.period == 'after_publication']

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
    first = signal + 1
    if first > last_sample:
        continue
    scheduled_last = first + 63 - 1
    last = min(scheduled_last, last_sample)
    rebuilt[first:last + 1] = 1
    sample_returns = daily.loc[market.index[first]:market.index[last], sid]
    trades.append({'strategy':sid, 'signal_date':str(market.index[signal].date()),
                   'buy_close_date':str(market.index[signal].date()),
                   'first_stock_return_date':str(market.index[first].date()),
                   'last_stock_return_date':str(market.index[last].date()),
                   'sell_close_date':str(market.index[last].date()) if scheduled_last <= last_sample else '',
                   'holding_return_days':last - first + 1,
                   'gross_stock_trade_return':float((1 + sample_returns).prod() - 1),
                   'status':'closed' if scheduled_last <= last_sample else 'open_at_sample_end'})
np.testing.assert_array_equal(pd.Series(rebuilt, index=market.index).loc[positions.index], positions[sid])
trades = pd.DataFrame(trades)
trades.to_csv(out / 'example_trades.csv', index=False)
trade_preview = table(['信号日', '信号收盘买入', '首个股票收益日', '卖出收盘日', '持有交易日', '该段股票收益'],
                      [[r.signal_date,r.buy_close_date,r.first_stock_return_date,r.sell_close_date,r.holding_return_days,f'{r.gross_stock_trade_return:.2%}'] for r in trades.head(5).itertuples()])

report = f'''# 抄底有用吗？AQR《Hold the Dip》公开数据复现结果

更新日期：2026-11-10。主样本为 **{summary['sample_start']} 至 {summary['sample_end']}，共 {summary['days']:,} 个交易日**。

## 先看结论

使用公开的标普500价格与3个月国债报价，本次196组抄底规则的平均夏普为 **{summary['mean_sharpe']:.3f}**，买入持有为 **{summary['benchmark']['sharpe']:.3f}**；**{int((grid.sharpe_gap < 0).sum())}/196组（{summary['underperform_share']:.1%}）**落后。平均年化alpha为 **{summary['mean_alpha_ann']:.3%}**。HAC口径下有{summary['hac_significant_count']}组通过5%单项检验，经Holm多重检验校正后为{summary['holm_significant_count']}组。

这组实验支持“抄底规则整体难以改善风险调整后表现”的方向性判断。完整网格、成本与统计检验用于解释各规则的表现及其差异。

## 0. 与原图逐参数对照

下面按照原文的回看期、跌幅和持有期顺序，对照全部196组规则。参考值来自原文PDF第6—8页的矢量柱高，使用坐标轴刻度换算；它们具有图形读数精度。橙线为最初的日频版本，紫线为使用French现金的月频版本，深色线为原图参考值，蓝色虚线为本次3个月现金估计版本。

![原图与修改前后叠加](figures/05_paper_overlay.png)

图1：夏普、年化alpha和OLS t值的逐参数叠加。每一点对应同一条参数规则。

| 指标 | 初版日频RMSE | 上版1M现金月频RMSE | 本次3M现金RMSE | 本次相关系数 |
|---|---|---|---|---|
| 夏普 | {prior.loc['sharpe', 'rmse']:.4f} | {previous.loc['sharpe', 'rmse']:.4f} | {aligned.loc['sharpe', 'rmse']:.4f} | {aligned.loc['sharpe', 'correlation']:.4f} |
| 年化alpha/百分点 | {prior.loc['alpha_ann', 'rmse'] * 100:.4f} | {previous.loc['alpha_ann', 'rmse'] * 100:.4f} | {aligned.loc['alpha_ann', 'rmse'] * 100:.4f} | {aligned.loc['alpha_ann', 'correlation']:.4f} |
| OLS t值 | {prior.loc['t_iid', 'rmse']:.4f} | {previous.loc['t_iid', 'rmse']:.4f} | {aligned.loc['t_iid', 'rmse']:.4f} | {aligned.loc['t_iid', 'correlation']:.4f} |

**当前对齐状态：{alignment_status}。**

{benchmark_table}

原图与本次结果分别按各自的买入持有基准计算落后比例；此项和基准夏普单独纳入对照检查，原图比例按图形读数近似呈现。

主要修正是收益统计频率和成交近似：保留原先的固定交易日网格与忽略重复信号规则，将每日收益按月份复利为`月收益=prod(1+日收益)-1`，随后进行月频夏普计算及市场回归。回归截距乘12得到年化alpha。主复现从信号后下一段收盘收益计入仓位；下一日收盘成交另作敏感性分析。

只修正成交近似、仍用日频统计时，平均年化alpha为{curve_metrics.loc[(curve_metrics.case == 'daily_lag1') & (curve_metrics.metric == 'alpha_ann'), 'reproduced_mean'].iloc[0]:.3%}；在相同French现金下使用月频后为{previous.loc['alpha_ann', 'reproduced_mean']:.3%}。这说明月频聚合解释了主要的alpha水平差距。日历月与持仓展期也作过诊断比较；最终保留与原图更吻合的固定交易日和忽略重复信号设定。

### 被动基准差异的核对

原文现金基准为ICE BofA 3M T-Bill指数。本次用公开的[13周国债报价](https://finance.yahoo.com/quote/%5EIRX/)构造3个月短债现金估计，每月末重新建立3个月到期的零息票据，持有期间计入贴现收益和价格变动。这与该指数[月末选择短期国债并滚动持有的定义](https://www.sec.gov/Archives/edgar/data/789281/000162363222000679/fist-form.htm)对应。

估值采用[TreasuryDirect的贴现定价公式](https://www.treasurydirect.gov/marketable-securities/understanding-pricing/)：`价格=面值×(1-贴现率×剩余日数/360)`，每日现金收益为相邻估值价格之比减1。使用三个日历月的初始期限、实际日数和360天计息基准；剩余期限按当日3个月报价估值，相当于平坦期限曲线假设。缺少报价的交易日沿用最近报价。该序列是公开数据估计，精确对齐ICE指数还涉及具体票据选择和期限曲线价格。

保持股票价格、参数和成交设定相同，现金口径的影响如下。年化平均现金收益为月收益均值乘12。

{cash_table}

将French日频现金改为其官方月度现金后，被动夏普仅小幅变化，说明日频小数舍入对差距影响很小。采用3个月月末滚动估计后，被动夏普由{audit['previous_benchmark_sharpe']:.4f}降至{audit['corrected_benchmark_sharpe']:.4f}，解释了相对原图基准差距的约{audit['cash_term_explained_fraction']:.1%}；夏普曲线RMSE从{previous.loc['sharpe', 'rmse']:.4f}降至{aligned.loc['sharpe', 'rmse']:.4f}。

频率诊断使用已经完成的自然统计周期，最后周期日期一并列出：

{frequency_table}

月频同时保留了与原图高度接近的参数结构。作为另一项定义诊断，以年化几何收益差除以月收益年化波动得到{audit['geometric_return_over_monthly_volatility']:.4f}；本项目继续采用算术超额收益计算夏普，[AQR其他方法说明](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/Driving-with-the-Rear-View-Mirror.pdf)也明确采用该定义。

原文正文称平均夏普落后约0.04、相对下降约16%，由这两个四舍五入数字隐含的被动夏普约为`0.04/0.16=0.25`。本次3个月现金估计得到{summary['benchmark']['sharpe']:.4f}，与这一近似参照相差{audit['text_implied_benchmark_difference']:+.4f}。原图基准线则约为{comparison['paper_figure_benchmark_sharpe']:.3f}，196根柱子的均值约为{aligned.loc['sharpe', 'paper_figure_mean']:.3f}，图形读数的平均差约为{comparison['paper_figure_mean_sharpe_gap']:+.3f}。正文摘要和原图读数保留为两类参照。

相对原图尚余约{summary['benchmark']['sharpe'] - comparison['paper_figure_benchmark_sharpe']:.4f}的被动夏普差距。在固定超额波动的均值分解中，若单靠现金平均收益解释这一差距，需要年化平均现金收益约{audit['annual_cash_mean_needed_for_figure_at_fixed_volatility']:.2%}，而本次估计约{audit['cash_methods'][2]['annual_mean_cash_return']:.2%}。进一步逐点对齐需要原文实际现金指数序列和计算说明；当前结果按公开数据3个月现金估计解释，整体状态继续标记为部分复现。

## 1. 与原文逐项比较

原文是AQR于2025-12-01发布的[《Hold the Dip》](https://www.aqr.com/insights/research/alternative-thinking/hold-the-dip)。下表原文数值来自[PDF第5—8页](https://www.aqr.com/-/media/AQR/Documents/Alternative-Thinking/AQR-Alternative-Thinking---Hold-the-Dip.pdf?sc_lang=en)；右列由本项目输出计算。

| 指标 | 原文 | 本次实测 |
|---|---|---|
| 策略数量 | 196 | {len(grid)} |
| 平均夏普减买入持有夏普 | 约-0.04 | {summary['mean_sharpe_gap']:+.3f} |
| 夏普落后的规则比例 | 超过60% | {summary['underperform_share']:.1%} |
| 平均年化alpha | 约0.5% | {summary['mean_alpha_ann']:.3%} |
| 传统t值绝对值超过1.96 | 16组 | {summary['iid_significant_count']}组（本项目普通OLS） |
| 月频HAC(3)的p值小于0.05 | 原文未明确此口径 | {summary['hac_significant_count']}组 |
| Holm校正后p值小于0.05 | 本项目新增评价 | {summary['holm_significant_count']}组 |

**复现程度：{alignment_status}。** 本次使用Yahoo价格、3个月国债月末滚动现金估计、固定交易日周期、信号收盘成交近似及月频统计。原文使用Bloomberg价格及3M现金基准，并未明确给出统计频率与全部执行细节；月频设定由逐参数对照提供支持。剩余差异保留在指标表和叠加图中。

## 2. 收益、风险与仓位一起看

{base_table}

表内为主样本毛收益，股票采用价格收益，空仓资金获得下述3个月短债现金估计收益。日收益先复利汇总到月度，再计算年化波动和夏普。三条示例规则覆盖短、中、长持有期，展示用途相同；全部196条规则在CSV中保留。每个年化收益按252交易日换算。

买入持有年化收益为{summary['benchmark']['cagr']:.2%}，最大回撤为{summary['benchmark']['max_drawdown']:.2%}。抄底规则的股票平均仓位在全网格上为{grid.exposure.mean():.1%}，长期空仓的现金收益构成了策略回报的一部分。196规则组合的夏普为{summary['baselines']['equal_weight_196']['sharpe']:.3f}，其计算方式是每天将资金等分给全部规则后取平均日收益。

趋势基线用“价格高于252日前价格则持股”的单一规则，夏普为{summary['baselines']['trend_252']['sharpe']:.3f}。这个基线帮助比较顺势与抄底的仓位路径；AQR原文的SG Trend部分使用另一组数据与策略范围。

![196组参数结果](figures/02_all_strategies.png)

图2：上图为196条规则的夏普与买入持有参照线，下图为市场回归年化alpha。排列顺序为回看期、跌幅、持有期，便于与grid_results.csv对应。

![净值与回撤](figures/03_nav_drawdown.png)

图3：对数净值和回撤。不同策略的股票暴露不同，同时观察收益增长和风险路径有助于理解这种差别。

## 3. 参数结构：短持有与长持有的差别

![夏普差热力图](figures/01_sharpe_heatmap.png)

图4：横轴是持有期，纵轴是回看期，每格对四种跌幅取平均；数值为抄底夏普减买入持有夏普。全网格平均值为{summary['mean_sharpe_gap']:+.3f}。

按持有期汇总，21日持有规则的平均夏普为{grid.loc[grid.hold_days == 21, 'sharpe'].mean():.3f}，1260日持有规则为{grid.loc[grid.hold_days == 1260, 'sharpe'].mean():.3f}；对应平均股票仓位从{grid.loc[grid.hold_days == 21, 'exposure'].mean():.1%}上升到{grid.loc[grid.hold_days == 1260, 'exposure'].mean():.1%}。延长持有期后，规则逐渐增加股票暴露，夏普也更接近买入持有。每格的细分参数和平均仓位可在grid_results.csv核对。全网格净开仓次数介于{int(grid.entries.min())}和{int(grid.entries.max())}次之间，评价具体参数时可将这些信息一起读。

## 4. 对实现方式敏感吗

{impl_table}

三种实现下，整组规则平均夏普均低于买入持有。落后幅度和比例随实施方式发生变化，说明在复现中记录重复信号处理和成交时点有实际意义。主复现使用“信号收盘价成交近似、持仓期间忽略新信号”；同时保留下一交易日收盘成交作为执行延迟敏感性。

## 5. 费用与统计检验

{cost_table}

表中“平均年化收益”和“平均回撤”为196条规则各自指标的算术平均。买入持有参照采用毛收益。单边成本是每次买入或卖出按资金比例扣除的简化费用，成本提高对高换手规则影响更明显。

![Alpha与成本敏感性](figures/04_alpha_cost.png)

图5：左侧是月频HAC(3)的alpha t值分布，右侧是四档单边成本下的平均及中位夏普。主实验平均年化alpha为{summary['mean_alpha_ann']:.3%}，个体显著结果经过196项Holm校正后，达到5%水平的规则数为{summary['holm_significant_count']}。

回归截距回答在这一样本中、控制市场超额收益后，策略还有多少剩余收益。HAC和多重检验把时间依赖与多参数比较纳入同一评价流程，结果解释据此保持在当前样本和规则范围内。

## 6. 不同历史阶段与发布后观察

{period_table}

历史分段沿用原有仓位。平均夏普和落后比例统一在夏普有效的规则中计算，表中列出有效规则数量及全程现金组数；平均股票仓位覆盖全部196组。

发布后窗口是**2025-12-02至2026-09-30**，共{int(periods.loc[periods.period == 'after_publication', 'days'].iloc[0])}个交易日。该窗口有{int(published.sharpe.notna().sum())}组规则具有有效夏普，另有{int((published.exposure == 0).sum())}组全程持有现金。平均夏普与落后比例使用前者作为分母。窗口中的规则表现优劣以当前观察为范围，后续可按相同设定持续更新。

## 7. 看一笔交易，核对时间线

示例规则：回看21个交易日、跌幅达到10%、持有63个股票收益区间。前五条交易如下，完整清单在example_trades.csv。

{trade_preview}

主复现按信号当日收盘价建立仓位，从下一交易日记录股票收益；这是原图对照采用的理想化收盘成交近似。逐日仓位文件可以核对每个买入、持有和退出区间。`holding_return_days`统计实际纳入的股票日收益数量；样本结束仍有仓位时，交易清单用open_at_sample_end标记。若到期时接续一个新持有周期，两个周期在同一次收盘衔接；股票净仓位保持不变，费用按净仓位变化计算。报告生成时，用逐个触发周期独立重建示例仓位，并与主程序输出逐日核对。

## 8. 已完成的复现与接下来可研究的问题

已完成公开数据下载和快照留存、196组主实验、收益风险评价、OLS与HAC回归、Holm校正、执行与成本敏感性、历史分段、发布后观察，以及{len(summary['self_checks'])}类针对时间、月份复利与资金逻辑的自检。

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



