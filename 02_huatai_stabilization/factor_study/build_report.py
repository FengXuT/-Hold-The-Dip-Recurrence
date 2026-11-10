"""生成因子研究的修正、泄漏审查与历史时间留出报告。"""
from io import StringIO
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from backtest import ROOT,OUT,NAMES

plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
    'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'font.size':11})
metrics=json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
audit=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
config=json.loads((OUT/'frozen_training_config.json').read_text(encoding='utf-8'))
nav=pd.read_csv(OUT/'nav.csv',index_col=0,parse_dates=True)
holdout=pd.read_csv(OUT/'holdout.csv')
holdout_nav=pd.read_csv(OUT/'holdout_nav.csv',index_col=0,parse_dates=True)
walk=pd.read_csv(OUT/'walk_forward.csv')
tail=pd.read_csv(OUT/'post_report_holdout.csv')
rejects=pd.read_csv(OUT/'post_report_rejected_orders.csv')
grid=pd.read_csv(OUT/'training_grid.csv')
sensitivity=pd.read_csv(OUT/'sensitivity.csv')
cross=pd.read_csv(OUT/'cross_index.csv')
stress=pd.read_csv(OUT/'cost_stress.csv')
ledgers={m:pd.read_csv(OUT/f'{m}_trades.csv',parse_dates=['confirm','entry','exit']) for m in NAMES}


def table(rows,columns):
    return '<div class="table">'+pd.DataFrame(rows,columns=columns).to_html(index=False,border=0)+'</div>'


fig,axes=plt.subplots(2,1,figsize=(11,6.2),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,1]})
for method,color in [('baseline','#8b959d'),('combined','#1b669a')]:
    axes[0].plot(nav.index,nav[method],color=color,label=NAMES[method],linewidth=1.9)
    axes[1].plot(nav.index,(nav[method]/nav[method].cummax().clip(lower=1)-1)*100,color=color,linewidth=1.4)
axes[0].legend(frameon=False);axes[0].set_ylabel('净值');axes[1].set_ylabel('回撤 / %')
axes[0].set_title('原研究区间：修正执行后的固定规则',loc='left')
buffer=StringIO();fig.savefig(buffer,format='svg');main_svg=buffer.getvalue();plt.close(fig)
fig,ax=plt.subplots(figsize=(11,3.7),layout='constrained')
labels={'baseline':'同阈值基准','fixed_combined':'固定因子1.0/3日','training_selected':'训练选择0.75/3日'}
for method in labels:
    ax.plot(holdout_nav.index,holdout_nav[method],label=labels[method],linewidth=1.7)
ax.legend(frameon=False);ax.set_ylabel('净值');ax.set_title('2024-01-02至2026-08-31：训练后封存参数的留出段',loc='left')
buffer=StringIO();fig.savefig(buffer,format='svg');holdout_svg=buffer.getvalue();plt.close(fig)

main_rows=[]
for method in NAMES:
    m=metrics[method]
    main_rows.append([NAMES[method],m['trades'],f'{m["annual_return"]*100:.2f}%',f'{m["max_drawdown"]*100:.2f}%',f'{m["sharpe"]:.3f}',f'{m["mean_exposure"]*100:.2f}%'])
main_table=table(main_rows,['策略','笔数','年化收益','最大回撤','夏普','平均名义敞口'])
old=audit['before_metrics'];new=metrics['combined']
before_table=table([[title,f'{old[key]*scale:.{decimals}f}'+unit,f'{new[key]*scale:.{decimals}f}'+unit]
    for title,key,scale,decimals,unit in [('期末净值','final_nav',1,5,''),('年化收益','annual_return',100,3,'%'),('最大回撤','max_drawdown',100,3,'%'),('夏普','sharpe',1,3,'')]],['组合指标','上次提交','修正后'])
checks_table=table([[z['cutoff'],z['orders'],z['open_trades'],f'{z["nav"]:.6f}'] for z in audit['prefix_snapshots']],['截断日','已形成订单数','未平仓笔数','截止日净值'])
holdout_table=table([[labels[z.method],int(z.trades),f'{z.total_return*100:.2f}%',f'{z.annual_return*100:.2f}%',f'{z.max_drawdown*100:.2f}%',f'{z.sharpe:.3f}'] for z in holdout.itertuples()],['策略','笔数','区间收益','年化收益','最大回撤','夏普'])
training_table=table([[z.ratio_limit,int(z.floor_days),int(z.closed_trades),f'{z.sharpe:.3f}'] for z in grid.itertuples()],['压力比上限','守底/日','训练已平仓笔数','训练夏普'])
walk_rows=[]
for year in sorted(walk.year.unique()):
    b=walk.loc[(walk.year==year)&(walk.method=='baseline')].iloc[0]
    c=walk.loc[(walk.year==year)&(walk.method=='selected')].iloc[0]
    walk_rows.append([str(year)+('（截至8月31日）' if year==2026 else ''),c.train_end,f'{c.threshold*100:.3f}%',f'{c.ratio_limit}/{int(c.floor_days)}',int(c.training_closed_trades),f'{b.total_return*100:.2f}%',f'{c.total_return*100:.2f}%'])
walk_table=table(walk_rows,['测试段','训练截止','训练大跌阈值','所选P/A','训练笔数','基准区间收益','所选组合区间收益'])
tail_names={'baseline':'严格上限：基准','combined':'严格上限：组合','market_target_budget':'目标预算：组合市价对照'}
tail_table=table([[tail_names[z.method],int(z.closed_trades),int(z.open_trades),int(z.rejected_orders),f'{z.total_return*100:.2f}%',f'{z.max_entry_margin*100:.3f}%'] for z in tail.itertuples()],['执行方式','已平仓','未平仓','开盘撤单','区间收益','实际最高保证金/本金'])
reject_table=table([[z.entry,z.contract,z.quantity,f'{z.opening_price:.1f}',f'{z.limit_price:.1f}'] for z in rejects.itertuples()],['日期','合约','手数','实际开盘价','已提交限价'])

old_ledger=ledgers['baseline'].copy();new_ledger=ledgers['combined'].copy()
old_ledger['month']=old_ledger.entry.dt.to_period('M');new_ledger['month']=new_ledger.entry.dt.to_period('M')
attribution=pd.DataFrame({'baseline':old_ledger.groupby('month').profit.sum(),'combined':new_ledger.groupby('month').profit.sum()}).fillna(0)
attribution['difference']=attribution.combined-attribution.baseline
attribution=attribution.loc[attribution.difference.abs()>.01]
attribution.to_csv(OUT/'trade_attribution.csv',index_label='entry_month')
changes_table=table([[str(d),f'{r.baseline/1e4:.2f}',f'{r.combined/1e4:.2f}',f'{r.difference/1e4:+.2f}'] for d,r in attribution.iterrows()],['开仓月份','基准净利润/万元','组合净利润/万元','差额/万元'])
sensitivity_table=table([[z.ratio_limit,int(z.floor_days),int(z.trades),f'{z.annual_return*100:.2f}%',f'{z.sharpe:.3f}'] for z in sensitivity.itertuples()],['压力比上限','守底/日','笔数','原区间年化','夏普'])
cross_names={'IM':'中证1000','IC':'中证500','IF':'沪深300','IH':'上证50'}
cross_table=table([[cross_names[z.prefix],NAMES[z.method],f'{z.annual_return*100:.2f}%',f'{z.max_drawdown*100:.2f}%',f'{z.sharpe:.3f}'] for z in cross.loc[cross.method.isin(['baseline','combined'])].itertuples()],['指数','策略','年化收益','最大回撤','夏普'])
trades_table=table([[z.confirm.strftime('%Y-%m-%d'),z.entry.strftime('%Y-%m-%d'),z.exit.strftime('%Y-%m-%d'),z.contract,z.quantity,f'{z.sizing_price:.1f}',f'{z.profit/1e4:.2f}'] for z in ledgers['combined'].itertuples()],['确认日','开仓日','平仓日','合约','手数','确认时参考价','净利润/万元'])
html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,">
<title>双因子策略：时点修正与时间留出检验</title><style>
body{{margin:0;background:#f2f5f6;color:#273b49;font:17px/1.85 "Microsoft YaHei","Noto Sans CJK SC",sans-serif}}main{{max-width:1040px;margin:36px auto;padding:48px 60px;background:white;border-top:7px solid #1a5f94}}h1{{font-size:34px;line-height:1.4;margin:16px 0}}h2{{font-size:25px;margin:44px 0 14px;color:#174763}}h3{{font-size:19px;margin-top:28px}}p{{margin:14px 0}}a{{color:#226e9c}}.tag{{font-size:14px;color:#687c88;letter-spacing:2px}}.lead{{font-size:19px}}.callout{{padding:20px 24px;background:#edf5fa;border-left:4px solid #4c92b8}}.table{{overflow-x:auto;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}}th,td{{padding:10px 12px;border-bottom:1px solid #dfe7ec;text-align:right}}th{{background:#edf3f6}}th:first-child,td:first-child{{text-align:left}}svg{{width:100%;height:auto}}.formula,pre{{padding:14px 18px;background:#f1f5f7;overflow-x:auto;font-family:Consolas,monospace;line-height:1.7}}li{{margin:8px 0}}footer{{margin-top:42px;border-top:1px solid #dfe7ec;padding-top:18px;font-size:14px}}@media(max-width:700px){{main{{margin:0;padding:24px 20px}}h1{{font-size:27px}}h2{{font-size:23px}}body{{font-size:16px}}}}@media print{{main{{margin:0;padding:0;border:0}}body{{background:white}}h2{{break-after:avoid}}table,svg{{break-inside:avoid}}}}
</style><main><div class="tag">自主因子研究 · 时点审查 · 历史时间留出</div>
<h1>双因子策略：<br>修正信息时点，再检验样本外表现</h1>
<p class="lead">本研究在大跌后企稳的框架上增加“下行压力衰减”和“连续三日守底”。本次更新修正区间起止边界与定仓信息时点，并补齐训练期封存、时间留出和成交方式对照。</p>
<div class="callout"><strong>核心发现：</strong>修正后，原研究区间组合年化收益为{new['annual_return']*100:.2f}%，最大回撤{new['max_drawdown']*100:.2f}%，夏普{new['sharpe']:.3f}。但仅用训练期挑出的较严格参数，在后续时间留出段年化为12.87%，低于同阈值基准18.36%。少量训练交易与参数选择不稳定，是当前研究的主要问题。</div>
<p>研究起点是<a href="https://www.htfc.com/wz_upload/png_upload/20260909/1788943656749408f5b.pdf">华泰期货研究院2026年9月1日正式研报</a>。上级<a href="../REPORT.html">原研报复现报告</a>保留其历史口径；本报告使用修正后的可下单规则开展因子检验。</p>
<h2>1. 策略研究什么</h2>
<p>原条件在中证1000五日收益低于−4.92%后等待七日，若期末收盘恢复至触发日之上，则考虑入场。新增因子分别度量下跌方向的波动与最近低点出现的时间。</p>
<div class="formula">D(n,u)=sqrt[mean(min(rᵢ,0)²)]，窗口包含全部n个交易日<br>P=D(3,确认日)/D(20,触发日前一日)，要求P≤1<br>A=确认日距“触发日至确认日最后一次最低价”的交易日数，要求A≥3</div>
<p>正收益在下行波动中贡献零；最低价并列时取最后一次。两项通过后，收盘形成次日订单。固定主实验沿用七日等待、20个交易间隔持有和研报阈值−4.92%。时间留出实验另用训练数据估计大跌阈值。</p>
<h2>2. 如何从数据发现代码问题</h2>
<h3>问题一：尚未平仓的合格信号被删除</h3>
<p>把行情截到2024年2月19日，触发日1月31日的信号已经完成确认：修复幅度5.11%，压力比为0，低点距确认日四个交易日。完整数据下旧代码保留该信号；相同历史价格截断后，信号却消失。</p>
<p>原因是候选生成提前检查“20日退出日期必须位于样本内”。这笔交易计划3月19日退出，旧条件因未来退出日期尚未出现而删除2月19日的决定。继续把终点移到2月20日，旧逻辑仍漏掉持仓；修正后当天产生约{audit['terminal_position_example']['first_day_net_profit']/1e4:.2f}万元净损益，并保留未平仓状态。</p>
<p>修正方式是按日处理“开盘退出、开盘成交、收盘估值、收盘形成新订单”。订单生成只依赖确认时已知信息；期末保留待执行订单和未平仓持仓，最后价格用于估值。此次原区间截止日恰好空仓，因此边界修正主要影响截断检验与新增区间。</p>
<h3>问题二：手数用了成交时才知道的开盘价</h3>
<p>2026年4月1日确认时，IM2606收盘为7518.6点；次日实际开盘为7460.0点。旧公式用次日开盘价算出67手，当日已知价格只能算66手。修改次日开盘价，会改变旧公式中的订单手数，这直接暴露了定仓信息时点问题。</p>
<p>现按确认日合约收盘价计算整数手数，并同步设置保证金上限对应的买入限价。开盘价高于限价则撤销该次开盘订单，收盘继续评估新信号；开盘价满足限价则模拟成交。这样，手数与限价都在成交前确定，初始保证金保持在1200万元以内。</p>
<div class="formula">手数=floor[1200万元/(确认日合约收盘价×合约乘数×保证金率)]<br>买入限价=floor[1200万元/(手数×合约乘数×保证金率)/0.2]×0.2</div>
<p>手续费仍为双边各万分之0.3。未来价格扰动检验中，把4月2日开盘价提高5%，4月1日提交的手数仍为66手；订单能否成交则由次日开盘与限价决定。</p>
<h3>问题三：原有核验只比较完成交易</h3>
<p>旧检查先删除截止日之后平仓的交易，再比较剩余交易。这使两边同时跳过边界持仓，错误被隐藏。本次改为比较截止时已经形成的全部订单、当时持仓和每日净值。</p>
{checks_table}
<p>五个截断点均保持订单前缀与净值一致。另把2024年2月19日之后的指数价格乘1.7、期货价格乘1.9，截止当日的订单与净值保持一致；把训练截止日之后价格整体扰动，训练阈值、九组训练结果与所选参数也保持一致。</p>
<h3>补充复查：区间首日漏掉前一收盘的订单</h3>
<p>单独从2024年2月20日开盘开始回测时，旧初始化只从当天收盘检查信号，漏掉2月19日已确认的103手订单，首日损益错误地变为零。相同订单在完整回放中，2月20日净损益为{audit['window_initialization']['actual_first_day_profit']/1e4:.2f}万元。</p>
<p>现在从区间首日前一个收盘初始化决策，账户仍以空仓和初始资金启动。成交与损益从区间首日计算。逐一把原区间13笔订单的开仓日作为独立窗口起点，订单和首日损益均与完整回放一致；另用9月3日验证首日跳空超过限价时正确撤单。</p>
<p>无交易区间的订单、持仓与撤单CSV现在保留完整表头，三类文件均通过写出再读入检查。单日窗口保留收益和回撤，波动率与夏普因观测不足而留空；2023年1月4日的单日亏损窗口也验证了回撤从初始净值1开始计算。此次复查后，主区间、时间留出与新增区间的汇总数值保持一致。</p>
<h2>3. 修正后的原区间结果</h2>
<p>区间为2022年7月22日至2026年8月31日，共997个交易日。各组采用相同执行规则、1亿元初始资金与1200万元保证金上限；IM历史保证金率沿用原复现中的15%→12%变化。</p>
{main_svg}{main_table}
<h3>与上次已提交报告比较</h3>
{before_table}
<p>修正前数字来自<a href="https://github.com/FengXuT/-Hold-The-Dip-Recurrence/commit/683f5ea">上次提交683f5ea</a>。收益差异主要来自确认日价格定仓后的少量手数变化。原区间双因子组合13笔交易均满足开盘限价，主结果中撤单数为零。</p>
<h3>改善集中在哪些交易</h3>
{changes_table}
<p>收益差异仍主要来自2023年5月跳过亏损交易和2025年1月延后入场；2024年11月与2025年4月分别损失了一部分收益。这种集中程度，使训练期样本量与后续时间检验尤其重要。</p>
<h2>4. 把训练与测试按时间分开</h2>
<p>阈值−4.92%是研报给定的复现常数，主历史对照沿用它。要检验训练后可执行的策略，本次单独封存训练配置：训练区间2022年7月22日至2023年11月30日，用其中331个五日收益观测估计5%分位数，得到{config['threshold']*100:.3f}%；再在九组P/A组合中按训练夏普选参，至少要求三笔已平仓交易。</p>
<p>训练选择P≤{config['ratio_limit']}、A≥{config['floor_days']}。随后保留21个交易日的间隔，测试从2024年1月2日开始，截止2026年8月31日；测试从空仓启动，所有参数在进入该段前封存。</p>
{training_table}
<p>训练选中组只有3笔已平仓交易，守底2、3、4日的训练结果完全相同。这里能观察到参数识别不足：训练表现由极少数交易决定，守底窗口在这些交易中缺乏区分度。</p>
{holdout_svg}{holdout_table}
<p>按训练期选出的严格压力比0.75，在测试段年化收益12.87%、最大回撤9.27%、夏普0.920；同阈值基准分别为18.36%、7.31%、1.265。固定因子1.0/3日的对照年化为19.04%，但主因子设计曾参考原研究区间，后续参数选择须继续服从训练期封存规则。</p>
<h3>逐年重建训练与留出流程</h3>
<p>分别以2023年11月底、2024年11月底和2025年11月底为训练截止，在后续年度独立从空仓开始检验。表内使用区间收益，2026年只截至8月31日。</p>
{walk_table}
<p>2025年选择1.0/3日时优于基准；2024年与2026年选择0.75/3日时收益较弱。2026年较严阈值只保留一笔交易，漏掉4月的主要反弹。这个失败案例是对过拟合风险的直接数据证据，测试结果完整保留。</p>
<p>本项目的因子是在查看原区间表现后提出的，因此2024—2026年的检验定位为重建训练流程的历史时间留出。完整前向观察从2026年11月10日规则冻结后开始积累；今天的结果用于检查时点、训练稳定性和执行约束。</p>
<h2>5. 原研究区间后的新增数据回放</h2>
<p>2026年9月1日至11月9日为此前因子主回测之外的区间。本次加入这些数据后，也用它检查了保证金约束与期末持仓处理，因此本节定位为修正后的新增区间回放。</p>
{tail_table}
<p>严格保证金上限下，基准与组合均只有一笔未平仓交易，区间收益均为0.90%；两次开盘订单因超过预先确定的限价而撤销。若采用确认日定仓、次日按开盘市价成交，并把保证金作为目标值，则区间收益为−4.03%，最高开仓保证金为本金的12.017%。</p>
{reject_table}
<p>这组差异主要由成交约束产生，基准与组合的因子差异在该段尚未形成额外交易收益。报告同时展示两种方式，以便后续把撮合、跳空与信号贡献分别核对。期末未平仓按11月9日收盘估值，计划退出日期和浮动损益保留在交易记录中。</p>
<h2>6. 参数附近与跨指数表现</h2>
<p>下表使用修正后的执行规则。参数邻域只用于展示敏感性，后续测试结果保持原样。</p>
{sensitivity_table}{cross_table}
<p>下行压力收紧至0.75后，一部分反弹收益被过滤；跨指数方面，中证500与沪深300的组合仍有改善，上证50的组合低于基准。固定两因子规则的效果依赖市场与行情阶段。</p>
<h2>7. 原区间组合全部交易</h2>
{trades_table}
<p>每笔记录包含确认日参考价、订单手数、限价、实际成交价、计划退出与实际状态。每日损益之和与已实现加未实现交易损益逐笔对账。</p>
<h2>8. 结论与复算</h2>
<p>本次审查修复了期末信号删除、期初订单初始化、成交价反推手数，以及只比较完成交易的核验缺口。原区间收益优势小幅保留；训练期选参在后续样本中表现不稳，训练交易数量与收益集中度是更重要的研究约束。接下来保留固定候选与训练封存规则，按新增行情累积前向记录。</p>
<pre>python fetch_evaluation_data.py
python backtest.py
python audit_validation.py
python build_report.py</pre>
<p>首次准备环境与原始数据时，先按上级README安装依赖并运行download_data.py。独立评估快照截至2026年11月9日，原研报复现快照保留原口径。绩效以250个年度交易日统计；最大回撤从初始净值1开始计算，期末持仓按收盘估值。</p>
<footer>数据：中证指数官方日线、国证指数交易日期、新浪实际期货合约行情。选约按中金所<a href="https://www.cffex.com.cn/u/cms/www/202207/18190308vxpv.pdf">合约规则</a>执行；未来日历仅补充至2027年1月8日，元旦安排依据<a href="https://www.gov.cn/zhengce/content/202611/content_7085101.htm">2026年11月5日公布的安排</a>。审查与报告更新日期：2026-11-10。</footer></main></html>'''
assert 'private_recent_year' not in html and '2025-11-10' not in html
(ROOT/'REPORT.html').write_text('\n'.join(line.rstrip() for line in html.splitlines())+'\n',encoding='utf-8')
print('Updated audit and holdout report generated; separate recent-year output excluded.')
