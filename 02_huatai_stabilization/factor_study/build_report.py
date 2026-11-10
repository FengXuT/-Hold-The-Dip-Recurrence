"""从因子实验输出生成独立的HTML研究报告。"""
from io import StringIO
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from backtest import ROOT, OUT, NAMES

plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'], 'axes.unicode_minus':False,
                     'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
                     'font.size':11,'axes.grid':True,'grid.alpha':.18})
metrics=json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
frozen=json.loads((OUT/'fixed_event_filter.json').read_text(encoding='utf-8'))
january=json.loads((OUT/'january_decomposition.json').read_text(encoding='utf-8'))
nav=pd.read_csv(OUT/'nav.csv',index_col=0,parse_dates=True)
annual=pd.read_csv(OUT/'annual.csv')
cross=pd.read_csv(OUT/'cross_index.csv')
sensitivity=pd.read_csv(OUT/'sensitivity.csv')
stress=pd.read_csv(OUT/'cost_stress.csv')
matched=pd.read_csv(OUT/'matched_exposure.csv')
cohort=pd.read_csv(OUT/'baseline_event_diagnostics.csv')
ledgers={method:pd.read_csv(OUT/f'{method}_trades.csv',parse_dates=['entry','exit']) for method in NAMES}


def table(rows, columns):
    return '<div class="table">'+pd.DataFrame(rows,columns=columns).to_html(index=False,border=0)+'</div>'


fig,axes=plt.subplots(2,1,figsize=(11,6.5),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,1]})
colors={'baseline':'#8b959d','pressure':'#b78231','floor':'#3b927e','combined':'#1a5f94'}
for method,name in NAMES.items():
    width=2.1 if method in ['baseline','combined'] else 1.1
    axes[0].plot(nav.index,nav[method],label=name,color=colors[method],linewidth=width)
    axes[1].plot(nav.index,(nav[method]/nav[method].cummax()-1)*100,color=colors[method],linewidth=width)
axes[0].set_ylabel('净值');axes[0].legend(loc='upper left',frameon=False,ncol=2)
axes[1].set_ylabel('回撤 / %')
axes[0].set_title('相同单笔保证金预算：原策略与因子扩展',loc='left')
buffer=StringIO();fig.savefig(buffer,format='svg');nav_svg=buffer.getvalue();plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for ax,key,title,factor in zip(axes,['annual_return','sharpe'],['年化收益 / %','夏普比率'],[100,1]):
    matrix=sensitivity.pivot(index='ratio_limit',columns='floor_days',values=key).sort_index()*factor
    im=ax.imshow(matrix.to_numpy(),cmap='YlGnBu',aspect='auto')
    ax.grid(False);ax.set_xticks(range(3),['2日','3日','4日']);ax.set_yticks(range(3),['0.75','1.00','1.25'])
    ax.set_xlabel('连续守底要求');ax.set_ylabel('下行压力比上限');ax.set_title(title,loc='left')
    for i in range(3):
        for j in range(3):
            value=matrix.iloc[i,j]
            ax.text(j,i,f'{value:.2f}',ha='center',va='center',color='white' if value>matrix.to_numpy().mean() else '#193044')
    ax.add_patch(Rectangle((.5,.5),1,1,fill=False,edgecolor='#d95329',linewidth=3))
buffer=StringIO();fig.savefig(buffer,format='svg');sensitivity_svg=buffer.getvalue();plt.close(fig)

rows=[]
for method,name in NAMES.items():
    m=metrics[method]
    rows.append([name,m['trades'],f'{m["annual_return"]*100:.2f}%',f'{m["max_drawdown"]*100:.2f}%',
                 f'{m["sharpe"]:.3f}',f'{m["volatility"]*100:.2f}%',f'{m["mean_exposure"]*100:.2f}%'])
main_table=table(rows,['策略','笔数','年化收益','最大回撤','夏普','年化波动','平均名义敞口'])

rows=[]
for method in ['pressure','floor','combined']:
    a=metrics[method];b=matched.loc[matched.method==method].iloc[0]
    rows.append([NAMES[method],f'{a["mean_exposure"]*100:.2f}%',f'{a["annual_return"]*100:.2f}%',
                 f'{b.annual_return*100:.2f}%',f'{a["sharpe"]:.3f}',f'{b.sharpe:.3f}'])
matched_table=table(rows,['因子策略','双方平均敞口','因子年化','等敞口基准年化','因子夏普','等敞口基准夏普'])

# 以实际开仓月份归集不同交易，显示信号被跳过或延后的资金收益差异。
old=ledgers['baseline'].copy();new=ledgers['combined'].copy()
old['month']=old.entry.dt.to_period('M');new['month']=new.entry.dt.to_period('M')
old_profit=old.groupby('month').profit.sum();new_profit=new.groupby('month').profit.sum()
changes=pd.DataFrame({'baseline':old_profit,'combined':new_profit}).fillna(0)
changes['difference']=changes.combined-changes.baseline
changes=changes.loc[changes.difference.abs()>.01]
rows=[]
for month,z in changes.iterrows():
    a=old.loc[old.month==month];b=new.loc[new.month==month]
    before=' / '.join(a.entry.dt.strftime('%m-%d')+' '+a.contract) or '空仓'
    after=' / '.join(b.entry.dt.strftime('%m-%d')+' '+b.contract) or '空仓'
    rows.append([str(month),before,after,f'{z.baseline/1e4:.2f}',f'{z.combined/1e4:.2f}',f'{z.difference/1e4:+.2f}'])
changes_table=table(rows,['事件月份','基准开仓','组合开仓','基准净收益/万元','组合净收益/万元','差额/万元'])
changes.to_csv(OUT/'trade_attribution.csv',index_label='entry_month')

rows=[]
for year in sorted(annual.year.unique()):
    a=annual.loc[(annual.year==year)&(annual.method=='baseline')].iloc[0]
    b=annual.loc[(annual.year==year)&(annual.method=='combined')].iloc[0]
    rows.append([str(year)+('（部分年度）' if year in [2022,2026] else ''),f'{a.profit/1e4:.2f}',f'{b.profit/1e4:.2f}',f'{(b.profit-a.profit)/1e4:+.2f}'])
annual_table=table(rows,['年度','基准净利润/万元','组合净利润/万元','差额/万元'])

rows=[]
for prefix,title in [('IM','中证1000'),('IC','中证500'),('IF','沪深300'),('IH','上证50')]:
    for method in ['baseline','pressure','combined']:
        m=cross.loc[(cross.prefix==prefix)&(cross.method==method)].iloc[0]
        rows.append([title,NAMES[method],int(m.trades),f'{m.annual_return*100:.2f}%',f'{m.max_drawdown*100:.2f}%',f'{m.sharpe:.3f}'])
cross_table=table(rows,['指数','策略','笔数','年化收益','最大回撤','夏普'])
rows=[]
for method in NAMES:
    m=stress.loc[stress.method==method].iloc[0]
    rows.append([NAMES[method],f'{m.annual_return*100:.2f}%',f'{m.max_drawdown*100:.2f}%',f'{m.sharpe:.3f}'])
cost_table=table(rows,['策略','成本压力下年化收益','最大回撤','夏普'])
rows=[]
for z in cohort.itertuples():
    rows.append([z.entry,f'{z.downside_ratio:.3f}',z.days_since_low,'通过' if z.combined_pass else '等待后续信号',f'{z.profit/1e4:.2f}'])
cohort_table=table(rows,['原策略开仓日','下行压力比','距最后低点/日','本次组合判断','原交易净收益/万元'])
rows=[]
for z in ledgers['combined'].itertuples():
    rows.append([z.entry.strftime('%Y-%m-%d'),z.exit.strftime('%Y-%m-%d'),z.contract,z.quantity,
                 f'{z.downside_ratio:.3f}',z.days_since_low,f'{z.profit/1e4:.2f}'])
trades_table=table(rows,['开仓日','平仓日','合约','手数','下行压力比','守底天数','净收益/万元'])

html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>从企稳到卖压衰竭：双因子策略回测</title><link rel="icon" href="data:,">
<style>
body{{margin:0;background:#f2f5f6;color:#273b49;font:17px/1.85 "Microsoft YaHei","Noto Sans CJK SC",sans-serif}}main{{max-width:1040px;margin:36px auto;padding:48px 60px;background:white;border-top:7px solid #1a5f94}}h1{{font-size:34px;line-height:1.4;margin:16px 0}}h2{{font-size:25px;margin:44px 0 14px;color:#174763}}h3{{font-size:19px;margin-top:28px}}p{{margin:14px 0}}a{{color:#226e9c}}.tag{{font-size:14px;color:#687c88;letter-spacing:2px}}.lead{{font-size:19px}}.callout{{padding:20px 24px;background:#edf5fa;border-left:4px solid #4c92b8}}.small,figcaption{{font-size:14px;color:#607382}}.table{{overflow-x:auto;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}}th,td{{padding:10px 12px;border-bottom:1px solid #dfe7ec;text-align:right}}th{{background:#edf3f6}}th:first-child,td:first-child{{text-align:left}}svg{{width:100%;height:auto}}.formula,pre{{padding:14px 18px;background:#f1f5f7;overflow-x:auto;font-family:Consolas,monospace;line-height:1.7}}li{{margin:8px 0}}footer{{margin-top:42px;border-top:1px solid #dfe7ec;padding-top:18px;font-size:14px}}@media(max-width:700px){{main{{margin:0;padding:24px 20px}}h1{{font-size:27px}}h2{{font-size:23px}}body{{font-size:16px}}}}@media print{{main{{margin:0;padding:0;border:0}}body{{background:white}}h2{{break-after:avoid}}table,svg{{break-inside:avoid}}}}
</style><main>
<div class="tag">自主因子研究 · 实际股指期货 · 固定规则回测</div>
<h1>从企稳到卖压衰竭：<br>下行压力衰减与新低停止的双因子实验</h1>
<p class="lead">在华泰期货“连续大跌后等待七日企稳”的框架上，本研究增加两个入场条件：近期下行压力已经降低，以及价格低点已经稳定三日。研究重点是观察这些条件能否改善交易时机，同时保留大反弹阶段的收益。</p>
<div class="callout"><strong>主要结果：</strong>2022年7月22日至2026年8月31日，双因子组合完成13笔交易，年化收益15.58%，夏普1.466，最大回撤6.56%；原企稳策略分别为15笔、14.98%、1.332和6.62%。平均名义敞口由29.10%降至25.11%。净利润增加363.63万元，改善主要来自2023年一次亏损交易的跳过与2025年1月入场时机的调整。</div>
<p>基准来自<a href="https://www.htfc.com/wz_upload/png_upload/20260909/1788943656749408f5b.pdf">华泰期货研究院2026年9月1日正式研报</a>，其完整复现见<a href="../REPORT.html">原研报复现报告</a>。本报告讨论的是在既有复现样本上的自主因子扩展，所有主参数在本次首次回测前固定，后续网格用于观察邻域变化。</p>
<h2>1. 为什么选择这两个因子</h2>
<p>原企稳条件只比较等待期末与大跌触发日的收盘价。两条价格路径即使终点相同，过程也可能不同：一条已经停止探底，另一条刚刚创出新低后快速拉回。观察下行波动与最低价出现的时间，可以补充收盘价比较的信息。</p>
<p>前一轮实验曾用总波动下降来缩减仓位，夏普降至1.17。2024年2月的强反弹使总波动升高，这笔交易却贡献了1441.36万元。因此，本次专门度量下跌方向的波动，并把因子用于入场筛选。</p>
<h2>2. 因子与交易规则</h2>
<h3>因子一：下行压力衰减</h3>
<p>日收益记为r，先把正收益置零，再计算平方收益的均值平方根。这样，上涨产生的波动会保留为零，下跌才贡献压力。</p>
<div class="formula">D(n,u) = sqrt[ mean(min(rᵢ,0)²), i=u−n+1,…,u ]<br>下行压力比 P = D(3,确认日) / D(20,触发日前一日)<br>通过条件：P ≤ 1</div>
<p>分母使用大跌触发日之前20个交易日，分子使用确认日及此前两日。P≤1表示近期下跌压力已经回到触发前的水平或更低。窗口内所有交易日都进入平均值，上涨日贡献零。</p>
<h3>因子二：新低停止</h3>
<div class="formula">A = 确认日 − [触发日至确认日期间最后一次最低价的日期]<br>通过条件：A ≥ 3个交易日</div>
<p>价格采用每日最低价；若最低价重复出现，取最后一次。A≥3意味着最低点之后已有三个交易日守住这一低点。</p>
<h3>从条件到可执行策略</h3>
<ol><li>中证1000五日收益低于−4.92%，记录触发日t；到t+7收盘检查原企稳条件Cₜ₊₇≥Cₜ。</li>
<li>原条件通过后，分别测试压力比P≤1、守底A≥3，以及两者同时通过的组合。</li>
<li>条件通过则下一交易日开盘入场；未通过时，后续交易日继续评估各自七日前形成的新信号。</li>
<li>按原规则持有20个交易间隔后开盘退出；同一时刻持有一笔多头，平仓当日收盘可重新确认信号。</li></ol>
<p>主实验统一采用固定七日等待期与20日持有期。初始资金1亿元，单笔保证金预算1200万元；使用剩余期限严格超过40个交易日的最近到期实际合约，整数手数向下取整。IM开仓保证金在2023年5月5日前取15%，之后取12%；双边手续费各万分之0.3，空仓资金收益取零。样本包含能在截止日前完成持有期的交易。</p>
<p>绩效采用250个年度交易日；年化收益为净值几何增长率，波动为日收益样本标准差年化，夏普为（年化收益−2%）/年化波动。沿用这些口径，基准回放逐日净值及全部15笔交易与原复现一致。</p>
<h2>3. 四种策略的完整结果</h2>
{nav_svg}
{main_table}
<p>组合的年化收益提高约0.60个百分点，波动下降约0.49个百分点，最大回撤下降约0.07个百分点。资金利用方面，持仓日数从300日降至260日，平均名义敞口下降3.99个百分点。这里的敞口按每笔开仓名义金额/初始本金计算，在持有期间逐日累计并取全样本平均。</p>
<p>组合保留了2024年2月和2026年4月两笔主要反弹交易。前三笔盈利占组合净利润的49.78%，原策略为47.85%，收益仍集中在少数较强的反弹阶段。</p>
<h3>在相同平均敞口下比较</h3>
<p>为了观察筛选收益，进一步把基准每日净损益等比例缩放，使其全区间平均名义敞口与各因子策略相同。下表是事后等敞口统计对照，缩放包含原有手续费。</p>
{matched_table}
<p>组合平均敞口为25.11%时，等敞口基准年化收益为13.26%，组合为15.58%。两者夏普分别为1.297和1.466。等敞口基准回撤约5.94%，组合为6.56%，因此本次改善主要体现在收益效率，回撤控制仍有继续研究的空间。</p>
<h2>4. 改善具体来自哪些交易</h2>
{changes_table}
<p>2023年5月，原交易的下行压力比为1.416，组合跳过这笔亏损143.17万元的交易。2024年11月，确认日刚出现等待期最低价，组合选择继续等待，随后少获得原交易的32.94万元盈利。</p>
<p>2025年1月，原确认日距最低价仅一天；组合等到新的信号通过后，于1月17日入场，较基准晚两个交易日，净收益从867.29万元升至1191.97万元。最近到期合约也从IM2503变为IM2506。按相同延后日期仍交易IM2503测算，净收益为{january['same_contract_profit']/1e4:.2f}万元；约{january['timing_and_size_gain']/1e4:.2f}万元改善来自交易时点及相应开仓手数变化，约{january['contract_gain']/1e4:.2f}万元来自本例合约选择变化。</p>
<p>2025年4月，原信号的下行压力比为1.085，组合在下一日的新确认通过后入场，这次延后少获得71.28万元。四处变化合计增加363.63万元净利润。</p>
<h3>筛选与继续等待的作用不同</h3>
<p>将原来的15笔交易固定，只删除未通过双因子的交易，可留下{frozen['trades']}笔，年化收益为{frozen['annual_return']*100:.2f}%，净利润为{frozen['total_profit']/1e4:.2f}万元。允许后续新信号继续确认后，组合完成13笔，年化收益15.58%。这说明本次结果的关键是“条件未成熟时继续观察，成熟后参与”，单纯删除交易会错过2025年1月和4月的修复行情。</p>
<h3>按年度查看利润来源</h3>
{annual_table}
<p>年度净利润统一除以相同初始资金口径理解，表中直接展示金额。2022年只覆盖7月22日之后，2026年截止8月31日。组合的利润提升集中在2023年与2025年，2022年和2026年的实际交易与基准相同。</p>
<h2>5. 稳健性：参数、交易成本与其他指数</h2>
<h3>参数附近的结果</h3>
{sensitivity_svg}
<figcaption>横轴为连续守底要求，纵轴为下行压力比上限；橙框标出主实验的1.0与3日。九组组合完整输出在sensitivity.csv中。</figcaption>
<p>压力比上限取1.0或1.25时，九宫格中对应六组的年化收益为14.92%—15.70%，夏普为1.388—1.479。上限收紧至0.75后，年化收益降至11.40%—12.12%，夏普降至1.071—1.158，强过滤会损失一部分后续反弹。主实验保留事先固定的1.0与3日，邻域表完整展示变化。</p>
<h3>增加执行成本</h3>
<p>压力情景把手续费提高至双边各万分之0.6，并在开仓价加1个指数点、平仓价减1个指数点；手数按加入冲击后的价格重新计算。</p>
{cost_table}
<p>成本压力下，组合年化收益为15.50%，夏普1.455；基准分别为14.89%和1.319。相同成本设定下，两者的差异基本保留。</p>
<h3>跨指数检验</h3>
<p>沿用各指数原研报的五日跌幅阈值，其他因子窗口和条件共用。IC、IF、IH分别使用其真实合约、乘数及原复现中的历史保证金率。</p>
{cross_table}
<p>中证500的组合年化收益由9.16%升至10.24%，回撤由9.21%降至8.84%；沪深300也有小幅改善。上证50的组合年化收益降至0.58%，低于基准1.31%。单独的下行压力因子在四个指数上均提高了本样本的年化收益与夏普，守底条件更适合优先在中小盘上继续研究。</p>
<h2>6. 全部信号与交易记录</h2>
<h3>原策略15笔交易的因子诊断</h3>
{cohort_table}
<p>表中判断只针对原交易当日。判为继续等待的事件，后续仍可能出现新的合格信号。2024年2月的压力比为零，表明确认前最近三日的下行平方收益均为零，强反弹因此顺利保留。</p>
<h3>组合实际执行的13笔交易</h3>
{trades_table}
<h2>7. 结论与后续研究</h2>
<p>这次实验支持把“企稳”进一步拆成“收盘恢复、下行压力减弱、低点稳定”三个条件。双因子组合在中证1000样本中，以较低平均敞口获得略高收益和夏普；最大回撤改善较小。逐笔归因显示，收益提升更依赖信号延后后的重新入场，尤其是2025年1月的调整。</p>
<p>后续优先保留下行压力因子作为通用候选，连续守底作为中小盘的附加条件；沿用本次固定参数，按新增时间段累计事件。研究记录同时保留未通过信号、后续重新入场、名义敞口及每笔收益，以继续检验等待带来的收益与机会成本。</p>
<h2>8. 如何复算</h2>
<p>首次准备数据时，在上级目录按README安装依赖并运行download_data.py；之后在本目录执行：</p>
<pre>python backtest.py
python build_report.py</pre>
<p>backtest.py生成每日净值、全部候选、逐笔交易、等敞口对照、年度利润、四指数、参数邻域和成本压力结果。程序检查基准逐日回放、逐笔现金对账，并把行情截断至两个历史日期，核验四种策略的历史因子和已完成交易保持一致。build_report.py从这些输出生成本报告。</p>
<footer>数据：中证指数官方OHLC、国证指数交易日期、新浪逐合约期货日线；历史保证金与基准规则沿用上级复现项目。研究区间：2022-07-22—2026-08-31，997个交易日。报告生成日期：2026-11-10。方法与运行说明见本目录README。</footer>
</main></html>'''
(ROOT/'REPORT.html').write_text('\n'.join(line.rstrip() for line in html.splitlines())+'\n',encoding='utf-8')
print('Factor REPORT.html generated.')
