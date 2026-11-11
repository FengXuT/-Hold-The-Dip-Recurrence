"""用回测结果生成双因子研究报告。"""
from io import StringIO
import json

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from backtest import ROOT, OUT, NAMES

plt.rcParams.update({
    'font.family': ['Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': .18, 'font.size': 11,
})

metrics = json.loads((OUT / 'metrics.json').read_text(encoding='utf-8'))
config = json.loads((OUT / 'frozen_training_config.json').read_text(encoding='utf-8'))
nav = pd.read_csv(OUT / 'nav.csv', index_col=0, parse_dates=True)
holdout = pd.read_csv(OUT / 'holdout.csv').set_index('method')
holdout_nav = pd.read_csv(OUT / 'holdout_nav.csv', index_col=0, parse_dates=True)
walk = pd.read_csv(OUT / 'walk_forward.csv')
tail = pd.read_csv(OUT / 'post_report_holdout.csv').set_index('method')
sensitivity = pd.read_csv(OUT / 'sensitivity.csv')
cross = pd.read_csv(OUT / 'cross_index.csv')
ledgers = {
    method: pd.read_csv(OUT / f'{method}_trades.csv', parse_dates=['confirm', 'entry', 'exit'])
    for method in ['baseline', 'combined']
}


def table(rows, columns):
    content = pd.DataFrame(rows, columns=columns).to_html(index=False, border=0)
    return '<div class="table">' + content + '</div>'


fig, axes = plt.subplots(
    2, 1, figsize=(10, 5.6), sharex=True, layout='constrained',
    gridspec_kw={'height_ratios': [2, 1]},
)
for method, color in [('baseline', '#8b959d'), ('combined', '#1b669a')]:
    curve = nav[method]
    axes[0].plot(curve.index, curve, color=color, label=NAMES[method], linewidth=1.9)
    drawdown = (curve / curve.cummax().clip(lower=1) - 1) * 100
    axes[1].plot(curve.index, drawdown, color=color, linewidth=1.4)
axes[0].legend(frameon=False)
axes[0].set_ylabel('净值')
axes[1].set_ylabel('回撤 / %')
buffer = StringIO()
fig.savefig(buffer, format='svg')
main_svg = buffer.getvalue()
plt.close(fig)

labels = {
    'baseline': '同阈值基准',
    'fixed_combined': '固定组合（1.0，3日）',
    'training_selected': f"训练所选组合（{config['ratio_limit']}，{config['floor_days']}日）",
}
fig, ax = plt.subplots(figsize=(10, 3.5), layout='constrained')
for method, label in labels.items():
    ax.plot(holdout_nav.index, holdout_nav[method], label=label, linewidth=1.7)
ax.legend(frameon=False)
ax.set_ylabel('净值')
buffer = StringIO()
fig.savefig(buffer, format='svg')
holdout_svg = buffer.getvalue()
plt.close(fig)

baseline, combined = metrics['baseline'], metrics['combined']
main_table = table([
    [NAMES[method], row['trades'], f"{row['annual_return']*100:.2f}%",
     f"{row['max_drawdown']*100:.2f}%", f"{row['sharpe']:.3f}", f"{row['mean_exposure']*100:.2f}%"]
    for method, row in metrics.items()
], ['策略', '交易笔数', '年化收益', '最大回撤', '夏普', '平均名义敞口'])

attribution = pd.DataFrame({
    method: ledger.groupby(ledger.entry.dt.to_period('M')).profit.sum()
    for method, ledger in ledgers.items()
}).fillna(0)
attribution['difference'] = attribution.combined - attribution.baseline
attribution = attribution.loc[attribution.difference.abs() > .01]
attribution.to_csv(OUT / 'trade_attribution.csv', index_label='entry_month')
changes_table = table([
    [str(month), f'{row.baseline/1e4:,.2f}', f'{row.combined/1e4:,.2f}', f'{row.difference/1e4:+,.2f}']
    for month, row in attribution.iterrows()
], ['开仓月份', '基准净利润（万元）', '组合净利润（万元）', '差额（万元）'])
net_gain = attribution.difference.sum()
january_gain = attribution.loc[pd.Period('2025-01', freq='M'), 'difference']
april_loss = -attribution.loc[pd.Period('2025-04', freq='M'), 'difference']

holdout_table = table([
    [labels[method], int(row.trades), f'{row.annual_return*100:.2f}%',
     f'{row.max_drawdown*100:.2f}%', f'{row.sharpe:.3f}']
    for method, row in holdout.iterrows()
], ['策略', '交易笔数', '年化收益', '最大回撤', '夏普'])
walk_returns = walk.pivot(index='year', columns='method', values='total_return')
walk_table = table([
    [str(row.year) if row.year < 2026 else '2026（截至8月31日）',
     f'{row.ratio_limit:g} / {int(row.floor_days)}日',
     f'{walk_returns.loc[row.year, "baseline"]*100:.2f}%', f'{row.total_return*100:.2f}%']
    for row in walk.loc[walk.method == 'selected'].itertuples()
], ['检验年份', '训练所选压力比上限 / 守底天数', '基准区间收益', '组合区间收益'])

sensitivity_table = table([
    [row.ratio_limit, int(row.floor_days), int(row.trades),
     f'{row.annual_return*100:.2f}%', f'{row.sharpe:.3f}']
    for row in sensitivity.itertuples()
], ['压力比上限', '守底天数', '交易笔数', '年化收益', '夏普'])
cross_names = {'IM': '中证1000', 'IC': '中证500', 'IF': '沪深300', 'IH': '上证50'}
cross_table = table([
    [cross_names[prefix], f"{rows.loc['baseline','annual_return']*100:.2f}%",
     f"{rows.loc['combined','annual_return']*100:.2f}%",
     f"{rows.loc['baseline','max_drawdown']*100:.2f}%",
     f"{rows.loc['combined','max_drawdown']*100:.2f}%"]
    for prefix, rows in cross.set_index('method').groupby('prefix', sort=False)
], ['指数', '基准年化收益', '组合年化收益', '基准最大回撤', '组合最大回撤'])
trades_table = table([
    [f'{row.confirm:%Y-%m-%d}', f'{row.entry:%Y-%m-%d}', f'{row.exit:%Y-%m-%d}',
     row.contract, row.quantity, f'{row.profit/1e4:,.2f}']
    for row in ledgers['combined'].itertuples()
], ['确认日', '开仓日', '平仓日', '合约', '手数', '净利润（万元）'])

html = f'''<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<title>大跌后的入场筛选｜下行压力与低点稳定性</title>
<style>
body {{ margin:0; background:#f2f5f6; color:#273b49; font:17px/1.85 "Microsoft YaHei","Noto Sans CJK SC",sans-serif; }}
main {{ max-width:980px; margin:36px auto; padding:44px 56px; background:white; border-top:6px solid #1a5f94; }}
h1 {{ font-size:34px; line-height:1.4; margin:0 0 18px; }}
h2 {{ font-size:24px; margin:40px 0 12px; color:#174763; }}
h3 {{ font-size:18px; margin:24px 0 8px; }}
p {{ margin:12px 0; }}
a {{ color:#226e9c; }}
.lead {{ font-size:19px; }}
.callout {{ padding:16px 20px; background:#edf5fa; border-left:4px solid #4c92b8; }}
.table {{ overflow-x:auto; margin:16px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; white-space:nowrap; }}
th, td {{ padding:9px 12px; border-bottom:1px solid #dfe7ec; text-align:right; }}
th {{ background:#edf3f6; }}
th:first-child, td:first-child {{ text-align:left; }}
figure {{ margin:20px 0; }}
figcaption, footer {{ font-size:14px; color:#526b7b; }}
svg {{ display:block; width:100%; height:auto; }}
.formula {{ padding:10px 16px; background:#f1f5f7; overflow-x:auto; }}
li {{ margin:7px 0; }}
details {{ margin-top:18px; }}
summary {{ cursor:pointer; color:#174763; font-weight:600; }}
footer {{ margin-top:32px; border-top:1px solid #dfe7ec; padding-top:14px; }}
@media(max-width:700px) {{ main {{ margin:0; padding:24px 20px; }} h1 {{ font-size:26px; }} h2 {{ font-size:21px; }} body {{ font-size:16px; }} }}
@media print {{ main {{ margin:0; padding:0; border:0; }} body {{ background:white; }} h2 {{ break-after:avoid; }} table, svg {{ break-inside:avoid; }} }}
</style>
<main>
<h1>大跌后的入场筛选<br>下行压力与低点稳定性</h1>
<p class="lead">中证1000大跌后，第七个交易日收盘恢复到触发日的水平，华泰期货研报将其视为企稳。本研究在这项条件上增加两个因子，分别考察近期下跌的幅度，以及观察期内的最低价已经维持了多久。</p>
<p class="callout">在{nav.index[0]:%Y-%m-%d}至{nav.index[-1]:%Y-%m-%d}的中证1000回测中，双因子组合年化收益为<strong>{combined['annual_return']*100:.2f}%</strong>，基准为{baseline['annual_return']*100:.2f}%。组合的净利润增量主要来自2025年1月的一次交易变化。按早期样本选出的较严门槛，在随后的历史留出段表现较弱。</p>

<h2>因子怎么来</h2>
<p>两次大跌后的第七日收盘都可能回到触发日的水平，期间的行情却可能很不同。一种走势是跌势逐渐减弱、低点维持了几天，另一种是确认日盘中再次探底、收盘拉回。两种走势都能通过研报的收盘条件，分别考察下跌幅度和低点日期，可以进一步区分它们。</p>
<h3>下行压力衰减</h3>
<p>用日收益衡量下跌压力。把正收益记为零，将窗口内全部日收益的下行部分平方、取平均，再开平方。跌幅较大的日期贡献更多，快速上涨的日期贡献为零。确认日及此前两天反映观察期末的下跌情况，大跌触发日之前的20个交易日提供参照。</p>
<p class="formula">下行压力比 P = 确认日及此前两日的下行收益均方根 ÷ 触发日前20日的下行收益均方根</p>
<p>主策略要求 <strong>P ≤ 1</strong>，即观察期末的下行压力已回到触发前的水平或更低。</p>
<h3>低点稳定性</h3>
<p>在触发日到确认日之间，用每日最低价找出这段行情的最低点，再数它距确认日有几个交易日。最低价多次出现时，从最后一次开始数。主策略要求 <strong>至少守底3个交易日</strong>。例如确认日盘中刚触及这段最低价，守底天数为零，即便当天收盘已经回升。</p>

<h2>回测怎么做</h2>
<p>主回测使用中证1000指数日线生成信号，用对应的IM期货实际合约模拟交易，覆盖{len(nav)}个交易日。比较原企稳条件、单独加入下行压力因子、单独加入守底因子和双因子组合。四组采用相同的选约、资金、成交和退出规则。</p>
<ol>
<li><strong>信号。</strong>指数五日累计收益低于−4.92%时记为触发日。七个交易日后，收盘价达到或超过触发日收盘价，原企稳条件通过。双因子组在当日收盘同时检查压力比和守底天数。每次最多持有一笔交易，后续信号可在空仓时入场。</li>
<li><strong>选约与下单。</strong>在确认日已上市、距计划开仓日剩余期限超过40个交易日的IM合约中，选到期最近的一只。初始本金1亿元，每笔初始保证金上限1200万元。用确认日合约收盘价乘合约乘数和保证金率，得到每手保证金，再用预算除以该金额向下取整。IM乘数为每点200元，回测中保证金率自2023年5月5日起为12%，此前为15%。</li>
<li><strong>成交与退出。</strong>把保证金上限除以手数、合约乘数和保证金率，得到最高买价，按0.2点一档向下取整。次日开盘价在限价以内时按开盘价成交，高于限价则撤单。开仓日算第一个持有日，持满20个交易日后在下一交易日开盘平仓。</li>
<li><strong>损益。</strong>开平仓手续费各为成交额的万分之0.3，主回测滑点取零。持仓期间按每日收盘价估值，期末仍持有的合约按最后一个交易日收盘价计入净值。各组以固定保证金预算定仓，累计净损益除以初始本金计入净值。</li>
</ol>
<p>以250个交易日为一年，将期末净值折算为几何年化收益。夏普使用年化收益减去2%后除以年化波动率。平均名义敞口按持仓日的开仓名义金额占初始本金的比例取日均，空仓日计零。</p>

<h2>回测结果与交易归因</h2>
{main_table}
<figure>{main_svg}<figcaption>原企稳策略与双因子组合的净值和回撤。</figcaption></figure>
<p>单因子中，守底带来的收益改善较大，下行压力因子略微降低了最大回撤。组合的交易笔数从{baseline['trades']}笔降到{combined['trades']}笔，平均敞口也随之下降。最大回撤只相差{(baseline['max_drawdown']-combined['max_drawdown'])*100:.2f}个百分点，收益增量主要集中在少数交易。</p>
<h3>四组交易变化</h3>
{changes_table}
<p>双因子组合比基准多赚{net_gain/1e4:,.2f}万元。2023年5月压力比约为1.42，组合避开了一笔亏损。2024年11月确认日盘中出现了观察期最低价，守底条件过滤了一笔小幅盈利的交易。</p>
<p>2025年1月，组合在后续信号通过后于1月17日入场，比基准晚两个交易日。按同一选约规则，交易合约由IM2503变为IM2506，进出场价格和手数也随之变化。这组交易差额为{january_gain/1e4:,.2f}万元，占净增益约{january_gain/net_gain:.0%}。2025年4月同样等到了后续信号，晚一个交易日入场的组合少赚了{april_loss/1e4:,.2f}万元。</p>
<details>
<summary>查看双因子组合逐笔交易</summary>
{trades_table}
</details>

<h2>时间留出检验</h2>
<p>用{config['train_start']}至{config['train_end']}的{config['threshold_observations']}个五日收益观测估计5%分位数，得到大跌阈值{config['threshold']*100:.3f}%。训练期比较压力比上限0.75、1.0、1.25与守底2、3、4日的九组组合，要求至少有三笔已平仓交易，再按夏普选择参数。同分时优先选接近主策略的设定。训练期选中的是<strong>压力比上限{config['ratio_limit']}、守底{config['floor_days']}日</strong>。</p>
<p>训练与检验之间隔开{config['embargo_sessions']}个交易日。检验区间为{holdout_nav.index[0]:%Y-%m-%d}至{holdout_nav.index[-1]:%Y-%m-%d}，从空仓开始。原企稳基准、固定双因子组合和训练所选组合共用训练期估计的大跌阈值。</p>
{holdout_table}
<figure>{holdout_svg}<figcaption>后段历史行情中的三组净值。</figcaption></figure>
<p>固定组合小幅领先基准。训练所选的0.75门槛过滤了更多反弹，年化收益降到{holdout.loc['training_selected','annual_return']*100:.2f}%。选中组在训练期只有{config['training_closed_trades']}笔已平仓交易，守底2、3、4日的训练结果相同。2026年4月的压力比约为0.84，这笔上涨行情能通过1.0门槛，超过了0.75门槛。</p>
<h3>逐年检验</h3>
<p>分别使用各测试年度前一年的11月底及以前的数据估计阈值和选择参数，再在后续年度独立检验。表中为各段累计收益。</p>
{walk_table}
<p>2025年采用1.0门槛时，组合领先基准。2024年和2026年的训练均选出0.75，随后收益低于基准。因子构思参考了主回测区间的行情，以上结果属于历史时间留出，反映从早期数据选出的参数在随后行情中的表现。</p>

<h2>参数与指数比较</h2>
<p>把压力比上限和守底天数在主参数附近变动。0.75门槛下收益整体较低，1.0和1.25在守底3日时的表现接近。守底从3日延长到4日后，两档的年化收益均下降。</p>
{sensitivity_table}
<p>将同样的双因子条件应用于四个指数，大跌阈值和期货合约按对应指数设定。组合在中证1000、中证500和沪深300上提高了年化收益，上证50则呈现收益下降、回撤收窄。</p>
{cross_table}
<p>成交假设也影响短区间结果。2026年9月1日至11月9日，采用开盘限价时，基准与组合的区间收益均为{tail.loc['combined','total_return']*100:.2f}%。组合有两次开盘撤单，期末有一笔持仓按收盘估值。以相同信号按开盘市价成交、把保证金作为目标值时，组合区间收益为{tail.loc['market_target_budget','total_return']*100:.2f}%。</p>

<h2>结论</h2>
<p>下行压力和守底天数为七日企稳条件补充了等待期间的价格信息。中证1000的固定双因子组合减少了交易和敞口，年化收益小幅提高，最大回撤接近基准。收益增量集中在少数入场时点的变化，训练所选的较严门槛则错过了部分大反弹。入场筛选的主要取舍，是延后入场可能带来的收益改善与错过反弹的成本。</p>

<footer>
<p>数据来源为中证指数官方日线、国证指数交易日期及新浪实际期货合约行情。<a href="https://www.cffex.com.cn/u/cms/www/202207/18190308vxpv.pdf">中证1000股指期货合约规则</a></p>
<p><a href="https://www.htfc.com/wz_upload/png_upload/20260909/1788943656749408f5b.pdf">华泰期货原研报</a> · <a href="../REPORT.html">原研报复现</a> · <a href="README.md">复算说明</a></p>
</footer>
</main>
</html>'''
(ROOT / 'REPORT.html').write_text(
    '\n'.join(line.rstrip() for line in html.splitlines()) + '\n', encoding='utf-8',
)
print('Factor research report generated.')
