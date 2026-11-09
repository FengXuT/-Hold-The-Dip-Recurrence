"""Create the standalone Chinese results report from computed outputs."""
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from download_data import AUTHOR_COMMIT
from run_research import BASE, RAW, OUT, FACTORS

NAMES = {'Mkt-RF': '市场', 'SMB': '规模', 'HML': '价值', 'RMW': '盈利', 'CMA': '投资', 'Mom': '动量'}
COLORS = ['#8091a5', '#d09a43', '#176b8a']


def image(path, alt):
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def table(headers, rows):
    cells = ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in row) + '</tr>' for row in rows)
    return '<div class="tablewrap"><table><thead><tr>' + ''.join(f'<th>{x}</th>' for x in headers) + '</tr></thead><tbody>' + cells + '</tbody></table></div>'


def main():
    summary = pd.read_csv(OUT / 'summary.csv').set_index('factor').loc[FACTORS]
    decomposition = pd.read_csv(OUT / 'decomposition.csv').set_index(['factor', 'period'])
    terciles = pd.read_csv(OUT / 'terciles.csv').set_index(['factor', 'period'])
    reference = pd.read_csv(RAW / 'reference_terciles.csv').set_index(['factor', 'period'])
    paths = pd.read_csv(OUT / 'monthly_paths.csv', parse_dates=['date'])
    comparison = pd.read_csv(OUT / 'comparison.csv')
    audit = json.loads((OUT / 'comparison.json').read_text())
    run = json.loads((OUT / 'run.json').read_text())
    crosscheck = json.loads((OUT / 'independent_crosscheck.json').read_text())
    assert audit['all_passed'] and crosscheck['max_abs_error'] < 1e-12
    printed = comparison[comparison.section == 'Table 1'].set_index(['key', 'metric']).paper_reference
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18, 'figure.constrained_layout.use': True})

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(6)
    for i, (column, label) in enumerate([('sr_orig', 'Original'), ('sr_canon', 'Prior-month RV, net'), ('sr_net', 'Daily EWMA, net')]):
        ax.bar(x+(i-1)*.24, summary[column], .23, label=label, color=COLORS[i])
        ax.scatter(x+(i-1)*.24, [printed.loc[(f, column)] for f in FACTORS], marker='_', s=110,
                   color='#111827', linewidth=1.1, zorder=5, label='Paper (rounded)' if i == 0 else None)
    ax.set_xticks(x, ['Mkt', 'SMB', 'HML', 'RMW', 'CMA', 'Mom'])
    ax.set_ylabel('Annualized Sharpe ratio'); ax.set_ylim(0, 1.08)
    ax.legend(loc='upper left', ncol=2, frameon=False)
    save(fig, '01_sharpe.png')

    mom = paths[paths.factor == 'Mom'].set_index('date')
    fig, ax = plt.subplots(figsize=(10, 4.1))
    for column, label, color, style in [('original', 'Original momentum', COLORS[0], '-'),
                                          ('net', 'Managed momentum, net', COLORS[2], '-'),
                                          ('combo', 'Real-time combination, net', '#b76649', '--')]:
        values = mom[column].dropna()
        cumulative = (values * .10 / (values.std(ddof=1)*np.sqrt(12))).cumsum()
        ax.plot(values.index, cumulative, color=color, lw=1.5, ls=style, label=label)
    ax.axvline(pd.Timestamp('2017-01-01'), color='#555', lw=.8, ls=':')
    ax.set_ylabel('Cumulative summed excess return at 10% volatility')
    ax.legend(loc='upper left', frameon=False)
    save(fig, '02_momentum.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, period, title in zip(axes, ['pre', 'post'], ['1973-07 to 2016-12', '2017-01 to 2026-08']):
        for j, (column, label, color) in enumerate([('turbulent', 'Turbulent', '#bb604c'), ('middle', 'Middle', '#b6bbc2'), ('calm', 'Calm', '#176b8a')]):
            own = np.array([terciles.loc[(f, period), column]*100 for f in FACTORS])
            ref = np.array([reference.loc[(f, period), column]*100 for f in FACTORS])
            ax.bar(x+(j-1)*.25, own, .24, label=label, color=color)
            ax.scatter(x+(j-1)*.25, ref, marker='_', s=90, c='black', linewidth=.9, zorder=5)
        ax.axhline(0, color='#555', lw=.8)
        ax.set_xticks(x, ['Mkt', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']);ax.set_title(title)
    axes[0].set_ylabel('Mean factor return, % per year')
    axes[0].legend(loc='upper left', frameon=False, fontsize=8)
    save(fig, '03_terciles.png')

    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True)
    for ax, factor, label in zip(axes, ['Mom', 'Mkt-RF'], ['Momentum', 'Market']):
        part = paths[(paths.factor == factor) & paths.date.between('2007-01-01', '2010-12-31')]
        ax.bar(part.date, part.original*100, width=22, color=np.where(part.original < 0, '#db9a8b', '#9db9c4'))
        ax.set_ylabel('Factor return, %'); ax.set_title(label, loc='left', fontsize=10)
        right = ax.twinx();right.grid(False);right.plot(part.date, part.weight, color='#176b8a', lw=1.6)
        right.set_ylim(0, 1.65);right.set_ylabel('Exposure')
    save(fig, '04_crisis_exposure.png')

    main_rows, full_rows = [], []
    for f in FACTORS:
        row = summary.loc[f]
        main_rows.append([NAMES[f], f'{printed.loc[(f,"sr_orig")]:.2f}', f'{row.sr_orig:.4f}',
                          f'{printed.loc[(f,"sr_net")]:.2f}', f'{row.sr_net:.4f}', f'{row.p_holm:.4f}'])
        full_rows.append([NAMES[f], f'{row.sr_canon:.4f}', f'{row.sr_orig_cw:.4f}', f'{row.sr_combo:.4f}',
                          f'{row.p_c_holm:.4f}', f'{row.dd_orig:.2f}', f'{row.dd_net:.2f}'])
    main_table = table(['因子', '原文：原始', '复现：原始', '原文：管理后', '复现：管理后', '复现 Holm p'], main_rows)
    full_table = table(['因子', '传统方差法夏普', '组合期原始夏普', '实时组合夏普', '组合 Holm p', '原始最大损失/点', '管理后最大损失/点'], full_rows)
    decomposition_rows = []
    for f in FACTORS:
        for period, label in [('pre', '1973—2016'), ('post', '2017—2026')]:
            row = decomposition.loc[(f, period)]
            decomposition_rows.append([NAMES[f], label, f'{row.avg_w:.2f}', *[f'{row[c]*100:.2f}' for c in ['mean_f','ew_ef','cov_wf','cost','mean_net']],
                                       '—' if pd.isna(row.M) else f'{row.M:.2f}', f'{row.R:.2f}'])
    decomposition_table = table(['因子','时期','平均仓位','因子均值/%','恒定仓位项/%','择时协方差/%','成本/%','管理后均值/%','M','R'], decomposition_rows)
    error_rows = [
        ['Table 1：夏普、p 值、最大损失', '54 / 54', '按原表位数核对'],
        ['Table 2：12 组时期收益分解', '93 / 93', '3 个原文留空的 M 随原表处理'],
        ['Figure 1：36 根分组三分位柱', '36 / 36', f'最大差 {audit["max_tercile_error_pp"]:.5f} 个百分点'],
        ['同输入下的独立实现交叉核对', '18 / 18 条序列', f'逐月最大绝对误差 {crosscheck["max_abs_error"]:.0g}'],
    ]
    error_table = table(['核对对象','通过数量','精度'], error_rows)
    details_rows = [[html.escape(str(row.section)), html.escape(str(row.key)), html.escape(str(row.metric)),
                     f'{row.reference:.7f}', f'{row.reproduced:.7f}', f'{row.abs_error:.7f}'] for row in comparison.itertuples()]
    details_table = table(['来源','因子/时期','指标','作者 CSV','复现','绝对差'], details_rows)
    period_rows = []
    for label, part in [('1973-07—2016-12', mom.loc[:'2016-12-31']), ('2017-01—2026-08', mom.loc['2017-01-01':])]:
        period_rows.append([label, str(len(part)), *[f'{part[c].mean()/part[c].std(ddof=1)*np.sqrt(12):.4f}' for c in ['original','net']]])
    periods_table = table(['动量观察期','月数','原始夏普','管理后夏普'], period_rows)
    dp = decomposition.loc[('Mom','pre')];dq = decomposition.loc[('Mom','post')]
    market = summary.loc['Mkt-RF'];momentum = summary.loc['Mom']
    text = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>平静月与剧烈波动月：动量因子的波动率管理复现</title><style>
:root{{--ink:#203247;--muted:#5d6d7b;--blue:#176b8a;--line:#dbe3e8}}*{{box-sizing:border-box}}
body{{margin:0;background:#f1f4f6;color:var(--ink);font:16px/1.85 'Microsoft YaHei','PingFang SC',sans-serif}}
main{{max-width:1100px;margin:30px auto;padding:48px 58px;background:white;border-radius:10px}}
h1{{font-size:31px;line-height:1.4;margin:0 0 12px}}h2{{font-size:23px;margin:44px 0 14px;border-top:1px solid var(--line);padding-top:23px}}
h3{{font-size:18px;margin:24px 0 8px}}p{{margin:12px 0}}.meta,.caption{{color:var(--muted);font-size:14px}}a{{color:var(--blue)}}
.lead{{font-size:18px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:25px 0}}
.card{{background:#edf5f7;border-top:3px solid var(--blue);padding:15px 18px}}.number{{display:block;font-size:26px;font-weight:700;color:var(--blue)}}
img{{width:100%;height:auto;display:block;margin:15px 0 3px}}figure{{margin:20px 0}}.tablewrap{{overflow-x:auto;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}}th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf2f5;font-weight:600}}tbody tr:nth-child(even){{background:#f9fbfc}}
.formula,pre{{background:#f5f7fa;border-left:3px solid #9db9c4;padding:14px 18px;overflow-x:auto;font:14px/1.8 Consolas,monospace;white-space:pre-wrap}}.note{{background:#f7f8fa;padding:15px 20px}}details{{margin:20px 0}}summary{{cursor:pointer;color:var(--blue);font-weight:600}}footer{{border-top:1px solid var(--line);margin-top:35px;padding-top:15px;font-size:13px;color:var(--muted)}}
@media(max-width:720px){{main{{margin:0;padding:25px 18px}}h1{{font-size:25px}}.cards{{grid-template-columns:1fr}}}}@media print{{body{{background:white}}main{{margin:0;padding:0;max-width:none}}details{{display:block}}figure,table{{break-inside:avoid}}}}
</style></head><body><main>
<p class="meta">量化研究复现 · 2026 年 11 月 10 日 · 1973-07—2026-08，638 个月</p>
<h1>平静月与剧烈波动月：<br>动量因子的波动率管理复现</h1>
<p class="meta">原研究：Tanishk Yadav，<em>Calm Months, Crash Months: Why Volatility Management Pays for Momentum and Not the Market</em>，2026 年 9 月，SSRN 预印本 7113079。</p>
<p class="lead">在波动较大时减少仓位、平静时增加仓位，对不同因子的作用差异很大。本次复现中，动量因子的年化夏普从 {momentum.sr_orig:.4f} 提高到 {momentum.sr_net:.4f}，市场因子则从 {market.sr_orig:.4f} 降到 {market.sr_net:.4f}。六因子主表、分时期收益分解和三分位图均已与原文逐项核对。</p>
<div class="cards"><div class="card">动量年化夏普<span class="number">{momentum.sr_orig:.4f} → {momentum.sr_net:.4f}</span>已计敞口调整成本</div><div class="card">动量等波动最大损失<span class="number">{abs(momentum.dd_orig):.2f} → {abs(momentum.dd_net):.2f}</span>累计收益点数</div><div class="card">Holm 校正后的 p 值<span class="number">{momentum.p_holm:.4f}</span>六因子共同检验</div></div>
<h2>1. 研究问题与复现范围</h2>
<p>动量策略买入过去表现较好的股票、卖出过去表现较差的股票。市场剧烈反弹时，先前跌幅较大的股票可能反弹更快，动量的空头部分随之承受损失。原研究关注一个简单的仓位规则：依据最近的风险水平，每个月调整一次因子敞口。</p>
<p>本文覆盖原文 Table 1 的六因子绩效、实时组合、显著性和最大损失，Table 2 的精确收益分解，以及 Figure 1 的分组三分位结果；同时重建作者公开附图中的动量累计曲线。输入全部来自 Kenneth French 免费因子数据。原文后半部分以 CRSP 个股资料研究动量的市场 beta，属于另一项个股层面的机制检验。</p>
<p>六个因子分别是市场超额收益 Mkt-RF、规模 SMB、价值 HML、盈利 RMW、投资 CMA 和动量 Mom。前五个来自五因子文件，动量来自单独的动量文件。样本从 1963 年 7 月开始，前 120 个月用于积累估计历史，正式持仓从 1973 年 7 月开始。</p>
<h2>2. 从原始数据到每月持仓</h2>
<h3>先把“收益”和“风险”放到各自对应的数据上</h3>
<p>每个月赚取的因子收益取官方月度文件；每天的因子收益用于估计方差。月收益和日风险序列在月末日期对齐。源文件以百分数表示收益，读入后统一除以 100。当前官方输入文件使用 2026 年 9 月数据版本，本次固定取到 2026 年 8 月末，共 758 个月。</p>
<h3>用最近的日收益估计下个月的风险</h3>
<p>主规则对每日收益的平方做指数加权平均，半衰期为 21 个交易日。较新的观测权重更大；约一个月以前的观测权重衰减一半。取月末估计值乘以 21，得到月度风险信号。作为对照，传统规则使用上个月日收益相对该月均值的平方离差之和。</p>
<div class="formula">λ = 2^(−1/21)
σ²[d] = λ × σ²[d−1] + (1−λ) × r[d]²
S[m] = 21 × σ²[该月最后一个交易日]</div>
<h3>风险越高，次月仓位越低</h3>
<p>月 m 的仓位在上月末确定。缩放常数 c 使用当时已经积累的历史估计，分母是历次“月收益 ÷ 前月风险信号”的标准差。缩放估计从 1963 年 8 月起具备前月风险信号的月份开始。仓位上限为 1.5。仓位计算从预热期的最后一个月开始，给正式首月提供前期仓位，因此第一笔换仓成本也能按同一规则计算。</p>
<div class="formula">c[m−1] = sd(f[1…m−1]) / sd(f[s] / S[s−1], s ≤ m−1)
w[m] = min(c[m−1] / S[m−1], 1.5)
净收益[m] = w[m] × f[m] − 0.0014 × |w[m]−w[m−1]|</div>
<p>每单位仓位变化收取 14 个基点。例如仓位从 0.8 调整到 0.5，成本为 0.0014 × 0.3，即当月收益的 0.042%。这项成本对应因子敞口调整。研究直接使用 French 已构造的多空因子收益。</p>
<h2>3. 六因子主表：动量的改善最明显</h2>
<p>夏普按月收益均值除以样本标准差，再乘 √12。动量的管理后夏普约为原来的 {momentum.sr_net/momentum.sr_orig:.2f} 倍。盈利因子也有所提高，六因子共同进行多重检验后，动量的校正 p 值为 {momentum.p_holm:.4f}；其他五个因子的校正 p 值均为 1.000。</p>
{main_table}
<figure>{image(OUT/'01_sharpe.png','六因子夏普与原文的对照')}<figcaption class="caption">图 1　复现结果。横向黑色短线为论文印刷值；柱形为独立计算值。传统方差法与日 EWMA 法都采用相同的仓位上限和成本。</figcaption></figure>
<p>原文还使用“原始因子 + 管理后因子”的实时组合：每月根据过去的均值和协方差估计组合比例，把两部分换算成最终因子敞口，再收取该敞口的调整成本。组合使用额外 60 个月历史，从 1978 年 7 月开始。动量组合夏普复现为 {momentum.sr_combo:.4f}，原文为 0.80。</p>
<details><summary>查看传统规则、实时组合和最大损失的完整结果</summary>{full_table}<p class="caption">最大损失沿用原表定义：月收益先缩放至原始因子的波动率，随后累加，计算累计收益的峰谷差，单位为收益点数。</p></details>
<h2>4. 累计曲线与历史中的仓位变化</h2>
<p>为了让不同波动率的序列便于比较，曲线统一缩放至 10% 年波动，再累加每月超额收益。灰线为原始动量，蓝线为管理后动量，虚线为实时组合。2017 年以后的区间单独列示，便于观察参数选择期以后的表现。</p>
<figure>{image(RAW/'reference_momentum.png','作者发布的动量累计曲线')}<figcaption class="caption">图 2a　作者固定版本发布的原图，来源为原项目 fig_momentum.png。</figcaption></figure>
<figure>{image(OUT/'02_momentum.png','独立复现的动量累计曲线')}<figcaption class="caption">图 2b　独立复现。采用原图相同的 10% 波动缩放和累加口径，主要拐点、曲线排序与长期斜率吻合。</figcaption></figure>
{periods_table}
<p>在 2007—2010 年的窗口中，风险积累后动量仓位明显下降，2009 年反弹阶段的损失敞口随之收缩。该规则通过前月风险调整本月仓位，月内保持给定的因子敞口。</p>
<figure>{image(OUT/'04_crisis_exposure.png','危机时期月收益与仓位')}<figcaption class="caption">图 3　月收益用柱形表示，蓝线为当月仓位。动量和市场分别使用各自的风险信号。</figcaption></figure>
<h2>5. 改善从哪里来：把收益拆开看</h2>
<p>平均管理后收益可以精确拆成三项：恒定平均仓位带来的收益、仓位与因子收益的协方差、仓位调整成本。协方差为正，意味着较高仓位平均对应较好的收益月份；协方差为负，则表明较低仓位平均对应较好的收益月份。</p>
<div class="formula">平均净收益 = 平均仓位 × 平均因子收益 + Cov(仓位, 因子收益) − 平均成本</div>
<p>1973—2016 年，动量的恒定仓位项为年化 {dp.ew_ef*100:.2f}%，择时协方差增加 {dp.cov_wf*100:.2f} 个百分点，成本消耗 {dp.cost*100:.2f} 个百分点，得到 {dp.mean_net*100:.2f}% 的平均净收益。2017—2026 年，相同三项为 {dq.ew_ef*100:.2f}%、{dq.cov_wf*100:.2f} 和 {dq.cost*100:.2f} 个百分点，得到 {dq.mean_net*100:.2f}%。两段时期的动量择时协方差均为正。</p>
{decomposition_table}
<p class="caption">表 2　独立复现的收益分解。收益、协方差和成本均为年化百分数。M 衡量平均收益渠道，R 衡量波动收缩渠道；M × R 等于管理后夏普与原始夏普之比。原表在因子平均收益为负的三行将 M 留空。</p>
<p>原文 Figure 1 按仓位将月份分成三组。低仓位组对应较高的前期风险，高仓位组对应较低的前期风险。动量在平静组的平均收益高于剧烈波动组；其他因子的分组关系各有差异，这解释了同一套规则产生不同的效果。仓位触及上限时，使用日期顺序打破并列，再等分三组，与作者实现保持一致。</p>
<figure>{image(RAW/'reference_terciles.png','论文Figure 1原图')}<figcaption class="caption">图 4a　论文 Figure 1 原图，按仓位三分位比较因子收益。</figcaption></figure>
<figure>{image(OUT/'03_terciles.png','独立复现的三分位因子收益')}<figcaption class="caption">图 4b　复现值与作者参考值的叠加。黑色短线为作者参考值；36 根柱的最大差为 {audit['max_tercile_error_pp']:.5f} 个百分点，RMSE 为 {audit['tercile_rmse_pp']:.5f} 个百分点。</figcaption></figure>
<h2>6. 与原结果究竟有多接近</h2>
{error_table}
<p>主表的五类夏普共 30 项，与作者四位小数 CSV 的最大绝对差为 {audit['max_sharpe_error']:.7f}。最大损失的最大差为 {audit['max_loss_error_pp']:.6f} 个收益点；收益分解中年化收益项的最大差为 {audit['max_return_item_error_pp']:.5f} 个百分点；两组 Holm 校正 p 值在计算精度内一致。</p>
<p>本次同时保留更细的 CSV 诊断：{audit['strict_csv_passed']} / {audit['total']} 项达到夏普及比率差 0.001、年化收益项和损失差 0.02 个百分点的严格目标。剩余两项是动量后期的 M（1.8681 对 1.8733，原表均显示 1.87）和低仓位组平均收益（−6.1108% 对 −6.14%）。论文表格按印刷位数验收，分组图的最大容差为年化 0.05 个百分点，全部 {audit['total']} 项通过。</p>
<p>为继续定位这两处差别，我把作者固定版本和独立实现放到完全相同的官方输入上重算。六因子的仓位、净收益及实时组合共 18 条月序列逐点一致，最大差为 {crosscheck['max_abs_error']:.0g}。French 说明每次数据更新都会重建历史；作者参考结果与当前输入快照存在细小数值差别。本文保留原始输入和误差明细，便于按版本继续核查。</p>
<details><summary>查看全部 183 项细粒度数值对照</summary><p class="caption">收益项及分组均值使用年化百分点；最大损失使用收益点数；夏普、M、R 和 p 值使用原数值。</p>{details_table}</details>
<h2>7. 复现时做了哪些检查</h2>
<p>检查从数据覆盖开始：四个文件分别识别日/月区段，统一单位，固定 1963-07—2026-08 的研究窗口。随后用递推算例核对 EWMA，再把 2008 年末及之后的收益、风险全部替换，检查此前仓位保持一致。成本核对直接从“仓位 × 当月收益 − 调整费用”重建净收益，12 组平均收益分解都在浮点精度内相等。</p>
<p>显著性沿用论文的学生化圆形区块重采样：区块长 12 个月，4,999 次，随机种子 20260928；638 月主样本每次抽取 54 个完整区块，共 648 个观测。每次检验单独重置种子，随后对六因子做 Holm 校正。这个完整区块约定和第一期成本的处理，会影响更细的 p 值与收益对照。</p>
<p>对照程序直接读取作者的 LaTeX 印刷表，以及保存到四位小数的参考 CSV。这样可以同时检查论文展示结果和更细的小数差异。参数始终采用论文的 21 日半衰期、1.5 倍上限及 14bp 成本，2017 年以后的结果另外列出。</p>
<h2>8. 如何重新运行与阅读输出</h2>
<p>在本项目文件夹安装 requirements.txt 中的依赖，按以下顺序运行。前两个脚本负责准备输入和计算；后三个分别核对作者实现、检验原表误差并生成这份报告。</p>
<pre>python -m pip install -r requirements.txt
python download_data.py
python run_research.py
python verify_implementation.py
python compare_results.py
python build_report.py</pre>
<p>原始文件及下载来源保存在 data/，月度路径、逐项误差和中间统计保存在 outputs/。REPORT.html 已嵌入全部图表，下载后可以离线阅读。初次下载后默认复用快照；download_data.py --refresh 会重新获取官方输入。</p>
<h2>资料来源</h2>
<p><a href="https://www.tanishkyadav.me/research/volatility-managed-factors">作者研究主页与发布日期</a>；
<a href="https://www.tanishkyadav.me/papers/yadav-2026-calm-months-crash-months.pdf">原论文 PDF</a>；
<a href="https://github.com/tanishhky/Equity-Factor-Volatility-Analysis/tree/{AUTHOR_COMMIT}">固定版本代码与原始结果</a>；
<a href="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html">French 官方数据与历史修订说明</a>。
原图来源于作者公开项目，仅用于研究结果对照。</p>
<footer>独立复现日期：2026-11-10。作者参照版本：{AUTHOR_COMMIT}。本次运行：NumPy {run['versions']['numpy']}，pandas {run['versions']['pandas']}。输入来源及快照信息记录于 data/manifest.json。</footer>
</main></body></html>'''
    assert all(phrase not in text for phrase in ['不代表', '不能认为', '不等于', '并非', '简历', '面试'])
    (BASE / 'REPORT.html').write_text(text, encoding='utf-8')
    print(f'Created REPORT.html: {len(text.encode("utf-8")):,} bytes; 6 embedded figures')


if __name__ == '__main__':
    main()
