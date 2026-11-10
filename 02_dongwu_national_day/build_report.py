"""Build the readable, self-contained Chinese results report."""
import base64
import io
import json
from html import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catalog import BASE, OUT, PAPER_URL, WINDOWS, SERIES

plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white'})


def picture(fig, alt):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=165, bbox_inches='tight')
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{escape(alt)}"></figure>'


def stats_table(names, own, reference):
    records = []
    for name in names:
        row = {'指数': name}
        for window in WINDOWS:
            a = own.loc[(name, window)]
            b = reference.loc[(name, window)]
            row[window] = f'{a.trimmed_mean:.2f} / {b.trimmed_mean:.2f}'
        records.append(row)
    return pd.DataFrame(records).to_html(index=False, classes='data', border=0)


def main():
    stats = pd.read_csv(OUT / 'source_stats.csv')
    valid = pd.read_csv(OUT / 'valid_stats.csv')
    ref = pd.read_csv(OUT / 'reference.csv')
    comparison = pd.read_csv(OUT / 'comparison.csv')
    coverage = pd.read_csv(OUT / 'coverage.csv', dtype={'code': str})
    events = pd.read_csv(OUT / 'event_returns.csv')
    paths = pd.read_csv(OUT / 'event_paths.csv')
    summary = json.loads((OUT / 'validation.json').read_text())
    source_stats = stats.set_index(['name', 'window'])
    reference = ref.set_index(['name', 'window'])
    selected = coverage.loc[coverage.included, 'name'].tolist()
    styles = [name for source, code, name in SERIES if source != 'sw']
    industries = [name for source, code, name in SERIES if source == 'sw' and name in selected]

    means = comparison.query("metric == 'trimmed_mean'")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    colors = {'cni': '#227c9d', 'csi': '#17a398', 'sw': '#7763ac', 'citic': '#e6a23c'}
    labels = {'cni': '国证', 'csi': '中证／上证', 'sw': '申万', 'citic': '中信'}
    for source, group in means.groupby('source'):
        axes[0].scatter(group.paper, group.own, s=20, color=colors[source], label=labels[source], alpha=.75)
    axes[0].plot([-3, 3], [-3, 3], color='#9aa7b0', lw=1)
    axes[0].set(xlabel='原表去极值均值（%）', ylabel='复现去极值均值（%）', title='128个均值：原表与复现')
    axes[0].legend(fontsize=8)
    axes[1].hist(means.error_pp, bins=np.linspace(-.04, .04, 33), color='#227c9d')
    axes[1].set(xlabel='复现 − 原表（百分点）', ylabel='统计项数量', title='逐项差异的分布')
    fig.tight_layout()
    alignment_chart = picture(fig, '128项均值的原表对照与误差分布')

    fig, axes = plt.subplots(1, 2, figsize=(11, 7.2), gridspec_kw={'width_ratios': [1, 1]})
    for ax, names, title in zip(axes, [styles, industries], ['风格与宽基', '具有完整20年历史的行业']):
        matrix = source_stats.trimmed_mean.unstack().reindex(index=names, columns=list(WINDOWS))
        image = ax.imshow(matrix, cmap='RdYlGn_r', vmin=-3, vmax=3, aspect='auto')
        ax.set_xticks(range(4), list(WINDOWS))
        ax.set_yticks(range(len(names)), names, fontsize=9)
        for i in range(len(names)):
            for j in range(4):
                ax.text(j, i, f'{matrix.iloc[i,j]:.2f}', ha='center', va='center', fontsize=8)
        ax.set_title(title)
    fig.colorbar(image, ax=axes.tolist(), label='去极值收益均值（%）', shrink=.6, pad=.03)
    heatmap = picture(fig, '32个指数在国庆前后四个窗口的收益均值')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    for name, color in [('沪深300', '#227c9d'), ('中证1000', '#17a398'), ('国证2000', '#b47b28')]:
        a = paths.query('year < 2026').loc[paths.name.eq(name)].groupby('offset').return_pct.apply(lambda x: np.sort(x.dropna())[1:-1].mean())
        axes[0].plot(a.index, a, label=name, color=color, lw=2)
    axes[0].axvline(0, color='#8b98a2', lw=1)
    axes[0].axhline(0, color='#ccd3d9', lw=.8)
    axes[0].set(title='补充观察：节前最后收盘归零', xlabel='相对交易日', ylabel='各点去极值累计收益（%）')
    axes[0].legend(fontsize=8)
    bars = events.query("name == '沪深300' and window == 'T+5'").sort_values('year')
    axes[1].bar(bars.year, bars.return_pct, color=['#d58840' if y == 2026 else '#227c9d' for y in bars.year])
    axes[1].axhline(0, color='#8b98a2', lw=.8)
    axes[1].set(title='沪深300：历年节后5日收益', xlabel='年份（橙色为发表后2026年）', ylabel='区间收益（%）')
    axes[1].set_xticks([2006, 2010, 2015, 2020, 2026])
    fig.tight_layout()
    paths_chart = picture(fig, '自主补充的宽基事件曲线及沪深300历年节后收益')

    # Resample entire holiday events, preserving one independent unit per year.
    a = events.query("name == '沪深300' and window == 'T+5' and year < 2026").return_pct.to_numpy()
    samples = np.random.default_rng(20260926).choice(a, size=(10000, len(a)), replace=True)
    boot_means = np.sort(samples, axis=1)[:, 1:-1].mean(axis=1)
    ci = np.quantile(boot_means, [.025, .975])
    sensitivity = []
    for label, group in [('2006—2015', events.query('2006 <= year <= 2015')),
                         ('2016—2025', events.query('2016 <= year <= 2025'))]:
        values = group.query("name == '沪深300' and window == 'T+5'").return_pct.to_numpy()
        sensitivity.append({'样本': label, '年份数': len(values), '胜率': f'{(values > 0).mean()*100:.0f}%',
                            '去极值均值': f'{np.sort(values)[1:-1].mean():.2f}%'})
    sensitivity.append({'样本': '2006—2025', '年份数': len(a), '胜率': f'{(a > 0).mean()*100:.0f}%',
                        '去极值均值': f'{np.sort(a)[1:-1].mean():.2f}%'})
    sensitivity_table = pd.DataFrame(sensitivity).to_html(index=False, border=0, classes='data')
    pd.DataFrame(sensitivity).to_csv(OUT / 'stability.csv', index=False, encoding='utf-8-sig')

    boundary = pd.read_csv(OUT / 'event_dates.csv').query('year == 2025')
    boundary['价格比'] = ['P[t] / P[t−11] − 1', 'P[t] / P[t−6] − 1', 'P[t+5] / P[t] − 1', 'P[t+10] / P[t] − 1']
    boundary = boundary.rename(columns={'window': '原表标签', 'start': '起始收盘日', 'end': '结束收盘日', 'intervals': '收益区间数'})
    boundary_table = boundary[['原表标签', '价格比', '起始收盘日', '结束收盘日', '收益区间数']].to_html(index=False, classes='data', border=0)
    audit = pd.read_csv(OUT / 'window_boundary_audit.csv')
    audit['boundary'] = audit.boundary.map({'original_boundary': '原表反推边界', 'standard_N_intervals': '常规N个收益区间'})
    audit.columns = ['计算边界', '窗口', '指数数', '均值精度内吻合数', '胜率精度内吻合数', '均值最大差异（百分点）']
    audit_table = audit.to_html(index=False, classes='data', border=0, float_format=lambda x: f'{x:.4f}')
    valid_2000 = valid.query("name == '国证2000' and window.str.startswith('T+')").copy()
    special_rows = []
    for row in valid_2000.to_dict('records'):
        s = source_stats.loc[('国证2000', row['window'])]
        special_rows.append({'窗口': row['window'], '有效样本': int(row['n']), '有效样本胜率': f"{row['win_rate']:.2f}%",
                             '有效样本均值': f"{row['trimmed_mean']:.4f}%", '原表对齐样本': int(s.n),
                             '原表对齐胜率': f'{s.win_rate:.0f}%', '原表对齐均值': f'{s.trimmed_mean:.4f}%'})
    special_table = pd.DataFrame(special_rows).to_html(index=False, classes='data', border=0)
    now = events.query("year == 2026 and name in ['沪深300', '中证1000', '国证2000'] and window.str.startswith('T+')")
    now_table = now.pivot(index='name', columns='window', values='return_pct').reindex(columns=['T+5', 'T+10'])
    now_table = now_table.map(lambda x: f'{x:.2f}%').rename_axis('指数').to_html(classes='data', border=0)
    errors = comparison.loc[~comparison.print_match, ['name', 'window', 'own', 'paper', 'error_pp']].copy()
    errors.columns = ['指数', '窗口', '复现均值（%）', '原表均值（%）', '差异（百分点）']
    error_table = errors.to_html(index=False, classes='data', border=0, float_format=lambda x: f'{x:.5f}')
    missing = coverage.loc[~coverage.included, ['name', 'first_date']].copy()
    missing.columns = ['行业', '本次取得的官方历史起点']
    missing_table = missing.to_html(index=False, classes='data', border=0)
    all_comparison = comparison[['name', 'window', 'metric', 'own', 'paper', 'error_pp']].copy()
    all_comparison['metric'] = all_comparison.metric.map({'win_rate': '胜率', 'trimmed_mean': '去极值均值'})
    all_comparison.columns = ['指数', '窗口', '统计量', '复现（%）', '原表（%）', '差异（百分点）']
    detail = all_comparison.to_html(index=False, classes='data', border=0, float_format=lambda x: f'{x:.5f}')

    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>东吴证券国庆季节效应研报复现</title><style>
:root{{--ink:#203747;--muted:#637684;--accent:#227c9d;--line:#dce5eb}}*{{box-sizing:border-box}}body{{margin:0;background:#f1f5f8;color:var(--ink);font:16px/1.85 "Microsoft YaHei","PingFang SC",sans-serif}}main{{max-width:1120px;margin:40px auto;padding:48px 62px;background:white;border-radius:12px}}h1{{font-size:32px;line-height:1.4;margin:12px 0 20px}}h2{{font-size:23px;margin-top:48px;border-bottom:2px solid var(--line);padding-bottom:9px}}h3{{font-size:18px;margin-top:28px}}p{{margin:14px 0}}a{{color:var(--accent)}}.eyebrow{{font-size:13px;letter-spacing:2px;color:var(--accent)}}.lead{{font-size:18px}}.meta,.caption{{color:var(--muted);font-size:14px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:28px 0}}.card{{background:#edf5f8;padding:18px;border-radius:8px}}.card strong{{display:block;font-size:29px;color:var(--accent)}}figure{{margin:24px 0}}img{{max-width:100%;height:auto}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px;line-height:1.6;margin:18px 0}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf5f8;color:#244e63}}details{{padding:14px 18px;border:1px solid var(--line);border-radius:7px;margin:20px 0}}summary{{cursor:pointer;font-weight:bold}}code{{background:#edf3f6;padding:2px 5px;border-radius:3px}}.callout{{border-left:4px solid var(--accent);padding:8px 20px;background:#f3f8fa}}li{{margin:7px 0}}@media(max-width:760px){{main{{margin:0;padding:24px 18px;border-radius:0}}h1{{font-size:26px}}.cards{{grid-template-columns:1fr}}}}@media print{{body{{background:white}}main{{margin:0;padding:10px;max-width:none}}h2{{break-after:avoid}}figure,tr{{break-inside:avoid}}}}
</style><main><div class="eyebrow">A股 · 券商研报复现 · 2026年11月10日</div>
<h1>国庆前后的 A股，历史上怎样变化？</h1>
<p class="lead">复现东吴证券《国庆假期的“季节效应”》：以2006—2025年的国庆假期为事件，核对宽基、风格和行业指数的胜率与收益均值。</p>
<p class="meta">原件：东吴证券研究所 · 策略周评 · 2026年9月26日 · 陈刚、林煜辉 · 5页。<a href="{PAPER_URL}">查看研报原始PDF</a>。使用封面日期；资料平台的收录日期为9月27日。</p>
<div class="cards"><div class="card"><strong>32 / 48</strong>原表指数覆盖</div><div class="card"><strong>{summary['print_matches']} / 256</strong>按原表精度吻合的统计项</div><div class="card"><strong>{summary['max_mean_error_pp']:.4f} pp</strong>均值最大绝对差异</div></div>
<p>本次复现对象是研报第4页的图3和图4，共覆盖17个风格／宽基指数、15个行业指数。128项胜率全部按显示精度对齐，128项收益均值中120项按两位小数对齐；其余8项来自中信风格指数，最大差异约3.23个基点。国证2000的节后四项统计包含缺失年份的口径反推，16年有效样本结果在下文并列展示。全部均值的均方根误差为{summary['rmse_mean_pp']:.4f}个百分点。</p>
<p>研究流程是：核验券商原件，提取原表数字，下载指数机构的公开历史行情，逐年定位假期，计算四个窗口，再逐格对照。对于差异，先检查窗口与样本口径，再检查数据来源。</p>

<h2>1　原研报研究什么，复现覆盖到哪里</h2>
<p>原文研究国庆前后市场、成交和风格的季节性变化，属于券商策略研究。它先观察市场在假期附近的表现，再比较宽基、风格与行业。本项目集中复现其中能够用公开官方价格独立核对的两张统计表。</p>
<div class="scroll"><table><tr><th>原文内容</th><th>本项目覆盖</th><th>数据来源</th></tr>
<tr><td>图3：18个风格／宽基</td><td>17个，136项</td><td>国证、中证、中信官方历史</td></tr>
<tr><td>图4：30个申万一级行业</td><td>15个，120项</td><td>申万宏源研究官方历史</td></tr>
<tr><td>图1／图2：Wind全A价格／成交路径</td><td>保留原文的研究背景</td><td>原文使用Wind全A；本项目另绘公开宽基事件路径</td></tr></table></div>
<p>图3余下的微盘股指数，以及图4余下15个行业的2006年起连续历史，是后续扩展所需的数据。行业进入本次核对的条件为：四个窗口均有20个有效年份。该条件在计算原表误差前按历史覆盖确定。</p>
<details><summary>查看15个行业的公开历史起点</summary><div class="scroll">{missing_table}</div><p>这些行业在官方接口中取得的历史起点较晚。原表通过Wind取得的长期回溯序列，是进一步扩大覆盖时需要补齐的资料。</p></details>

<h2>2　怎样把原表的口径还原出来</h2>
<h3>交易日、收益和去极值</h3>
<p>以每年10月1日之前最后一个交易日为锚点 <code>t</code>。交易日历取自国证实际日线日期；各指数按该日历对齐。价格采用收盘点位，区间收益为结束收盘价除以起始收盘价再减一。停市日期保持空缺，所有价格沿用来源数值。</p>
<p>每个窗口先得到20个年度收益。胜率为正收益年份数除以有效年份数，零收益计入分母；收益均值剔除该窗口的一个最大值和一个最小值后计算。20年完整样本对应18个收益取平均。原表注释明确给出均值去极值规则。</p>
<p>通过6个国证风格与5个中证／上证宽基的联合核对，原表节前标签对应的边界可还原为下面的价格比。节前两列分别包含11、6个收盘收益区间。程序把这套边界写成固定配置，并额外输出通常的10、5个区间计算结果，便于检查边界的影响。</p>
<div class="scroll">{boundary_table}</div>
<p class="caption">示例年份为2025年。每年对应的真实起止日期写入 <code>outputs/event_dates.csv</code>，通常10／5个区间的结果写入 <code>standard_window_stats.csv</code>。原件给出了窗口标签；这里的精确收盘边界来自多指数联合反推。</p>

<details><summary>两种节前边界在11个完整指数上的联合核对</summary><div class="scroll">{audit_table}</div><p>逐项结果保存在 <code>window_boundary_audit.csv</code>。原表反推边界在两列共44项统计中全部对齐。</p></details>
<h3>国证2000：分别展示“原表对齐”和“有效样本”</h3>
<p>公开的国证2000行情覆盖2010—2025年的16次国庆事件。节前两列按16个有效年份计算，能够对齐原表。节后两列加入4个零收益观察、按20个年份计算，能够同时对齐原表的胜率与去极值均值；这一处理是由原表数字反推的口径。真实价格和逐年收益始终保留缺失值，零值只在“原表对齐统计”中加入。</p>
<div class="scroll">{special_table}</div>
<p>有效样本结果单独保存于 <code>valid_stats.csv</code>。解释大小盘的历史相对表现时，可以先统一年份，再计算每年相对收益；原表分列均值之间的差值同时受到样本数量与去极值的影响。</p>

<h2>3　复现与原表有多接近</h2>
{alignment_chart}
<p>国证、中证／上证与15个申万行业合计216项统计，全部在原表显示精度内吻合。中信风格的20项胜率也全部吻合，20项均值中12项吻合。下表列出剩余8项，数值直接来自官方历史数据的计算结果。</p>
<div class="scroll">{error_table}</div>
<p>原表注明数据源为Wind；本次使用中信官网点位。现有资料可以把剩余差异定位到这组风格序列，继续确认其历史版本差异需要同日期的Wind原始点位。当前保持同一套时间窗口与官方原始行情，按实际误差呈现结果。</p>
<details><summary>逐格核对全部256项统计</summary><div class="scroll">{detail}</div></details>

<h2>4　结果怎样理解</h2>
<p>在覆盖的指数中，节后5日收益整体强于节前。沪深300节后5日上涨14次，胜率70%，去极值均值1.56%；节后10日均值降至0.96%。金融风格节后5日和10日的均值分别约1.91%和2.01%，相对更有持续性。节前行业中，食品饮料与医药生物的收益均值较高。</p>
{heatmap}
<p class="caption">单位为%。红色表示正收益，绿色表示负收益；按原表对齐口径绘制。国证2000的有效样本值见前表。</p>
<details><summary>风格、宽基与行业：复现均值／原表均值</summary><h3>17个风格／宽基</h3><div class="scroll">{stats_table(styles, source_stats, reference)}</div><h3>15个行业</h3><div class="scroll">{stats_table(industries, source_stats, reference)}</div><p>表格单位为%，每格先列复现值，再列原表值。完整胜率与误差可在256项核对表中查看。</p></details>

<h3>补充观察：年度波动与发表后的2026年</h3>
{paths_chart}
<p class="caption">左图为本项目补充绘制的公开宽基事件曲线，按各年节前最后收盘归零，再逐点去极值求均值。国证2000使用16年有效历史；沪深300、中证1000使用20年。原研报图1绘制的是Wind全A。右图保留每年真实波动，2026年用橙色标示。</p>
<p>把沪深300节后5日的20个年度事件分成前后两段，可看到季节性均值随样本变化。按年度事件进行10,000次自助重采样，去极值均值的95%区间为 [{ci[0]:.2f}%, {ci[1]:.2f}%]。这里以“年”为重采样单位，保留同一假期作为一次观察。</p>
<div class="scroll">{sensitivity_table}</div>
<p>研报发表于2026年9月26日，2026年的节后收益可以单独观察。以9月30日收盘为起点，节后第5／10个交易日分别为10月14日和10月21日：</p>
<div class="scroll">{now_table}</div>
<p>2026年这三个指数节后均回落。结合年度收益图和分段统计，后续研究可以围绕“哪些市场状态更容易出现节后回升”展开，把历史均值转化成有条件的交易假设。</p>

<h2>5　从复现走向新的研究问题</h2>
<p><strong>假期前后的相对强弱。</strong>在统一年份上，分别计算消费相对沪深300的节前超额、金融相对沪深300的节后超额，研究两段是否具有稳定的风格切换。先用截至上一年的数据估计，再在下一年检验，可以形成年度滚动的选取规则。</p>
<p><strong>节前跌幅与节后修复。</strong>把每年节前最后5个常规收益区间的跌幅作为状态变量，观察其与节后5日收益、大小盘相对收益的关系。进一步加入节前成交收缩比例，可以检验“卖压释放后修复”的条件。新增因子进入回测时，信号形成日、下一可成交日和交易成本需一并记录。</p>
<p>这两个方向首先需要比较统一样本的年度分布，再逐步形成规则。现有代码已提供逐年、逐指数、逐窗口收益，方便继续研究。</p>

<h2>6　读者怎样重现这份结果</h2>
<p>从仓库中的 <code>02_dongwu_national_day</code> 开始，安装 <code>requirements.txt</code>，依次运行 <code>download_data.py</code>、<code>run_research.py</code>、<code>compare_paper.py</code>、<code>build_report.py</code>。下载程序保存原始PDF和47个官方指数历史文件，计算程序据历史覆盖选择32个指数，比较程序逐项读取原件表格，最后生成这份内嵌图表的HTML报告。</p>
<p>README列出完整命令与各输出的用途。均值核对以0.00501个百分点检查两位小数的显示精度；中信组另设0.04个百分点的后续运行报警线，并在本报告完整展示8项显示差异。结果评价以实际误差为依据。</p>
<p class="meta">原始价格快照截至2026年11月6日；历史对齐样本固定为2006—2025年。公开数据提供方：<a href="https://www.cnindex.com.cn/">国证指数</a>、<a href="https://www.csindex.com.cn/">中证指数</a>、<a href="https://research.citicsinfo.com/idx">中信证券指数研究</a>、<a href="https://www.swsresearch.com/">申万宏源研究</a>。下载地址、请求参数与快照信息保存在本地 <code>data/manifest.json</code>。</p>
</main></html>'''
    (BASE / 'REPORT.html').write_text(html, encoding='utf-8')
    print(f'REPORT.html: {len(html.encode()):,} bytes; 3 embedded figures')


if __name__ == '__main__':
    main()
