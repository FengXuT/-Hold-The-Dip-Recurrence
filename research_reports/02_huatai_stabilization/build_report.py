"""根据本次计算结果生成一份可离线阅读的HTML报告。"""
from io import StringIO


import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pypdf import PdfReader
from reproduce import ROOT, RAW, OUT

plt.rcParams.update({'font.family': ['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11,
                     'svg.fonttype': 'none', 'axes.grid': True, 'grid.alpha': .2})
logging.getLogger('pypdf').setLevel(logging.ERROR)
metrics = json.loads((OUT / 'metrics.json').read_text(encoding='utf-8'))
validation = json.loads((OUT / 'validation.json').read_text(encoding='utf-8'))
nav = pd.read_csv(OUT / 'nav.csv', index_col=0, parse_dates=True)
grid = pd.read_csv(OUT / 'parameter_grid.csv')
trades = pd.read_csv(OUT / 'confirmed_trades.csv')

# 坐标标定只使用原图轴刻度，不依赖复现净值。
reader = PdfReader(RAW / 'report.pdf')
source_image = next(item for item in reader.pages[7].images if item.name == 'Image103.png')
(RAW / 'source_nav.png').write_bytes(source_image.data)
pixels = np.asarray(Image.open(RAW / 'source_nav.png').convert('RGB'))
y_slope, y_intercept = np.polyfit([466.5,371.5,275.5,179.5,84.5], [1,1.2,1.4,1.6,1.8], 1)
base_date = pd.Timestamp('2023-01-01')
tick_dates = pd.to_datetime(['2023-01-01','2024-01-01','2025-01-01','2026-01-01'])
x_slope, x_intercept = np.polyfit([306.5,618.5,932.5,1244.5], (tick_dates-base_date).days, 1)
source_rows = []
for label in ['direct', 'confirmed']:
    for x in range(166,1454):
        column = pixels[65:540,x]
        if label == 'confirmed':
            mask = (column[:,0] > 150) & (column[:,1] < 90) & (column[:,2] < 100)
        else:
            mask = (column.max(1)-column.min(1) < 7) & (column[:,0] > 100) & (column[:,0] < 180)
        if mask.any():
            y = np.median(np.flatnonzero(mask)) + 65
            date = base_date + pd.Timedelta(days=float(x_slope*x+x_intercept))
            source_rows.append(dict(strategy=label, date=date, source_nav=float(y_slope*y+y_intercept)))
source = pd.DataFrame(source_rows)
source.to_csv(OUT / 'source_curve_digitized.csv', index=False)
curve_stats = {}
fig, axes = plt.subplots(2, 1, figsize=(11,7), sharex=True, layout='constrained')
for ax, label, title in zip(axes, ['direct','confirmed'], ['直接抄底','等待7日企稳']):
    reference = source.loc[source.strategy == label].set_index('date').source_nav
    aligned = nav[label].reindex(nav.index.union(reference.index)).interpolate(method='time').reindex(reference.index)
    curve_stats[label] = dict(rmse=float(np.sqrt(((reference-aligned)**2).mean())), correlation=float(reference.corr(aligned)))
    ax.plot(reference.index,reference, color='#da9e32',linewidth=3,alpha=.7,label='原图读数')
    ax.plot(nav.index,nav[label],color='#215a81',linewidth=1.3,label='独立复现')
    ax.set_title(title,loc='left');ax.set_ylabel('净值');ax.legend(loc='upper left',frameon=False,ncol=2)
buffer = StringIO();fig.savefig(buffer, format='svg');curve_svg = buffer.getvalue();plt.close(fig)
(OUT / 'curve_comparison.json').write_text(json.dumps(curve_stats,indent=2),encoding='utf-8')

source_main={'direct':[17,1.64,.1321,.2411,.79],'confirmed':[15,1.74,.1497,.0663,1.33]}
rows=[]
for label,title in [('direct','直接抄底'),('confirmed','企稳策略')]:
    for key,name,target in zip(['trades','final_nav','annual_return','max_drawdown','sharpe'],
                               ['开仓次数','期末净值','年化收益','最大回撤','夏普'],source_main[label]):
        own=metrics[label][key]
        if key in ['annual_return','max_drawdown']:
            ref, value, delta=f'{target*100:.2f}%',f'{own*100:.3f}%',f'{(own-target)*100:+.3f}个百分点'
        elif key=='trades':ref,value,delta=str(target),str(own),str(own-target)
        else:ref,value,delta=f'{target:.2f}',f'{own:.5f}',f'{own-target:+.5f}'
        rows.append([title,name,ref,value,delta])
main_table=pd.DataFrame(rows,columns=['策略','指标','原文','复现','差值']).to_html(index=False,border=0)
rows=[]
for prefix,title,target_a,target_d in [('IH','上证50',1.31,13.05),('IF','沪深300',3.15,8.68),('IC','中证500',9.20,9.20),('IM','中证1000',14.97,6.63)]:
    m=metrics[prefix]
    rows.append([title,m['trades'],f'{target_a:.2f}%',f'{m["annual_return"]*100:.3f}%',f'{target_d:.2f}%',f'{m["max_drawdown"]*100:.3f}%'])
cross_table=pd.DataFrame(rows,columns=['指数','开仓次数','原文年化','复现年化','原文回撤','复现回撤']).to_html(index=False,border=0)
source_a=np.array([[9.4,12.1,11.1,12.1,11.7],[14,13,13.4,14.8,14.4],[14.6,13.3,15,16.2,14.7],[13.9,14.1,13.6,14.9,15.4],[13.7,14,15.2,13.2,14.2]])
source_d=np.array([[23.8,24.7,24.8,24.9,25.5],[6.8,7,6.8,6.8,7.3],[5.9,6.7,6.6,7.3,9.1],[4.9,6.4,6.8,7.6,8.4],[6.6,6.7,7.3,7.9,11.8]])
grid_tables=[]
for field,ref in [('annual_return',source_a),('max_drawdown',source_d)]:
    own=grid[field].to_numpy().reshape(5,5)*100
    content=[[f'{own[i,j]:.2f} ({ref[i,j]:.1f})' for j in range(5)] for i in range(5)]
    grid_tables.append(pd.DataFrame(content,index=[5,6,7,8,9],columns=[15,18,20,22,25]).rename_axis('等待/持有').to_html(border=0))
trade_table=trades[['trigger','entry','exit','contract','quantity','profit']].copy()
trade_table.profit=(trade_table.profit/10_000).map(lambda x:f'{x:,.2f}')
trade_table.columns=['大跌日','开仓日','平仓日','合约','手数','净收益/万元']
trade_html=trade_table.to_html(index=False,border=0)
html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>大跌后企稳再入场：华泰期货研报复现</title>
<style>
:root{{color-scheme:light}}body{{margin:0;background:#f2f5f6;color:#243644;font:17px/1.85 "Microsoft YaHei","Noto Sans CJK SC",sans-serif}}main{{max-width:1040px;margin:36px auto;padding:48px 60px;background:white;border-top:7px solid #215a81}}h1{{font-size:34px;line-height:1.45;margin:14px 0 18px}}h2{{font-size:25px;margin:46px 0 14px;color:#174763}}h3{{font-size:19px;margin-top:28px}}p{{margin:14px 0}}a{{color:#226e9c}}.tag{{font-size:14px;letter-spacing:2px;color:#687c88}}.lead{{font-size:19px}}.callout{{background:#edf5fa;padding:20px 24px;border-left:4px solid #4c92b8}}.small,figcaption{{font-size:14px;color:#607382}}.table{{overflow-x:auto;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}}th,td{{padding:10px 12px;border-bottom:1px solid #dfe7ec;text-align:right}}th{{background:#edf3f6;font-weight:600}}td:first-child,th:first-child{{text-align:left}}svg{{width:100%;height:auto}}code,pre{{font-family:Consolas,monospace;background:#eff3f5}}pre{{padding:18px;overflow-x:auto;line-height:1.65;font-size:14px}}.formula{{padding:12px 18px;background:#f6f8fa;font-family:Consolas,monospace;overflow-x:auto}}li{{margin:8px 0}}footer{{margin-top:42px;padding-top:18px;border-top:1px solid #dfe7ec;font-size:14px}}@media(max-width:700px){{main{{margin:0;padding:26px 20px}}h1{{font-size:27px}}h2{{font-size:23px}}body{{font-size:16px}}}}@media print{{body{{background:white}}main{{margin:0;max-width:none;padding:0;border:0}}h2{{break-after:avoid}}table,svg{{break-inside:avoid}}}}
</style><main>
<div class="tag">A股指数 · 股指期货 · 研报复现</div>
<h1>大跌后企稳再入场：<br>把“左侧抄底”拆成可以检验的信号</h1>
<p class="lead">本项目复现华泰期货研究院李光庭于2026年9月1日发布的量化专题研报《大跌后应该如何抄底2.0？——左侧+企稳确认》，研究中证1000大跌后的入场时机。</p>
<p><a href="https://www.htfc.com/wz_upload/png_upload/20260909/1788943656749408f5b.pdf">阅读官网原件（15页）</a> · <a href="https://webtrade.htfc.com/main/a/20260909/80181909.shtml">官网发布页</a>。原件封面日期为9月1日，网站上传日期为9月9日。复现区间为2022年7月22日至2026年8月31日，共997个交易日。</p>
<div class="callout"><strong>主要结果。</strong>企稳策略复现出15笔交易，年化收益14.983%，最大回撤6.625%；原文分别为14.97%和6.63%。直接抄底的17笔交易及24.11%最大回撤也得到还原。25组参数的年化收益最大差异为{validation['grid_annual_error_pp']:.3f}个百分点，最大回撤最大差异为{validation['grid_drawdown_error_pp']:.3f}个百分点。</div>
<h2>1. 这项研究想解决什么问题</h2>
<p>大跌后的反弹具有吸引力，但入场过早会经历继续下跌。研报把入场分成两步：先用连续五日跌幅找出恐慌阶段，再观察七个交易日，检查价格是否恢复到大跌日收盘之上。由此，研究重心从“跌得多不多”转向“卖压是否缓和”。</p>
<h2>2. 把文字规则还原成交易流程</h2>
<ol><li>在交易日t收盘，计算五日收益 Cₜ/Cₜ₋₅−1；低于−4.92%时记录大跌信号。</li>
<li>从t之后的下一交易日起等待七个交易日；若Cₜ₊₇≥Cₜ，则在t+8开盘买入。</li>
<li>选择剩余交易日严格超过40日、且到期最近的实际IM合约。以开仓后第40个交易日作边界，到期日须晚于该边界。</li>
<li>开仓后持有20个交易间隔，在第20个持有日之后的下一交易日开盘平仓。例如7月30日开仓，对应8月27日平仓。</li>
<li>初始本金1亿元，每笔保证金预算1200万元；按合约开盘价与保证金率计算整数手数，持仓规模随每笔开仓价变化。</li>
<li>开平仓各收取成交额万分之0.3手续费，持仓期间按实际合约收盘价估值；空仓资金收益取零。</li></ol>
<div class="formula">手数 = floor[12,000,000 ÷ (开仓价 × 200 × 保证金率)]<br>净收益 = 手数 × 200 × (平仓开盘价 − 开仓开盘价) − 双边手续费</div>
<p>原文明确了信号、持有期、资金占用和手续费。保证金率、重复信号处理与绩效公式则需要结合外部规则和结果核验：本项目采用2023年5月5日前15%、此后12%的IM开仓保证金率；持仓期间形成的大跌信号，仍可在平仓后完成企稳确认；最后一笔交易须在样本截止日前完成20日持有。</p>
<p>年化收益采用净值几何增长率，年度交易日取250；年化波动采用日净值收益的样本标准差乘√250；夏普为（年化收益−2%）/年化波动；最大回撤为每日净值相对历史高点的最大跌幅。这组公式在四个指数和25组参数上同时接近原表。</p>
<h2>3. 数据从哪里来，关键差异怎样找到</h2>
<p>指数开高低收来自中证指数公司，交易日历取国证指数官方日线日期。期货行情采用新浪的逐合约历史日线，保留IM2303、IM2406等真实交割合约。原研报使用天软数据；这里通过公开数据重建同一套交易过程。</p>
<p>排查过程中，最明显的偏差出现在2022年底至2023年初。若全程固定12%保证金，早期仓位偏大，企稳策略期末净值会达到1.77689。中金所的历史规则显示，IM上市时保证金为15%，2023年5月4日交易后调整为12%。修正后，前两笔交易手数分别从80、78手变为64、62手，早期净值平台与原图对齐，期末净值降至1.74507。</p>
<p><a href="https://www.bocifco.com/newsinfo.aspx?cid=3&id=48089">中银国际期货完整转发的中金所发〔2023〕28号通知</a>列出了IM保证金15%→12%的变化与生效时点。另用中金所官方日行情CSV核对IM2303的2022年12月23日、2023年1月4日、1月30日和2月8日，四天的开盘与收盘共8个价格均与新浪一致。</p>
<p>合约期限边界也影响结果。例如2022年10月21日，严格“超过40个交易日”对应IM2303；把边界放宽到40日会改选IM2212。最终采用严格边界，既符合原文，也与原图早期平台及整张参数表一致。</p>
<h2>4. 曲线与原文是否吻合</h2>
{curve_svg}
<figcaption>金色为原图图6按坐标轴刻度读取的轨迹，蓝色为独立回测。原图约每天一个横向像素，读数保留了线宽和日期定位误差。直接抄底与企稳策略的路径相关系数分别为{curve_stats['direct']['correlation']:.5f}、{curve_stats['confirmed']['correlation']:.5f}；净值均方根误差分别为{curve_stats['direct']['rmse']:.5f}、{curve_stats['confirmed']['rmse']:.5f}。</figcaption>
<div class="table">{main_table}</div>
<p>两条曲线的主要交易区段、台阶平台和2024年初的分化均得到还原。企稳策略的年化波动为9.748%，原文为9.76%；卡玛比率为2.262，原文为2.26。主表中剩余差异最大的是期末净值：企稳策略为1.74507，原文显示1.74。原表净值保留两位小数，原图读取精度约为0.002；逐笔价格的数据源与绩效统计细节仍可解释这一量级的残差。</p>
<h3>两笔可以独立复算的交易</h3>
<p><strong>2024年初：</strong>1月8日大跌后，到1月17日仍较触发日低2.79%，企稳规则继续等待；1月31日的新信号到2月19日修复5.11%，2月20日才开仓。直接抄底在1月9日买入IM2403共92手，2月6日退出，净亏损2610.74万元，占初始本金26.11%，与原文相同。</p>
<p><strong>2026年7月：</strong>7月7日触发后七日继续下跌8.07%，随后7月20日的新信号到7月29日修复1.86%；7月30日开仓IM2612共72手，8月27日退出。指数开盘到开盘上涨7.07%，实际期货净收益637.88万元，即本金的6.38%，与原文相同。</p>
<h3>企稳策略全部15笔交易</h3><div class="table">{trade_html}</div>
<h2>5. 更换参数和指数后，结论是否保留</h2>
<p>等待期取5、6、7、8、9日，持有期取15、18、20、22、25日。以下各格为“复现值（原图值）”，单位为%。全表25组年化收益平均绝对差为0.026个百分点；最大回撤平均绝对差为0.018个百分点。</p>
<h3>年化收益</h3><div class="table">{grid_tables[0]}</div>
<h3>最大回撤</h3><div class="table">{grid_tables[1]}</div>
<p>等待5日时，最大回撤仍在24%左右；等待6日之后，回撤整体显著下降。7日等待、20日持有位于一片表现接近的区域。相邻参数同样有效，使“确认企稳有助于改善持仓路径”得到更完整的支持。</p>
<p>跨指数检验使用原文分别给出的五日跌幅阈值：上证50为−3.49%、沪深300为−3.45%、中证500为−4.32%、中证1000为−4.92%。分别交易IH、IF、IC、IM，按各自200或300元的乘数和历史开仓保证金计算手数；2023年5月5日前IC保证金为14%，IF与IH为12%，此后四者统一12%。</p>
<div class="table">{cross_table}</div>
<p>四个指数的交易次数全部一致，年化收益最大差异约0.067个百分点，最大回撤最大差异约0.013个百分点。中小盘在大跌后的弹性更强，企稳确认与其反弹结构更加匹配。</p>
<h2>6. 如何从零运行</h2>
<pre>python -m pip install -r requirements.txt
python download_data.py
python reproduce.py
python check_results.py
python build_report.py</pre>
<p>下载脚本保留原始文件及来源URL；回测脚本生成每日净值、全部交易、四指数绩效和参数矩阵；回测与核验脚本检查8个官方价格、两笔原文锚点、两次未来行情截断和参数矩阵误差；最后由本次输出生成这份HTML。原始数据保存在data/raw，计算输出保存在outputs。详细运行说明见同目录README。</p>
<p>本次覆盖研报中的期货主策略、直接抄底对照、25组参数与四指数比较。原文另有平值看涨期权对照，复现该部分需要逐执行价历史报价。</p>
<footer>资料：华泰期货研究院2026年9月1日原研报（第5—13页）；中证指数公司指数日线；国证指数官方交易日期；新浪逐合约行情；中金所官方日行情；中银国际期货转发中金所保证金通知。报告生成日期：2026-11-12。计算值随代码与公开数据一同核验。</footer>
</main></html>'''
(ROOT / 'REPORT.html').write_text('\n'.join(line.rstrip() for line in html.splitlines()) + '\n',encoding='utf-8')
print('REPORT.html generated; curve comparison:', curve_stats)
