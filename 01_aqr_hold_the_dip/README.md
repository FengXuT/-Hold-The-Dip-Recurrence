# AQR《Hold the Dip》公开数据复现

复现AQR于2025年12月1日发布的《Hold the Dip》，检验逢跌买入相对持续持有标普500的表现。主样本覆盖1965—2025年15,288个交易日，比较196组参数，并评估执行延迟、交易成本、历史阶段和文章发布后表现。

## 结果报告

[REPORT.html](REPORT.html) 是唯一结果报告，含原图对照、被动基准差异分解、实测表格和五张研究图。下载后用浏览器打开，图表已嵌入，可离线阅读。

本轮将现金改为公开13周国债报价构造的3个月月末滚动估计。平均夏普为0.195，买入持有为0.247；147/196组规则落后，比例75.0%；平均年化alpha约0.461%；普通OLS显著组数16，Holm校正后为0。

## 与原文的对照

原图196组参数的夏普、alpha和OLS t值由PDF矢量柱高提取。当前相关系数分别为0.9982、0.9990和0.9992；夏普RMSE为0.0093，年化alpha RMSE约0.0507个百分点，显著组数与原文同为16组。

被动夏普从French 1M现金口径的0.2632降至3个月现金估计下的0.2471，相对原图0.2060的差距缩小约28.1%。原文正文“落后约0.04、下降约16%”隐含基准约0.25，与图中的0.206存在差异，报告分别列示两类参照。

当前状态为部分复现：参数曲线高度接近，原图基准及相对排名仍有差距。现金期限、月末滚动和贴现计价已按公开数据对齐；具体票据选择、剩余期限估值仍采用估计，精确对齐需要原文实际ICE现金指数序列及计算说明。

## 文件

| 文件 | 用途 |
|---|---|
| `download_data.py` | 下载股价、13周国债报价、French日/月数据及原文PDF |
| `run_research.py` | 现金估值、196组日频回测、月频统计、回归与敏感性分析 |
| `compare_paper.py` | 从PDF提取196组参考值，对照曲线、基准和相对表现 |
| `audit_baseline.py` | 分解现金舍入、期限、统计频率及正文/图示差异 |
| `build_report.py` | 根据实际计算结果生成内嵌图表的报告 |
| `requirements.txt` | 固定Python依赖版本 |
| `REPORT.html` | 当前结果报告 |

运行生成的`data/`与`outputs/`保留在本地，由`.gitignore`忽略。

## 运行

推荐Python 3.12或3.13。在本子项目目录创建环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

Linux / WSL：

```bash
source .venv/bin/activate
```

随后依次执行：

```bash
python -m pip install -r requirements.txt
python download_data.py
python run_research.py --self-check
python compare_paper.py --self-check
python audit_baseline.py
python build_report.py
```

首次运行需要访问公开数据源，后续复用本地快照。更新数据使用`python download_data.py --refresh`，再依次重跑研究、对照、归因和报告脚本。固定数据截止日为2026-09-30。

## 数据与现金估计

股票取Yahoo Finance的`^GSPC`收盘价。主样本为1965-01-01至2025-09-30，首个交易日为1965-01-04；1963—1964年提供回看历史。发布后观察为2025-12-02至2026-09-30，延续之前仓位。

现金取Yahoo Finance的`^IRX`百分数报价，除以100得到年化贴现率。每月首个交易日对应的持仓，在上一交易月末建立，初始到期日为该月末后三个日历月。估值公式为：

`P = 1 - discount_yield * remaining_calendar_days / 360`

每日现金收益为相邻估值价格之比减1，月末滚动换入新票据。剩余期限采用当日3个月报价估值，即平坦曲线假设；无报价的交易日沿用最近报价。由此同时计入持有收益和利率变化引起的价格变动。

French日频RF保留为旧版对照，官方月度RF用于核查日频舍入影响。处理结果的`rf`列为本次3个月现金估计，`rf_french`为原French现金收益。

## 策略与统计

| 参数 | 取值 |
|---|---|
| 跌幅阈值 | 5%、10%、15%、20% |
| 回看期L | 5、10、15、21、63、126、252个交易日 |
| 持有期H | 21、42、63、126、252、756、1260个交易日 |

当`close[t] / close[t-L] - 1 <= -阈值`时触发信号。空仓时建仓，持有H个股票日收益区间，实际退出前忽略新信号。主复现采用信号收盘价成交近似，仓位从下一交易日收益生效；另测下一交易日收盘成交。

每日收益为`RF + 股票仓位 * (市场收益 - RF)`，仓位为0或1。费用按净仓位变化计，评估单边0、5、10、25bps。

日收益按自然月复利汇总为`prod(1+日收益)-1`。夏普采用月超额收益均值除以其样本标准差，再乘`sqrt(12)`；波动率使用月收益标准差乘`sqrt(12)`；CAGR按每日复利净值计算。市场回归使用策略和市场的月超额收益，截距乘12得到年化alpha，同时记录OLS、月频HAC(3)和196项Holm校正。

股票采用原文长样本的价格收益口径。固定交易日换算、收盘成交近似、月频统计和现金估值假设在报告中明确说明。

## 验证与输出

自检覆盖成交时序、退出前忽略信号、持有长度、月度复利、3个月票据的持有收益与价格重估、收益重建、成本单调性和历史仓位因果性。报告生成时另用触发周期重建示例仓位并核对。

阶段平均夏普与落后比例统一使用夏普有效的规则，另列全程现金组数；平均仓位覆盖全部196组。原图对照分别输出曲线形态、基准和整体对齐状态。

`outputs/`保存完整指标、逐日仓位、日收益与净值、敏感性结果、自检及运行版本。基准归因另存`baseline_cash_audit.csv`、`baseline_frequency_audit.csv`与`baseline_audit.json`。本轮报告中的数值均由这些输出生成。

## 来源

- [AQR官方文章与PDF入口](https://www.aqr.com/insights/research/alternative-thinking/hold-the-dip)
- [Yahoo S&P500](https://finance.yahoo.com/quote/%5EGSPC/)及[13周国债报价](https://finance.yahoo.com/quote/%5EIRX/)
- [TreasuryDirect国债贴现定价](https://www.treasurydirect.gov/marketable-securities/understanding-pricing/)
- [ICE 3M现金基准月末滚动定义的SEC披露](https://www.sec.gov/Archives/edgar/data/789281/000162363222000679/fist-form.htm)
- [French因子说明](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html)
- [AQR关于算术超额收益夏普的说明](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/Driving-with-the-Rear-View-Mirror.pdf)
