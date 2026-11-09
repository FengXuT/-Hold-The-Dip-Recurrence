# AQR《Hold the Dip》公开数据复现

复现AQR于2025年12月1日发布的《Hold the Dip》，检验逢跌买入相对持续持有标普500的表现。实验覆盖1965—2025年15,288个交易日，比较196组参数，并补充执行方式、交易成本、历史分段和文章发布后观察。

## 结果报告

[REPORT.html](REPORT.html) 是本仓库唯一的结果报告，包含原文对照、实测表格和五张研究图。下载后用浏览器打开，图表已嵌入文件，可离线阅读。

主实验结果：抄底规则平均夏普0.203，买入持有0.263；153/196组规则落后；平均年化alpha约0.459%；Holm校正后显著组数为0。公开数据复现的结论方向与原文相符，数值差异在报告中逐项说明。

## 与原图的吻合程度

从原文PDF第6—8页的矢量柱高提取夏普、alpha和OLS t值，按相同参数顺序比较。修正后相关系数分别为0.9942、0.9989和0.9988；夏普RMSE为0.0176，年化alpha RMSE约0.0532个百分点。普通OLS显著组数为15组，原文为16组。被动基准夏普仍为0.263对0.206，落后比例为78.1%对原图约60%；当前整体状态为部分复现。

主要修正是将原先的日频统计改为月度复利统计，并将主复现改为信号后一段收益。参数网格与价格数据保持不变。报告给出修改前后叠加图，曲线高低结构已基本吻合，夏普水平仍有差距。对照脚本分别输出曲线形态、基准和整体对齐状态，报告列出基准线、落后比例及平均夏普差。图中提取值与原文正文摘要分别列示。

## 文件

| 文件 | 用途 |
|---|---|
| `download_data.py` | 获取公开价格与现金利率，保存本地数据及来源记录 |
| `run_research.py` | 清洗数据、计算196组回测、回归和敏感性分析，生成中间结果及图表 |
| `compare_paper.py` | 从原文PDF矢量图提取196组参考值，比较修改前后的相关系数和RMSE |
| `build_report.py` | 根据计算结果生成内嵌图表的单文件报告 |
| `requirements.txt` | 固定Python依赖版本 |
| `REPORT.html` | 本次结果报告 |

运行生成的`data/`和`outputs/`留在本地，由`.gitignore`忽略。

## 运行

推荐Python 3.12或3.13，在仓库目录执行：

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

安装依赖后依次运行：

```bash
python -m pip install -r requirements.txt
python download_data.py
python run_research.py --self-check
python compare_paper.py --self-check
python build_report.py
```

首次运行需要网络访问价格、现金数据源与AQR原文PDF，后续默认复用本地快照。更新数据使用`python download_data.py --refresh`，然后重新运行研究和报告脚本。固定数据截止日为2026-09-30。

## 数据与实验口径

股票数据取自Yahoo Finance的`^GSPC`收盘价；现金使用Kenneth French日频文件的RF列，百分数除以100。主样本为1965-01-01至2025-09-30，首个交易日是1965-01-04；1963—1964年的数据用于计算回看窗口。发布后观察区间为2025-12-02至2026-09-30，沿用此前仓位。

| 参数 | 取值 |
|---|---|
| 跌幅阈值 | 5%、10%、15%、20% |
| 回看期L | 5、10、15、21、63、126、252个交易日 |
| 持有期H | 21、42、63、126、252、756、1260个交易日 |

当`close[t] / close[t-L] - 1 <= -跌幅阈值`时触发信号。空仓时买入，持有H个股票日收益区间，持仓期间忽略新信号。主复现按信号当日收盘价建仓，从下一交易日记录股票收益，即仓位相对信号滞后一行。这是理想化的收盘成交近似；敏感性分析另测下一交易日收盘成交，仓位滞后两行。

每日策略收益为`RF + 股票仓位 × (市场收益 - RF)`，股票仓位为0或1。费用按单边换手计算，分别评估0、5、10、25bps。连续持有周期衔接时，按股票净仓位变化计费。

日收益先按自然月复利汇总为`prod(1+日收益)-1`。夏普使用月超额收益均值除以其样本标准差，再乘`sqrt(12)`；年化波动使用月收益标准差乘`sqrt(12)`。年化收益按每日复利净值计算。市场回归使用策略与市场的月超额收益，alpha按12倍月截距年化；同时记录普通OLS和月频HAC(3)统计量，并对196组HAC检验进行Holm校正。

原文长样本采用价格收益，本项目按相同收益口径计算；原文的Bloomberg价格和3M现金基准，与本项目的Yahoo价格和French 1M现金基准存在数据差异。固定交易日换算、收盘成交近似、月频统计和重复信号处理为本项目明确设定。原文未明确统计频率；月频设定通过逐参数原图对照得到支持，报告包含相应敏感性结果。

## 验证与本地输出

`--self-check`检查信号延迟、实际持仓结束前忽略新信号、持有长度、重复信号处理、空仓情况、月份复利、收益重建、成本单调性和历史仓位的时间因果性。报告生成时，另用触发周期独立重建示例仓位并逐日核对。

阶段报告同时列出有效夏普规则数和全程现金组数。平均夏普与落后比例使用同一有效样本，平均股票仓位覆盖全部196组。

`outputs/`在本地保存196组完整指标、逐日仓位、示例日收益与净值、成本和执行敏感性、分阶段结果、运行环境及自检记录。`build_report.py`读取这些文件后更新根目录的`REPORT.html`。

## 来源

- [AQR官方文章页](https://www.aqr.com/insights/research/alternative-thinking/hold-the-dip)
- [AQR原文PDF](https://www.aqr.com/-/media/AQR/Documents/Alternative-Thinking/AQR-Alternative-Thinking---Hold-the-Dip.pdf?sc_lang=en)
- [Yahoo Finance S&P500](https://finance.yahoo.com/quote/%5EGSPC/)
- [Kenneth French官方因子说明](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html)
- [French日频数据](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip)
