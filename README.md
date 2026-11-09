# AQR《Hold the Dip》公开数据复现

复现AQR于2025年12月1日发布的《Hold the Dip》，检验逢跌买入相对持续持有标普500的表现。实验覆盖1965—2025年15,288个交易日，比较196组参数，并补充执行方式、交易成本、历史分段和文章发布后观察。

## 结果报告

[REPORT.html](REPORT.html) 是本仓库唯一的结果报告，包含原文对照、实测表格和四张研究图。下载后用浏览器打开，图表已嵌入文件，可离线阅读。

主实验结果：抄底规则平均夏普0.168，买入持有0.254；171/196组规则落后；平均年化alpha约0.086%；Holm校正后显著组数为0。公开数据复现的结论方向与原文相符，数值差异在报告中逐项说明。

## 文件

| 文件 | 用途 |
|---|---|
| `download_data.py` | 获取公开价格与现金利率，保存本地数据及来源记录 |
| `run_research.py` | 清洗数据、计算196组回测、回归和敏感性分析，生成中间结果及图表 |
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
python build_report.py
```

首次运行需要网络访问数据源，后续默认复用本地快照。更新数据使用`python download_data.py --refresh`，然后重新运行研究和报告脚本。固定数据截止日为2026-09-30。

## 数据与实验口径

股票数据取自Yahoo Finance的`^GSPC`收盘价；现金使用Kenneth French日频文件的RF列，百分数除以100。主样本为1965-01-01至2025-09-30，首个交易日是1965-01-04；1963—1964年的数据用于计算回看窗口。发布后观察区间为2025-12-02至2026-09-30，沿用此前仓位。

| 参数 | 取值 |
|---|---|
| 跌幅阈值 | 5%、10%、15%、20% |
| 回看期L | 5、10、15、21、63、126、252个交易日 |
| 持有期H | 21、42、63、126、252、756、1260个交易日 |

当`close[t] / close[t-L] - 1 <= -跌幅阈值`时触发信号。空仓时买入，持有H个股票日收益区间，持仓期间忽略新信号。信号形成后下一交易日收盘成交，首次股票收益记录在随后交易日；代码中仓位相对信号滞后两行。

每日策略收益为`RF + 股票仓位 × (市场收益 - RF)`，股票仓位为0或1。费用按单边换手计算，分别评估0、5、10、25bps。连续持有周期衔接时，按股票净仓位变化计费。

夏普使用日超额收益均值除以其样本标准差，再乘`sqrt(252)`。年化收益按复利净值计算。市场回归使用策略与市场的超额收益，alpha按252倍日截距年化；同时记录普通OLS和HAC(21)统计量，并对196组HAC检验进行Holm校正。

原文长样本采用价格收益，本项目按相同收益口径计算；原文的Bloomberg价格和3M现金基准，与本项目的Yahoo价格和French 1M现金基准存在数据差异。固定交易日换算、执行时点和重复信号处理为本项目明确设定，报告包含相应敏感性结果。

## 验证与本地输出

`--self-check`检查信号延迟、持有长度、重复信号处理、空仓情况、收益重建、成本单调性和历史仓位的时间因果性。报告生成时，另用触发周期独立重建示例仓位并逐日核对。

`outputs/`在本地保存196组完整指标、逐日仓位、示例日收益与净值、成本和执行敏感性、分阶段结果、运行环境及自检记录。`build_report.py`读取这些文件后更新根目录的`REPORT.html`。

## 来源

- [AQR官方文章页](https://www.aqr.com/insights/research/alternative-thinking/hold-the-dip)
- [AQR原文PDF](https://www.aqr.com/-/media/AQR/Documents/Alternative-Thinking/AQR-Alternative-Thinking---Hold-the-Dip.pdf?sc_lang=en)
- [Yahoo Finance S&P500](https://finance.yahoo.com/quote/%5EGSPC/)
- [Kenneth French官方因子说明](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html)
- [French日频数据](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip)
