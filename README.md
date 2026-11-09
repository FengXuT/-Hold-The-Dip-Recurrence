# AQR《Hold the Dip》公开数据复现

研究逢跌买入相对长期持有的风险收益表现。原文发表于 2025-12-01；本项目复现其196组参数实验，补充交易成本、执行方式、历史阶段和发布后观察。

## 先看成果

1. [完整复现流程](docs/复现流程.md)：背景、数据、公式、时间线、运行方法与结果解释。
2. [实际结果](outputs/复现结果.md) / [浏览器版](outputs/复现结果.html)：原文对照、实测表格、四张研究图。
3. [交互 Notebook](research_walkthrough.ipynb)：阅读数据和结果、检查逐笔交易。
4. [简历与面试准备](docs/简历与面试.md)。

核心实测结果：196组规则平均夏普0.168，买入持有0.254；171组落后；平均年化alpha约0.086%；Holm校正后显著组数为0。主样本1965-01-04至2025-09-30，使用价格收益和French现金利率。

## 运行

推荐 Python 3.12 或 3.13。项目核心运行依赖前四个库，Markdown用于生成HTML。Notebook可在VS Code的Jupyter环境或现有Jupyter中打开，已保留本次执行输出。

```bash
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.venv\Scripts\Activate.ps1
```

Linux / WSL 激活：

```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
python download_data.py
python run_research.py --self-check
python build_report.py
```

本次研究的原始数据快照保留在本地，Git默认忽略原始与处理后数据。首次克隆仓库后运行`download_data.py`获取数据，再运行研究脚本生成处理结果；已有数据时默认复用缓存。`--refresh` 用于主动更新数据。固定研究截止日为2026-09-30。每次重算后用`build_report.py`更新结果报告。

## 目录

- `download_data.py`：下载公开数据和保存来源清单。
- `run_research.py`：集中清洗、196组实验、统计评估、敏感性分析和四张图。
- `build_report.py`：读取结果、生成可读报告和示例交易清单。
- `data/`：原始与处理后数据，Git默认忽略。
- `outputs/`：可核对的结果表、图、研究报告。
- `docs/`：完整流程与面试材料。

## 口径

这是公开数据近似复现：Yahoo S&P500价格序列、French 1M现金基准、固定交易日周期、信号后下一交易日收盘成交、持仓中忽略新信号。原文采用Bloomberg价格和3M现金基准，部分执行细节留有实现空间。结果报告保留这些差异及其敏感性。

本次代码在Windows Python 3.13.9执行；环境版本和自检结果见`outputs/summary.json`。原文和数据来源见完整流程末尾。项目代码与报告可独立整理为个人作品，原始数据快照供本地复核使用。


