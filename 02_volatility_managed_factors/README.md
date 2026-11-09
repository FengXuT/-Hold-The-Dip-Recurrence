# 平静月与剧烈波动月：六因子的波动率管理复现

复现 Tanishk Yadav 于 **2026 年 9 月**发布的量化研究预印本 *Calm Months, Crash Months: Why Volatility Management Pays for Momentum and Not the Market*，SSRN 7113079。

- [研究主页](https://www.tanishkyadav.me/research/volatility-managed-factors)
- [原论文](https://www.tanishkyadav.me/papers/yadav-2026-calm-months-crash-months.pdf)
- [作者固定版本](https://github.com/tanishhky/Equity-Factor-Volatility-Analysis/tree/66104218e702293b6b46d94c0c14ade4fb0cf0f6)
- **[阅读结果报告 REPORT.html](REPORT.html)**：下载后用浏览器打开，图表已嵌入。

## 结果

| 指标 | 论文 | 复现 |
|---|---:|---:|
| 原始动量夏普 | 0.43 | 0.4308 |
| 管理后动量夏普，计成本 | 0.94 | 0.9374 |
| 动量 Holm 校正 p 值 | 0.023 | 0.0228 |
| 动量原始等波动最大损失/收益点 | 76 | 75.75 |
| 动量管理后等波动最大损失/收益点 | 23 | 23.09 |

Table 1 全部 54 项、Table 2 的 93 项有效数字和 Figure 1 的 36 根柱均通过核对。表格按原文展示位数验收；图形对照采用年化收益差不超过 0.05 个百分点的标准。

对作者四位小数 CSV，30 项夏普的最大差为 **0.0001768**，分组三分位收益最大差为 **0.02923 个百分点**，收益分解中年化收益项最大差为 **0.01103 个百分点**。另保留更严格的 CSV 诊断，181/183 项通过；两处更细差异及其数值见报告第 6 节。把作者版本和独立实现放到相同官方输入上运行，18 条月度仓位、净收益和组合收益序列逐点一致。

## 范围和计算口径

复现范围为原论文 Table 1、Table 2、Figure 1，以及作者项目的动量累计收益附图。原文后半部分的个股 beta 机制研究使用 CRSP 个股资料。

输入为 French 五因子和动量的官方日、月文件。固定范围为 1963-07—2026-08，120 个月预热后，正式回测为 1973-07—2026-08，共 638 个月。月收益取官方月数据；日数据只估计风险。

主规则：日收益平方的 EWMA 半衰期 21 个交易日，月末读数乘 21；次月仓位等于历史扩展窗口估计的缩放常数除以前月风险，最高 1.5 倍；每单位敞口变化计 14bp 成本。传统规则以此前月内平方离差之和替代 EWMA。实时组合再用过去 60 个月及后续累积历史估计原始与管理后因子的组合敞口，正式从 1978-07 开始。

夏普按月均值/月样本标准差乘 √12。最大损失沿用原表的累计月收益峰谷差；管理后序列先缩放至原始因子的波动率。累计曲线使用 10% 年波动缩放后的月收益累加。显著性检验为学生化圆形区块 bootstrap，12 月区块，4,999 次，种子 20260928，并做六因子 Holm 校正。

## 运行

推荐 Python 3.12 或 3.13，在本子文件夹运行：

```bash
python -m venv .venv
```

Windows PowerShell 激活：`.venv\Scripts\Activate.ps1`；Linux / WSL 激活：`source .venv/bin/activate`。

```bash
python -m pip install -r requirements.txt
python download_data.py
python run_research.py
python verify_implementation.py
python compare_results.py
python build_report.py
```

首次需要网络下载官方数据、固定版本论文和原始参考结果。后续默认使用本地快照；刷新输入使用 `python download_data.py --refresh`。参考文件锁定在作者提交 `66104218e702293b6b46d94c0c14ade4fb0cf0f6`。

French 会重建历史数据，`data/manifest.json` 记录本次输入的来源、获取时间、文件大小和 SHA-256，便于识别数据版本。当前因子文件头为 202609 CRSP 版本，研究区间固定截至 2026-08。

## 文件

| 文件 | 内容 |
|---|---|
| `download_data.py` | 官方因子输入、固定版论文、表格与原图下载 |
| `run_research.py` | 独立实现仓位、成本、组合、检验及收益分解 |
| `verify_implementation.py` | 临时下载作者固定实现，逐月交叉核对并自动清理 |
| `compare_results.py` | 直接读取印刷 LaTeX 表和参考 CSV，检验 183 项结果 |
| `build_report.py` | 从实际结果生成研究图表及内嵌图像的 HTML 报告 |
| `requirements.txt` | 依赖版本 |
| `REPORT.html` | 唯一结果报告 |

`data/` 和 `outputs/` 由脚本生成并在 Git 中忽略。主要输出包括：`monthly_paths.csv`（逐月路径）、`summary.csv`（六因子绩效）、`decomposition.csv`（12 组收益分解）、`terciles.csv`（36 根柱）、`comparison.csv/json`（逐项误差）和 `independent_crosscheck.json`（作者实现交叉核对）。

检查覆盖月数据完整性、EWMA 递推、仓位因果性、上限、费用重建、分解恒等式、Holm 调整和 bootstrap 梯度。报告中的数字和图表均从这些结果生成。
