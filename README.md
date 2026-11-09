# 量化研究复现

两个项目放在同一母文件夹中，各自保留代码、README 和一份独立结果报告。

| 项目 | 原文日期 | 研究内容 | 结果报告 |
|---|---|---|---|
| [01_aqr_hold_the_dip](01_aqr_hold_the_dip/) | 2025-12-01 | 196 组逢跌买入规则；参数曲线高度接近，被动基准仍有差距 | [AQR 报告](01_aqr_hold_the_dip/REPORT.html) |
| [02_volatility_managed_factors](02_volatility_managed_factors/) | 2026 年 9 月 | 六因子波动率管理；两张主表与三分位图共 183 项达到论文展示精度及图形误差标准 | [新复现报告](02_volatility_managed_factors/REPORT.html) |

新项目复现 Tanishk Yadav 的 *Calm Months, Crash Months: Why Volatility Management Pays for Momentum and Not the Market*。动量夏普为 **0.4308 → 0.9374**，原文为 **0.43 → 0.94**；校正 p 值为 **0.0228**，原文为 **0.023**。按相同官方输入逐月交叉计算，独立实现和作者固定版本的 18 条仓位与收益序列完全一致。

进入对应子文件夹，按各自 README 运行。两份 `REPORT.html` 均已嵌入图表，下载后用浏览器打开即可离线阅读。原始输入和运行输出由脚本生成，保留在本地 `data/`、`outputs/`。
