# 量化研报复现

两个项目统一放在本仓库中，各自包含复现代码、README和结果报告。

| 项目 | 原文日期 | 研究内容 | 结果报告 |
|---|---|---|---|
| [01_aqr_hold_the_dip](01_aqr_hold_the_dip/) | 2025-12-01 | AQR逢跌买入研究；参数曲线接近，被动基准仍有差距 | [结果报告](01_aqr_hold_the_dip/REPORT.html) |
| [02_huatai_stabilization](02_huatai_stabilization/) | 2026-09-01 | 华泰期货正式研报：大跌后企稳确认、真实股指期货交易及连续因子扩展；25组参数年化最大差异0.05个百分点 | [结果报告](02_huatai_stabilization/REPORT.html) |

新项目优先从持续计算的价格、波动与期货基差信号开展研究。完整规则、数据来源、逐笔交易和原文对照见各项目报告；运行命令见各目录README。

基于第二份研报的[自主双因子回测](02_huatai_stabilization/factor_study/README.md)及[独立报告](02_huatai_stabilization/factor_study/REPORT.html)已经完成，研究下行压力衰减与新低停止对交易时机的影响，并已完成信息时点修正与历史时间留出检验。
