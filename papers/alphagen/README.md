# AlphaGen 复现运行说明

本目录复现 CSI300、因子池容量 10、seed 1 的实验。论文对照、预测指标和回测图见 [实验报告](REPORT.md)。

## 1. 安装环境

在 Recurrence 仓库根目录进入 `papers/alphagen`。已验证环境为 Linux/WSL、Python 3.10、PyTorch 2.7.1 和 CUDA 12.8。训练需要 CUDA GPU，回测用 CPU。

```bash
cd papers/alphagen
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

获取固定版本的作者代码，并应用本次回测使用的换仓条件修正。训练和回测从这个版本导入表达式、因子池和策略模块。

```bash
mkdir -p data
git clone https://github.com/ICT-FinD-Lab/alphagen.git data/alphagen-upstream
git -C data/alphagen-upstream checkout --detach 3f3f179f72eaeb04658eb31fd2ddffa2ab0ecb65
git -C data/alphagen-upstream apply "$PWD/qlib-swap.patch"
export PYTHONPATH="$PWD/data/alphagen-upstream"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

新开终端后先激活虚拟环境，并重新设置上面的环境变量。[作者代码的固定版本](https://github.com/ICT-FinD-Lab/alphagen/tree/3f3f179f72eaeb04658eb31fd2ddffa2ab0ecb65)

## 2. 准备数据

下载 2026-10-27 版社区 Qlib 归档。脚本从归档读取 CSI300 历史成分股、交易日和基准指数收盘价。个股行情由 BaoStock 下载并按 2023-01-15 重设前复权基准，随后转换成 Qlib 二进制格式。[社区数据发布页](https://github.com/chenditc/investment_data/releases/tag/2026-10-27)

```bash
mkdir -p data/community
curl -fL https://github.com/chenditc/investment_data/releases/download/2026-10-27/qlib_bin.tar.gz -o data/qlib_bin-2026-10-27.tar.gz
tar -xzf data/qlib_bin-2026-10-27.tar.gz --strip-components=1 -C data/community
python prepare_data.py --community-dir data/community
```

生成 `data/qlib_data/`、`data/csi300_benchmark.csv` 和 `data/baostock_csv/`。重新执行数据准备会复用已下载的单股 CSV。有同一复权口径的缓存时，可用 `--stock-csv /path/to/stock_csv` 指定。缓存文件按 `SH600000.csv` 命名，字段包含 `date,code,open,high,low,close,volume,vwap,factor`。

如果已有准备好的 Qlib 二进制目录和基准 CSV，直接在下面命令中用 `--data-dir` 和 `--benchmark-csv` 指定路径。

## 3. 训练 seed 1

```bash
python train.py --seed 1 --steps 250000 --data-dir data/qlib_data --output-dir outputs/seed1
```

程序以容量 10 从头训练，默认按完整 PPO 批次结束。运行后可在 `outputs/seed1/` 查看 `final_model.zip`、`final_pool.json`、`metrics.json` 和训练检查点。再次训练时指定新的输出目录。

## 4. 跑回测

```bash
python backtest.py --pool outputs/seed1/final_pool.json --data-dir data/qlib_data --benchmark-csv data/csi300_benchmark.csv --output-dir outputs/seed1/backtest
```

回测区间为 2020—2021 年，目标持股 50 只，每日最多替换 5 只。输出包括 `metrics.json`、`daily.csv` 和可离线打开的 `nav.html`。如需直接复算报告中的回测，将 `--pool` 改为随仓库提供的 [seed1_pool.json](seed1_pool.json)。
