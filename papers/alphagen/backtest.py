import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import qlib
import torch
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis

from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import StockData
from alphagen_qlib.strategy import TopKSwapNStrategy
from alphagen_qlib.utils import load_alpha_pool_by_path


def main():
    parser = argparse.ArgumentParser(description="Run AlphaGen's CSI300 Top 50 / Drop 5 backtest.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/qlib_data"))
    parser.add_argument("--benchmark-csv", type=Path, default=Path("data/csi300_benchmark.csv"))
    parser.add_argument("--pool", type=Path, required=True, help="Factor pool saved by a trusted AlphaGen run")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/seed1/backtest"))
    args = parser.parse_args()

    torch.set_num_threads(1)
    qlib.init(provider_uri=str(args.data_dir.expanduser().resolve()), region="cn", kernels=1)
    StockData._qlib_initialized = True
    exprs, weights = load_alpha_pool_by_path(str(args.pool.expanduser()))
    stocks = StockData("csi300", "2020-01-01", "2021-12-31",
                       max_future_days=0, device=torch.device("cpu"))
    calculator = QLibStockDataCalculator(stocks, None)
    signal = stocks.make_dataframe(calculator.make_ensemble_alpha(exprs, weights), ["score"])["score"]
    signal = signal.sort_index()
    dates = signal.index.get_level_values(0)
    index = pd.read_csv(args.benchmark_csv, parse_dates=["date"], index_col="date")
    benchmark = index["close"].pct_change(fill_method=None)

    portfolios, _ = backtest(
        strategy=TopKSwapNStrategy(K=50, n_swap=5, min_hold_days=1, only_tradable=True, signal=signal),
        executor=executor.SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True),
        start_time=dates.min(), end_time=dates.max(), account=100_000_000, benchmark=benchmark,
        exchange_kwargs={
            "freq": "day", "deal_price": "close", "limit_threshold": 0.095,
            "open_cost": 0.0015, "close_cost": 0.0015, "min_cost": 5,
        },
    )
    daily, _ = portfolios["1day"]
    net = daily["return"] - daily["cost"]
    strategy_risk = risk_analysis(net)["risk"]
    excess_risk = risk_analysis(net - daily["bench"])["risk"]
    metrics = {
        "annual_return": float(strategy_risk["annualized_return"]),
        "sharpe": float(strategy_risk["information_ratio"]),
        "max_drawdown": float(strategy_risk["max_drawdown"]),
        "annual_excess_return": float(excess_risk["annualized_return"]),
        "information_ratio": float(excess_risk["information_ratio"]),
        "excess_max_drawdown": float(excess_risk["max_drawdown"]),
    }

    out = args.output_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / "daily.csv")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    fig = go.Figure([
        go.Scatter(x=daily.index, y=(1 + net).cumprod(), name="AlphaGen after costs"),
        go.Scatter(x=daily.index, y=(1 + daily["bench"]).cumprod(), name="CSI 300"),
    ])
    fig.update_layout(title="AlphaGen and CSI 300", yaxis_title="Net asset value", template="plotly_white")
    fig.write_html(out / "nav.html", include_plotlyjs=True)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
