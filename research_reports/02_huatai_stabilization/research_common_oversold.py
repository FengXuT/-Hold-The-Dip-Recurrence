"""用宽基指数日线检验中小盘共同超跌。"""
from pathlib import Path
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
CODES = {"CSI500": "000905", "CSI1000": "000852", "CSI300": "000300"}
START, END = "2015-01-01", "2026-11-06"
OOS = pd.Timestamp("2022-01-01")
LOOKBACK, SHOCK, HORIZON = 120, 5, 20
BLOCK, DRAWS = 60, 1500
GROUPS = ["zero", "low", "middle", "high"]


def load_prices():
    calendar = json.loads((RAW / "calendar.json").read_text(encoding="utf-8"))
    dates = pd.DatetimeIndex(pd.to_datetime([row[0] for row in calendar["data"]["data"]])).sort_values()
    dates = dates[(dates >= START) & (dates <= END)]
    prices = {}
    for name, code in CODES.items():
        rows = json.loads((RAW / f"{code}.json").read_text(encoding="utf-8"))["data"]
        frame = pd.DataFrame(rows)
        frame.index = pd.to_datetime(frame["tradeDate"], format="%Y%m%d")
        prices[name] = frame[["open", "close"]].astype(float).reindex(dates)
        if not ((prices[name] > 0) & np.isfinite(prices[name])).all().all():
            raise ValueError(f"指数日线有缺失或无效价格：{name}")
    return pd.concat(prices, axis=1)


def calculate_factor(returns, lookback=LOOKBACK, shock=SHOCK):
    daily = returns.to_numpy()
    cumulative = returns.rolling(shock).sum().to_numpy()
    z = np.full((len(daily), 2), np.nan)
    for t in range(lookback + shock - 1, len(daily)):
        stop = t - shock + 1
        start = stop - lookback
        history = daily[start:stop]
        market = np.column_stack([np.ones(lookback), history[:, 2]])
        intercept, beta = np.linalg.lstsq(market, history[:, :2], rcond=None)[0]

        # 只取估计期内完整的五日窗口，所有窗口使用当日这组回归系数。
        past = cumulative[start + shock - 1:stop]
        past_residual = past[:, :2] - shock * intercept - past[:, 2, None] * beta
        current = cumulative[t, :2] - shock * intercept - cumulative[t, 2] * beta
        z[t] = current / past_residual.std(axis=0, ddof=1)
    factor = pd.DataFrame(z, index=returns.index, columns=["z500", "z1000"])
    factor["own"] = (-factor["z1000"]).clip(lower=0)
    factor["common"] = (-factor[["z500", "z1000"]].max(axis=1, skipna=False)).clip(lower=0)
    return factor


def forward_returns(opens, horizon):
    period = np.log(opens.shift(-horizon - 1) / opens.shift(-1))
    return pd.DataFrame({
        "target": period["CSI1000"] - period["CSI300"],
        "target500": period["CSI500"] - period["CSI300"],
        "label_end": opens.index.to_series().shift(-horizon - 1),
    })


def rank_ic(a, b):
    ranks = pd.DataFrame({"a": a, "b": b}).rank()
    return float(ranks["a"].corr(ranks["b"]))


def fit_walk_forward(frame):
    tested = []
    for year, batch in frame.loc[OOS:].groupby(frame.loc[OOS:].index.year):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train = frame.loc[frame["label_end"] < cutoff]
        batch = batch.copy()
        for name, features in [("baseline", ["own"]), ("augmented", ["own", "common"])]:
            train_x = np.column_stack([np.ones(len(train)), train[features]])
            test_x = np.column_stack([np.ones(len(batch)), batch[features]])
            coef = np.linalg.lstsq(train_x, train["target"], rcond=None)[0]
            batch[name] = test_x @ coef
        tested.append(batch)
    return pd.concat(tested)


def group_means(frame, column, order):
    means = frame.groupby(column, observed=True)["target"].agg(["size", "mean"]).reindex(order)
    return [{"group": str(group), "n": int(row["size"]), "mean": float(row["mean"])}
            for group, row in means.dropna().iterrows()]


def statistics(frame):
    y = frame["target"]
    base_ic = rank_ic(frame["baseline"], y)
    aug_ic = rank_ic(frame["augmented"], y)
    high = frame.loc[frame["group"] == "high", "target"].mean()
    low = frame.loc[frame["group"] == "low", "target"].mean()
    return {
        "factor_ic": rank_ic(frame["common"], y),
        "own_ic": rank_ic(frame["own"], y),
        "baseline_ic": base_ic,
        "augmented_ic": aug_ic,
        "delta_ic": aug_ic - base_ic,
        "high_low": float(high - low),
    }


def bootstrap_intervals(tested):
    rng = np.random.default_rng(42)
    offsets = np.arange(BLOCK)
    block_count = (len(tested) + BLOCK - 1) // BLOCK
    results = []
    for _ in range(DRAWS):
        starts = rng.integers(0, len(tested) - BLOCK + 1, size=block_count)
        sample = (starts[:, None] + offsets).ravel()[:len(tested)]
        results.append(statistics(tested.iloc[sample]))
    return {name: list(pd.Series([row[name] for row in results]).quantile([.025, .975]))
            for name in ("factor_ic", "delta_ic", "high_low")}


def draw_chart(groups, annual):
    colors = ["#9ca3af", "#7aa5b6", "#37768b", "#144e61"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.35), layout="constrained")
    axes[0].bar([row["group"].title() for row in groups],
                [100 * row["mean"] for row in groups], color=colors)
    axes[0].set(title="Out-of-sample groups", ylabel="Mean 20-day relative log return (%)")
    years = [str(row["period"]) for row in annual]
    axes[1].bar(years, [row["ic"] for row in annual],
                color=["#b56449" if row["ic"] < 0 else "#37768b" for row in annual])
    axes[1].set(title="Factor Rank IC by year", ylabel="Spearman correlation")
    for ax in axes:
        ax.axhline(0, color="#4b5563", linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.savefig(OUT / "common_oversold.png", dpi=180)
    plt.close(fig)


def main():
    prices = load_prices()
    returns = np.log(prices.xs("close", level=1, axis=1)).diff().iloc[1:]
    opens = prices.xs("open", level=1, axis=1)
    frame = calculate_factor(returns).join(forward_returns(opens, HORIZON)).dropna()
    frame["market60"] = returns["CSI300"].rolling(60).sum()
    train = frame.loc[frame["label_end"] < OOS]
    positive = train.loc[train["common"] > 0, "common"]
    q1, q2 = positive.quantile([1 / 3, 2 / 3])
    frame["group"] = pd.cut(frame["common"], [-np.inf, 0, q1, q2, np.inf], labels=GROUPS)

    tested = fit_walk_forward(frame)
    groups = group_means(tested, "group", GROUPS)
    annual = [{
        "period": int(year), "n": len(part), "ic": rank_ic(part["common"], part["target"]),
        "delta_ic": rank_ic(part["augmented"], part["target"]) - rank_ic(part["baseline"], part["target"]),
    } for year, part in tested.groupby(tested.index.year)]
    regimes = []
    for name, part in tested.groupby(np.where(tested["market60"] >= 0, "up", "down")):
        regimes.append({"period": name, "n": len(part), "ic": rank_ic(part["common"], part["target"])})

    own_q1, own_q2 = train.loc[train["own"] > 0, "own"].quantile([1 / 3, 2 / 3])
    conditional = tested.loc[tested["own"] > 0].copy()
    conditional["own_group"] = pd.cut(conditional["own"], [0, own_q1, own_q2, np.inf],
                                     labels=["low", "middle", "high"])
    conditional["coincident"] = np.where(conditional["common"] > 0, "joint", "solo")
    matched = [{"own_group": str(name), "groups": group_means(part, "coincident", ["solo", "joint"])}
               for name, part in conditional.groupby("own_group", observed=True)]

    variants = [(120, 5, 20), (90, 5, 20), (150, 5, 20), (120, 4, 20),
                (120, 6, 20), (120, 5, 10), (120, 5, 30)]
    last_common_date = opens.index[-32]
    robust = []
    for lookback, shock, horizon in variants:
        sample = calculate_factor(returns, lookback, shock).join(forward_returns(opens, horizon))
        sample = sample.loc[OOS:last_common_date].dropna()
        robust.append({"lookback": lookback, "shock": shock, "horizon": horizon, "n": len(sample),
                       "ic": rank_ic(sample["common"], sample["target"])})

    summary = {
        "train_dates": [str(train.index[0].date()), str(train.index[-1].date())],
        "oos_dates": [str(tested.index[0].date()), str(tested.index[-1].date())],
        "train_n": len(train), "oos_n": len(tested),
        "factor_positive_n": int(tested["common"].gt(0).sum()),
        "group_thresholds": [float(q1), float(q2)],
        "own_thresholds": [float(own_q1), float(own_q2)],
        "train_ic": rank_ic(train["common"], train["target"]),
        "oos_pearson": float(tested["common"].corr(tested["target"])),
        "statistics": statistics(tested),
        "intervals": bootstrap_intervals(tested),
        "factor_vs_own_ic": rank_ic(tested["common"], tested["own"]),
        "groups": groups, "annual": annual, "regimes": regimes, "conditional": matched,
        "robustness": robust,
        "target500_ic": rank_ic(tested["common"], tested["target500"]),
        "through_2025_ic": rank_ic(tested.loc[:"2025", "common"], tested.loc[:"2025", "target"]),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "common_oversold_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tested.to_csv(OUT / "common_oversold_daily.csv", index_label="date")
    draw_chart(groups, annual)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
