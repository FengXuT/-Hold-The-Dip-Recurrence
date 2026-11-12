"""检验 A 股承压日相对波动因子。"""
from argparse import ArgumentParser
import json
from pathlib import Path
from shutil import copyfileobj
from urllib.request import Request, urlopen

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "factor"
OUT = ROOT / "outputs" / "factor"
START, END = "2011-01-04", "2026-09-24"
RESEARCH = ("2012-01-01", "2018-12-31")
VALIDATION = ("2019-01-01", "2022-12-31")
OOS = ("2023-01-01", END)
HOLD = 20
MIN_CROSS_SECTION = 40
DB = DATA / "ashare_pit.duckdb"
DB_URL = "https://github.com/noblechang/a-share-data/releases/download/FREE_DATA_v0/ashare_pit.duckdb"
INDEX_URL = "https://www.csindex.com.cn/csindex-home/perf/index-perf?indexCode={}&startDate=20100101&endDate=20260924"
INDEX_FILES = {
    "000905": ROOT / "data" / "raw" / "000905.json",
    "H00905": DATA / "H00905.json",
}
BOARDS = {"main": r"^(60[0135]|00[0123])", "growth": r"^(30|688|689)"}


def load_panel(start, end, board):
    with duckdb.connect(str(DB), read_only=True) as db:
        db.execute("SET threads=4")
        dates = pd.DatetimeIndex(db.execute("""
            SELECT calendar_date FROM trade_calendar
            WHERE is_trading_day=1 AND calendar_date BETWEEN ?::DATE AND ?::DATE
            ORDER BY calendar_date
        """, [start, end]).fetchnumpy()["calendar_date"])
        bars = db.execute("""
            SELECT b.trade_date, s.symbol, b.open::FLOAT AS open, b.close::FLOAT AS close,
                   b.preclose::FLOAT AS prev, b.amount::FLOAT AS amount,
                   b.tradestatus=1 AND b.volume>0 AS active, b.isST=1 AS st
            FROM bars b JOIN security_master s USING (security_id)
            WHERE b.trade_date BETWEEN ?::DATE AND ?::DATE
              AND regexp_matches(s.symbol, ?)
              AND b.trade_date >= s.list_date
              AND (s.delist_date IS NULL OR b.trade_date <= s.delist_date)
        """, [start, end, BOARDS[board]]).df()
        delist = db.execute("SELECT symbol, delist_date FROM security_master WHERE regexp_matches(symbol, ?)",
                            [BOARDS[board]]).df().set_index("symbol").delist_date
    print(f"{board}: {len(bars):,} bars, {bars.symbol.nunique():,} stocks, {len(dates):,} sessions", flush=True)
    return bars.set_index(["trade_date", "symbol"]).unstack().reindex(dates), delist


def index_close(code, dates):
    rows = json.loads(INDEX_FILES[code].read_text(encoding="utf-8"))["data"]
    close = pd.Series({pd.Timestamp(row["tradeDate"]): float(row["close"]) for row in rows})
    return close.reindex(dates)


def prepare(panel):
    close, op, prev = (panel[k].where(panel[k].gt(0)) for k in ("close", "open", "prev"))
    active = panel["active"].eq(True) & close.notna() & op.notna() & prev.notna()
    st = panel["st"].eq(True)
    log_return = np.log(close / prev)
    # 用除权参考昨收连接相邻交易日，保留开盘和收盘的相对位置。
    log_close = log_return.fillna(0).astype("float64").cumsum()
    log_open = log_close + np.log(op / close)
    liquid = panel["amount"].where(active, 0).rolling(20).median().ge(20e6)
    eligible = active & ~st & close.ge(3) & liquid & active.rolling(60).sum().ge(57)
    return active, st, eligible, log_return.where(active), log_open, log_close


def pressure_score(excess, index, eligible, lookback=20, market_window=5):
    log_index = np.log(index)
    pressure = log_index.diff().lt(0) & log_index.diff(market_window).lt(0)
    count = pressure.rolling(lookback).sum().replace(0, np.nan)
    squared = excess.where(pressure, 0, axis=0).pow(2).rolling(lookback).sum()
    return -np.sqrt(squared.div(count, axis=0)).where(eligible), count


def forward(log_open, active, holding):
    can_label = active.shift(-1, fill_value=False) & active.shift(-1-holding, fill_value=False)
    return np.expm1(log_open.shift(-1-holding) - log_open.shift(-1)).where(can_label)


def pearson(score, target):
    x, y = np.asarray(score, dtype="float64"), np.asarray(target, dtype="float64")
    valid = np.isfinite(x) & np.isfinite(y)
    n = valid.sum(axis=1)
    x, y = np.where(valid, x, 0), np.where(valid, y, 0)
    sx, sy = x.sum(axis=1), y.sum(axis=1)
    dx = ((x*x).sum(axis=1)*n - sx*sx).clip(min=0)
    dy = ((y*y).sum(axis=1)*n - sy*sy).clip(min=0)
    den = np.sqrt(dx*dy)
    return np.divide((x*y).sum(axis=1)*n - sx*sy, den, out=np.full(len(n), np.nan),
                     where=(n >= MIN_CROSS_SECTION) & (den > 0))


def statistics(values):
    z = pd.Series(values).dropna()
    return dict(n=len(z), ic=float(z.mean()), icir=float(z.mean()/z.std()),
                positive=float(z.gt(0).mean()))


def bootstrap(values, block):
    z = pd.Series(values).dropna().to_numpy()
    rng = np.random.default_rng(261112)
    starts = rng.integers(len(z), size=(5000, (len(z)+block-1)//block))
    draws = z[((starts[:, :, None]+np.arange(block)) % len(z)).reshape(5000, -1)[:, :len(z)]]
    mean = draws.mean(axis=1)
    return dict(mean_ci=np.quantile(mean, [.025, .975]).tolist(),
                icir_ci=np.quantile(mean/draws.std(axis=1, ddof=1), [.025, .975]).tolist())


def trade_portfolio(score, panel, delist, prices, period, fee_bp=10):
    active, st, _, _, log_open, log_close = prices
    start, end = period
    dates = score.loc[start:end].index
    close_mark = log_close.where(active).ffill()
    open_mark = log_open.where(active).fillna(close_mark.shift(1)).fillna(close_mark)
    market_open = np.exp(open_mark.loc[dates].fillna(0)).to_numpy()
    market_close = np.exp(close_mark.loc[dates].fillna(0)).to_numpy()
    raw_open = np.rint(panel["open"].loc[dates].to_numpy(dtype="float64")*100)
    prev = np.round(panel["prev"].loc[dates].to_numpy(dtype="float64"), 2)
    warned = st.loc[dates].to_numpy()
    limit = np.where(warned & (dates.to_numpy()[:, None] < np.datetime64("2026-07-06")), .05, .10)
    upper = np.floor(prev*(1+limit)*100+.500001)
    lower = np.floor(prev*(1-limit)*100+.500001)
    traded = active.loc[dates].to_numpy()
    buyable, sellable = traded & (raw_open < upper) & ~warned, traded & (raw_open > lower)
    expiry = pd.DatetimeIndex(delist.reindex(score.columns)).to_numpy()
    signals = score.loc[dates].to_numpy()
    position = np.zeros((HOLD, len(score.columns)))
    pending = np.zeros_like(position, dtype=bool)
    cash = np.full(HOLD, 1/HOLD)
    fee = fee_bp/10000
    tape = [(dates[0], 1., 1., 0., 0., 0., 0, 0)]
    for i, day in enumerate(dates[1:], 1):
        position *= market_open[i]/market_open[i-1]
        position[:, day.to_datetime64() > expiry] = 0.
        sleeve = (i-1) % HOLD
        nav_open = cash.sum()+position.sum()
        stamp = .001 if day < pd.Timestamp("2023-08-28") else .0005
        signal = signals[i-1]
        candidates = np.flatnonzero(np.isfinite(signal))
        target = np.zeros(len(signal))
        if len(candidates) >= MIN_CROSS_SECTION:
            count = int(np.ceil(.20*len(candidates)))
            picks = candidates[np.argsort(-signal[candidates], kind="stable")[:count]]
            target[picks] = (cash[sleeve]+position[sleeve].sum())/count
        sold = np.where(pending & sellable[i], position, 0.)
        sold[sleeve] = np.where(sellable[i], (position[sleeve]-target).clip(min=0), 0.)
        position -= sold
        cash += sold.sum(axis=1)*(1-fee-stamp)
        pending &= position > 0
        pending[sleeve] = position[sleeve] > target+1e-12
        buy = np.where(buyable[i], (target-position[sleeve]).clip(min=0), 0.)
        if buy.sum() > 0:
            buy *= min(1., cash[sleeve]/(buy.sum()*(1+fee)))
        position[sleeve] += buy
        cash[sleeve] -= buy.sum()*(1+fee)
        nav = cash.sum() + (position*(market_close[i]/market_open[i])).sum()
        tape.append((day, nav, cash.sum()/nav, buy.sum()/nav_open, sold.sum()/nav_open,
                     (buy.sum()*fee+sold.sum()*(fee+stamp))/nav_open, len(candidates),
                     int((position.sum(axis=0)>0).sum())))
    return pd.DataFrame(tape, columns=["date", "nav", "cash_weight", "buy", "sell", "cost", "candidates",
                                       "positions"]).set_index("date")


def nav_stats(nav):
    ret = nav.pct_change().dropna()
    return dict(cagr=float(nav.iloc[-1]**(252/len(ret))-1), total=float(nav.iloc[-1]-1),
                max_drawdown=float((nav/nav.cummax()-1).min()),
                sharpe=float(ret.mean()/ret.std()*np.sqrt(252)))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    panel, delist = load_panel(START, END, "main")
    prices = prepare(panel)
    active, st, eligible, stock_return, log_open, _ = prices
    market = stock_return.where(~st).median(axis=1)
    excess = stock_return.sub(market, axis=0)
    benchmark = index_close("000905", panel.index)
    score, pressure_n = pressure_score(excess, benchmark, eligible)
    target = forward(log_open, active, HOLD)
    ic = pd.Series(pearson(score, target), index=score.index, name="ic")
    dates = score.loc[slice(*OOS)].index[:-HOLD-1]
    first, last = dates[0], dates[-1]
    test_ic = ic.reindex(dates)
    period_ic = {name: statistics(ic.loc[slice(*period)].iloc[:-HOLD-1]) for name, period in
                 (("research", RESEARCH), ("validation", VALIDATION), ("oos", OOS))}
    print("Pearson IC:", period_ic, flush=True)
    valid = (score.loc[dates].notna() & target.loc[dates].notna()).sum(axis=1)
    valid_market = pressure_n.reindex(dates).fillna(0)
    day_stats = pd.DataFrame({"pearson_ic": ic, "pressure_days": pressure_n,
                              "scored_stocks": score.notna().sum(axis=1)})
    day_stats.to_csv(OUT / "daily_ic.csv", index_label="date")
    years = {str(k): statistics(g) for k, g in test_ic.groupby(test_ic.index.year)}
    phases = {str(k): statistics(test_ic.iloc[k::HOLD]) for k in range(HOLD)}
    regime_down = benchmark.pct_change(60).reindex(dates).lt(0)
    regimes = {"csi500_60d_down": statistics(test_ic[regime_down]),
               "csi500_60d_up": statistics(test_ic[~regime_down])}

    params = []
    for lookback, market_window, h in ((15,5,20), (25,5,20), (20,3,20), (20,7,20), (20,5,15), (20,5,25)):
        s, _ = pressure_score(excess, benchmark, eligible, lookback, market_window)
        label = target.loc[dates] if h == HOLD else forward(log_open.loc[first:], active.loc[first:], h).loc[dates]
        params.append(dict(lookback=lookback, market_window=market_window, holding=h,
                           **statistics(pearson(s.loc[dates], label))))
    del s, label

    x, y = score.loc[dates], target.loc[dates]
    mean_rank = x.rank(axis=1, pct=True)
    groups = pd.DataFrame({f"Q{k}": y.where(np.ceil(mean_rank*5).eq(k)).mean(axis=1)
                           for k in range(1,6)}).where(test_ic.notna(), axis=0)
    groups["Q5-Q1"] = groups.Q5-groups.Q1
    groups.to_csv(OUT / "groups_20d.csv", index_label="signal_date")
    group_stats = dict(mean=groups.mean().to_dict(), spread_ci=bootstrap(groups["Q5-Q1"], 40)["mean_ci"])

    daily_rms = -np.sqrt(excess.pow(2).rolling(20).mean()).where(eligible)
    market_up = benchmark.pct_change().ge(0)
    rising_rms = -np.sqrt(excess.where(market_up,0,axis=0).pow(2).rolling(20).mean()).where(eligible)
    comparators = {}
    for name, baseline in (("all_day_volatility", daily_rms), ("up_day_volatility", rising_rms)):
        common = x.notna() & baseline.loc[dates].notna() & y.notna()
        fx, bx, yy = x.where(common), baseline.loc[dates].where(common), y.where(common)
        comparators[name] = dict(**statistics(pearson(bx, yy)),
                                 factor_correlation=float(np.nanmean(pearson(fx, bx))))
    bounded_y = y.clip(y.quantile(.01, axis=1), y.quantile(.99, axis=1), axis=0)
    bounded = statistics(pearson(x, bounded_y))
    for name, prefix in (("shanghai_main", "60"), ("shenzhen_main", "00")):
        cols = x.columns.str.startswith(prefix)
        regimes[name] = statistics(pearson(x.loc[:,cols], y.loc[:,cols]))

    portfolios = {}
    traces = {}
    for name, period, fee in (("validation", VALIDATION, 10), ("oos", OOS, 10),
                               ("oos_high_cost", OOS, 25)):
        port = trade_portfolio(score, panel, delist, prices, period, fee)
        for kind, code in (("csi500_tr","H00905"), ("csi500_price","000905")):
            br = index_close(code, port.index)
            port[kind] = br/br.iloc[0]
        portfolios[name] = {k: nav_stats(port[k]) for k in ("nav", "csi500_tr", "csi500_price")}
        portfolios[name].update(cash=float(port.cash_weight.iloc[1:].mean()),
                                one_way_annual=float((port.buy+port.sell).sum()/2*252/(len(port)-1)),
                                mean_positions=float(port.positions.iloc[1:].mean()),
                                start=str(port.index[0].date()), end=str(port.index[-1].date()),
                                n_sessions=len(port)-1)
        traces[name] = port
        port.to_csv(OUT / f"portfolio_{name}.csv", index_label="date")
        print(name, portfolios[name], flush=True)

    del panel, prices, active, st, eligible, stock_return, log_open, excess, target
    del score, x, y, mean_rank, daily_rms, rising_rms, baseline, common, fx, bx, yy, bounded_y
    growth, _ = load_panel("2022-01-04", END, "growth")
    a_g, _, eligible_g, ret_g, o_g, _ = prepare(growth)
    b_g = benchmark.reindex(growth.index)
    score_g, _ = pressure_score(ret_g.sub(market.reindex(growth.index), axis=0), b_g, eligible_g)
    y_g = forward(o_g, a_g, HOLD)
    regimes["growth"] = statistics(pearson(score_g.loc[dates], y_g.loc[dates]))

    summary = dict(periods=period_ic, years=years, phases=phases, regimes=regimes,
                   block40=bootstrap(test_ic,40), block60=bootstrap(test_ic,60), params=params,
                   groups=group_stats, comparators=comparators, bounded_response=bounded,
                   portfolios=portfolios, median_labelled=float(valid[test_ic.notna()].median()),
                   pressure_median=float(valid_market.median()), zero_pressure=int(valid_market.eq(0).sum()),
                   signal_start=str(first.date()), signal_end=str(last.date()))
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Summary:", json.dumps(summary, ensure_ascii=False), flush=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.7), layout="constrained")
    axes[0].plot(test_ic.rolling(60, min_periods=30).mean(), color="#165c69", lw=1.6)
    axes[0].axhline(0, color="#7c7975", lw=.8)
    axes[0].set_title("60-session rolling mean Pearson IC")
    axes[1].bar([f"Q{k}" for k in range(1,6)], groups.iloc[:,:5].mean()*100,
                color=["#9daeb3"]*4+["#147b6c"])
    axes[1].set_title("Mean 20-session return by score quintile")
    axes[1].set_ylabel("Return (%)")
    fig.savefig(OUT / "predictability.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11.5, 3.8), layout="constrained")
    port = traces["oos"]
    ax.plot(port.nav, color="#145d68", label="Strategy, 10 bp per side + sell tax")
    ax.plot(port.csi500_tr, color="#a37352", label="CSI 500 Total Return")
    ax.plot(traces["oos_high_cost"].nav, color="#7d899b", ls="--", label="Strategy, 25 bp per side + sell tax")
    ax.axhline(1, color="#9c9c9c", lw=.8)
    ax.set_ylabel("Net asset value")
    ax.legend(loc="upper left")
    fig.savefig(OUT / "portfolio.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="下载缺失的行情文件（A 股快照约 1 GB）")
    if parser.parse_args().download:
        for path, url in [(DB, DB_URL), *((path, INDEX_URL.format(code)) for code, path in INDEX_FILES.items())]:
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".download")
            req = Request(url, headers={"User-Agent":"Mozilla/5.0"})
            with urlopen(req, timeout=60) as remote, partial.open("wb") as output:
                copyfileobj(remote, output)
            partial.replace(path)
    run()
