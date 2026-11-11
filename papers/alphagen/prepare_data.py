import argparse
from bisect import bisect_left
from pathlib import Path
import shutil
import socket

import baostock as bs
import numpy as np
import pandas as pd

from data_collection.qlib_dump_bin import DumpDataAll


ADJUST_DATE = "2023-01-15"
PRICE_END = "2023-01-13"
PRICE_FIELDS = ["open", "high", "low", "close"]
DUMP_FIELDS = PRICE_FIELDS + ["volume", "vwap", "factor"]


def read_result(result):
    rows = []
    while result.error_code == "0" and result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock {result.error_code}: {result.error_msg}")
    return pd.DataFrame(rows, columns=result.fields)


def download_stock(symbol, output):
    code = f"{symbol[:2].lower()}.{symbol[2:]}"
    fields = ["date", *PRICE_FIELDS, "volume", "amount"]
    prices = read_result(bs.query_history_k_data_plus(
        code, ",".join(fields), start_date="1990-12-19", end_date=PRICE_END,
        frequency="d", adjustflag="2",
    )).set_index("date")
    if prices.empty:
        raise ValueError(f"No daily prices for {symbol}")
    prices = prices.replace("", "0").astype(float)

    adjustments = read_result(bs.query_adjust_factor(code, start_date="1990-12-19"))
    factors = (adjustments.drop_duplicates("dividOperateDate")
               .set_index("dividOperateDate")["foreAdjustFactor"]
               .astype(float).sort_index())
    factor_on_date = factors.reindex(prices.index, method="ffill").fillna(1.0)
    reference = factors.loc[:ADJUST_DATE]
    reference_factor = reference.iloc[-1] if len(reference) else 1.0

    # BaoStock's forward-adjusted prices use its current reference factor.
    prices[PRICE_FIELDS] /= reference_factor
    prices["factor"] = factor_on_date / reference_factor
    prices["volume"] /= prices["factor"]
    prices["vwap"] = prices["amount"] / prices["volume"]
    prices["code"] = symbol
    temporary = output.with_suffix(".tmp")
    prices.to_csv(temporary, columns=["code", *DUMP_FIELDS])
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description="Build AlphaGen's CSI300 data from BaoStock and a community Qlib archive.")
    parser.add_argument("--community-dir", type=Path, required=True, help="Extracted community Qlib archive")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--stock-csv", type=Path, help="Optional cache of stock CSVs adjusted to 2023-01-15")
    args = parser.parse_args()

    community = args.community_dir.expanduser()
    out = args.output_dir.expanduser()
    stock_csv = args.stock_csv.expanduser() if args.stock_csv else out / "baostock_csv"
    stock_csv.mkdir(parents=True, exist_ok=True)

    calendar = (community / "calendars/day.txt").read_text().splitlines()
    membership = community / "instruments/csi300.txt"
    history_start = calendar[bisect_left(calendar, "2009-01-01") - 100]
    history_end = calendar[bisect_left(calendar, "2021-12-31") + 30]
    intervals = [line.split() for line in membership.read_text().splitlines() if line.strip()]
    symbols = sorted({symbol for symbol, start, end in intervals
                      if start <= history_end and end >= history_start})
    missing = [symbol for symbol in symbols if not (stock_csv / f"{symbol}.csv").is_file()]

    if missing:
        socket.setdefaulttimeout(30)
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {login.error_msg}")
        try:
            for number, symbol in enumerate(missing, 1):
                download_stock(symbol, stock_csv / f"{symbol}.csv")
                print(f"BaoStock {number}/{len(missing)} {symbol}", flush=True)
        finally:
            bs.logout()

    qlib_dir = out / "qlib_data"
    dump = DumpDataAll(
        csv_path=str(stock_csv), qlib_dir=str(qlib_dir), max_workers=4,
        symbol_field_name="code", include_fields=",".join(DUMP_FIELDS),
    )
    dump.csv_files = [stock_csv / f"{symbol}.csv" for symbol in symbols]
    dump.dump()
    shutil.copyfile(membership, qlib_dir / "instruments/csi300.txt")
    shutil.copyfile(qlib_dir / "calendars/day.txt", qlib_dir / "calendars/day_future.txt")

    benchmark = np.fromfile(community / "features/sh000300/close.day.bin", dtype="<f4")
    offset = int(benchmark[0])
    close = pd.Series(benchmark[1:], index=calendar[offset:offset + len(benchmark) - 1], name="close")
    close.dropna().to_csv(out / "csi300_benchmark.csv", index_label="date")
    print(f"Prepared {len(symbols)} stocks in {qlib_dir}")


if __name__ == "__main__":
    main()
