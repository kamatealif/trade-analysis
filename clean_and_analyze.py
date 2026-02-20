#!/usr/bin/env python3
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FG_INPUT = BASE_DIR / "fear_greed_index.csv"
HIST_INPUT = BASE_DIR / "historical_data.csv"

FG_CLEAN_OUTPUT = BASE_DIR / "fear_greed_index_clean.csv"
HIST_CLEAN_OUTPUT = BASE_DIR / "historical_data_clean.csv"
DAILY_OUTPUT = BASE_DIR / "daily_trade_metrics.csv"
CLASS_OUTPUT = BASE_DIR / "fear_greed_class_summary.csv"
COIN_OUTPUT = BASE_DIR / "coin_summary.csv"
REPORT_OUTPUT = BASE_DIR / "analysis_report.md"


def sci_to_int_string(value: object) -> pd._libs.missing.NAType | str:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    try:
        return str(int(Decimal(text)))
    except (InvalidOperation, OverflowError, ValueError):
        return pd.NA


def clean_fear_greed(path: Path) -> pd.DataFrame:
    fg = pd.read_csv(path)
    fg = fg.rename(
        columns={
            "timestamp": "fg_timestamp",
            "value": "fg_value",
            "classification": "fg_classification",
            "date": "fg_date",
        }
    )

    fg["fg_timestamp"] = pd.to_numeric(fg["fg_timestamp"], errors="coerce").astype("Int64")
    fg["fg_date"] = pd.to_datetime(fg["fg_date"], errors="coerce")
    fg["fg_classification"] = fg["fg_classification"].astype(str).str.strip().str.title()
    fg["fg_value"] = pd.to_numeric(fg["fg_value"], errors="coerce")

    fg["fg_date_from_timestamp"] = pd.to_datetime(
        fg["fg_timestamp"], errors="coerce", unit="s", utc=True
    ).dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    fg["fg_timestamp_date_mismatch"] = fg["fg_date"].dt.normalize() != fg["fg_date_from_timestamp"]

    fg = fg.dropna(subset=["fg_date", "fg_value"]).drop_duplicates(subset=["fg_date"], keep="last")
    fg = fg.sort_values("fg_date").reset_index(drop=True)
    fg["fg_date"] = fg["fg_date"].dt.date
    fg["fg_date_from_timestamp"] = fg["fg_date_from_timestamp"].dt.date
    fg["fg_value"] = fg["fg_value"].astype(int)
    return fg


def clean_historical(path: Path) -> pd.DataFrame:
    hist = pd.read_csv(
        path,
        dtype={"Trade ID": "string", "Timestamp": "string", "Order ID": "string"},
    )
    hist = hist.rename(
        columns={
            "Account": "account",
            "Coin": "coin",
            "Execution Price": "execution_price",
            "Size Tokens": "size_tokens",
            "Size USD": "size_usd",
            "Side": "side",
            "Timestamp IST": "timestamp_ist",
            "Start Position": "start_position",
            "Direction": "direction",
            "Closed PnL": "closed_pnl",
            "Transaction Hash": "transaction_hash",
            "Order ID": "order_id",
            "Crossed": "crossed",
            "Fee": "fee",
            "Trade ID": "trade_id_raw",
            "Timestamp": "timestamp_raw",
        }
    )

    for column in ["account", "coin", "side", "direction", "transaction_hash", "order_id"]:
        hist[column] = hist[column].astype(str).str.strip()

    hist["coin"] = hist["coin"].str.upper()
    hist["side"] = hist["side"].str.upper()
    hist["direction"] = hist["direction"].str.title()

    numeric_columns = [
        "execution_price",
        "size_tokens",
        "size_usd",
        "start_position",
        "closed_pnl",
        "fee",
    ]
    for column in numeric_columns:
        hist[column] = pd.to_numeric(hist[column], errors="coerce")

    hist["timestamp_ist"] = pd.to_datetime(
        hist["timestamp_ist"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    hist["timestamp_utc"] = (
        hist["timestamp_ist"]
        .dt.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )
    hist["trade_date"] = hist["timestamp_ist"].dt.date

    hist["trade_id"] = hist["trade_id_raw"].map(sci_to_int_string).astype("string")
    hist["timestamp_ms_approx"] = hist["timestamp_raw"].map(sci_to_int_string).astype("string")

    timestamp_ms_numeric = pd.to_numeric(hist["timestamp_ms_approx"], errors="coerce")
    hist["timestamp_approx_utc"] = pd.to_datetime(
        timestamp_ms_numeric, unit="ms", utc=True, errors="coerce"
    )
    hist["timestamp_approx_ist"] = hist["timestamp_approx_utc"].dt.tz_convert("Asia/Kolkata")
    hist["timestamp_approx_day_match"] = (
        hist["timestamp_approx_ist"].dt.tz_localize(None).dt.date == hist["trade_date"]
    )
    hist["timestamp_raw_unique_values"] = hist["timestamp_raw"].nunique(dropna=True)
    hist["timestamp_raw_low_resolution"] = hist["timestamp_raw_unique_values"] <= 100

    hist = hist.dropna(subset=["timestamp_ist"]).reset_index(drop=True)
    return hist


def build_analysis(hist: pd.DataFrame, fg: pd.DataFrame) -> dict[str, pd.DataFrame | float | int]:
    merged = hist.merge(
        fg[["fg_date", "fg_value", "fg_classification"]],
        left_on="trade_date",
        right_on="fg_date",
        how="left",
    )

    daily = (
        merged.groupby("trade_date", as_index=False)
        .agg(
            trades=("coin", "size"),
            total_volume_usd=("size_usd", "sum"),
            total_fees=("fee", "sum"),
            total_closed_pnl=("closed_pnl", "sum"),
            nonzero_closed_pnl=("closed_pnl", lambda s: s[s != 0].sum()),
            unique_coins=("coin", "nunique"),
            buy_ratio=("side", lambda s: (s == "BUY").mean()),
            fg_value=("fg_value", "mean"),
        )
        .sort_values("trade_date")
    )

    class_summary = (
        merged.dropna(subset=["fg_classification"])
        .groupby("fg_classification", as_index=False)
        .agg(
            trades=("coin", "size"),
            avg_trade_usd=("size_usd", "mean"),
            total_volume_usd=("size_usd", "sum"),
            total_closed_pnl=("closed_pnl", "sum"),
            avg_fee=("fee", "mean"),
            buy_ratio=("side", lambda s: (s == "BUY").mean()),
        )
        .sort_values("trades", ascending=False)
    )

    coin_summary = (
        merged.groupby("coin", as_index=False)
        .agg(
            trades=("coin", "size"),
            total_volume_usd=("size_usd", "sum"),
            total_closed_pnl=("closed_pnl", "sum"),
            mean_trade_usd=("size_usd", "mean"),
        )
        .sort_values("total_volume_usd", ascending=False)
    )

    direction_summary = (
        merged.groupby("direction", as_index=False)
        .agg(
            trades=("direction", "size"),
            total_volume_usd=("size_usd", "sum"),
            total_closed_pnl=("closed_pnl", "sum"),
        )
        .sort_values("trades", ascending=False)
    )

    hourly_summary = (
        merged.assign(hour=merged["timestamp_ist"].dt.hour)
        .groupby("hour", as_index=False)
        .agg(
            trades=("coin", "size"),
            total_volume_usd=("size_usd", "sum"),
            total_closed_pnl=("closed_pnl", "sum"),
        )
        .sort_values("trades", ascending=False)
    )

    daily_with_fg = daily.dropna(subset=["fg_value"]).copy()
    corr_volume = daily_with_fg["fg_value"].corr(daily_with_fg["total_volume_usd"])
    corr_pnl = daily_with_fg["fg_value"].corr(daily_with_fg["nonzero_closed_pnl"])
    corr_buy_ratio = daily_with_fg["fg_value"].corr(daily_with_fg["buy_ratio"])

    quality = {
        "hist_rows": int(len(hist)),
        "matched_fg_rows": int(merged["fg_value"].notna().sum()),
        "daily_rows": int(len(daily)),
        "date_min": str(daily["trade_date"].min()),
        "date_max": str(daily["trade_date"].max()),
        "negative_fee_rows": int((hist["fee"] < 0).sum()),
        "zero_usd_rows": int((hist["size_usd"] == 0).sum()),
        "nonzero_pnl_rows": int((hist["closed_pnl"] != 0).sum()),
        "timestamp_day_match_ratio": float(hist["timestamp_approx_day_match"].mean()),
        "timestamp_raw_unique_values": int(hist["timestamp_raw"].nunique(dropna=True)),
    }

    return {
        "merged": merged,
        "daily": daily,
        "class_summary": class_summary,
        "coin_summary": coin_summary,
        "direction_summary": direction_summary,
        "hourly_summary": hourly_summary,
        "corr_volume": float(corr_volume),
        "corr_pnl": float(corr_pnl),
        "corr_buy_ratio": float(corr_buy_ratio),
        "quality": quality,
    }


def format_table(
    df: pd.DataFrame,
    columns: list[str],
    rows: int = 10,
    float_digits: int = 2,
) -> str:
    view = df.loc[:, columns].head(rows).copy()
    for column in view.select_dtypes(include=["float", "float64"]).columns:
        view[column] = view[column].round(float_digits)
    return "```text\n" + view.to_string(index=False) + "\n```"


def build_report(
    analysis: dict[str, pd.DataFrame | float | int],
    fg: pd.DataFrame,
    hist: pd.DataFrame,
) -> str:
    daily = analysis["daily"]
    class_summary = analysis["class_summary"]
    coin_summary = analysis["coin_summary"]
    direction_summary = analysis["direction_summary"]
    hourly_summary = analysis["hourly_summary"]
    quality = analysis["quality"]

    assert isinstance(daily, pd.DataFrame)
    assert isinstance(class_summary, pd.DataFrame)
    assert isinstance(coin_summary, pd.DataFrame)
    assert isinstance(direction_summary, pd.DataFrame)
    assert isinstance(hourly_summary, pd.DataFrame)
    assert isinstance(quality, dict)

    corr_volume = analysis["corr_volume"]
    corr_pnl = analysis["corr_pnl"]
    corr_buy_ratio = analysis["corr_buy_ratio"]

    assert isinstance(corr_volume, float)
    assert isinstance(corr_pnl, float)
    assert isinstance(corr_buy_ratio, float)

    fear_high = class_summary.sort_values("total_volume_usd", ascending=False).head(1)
    best_pnl_coin = coin_summary.sort_values("total_closed_pnl", ascending=False).head(1)
    worst_pnl_coin = coin_summary.sort_values("total_closed_pnl", ascending=True).head(1)

    lines: list[str] = []
    lines.append("# Dataset Cleaning and Pattern Analysis")
    lines.append("")
    lines.append("## Coverage and Data Quality")
    lines.append(f"- Fear & Greed rows (clean): **{len(fg):,}**")
    lines.append(f"- Historical rows (clean): **{len(hist):,}**")
    lines.append(
        f"- Historical rows matched to Fear & Greed date: **{quality['matched_fg_rows']:,} / {quality['hist_rows']:,}**"
    )
    lines.append(f"- Trading date window: **{quality['date_min']} to {quality['date_max']}**")
    lines.append(
        f"- `Timestamp` raw quality: only **{quality['timestamp_raw_unique_values']} unique values** "
        f"for {quality['hist_rows']:,} rows; day match vs `Timestamp IST` is **{quality['timestamp_day_match_ratio']:.2%}**"
    )
    lines.append(
        "- Cleaning rule used: treat `Timestamp IST` as authoritative event time and keep `Timestamp` as low-resolution metadata."
    )
    lines.append(
        f"- Negative fee rows (likely maker rebates): **{quality['negative_fee_rows']:,}** | "
        f"Zero-USD rows: **{quality['zero_usd_rows']:,}** | Non-zero PnL rows: **{quality['nonzero_pnl_rows']:,}**"
    )
    lines.append("")
    lines.append("## Fear & Greed Relationships")
    lines.append(f"- Correlation: Fear/Greed value vs daily total volume = **{corr_volume:.3f}**")
    lines.append(f"- Correlation: Fear/Greed value vs daily non-zero closed PnL = **{corr_pnl:.3f}**")
    lines.append(f"- Correlation: Fear/Greed value vs daily buy ratio = **{corr_buy_ratio:.3f}**")
    if not fear_high.empty:
        row = fear_high.iloc[0]
        lines.append(
            f"- Highest aggregate volume occurred in **{row['fg_classification']}** "
            f"({row['total_volume_usd']:.2f} USD across {int(row['trades']):,} trades)."
        )
    lines.append("")
    lines.append("### By Fear/Greed Classification")
    lines.append(
        format_table(
            class_summary.sort_values("trades", ascending=False),
            [
                "fg_classification",
                "trades",
                "avg_trade_usd",
                "total_volume_usd",
                "total_closed_pnl",
                "buy_ratio",
            ],
            rows=10,
        )
    )
    lines.append("")
    lines.append("## Coin Patterns")
    if not best_pnl_coin.empty and not worst_pnl_coin.empty:
        best = best_pnl_coin.iloc[0]
        worst = worst_pnl_coin.iloc[0]
        lines.append(
            f"- Best total PnL coin: **{best['coin']}** ({best['total_closed_pnl']:.2f} USD)."
        )
        lines.append(
            f"- Worst total PnL coin: **{worst['coin']}** ({worst['total_closed_pnl']:.2f} USD)."
        )
    lines.append("")
    lines.append("### Top Coins by Volume")
    lines.append(
        format_table(
            coin_summary.sort_values("total_volume_usd", ascending=False),
            ["coin", "trades", "total_volume_usd", "total_closed_pnl", "mean_trade_usd"],
            rows=15,
        )
    )
    lines.append("")
    lines.append("### Top and Bottom Coins by PnL")
    lines.append("Top 10:")
    lines.append(
        format_table(
            coin_summary.sort_values("total_closed_pnl", ascending=False),
            ["coin", "trades", "total_closed_pnl", "total_volume_usd"],
            rows=10,
        )
    )
    lines.append("")
    lines.append("Bottom 10:")
    lines.append(
        format_table(
            coin_summary.sort_values("total_closed_pnl", ascending=True),
            ["coin", "trades", "total_closed_pnl", "total_volume_usd"],
            rows=10,
        )
    )
    lines.append("")
    lines.append("## Direction and Intraday Patterns")
    lines.append("### Direction Breakdown")
    lines.append(
        format_table(
            direction_summary,
            ["direction", "trades", "total_volume_usd", "total_closed_pnl"],
            rows=12,
        )
    )
    lines.append("")
    lines.append("### Most Active IST Hours")
    lines.append(
        format_table(
            hourly_summary.sort_values("trades", ascending=False),
            ["hour", "trades", "total_volume_usd", "total_closed_pnl"],
            rows=8,
        )
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append("- `fear_greed_index_clean.csv`")
    lines.append("- `historical_data_clean.csv`")
    lines.append("- `daily_trade_metrics.csv`")
    lines.append("- `fear_greed_class_summary.csv`")
    lines.append("- `coin_summary.csv`")
    lines.append("- `analysis_report.md` (this report)")
    lines.append("")
    lines.append("## Notes")
    lines.append(
        "- `Trade ID` and `Timestamp` appear rounded in scientific notation in source data; "
        "numeric conversion preserves only approximate integer values."
    )
    lines.append(
        "- If you can export raw IDs and millisecond timestamps without scientific notation, "
        "trade-level sequencing and reconciliation will improve significantly."
    )

    return "\n".join(lines)


def main() -> None:
    fg_clean = clean_fear_greed(FG_INPUT)
    hist_clean = clean_historical(HIST_INPUT)
    analysis = build_analysis(hist_clean, fg_clean)

    fg_clean.to_csv(FG_CLEAN_OUTPUT, index=False)
    hist_clean.to_csv(HIST_CLEAN_OUTPUT, index=False)

    daily = analysis["daily"]
    class_summary = analysis["class_summary"]
    coin_summary = analysis["coin_summary"]
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(class_summary, pd.DataFrame)
    assert isinstance(coin_summary, pd.DataFrame)

    daily.to_csv(DAILY_OUTPUT, index=False)
    class_summary.to_csv(CLASS_OUTPUT, index=False)
    coin_summary.to_csv(COIN_OUTPUT, index=False)

    report = build_report(analysis, fg_clean, hist_clean)
    REPORT_OUTPUT.write_text(report, encoding="utf-8")

    print("Saved cleaned and analysis outputs:")
    for path in [
        FG_CLEAN_OUTPUT,
        HIST_CLEAN_OUTPUT,
        DAILY_OUTPUT,
        CLASS_OUTPUT,
        COIN_OUTPUT,
        REPORT_OUTPUT,
    ]:
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
