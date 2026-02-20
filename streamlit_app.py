from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Trader Behavior Dashboard", layout="wide")


BASE_DIR = Path(__file__).resolve().parent


def max_drawdown_from_pnl(pnl_series: np.ndarray) -> float:
    pnl = np.asarray(pnl_series, dtype=float)
    if len(pnl) == 0:
        return np.nan
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    return float(np.min(drawdown))


def strategy_stats(pnl_series: np.ndarray) -> dict[str, float]:
    pnl = np.asarray(pnl_series, dtype=float)
    if len(pnl) == 0:
        return {
            "days": 0,
            "total_pnl": np.nan,
            "mean_daily_pnl": np.nan,
            "std_daily_pnl": np.nan,
            "sharpe_like": np.nan,
            "max_drawdown": np.nan,
        }
    std = np.std(pnl, ddof=1) if len(pnl) > 1 else 0.0
    sharpe_like = (np.mean(pnl) / std) * np.sqrt(252) if std > 0 else np.nan
    return {
        "days": int(len(pnl)),
        "total_pnl": float(np.sum(pnl)),
        "mean_daily_pnl": float(np.mean(pnl)),
        "std_daily_pnl": float(std),
        "sharpe_like": float(sharpe_like) if pd.notna(sharpe_like) else np.nan,
        "max_drawdown": max_drawdown_from_pnl(pnl),
    }


def tail_risk_metrics(pnl_series: np.ndarray) -> dict[str, float]:
    pnl = np.asarray(pnl_series, dtype=float)
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) == 0:
        return {
            "days": 0,
            "mean_daily_pnl": np.nan,
            "std_daily_pnl": np.nan,
            "var_95": np.nan,
            "cvar_95": np.nan,
            "max_drawdown": np.nan,
        }
    var95 = np.quantile(pnl, 0.05)
    cvar95 = pnl[pnl <= var95].mean() if np.any(pnl <= var95) else var95
    std = np.std(pnl, ddof=1) if len(pnl) > 1 else 0.0
    return {
        "days": int(len(pnl)),
        "mean_daily_pnl": float(np.mean(pnl)),
        "std_daily_pnl": float(std),
        "var_95": float(var95),
        "cvar_95": float(cvar95),
        "max_drawdown": max_drawdown_from_pnl(pnl),
    }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(y_true) == 0:
        return {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan}

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if pd.notna(precision) and pd.notna(recall) and (precision + recall)
        else np.nan
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "r2_like": np.nan}

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    var = np.var(y_true)
    r2_like = float(1 - np.mean((y_true - y_pred) ** 2) / var) if var > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "r2_like": r2_like}


@st.cache_data
def load_data(base_dir: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    files = {
        "fg_clean": ("fear_greed_index_clean.csv", ["fg_date"]),
        "hist_clean": ("historical_data_clean.csv", ["timestamp_ist"]),
        "daily_metrics": ("daily_trade_metrics.csv", ["trade_date"]),
        "class_summary": ("fear_greed_class_summary.csv", None),
        "coin_summary": ("coin_summary.csv", None),
        "account_daily": ("account_daily_metrics.csv", ["trade_date"]),
        "trader_features": ("trader_features_segments.csv", None),
        "daily_market": ("daily_market_metrics.csv", ["trade_date"]),
    }

    data: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for key, (filename, date_cols) in files.items():
        path = base_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        if date_cols:
            data[key] = pd.read_csv(path, parse_dates=date_cols)
        else:
            data[key] = pd.read_csv(path)
    return data, missing


def main() -> None:
    st.title("Trader Performance vs Market Sentiment")
    st.caption("Interactive dashboard for sentiment-performance analysis on Hyperliquid.")

    data, missing = load_data(BASE_DIR)
    if missing:
        st.warning(
            "Missing files: "
            + ", ".join(missing)
            + ". Run `main.ipynb` or `python clean_and_analyze.py` first."
        )
    if not data:
        st.stop()

    fg = data.get("fg_clean", pd.DataFrame())
    daily = data.get("daily_metrics", pd.DataFrame())
    cls = data.get("class_summary", pd.DataFrame())
    coin = data.get("coin_summary", pd.DataFrame())
    hist = data.get("hist_clean", pd.DataFrame())
    account_daily = data.get("account_daily", pd.DataFrame())
    trader_features = data.get("trader_features", pd.DataFrame())

    if not daily.empty and "trade_date" in daily.columns:
        min_date = pd.to_datetime(daily["trade_date"]).min().date()
        max_date = pd.to_datetime(daily["trade_date"]).max().date()
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    else:
        date_range = None

    tabs = st.tabs(
        [
            "Overview",
            "Sentiment",
            "Coins",
            "Segment Risk",
            "Out-of-Sample Validation",
            "Archetype Clustering",
            "Predictive Baseline",
        ]
    )

    with tabs[0]:
        st.subheader("Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("FG rows", f"{len(fg):,}" if not fg.empty else "N/A")
        c2.metric("Trade rows", f"{len(hist):,}" if not hist.empty else "N/A")
        c3.metric("Daily rows", f"{len(daily):,}" if not daily.empty else "N/A")
        if not daily.empty:
            c4.metric("Total closed PnL", f"{daily['total_closed_pnl'].sum():,.2f}")
        else:
            c4.metric("Total closed PnL", "N/A")

        if not daily.empty and not fg.empty:
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

            fg_plot = fg.sort_values("fg_date").copy()
            fg_plot["fg_30d"] = fg_plot["fg_value"].rolling(30, min_periods=1).mean()
            axes[0].plot(fg_plot["fg_date"], fg_plot["fg_value"], linewidth=1, alpha=0.7, label="FG value")
            axes[0].plot(fg_plot["fg_date"], fg_plot["fg_30d"], linewidth=2, label="FG 30D MA")
            axes[0].set_title("Fear & Greed Trend")
            axes[0].legend()

            daily_plot = daily.sort_values("trade_date").copy()
            daily_plot["pnl_7d"] = daily_plot["total_closed_pnl"].rolling(7, min_periods=1).mean()
            axes[1].plot(
                daily_plot["trade_date"], daily_plot["total_closed_pnl"], alpha=0.35, label="Daily closed PnL"
            )
            axes[1].plot(daily_plot["trade_date"], daily_plot["pnl_7d"], linewidth=2, label="PnL 7D MA")
            axes[1].set_title("Daily PnL Trend")
            axes[1].legend()
            plt.tight_layout()
            st.pyplot(fig)

    with tabs[1]:
        st.subheader("Sentiment Relationships")
        if daily.empty:
            st.info("`daily_trade_metrics.csv` not found.")
        else:
            corr_df = daily.dropna(subset=["fg_value"]).copy()
            if len(corr_df) > 0:
                corr1 = corr_df["fg_value"].corr(corr_df["total_volume_usd"])
                corr2 = corr_df["fg_value"].corr(corr_df["nonzero_closed_pnl"])
                st.write(f"corr(FG, volume): **{corr1:.3f}**")
                st.write(f"corr(FG, nonzero closed PnL): **{corr2:.3f}**")

                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
                axes[0].scatter(corr_df["fg_value"], corr_df["total_volume_usd"], alpha=0.5)
                axes[0].set_title("FG vs Total Volume")
                axes[0].set_xlabel("FG value")
                axes[0].set_ylabel("Volume")

                axes[1].scatter(corr_df["fg_value"], corr_df["nonzero_closed_pnl"], alpha=0.5, color="tab:orange")
                axes[1].set_title("FG vs Nonzero Closed PnL")
                axes[1].set_xlabel("FG value")
                axes[1].set_ylabel("PnL")
                plt.tight_layout()
                st.pyplot(fig)

            if not cls.empty:
                st.write("Sentiment class summary")
                st.dataframe(cls, width="stretch")

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(cls["fg_classification"], cls["total_closed_pnl"])
                ax.set_title("Total Closed PnL by Sentiment Class")
                ax.tick_params(axis="x", rotation=20)
                st.pyplot(fig)

    with tabs[2]:
        st.subheader("Coin-Level Outcomes")
        if coin.empty:
            st.info("`coin_summary.csv` not found.")
        else:
            top_n = st.slider("Top N coins", min_value=5, max_value=30, value=12)

            top_vol = coin.sort_values("total_volume_usd", ascending=False).head(top_n)
            top_pnl = coin.sort_values("total_closed_pnl", ascending=False).head(top_n)
            bottom_pnl = coin.sort_values("total_closed_pnl", ascending=True).head(top_n)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Top volume")
                st.dataframe(top_vol[["coin", "trades", "total_volume_usd", "total_closed_pnl"]], width="stretch")
            with col2:
                st.write("Top PnL")
                st.dataframe(top_pnl[["coin", "trades", "total_closed_pnl", "total_volume_usd"]], width="stretch")
            with col3:
                st.write("Bottom PnL")
                st.dataframe(bottom_pnl[["coin", "trades", "total_closed_pnl", "total_volume_usd"]], width="stretch")

    with tabs[3]:
        st.subheader("Risk Metrics by Segment")
        if account_daily.empty or trader_features.empty:
            st.info("`account_daily_metrics.csv` or `trader_features_segments.csv` not found.")
        else:
            seg_col = st.selectbox(
                "Segment type",
                options=["segment_consistency", "segment_frequency", "segment_size"],
                index=0,
            )

            seg_daily = account_daily.merge(
                trader_features[["account", "segment_consistency", "segment_frequency", "segment_size"]],
                on="account",
                how="left",
            )

            rows = []
            for seg_value, grp in seg_daily.groupby(seg_col):
                daily_pnl = grp.groupby("trade_date")["total_pnl"].sum().values
                stats = tail_risk_metrics(daily_pnl)
                stats[seg_col] = seg_value
                rows.append(stats)

            risk_df = pd.DataFrame(rows).sort_values("mean_daily_pnl", ascending=False)
            st.dataframe(risk_df, width="stretch")

            if len(risk_df) > 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].bar(risk_df[seg_col], risk_df["mean_daily_pnl"])
                axes[0].set_title("Mean Daily PnL by Segment")
                axes[0].tick_params(axis="x", rotation=20)

                axes[1].bar(risk_df[seg_col], risk_df["cvar_95"], color="tab:red")
                axes[1].set_title("CVaR 95 by Segment")
                axes[1].tick_params(axis="x", rotation=20)
                plt.tight_layout()
                st.pyplot(fig)

    with tabs[4]:
        st.subheader("Out-of-Sample Validation (Time Split)")
        if account_daily.empty:
            st.info("`account_daily_metrics.csv` not found.")
        else:
            df = account_daily[account_daily["sentiment_bucket"].isin(["Fear", "Greed"])].copy()
            all_dates = np.array(sorted(df["trade_date"].dropna().unique()))

            if len(all_dates) < 20:
                st.info("Not enough dates for robust time-split validation.")
            else:
                split_idx = int(len(all_dates) * 0.7)
                split_idx = min(max(split_idx, 1), len(all_dates) - 1)
                split_date = all_dates[split_idx]

                train = df[df["trade_date"] <= split_date].copy()
                test = df[df["trade_date"] > split_date].copy()
                test_dates = np.array(sorted(test["trade_date"].unique()))

                train_regime = train.groupby("sentiment_bucket", as_index=False)["total_pnl"].mean()
                chosen_regime = train_regime.sort_values("total_pnl", ascending=False).iloc[0]["sentiment_bucket"]

                baseline = test.groupby("trade_date")["total_pnl"].sum().reindex(test_dates, fill_value=0.0)
                regime_only = (
                    test[test["sentiment_bucket"] == chosen_regime]
                    .groupby("trade_date")["total_pnl"]
                    .sum()
                    .reindex(test_dates, fill_value=0.0)
                )

                train_accounts = (
                    train.groupby("account", as_index=False)
                    .agg(total_pnl=("total_pnl", "sum"), wins=("wins", "sum"), pnl_nonzero=("pnl_nonzero_count", "sum"))
                )
                train_accounts["win_rate"] = np.where(
                    train_accounts["pnl_nonzero"] > 0,
                    train_accounts["wins"] / train_accounts["pnl_nonzero"],
                    np.nan,
                )
                selected_accounts = set(
                    train_accounts[(train_accounts["win_rate"] >= 0.55) & (train_accounts["total_pnl"] > 0)]["account"]
                )
                selected_only = (
                    test[test["account"].isin(selected_accounts)]
                    .groupby("trade_date")["total_pnl"]
                    .sum()
                    .reindex(test_dates, fill_value=0.0)
                )

                rows = []
                for name, series in [
                    ("Baseline (all test trades)", baseline.values),
                    (f"Regime-only ({chosen_regime})", regime_only.values),
                    ("Selected accounts only", selected_only.values),
                ]:
                    row = {"strategy": name}
                    row.update(strategy_stats(series))
                    rows.append(row)
                val_df = pd.DataFrame(rows)

                st.write(f"Split date: **{pd.to_datetime(split_date).date()}**")
                st.write(f"Chosen train regime: **{chosen_regime}**")
                st.dataframe(val_df, width="stretch")

                cum_df = pd.DataFrame(
                    {
                        "trade_date": test_dates,
                        "baseline": np.cumsum(baseline.values),
                        f"regime_{chosen_regime.lower()}": np.cumsum(regime_only.values),
                        "selected_accounts": np.cumsum(selected_only.values),
                    }
                )
                fig, ax = plt.subplots(figsize=(12, 4.5))
                for col in ["baseline", f"regime_{chosen_regime.lower()}", "selected_accounts"]:
                    ax.plot(cum_df["trade_date"], cum_df[col], label=col)
                ax.set_title("Out-of-Sample Cumulative PnL")
                ax.legend()
                st.pyplot(fig)

    with tabs[5]:
        st.subheader("Clustering: Behavioral Archetypes")
        if trader_features.empty:
            st.info("`trader_features_segments.csv` not found.")
        else:
            base_cols = [
                "total_trades",
                "active_days",
                "total_pnl",
                "avg_trade_size_usd",
                "avg_trades_per_day",
                "win_rate",
            ]
            use_cols = [c for c in base_cols if c in trader_features.columns]

            if len(use_cols) < 2:
                st.info("Not enough numeric features to cluster traders.")
            else:
                cluster_df = trader_features.copy()
                for c in use_cols:
                    cluster_df[c] = pd.to_numeric(cluster_df[c], errors="coerce")
                    cluster_df[c] = cluster_df[c].fillna(cluster_df[c].median())

                n_max = min(8, max(3, len(cluster_df)))
                n_clusters = st.slider("Number of archetypes", min_value=3, max_value=n_max, value=min(4, n_max))

                method = "fallback"
                try:
                    from sklearn.cluster import KMeans
                    from sklearn.preprocessing import StandardScaler

                    scaler = StandardScaler()
                    X = scaler.fit_transform(cluster_df[use_cols].values)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
                    cluster_df["cluster_id"] = kmeans.fit_predict(X)
                    method = "kmeans"
                except Exception:
                    # Fallback when sklearn is unavailable: activity/pnl quantile archetypes.
                    activity_q = cluster_df["avg_trades_per_day"].median() if "avg_trades_per_day" in cluster_df else 0
                    pnl_q = cluster_df["total_pnl"].median() if "total_pnl" in cluster_df else 0

                    def fallback_cluster(row: pd.Series) -> int:
                        a = int(row.get("avg_trades_per_day", 0) >= activity_q)
                        p = int(row.get("total_pnl", 0) >= pnl_q)
                        return a * 2 + p

                    cluster_df["cluster_id"] = cluster_df.apply(fallback_cluster, axis=1)

                summary = (
                    cluster_df.groupby("cluster_id", as_index=False)
                    .agg(
                        traders=("account", "nunique"),
                        mean_total_pnl=("total_pnl", "mean"),
                        mean_win_rate=("win_rate", "mean"),
                        mean_trades_per_day=("avg_trades_per_day", "mean"),
                        mean_trade_size=("avg_trade_size_usd", "mean"),
                    )
                    .sort_values("mean_total_pnl", ascending=False)
                )

                # Assign human-readable archetype labels.
                summary = summary.reset_index(drop=True)
                labels = []
                for _, row in summary.iterrows():
                    pnl_sign = "Winner" if row["mean_total_pnl"] >= 0 else "Loser"
                    activity = "HighActivity" if row["mean_trades_per_day"] >= summary["mean_trades_per_day"].median() else "LowActivity"
                    labels.append(f"{activity}_{pnl_sign}")
                summary["archetype"] = labels

                cluster_df = cluster_df.merge(summary[["cluster_id", "archetype"]], on="cluster_id", how="left")

                st.write(f"Clustering method: **{method}**")
                st.dataframe(summary[["archetype", "cluster_id", "traders", "mean_total_pnl", "mean_win_rate", "mean_trades_per_day", "mean_trade_size"]], width="stretch")

                preview_cols = ["account", "archetype"] + [c for c in use_cols if c in cluster_df.columns]
                st.dataframe(cluster_df[preview_cols].head(200), width="stretch")

                if "avg_trades_per_day" in cluster_df.columns and "total_pnl" in cluster_df.columns:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    for archetype, grp in cluster_df.groupby("archetype"):
                        ax.scatter(grp["avg_trades_per_day"], grp["total_pnl"], alpha=0.6, label=archetype)
                    ax.axhline(0, color="black", linewidth=1)
                    ax.set_xlabel("Avg Trades per Day")
                    ax.set_ylabel("Total PnL")
                    ax.set_title("Trader Archetypes: Activity vs PnL")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)

    with tabs[6]:
        st.subheader("Predictive Modeling")
        mode = st.radio(
            "Select modeling target",
            options=[
                "Market-day next-day profitability (bucket)",
                "Trader-day next-day profitability + volatility",
            ],
            horizontal=True,
        )

        if mode == "Market-day next-day profitability (bucket)":
            if daily.empty:
                st.info("`daily_trade_metrics.csv` not found.")
            else:
                model_df = daily.sort_values("trade_date").copy()
                model_df["target_next_day_positive"] = (model_df["total_closed_pnl"].shift(-1) > 0).astype(int)

                feature_cols = [
                    "fg_value",
                    "trades",
                    "total_volume_usd",
                    "buy_ratio",
                    "nonzero_closed_pnl",
                    "total_closed_pnl",
                ]
                for col in feature_cols:
                    model_df[f"{col}_lag1"] = model_df[col].shift(1)
                used_features = feature_cols + [f"{c}_lag1" for c in feature_cols]

                model_df = model_df.dropna(subset=used_features + ["target_next_day_positive"]).reset_index(drop=True)
                if len(model_df) < 50:
                    st.info("Not enough rows to train predictive baseline.")
                else:
                    split_idx = int(len(model_df) * 0.7)
                    split_idx = min(max(split_idx, 10), len(model_df) - 10)

                    train_df = model_df.iloc[:split_idx].copy()
                    test_df = model_df.iloc[split_idx:].copy()
                    X_train = train_df[used_features].values
                    y_train = train_df["target_next_day_positive"].values.astype(int)
                    X_test = test_df[used_features].values
                    y_test = test_df["target_next_day_positive"].values.astype(int)

                    majority = int(np.round(np.mean(y_train) >= 0.5))
                    y_pred_base = np.full_like(y_test, majority)
                    base_m = binary_metrics(y_test, y_pred_base)

                    out_rows = [{"model": "Majority baseline", **base_m}]

                    try:
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.pipeline import Pipeline
                        from sklearn.preprocessing import StandardScaler

                        clf = Pipeline(
                            [
                                ("scaler", StandardScaler()),
                                ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
                            ]
                        )
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        m = binary_metrics(y_test, y_pred)
                        out_rows.append({"model": "Logistic regression", **m})

                        coef = clf.named_steps["lr"].coef_[0]
                        coef_df = pd.DataFrame({"feature": used_features, "coef": coef}).sort_values("coef", ascending=False)
                        st.write("Top coefficients")
                        st.dataframe(coef_df.head(10), width="stretch")
                    except Exception as e:
                        st.info(f"scikit-learn unavailable for logistic model: {e}")

                    st.dataframe(pd.DataFrame(out_rows), width="stretch")

        else:
            if account_daily.empty:
                st.info("`account_daily_metrics.csv` not found.")
            else:
                tdf = account_daily.copy().sort_values(["account", "trade_date"]).reset_index(drop=True)
                if not daily.empty and "fg_value" in daily.columns:
                    tdf = tdf.merge(daily[["trade_date", "fg_value"]], on="trade_date", how="left")
                else:
                    tdf["fg_value"] = np.nan

                tdf["next_day_pnl"] = tdf.groupby("account")["total_pnl"].shift(-1)
                tdf["target_profit_bucket"] = (tdf["next_day_pnl"] > 0).astype(int)
                tdf["target_volatility"] = tdf["next_day_pnl"].abs()

                feature_candidates = [
                    "fg_value",
                    "trades",
                    "total_pnl",
                    "avg_trade_size_usd",
                    "total_volume_usd",
                    "win_rate",
                    "long_short_ratio",
                    "drawdown_proxy",
                    "avg_fee",
                ]
                feats = [c for c in feature_candidates if c in tdf.columns]
                if len(feats) < 3:
                    st.info("Not enough trader-day features available for this model.")
                else:
                    model_df = tdf.dropna(subset=["trade_date", "next_day_pnl"]).copy()
                    for c in feats:
                        model_df[c] = pd.to_numeric(model_df[c], errors="coerce")
                        model_df[c] = model_df[c].fillna(model_df[c].median())

                    dates = np.array(sorted(model_df["trade_date"].unique()))
                    if len(dates) < 30:
                        st.info("Not enough dates for robust trader-level time split modeling.")
                    else:
                        split_idx = int(len(dates) * 0.7)
                        split_idx = min(max(split_idx, 10), len(dates) - 10)
                        split_date = dates[split_idx]

                        train_df = model_df[model_df["trade_date"] <= split_date].copy()
                        test_df = model_df[model_df["trade_date"] > split_date].copy()

                        X_train = train_df[feats].values
                        X_test = test_df[feats].values

                        y_train_cls = train_df["target_profit_bucket"].values.astype(int)
                        y_test_cls = test_df["target_profit_bucket"].values.astype(int)
                        y_train_reg = train_df["target_volatility"].values.astype(float)
                        y_test_reg = test_df["target_volatility"].values.astype(float)

                        # Classification baselines
                        majority = int(np.round(np.mean(y_train_cls) >= 0.5))
                        pred_cls_base = np.full_like(y_test_cls, majority)
                        cls_rows = [{"model": "Majority baseline", **binary_metrics(y_test_cls, pred_cls_base)}]

                        # Regression baseline
                        mean_vol = float(np.mean(y_train_reg))
                        pred_reg_base = np.full_like(y_test_reg, mean_vol, dtype=float)
                        reg_rows = [{"model": "Mean baseline", **regression_metrics(y_test_reg, pred_reg_base)}]

                        try:
                            from sklearn.linear_model import LogisticRegression, Ridge
                            from sklearn.pipeline import Pipeline
                            from sklearn.preprocessing import StandardScaler

                            cls_model = Pipeline(
                                [
                                    ("scaler", StandardScaler()),
                                    ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
                                ]
                            )
                            cls_model.fit(X_train, y_train_cls)
                            pred_cls = cls_model.predict(X_test)
                            cls_rows.append({"model": "Logistic regression", **binary_metrics(y_test_cls, pred_cls)})

                            reg_model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
                            reg_model.fit(X_train, y_train_reg)
                            pred_reg = reg_model.predict(X_test)
                            reg_rows.append({"model": "Ridge regression", **regression_metrics(y_test_reg, pred_reg)})
                        except Exception as e:
                            st.info(f"scikit-learn unavailable for advanced trader-level models: {e}")

                        st.write(f"Trader-level time split date: **{pd.to_datetime(split_date).date()}**")
                        st.write("Profitability bucket model metrics")
                        st.dataframe(pd.DataFrame(cls_rows), width="stretch")
                        st.write("Next-day volatility model metrics")
                        st.dataframe(pd.DataFrame(reg_rows), width="stretch")


if __name__ == "__main__":
    main()
