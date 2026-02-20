from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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


def run_market_day_profitability_model(
    daily_df: pd.DataFrame, train_ratio: float = 0.7
) -> dict[str, Any]:
    model_df = daily_df.sort_values("trade_date").copy()
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
        return {"status": "insufficient_data", "message": "Not enough rows to train predictive baseline."}

    split_idx = int(len(model_df) * train_ratio)
    split_idx = min(max(split_idx, 10), len(model_df) - 10)
    train_df = model_df.iloc[:split_idx].copy()
    test_df = model_df.iloc[split_idx:].copy()

    X_train = train_df[used_features].values
    y_train = train_df["target_next_day_positive"].values.astype(int)
    X_test = test_df[used_features].values
    y_test = test_df["target_next_day_positive"].values.astype(int)

    majority = int(np.round(np.mean(y_train) >= 0.5))
    y_pred_base = np.full_like(y_test, majority)
    out_rows = [{"model": "Majority baseline", **binary_metrics(y_test, y_pred_base)}]
    coef_df = None

    sklearn_error = None
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
        out_rows.append({"model": "Logistic regression", **binary_metrics(y_test, y_pred)})

        coef = clf.named_steps["lr"].coef_[0]
        coef_df = pd.DataFrame({"feature": used_features, "coef": coef}).sort_values("coef", ascending=False)
    except Exception as e:
        sklearn_error = str(e)

    return {
        "status": "ok",
        "metrics_df": pd.DataFrame(out_rows),
        "coef_df": coef_df,
        "split_date": test_df["trade_date"].min() if len(test_df) else None,
        "sklearn_error": sklearn_error,
    }


def run_trader_day_models(
    account_daily_df: pd.DataFrame,
    daily_metrics_df: pd.DataFrame | None = None,
    train_ratio: float = 0.7,
) -> dict[str, Any]:
    pred_df = account_daily_df.copy().sort_values(["account", "trade_date"]).reset_index(drop=True)
    if daily_metrics_df is not None and "fg_value" in daily_metrics_df.columns:
        pred_df = pred_df.merge(daily_metrics_df[["trade_date", "fg_value"]], on="trade_date", how="left")
    else:
        pred_df["fg_value"] = np.nan

    pred_df["next_day_pnl"] = pred_df.groupby("account")["total_pnl"].shift(-1)
    pred_df["target_profit_bucket"] = (pred_df["next_day_pnl"] > 0).astype(int)
    pred_df["target_volatility"] = pred_df["next_day_pnl"].abs()

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
    model_features = [c for c in feature_candidates if c in pred_df.columns]
    if len(model_features) < 3:
        return {"status": "insufficient_data", "message": "Not enough trader-day features available."}

    for c in model_features:
        pred_df[c] = pd.to_numeric(pred_df[c], errors="coerce")
        pred_df[c] = pred_df[c].fillna(pred_df[c].median())

    model_df = pred_df.dropna(subset=["trade_date", "next_day_pnl"]).copy()
    if len(model_df) < 100:
        return {"status": "insufficient_data", "message": "Not enough trader-day rows for robust modeling."}

    dates = np.array(sorted(model_df["trade_date"].unique()))
    if len(dates) < 30:
        return {"status": "insufficient_data", "message": "Not enough unique dates for time-split modeling."}

    split_idx = int(len(dates) * train_ratio)
    split_idx = min(max(split_idx, 10), len(dates) - 10)
    split_date = dates[split_idx]

    train_df = model_df[model_df["trade_date"] <= split_date].copy()
    test_df = model_df[model_df["trade_date"] > split_date].copy()

    X_train = train_df[model_features].values
    X_test = test_df[model_features].values

    y_train_cls = train_df["target_profit_bucket"].values.astype(int)
    y_test_cls = test_df["target_profit_bucket"].values.astype(int)
    y_train_reg = train_df["target_volatility"].values.astype(float)
    y_test_reg = test_df["target_volatility"].values.astype(float)

    majority = int(np.round(np.mean(y_train_cls) >= 0.5))
    pred_cls_base = np.full_like(y_test_cls, majority)
    cls_rows = [{"model": "Majority baseline", **binary_metrics(y_test_cls, pred_cls_base)}]

    mean_vol = float(np.mean(y_train_reg))
    pred_reg_base = np.full_like(y_test_reg, mean_vol, dtype=float)
    reg_rows = [{"model": "Mean baseline", **regression_metrics(y_test_reg, pred_reg_base)}]

    coef_cls_df = None
    coef_reg_df = None
    sklearn_error = None

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

        coef_cls_df = pd.DataFrame(
            {"feature": model_features, "coef": cls_model.named_steps["lr"].coef_[0]}
        ).sort_values("coef", ascending=False)
        coef_reg_df = pd.DataFrame(
            {"feature": model_features, "coef": reg_model.named_steps["ridge"].coef_}
        ).sort_values("coef", ascending=False)
    except Exception as e:
        sklearn_error = str(e)

    return {
        "status": "ok",
        "cls_metrics_df": pd.DataFrame(cls_rows),
        "reg_metrics_df": pd.DataFrame(reg_rows),
        "coef_cls_df": coef_cls_df,
        "coef_reg_df": coef_reg_df,
        "split_date": split_date,
        "sklearn_error": sklearn_error,
    }


def cluster_traders(trader_features_df: pd.DataFrame, n_clusters: int = 4) -> dict[str, Any]:
    cluster_df = trader_features_df.copy()
    cluster_features = [
        "total_trades",
        "active_days",
        "total_pnl",
        "avg_trade_size_usd",
        "avg_trades_per_day",
        "win_rate",
    ]
    cluster_features = [c for c in cluster_features if c in cluster_df.columns]

    if len(cluster_features) < 2:
        return {"status": "insufficient_data", "message": "Not enough features for clustering."}

    for c in cluster_features:
        cluster_df[c] = pd.to_numeric(cluster_df[c], errors="coerce")
        cluster_df[c] = cluster_df[c].fillna(cluster_df[c].median())

    method = "fallback"
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(cluster_df[cluster_features].values)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        cluster_df["cluster_id"] = km.fit_predict(X)
        method = "kmeans"
    except Exception:
        activity_med = cluster_df["avg_trades_per_day"].median() if "avg_trades_per_day" in cluster_df else 0
        pnl_med = cluster_df["total_pnl"].median() if "total_pnl" in cluster_df else 0

        def fallback_cluster(row: pd.Series) -> int:
            a = int(row.get("avg_trades_per_day", 0) >= activity_med)
            p = int(row.get("total_pnl", 0) >= pnl_med)
            return a * 2 + p

        cluster_df["cluster_id"] = cluster_df.apply(fallback_cluster, axis=1)

    summary = (
        cluster_df.groupby("cluster_id", as_index=False)
        .agg(
            traders=("account", "nunique"),
            mean_total_pnl=("total_pnl", "mean"),
            median_total_pnl=("total_pnl", "median"),
            mean_win_rate=("win_rate", "mean"),
            mean_trades_per_day=("avg_trades_per_day", "mean"),
            mean_trade_size=("avg_trade_size_usd", "mean"),
        )
        .sort_values("mean_total_pnl", ascending=False)
        .reset_index(drop=True)
    )

    names = []
    activity_med = summary["mean_trades_per_day"].median()
    for _, row in summary.iterrows():
        act = "HighActivity" if row["mean_trades_per_day"] >= activity_med else "LowActivity"
        pnl = "Winner" if row["mean_total_pnl"] >= 0 else "Loser"
        names.append(f"{act}_{pnl}")
    summary["archetype"] = names

    cluster_df = cluster_df.merge(summary[["cluster_id", "archetype"]], on="cluster_id", how="left")
    return {
        "status": "ok",
        "method": method,
        "cluster_df": cluster_df,
        "summary_df": summary,
        "features": cluster_features,
    }
