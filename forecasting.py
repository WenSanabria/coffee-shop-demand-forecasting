"""Reusable forecasting utilities for the portfolio project.

The public repository runs on synthetic POS data. The original academic project
used private business data, which is intentionally not distributed.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from ftfy import fix_text
except ImportError:  # Lightweight fallback for environments without optional text-cleaning dependency
    def fix_text(value):
        return value

try:
    from rapidfuzz import fuzz
    def token_sort_ratio(a, b):
        return fuzz.token_sort_ratio(a, b)
except ImportError:
    from difflib import SequenceMatcher
    def token_sort_ratio(a, b):
        a = " ".join(sorted(str(a).split()))
        b = " ".join(sorted(str(b).split()))
        return SequenceMatcher(None, a, b).ratio() * 100
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

HOLIDAYS = pd.to_datetime([
    "2024-01-01", "2024-07-04", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-07-04", "2025-11-27", "2025-12-25",
])

FEATURES = [
    "day_of_week_num", "is_weekend", "month", "year",
    "lag_1", "lag_7", "roll7_mean", "avg_net_price",
    "is_holiday", "is_holiday_eve",
    "tmax", "tmin", "is_rainy", "is_snowy", "is_cold_day",
]


def repair_menu_text(value):
    if pd.isna(value):
        return value
    text = fix_text(str(value))
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.strip().strip('"').strip("'").split())


def canonical_key(value):
    if pd.isna(value):
        return value
    text = unicodedata.normalize("NFKD", str(value).lower().strip())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    return " ".join(text.split())


def clean_transactions(df: pd.DataFrame, fuzzy_threshold: int = 92) -> pd.DataFrame:
    data = df.copy()
    data = data[data["Void?"] == False].copy()  # noqa: E712
    data["Sent Date"] = pd.to_datetime(data["Sent Date"], errors="coerce", format="mixed")
    data = data.dropna(subset=["Sent Date"]).copy()
    data["date"] = data["Sent Date"].dt.floor("D")

    for col in ["Menu Item", "Menu Group", "Menu", "Sales Category"]:
        data[col] = data[col].apply(repair_menu_text)

    data["menu_item_key"] = data["Menu Item"].apply(canonical_key)
    data["menu_item_tokens_key"] = data["menu_item_key"].apply(
        lambda x: " ".join(sorted(str(x).split()))
    )

    unique_keys = sorted(data["menu_item_tokens_key"].dropna().unique())
    clusters = {}
    for key in unique_keys:
        if key in clusters:
            continue
        clusters[key] = key
        for other in unique_keys:
            if other in clusters:
                continue
            if token_sort_ratio(key, other) >= fuzzy_threshold:
                clusters[other] = key

    data["menu_item_cluster_key"] = data["menu_item_tokens_key"].map(clusters)
    reps = (
        data.groupby(["menu_item_cluster_key", "Menu Item"])
        .size().reset_index(name="n")
        .sort_values(["menu_item_cluster_key", "n"], ascending=[True, False])
    )
    best_display = (
        reps.drop_duplicates("menu_item_cluster_key")
        .set_index("menu_item_cluster_key")["Menu Item"].to_dict()
    )
    data["Menu Item Final"] = data["menu_item_cluster_key"].map(best_display)
    return data


def _seasonal_weather_proxy(dates: pd.Series) -> pd.DataFrame:
    """Deterministic Denver-like weather proxy to keep the demo self-contained."""
    date = pd.to_datetime(dates)
    doy = date.dt.dayofyear.to_numpy()
    # Approximate seasonal temperatures; deterministic by date.
    tavg = 10 + 13 * np.sin((doy - 100) / 365.25 * 2 * np.pi)
    tmax = tavg + 7
    tmin = tavg - 7
    # Deterministic precipitation flags so a clone runs without an API key/network.
    rainy = ((doy * 17) % 29 < 4).astype(int)
    snowy = ((date.dt.month.isin([11, 12, 1, 2, 3])) & (((doy * 11) % 31) < 3)).astype(int)
    return pd.DataFrame({
        "date": date.dt.floor("D"), "tavg": tavg, "tmin": tmin, "tmax": tmax,
        "is_rainy": rainy, "is_snowy": snowy, "is_cold_day": (tmax <= 5).astype(int),
    }).drop_duplicates("date")


def build_model_table(clean_df: pd.DataFrame, min_rows: int = 20, min_total_qty: int = 50) -> pd.DataFrame:
    stats = clean_df.groupby("Menu Item Final").agg(
        rows=("Menu Item Final", "size"), total_qty=("Qty", "sum")
    )
    keep = stats[(stats["rows"] >= min_rows) & (stats["total_qty"] >= min_total_qty)].index
    data = clean_df[clean_df["Menu Item Final"].isin(keep)].copy()

    daily = (
        data.groupby(["date", "Menu Item Final"], as_index=False)
        .agg(y=("Qty", "sum"), avg_net_price=("Net Price", "mean"))
        .sort_values(["Menu Item Final", "date"])
    )
    daily["day_of_week_num"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = daily["day_of_week_num"].isin([5, 6]).astype(int)
    daily["month"] = daily["date"].dt.month
    daily["year"] = daily["date"].dt.year
    daily["lag_1"] = daily.groupby("Menu Item Final")["y"].shift(1)
    daily["lag_7"] = daily.groupby("Menu Item Final")["y"].shift(7)
    daily["roll7_mean"] = daily.groupby("Menu Item Final")["y"].transform(
        lambda s: s.shift(1).rolling(7).mean()
    )
    daily["is_holiday"] = daily["date"].isin(HOLIDAYS).astype(int)
    daily["is_holiday_eve"] = daily["date"].isin(HOLIDAYS - pd.Timedelta(days=1)).astype(int)
    weather = _seasonal_weather_proxy(daily["date"])
    daily = daily.merge(weather, on="date", how="left")
    return daily.dropna(subset=["lag_1", "lag_7", "roll7_mean"]).copy()


def time_split(model_df: pd.DataFrame, train_fraction: float = 0.80):
    dates = np.sort(model_df["date"].unique())
    cutoff = dates[int(len(dates) * train_fraction)]
    train = model_df[model_df["date"] <= cutoff].copy()
    test = model_df[model_df["date"] > cutoff].copy()
    return train, test, pd.Timestamp(cutoff)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(y_true == 0, 1, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def train_and_evaluate(model_df: pd.DataFrame):
    train, test, cutoff = time_split(model_df)
    X_train, y_train = train[FEATURES], train["y"]
    X_test, y_test = test[FEATURES], test["y"]

    models = {
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    rows = [{
        "Model": "Baseline (Lag-1)",
        "MAE": mean_absolute_error(y_test, test["lag_1"]),
        "RMSE": rmse(y_test, test["lag_1"]),
        "MAPE": mape(y_test, test["lag_1"]),
    }]
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fitted[name] = model
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": rmse(y_test, pred),
            "MAPE": mape(y_test, pred),
        })
    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return fitted, results, train, test, cutoff


def build_next_day_feature_row(model_df: pd.DataFrame, menu_item: str, pred_date):
    """Build a leakage-safe one-day-ahead feature row from observed history.

    The public demo intentionally supports the next calendar day only. Multi-step
    forecasting would require recursive predictions or a different model design.
    """
    pred_date = pd.Timestamp(pred_date).floor("D")
    hist = model_df[model_df["Menu Item Final"] == menu_item].sort_values("date")
    if hist.empty:
        raise ValueError(f"No history found for item: {menu_item}")
    latest = hist["date"].max()
    expected = latest + pd.Timedelta(days=1)
    if pred_date != expected:
        raise ValueError(f"This demo supports one-day-ahead forecasts. Choose {expected.date()}.")

    # y values are observed historical demand; they are safe for next-day forecasting.
    by_date = hist.set_index("date")["y"]
    lag_1 = float(by_date.iloc[-1])
    lag_7_date = pred_date - pd.Timedelta(days=7)
    lag_7 = float(by_date.get(lag_7_date, by_date.tail(7).mean()))
    roll7 = float(by_date.tail(7).mean())
    avg_price = float(hist["avg_net_price"].iloc[-1])
    weather = _seasonal_weather_proxy(pd.Series([pred_date])).iloc[0]

    row = {
        "day_of_week_num": pred_date.dayofweek,
        "is_weekend": int(pred_date.dayofweek in [5, 6]),
        "month": pred_date.month,
        "year": pred_date.year,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "roll7_mean": roll7,
        "avg_net_price": avg_price,
        "is_holiday": int(pred_date in set(HOLIDAYS)),
        # A date is a holiday eve when tomorrow is in the holiday set.
        "is_holiday_eve": int((pred_date + pd.Timedelta(days=1)) in set(HOLIDAYS)),
        "tmax": float(weather["tmax"]),
        "tmin": float(weather["tmin"]),
        "is_rainy": int(weather["is_rainy"]),
        "is_snowy": int(weather["is_snowy"]),
        "is_cold_day": int(weather["is_cold_day"]),
    }
    return pd.DataFrame([row])[FEATURES]


def run_pipeline(data_path: str | Path):
    raw = pd.read_csv(data_path)
    clean = clean_transactions(raw)
    model_df = build_model_table(clean)
    return (*train_and_evaluate(model_df), clean, model_df)
