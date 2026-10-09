from __future__ import annotations

import pandas as pd


def build_balance_features_per_product(
    eod: pd.DataFrame,
    prod_id_col: str = "prod_id",
    date_col: str = "ymd",
    bal_col: str = "bal_eod_eur",
) -> pd.DataFrame:
    """
    Build simple product-level features from daily end-of-day balances.

    Features:
    - n_days: number of daily observations
    - bal_mean, bal_std, bal_min, bal_max
    - neg_days: number of days with balance < 0
    - neg_ratio: share of days with balance < 0

    Parameters
    ----------
    eod:
        Daily balance dataset.
    prod_id_col:
        Product identifier column.
    date_col:
        Date column name.
    bal_col:
        Balance column name.

    Returns
    -------
    pd.DataFrame
        One row per product with engineered features.
    """
    d = eod[[prod_id_col, date_col, bal_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[prod_id_col, date_col, bal_col])

    d["is_neg"] = (d[bal_col] < 0).astype(int)

    prod_features = (
        d.groupby(prod_id_col)
        .agg(
            n_days=(bal_col, "size"),
            bal_mean=(bal_col, "mean"),
            bal_std=(bal_col, "std"),
            bal_min=(bal_col, "min"),
            bal_max=(bal_col, "max"),
            neg_days=("is_neg", "sum"),
            neg_ratio=("is_neg", "mean"),
        )
        .reset_index()
    )

    return prod_features


def aggregate_product_features_to_customer(
    prod_features: pd.DataFrame,
    cust_prod_mapping: pd.DataFrame,
    cust_id_col: str = "cust_id",
    prod_id_col: str = "prod_id",
) -> pd.DataFrame:
    """
    Map product-level features to customers and aggregate to customer level.

    Aggregation:
    - counts (n_days, neg_days): sum across products
    - balance stats / ratios: mean across products

    Parameters
    ----------
    prod_features:
        Product-level feature dataset.
    cust_prod_mapping:
        Table mapping customers to products.
    cust_id_col:
        Customer identifier column.
    prod_id_col:
        Product identifier column.

    Returns
    -------
    pd.DataFrame
        One row per customer with aggregated balance features.
    """
    merged = cust_prod_mapping.merge(prod_features, on=prod_id_col, how="left")

    cust_features = (
        merged.groupby(cust_id_col)
        .agg(
            n_days_sum=("n_days", "sum"),
            neg_days_sum=("neg_days", "sum"),
            bal_mean_mean=("bal_mean", "mean"),
            bal_std_mean=("bal_std", "mean"),
            bal_min_mean=("bal_min", "mean"),
            bal_max_mean=("bal_max", "mean"),
            neg_ratio_mean=("neg_ratio", "mean"),
        )
        .reset_index()
    )

    return cust_features
