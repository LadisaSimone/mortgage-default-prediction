from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def print_basic_info(df: pd.DataFrame, name: str) -> None:
    """
    Print basic shape and missingness information for a DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame.
    name:
        Human-readable dataset name for logging.
    """
    print(f"=== {name} ===")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    na_rate = df.isna().mean().sort_values(ascending=False)
    print("top missingness:")
    print(na_rate.head(10))
    print()


def plot_side_by_side_percent(
    left: pd.Series,
    right: pd.Series,
    title_left: str,
    title_right: str,
    suptitle: str,
    order: Optional[Sequence[str]] = None,
    rotation: int = 45,
) -> None:
    """
    Plot two percentage distributions side-by-side for easy comparison.

    Parameters
    ----------
    left:
        Categorical series for the full population.
    right:
        Categorical series for the subset (e.g., defaulters).
    title_left:
        Title for the left plot.
    title_right:
        Title for the right plot.
    suptitle:
        Overall figure title.
    order:
        Optional explicit category order.
    rotation:
        X-axis label rotation.
    """
    left_counts = left.value_counts(normalize=True, dropna=False) * 100
    right_counts = right.value_counts(normalize=True, dropna=False) * 100

    if order is None:
        categories = sorted(set(left_counts.index).union(set(right_counts.index)), key=str)
    else:
        categories = list(order)

    left_vals = [float(left_counts.get(cat, 0.0)) for cat in categories]
    right_vals = [float(right_counts.get(cat, 0.0)) for cat in categories]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    axes[0].bar(categories, left_vals)
    axes[0].set_title(title_left)
    axes[0].set_ylabel("Percentage (%)")
    axes[0].tick_params(axis="x", rotation=rotation)

    axes[1].bar(categories, right_vals)
    axes[1].set_title(title_right)
    axes[1].tick_params(axis="x", rotation=rotation)

    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def default_rate_by_group(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = "target",
) -> pd.DataFrame:
    """
    Compute default rate (mean target) by a categorical grouping column.

    Parameters
    ----------
    df:
        Customer-level dataset containing group_col and target_col.
    group_col:
        Column used to group customers (e.g., age_group).
    target_col:
        Binary target column where 1 indicates default.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - group_col
        - n_customers
        - default_rate
    """
    out = (
        df.groupby(group_col)[target_col]
        .agg(n_customers="size", default_rate="mean")
        .reset_index()
        .sort_values(group_col)
    )
    return out


def plot_default_rate_bar(
    rate_df: pd.DataFrame,
    group_col: str,
    rate_col: str = "default_rate",
    title: str = "Default rate by group",
    rotation: int = 45,
) -> None:
    """
    Plot default rate by group as a bar chart.

    Parameters
    ----------
    rate_df:
        Output of default_rate_by_group.
    group_col:
        Grouping column name.
    rate_col:
        Default rate column name.
    title:
        Plot title.
    rotation:
        X-axis label rotation.
    """
    x = rate_df[group_col].astype(str).tolist()
    y = (rate_df[rate_col].astype(float) * 100).tolist()

    plt.figure(figsize=(8, 4))
    plt.bar(x, y)
    plt.title(title)
    plt.ylabel("Default rate (%)")
    plt.xlabel(group_col)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()
