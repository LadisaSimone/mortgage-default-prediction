from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class CaseDatasets:
    """Container for the case datasets loaded from disk."""
    eod_balances: pd.DataFrame
    cust_prod_mapping: pd.DataFrame
    misc_features: pd.DataFrame
    target: pd.DataFrame


def load_case_datasets(case_data_dir: str | Path) -> CaseDatasets:
    """
    Load the four case datasets from a local directory.

    Parameters
    ----------
    case_data_dir:
        Path to the folder containing the CSV files:
        - case_eod_balances.csv
        - case_cust_prod_mapping.csv
        - case_misc_features.csv
        - case_target.csv

    Returns
    -------
    CaseDatasets
        A dataclass containing four pandas DataFrames.

    Notes
    -----
    This function performs minimal parsing and does not modify the raw data.
    """
    case_data_dir = Path(case_data_dir)

    eod_balances = pd.read_csv(case_data_dir / "case_eod_balances.csv")
    cust_prod_mapping = pd.read_csv(case_data_dir / "case_cust_prod_mapping.csv")
    misc_features = pd.read_csv(case_data_dir / "case_misc_features.csv")
    target = pd.read_csv(case_data_dir / "case_target.csv")

    return CaseDatasets(
        eod_balances=eod_balances,
        cust_prod_mapping=cust_prod_mapping,
        misc_features=misc_features,
        target=target,
    )