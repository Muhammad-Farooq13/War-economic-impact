"""Shared test fixtures."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sample_raw_df():
    """Minimal synthetic raw DataFrame that matches the dataset schema."""
    n = 200
    rng = np.random.default_rng(42)

    conflict_types = ["Civil War", "World War", "Asymmetric War", "Interstate/Counter-insurgency"]
    regions = ["Middle East", "Europe", "South Asia", "East Asia", "Sub-Saharan Africa"]
    bm_levels = ["Low", "Moderate", "High", "Dominant"]
    sectors = ["Tourism", "Construction", "Energy", "Manufacturing"]

    df = pd.DataFrame(
        {
            "Conflict_Name": [f"Conflict_{i}" for i in range(n)],
            "Conflict_Type": rng.choice(conflict_types, n),
            "Region": rng.choice(regions, n),
            "Start_Year": rng.integers(1939, 2020, n),
            "End_Year": rng.integers(2000, 2026, n),
            "Status": rng.choice(["Ongoing", "Resolved"], n),
            "Primary_Country": [f"Country_{i}" for i in range(n)],
            "Pre_War_Unemployment_%": rng.uniform(2, 20, n),
            "During_War_Unemployment_%": rng.uniform(10, 45, n),
            "Unemployment_Spike_Percentage_Points": rng.uniform(0, 30, n),
            "Most_Affected_Sector": rng.choice(sectors, n),
            "Youth_Unemployment_Change_%": rng.uniform(0, 50, n),
            "Pre_War_Poverty_Rate_%": rng.uniform(5, 40, n),
            "During_War_Poverty_Rate_%": rng.uniform(10, 60, n),
            "Extreme_Poverty_Rate_%": rng.uniform(2, 25, n),
            "Food_Insecurity_Rate_%": rng.uniform(5, 40, n),
            "Households_Fallen_Into_Poverty_Estimate": rng.integers(1000, 500000, n),
            "GDP_Change_%": rng.uniform(-80, 10, n),
            "Inflation_Rate_%": rng.uniform(5, 200, n),
            "Currency_Devaluation_%": rng.uniform(0, 500, n),
            "Cost_of_War_USD": rng.uniform(1e9, 5e11, n),
            "Estimated_Reconstruction_Cost_USD": rng.uniform(2e9, 1e12, n),
            "Informal_Economy_Size_Pre_War_%": rng.uniform(10, 50, n),
            "Informal_Economy_Size_During_War_%": rng.uniform(20, 90, n),
            "Black_Market_Activity_Level": rng.choice(bm_levels, n),
            "Primary_Black_Market_Goods": rng.choice(["food", "fuel", "medicine", "currency"], n),
            "Currency_Black_Market_Rate_Gap_%": rng.uniform(0, 500, n),
            "War_Profiteering_Documented": rng.choice(["Yes", "No"], n),
        }
    )
    return df


@pytest.fixture(scope="session")
def cfg():
    from src.utils import load_config

    return load_config(ROOT / "config" / "config.yaml")
