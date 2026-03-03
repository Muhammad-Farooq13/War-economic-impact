# Data Directory

## Structure

```
data/
├── raw/                         ← Original, immutable data
│   └── war_economic_impact_dataset.csv
├── processed/                   ← Cleaned + engineered data (auto-generated)
│   ├── war_economic_processed.parquet    ← Output of make_dataset.py
│   └── war_economic_features.parquet    ← Output of build_features.py
└── external/                    ← Any manually sourced supplementary data
```

## Reproducibility

The `processed/` directory is gitignored and fully reproducible:

```bash
make data      # generates war_economic_processed.parquet
make features  # generates war_economic_features.parquet
```

## Raw Data Schema

| Column | Type | Description |
|--------|------|-------------|
| Conflict_Name | string | Name of the conflict |
| Conflict_Type | string | Civil War, World War, Asymmetric War, etc. |
| Region | string | Geographic region |
| Start_Year | int | Year conflict began |
| End_Year | int | Year conflict ended / current year if ongoing |
| Status | string | Ongoing or Resolved |
| Primary_Country | string | Main affected country |
| Pre_War_Unemployment_% | float | Unemployment rate before conflict |
| During_War_Unemployment_% | float | Unemployment rate during conflict |
| Unemployment_Spike_Percentage_Points | float | Difference (During - Pre) |
| Most_Affected_Sector | string | Hardest-hit economic sector |
| Youth_Unemployment_Change_% | float | Change in youth unemployment |
| Pre_War_Poverty_Rate_% | float | Poverty rate before conflict |
| During_War_Poverty_Rate_% | float | Poverty rate during conflict |
| Extreme_Poverty_Rate_% | float | Extreme poverty during conflict |
| Food_Insecurity_Rate_% | float | Food insecurity rate during conflict |
| Households_Fallen_Into_Poverty_Estimate | int | Estimated households impoverished |
| **GDP_Change_%** | **float** | **Target: GDP change during conflict** |
| Inflation_Rate_% | float | Inflation rate during conflict |
| Currency_Devaluation_% | float | Currency devaluation % |
| Cost_of_War_USD | float | Total estimated cost of conflict |
| Estimated_Reconstruction_Cost_USD | float | Post-war reconstruction estimate |
| Informal_Economy_Size_Pre_War_% | float | Informal economy before conflict |
| Informal_Economy_Size_During_War_% | float | Informal economy during conflict |
| Black_Market_Activity_Level | string | Low / Moderate / High / Dominant |
| Primary_Black_Market_Goods | string | Main black market commodities |
| Currency_Black_Market_Rate_Gap_% | float | Parallel market rate gap |
| War_Profiteering_Documented | string | Yes / No |
