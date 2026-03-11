"""
visualize.py
────────────
Reusable plotting functions shared between notebooks and the Streamlit app.

All functions return Matplotlib Figure objects so they can be rendered
in notebooks, saved to disk, or passed to st.pyplot() in Streamlit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# ── Style defaults ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
PALETTE = sns.color_palette("muted")
FIG_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, fname: str) -> None:
    path = FIG_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Distribution Plots
# ═══════════════════════════════════════════════════════════════════════════════


def plot_gdp_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Histogram + KDE of GDP_Change_% across all conflicts."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["GDP_Change_%"], bins=60, kde=True, color=PALETTE[0], ax=ax, stat="density")
    ax.axvline(
        df["GDP_Change_%"].median(),
        color="red",
        ls="--",
        label=f"Median: {df['GDP_Change_%'].median():.1f}%",
    )
    ax.axvline(
        df["GDP_Change_%"].mean(),
        color="orange",
        ls="--",
        label=f"Mean: {df['GDP_Change_%'].mean():.1f}%",
    )
    ax.set_xlabel("GDP Change (%)")
    ax.set_title("Distribution of GDP Change During Conflict")
    ax.legend()
    fig.tight_layout()
    if save:
        _save(fig, "gdp_distribution.png")
    return fig


def plot_numeric_distributions(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    save: bool = True,
) -> plt.Figure:
    """Grid of histograms for key numeric columns."""
    if cols is None:
        cols = [
            "GDP_Change_%",
            "Inflation_Rate_%",
            "Currency_Devaluation_%",
            "Pre_War_Unemployment_%",
            "During_War_Unemployment_%",
            "Pre_War_Poverty_Rate_%",
            "During_War_Poverty_Rate_%",
            "Food_Insecurity_Rate_%",
        ]
    cols = [c for c in cols if c in df.columns]
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(df[col], bins=40, ax=axes[i], color=PALETTE[i % len(PALETTE)], kde=True)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Numeric Feature Distributions", fontsize=13, y=1.01)
    fig.tight_layout()
    if save:
        _save(fig, "numeric_distributions.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Categorical Plots
# ═══════════════════════════════════════════════════════════════════════════════


def plot_gdp_by_conflict_type(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Box plot of GDP change split by conflict type."""
    fig, ax = plt.subplots(figsize=(11, 5))
    order = df.groupby("Conflict_Type")["GDP_Change_%"].median().sort_values().index.tolist()
    sns.boxplot(
        data=df,
        x="Conflict_Type",
        y="GDP_Change_%",
        order=order,
        palette="muted",
        ax=ax,
        width=0.5,
    )
    ax.set_xlabel("Conflict Type")
    ax.set_ylabel("GDP Change (%)")
    ax.set_title("GDP Change by Conflict Type")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    if save:
        _save(fig, "gdp_by_conflict_type.png")
    return fig


def plot_gdp_by_region(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Violin plot of GDP change by region."""
    fig, ax = plt.subplots(figsize=(12, 5))
    order = df.groupby("Region")["GDP_Change_%"].median().sort_values().index.tolist()
    sns.violinplot(
        data=df,
        x="Region",
        y="GDP_Change_%",
        order=order,
        palette="pastel",
        inner="box",
        ax=ax,
    )
    ax.set_xlabel("Region")
    ax.set_ylabel("GDP Change (%)")
    ax.set_title("GDP Change Distribution by Region")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    if save:
        _save(fig, "gdp_by_region.png")
    return fig


def plot_severity_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Bar chart of the engineered severity label."""
    if "Severity_Label" not in df.columns:
        raise ValueError("Column 'Severity_Label' not found. Run preprocessing first.")
    label_map = {0: "Mild", 1: "Moderate", 2: "Severe", 3: "Catastrophic"}
    counts = df["Severity_Label"].map(label_map).value_counts().reindex(label_map.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2196F3", "#FF9800", "#F44336", "#B71C1C"]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.6)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 300,
            f"{bar.get_height():,}",
            ha="center",
            fontsize=10,
        )
    ax.set_xlabel("Severity Class")
    ax.set_ylabel("Count")
    ax.set_title("Economic Severity Label Distribution")
    fig.tight_layout()
    if save:
        _save(fig, "severity_distribution.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Correlation & Heatmap
# ═══════════════════════════════════════════════════════════════════════════════


def plot_correlation_heatmap(
    df: pd.DataFrame,
    top_n: int = 18,
    save: bool = True,
) -> plt.Figure:
    """Correlation heatmap of top-N numeric features correlated with GDP."""
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    if "GDP_Change_%" in corr.columns:
        top_cols = (
            corr["GDP_Change_%"].abs().sort_values(ascending=False).head(top_n).index.tolist()
        )
    else:
        top_cols = num_df.columns[:top_n].tolist()
    sub_corr = corr.loc[top_cols, top_cols]

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(sub_corr, dtype=bool))
    sns.heatmap(
        sub_corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title(f"Feature Correlation Matrix (top {top_n} by |corr| with GDP)", fontsize=13)
    fig.tight_layout()
    if save:
        _save(fig, "correlation_heatmap.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Time / Duration Analysis
# ═══════════════════════════════════════════════════════════════════════════════


def plot_duration_vs_gdp(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Scatter: conflict duration vs GDP change, coloured by severity."""
    if "Conflict_Duration_Years" not in df.columns:
        raise ValueError("Column 'Conflict_Duration_Years' not found.")
    fig, ax = plt.subplots(figsize=(10, 6))
    hue_col = "Severity_Label" if "Severity_Label" in df.columns else None
    scatter_data = df.copy()
    if hue_col:
        label_map = {0: "Mild", 1: "Moderate", 2: "Severe", 3: "Catastrophic"}
        scatter_data["Severity"] = scatter_data[hue_col].map(label_map)
        hue_col = "Severity"
    sns.scatterplot(
        data=scatter_data,
        x="Conflict_Duration_Years",
        y="GDP_Change_%",
        hue=hue_col,
        palette=["#2196F3", "#FF9800", "#F44336", "#B71C1C"],
        alpha=0.5,
        s=20,
        ax=ax,
    )
    ax.set_xlabel("Conflict Duration (Years)")
    ax.set_ylabel("GDP Change (%)")
    ax.set_title("Conflict Duration vs Economic Impact")
    ax.axhline(0, color="grey", ls="--", linewidth=0.8)
    fig.tight_layout()
    if save:
        _save(fig, "duration_vs_gdp.png")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Plotly Interactive
# ═══════════════════════════════════════════════════════════════════════════════


def plotly_gdp_choropleth(df: pd.DataFrame) -> go.Figure:
    """Plotly choropleth of mean GDP change by country/region."""
    agg = df.groupby("Region")["GDP_Change_%"].mean().reset_index()
    fig = px.bar(
        agg.sort_values("GDP_Change_%"),
        x="GDP_Change_%",
        y="Region",
        orientation="h",
        color="GDP_Change_%",
        color_continuous_scale="RdYlGn",
        title="Mean GDP Change (%) by Region",
        labels={"GDP_Change_%": "Mean GDP Change (%)"},
    )
    fig.update_layout(height=500, xaxis_title="Mean GDP Change (%)")
    return fig


def plotly_scatter_matrix(df: pd.DataFrame, cols: Optional[list[str]] = None) -> go.Figure:
    """Interactive scatter matrix for selected features."""
    if cols is None:
        cols = [
            "GDP_Change_%",
            "Inflation_Rate_%",
            "Unemployment_Spike_Percentage_Points",
            "During_War_Poverty_Rate_%",
            "Currency_Devaluation_%",
        ]
    cols = [c for c in cols if c in df.columns]
    color_col = "Severity_Label" if "Severity_Label" in df.columns else None
    fig = px.scatter_matrix(
        df.sample(min(5000, len(df)), random_state=42),
        dimensions=cols,
        color=color_col,
        title="Feature Scatter Matrix (sample n=5000)",
        opacity=0.4,
    )
    fig.update_traces(diagonal_visible=False, marker={"size": 3})
    return fig


def plotly_inflation_boxplot(df: pd.DataFrame) -> go.Figure:
    """Plotly box plot of inflation rate by conflict type."""
    fig = px.box(
        df,
        x="Conflict_Type",
        y="Inflation_Rate_%",
        color="Conflict_Type",
        title="Inflation Rate Distribution by Conflict Type",
        points=False,
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    return fig
