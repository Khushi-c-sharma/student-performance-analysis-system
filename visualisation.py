"""
This module consumes analytical outputs (DataFrames)
and produces matplotlib Figure objects without rendering.

Responsibilities:
- Visualize department performance
- Visualize CGPA vs salary relationship
- Visualize grade distributions

No data transformation is performed here.
"""

from data_analysis import Analysis
from typing import Optional, List
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "university_data.csv"

OUTPUT_DIR = PROJECT_ROOT / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_department_performance(
    dept_df: pd.DataFrame,
    metric: str = "Avg_Salary_LPA",
) -> plt.Figure:
    """
    Create a bar chart for department-wise performance.

    Args:
        dept_df (pd.DataFrame): Output from department placement analysis
                                Expected to have 'Department' column and the metric column
        metric (str): Column name to visualize (default: "Avg_Salary_LPA")

    Returns:
        matplotlib.figure.Figure: Bar chart showing department performance
    """
    # Create figure and axis objects with specified size
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create bar chart with departments on x-axis and metric values on y-axis
    ax.bar(dept_df["Department"], dept_df[metric], color="skyblue", edgecolor="navy")

    # Set axis labels with readable formatting (replace underscores with spaces)
    ax.set_xlabel("Department", fontsize=12)
    ax.set_ylabel(metric.replace("_", " "), fontsize=12)
    ax.set_title(
        f"Department-wise {metric.replace('_', ' ')}", fontsize=14, fontweight="bold"
    )

    # Rotate x-axis labels for better readability
    ax.tick_params(axis="x", rotation=45)

    # Add grid for easier reading of values
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Adjust layout to prevent label cutoff
    fig.tight_layout()
    return fig


def plot_cgpa_vs_salary(
    df: pd.DataFrame,
    cgpa_col: str = "CGPA",
    salary_col: str = "Placement_Package_LPA",
) -> plt.Figure:
    """
    Scatter plot showing CGPA vs placement package with trend line.

    Args:
        df (pd.DataFrame): Cleaned student-level dataset
        cgpa_col (str): Name of CGPA column (default: "CGPA")
        salary_col (str): Name of salary column (default: "Placement_Package_LPA")

    Returns:
        matplotlib.figure.Figure: Scatter plot with linear trend line and correlation
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))

    # Create scatter plot with semi-transparent points
    ax.scatter(
        df[cgpa_col], df[salary_col], alpha=0.6, color="coral", edgecolors="black"
    )

    # Calculate and plot linear trend line
    # np.polyfit returns coefficients for polynomial fit (degree 1 = linear)
    z = np.polyfit(df[cgpa_col], df[salary_col], 1)
    p = np.poly1d(z)  # Create polynomial function from coefficients
    ax.plot(
        df[cgpa_col],
        p(df[cgpa_col]),
        linestyle="--",
        color="darkred",
        linewidth=2,
        label="Trend Line",
    )

    # Calculate Pearson correlation coefficient
    corr = df[cgpa_col].corr(df[salary_col])

    # Set labels and title
    ax.set_xlabel("CGPA", fontsize=12)
    ax.set_ylabel("Placement Package (LPA)", fontsize=12)
    ax.set_title(f"CGPA vs Package (corr = {corr:.2f})", fontsize=14, fontweight="bold")

    # Add legend and grid
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    return fig


def plot_grade_distribution(
    grade_series: pd.Series,
) -> plt.Figure:
    """
    Pie chart for grade distribution.

    Args:
        grade_series (pd.Series): Output of Analysis.grade_distribution()
                                  Index = grade labels, values = counts
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.pie(
        grade_series.values,
        labels=grade_series.index,
        autopct="%1.1f%%",  # Show percentages with 1 decimal place
        startangle=90,  # Start from top
        colors=plt.cm.Set3.colors,
    )

    ax.set_title("Grade Distribution", fontsize=14, fontweight="bold")
    ax.axis("equal")  # Equal aspect ratio ensures circular pie chart

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Correlation Matrix",
) -> plt.Figure:
    """
    Plot a correlation matrix heatmap for numeric features.

    This function:
    - Selects numeric columns
    - Computes the Pearson correlation matrix
    - Visualizes it as a heatmap
    - Returns a Matplotlib Figure (no plt.show())

    Args:
        df (pd.DataFrame): Cleaned DataFrame
        columns (List[str], optional): Specific numeric columns to include.
            If None, all numeric columns are used.
        title (str): Plot title

    Returns:
        matplotlib.figure.Figure: Correlation heatmap figure
    """
    # Select numeric columns
    if columns is None:
        data = df.select_dtypes(include="number")
    else:
        data = df[columns]

    if data.empty:
        raise ValueError("No numeric columns available for correlation heatmap")

    # Compute correlation matrix
    corr = data.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    # Axis ticks and labels
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    # Add correlation values on cells
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )

    # Colorbar
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    fig.tight_layout()

    return fig


def save_figure(
    fig: plt.Figure,
    path: Path,
    dpi: int = 300,
    close: bool = True,
) -> None:
    """
    Save a matplotlib Figure to disk safely.

    Args:
        fig (plt.Figure): Figure object to save
        path (Path): Output file path (must include extension like .png, .pdf, .jpg)
        dpi (int): Image resolution in dots per inch (default: 300 for high quality)
        close (bool): Whether to close the figure after saving to free memory (default: True)

    Raises:
        ValueError: If path has no file extension
    """
    # Ensure path is a Path object
    path = Path(path)

    # Validate that file extension is provided
    if path.suffix == "":
        raise ValueError("Output path must include a file extension (e.g., .png)")

    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure with tight bounding box to avoid whitespace
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Figure saved to {path}")

    # Close figure to free memory if requested
    if close:
        plt.close(fig)


if __name__ == "__main__":
    # Initialize analysis
    analyzer = Analysis(CSV_PATH)
    df = analyzer.load_data()
    df = analyzer.preprocess_data(df)

    # --- Department performance ---
    dept_df = analyzer.department_placement_stats()
    fig = plot_department_performance(dept_df)
    save_figure(fig, OUTPUT_DIR / "department_performance.png")

    # --- CGPA vs Salary ---
    fig = plot_cgpa_vs_salary(df)
    save_figure(fig, OUTPUT_DIR / "cgpa_vs_salary.png")

    # --- Grade distribution ---
    grade_df = analyzer.grade_distribution()
    fig = plot_grade_distribution(grade_df)
    save_figure(fig, OUTPUT_DIR / "grade_distribution.png")

    logger.info("Visualization preview completed successfully")
