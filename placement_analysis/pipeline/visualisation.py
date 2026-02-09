from pathlib import Path
from typing import Optional
import logging

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline.config import FigureSpec, FIGURE_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "processed"
CSV_PATH = DATA_DIR / "cleaned_placement_data.csv"


class PlacementVisualizer:
    """
    Centralized visualization engine for placement analytics.
    All plot methods return matplotlib Figure objects.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        style: str = "whitegrid",
        fig_size: tuple[int, int] = (12, 8),
        font_size: int = 10,
        show_plots: bool = False,
    ):
        self.output_dir = output_dir
        self.show_plots = show_plots
        self.logger = logging.getLogger(self.__class__.__name__)

        self._configure_style(style, fig_size, font_size)
        self._create_directories()

    def _configure_style(
        self,
        style: str,
        fig_size: tuple[int, int],
        font_size: int,
    ) -> None:
        sns.set_style(style)
        plt.rcParams.update(
            {
                "figure.figsize": fig_size,
                "font.size": font_size,
            }
        )

        self.logger.debug(
            "Style configured | style=%s fig_size=%s font_size=%s",
            style,
            fig_size,
            font_size,
        )

    def _create_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_figure(
        self,
        fig: plt.Figure,
        name: str,
        *,
        dpi: int = 300,
        close: bool = True,
    ) -> Path:
        """
        Persist a figure to disk.

        Returns:
            Path to saved figure
        """
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        self.logger.info("Figure saved: %s", path)

        if self.show_plots:
            plt.show()
        elif close:
            plt.close(fig)

        return path

    def plot_placement_overview(
        self,
        df: pd.DataFrame,
    ) -> plt.Figure:
        """
        Visualize overall placement distribution.

        Args:
            df: Cleaned placement DataFrame
            save_name: Output file name for the plot
        """
        if "Placed" not in df.columns:
            raise KeyError("Column 'Placed' not found")

        counts = df["Placed"].value_counts()

        fig, ax = plt.subplots()

        sns.barplot(
            x=counts.index.map({True: "Placed", False: "Not Placed"}),
            y=counts.values,
            ax=ax,
        )

        ax.set_title("Overall Placement Distribution")
        ax.set_xlabel("Placement Status")
        ax.set_ylabel("Number of Students")

        return fig

    def plot_department_placements(
        self,
        df: pd.DataFrame,
    ) -> plt.Figure:
        """
        Plot number of placed vs unplaced students across departments.

        Args:
            df: Cleaned placement DataFrame
            save_name: Output file name
        """
        required = {"Department", "Placed"}
        if not required.issubset(df.columns):
            raise KeyError(f"Missing columns: {required - set(df.columns)}")

        fig, ax = plt.subplots()

        sns.countplot(
            data=df,
            x="Department",
            hue="Placed",
            order=df["Department"].value_counts().index,
            ax=ax,
        )

        ax.set_title("Placement Count by Department")
        ax.set_xlabel("Department")
        ax.set_ylabel("Number of Students")
        ax.tick_params(axis="x", rotation=45)

        return fig

    def plot_cgpa_vs_package(
        self,
        df: pd.DataFrame,
    ) -> plt.Figure:
        required = {"CGPA", "Placement_Package_LPA", "Placed"}
        if not required.issubset(df.columns):
            raise KeyError("Missing required columns")

        placed_df = df[df["Placed"]]

        fig, ax = plt.subplots()

        sns.scatterplot(
            data=placed_df,
            x="CGPA",
            y="Placement_Package_LPA",
            alpha=0.7,
            ax=ax,
        )

        corr = placed_df["CGPA"].corr(placed_df["Placement_Package_LPA"])

        ax.set_title(f"CGPA vs Placement Package (corr={corr:.2f})")
        ax.set_xlabel("CGPA")
        ax.set_ylabel("Package (LPA)")

        return fig

    def plot_cgpa_binned_placement_rate(
        self,
        df: pd.DataFrame,
        *,
        bins: int = 5,
    ) -> plt.Figure:
        df = df[["CGPA", "Placed"]].dropna()
        df["CGPA_Bin"] = pd.cut(df["CGPA"], bins=bins)

        rates = df.groupby("CGPA_Bin")["Placed"].mean().mul(100).reset_index()

        fig, ax = plt.subplots()

        sns.barplot(
            data=rates,
            x="CGPA_Bin",
            y="Placed",
            ax=ax,
        )

        ax.set_title("Placement Rate by CGPA Range")
        ax.set_xlabel("CGPA Range")
        ax.set_ylabel("Placement Rate (%)")
        ax.tick_params(axis="x", rotation=45)

        return fig

    def plot_department_placement_rate(
        self,
        df: pd.DataFrame,
    ) -> plt.Figure:
        required = {"Department", "Placed"}
        if not required.issubset(df.columns):
            raise KeyError(f"Missing columns: {required - set(df.columns)}")

        rates = (
            df.groupby("Department")["Placed"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots()

        sns.barplot(
            x=rates.index,
            y=rates.values,
            ax=ax,
        )

        ax.set_title("Placement Rate by Department (%)")
        ax.set_xlabel("Department")
        ax.set_ylabel("Placement Rate (%)")
        ax.tick_params(axis="x", rotation=45)

        return fig

    def plot_department_package_distribution(
        self,
        df: pd.DataFrame,
    ) -> plt.Figure:
        required = {"Department", "Placed", "Placement_Package_LPA"}
        if not required.issubset(df.columns):
            raise KeyError("Missing required columns")

        placed_df = df[df["Placed"]]

        fig, ax = plt.subplots()

        sns.boxplot(
            data=placed_df,
            x="Department",
            y="Placement_Package_LPA",
            ax=ax,
        )

        ax.set_title("Placement Package Distribution by Department")
        ax.set_xlabel("Department")
        ax.set_ylabel("Package (LPA)")
        ax.tick_params(axis="x", rotation=45)

        return fig

    def plot_cgpa_distribution(
        self,
        df: pd.DataFrame,
    ) -> plt.Figure:
        required = {"CGPA", "Placed"}
        if not required.issubset(df.columns):
            raise KeyError("Missing required columns")

        fig, ax = plt.subplots()

        sns.kdeplot(
            data=df,
            x="CGPA",
            hue="Placed",
            fill=True,
            common_norm=False,
            ax=ax,
        )

        ax.set_title("CGPA Distribution by Placement Status")
        ax.set_xlabel("CGPA")
        ax.set_ylabel("Density")

        return fig

    def plot_backlog_impact(
        self,
        df: pd.DataFrame,
    ) -> plt.Figure:
        required = {"Backlogs", "Placed"}
        if not required.issubset(df.columns):
            raise KeyError("Missing required columns")

        stats = df.groupby("Backlogs")["Placed"].mean().mul(100).reset_index()

        fig, ax = plt.subplots()

        sns.lineplot(
            data=stats,
            x="Backlogs",
            y="Placed",
            marker="o",
            ax=ax,
        )

        ax.set_title("Placement Rate vs Number of Backlogs")
        ax.set_xlabel("Number of Backlogs")
        ax.set_ylabel("Placement Rate (%)")

        return fig

    def plot_internship_impact(
        self,
        df: pd.DataFrame,
    ) -> plt.Figure:
        required = {"Has_Internship", "Placed", "Placement_Package_LPA"}
        if not required.issubset(df.columns):
            raise KeyError("Missing required columns")

        fig, ax = plt.subplots()

        sns.boxplot(
            data=df[df["Placed"]],
            x="Has_Internship",
            y="Placement_Package_LPA",
            ax=ax,
        )

        ax.set_title("Internship Impact on Placement Package")
        ax.set_xlabel("Has Internship")
        ax.set_ylabel("Package (LPA)")

        return fig

    def plot_gender_placement_outcomes(
        self,
        df: pd.DataFrame,
    ) -> plt.Figure:
        required = {"Gender", "Placed"}
        if not required.issubset(df.columns):
            raise KeyError("Missing required columns")

        fig, ax = plt.subplots()

        sns.barplot(
            data=df,
            x="Gender",
            y="Placed",
            estimator=lambda x: x.mean() * 100,
            ax=ax,
        )

        ax.set_title("Placement Rate by Gender")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Placement Rate (%)")

        return fig

    def plot_department_heatmap(
        self,
        df: pd.DataFrame,
    ) -> plt.Figure:
        required = {"Department", "Placed"}
        if not required.issubset(df.columns):
            raise KeyError("Missing required columns")

        stats = df.groupby("Department")["Placed"].mean().mul(100).to_frame()

        fig, ax = plt.subplots()

        sns.heatmap(
            stats,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            ax=ax,
        )

        ax.set_title("Department-wise Placement Rate (%)")
        ax.set_xlabel("")
        ax.set_ylabel("Department")

        return fig

    def plot_gender_department_segmentation(
        self,
        df: pd.DataFrame,
    ) -> plt.Figure:
        required = {"Department", "Gender", "Placed"}
        if not required.issubset(df.columns):
            raise KeyError("Missing required columns")

        pivot = df.groupby(["Department", "Gender"])["Placed"].mean().mul(100).unstack()

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="coolwarm",
            ax=ax,
        )

        ax.set_title("Placement Rate (%) by Department and Gender")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Department")

        return fig

    def plot_high_risk_distribution(
        self,
        df: pd.DataFrame,
    ) -> plt.Figure:
        required = {"Department", "CGPA", "Backlogs", "Has_Internship"}
        if not required.issubset(df.columns):
            raise KeyError("Missing required columns")

        high_risk = df[
            (df["CGPA"] < 7.0) & (df["Backlogs"] > 2) & (~df["Has_Internship"])
        ]

        fig, ax = plt.subplots()

        sns.countplot(
            data=high_risk,
            y="Department",
            order=high_risk["Department"].value_counts().index,
            ax=ax,
        )

        ax.set_title("High-Risk Students by Department")
        ax.set_xlabel("Student Count")
        ax.set_ylabel("Department")

        return fig

    def plot_top_performers_profile(
        self,
        df: pd.DataFrame,
    ) -> plt.Figure:
        required = {"CGPA", "Placed", "Placement_Package_LPA", "Department"}
        if not required.issubset(df.columns):
            raise KeyError("Missing required columns")

        median_pkg = df.loc[df["Placed"], "Placement_Package_LPA"].median()

        top = df[
            (df["Placed"])
            & (df["CGPA"] > 8.5)
            & (df["Placement_Package_LPA"] > median_pkg)
        ]

        fig, ax = plt.subplots()

        sns.scatterplot(
            data=top,
            x="CGPA",
            y="Placement_Package_LPA",
            hue="Department",
            ax=ax,
        )

        ax.set_title("Top Performers: CGPA vs Package")
        ax.set_xlabel("CGPA")
        ax.set_ylabel("Package (LPA)")

        return fig

    def generate_all(
        self,
        df: pd.DataFrame,
        fig_registry: dict[str, FigureSpec],
    ) -> None:
        """
        Generate and persist all registered figures.

        Args:
            df: Cleaned placement DataFrame
            fig_registry: Mapping of figure keys to FigureSpec definitions
        """
        self.logger.info(
            "Starting batch visualization generation | total_plots=%d",
            len(fig_registry),
        )

        for fig_key, spec in fig_registry.items():
            self.logger.info("Generating plot: %s", fig_key)

            try:
                plot_fn = getattr(self, spec.method_name)

                fig = plot_fn(df, **(spec.kwargs or {}))

                output_path = self.output_dir / spec.output_name
                self.save_figure(fig, output_path)

                self.logger.info("Plot completed successfully: %s", fig_key)

            except Exception as exc:
                self.logger.exception(
                    "Plot generation failed | plot=%s | reason=%s",
                    fig_key,
                    exc,
                )

        self.logger.info("Visualization batch completed")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    try:
        logger.info("STARTING VISUALIZATION PIPELINE")

        df = pd.read_csv(CSV_PATH)

        visualizer = PlacementVisualizer(
            output_dir=PROJECT_ROOT / "outputs" / "figures",
            show_plots=False,
        )

        visualizer.generate_all(
            df=df,
            fig_registry=FIGURE_REGISTRY,
        )

        logger.info("VISUALIZATION PIPELINE COMPLETED")

    except Exception as e:
        logger.critical("Visualization pipeline failed")
        raise
