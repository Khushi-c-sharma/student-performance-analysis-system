import pandas as pd
from typing import Dict
from .config import PROCESSED_DATA_DIR
import logging
import os


logger = logging.getLogger(__name__)

CSV_PATH = PROCESSED_DATA_DIR / "cleaned_placement_data.csv"


class PlacementDataAnalyzer:
    """
    Perform comprehensive statistical and exploratory analysis
    on the cleaned placement dataset.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize analyzer with cleaned dataset.

        Args:
            df (pd.DataFrame): Cleaned placement dataset
        """
        self.df = df.copy()

    def dataset_health_check(self) -> Dict:
        """
        Perform basic dataset sanity checks.

        Returns:
            dict: Summary of missing values and dataset size
        """
        return {
            "total_records": len(self.df),
            "missing_values": self.df.isna().sum().to_dict(),
        }

    def placement_summary(self) -> Dict:
        """
        Compute high-level placement statistics.

        Returns:
            dict: Placement counts, rates, and package statistics
        """
        placed_df = self.df[self.df["Placed"] == True]

        summary = {
            "total_students": len(self.df),
            "total_placed": len(placed_df),
            "total_unplaced": len(self.df) - len(placed_df),
            "placement_rate_%": round((len(placed_df) / len(self.df)) * 100, 2),
        }

        packages = placed_df[placed_df["Placement_Package_LPA"] > 0][
            "Placement_Package_LPA"
        ]

        summary["package_stats"] = {
            "mean": round(packages.mean(), 2),
            "median": round(packages.median(), 2),
            "std": round(packages.std(), 2),
            "min": round(packages.min(), 2),
            "max": round(packages.max(), 2),
            "q25": round(packages.quantile(0.25), 2),
            "q75": round(packages.quantile(0.75), 2),
            "q90": round(packages.quantile(0.90), 2),
            "q95": round(packages.quantile(0.95), 2),
            "q99": round(packages.quantile(0.99), 2),
        }

        return summary

    def academic_performance_summary(self) -> Dict:
        """
        Analyze academic performance metrics.

        Returns:
            dict: CGPA, backlog, and internship statistics
        """
        cgpa = self.df["CGPA"].dropna()

        return {
            "cgpa_stats": {
                "mean": round(cgpa.mean(), 2),
                "median": round(cgpa.median(), 2),
                "std": round(cgpa.std(), 2),
                "min": round(cgpa.min(), 2),
                "max": round(cgpa.max(), 2),
            },
            "backlogs": {
                "students_with_backlogs": int((self.df["Backlogs"] > 0).sum()),
                "percentage_with_backlogs": round(
                    ((self.df["Backlogs"] > 0).mean()) * 100, 2
                ),
                "average_backlogs": round(self.df["Backlogs"].mean(), 2),
            },
            "internships": {
                "students_with_internship": int(self.df["Has_Internship"].sum()),
                "percentage_with_internship": round(
                    (self.df["Has_Internship"].mean()) * 100, 2
                ),
            },
        }

    def demographic_distribution(self) -> Dict:
        """
        Analyze demographic composition of the dataset.

        Returns:
            dict: Gender, department, and category distributions
        """
        demographics = {}

        if "Gender" in self.df.columns:
            demographics["gender"] = self.df["Gender"].value_counts().to_dict()

        if "Department" in self.df.columns:
            demographics["department"] = self.df["Department"].value_counts().to_dict()

        if "Category" in self.df.columns:
            demographics["category"] = self.df["Category"].value_counts().to_dict()

        return demographics

    def cgpa_segment_analysis(self, bins: int = 5) -> pd.DataFrame:
        """
        Analyze placement outcomes across CGPA ranges.

        Args:
            bins (int): Number of CGPA bins

        Returns:
            pd.DataFrame: CGPA segment-wise placement statistics
        """
        logger.info("Running CGPA segmentation analysis")

        df = self.df[self.df["CGPA"].notna()].copy()
        df["CGPA_Range"] = pd.cut(df["CGPA"], bins=bins)

        result = (
            df.groupby("CGPA_Range")
            .agg(
                {
                    "Placed": ["count", "sum", "mean"],
                    "Placement_Package_LPA": "mean",
                }
            )
            .round(3)
        )

        result.columns = [
            "Total_Students",
            "Placed_Count",
            "Placement_Rate",
            "Avg_Package_LPA",
        ]

        result["Placement_Rate"] = (result["Placement_Rate"] * 100).round(2)

        logger.info("Completed CGPA segmentation analysis")
        return result.reset_index()

    def placement_rate_by_cgpa_threshold(self, threshold: float = 8.0) -> pd.DataFrame:
        """
        Compare placement rates above and below a CGPA threshold.

        Args:
            threshold (float): CGPA cutoff

        Returns:
            pd.DataFrame: Threshold-based comparison
        """
        logger.info("Analyzing placement rate by CGPA threshold: %.2f", threshold)

        above = self.df[self.df["CGPA"] >= threshold]
        below = self.df[self.df["CGPA"] < threshold]

        result = pd.DataFrame(
            {
                "Category": [f"CGPA ≥ {threshold}", f"CGPA < {threshold}"],
                "Total_Students": [len(above), len(below)],
                "Placed": [above["Placed"].sum(), below["Placed"].sum()],
                "Placement_Rate_%": [
                    round(above["Placed"].mean() * 100, 2),
                    round(below["Placed"].mean() * 100, 2),
                ],
                "Avg_Package_LPA": [
                    round(above[above["Placed"]]["Placement_Package_LPA"].mean(), 2),
                    round(below[below["Placed"]]["Placement_Package_LPA"].mean(), 2),
                ],
            }
        )

        return result

    def department_performance(self) -> pd.DataFrame:
        """
        Analyze placement and academic performance by department.

        Returns:
            pd.DataFrame: Department-level statistics
        """
        logger.info("Computing department-wise performance metrics")

        dept_stats = (
            self.df.groupby("Department")
            .agg(
                {
                    "Student_ID": "count",
                    "Placed": ["sum", "mean"],
                    "Placement_Package_LPA": ["mean", "median", "max"],
                    "CGPA": "mean",
                    "Backlogs": "mean",
                    "Has_Internship": "mean",
                }
            )
            .round(2)
        )

        dept_stats.columns = [
            "Total_Students",
            "Placed_Count",
            "Placement_Rate",
            "Avg_Package",
            "Median_Package",
            "Max_Package",
            "Avg_CGPA",
            "Avg_Backlogs",
            "Internship_Rate",
        ]

        dept_stats["Placement_Rate"] = (dept_stats["Placement_Rate"] * 100).round(2)
        dept_stats["Internship_Rate"] = (dept_stats["Internship_Rate"] * 100).round(2)

        logger.info("Department-wise performance computed")
        return dept_stats.sort_values("Placement_Rate", ascending=False).reset_index()

    def top_departments_by_package(self, top_n: int = 5) -> pd.DataFrame:
        """
        Identify departments with highest average placement packages.

        Args:
            top_n (int): Number of top departments

        Returns:
            pd.DataFrame: Top departments by package
        """
        logger.info("Identifying top %d departments by package", top_n)

        placed_df = self.df[
            (self.df["Placed"] == True) & (self.df["Placement_Package_LPA"] > 0)
        ]

        result = (
            placed_df.groupby("Department")
            .agg({"Placement_Package_LPA": ["mean", "median", "max", "count"]})
            .round(2)
        )

        result.columns = [
            "Avg_Package",
            "Median_Package",
            "Max_Package",
            "Placed_Count",
        ]

        return (
            result.sort_values("Avg_Package", ascending=False).head(top_n).reset_index()
        )

    def gender_gap_analysis(self) -> Dict:
        """
        Analyze gender-based disparities in placement and packages.

        Returns:
            dict: Gender gap metrics
        """
        logger.info("Running gender gap analysis")

        male = self.df[self.df["Gender"] == "M"]
        female = self.df[self.df["Gender"] == "F"]

        analysis = {
            "counts": {
                "male": len(male),
                "female": len(female),
            },
            "placement_rates_%": {
                "male": round(male["Placed"].mean() * 100, 2),
                "female": round(female["Placed"].mean() * 100, 2),
            },
            "avg_package_LPA": {
                "male": round(male[male["Placed"]]["Placement_Package_LPA"].mean(), 2),
                "female": round(
                    female[female["Placed"]]["Placement_Package_LPA"].mean(), 2
                ),
            },
        }

        analysis["gaps"] = {
            "placement_rate_gap_%": round(
                analysis["placement_rates_%"]["male"]
                - analysis["placement_rates_%"]["female"],
                2,
            ),
            "package_gap_LPA": round(
                analysis["avg_package_LPA"]["male"]
                - analysis["avg_package_LPA"]["female"],
                2,
            ),
        }

        return analysis

    def internship_impact(self) -> Dict:
        """
        Analyze impact of internships on placement outcomes.

        Returns:
            dict: Internship ROI metrics
        """
        logger.info("Analyzing internship impact on placements")

        with_int = self.df[self.df["Has_Internship"] == True]
        without_int = self.df[self.df["Has_Internship"] == False]

        analysis = {
            "with_internship": {
                "count": len(with_int),
                "placement_rate_%": round(with_int["Placed"].mean() * 100, 2),
                "avg_package_LPA": round(
                    with_int[with_int["Placed"]]["Placement_Package_LPA"].mean(), 2
                ),
            },
            "without_internship": {
                "count": len(without_int),
                "placement_rate_%": round(without_int["Placed"].mean() * 100, 2),
                "avg_package_LPA": round(
                    without_int[without_int["Placed"]]["Placement_Package_LPA"].mean(),
                    2,
                ),
            },
        }

        analysis["impact"] = {
            "placement_rate_boost_%": round(
                analysis["with_internship"]["placement_rate_%"]
                - analysis["without_internship"]["placement_rate_%"],
                2,
            ),
            "package_premium_LPA": round(
                analysis["with_internship"]["avg_package_LPA"]
                - analysis["without_internship"]["avg_package_LPA"],
                2,
            ),
        }

        return analysis

    def correlation_analysis(self) -> pd.DataFrame:
        """
        Compute correlation between numeric features and placement outcome.

        Returns:
            pd.DataFrame: Sorted correlation matrix with 'Placed'
        """
        logger.info("Running correlation analysis")

        numeric_cols = [
            "CGPA",
            "Backlogs",
            "Placement_Package_LPA",
            "Has_Internship",
            "Placed",
        ]

        corr_df = (
            self.df[numeric_cols]
            .corr()
            .round(3)
            .loc[:, ["Placed"]]
            .sort_values("Placed", ascending=False)
        )

        logger.info("Correlation analysis completed")
        return corr_df.reset_index().rename(columns={"index": "Feature"})

    def internship_placement_chi_square(self) -> Dict:
        """
        Test association between internships and placement using Chi-Square test.

        Returns:
            dict: Test statistics and interpretation
        """
        from scipy.stats import chi2_contingency

        logger.info("Running chi-square test: Internship vs Placement")

        contingency = pd.crosstab(self.df["Has_Internship"], self.df["Placed"])

        chi2, p, dof, expected = chi2_contingency(contingency)

        result = {
            "chi_square_stat": round(chi2, 3),
            "p_value": round(p, 5),
            "degrees_of_freedom": dof,
            "significant": p < 0.05,
            "interpretation": (
                "Significant association between internship and placement"
                if p < 0.05
                else "No significant association detected"
            ),
        }

        logger.info("Chi-square test completed | p-value = %.5f", p)
        return result

    def cgpa_package_ttest(self, threshold: float = 8.0) -> Dict:
        """
        Compare placement packages for high vs low CGPA students.

        Args:
            threshold (float): CGPA cutoff

        Returns:
            dict: t-test statistics
        """
        from scipy.stats import ttest_ind

        logger.info(
            "Running t-test for package comparison (CGPA threshold = %.2f)", threshold
        )

        high = self.df[(self.df["Placed"]) & (self.df["CGPA"] >= threshold)][
            "Placement_Package_LPA"
        ]

        low = self.df[(self.df["Placed"]) & (self.df["CGPA"] < threshold)][
            "Placement_Package_LPA"
        ]

        t_stat, p_val = ttest_ind(high, low, equal_var=False)

        result = {
            "t_statistic": round(t_stat, 3),
            "p_value": round(p_val, 5),
            "significant": p_val < 0.05,
            "mean_package_high_cgpa": round(high.mean(), 2),
            "mean_package_low_cgpa": round(low.mean(), 2),
        }

        logger.info("t-test completed | p-value = %.5f", p_val)
        return result

    def identify_high_risk_students(
        self,
        cgpa_threshold: float = 6.5,
        backlog_threshold: int = 3,
    ) -> pd.DataFrame:
        """
        Identify students at high risk of non-placement.

        Returns:
            pd.DataFrame: High-risk student subset
        """
        logger.info("Identifying high-risk students")

        high_risk = self.df[
            (self.df["Placed"] == False)
            & (
                (self.df["CGPA"] < cgpa_threshold)
                | (self.df["Backlogs"] >= backlog_threshold)
            )
        ].copy()

        high_risk["Risk_Flag"] = "HIGH"

        logger.info("High-risk students identified: %d", len(high_risk))
        return high_risk.reset_index(drop=True)

    def top_performer_profile(
        self,
        cgpa_cutoff: float = 8.5,
        top_percentile: float = 0.9,
    ) -> pd.DataFrame:
        """
        Profile top-performing students based on CGPA and package percentile.

        Returns:
            pd.DataFrame: Top performer cohort
        """
        logger.info("Profiling top performers")

        package_cutoff = self.df["Placement_Package_LPA"].quantile(top_percentile)

        top_students = self.df[
            (self.df["Placed"] == True)
            & (self.df["CGPA"] >= cgpa_cutoff)
            & (self.df["Placement_Package_LPA"] >= package_cutoff)
        ].copy()

        top_students["Performer_Tag"] = "TOP"

        logger.info("Top performers identified: %d", len(top_students))
        return top_students.reset_index(drop=True)

    def strategic_student_segmentation(self) -> pd.DataFrame:
        """
        Segment students into strategic categories for placement action.

        Returns:
            pd.DataFrame: Students with segment labels
        """
        logger.info("Running strategic student segmentation")

        def segment(row):
            if row["Placed"] and row["CGPA"] >= 8.0:
                return "High Value"
            if not row["Placed"] and row["CGPA"] >= 7.0:
                return "Trainable"
            if not row["Placed"] and row["Backlogs"] >= 3:
                return "High Risk"
            return "Average"

        df_seg = self.df.copy()
        df_seg["Segment"] = df_seg.apply(segment, axis=1)

        logger.info("Segmentation completed")
        return df_seg

    def executive_summary(self) -> Dict:
        """
        Generate a high-level KPI summary for stakeholders.

        Returns:
            dict: Executive metrics
        """
        logger.info("Generating executive summary")

        summary = {
            "total_students": len(self.df),
            "overall_placement_rate_%": round(self.df["Placed"].mean() * 100, 2),
            "avg_package_LPA": round(
                self.df[self.df["Placed"]]["Placement_Package_LPA"].mean(), 2
            ),
            "internship_impact_%": self.internship_impact()["impact"][
                "placement_rate_boost_%"
            ],
            "top_department": self.department_performance().iloc[0]["Department"],
            "high_risk_students": len(self.identify_high_risk_students()),
        }

        logger.info("Executive summary generated")
        return summary


"""
if __name__ == "__main__":
    # setup_logging()
    logger = logging.getLogger("MAIN")

    try:
        logger.info("===== STARTING PLACEMENT DATA PIPELINE =====")

        # --------------------------------------------------
        # STEP 2: DATA ANALYSIS
        # --------------------------------------------------
        logger.info("Initializing data analysis pipeline")

        df = pd.read_csv(CSV_PATH)

        analyzer = PlacementDataAnalyzer(df)

        logger.info("Computing summary statistics")
        stats_report = analyzer.placement_summary()

        logger.info("Computing department-wise performance")
        dept_stats = analyzer.department_performance()

        logger.info("Computing CGPA segmentation")
        cgpa_segmentation = analyzer.strategic_student_segmentation()

        logger.info("Computing correlation analysis")
        correlations = analyzer.correlation_analysis()

        logger.info("Identifying high-risk students")
        high_risk_df = analyzer.identify_high_risk_students()

        # --------------------------------------------------
        # STEP 3: PERSISTENCE
        # --------------------------------------------------
        logger.info("Persisting analysis outputs")

        persistence = AnalysisPersistenceManager(base_output_dir="outputs")

        persistence.persist(stats_report, "placement_summary")
        persistence.persist(dept_stats, "department_performance")
        persistence.persist(cgpa_segmentation, "cgpa_segmentation")
        persistence.persist(correlations.reset_index(), "feature_correlations")
        persistence.persist(high_risk_df, "high_risk_students")

        logger.info("===== PIPELINE COMPLETED SUCCESSFULLY =====")

    except Exception as exc:
        logger.critical(
            "PIPELINE FAILED | %s: %s",
            exc.__class__.__name__,
            exc,
            exc_info=True,
        )
        raise
"""
