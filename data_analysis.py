"""
This module provides functionality for loading, preprocessing, and analyzing
university datasets. It includes data validation, missing value imputation,
duplicate removal, and outlier detection.
"""

import pandas as pd
from pathlib import Path
import sys
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "university_data.csv"


class Analysis:
    """
    A class for analyzing university data with preprocessing capabilities.

    This class handles the complete data analysis pipeline including:
    - Loading data from CSV files
    - Data validation and quality checks
    - Missing value imputation
    - Duplicate removal
    - Outlier detection and removal

    Attributes:
        CSV_PATH (Path): Path to the CSV file containing university data
        df (pd.DataFrame): The loaded and preprocessed DataFrame
    """

    def __init__(self, CSV_PATH: Path):
        """
        Initialize the Analysis class with a CSV file path.

        Args:
            CSV_PATH (Path): Path object pointing to the CSV file location
        """
        self.CSV_PATH = CSV_PATH
        self.df = None  # Will store the loaded DataFrame

    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from CSV file and perform initial exploration.

        This method:
        - Checks if the file exists
        - Loads the CSV into a pandas DataFrame
        - Logs dataset metadata (shape, columns, types, null counts, memory usage)

        Returns:
            pd.DataFrame: The loaded dataset

        Raises:
            FileNotFoundError: If the CSV file doesn't exist at the specified path
            Exception: If there's an error reading the CSV file
        """
        # Verify file exists before attempting to load
        if not self.CSV_PATH.exists():
            raise FileNotFoundError(f"CSV not found at: {self.CSV_PATH}")

        try:
            # Load CSV file into DataFrame
            df = pd.read_csv(self.CSV_PATH)
            logger.info("Dataset loaded successfully")
        except Exception:
            logger.exception("Failed to load dataset")
            raise

        # Log dataset overview information
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Dtypes:\n{df.dtypes}")
        logger.info(f"Non-null counts:\n{df.notnull().sum()}")
        logger.info(f"Memory usage (bytes):\n{df.memory_usage(deep=True)}")

        # Store the DataFrame in the instance
        self.df = df
        return df

    def impute_missing_values(
        self,
        df: pd.DataFrame,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
    ) -> pd.DataFrame:
        """
        Impute missing values in the dataset using specified strategies.

        This method handles missing data by:
        - Identifying numeric and categorical columns
        - Applying appropriate imputation strategies for each type
        - Logging imputation actions

        Args:
            df (pd.DataFrame): DataFrame with potential missing values
            numeric_strategy (str): Strategy for numeric columns. Options:
                - "mean": Replace with column mean
                - "median": Replace with column median (default, robust to outliers)
                - "zero": Replace with 0
            categorical_strategy (str): Strategy for categorical columns. Options:
                - "mode": Replace with most frequent value (default)
                - "missing": Replace with "Missing" string

        Returns:
            pd.DataFrame: DataFrame with imputed values
        """
        logger.info("Starting missing value imputation...")

        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Impute numeric columns
        for col in numeric_cols:
            missing_count = df[col].isna().sum()

            if missing_count > 0:
                if numeric_strategy == "mean":
                    fill_value = df[col].mean()
                    strategy_name = "mean"
                elif numeric_strategy == "median":
                    fill_value = df[col].median()
                    strategy_name = "median"
                elif numeric_strategy == "zero":
                    fill_value = 0
                    strategy_name = "zero"
                else:
                    # Default to median if invalid strategy provided
                    fill_value = df[col].median()
                    strategy_name = "median (default)"

                df[col].fillna(fill_value, inplace=True)
                logger.info(
                    f"Imputed {missing_count} missing values in '{col}' "
                    f"with {strategy_name}: {fill_value:.2f}"
                )

        # Impute categorical columns
        for col in categorical_cols:
            missing_count = df[col].isna().sum()

            if missing_count > 0:
                if categorical_strategy == "mode":
                    # Get the most frequent value (mode)
                    mode_value = df[col].mode()
                    if not mode_value.empty:
                        fill_value = mode_value[0]
                        strategy_name = "mode"
                    else:
                        # If no mode exists, use "Unknown"
                        fill_value = "Unknown"
                        strategy_name = "Unknown (no mode found)"
                elif categorical_strategy == "missing":
                    fill_value = "Missing"
                    strategy_name = "Missing"
                else:
                    # Default to mode if invalid strategy provided
                    mode_value = df[col].mode()
                    fill_value = mode_value[0] if not mode_value.empty else "Unknown"
                    strategy_name = "mode (default)"

                df[col].fillna(fill_value, inplace=True)
                logger.info(
                    f"Imputed {missing_count} missing values in '{col}' "
                    f"with {strategy_name}: '{fill_value}'"
                )

        logger.info("Missing value imputation completed")
        return df

    def preprocess_data(
        self,
        df: pd.DataFrame,
        value_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        impute: bool = True,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
    ) -> pd.DataFrame:
        """
        Preprocess the dataset by handling missing values, duplicates, and outliers.

        This comprehensive preprocessing pipeline:
        1. Reports missing values
        2. Optionally imputes missing values
        3. Removes duplicate rows
        4. Validates values against specified ranges
        5. Removes invalid/out-of-range values

        Args:
            df (pd.DataFrame): Raw DataFrame to preprocess
            value_ranges (Dict[str, Tuple[float, float]], optional):
                Dictionary mapping column names to (min, max) tuples for validation.
                Example: {"age": (0, 100), "cgpa": (0.0, 10.0)}
            impute (bool): Whether to perform missing value imputation (default: True)
            numeric_strategy (str): Imputation strategy for numeric columns
            categorical_strategy (str): Imputation strategy for categorical columns

        Returns:
            pd.DataFrame: Cleaned and preprocessed DataFrame
        """
        logger.info("Starting data preprocessing...")

        # Step 1: Check for missing values
        missing = df.isna().sum()
        missing = missing[missing > 0]

        if not missing.empty:
            logger.info(f"Missing values detected:\n{missing}")

            # Step 2: Impute missing values if requested
            if impute:
                df = self.impute_missing_values(
                    df,
                    numeric_strategy=numeric_strategy,
                    categorical_strategy=categorical_strategy,
                )
            else:
                logger.info("Skipping imputation (impute=False)")
        else:
            logger.info("No missing values detected")

        # Step 3: Remove duplicate rows
        og_len = len(df)
        df = df.drop_duplicates()
        duplicates_removed = og_len - len(df)
        logger.info(f"Removed {duplicates_removed} duplicate rows")

        # Step 4: Validate value ranges (if provided)
        if value_ranges:
            logger.info("Validating value ranges...")

            for col, (min_val, max_val) in value_ranges.items():
                if col in df.columns:
                    # Check if column is numeric
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        logger.warning(
                            f"Skipping range validation for non-numeric column '{col}'"
                        )
                        continue

                    # Identify values outside the valid range
                    invalid = ~df[col].between(min_val, max_val)
                    count = invalid.sum()

                    if count > 0:
                        logger.warning(
                            f"{count} invalid values in '{col}' "
                            f"(expected range {min_val}–{max_val})"
                        )
                        # Remove rows with invalid values
                        df = df[~invalid]
                else:
                    logger.warning(
                        f"Column '{col}' not found in DataFrame, skipping validation"
                    )

        # Step 5: Log final dataset shape
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info("Data preprocessing completed successfully")

        # Update the instance DataFrame
        self.df = df
        return df

    def get_summary_statistics(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate summary statistics for the dataset.

        Args:
            df (pd.DataFrame, optional): DataFrame to summarize.
                If None, uses the instance's stored DataFrame.

        Returns:
            pd.DataFrame: Summary statistics including count, mean, std, min, max, etc.
        """
        if df is None:
            df = self.df

        if df is None:
            raise ValueError("No DataFrame available. Load data first.")

        logger.info("Generating summary statistics...")
        return df.describe(include="all")

    def department_placement_stats(
        self, df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute department-wise placement statistics.

        Returns:
            pd.DataFrame: Aggregated department-level metrics
        """
        if df is None:
            df = self.df

        if df is None:
            raise ValueError("No DataFrame available. Load data first.")

        logger.info("Computing department-wise placement statistics")

        return (
            df.groupby("Department")
            .agg(
                Avg_CGPA=("CGPA", "mean"),
                Avg_Salary_LPA=("Placement_Package_LPA", "mean"),
                Highest_Package=("Placement_Package_LPA", "max"),
                Total_Students=("Department", "count"),
            )
            .round(2)
            .reset_index()
        )

    def top_performers(
        self,
        cgpa_threshold: float = 9.0,
        top_n: int = 10,
        placed_only: bool = True,
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Identify top-performing students based on CGPA and placement package.
        """
        if df is None:
            df = self.df

        if df is None:
            raise ValueError("No DataFrame available. Load data first.")

        logger.info(f"Selecting top performers (CGPA > {cgpa_threshold}, top {top_n})")

        data = df[df["CGPA"] > cgpa_threshold]

        if placed_only and "Placed" in data.columns:
            data = data[data["Placed"]]

        return data.sort_values("Placement_Package_LPA", ascending=False).head(top_n)[
            ["Name", "Department", "CGPA", "Placement_Package_LPA"]
        ]

    def grade_distribution(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        bins: tuple[float, ...] = (0.0, 7.0, 8.5, 10.0),
        labels: tuple[str, ...] = ("Second Class", "First Class", "Distinction"),
    ) -> pd.Series:
        """
        Compute the distribution of academic grades based on CGPA ranges.
        """

        # Resolve the DataFrame source:
        # Prefer the explicitly passed DataFrame; otherwise, use internal state.
        if df is None:
            df = self.df

        # Fail fast if no data is available at all
        if df is None:
            raise ValueError("No DataFrame available. Load data first.")

        # Ensure the required column exists before proceeding
        if "CGPA" not in df.columns:
            raise KeyError("CGPA column missing from dataset")

        # Validate grading configuration:
        # Number of labels must match the number of CGPA intervals
        if len(bins) - 1 != len(labels):
            raise ValueError("Number of labels must be exactly len(bins) - 1")

        logger.info("Computing grade distribution")

        # Convert CGPA to float explicitly to avoid dtype-related issues,
        # then bin values into categorical grade labels
        grades = pd.cut(
            df["CGPA"].astype(float),
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

        # Return a sorted distribution of grades for stable reporting
        return grades.value_counts().sort_index()


if __name__ == "__main__":
    """
    Main execution function demonstrating the Analysis class usage.

    This script:
    - Loads the university dataset
    - Applies preprocessing and validation
    - Runs core analytics methods
    - Logs key analytical outputs
    """
    try:
        # Initialize analyzer with CSV path
        analyzer = Analysis(CSV_PATH)

        # Load the data
        df = analyzer.load_data()

        # Define valid ranges for specific columns (example)
        # Adjust these ranges based on your actual data
        value_ranges = {
            "Age": (18, 100),  # Student age range
            "CGPA": (0.0, 10.0),  # CGPA scale
        }

        # Preprocess data with imputation enabled
        df_clean = analyzer.preprocess_data(
            df,
            value_ranges=value_ranges,
            impute=True,
            numeric_strategy="median",
            categorical_strategy="mode",
        )

        # Display summary statistics
        summary = analyzer.get_summary_statistics(df_clean)
        logger.info(f"Summary Statistics:\n{summary}")

        logger.info("Analysis completed successfully!")

        # Department-level analysis
        dept_stats = analyzer.department_placement_stats()
        logger.info(f"Department placement stats:\n{dept_stats}")

        # Top-performing students
        top_students = analyzer.top_performers(
            cgpa_threshold=9.0,
            top_n=5,
        )
        logger.info(f"Top-performing students:\n{top_students}")

        # Grade distribution
        grade_dist = analyzer.grade_distribution()
        logger.info(f"Grade distribution:\n{grade_dist}")

        logger.info("Analysis pipeline executed successfully")

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)
