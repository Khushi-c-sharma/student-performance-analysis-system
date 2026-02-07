import pandas as pd
from pathlib import Path

import logging
from typing import Dict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "placement_data.csv"

PROCESSED_DATA_DIR = PROJECT_ROOT / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


DEPARTMENT_MAPPING = {
    "Cse": "Computer Science",
    "Cs": "Computer Science",
    "Comp Sci": "Computer Science",
    "It": "Information Technology",
    "Info Tech": "Information Technology",
    "Ece": "Electronics And Communication",
    "E&C": "Electronics And Communication",
    "Electronics": "Electronics And Communication",
    "Eee": "Electrical Engineering",
    "Ee": "Electrical Engineering",
    "Electrical": "Electrical Engineering",
    "Me": "Mechanical Engineering",
    "Mech": "Mechanical Engineering",
    "Mechanical": "Mechanical Engineering",
    "Ce": "Civil Engineering",
    "Civil": "Civil Engineering",
}

STATE_MAPPING = {
    "Mh": "Maharashtra",
    "Maharashtra": "Maharashtra",
    "Tn": "Tamil Nadu",
    "Tamil Nadu": "Tamil Nadu",
    "Tamilnadu": "Tamil Nadu",
    "Up": "Uttar Pradesh",
    "U.P.": "Uttar Pradesh",
    "Wb": "West Bengal",
    "W Bengal": "West Bengal",
    "Ap": "Andhra Pradesh",
    "A.P.": "Andhra Pradesh",
}


class PlacementDataCleaningPipeline:
    """
    End-to-end data cleaning pipeline for the student placement dataset.

    Responsibilities:
    - Load and validate raw data
    - Standardize categorical fields
    - Fix logical inconsistencies
    - Handle missing values safely
    """

    def __init__(
        self,
        csv_path: str,
        department_mapping: Dict[str, str],
        state_mapping: Dict[str, str],
    ) -> None:
        """
        Initialize the pipeline with configuration.

        Args:
            csv_path (str): Path to the input CSV file
            department_mapping (Dict[str, str]): Department normalization mapping
            state_mapping (Dict[str, str]): State normalization mapping
        """
        self.csv_path = csv_path
        self.department_mapping = department_mapping
        self.state_mapping = state_mapping
        self.df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Load dataset from disk."""
        try:
            self.df = pd.read_csv(self.csv_path)
            logger.info("Dataset loaded successfully")
        except Exception as e:
            logger.exception("Failed to load dataset")
            raise e

    # ------------------------------------------------------------------
    # Diagnostics & Validation
    # ------------------------------------------------------------------

    def log_basic_diagnostics(self) -> None:
        """Log shape, missing values, and duplicate counts."""
        try:
            df = self.df
            logger.info("Dataset shape: %s", df.shape)
            logger.info("Missing values summary:\n%s", df.isna().sum())

            logger.info(
                "Duplicates (Name, Department, CGPA): %d",
                df.duplicated(subset=["Name", "Department", "CGPA"]).sum(),
            )
            logger.info(
                "Duplicates (Name, Age, Department, CGPA): %d",
                df.duplicated(subset=["Name", "Age", "Department", "CGPA"]).sum(),
            )
        except Exception as e:
            logger.exception("Error during diagnostics")
            raise e

    def validate_numeric_ranges(self) -> None:
        """Validate numeric ranges and log anomalies."""
        try:
            df = self.df

            logger.info(
                "Invalid CGPA count: %d",
                df[(df["CGPA"] < 0) | (df["CGPA"] > 10)].shape[0],
            )
            logger.info(
                "Invalid Age count: %d",
                df[(df["Age"] < 17) | (df["Age"] > 30)].shape[0],
            )
            logger.info(
                "Invalid 10th Percentage count: %d",
                df[(df["Tenth_Percentage"] < 0) | (df["Tenth_Percentage"] > 100)].shape[
                    0
                ],
            )
            logger.info(
                "Invalid 12th Percentage count: %d",
                df[
                    (df["Twelfth_Percentage"] < 0) | (df["Twelfth_Percentage"] > 100)
                ].shape[0],
            )
            logger.info(
                "Negative placement packages count: %d",
                df[df["Placement_Package_LPA"] < 0].shape[0],
            )

        except Exception as e:
            logger.exception("Numeric validation failed")
            raise e

    # ------------------------------------------------------------------
    # Standardization
    # ------------------------------------------------------------------

    def standardize_departments(self) -> None:
        """Normalize department names."""
        try:
            df = self.df
            before = df["Department"].nunique(dropna=True)

            df["Department"] = (
                df["Department"]
                .astype(str)
                .str.strip()
                .str.title()
                .replace(self.department_mapping)
            )

            after = df["Department"].nunique(dropna=True)
            logger.info("Departments standardized: %d → %d", before, after)

        except Exception as e:
            logger.exception("Department standardization failed")
            raise e

    def standardize_states(self) -> None:
        """Normalize state names."""
        try:
            df = self.df
            before = df["State_of_Residence"].nunique(dropna=True)

            df["State_of_Residence"] = (
                df["State_of_Residence"]
                .astype(str)
                .str.strip()
                .str.title()
                .replace(self.state_mapping)
            )

            after = df["State_of_Residence"].nunique(dropna=True)
            logger.info("States standardized: %d → %d", before, after)

        except Exception as e:
            logger.exception("State standardization failed")
            raise e

    def standardize_names(self) -> None:
        """Clean and normalize student names."""
        try:
            self.df["Name"] = self.df["Name"].astype(str).str.strip().str.title()
            logger.info("Student names standardized")
        except Exception as e:
            logger.exception("Name standardization failed")
            raise e

    # ------------------------------------------------------------------
    # Logical Consistency
    # ------------------------------------------------------------------

    def fix_placement_anomalies(self) -> None:
        """
        Fix inconsistencies between placement status and package values.
        """
        try:
            df = self.df

            mask = (df["Placed"] == True) & (
                (df["Placement_Package_LPA"] == 0)
                | (df["Placement_Package_LPA"].isna())
            )

            logger.info(
                "Placed students with missing/zero package: %d",
                mask.sum(),
            )

            dept_median = (
                df[(df["Placed"] == True) & (df["Placement_Package_LPA"] > 0)]
                .groupby("Department")["Placement_Package_LPA"]
                .median()
            )

            df.loc[mask, "Placement_Package_LPA"] = (
                df.loc[mask, "Department"]
                .map(dept_median)
                .fillna(6.0)  # documented fallback
            )

            inconsistent_unplaced = (
                (df["Placed"] == False) & (df["Placement_Package_LPA"] > 0)
            ).sum()

            logger.info(
                "Unplaced students with non-zero package: %d",
                inconsistent_unplaced,
            )

        except Exception as e:
            logger.exception("Placement anomaly correction failed")
            raise e

    # ------------------------------------------------------------------
    # Missing Value Handling
    # ------------------------------------------------------------------

    def impute_missing_values(self) -> None:
        """Impute missing values using robust statistical strategies."""
        try:
            df = self.df

            # Percentages → median
            for col in ("Tenth_Percentage", "Twelfth_Percentage"):
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())

            # CGPA → department median → global median
            dept_median = df.groupby("Department")["CGPA"].transform("median")
            df["CGPA"] = df["CGPA"].fillna(dept_median)
            df["CGPA"] = df["CGPA"].fillna(df["CGPA"].median())

            logger.info("Missing values imputed successfully")
            logger.info("Remaining null values:\n%s", df.isna().sum())

        except Exception as e:
            logger.exception("Missing value imputation failed")
            raise e

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_cleaned_data(self, output_path: str, index: bool = False) -> None:
        """
        Save the cleaned dataset to disk.

        Args:
            output_path (str): Path where the cleaned CSV will be saved
            index (bool): Whether to write row indices to file
        """
        try:
            if self.df is None:
                raise ValueError("No data available to save. Run cleaning steps first.")

            self.df.to_csv(output_path, index=index)
            logger.info("Cleaned data saved successfully at: %s", output_path)

        except Exception as e:
            logger.exception("Failed to save cleaned data")
            raise e


if __name__ == "__main__":
    try:
        logger.info("STARTING DATA CLEANING PIPELINE")

        cleaner = PlacementDataCleaningPipeline(
            csv_path=CSV_PATH,
            department_mapping=DEPARTMENT_MAPPING,
            state_mapping=STATE_MAPPING,
        )

        # Step 1: Load
        cleaner.load_data()

        # Step 2: Diagnostics & validation
        cleaner.log_basic_diagnostics()
        cleaner.validate_numeric_ranges()

        # Step 3: Standardization
        cleaner.standardize_departments()
        cleaner.standardize_states()
        cleaner.standardize_names()

        # Step 4: Logical consistency fixes
        cleaner.fix_placement_anomalies()

        # Step 5: Missing value imputation
        cleaner.impute_missing_values()

        # Step 6: Persist cleaned data
        OUTPUT_PATH = PROCESSED_DATA_DIR / "cleaned_placement_data.csv"
        cleaner.save_cleaned_data(OUTPUT_PATH)

        logger.info("DATA CLEANING PIPELINE COMPLETED SUCCESSFULLY")

    except Exception as e:
        logger.critical("Pipeline execution failed")
        raise e
