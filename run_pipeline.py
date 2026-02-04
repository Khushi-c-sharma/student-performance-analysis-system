import logging
import logging.config
from pathlib import Path

from data_analysis import Analysis
from visualisation import (
    plot_department_performance,
    plot_cgpa_vs_salary,
    plot_grade_distribution,
    save_figure,
    plot_correlation_heatmap,
)


# Resolve project root dynamically (safe for CLI execution)
PROJECT_ROOT = Path(__file__).resolve().parent

# Directory for log files
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Log file path
LOG_FILE = LOG_DIR / "pipeline.log"

# Dataset location
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "university_data.csv"

# Output directory for preprocessed data
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "cleaned_data.csv"


# Output directory for generated visualizations
FIGURES_DIR = PROJECT_ROOT / "visualizations"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Centralized logging configuration (APPLICATION LEVEL)
# -------------------------------------------------

LOGGING_CONFIG = {
    "version": 1,
    # Allow library loggers (analysis, visualization) to propagate
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        # Console output for real-time feedback
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        # Persistent file logs for auditing and debugging
        "file": {
            "class": "logging.FileHandler",
            "filename": str(LOG_FILE),
            "formatter": "standard",
            "level": "INFO",
        },
    },
    # Root logger applies to the entire application
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
}

# Apply logging configuration ONCE (entry point only)
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """
    Execute the end-to-end analytics & visualization pipeline.

    Stages:
    1. Data ingestion
    2. Data validation & preprocessing
    3. Analytical computation
    4. Visualization generation
    5. Artifact persistence

    Raises:
        FileNotFoundError: If the dataset is missing
        Exception: For any unexpected runtime failure
    """
    logger.info("Starting analytics & visualization pipeline")
    try:
        # Initialize analysis component with dataset path
        analyzer = Analysis(CSV_PATH)
        logger.info("Stage 1: Loading raw dataset")

        # Load raw dataset
        df = analyzer.load_data()

        logger.info("Stage 2: Preprocessing & data validation")

        # Clean and preprocess dataset
        df_clean = analyzer.preprocess_data(
            df,
            value_ranges={
                "CGPA": (0.0, 10.0),
                "Placement_Package_LPA": (0.0, 100.0),
            },
            impute=True,
            numeric_strategy="median",
            categorical_strategy="mode",
        )

        # Save the cleaned dataset
        df_clean.to_csv(PROCESSED_CSV_PATH, index=False)
        logger.info(
            "Preprocessed dataset saved to %s",
            PROCESSED_CSV_PATH,
        )

        logger.info("Stage 3: Computing analytical summaries")

        summary_stats = analyzer.get_summary_statistics(df_clean)
        dept_stats = analyzer.department_placement_stats(df_clean)
        grade_dist = analyzer.grade_distribution(
            df=df_clean,
            bins=(0.0, 6.0, 7.5, 10.0),
            labels=("Pass", "Merit", "Distinction"),
        )
        top_students = analyzer.top_performers(
            top_n=5,
            df=df_clean,
        )

        logger.info("Summary statistics computed")
        logger.info("Department placement stats computed")
        logger.info("Grade distribution computed")
        logger.info("Top-performing students identified")

        logger.debug("Summary statistics:\n%s", summary_stats)
        logger.debug("Department stats:\n%s", dept_stats)
        logger.debug("Grade distribution:\n%s", grade_dist)
        logger.debug("Top performers:\n%s", top_students)

        logger.info("Stage 4: Generating visualizations")

        fig = plot_department_performance(dept_stats)
        save_figure(fig, FIGURES_DIR / "department_performance.png")

        fig = plot_cgpa_vs_salary(df_clean)
        save_figure(fig, FIGURES_DIR / "cgpa_vs_salary.png")

        fig = plot_grade_distribution(grade_dist)
        save_figure(fig, FIGURES_DIR / "grade_distribution.png")

        fig = plot_correlation_heatmap(
            df_clean,
            columns=["CGPA", "Placement_Package_LPA", "Age"],
        )
        save_figure(fig, FIGURES_DIR / "correlation_matrix.png")

        logger.info("All visualizations generated and saved")

        logger.info("Pipeline completed successfully")

    except FileNotFoundError as e:
        logger.error("Dataset not found: %s", e)
        raise

    except Exception as e:
        logger.exception("Pipeline execution failed due to an unexpected error")
        raise RuntimeError("Analytics pipeline failed") from e


# -------------------------------------------------
# Script entry point
# -------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
