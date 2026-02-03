import logging
import logging.config
from pathlib import Path
import sys

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
    Execute the full analytics and visualization workflow.

    Pipeline steps:
    1. Load raw dataset from CSV
    2. Clean and preprocess data (validation, imputation)
    3. Compute analytical summaries
    4. Generate visualizations
    5. Persist figures to disk

    Raises:
        FileNotFoundError: If the dataset is missing
        Exception: For any unexpected runtime failure
    """
    logger.info("Starting analytics & visualization pipeline")

    # Initialize analysis component with dataset path
    analyzer = Analysis(CSV_PATH)

    # Load raw dataset
    df = analyzer.load_data()

    # Clean and preprocess dataset
    df_clean = analyzer.preprocess_data(
        df,
        value_ranges={
            "CGPA": (0.0, 10.0),
            "Placement_Package_LPA": (0.0, 100.0),
        },
    )

    dept_df = analyzer.department_placement_stats()
    grade_df = analyzer.grade_distribution()

    fig = plot_department_performance(dept_df)
    save_figure(fig, FIGURES_DIR / "department_performance.png")

    fig = plot_cgpa_vs_salary(df_clean)
    save_figure(fig, FIGURES_DIR / "cgpa_vs_salary.png")

    fig = plot_grade_distribution(grade_df)
    save_figure(fig, FIGURES_DIR / "grade_distribution.png")

    fig = plot_correlation_heatmap(
        df_clean,
        columns=["CGPA", "Placement_Package_LPA", "Age"],
    )
    save_figure(fig, FIGURES_DIR / "correlation_matrix.png")

    logger.info("Pipeline completed successfully")


# -------------------------------------------------
# Script entry point
# -------------------------------------------------

if __name__ == "__main__":
    try:
        run_pipeline()
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception:
        # Logs full traceback for post-mortem debugging
        logger.exception("Pipeline failed")
        sys.exit(1)
