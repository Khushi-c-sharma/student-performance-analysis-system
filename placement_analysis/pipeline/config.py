from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class FigureSpec:
    method_name: str
    output_name: str
    kwargs: Dict[str, Any] | None = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
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

FIGURE_REGISTRY = {
    "placement_overview": FigureSpec(
        method_name="plot_placement_overview",
        output_name="placement_overview.png",
    ),
    "dept_placement_rate": FigureSpec(
        method_name="plot_department_placement_rate",
        output_name="department_placement_rate.png",
    ),
    "cgpa_vs_package": FigureSpec(
        method_name="plot_cgpa_vs_package",
        output_name="cgpa_vs_package.png",
    ),
    "cgpa_distribution": FigureSpec(
        method_name="plot_cgpa_distribution",
        output_name="cgpa_distribution.png",
    ),
    "backlog_impact": FigureSpec(
        method_name="plot_backlog_impact",
        output_name="backlog_impact.png",
    ),
    "internship_impact": FigureSpec(
        method_name="plot_internship_impact",
        output_name="internship_impact.png",
    ),
    "gender_placement_outcomes": FigureSpec(
        method_name="plot_gender_placement_outcomes",
        output_name="gender_placement_outcomes.png",
    ),
    "department_heatmap": FigureSpec(
        method_name="plot_department_heatmap",
        output_name="department_heatmap.png",
    ),
    "gender_department_segmentation": FigureSpec(
        method_name="plot_gender_department_segmentation",
        output_name="gender_department_segmentation.png",
    ),
    "high_risk_distribution": FigureSpec(
        method_name="plot_high_risk_distribution",
        output_name="high_risk_distribution.png",
    ),
    "top_performers": FigureSpec(
        method_name="plot_top_performers_profile",
        output_name="top_performers.png",
    ),
}
