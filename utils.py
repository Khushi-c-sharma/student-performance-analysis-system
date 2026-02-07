import numpy as np
from typing import Dict
import random


def calculate_placement_probability(
    cgpa: float,
    dept: str,
    gender: str,
    has_internship: bool,
    num_backlogs: int,
    dept_placement_multiplier: Dict[str, float],
) -> float:
    """
    Calculate realistic placement probability based on multiple factors.

    This function models the complex relationship between academic performance,
    department, and other factors in determining placement outcomes.

    Args:
        cgpa: Student's CGPA (0-10 scale, can be NaN)
        dept: Department name (may have variations)
        gender: 'M' or 'F'
        has_internship: Whether student completed an internship
        num_backlogs: Number of backlogs/arrears
        dept_placement_multiplier: Dictionary mapping departments to placement rate multipliers

    Returns:
        float: Probability of placement (0.0 to 1.0)

    Notes:
        - Base probability ranges from 20% (CGPA=6.0) to 80% (CGPA=10.0)
        - Department multipliers reflect market demand
        - Internships increase probability significantly
        - Backlogs decrease probability
        - Small gender gap in tech placements (realistic but unfortunate)
    """
    # Handle missing CGPA (assume below-average probability)
    if np.isnan(cgpa):
        base_prob = 0.30
    else:
        # Linear scaling: 6.0 CGPA → 20%, 10.0 CGPA → 80%
        base_prob = 0.20 + (cgpa - 6.0) / 4.0 * 0.60
        base_prob = max(0.10, min(base_prob, 0.90))  # Clamp to reasonable range

    # Extract base department name (handle variants like "CSE", "Computer Science")
    base_dept = dept.split()[0] if " " in dept else dept

    # Map common variants to base names
    dept_mapping = {
        "CSE": "Computer Science",
        "CS": "Computer Science",
        "IT": "Information Technology",
        "ECE": "Electronics and Communication",
        "EEE": "Electrical Engineering",
        "ME": "Mechanical Engineering",
        "CE": "Civil Engineering",
    }

    base_dept = dept_mapping.get(base_dept, dept)

    # Apply department-wise placement rate multiplier
    dept_mult = dept_placement_multiplier.get(base_dept, 1.0)
    base_prob *= dept_mult

    # Internship boost (major factor in Indian placements)
    if has_internship:
        base_prob *= 1.25

    # Backlogs penalty (companies often have strict criteria)
    if num_backlogs > 0:
        penalty = 1.0 - (num_backlogs * 0.08)  # 8% reduction per backlog
        base_prob *= max(penalty, 0.40)  # Don't reduce below 40% of original

    # Gender gap in tech placements (small but realistic)
    # Women in tech face slightly lower placement rates in some regions
    if gender == "F" and base_dept in [
        "Computer Science",
        "Information Technology",
        "Data Science",
        "Artificial Intelligence",
    ]:
        base_prob *= 0.97  # 3% reduction

    # Cap probability at 95% (even top students aren't guaranteed)
    return min(base_prob, 0.95)


def generate_placement_package(
    is_placed: bool,
    dept: str,
    cgpa: float,
    has_internship: bool,
    dept_base_salary: Dict[str, float],
) -> float:
    """
    Generate realistic placement package (salary in Lakhs Per Annum).

    Salary generation considers department, CGPA, internship experience,
    and includes realistic outliers for top performers and premium companies.

    Args:
        is_placed: Whether student secured placement
        dept: Department name
        cgpa: Student's CGPA
        has_internship: Whether student has internship experience
        dept_base_salary: Dictionary mapping departments to base salaries

    Returns:
        float: Placement package in Lakhs Per Annum (0.0 if not placed)

    Notes:
        - Base salary varies by department (CS/IT highest)
        - CGPA has non-linear impact (stronger for high performers)
        - Internships add 0.5-1.5 LPA premium
        - 1-2% students get 15-30 LPA (product companies, startups)
        - 0.1% get 30-50 LPA (top-tier companies, international offers)
    """
    if not is_placed:
        return 0.0

    # Extract base department for salary lookup
    base_dept = dept.split()[0] if " " in dept else dept

    dept_mapping = {
        "CSE": "Computer Science",
        "CS": "Computer Science",
        "IT": "Information Technology",
        "ECE": "Electronics and Communication",
        "EEE": "Electrical Engineering",
        "ME": "Mechanical Engineering",
        "CE": "Civil Engineering",
    }

    base_dept = dept_mapping.get(base_dept, dept)

    # Get base salary for department (default to 6.0 LPA if not found)
    base_salary = dept_base_salary.get(base_dept, 6.0)

    # Calculate CGPA boost (non-linear, stronger for tech roles)
    if not np.isnan(cgpa):
        # Tech departments have stronger CGPA impact
        if base_dept in [
            "Computer Science",
            "Data Science",
            "Artificial Intelligence",
            "Information Technology",
        ]:
            cgpa_boost = max(0, (cgpa - 7.0)) * 0.30  # 30% boost per CGPA point above 7
        else:
            cgpa_boost = max(0, (cgpa - 7.0)) * 0.18  # 18% boost for other departments
    else:
        cgpa_boost = -0.3  # Penalty for missing CGPA

    # Internship premium (0.5-1.5 LPA additional)
    internship_premium = random.uniform(0.5, 1.5) if has_internship else 0.0

    # Random noise (interview performance, company tier, negotiation)
    # Represents factors beyond academic performance
    noise = random.uniform(-0.8, 1.8)

    # Calculate final package
    package = base_salary * (1 + cgpa_boost) + internship_premium + noise

    # Enforce minimum package floor (mass recruiters offer ~3.5-4 LPA minimum)
    package = max(package, 3.5)

    # ========================================================================
    # Generate realistic outliers
    # ========================================================================

    # High packages (15-30 LPA): Product companies, funded startups, niche roles
    # 1.5% of placed students, requires CGPA > 8.0
    if random.random() < 0.015 and (np.isnan(cgpa) or cgpa > 8.0):
        package = round(random.uniform(15, 30), 2)

    # Very high packages (30-50 LPA): Top-tier companies, international offers
    # 0.1% of placed students, requires CGPA > 9.0
    elif random.random() < 0.001 and (not np.isnan(cgpa) and cgpa > 9.0):
        package = round(random.uniform(30, 50), 2)

    return round(package, 2)
