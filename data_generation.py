import numpy as np
import pandas as pd
import random
from pathlib import Path
import logging
from utils import calculate_placement_probability, generate_placement_package


def generate_university_dataframe(num_records: int = 100, seed: int = 42):
    """
    Generate a synthetic Indian university student placement dataset.

    This function creates realistic student records with academic performance,
    demographic information, and placement outcomes. It includes realistic
    correlations (e.g., CGPA vs placement probability) and common data quality
    issues (missing values, outliers, inconsistent formatting).
    """
    random.seed(seed)

    # Gender-segregated first names (common Indian names)
    male_names = [
        "Aarav",
        "Ishaan",
        "Vivek",
        "Arjun",
        "Aditya",
        "Rohan",
        "Kabir",
        "Rahul",
        "Aryan",
        "Yash",
        "Amit",
        "Vikram",
        "Karan",
        "Deepak",
        "Siddharth",
        "Abhimanyu",
        "Vedant",
        "Swaraj",
        "Pranjal",
        "Harsh",
        "Nikhil",
        "Pranav",
        "Raghav",
        "Shubham",
        "Varun",
        "Ayush",
        "Manav",
    ]

    female_names = [
        "Ananya",
        "Sana",
        "Meera",
        "Priyanka",
        "Riya",
        "Sneha",
        "Tanvi",
        "Kavya",
        "Diya",
        "Ishani",
        "Pooja",
        "Neha",
        "Simran",
        "Anjali",
        "Ishita",
        "Naina",
        "Muskaan",
        "Khushi",
        "Pranjal",
        "Aditi",
        "Shreya",
        "Sakshi",
        "Divya",
        "Ritika",
        "Pallavi",
        "Kritika",
    ]

    # Common Indian surnames
    last_names = [
        "Sharma",
        "Iyer",
        "Gupta",
        "Khan",
        "Deshmukh",
        "Nair",
        "Reddy",
        "Das",
        "Verma",
        "Kapoor",
        "Bhattacharjee",
        "Deshpande",
        "Pande",
        "Singh",
        "Patel",
        "Chatterjee",
        "Joshi",
        "Malhotra",
        "Choudhury",
        "Bose",
        "Kulkarni",
        "Menon",
        "Pillai",
        "Mahajan",
        "Agarwal",
        "Sinha",
        "Rao",
        "Kumar",
        "Saxena",
        "Mishra",
    ]

    # Base salary expectations by department (in Lakhs Per Annum)
    # Reflects Indian job market trends as of 2024-25
    dept_base_salary = {
        "Computer Science": 8.5,
        "Data Science": 9.0,
        "Artificial Intelligence": 9.5,
        "Information Technology": 8.0,
        "Electronics and Communication": 6.5,
        "Electrical Engineering": 6.0,
        "Mechanical Engineering": 5.5,
        "Civil Engineering": 5.0,
        "Chemical Engineering": 5.5,
        "Biotechnology": 5.0,
        "Metallurgy": 5.0,
    }

    # Department name variations (simulates data entry inconsistencies)
    dept_variants = {
        "Computer Science": ["Computer Science", "CSE", "Comp Sci", "CS"],
        "Information Technology": ["Information Technology", "IT", "Info Tech"],
        "Electronics and Communication": ["ECE", "Electronics", "E&C"],
        "Electrical Engineering": ["EEE", "Electrical", "EE"],
        "Mechanical Engineering": ["Mechanical", "ME", "Mech"],
        "Civil Engineering": ["Civil", "CE"],
    }

    # Department-wise placement rate multipliers
    # Tech departments have higher placement rates
    dept_placement_multiplier = {
        "Computer Science": 1.35,
        "Data Science": 1.40,
        "Artificial Intelligence": 1.38,
        "Information Technology": 1.25,
        "Electronics and Communication": 1.05,
        "Electrical Engineering": 0.95,
        "Mechanical Engineering": 0.85,
        "Civil Engineering": 0.70,
        "Chemical Engineering": 0.80,
        "Biotechnology": 0.65,
        "Metallurgy": 0.75,
    }

    # Indian states with common variations
    states = [
        "Maharashtra",
        "Tamil Nadu",
        "Delhi",
        "Uttar Pradesh",
        "Kerala",
        "Telangana",
        "West Bengal",
        "Karnataka",
        "Gujarat",
        "Rajasthan",
        "Andhra Pradesh",
        "Punjab",
        "Haryana",
        "Madhya Pradesh",
    ]

    state_variants = {
        "Maharashtra": ["Maharashtra", "MH", "maharashtra"],
        "Tamil Nadu": ["Tamil Nadu", "Tamil nadu", "TN", "Tamilnadu"],
        "Uttar Pradesh": ["Uttar Pradesh", "UP", "U.P."],
        "West Bengal": ["West Bengal", "WB", "W Bengal"],
        "Andhra Pradesh": ["Andhra Pradesh", "AP", "A.P."],
    }

    # Admission categories (realistic distribution in Indian universities)
    categories = ["General", "OBC", "SC", "ST", "EWS"]
    category_weights = [0.40, 0.30, 0.15, 0.10, 0.05]

    data = {
        "Student_ID": [],
        "Name": [],
        "Age": [],
        "Gender": [],
        "State_of_Residence": [],
        "Category": [],
        "Department": [],
        "CGPA": [],
        "Tenth_Percentage": [],
        "Twelfth_Percentage": [],
        "Backlogs": [],
        "Has_Internship": [],
        "Has_Gap_Year": [],
        "Placed": [],
        "Placement_Package_LPA": [],
    }

    for i in range(1, num_records + 1):
        # 1. Identity Logic
        gender = random.choice(["M", "F"])
        first_name = (
            random.choice(male_names) if gender == "M" else random.choice(female_names)
        )
        last_name = random.choice(last_names)
        full_name = f"{first_name} {last_name}"

        # Add middle initials for realism (15% of students)
        if random.random() < 0.15:
            middle_initial = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            full_name = f"{first_name} {middle_initial}. {last_name}"

        # Simulate name formatting inconsistencies (data quality issues)
        name_format_rand = random.random()
        if name_format_rand < 0.03:  # 3% all uppercase
            full_name = full_name.upper()
        elif name_format_rand < 0.05:  # 2% all lowercase
            full_name = full_name.lower()

        # Simulate duplicate entries (data entry errors, ~2%)
        if i > 10 and random.random() < 0.02:
            duplicate_idx = random.randint(0, len(data["Name"]) - 1)
            full_name = data["Name"][duplicate_idx]

        # Rare age outliers: younger (advanced entry) or older (gap years)
        has_gap_year = False  # IMPORTANT: initialize to avoid runtime error

        age_rand = random.random()
        if age_rand < 0.02:
            age = 17  # Early admission / prodigy
        elif age_rand < 0.10:
            age = random.randint(25, 27)
            has_gap_year = True
        else:
            age = random.randint(18, 24)

        # State of residence
        state = random.choice(states)

        # Apply state name variations (10% inconsistent labels)
        if state in state_variants and random.random() < 0.10:
            state = random.choice(state_variants[state])

        # Admission category
        category = random.choices(categories, weights=category_weights)[0]

        # 2. Generate academic information
        # Department selection
        dept = random.choice(list(dept_base_salary.keys()))

        # Apply department label variations (8% inconsistent naming)
        if dept in dept_variants and random.random() < 0.08:
            dept = random.choice(dept_variants[dept])

        # CGPA generation (6.0-10.0 scale, realistic distribution)
        # Most students cluster around 7.0-8.5
        cgpa = round(random.triangular(6.0, 9.8, 7.5), 2)

        # Introduce missing CGPA values (7% - students who dropped out or data issues)
        if random.random() < 0.07:
            cgpa = np.nan
        # Rare outliers below scale or above scale (data entry errors)
        elif random.random() < 0.02:
            cgpa = round(random.uniform(4.5, 10.5), 2)

        # Age-CGPA correlation: older students often have lower CGPA
        if age > 24 and not np.isnan(cgpa):
            cgpa = min(cgpa, round(random.uniform(6.0, 7.8), 2))

        # 10th percentage (60-95%, correlated with CGPA)
        if not np.isnan(cgpa) and cgpa > 8.0:
            tenth_pct = round(random.uniform(75, 95), 2)
        else:
            tenth_pct = round(random.uniform(60, 88), 2)

        # Rare missing 10th marks (3%)
        if random.random() < 0.03:
            tenth_pct = np.nan

        # 12th percentage (55-95%, stronger correlation with CGPA)
        if not np.isnan(cgpa) and cgpa > 8.5:
            twelfth_pct = round(random.uniform(80, 95), 2)
        elif not np.isnan(cgpa) and cgpa > 7.5:
            twelfth_pct = round(random.uniform(70, 90), 2)
        else:
            twelfth_pct = round(random.uniform(55, 85), 2)

        # Rare missing 12th marks (3%)
        if random.random() < 0.03:
            twelfth_pct = np.nan

        # Backlogs (arrears/failed subjects)
        # Students with lower CGPA more likely to have backlogs
        if np.isnan(cgpa) or cgpa < 6.5:
            num_backlogs = random.choices(
                [0, 1, 2, 3, 4, 5], weights=[0.25, 0.25, 0.20, 0.15, 0.10, 0.05]
            )[0]
        elif cgpa < 7.5:
            num_backlogs = random.choices(
                [0, 1, 2, 3], weights=[0.50, 0.30, 0.15, 0.05]
            )[0]
        else:
            num_backlogs = random.choices([0, 1], weights=[0.90, 0.10])[0]

        # Logical consistency: students with many backlogs have lower CGPA
        if num_backlogs > 2 and not np.isnan(cgpa):
            cgpa = min(cgpa, 7.5)

        # Internship completion (strongly correlated with CGPA and placement)
        if not np.isnan(cgpa) and cgpa > 8.0:
            has_internship = random.random() < 0.75  # 75% of good students
        elif not np.isnan(cgpa) and cgpa > 7.0:
            has_internship = random.random() < 0.50  # 50% of average students
        else:
            has_internship = random.random() < 0.25  # 25% of weak students

        # 3. Calculate placement probability
        placement_prob = calculate_placement_probability(
            cgpa=cgpa,
            dept=dept,
            gender=gender,
            has_internship=has_internship,
            num_backlogs=num_backlogs,
            dept_placement_multiplier=dept_placement_multiplier,
        )

        # Determine if student is placed
        is_placed = random.random() < placement_prob

        # 4. Placement Package Logic (LPA)
        package = generate_placement_package(
            is_placed=is_placed,
            dept=dept,
            cgpa=cgpa,
            has_internship=has_internship,
            dept_base_salary=dept_base_salary,
        )

        # 5. Introduce data quality issues
        # Logical error: Student marked as placed but package is 0 or missing
        # (Offer not disclosed, delayed joining, or data entry error)
        if is_placed and random.random() < 0.03:
            if random.random() < 0.5:
                package = np.nan  # Missing package data
            else:
                package = 0.0  # Package not disclosed

        # Logical error: Student not placed but has package
        # (Off-campus placement not recorded properly)
        if not is_placed and random.random() < 0.015:
            package = round(random.uniform(3.5, 6.5), 2)
            # Note: We keep is_placed as False to simulate data inconsistency

        # 6. Populate the records
        data["Student_ID"].append(1000 + i)
        data["Name"].append(full_name)
        data["Age"].append(age)
        data["Gender"].append(gender)
        data["State_of_Residence"].append(state)
        data["Category"].append(category)
        data["Department"].append(dept)
        data["CGPA"].append(cgpa)
        data["Tenth_Percentage"].append(tenth_pct)
        data["Twelfth_Percentage"].append(twelfth_pct)
        data["Backlogs"].append(num_backlogs)
        data["Has_Internship"].append(has_internship)
        data["Has_Gap_Year"].append(has_gap_year)
        data["Placed"].append(is_placed)
        data["Placement_Package_LPA"].append(package)

    return pd.DataFrame(data)


if __name__ == "__main__":
    # This sets the format to include the timestamp, the level of importance, and the message.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # --- Directory Setup ---
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / "data"

    try:
        # Check if directory exists, if not, create it
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(parents=True)
            logger.info(f"Created new directory at: {DATA_DIR}")
        else:
            logger.info(f"Directory already exists at: {DATA_DIR}")

        # Generate Data
        df = generate_university_dataframe(seed=0)
        logger.info("Successfully generated 100 rows of university data.")

        # Save Data
        file_path = DATA_DIR / "placement_data.csv"
        df.to_csv(file_path, index=False, encoding="utf-8")

        # Use an f-string in the logger to confirm the save location
        logger.info(f"DataFrame exported successfully to: {file_path}")

    except Exception as e:
        # If something goes wrong (e.g., Permission denied), log it as an error
        logger.error(f"An error occurred during the data generation/save process: {e}")
