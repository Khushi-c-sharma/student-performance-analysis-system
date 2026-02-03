"""
Generate a synthetic Indian university student dataset with placement outcomes.

Returns:
    pd.DataFrame: Student-level academic and placement data including
                  Placement_Package_LPA (Lakhs Per Annum).
"""

import pandas as pd
import random
from pathlib import Path
import logging


def generate_university_dataframe(num_records=100):
    """
    This script generates a synthetic dataset for an Indian university.

    Returns:
        pd.DataFrame: Includes 'Placement_Package_LPA' (Lakhs Per Annum).
    """
    random.seed(42)

    # --- Gender-Segregated Name Pools ---
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
    ]

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
    ]

    # Base salaries (in Lakhs) for different departments to simulate market trends
    dept_base_salary = {
        "Computer Science": 8.0,
        "Information Technology": 7.5,
        "Electronics": 6.0,
        "Biotechnology": 5.5,
        "Mechanical": 5.0,
        "Chemical Engineering": 5.0,
        "Civil": 4.5,
    }

    states = [
        "Maharashtra",
        "Tamil Nadu",
        "Delhi",
        "Uttar Pradesh",
        "Kerala",
        "Telangana",
        "West Bengal",
        "Karnataka",
    ]

    data = {
        "Student_ID": [],
        "Name": [],
        "Age": [],
        "Gender": [],
        "State of Residence": [],
        "Department": [],
        "CGPA": [],
        "Placed": [],
        "Placement_Package_LPA": [],
    }

    for i in range(1, num_records + 1):
        # 1. Identity Logic
        gender = random.choice(["M", "F"])
        first_name = (
            random.choice(male_names) if gender == "M" else random.choice(female_names)
        )
        full_name = f"{first_name} {random.choice(last_names)}"
        age = random.randint(18, 24)
        state = random.choice(states)

        # 2. Academic Logic
        dept = random.choice(list(dept_base_salary.keys()))
        cgpa = round(random.uniform(6.0, 9.8), 2)
        is_placed = cgpa > 8.5 or random.random() < 0.5

        # 3. Placement Package Logic (LPA)
        if is_placed:
            base = dept_base_salary[dept]
            # Performance bonus: 10% increase for every point above 7.0 CGPA
            multiplier = 1 + (max(0, cgpa - 7.0) * 0.15)
            package = round(min(base * multiplier, 25.0), 2)
        else:
            package = 0.0

        # Populate
        data["Student_ID"].append(1000 + i)
        data["Name"].append(full_name)
        data["Age"].append(age)
        data["Gender"].append(gender)
        data["State of Residence"].append(state)
        data["Department"].append(dept)
        data["CGPA"].append(cgpa)
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
        df = generate_university_dataframe(100)
        logger.info("Successfully generated 100 rows of university data.")

        # Save Data
        file_path = DATA_DIR / "university_data.csv"
        df.to_csv(file_path, index=False, encoding="utf-8")

        # Use an f-string in the logger to confirm the save location
        logger.info(f"DataFrame exported successfully to: {file_path}")

    except Exception as e:
        # If something goes wrong (e.g., Permission denied), log it as an error
        logger.error(f"An error occurred during the data generation/save process: {e}")
