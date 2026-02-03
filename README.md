# 🎓 Student Performance Analysis System 

> A production-grade Python toolkit for analyzing university student placement data and generating reproducible visual insights.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Code Style](https://img.shields.io/badge/code%20style-clean-brightgreen.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Structure](#️-project-structure)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Analytics & Visualizations](#-analytics--visualizations)
- [Design Principles](#-design-principles)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

This project implements a **modular analytics pipeline** that cleanly separates data processing, analysis, and visualization concerns. Built with software engineering best practices, it's designed to be maintainable, testable, and production-ready.

**Perfect for:**
- Academic mini-projects and coursework
- Data science portfolio demonstrations
- Internal analytics tools
- Learning production-grade Python architecture

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📊 **Structured Data Pipeline** | Clean separation between ETL, analytics, and visualization layers |
| 🧮 **Pure Analytics Functions** | Stateless, reusable analysis modules with no side effects |
| 📈 **Figure-First Visualization** | Returns matplotlib `Figure` objects for maximum flexibility |
| 🧵 **Centralized Logging** | Professional logging setup with console and file outputs |
| 🗂️ **Scalable Architecture** | Modular design ready for growth and testing |
| 🔒 **Type Safety** | Type hints throughout for better IDE support and maintainability |

---

## 🏗️ Project Structure
```text
university-analytics-pipeline/
│
├── data/                           # Data directory
│   └── university_data.csv         # Raw placement dataset
│
├── logs/                           # Application logs
│   └── pipeline.log                # Execution logs with timestamps
│
├── visualizations/                 # Generated plots
│   ├── department_performance.png  # Department-wise metrics
│   ├── cgpa_vs_salary.png         # Correlation analysis
│   └── grade_distribution.png      # Grade breakdown
│
├── data_analysis.py               # 📊 Analytics layer
├── visualisation.py               # 📈 Visualization layer
├── pipeline.py                    # 🚀 Orchestrator & entry point
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 🔍 Architecture

This project follows a **layered architecture** pattern with clear separation of concerns:

### 1️⃣ **Analytics Layer** (`data_analysis.py`)

**Responsibilities:**
- Load and validate raw CSV data
- Handle missing values with statistical imputation (median/mode)
- Perform data quality checks and range validation
- Compute analytical summaries and aggregations

**Design Constraints:**
```python
✅ Pure functions (data in → data out)
✅ No visualization logic
✅ No logging configuration
✅ Type-annotated interfaces
❌ No plotting side effects
❌ No direct file I/O for outputs
```

**Key Functions:**
- `load_data()` - Robust CSV loading with validation
- `clean_data()` - Intelligent missing value imputation
- `department_placement_analysis()` - Aggregate department metrics
- `grade_distribution()` - Student performance categorization

---

### 2️⃣ **Visualization Layer** (`visualisation.py`)

**Responsibilities:**
- Transform analytical outputs into publication-ready plots
- Return `Figure` objects without rendering
- Provide utilities for saving figures to disk

**Design Constraints:**
```python
✅ Returns matplotlib.figure.Figure objects
✅ No plt.show() calls
✅ No logging configuration
✅ Customizable aesthetics
❌ No data transformation logic
❌ No direct analytics computations
```

**Key Functions:**
- `plot_department_performance()` - Bar charts with customizable metrics
- `plot_cgpa_vs_salary()` - Scatter plots with trend lines
- `plot_grade_distribution()` - Pie charts with percentage labels
- `save_figure()` - Safe file persistence with validation

---

### 3️⃣ **Orchestration Layer** (`pipeline.py`)

**Responsibilities:**
- **Single point of logging configuration**
- Coordinate analysis → visualization workflow
- Handle errors gracefully with proper exit codes
- Manage file I/O and directory creation

**This is the ONLY module that:**
```python
✅ Calls logging.config.dictConfig()
✅ Contains __main__ execution block
✅ Orchestrates cross-module workflows
✅ Handles top-level error recovery
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/Khushi-c-sharma/student-performance-analysis-system.git
cd student-performance-analysis-pipeline

# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install pandas numpy matplotlib
```

---

## 🚀 Usage

### Basic Execution
```bash
python pipeline.py
```

### Expected Output
```
2024-02-03 10:30:15 | INFO | Starting University Analytics Pipeline
2024-02-03 10:30:15 | INFO | Loading data from data/university_data.csv
2024-02-03 10:30:16 | INFO | Data cleaned: 500 records processed
2024-02-03 10:30:16 | INFO | Generating department performance analysis...
2024-02-03 10:30:17 | INFO | Creating visualizations...
2024-02-03 10:30:18 | INFO | Figure saved to visualizations/department_performance.png
2024-02-03 10:30:19 | INFO | Figure saved to visualizations/cgpa_vs_salary.png
2024-02-03 10:30:20 | INFO | Figure saved to visualizations/grade_distribution.png
2024-02-03 10:30:20 | INFO | Pipeline completed successfully
```

### Programmatic Usage
```python
from data_analysis import load_data, clean_data, department_placement_analysis
from visualisation import plot_department_performance, save_figure
from pathlib import Path

# Load and process data
df = load_data("data/university_data.csv")
df_clean = clean_data(df)

# Perform analysis
dept_stats = department_placement_analysis(df_clean)

# Generate visualization
fig = plot_department_performance(dept_stats, metric="Avg_Salary_LPA")
save_figure(fig, Path("output/my_chart.png"), dpi=300)
```

---

## 📊 Analytics & Visualizations

### Analytics Performed

| Analysis | Metrics | Output |
|----------|---------|--------|
| **Department Performance** | Avg. salary, placement rate, student count | DataFrame with dept-level aggregations |
| **CGPA-Salary Correlation** | Pearson correlation, linear regression | Scatter plot with trend line |
| **Grade Distribution** | Count & percentage by grade category | Categorical breakdown |
| **Data Quality Summary** | Missing values, outliers, data types | Statistical summary |

### Visualizations Generated

#### 1. Department Performance
![Department Performance](https://img.shields.io/badge/Chart-Bar-blue)
- **Type:** Horizontal/Vertical bar chart
- **Metrics:** Average salary, placement rate (configurable)
- **Features:** Grid lines, rotated labels, custom colors

#### 2. CGPA vs Salary
![CGPA Scatter](https://img.shields.io/badge/Chart-Scatter-green)
- **Type:** Scatter plot with regression line
- **Metrics:** Pearson correlation coefficient
- **Features:** Trend line, correlation annotation, transparency

#### 3. Grade Distribution
![Grade Pie](https://img.shields.io/badge/Chart-Pie-orange)
- **Type:** Pie chart
- **Metrics:** Percentage breakdown by grade
- **Features:** Auto-percentage labels, color scheme

---

## 🎯 Design Principles

### 1. **Separation of Concerns**
```
Data Layer → Analytics Layer → Visualization Layer → Orchestration
```
Each layer has a single, well-defined responsibility.

### 2. **Testability**
Pure functions with no side effects make unit testing straightforward:
```python
def test_department_analysis():
    sample_df = create_sample_data()
    result = department_placement_analysis(sample_df)
    assert result.shape[0] > 0
    assert "Avg_Salary_LPA" in result.columns
```

### 3. **Reusability**
Visualization functions return `Figure` objects, enabling:
- Integration into dashboards (Streamlit, Dash)
- Batch report generation
- A/B testing of different aesthetics
- Programmatic figure manipulation

### 4. **Logging Hygiene**
```python
# ✅ In library modules
logger = logging.getLogger(__name__)

# ❌ Never in library modules
logging.basicConfig(...)  # Only in pipeline.py
```

### 5. **Path Independence**
Using `pathlib.Path` ensures cross-platform compatibility:
```python
Path("data") / "file.csv"  # Works on Windows, Linux, macOS
```

---

## 🔮 Future Roadmap

### Phase 1: Enhanced Functionality
- [ ] CLI support with `argparse` or `click`
- [ ] Configuration management via YAML/JSON
- [ ] Export analytics to CSV/Excel/Parquet
- [ ] Interactive HTML reports

### Phase 2: Testing & Quality
- [ ] Unit tests with `pytest` (target: 80%+ coverage)
- [ ] Integration tests for full pipeline
- [ ] Pre-commit hooks with `black` and `flake8`
- [ ] Type checking with `mypy`

### Phase 3: Advanced Analytics
- [ ] Time-series analysis (multi-year trends)
- [ ] Predictive modeling (salary prediction)
- [ ] Statistical hypothesis testing
- [ ] Outlier detection and handling

### Phase 4: Deployment
- [ ] Docker containerization
- [ ] Streamlit web dashboard
- [ ] Scheduled execution with `cron`/`Airflow`
- [ ] Cloud deployment (AWS/GCP/Azure)

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Code Standards:**
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for public APIs
- Write tests for new features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Data Source:** Synthetic university placement dataset
- **Inspiration:** Production-grade data engineering practices
- **Tools:** Built with pandas, NumPy, and Matplotlib

---

## 📬 Contact

**Khushi Sharma**  
Data Science Enthusiast | Analytics Engineer | MLOps Practitioner

[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/Khushi-c-sharma)
[![Email](https://img.shields.io/badge/Email-Contact-red)](mailto:your.email@example.com)

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

Made with ❤️ and Python

</div>
