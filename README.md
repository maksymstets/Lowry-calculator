# Lowry-calculator
An interactive web application for calculating protein concentrations from Lowry assay data. Built with Streamlit, it handles calibration curve fitting, sample concentration calculation, and PDF report generation.

---

## Features

- Interactive data entry for calibration standards and samples
- Linear regression of the calibration curve with R² quality feedback
- Automatic protein concentration calculation from up to three absorbance replicates
- Mean and standard deviation per sample
- Dilution factor support
- Warnings for negative absorbances and out-of-range concentrations
- Bar chart of results with error bars
- Downloadable PDF report with calibration curve equation, results table, and chart

---

## Requirements

- Python 3.13.9
- The following packages:
fpdf2==2.8.7
matplotlib==3.10.9
numpy==2.4.6
pandas==3.0.3
pytest==8.4.2
scipy==1.17.1
streamlit==1.51.0
```

Install all dependencies with:

```bash
pip install streamlit pandas numpy scipy matplotlib fpdf2
```

---

## Running the App

```bash
streamlit run calculator.py
```

The app will open in your browser at `http://localhost:....`.

---

## Usage

### 1. Enter calibration data

Fill in the calibration table with your standard concentrations (mg/mL) and their corresponding absorbance values. At least two points with different concentrations are required.

### 2. Enter sample data

Fill in the sample table with:
- **Sample name** — left blank defaults to `Unknown`
- **Dilution factor** — defaults to `1` if not changed
- **Absorbance1, 2, 3** — up to three replicates; at least one is required per sample

### 3. Calculate

Click **Calculate** to:
- Fit the calibration curve and display R²
- Calculate protein concentrations for all samples
- Display results as a table and bar chart

### 4. Download report

Click **Download PDF Report** to save a report containing the calibration equation, results table, and bar chart.

---

## Warnings

The app will notify you if:
- Fewer than two calibration points are entered
- All calibration concentrations are identical
- Negative absorbance values are detected in standards or samples
- Calculated protein concentrations are negative (sample likely below the detection range of the standards)

---


## Project Structure

```
.
├── calculator.py        # Main application
├── test_calculator.py   # Unit tests
└── README.md
```

---

## Limitations

- Sample names longer than approximately 20 characters may be truncated in the PDF table
- The app is designed for single-session use; no data is saved between sessions
- PDF report does not include the calibration curve plot, only the sample concentration chart and calibration equation
