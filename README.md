# ☀️ Solar PV Data Analysis & Forecasting (2021–2030)
### 📊 Chapter 4: Data Analysis and Findings – South Africa

---

## 🧭 Overview
This project presents the **data analysis, forecasting, and visualization** of solar photovoltaic (PV) generation in **South Africa (2021–2030)**.  
The Python workflow explores:
- 📈 Growth and trends in PV generation  
- 🤖 Predictive modeling using Prophet and Random Forest  
- 🌤️ Solar irradiance distribution  
- 🔍 Correlation between capacity and energy output  
- 📉 Seasonal and performance variations  

This analysis supports sustainable energy research and aligns with **South Africa’s renewable energy transition goals**.

---

## ⚙️ Technologies & Libraries
Developed and tested using **Python 3.13** 🐍  
Before running, ensure these libraries are installed:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn prophet openpyxl statsmodels

🧩 Project Structure
Chapter4/
│
├── 📂 Data/
│   └── ESK-Solar PV Data.xlsx              # Raw dataset
│
├── 📜 Chapter4_Analysis.py                 # Main analysis script
├── 📜 Histogram.py                         # Script for Figure 4.8 (Solar Irradiance)
├── 📊 Chapter4_Figures/                    # Auto-generated charts & plots
│   ├── Figure_4_1.png
│   ├── Figure_4_2.png
│   ├── ...
│
├── 📑 Summary_Table.csv                    # Yearly PV growth data
├── 📈 PV_Forecast_2026_2030.csv            # Prophet forecast output
├── 💻 Dashboard.html                       # Interactive Plotly visualization
└── 🧾 README.md                            # Documentation (this file)

## 📉 Key Figures & Outputs

The following figures illustrate the analytical outcomes and visual insights generated from the dataset.  
All figures are automatically saved to the folder: `Chapter4_Figures/`.

| **Figure** | **Title** | **Description** |
|:------------|:-----------|:----------------|
| **Figure 4.1** | Workflow Diagram | Overview of analysis and forecasting pipeline |
| **Figure 4.2** | Sample Dataset | Structure of solar PV data |
| **Figure 4.3** | Data Points Count | Summary of cleaned and validated records |
| **Figure 4.4** | PV Production (5 Years) | Historical solar PV trend (2021–2025) |
| **Figure 4.5–4.6** | Distribution & Boxplots | Visual representation of PV generation variability |
| **Figure 4.7** | Boxplot of PV Output | Outlier detection and distribution visualization |
| **Figure 4.8** | Histogram of Solar Irradiance ☀️ | Frequency of irradiance (kWh/m²/day) |
| **Figure 4.9–4.10** | Yearly PV Growth | Annual total PV output and growth rate trends |
| **Figure 4.11–4.12** | Forecast 2026–2030 | Prophet model predictions and long-term trend |
| **Figure 4.13–4.14** | Random Forest | Model performance and feature importance ranking |
| **Figure 4.15** | Seasonal Decomposition | Trend, seasonality, and residual analysis |
| **Figure 4.16** | Correlation Heatmap 🔥 | Interrelationships among key performance variables |

---

## 🧮 Analytical Workflow

A six-step methodology was followed to ensure rigorous, repeatable, and transparent analysis.

### 🧹 Step 1: Data Cleaning
- Handled missing values using **median imputation**  
- Removed outliers based on **Z-score threshold**  
- Filtered analysis period to **2021–2025**

### 📊 Step 2: Descriptive Analysis
- Computed **summary statistics** (mean, median, range)  
- Visualized data using **histograms**, **boxplots**, and **growth curves**

### 🕒 Step 3: Trend and Growth Analysis
- Calculated **annual totals** and **percentage growth rates**  
- Analyzed **monthly averages** and **seasonal variations**

### 🤖 Step 4: Forecasting (2026–2030)
- Applied **Prophet time-series model** for 5-year prediction  
- Generated **upper and lower confidence intervals** for expected PV output  

### 🧠 Step 5: Machine Learning
- Trained **Linear Regression** and **Random Forest Regressor** models  
- Evaluated using **RMSE** and **R²** performance metrics  
- Identified **top predictive features** influencing PV output

### 🌤️ Step 6: Visualization
- Created and saved all figures automatically in `/Chapter4_Figures/`  
- Exported an **interactive HTML dashboard** for visual data exploration  

---

## 📈 Key Insights

### ✨ Findings Summary
- 📅 **Steady Growth (2021–2025):** Solar PV generation showed a consistent upward trajectory, indicating strong adoption and utilization of renewable energy technologies.  
- ⚠️ **Forecasted Dip in 2025:** A projected decline to approximately **2.06 million MWh** suggests short-term variability possibly due to weather anomalies, system degradation, or reduced solar availability.  
- 🚀 **Future Outlook (2026–2030):** PV generation is predicted to **increase steadily**, maintaining a positive long-term growth trend.  
- ☀️ **Performance Drivers:** *Solar irradiance* and *installed capacity* were the two most influential variables driving PV output.  
- 🤖 **Model Performance:** The **Random Forest model** outperformed **Linear Regression**, achieving higher accuracy and R² scores.  
- 🔁 **Seasonal Cycles:** Decomposition analysis revealed **strong annual and weather-linked periodic patterns** influencing solar PV generation.  

---

### 🧩 Interpretation
The combined analytical workflow demonstrates the value of **machine learning** and **time-series modeling** in understanding renewable energy dynamics.  
The results emphasize the importance of **data-driven decision-making** for improving energy forecasting, infrastructure planning, and sustainable development in South Africa’s solar sector.

---

> 🪪 *These figures and findings form part of Chapter 4 of the research report: “Data Analysis and Forecasting of Solar PV Electricity Generation in South Africa (2021–2030)” by Yolisa Qadi (2025).*


