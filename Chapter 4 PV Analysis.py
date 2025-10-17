# ----------------------------------------------
# Chapter 4: Data Analysis and Findings – Solar PV South Africa
# ----------------------------------------------

# -------------------------------
# Installed Python Libraries
# -------------------------------
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from prophet import Prophet
import openpyxl
from statsmodels.tsa.seasonal import seasonal_decompose

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# -------------------------------
# Helper function for saving figures
# -------------------------------
output_dir = "Chapter4_Figures"
os.makedirs(output_dir, exist_ok=True)
fig_counter = 1  # start numbering figures


def save_fig(title=None, prefix="Figure_4"):
    """
    Saves the current matplotlib figure with running number.
    Example: Figure_4_1.png, Figure_4_2.png, ...
    """
    global fig_counter
    filename = f"{prefix}_{fig_counter}.png"
    filepath = os.path.join(output_dir, filename)

    if title:
        plt.title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {filepath}")

    fig_counter += 1
    plt.close()


# -------------------------------
# Workflow Illustration
# -------------------------------
plt.figure(figsize=(12, 7))
plt.axis('off')

steps = [
    "Raw Dataset Import\n(Excel: 'ESK-Solar PV Data.xlsx')",
    "Column Standardization\n(strip, lower, replace spaces)",
    "Filter Last 5 Years\n(2021-2025)",
    "Fill Missing Values\n(Median for numeric columns)",
    "Remove Outliers\n(Z-score threshold < 3)",
    "Descriptive Analysis\n(Stats, Boxplots, Histograms)",
    "Trend Analysis\n(Yearly total & growth, Monthly averages,Proportion of PV Production per Year (Last 5 years) )",
    "Forecasting 2026–2030\n(Prophet: daily to yearly totals)",
    "Machine Learning Models\n(LR & Random Forest: RMSE, R², Feature Importance)",
    "Seasonal Decomposition\n(Trend, Seasonal, Residual)",
    "Correlation Analysis\n(Heatmap of key variables)",
    "Interactive Dashboard\n(Plotly HTML with actual & forecasted PV)"
]

for i, step in enumerate(steps):
    plt.text(0.5, 1 - (i * 0.08), step, ha='center', va='center',
             fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="skyblue", edgecolor="black"))
    if i < len(steps) - 1:
        plt.arrow(0.5, 1 - (i * 0.08) - 0.03, 0, -0.03,
                  head_width=0.02, head_length=0.02, fc='black', ec='black')

plt.title("Data Analysis, Forecasting for Solar PV (2021-2030), and Findings",
          fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig(title="Workflow of Data Analysis & Forecasting for Solar PV (2021-2030)", prefix="Figure_4")
plt.show()

# -------------------------------
# Load Dataset
# -------------------------------
file_path = 'ESK-Solar PV Data.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# Standardize column names
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
print("Columns in dataset:", df.columns.tolist())
# -------------------------------
#Sample Dataset Structure Visualization
# -------------------------------
plt.figure(figsize=(12, 3))
plt.axis('off')
sample_table = df.head()
table = plt.table(cellText=sample_table.values,
                  colLabels=sample_table.columns,
                  cellLoc='center',
                  loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(sample_table.columns))))
plt.title("Dataset: Sample Dataset Structure", fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig(title="Sample Dataset Structure")
plt.show()

# -------------------------------
# Data Cleaning / Preparation
# -------------------------------
date_col = 'date_time_hour_beginning'
pv_col = 'pv'

# Convert 'date_time_hour_beginning' to datetime
df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)

# Drop rows with invalid dates or missing PV values
df = df.dropna(subset=[date_col, pv_col]).sort_values(by=date_col)

# Filter data for last 5 years
latest_date = df[date_col].max()
start_date = latest_date - pd.DateOffset(years=5)
df_5yrs = df[df[date_col] >= start_date].copy()

print(f"Data filtered for the last 5 years ({start_date.date()} to {latest_date.date()})")

# Fill missing numeric values with median
numeric_cols = df_5yrs.select_dtypes(include=np.number).columns.tolist()
df_5yrs[numeric_cols] = df_5yrs[numeric_cols].fillna(df_5yrs[numeric_cols].median())

# Remove outliers using Z-score (threshold < 3)
for col in numeric_cols:
    z_scores = np.abs(stats.zscore(df_5yrs[col]))
    df_5yrs = df_5yrs[z_scores < 3]

print(f"Dataset shape after cleaning: {df_5yrs.shape}")

# -------------------------------
# Visualization: Data Remaining After Cleaning
# -------------------------------
plt.figure(figsize=(6, 6))
total_points = df_5yrs.shape[0]
plt.bar(0, total_points, color='navy', alpha=0.7, width=0.4)
plt.xticks([0], ['Data Remaining'])
plt.ylabel('Number of Data Points')
plt.title('Total Data Points Remaining After Cleaning')

# Add text label on top of the bar
plt.text(0, total_points * 1.01, f"{total_points}", ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig(title="Total Data Points Remaining After Cleaning")
plt.show()


# Print counts before and after cleaning
print(f"Original dataset rows: {df.shape[0]}")
print(f"Rows after filtering last 5 years: {df[df[date_col] >= start_date].shape[0]}")
print(f"Rows after cleaning (dropping NAs and outliers): {df_5yrs.shape[0]}")

# -------------------------------
# Total PV Electricity Produced
# -------------------------------
total_pv_5yrs = df_5yrs[pv_col].sum()
print(f"\nTotal Solar PV electricity produced in the last 5 years: "
      f"{total_pv_5yrs:.2f} MWh ({total_pv_5yrs / 1e6:.2f} TWh)")

# -------------------------------
# Basic Visualization
# -------------------------------
# PV Production Over Time
plt.figure(figsize=(12, 6))
plt.plot(df_5yrs[date_col], df_5yrs[pv_col], color='orange')
plt.xlabel('Date')
plt.ylabel('PV Production (MWh)')
plt.title('Solar PV Production Over Last 5 Years')
plt.tight_layout()
save_fig(title="Solar PV Production Over Last 5 Years")
plt.show()

# Histogram of PV values
plt.figure(figsize=(8, 5))
sns.histplot(df_5yrs[pv_col], bins=50, kde=True, color='green')
plt.xlabel('PV Production (MWh)')
plt.ylabel('Frequency')
plt.title('Distribution of Solar PV Production')
plt.tight_layout()
save_fig(title="Distribution of Solar PV Production")
plt.show()

# Box plot of PV values
plt.figure(figsize=(6, 4))
sns.boxplot(x=df_5yrs[pv_col], color='purple')
plt.xlabel('PV Production (MWh)')
plt.title('Box Plot of Solar PV Production')
plt.tight_layout()
save_fig(title="Box Plot of Solar PV Production")
plt.show()

# -------------------------------
# Descriptive Statistics
# -------------------------------
desc_stats = df_5yrs[numeric_cols].describe()
print("\nDescriptive Statistics:\n", desc_stats)

# Boxplot of PV output
plt.figure(figsize=(10, 5))
sns.boxplot(y=df_5yrs[pv_col], color='orange')
plt.ylabel("PV Output (MWh)")
plt.title("Boxplot of PV Output (MWh)")
plt.tight_layout()
save_fig(title="Boxplot of PV Output (MWh)")
plt.show()

if 'solar_irradiance' in df_5yrs.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df_5yrs['solar_irradiance'], bins=30, kde=True, color='skyblue')
    plt.xlabel("Solar Irradiance (kWh/m²/day)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Solar Irradiance (kWh/m²/day)")
    plt.tight_layout()
    save_fig(title="Histogram of Solar Irradiance")
    plt.show()

# -------------------------------
# Trend Analysis (2021 – 2025)
# -------------------------------
df_5yrs['year'] = df_5yrs[date_col].dt.year
df_5yrs['month'] = df_5yrs[date_col].dt.month

yearly_total = df_5yrs.groupby('year')[pv_col].sum()
yearly_growth = yearly_total.pct_change() * 100

print("\nYearly PV Total and Growth (%):")
print(pd.concat([yearly_total, yearly_growth], axis=1, keys=['Total PV (MWh)', 'Growth (%)']))

# PV Trend Line
plt.figure(figsize=(12, 6))
plt.plot(df_5yrs[date_col], df_5yrs[pv_col], color='orange', linewidth=1)
plt.title("Solar PV Electricity Generation (2021-2025)")
plt.xlabel("Date")
plt.ylabel("PV Output (MWh)")
plt.grid(True)
plt.tight_layout()
save_fig(title="Solar PV Electricity Generation (2021-2025)")
plt.show()

# Average monthly PV production
monthly_avg = df_5yrs.groupby('month')[pv_col].mean()
monthly_avg.plot(kind='bar', color='gold', figsize=(10, 5))
plt.title("Average Monthly PV Production (2021-2025)")
plt.xlabel("Month")
plt.ylabel("PV Output (MWh)")
plt.tight_layout()
save_fig(title="Average Monthly PV Production (2021-2025)")
plt.show()

# -------------------------------
# Pie Chart: Proportion of PV Production per Year (Last 5 years)
# -------------------------------
plt.figure(figsize=(8, 8))
colors = sns.color_palette('pastel')[0:len(yearly_total)]
plt.pie(yearly_total, labels=yearly_total.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title("Proportion of Total Solar PV Production Per Year (Last 5 Years)")
save_fig(title="Proportion of Solar PV Production Per Year")
plt.show()

# -------------------------------
# Summary Table for Yearly Totals and Growth
# -------------------------------
summary_table = pd.DataFrame({
    'Year': yearly_total.index,
    'Total PV (MWh)': yearly_total.values,
    'Growth (%)': yearly_growth.values
})
summary_table.to_csv("Summary_Table.csv", index=False)

# Yearly PV Total & Growth Plot
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

ax1 = sns.lineplot(x='Year', y='Total PV (MWh)', data=summary_table,
                   marker='o', label='Total PV (MWh)', color='blue')

ax2 = ax1.twinx()
sns.lineplot(x='Year', y='Growth (%)', data=summary_table,
             marker='o', label='Growth (%)', color='red', ax=ax2)

ax1.set_ylabel("Total PV (MWh)", color='blue', fontsize=12)
ax2.set_ylabel("Growth (%)", color='red', fontsize=12)
ax1.set_xlabel("Year", fontsize=12)
ax1.set_title("Yearly Solar PV Total and Growth (2021-2025)", fontsize=14)
ax1.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='red')

# Legends combined
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
save_fig(title="")
plt.show()

# -------------------------------
# Forecasting (2026–2030) using Prophet
# -------------------------------
df_daily = df_5yrs.resample('D', on=date_col)[pv_col].sum().reset_index()
df_prophet = df_daily.rename(columns={date_col: 'ds', pv_col: 'y'})

prophet_model = Prophet(yearly_seasonality=True, daily_seasonality=False)
prophet_model.fit(df_prophet)

future = prophet_model.make_future_dataframe(periods=5 * 365, freq='D')
forecast = prophet_model.predict(future)

# Filter forecast for 2026-2030
forecast['ds'] = pd.to_datetime(forecast['ds'])
forecast['year'] = forecast['ds'].dt.year
forecast_26_30 = forecast[(forecast['year'] >= 2026) & (forecast['year'] <= 2030)]

def aggregate_annual_forecast(forecast_subset):
    annual_forecast = (
        forecast_subset.groupby('year')[['yhat', 'yhat_lower', 'yhat_upper']]
        .sum()
        .reset_index()
    )
    annual_forecast.rename(columns={
        'yhat': 'Predicted_PV_MWh',
        'yhat_lower': 'Lower_Bound_MWh',
        'yhat_upper': 'Upper_Bound_MWh'
    }, inplace=True)
    return annual_forecast

annual_forecast = aggregate_annual_forecast(forecast_26_30)

print("\nForecast Table 2026–2030:\n", annual_forecast)
annual_forecast.to_csv("PV_Forecast_2026_2030.csv", index=False)
print("\nSaved as PV_Forecast_2026_2030.csv")

# Bar plot with error bars for forecast
plt.figure(figsize=(10, 5))
plt.bar(annual_forecast['year'], annual_forecast['Predicted_PV_MWh'],
        color='green', alpha=0.7, label='Predicted PV')
plt.errorbar(annual_forecast['year'], annual_forecast['Predicted_PV_MWh'],
             yerr=[annual_forecast['Predicted_PV_MWh'] - annual_forecast['Lower_Bound_MWh'],
                   annual_forecast['Upper_Bound_MWh'] - annual_forecast['Predicted_PV_MWh']],
             fmt='o', color='black', label='Prediction Interval')
plt.xlabel("Year")
plt.ylabel("Total PV Output (MWh)")
plt.title("Next 5 Years Final Output: Predicted Solar PV Output 2026-2030")
plt.legend()
plt.tight_layout()
save_fig(title="Predicted Solar PV Output 2026-2030")
plt.show()

# Forecast vs Actual Plot
plt.figure(figsize=(12, 6))
plt.plot(df_prophet['ds'], df_prophet['y'], color='orange', label='Actual PV Output')
plt.plot(forecast['ds'], forecast['yhat'], color='green', label='Predicted PV Output')
plt.fill_between(forecast['ds'], forecast['yhat_lower'],
                 forecast['yhat_upper'], color='green', alpha=0.2)
plt.title("Predicted vs Actual PV Output (2021-2030)")
plt.xlabel("Date")
plt.ylabel("PV Output (MWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_fig(title="Predicted vs Actual PV Output (2021-2030)")
plt.show()

# -------------------------------
# Machine Learning Models (Linear Regression & Random Forest)
# -------------------------------
ml_cols = [col for col in ['installed_capacity', 'solar_irradiance', 'performance_ratio'] if col in df_5yrs.columns]
print("\nSelected ML features:", ml_cols)

if ml_cols:
    X = df_5yrs[ml_cols]
    y = df_5yrs[pv_col]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # Performance
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f"\nLinear Regression RMSE: {rmse_lr:.2f}, R²: {r2_lr:.2f}")
    print(f"Random Forest RMSE: {rmse_rf:.2f}, R²: {r2_rf:.2f}")

    # Plot Predicted vs Actual - Random Forest
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred_rf, alpha=0.5, label='Random Forest')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual PV Output (MWh)")
    plt.ylabel("Predicted PV Output (MWh)")
    plt.title("Random Forest: Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    save_fig(title="Random Forest: Predicted vs Actual")
    plt.show()

    # Feature Importance
    importances = rf_model.feature_importances_
    features = X.columns
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=features)
    plt.title("Factors Most Affecting PV Output: Random Forest Model")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    save_fig(title="Random Forest Feature Importance")
    plt.show()
else:
    # Placeholder figure if features are missing
    plt.figure(figsize=(6, 4))
    plt.text(0.5, 0.5, "No ML Features Available\n(Random Forest Skipped)",
             ha='center', va='center', fontsize=12,
             transform=plt.gca().transAxes,  # << ensures text uses axes coordinates
             bbox=dict(facecolor='lightgrey', alpha=0.7))
    plt.axis("off")
    plt.title("Random Forest Feature Importance (Unavailable)")
    save_fig(title="Random Forest Feature Importance (Unavailable)")
    plt.show()

# -------------------------------
# Seasonal Decomposition of Daily PV Output
# -------------------------------
# Ensure continuous daily data
df_daily = df_daily.set_index('date_time_hour_beginning').asfreq('D').fillna(0).reset_index()

try:
    decomp = seasonal_decompose(df_daily[pv_col], model='additive', period=365)
    decomp.plot()
    plt.tight_layout()
    save_fig(title="Seasonal Decomposition of Daily PV Output")
    plt.show()
except Exception as e:
    print("⚠️ Seasonal decomposition failed:", e)
    plt.figure(figsize=(6, 4))
    plt.text(0.5, 0.5, "Seasonal Decomposition Failed", 
             ha='center', va='center', fontsize=12, 
             bbox=dict(facecolor='lightgrey', alpha=0.7))
    plt.axis("off")
    plt.title("Seasonal Decomposition of Daily PV Output (Unavailable)")
    save_fig(title="Seasonal Decomposition of Daily PV Output (Unavailable)")
    plt.show()

# -------------------------------
# Correlation Analysis
# -------------------------------
if len(numeric_cols) > 1:
    corr = df_5yrs[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap of Key Variables")
    plt.tight_layout()
    save_fig(title="Correlation Heatmap of Key Variables")
    plt.show()
else:
    print("Not enough numeric columns for correlation analysis.")

# -------------------------------
# Interactive Dashboard (Plotly)
# -------------------------------
fig_dashboard = px.line(df_5yrs, x='date_time_hour_beginning', y='pv',
                        title="Solar PV Output Dashboard")
fig_dashboard.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted PV Output')
fig_dashboard.write_html("Dashboard.html")
print("\nChapter 4 Analysis Completed. All Figures and tables saved successfully.")
