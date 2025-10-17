import pandas as pd
import matplotlib.pyplot as plt
import os

# Optional: Check if file exists
file_path = 'ESK-Solar PV Data.xlsx'
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    data = pd.read_excel(file_path)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data['Irradiance'], bins=20, color='gold', edgecolor='black')
    plt.title('Figure 4.8: Histogram of Solar Irradiance (kWh/m²/day)', fontsize=14)
    plt.xlabel('Solar Irradiance (kWh/m²/day)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
