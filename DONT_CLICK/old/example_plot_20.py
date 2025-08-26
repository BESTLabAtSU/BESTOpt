import pandas as pd
import matplotlib.pyplot as plt

combined_df = pd.read_csv("results/all_hvac_shift_summary.csv")
baseline = combined_df.groupby('timestamp')['baseline_hvac'].mean()
precool = combined_df.groupby('timestamp')['precool_hvac'].mean()

plt.figure(figsize=(6, 3), dpi=300)
plt.plot(baseline.index, baseline.values,
         label='Baseline (kW)', color='blue')
plt.plot(precool.index, precool.values,
         label='Precool (kW)', color='red')

# plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel("Time")
# plt.ylabel("Avg Load Shift (kW)")
plt.title("Average HVAC Load Shift Across 20 Datasets")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
# plt.savefig("./results/aggregate_hvac_shift_plot.png")
plt.show()