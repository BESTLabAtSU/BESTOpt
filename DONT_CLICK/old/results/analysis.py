import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.dates as mdates
from glob import glob
# combined_df = pd.read_csv("all_hvac_shift_summary.csv", parse_dates=['timestamp'])

# Load all baseline and precool CSVs
baseline_files = sorted(glob("baseline_results_*.csv"))
precool_files = sorted(glob("precool_results_*.csv"))

baseline_list = []
precool_list = []

for bfile, pfile in zip(baseline_files, precool_files):
    baseline_df = pd.read_csv(bfile, parse_dates=['timestamp'])
    precool_df = pd.read_csv(pfile, parse_dates=['timestamp'])

    # Keep only timestamp + hvac_electric
    df = pd.DataFrame({
        'timestamp': baseline_df['timestamp'],
        'baseline_hvac': baseline_df['hvac_electric'],
        'precool_hvac': precool_df['hvac_electric'],
        'baseline_light': baseline_df['lighting_load'],
        'precool_light': precool_df['lighting_load'],
        'baseline_app': baseline_df['appliance_total'],
        'precool_app': precool_df['appliance_total'],
        'baseline_total': baseline_df['hvac_electric']+baseline_df['lighting_load']+baseline_df['appliance_total'],
        'precool_total': precool_df['hvac_electric']+precool_df['lighting_load']+precool_df['appliance_total'],

    })

    baseline_list.append(df)

# Combine all datasets
combined_df = pd.concat(baseline_list, ignore_index=True)

def plot_avg_hvac_shift(combined_df, figsize=(4, 1.5), dpi=300):
    """
    Plot average HVAC load shift (pre-cooling vs baseline) from combined_df.

    Args:
        combined_df: Aggregated DataFrame with 'timestamp' and 'hvac_shift'
        figsize: Tuple for figure size
        dpi: Plot DPI
        save_path: Optional path to save the plot as PNG
    """
    # Group by timestamp to get average shift
    grouped = combined_df.groupby('timestamp').mean()
    # Ensure datetime index
    grouped = grouped.copy()
    grouped.index = pd.to_datetime(grouped.index)

    # Filter only peak hours: 4 PM to 8 PM
    peak_data = grouped.between_time("16:00", "20:00")

    # Calculate energy (kWh), assuming 1-minute resolution (or adjust as needed)
    # Duration in hours between timestamps
    delta_t_hours = (grouped.index[1] - grouped.index[0]).seconds / 3600

    baseline_energy = peak_data['baseline_total'].sum() * delta_t_hours / 1000 * 20  # kWh
    precool_energy = peak_data['precool_total'].sum() * delta_t_hours / 1000 * 20   # kWh

    reduction = baseline_energy - precool_energy
    reduction_pct = reduction / baseline_energy * 100

    print(f"Peak-hour energy (Baseline): {baseline_energy:.2f} kWh")
    print(f"Peak-hour energy (Pre-cooling): {precool_energy:.2f} kWh")
    print(f"Energy reduction: {reduction:.2f} kWh ({reduction_pct:.1f}%)")


    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi, sharex=True)
    ax.plot(grouped.index, grouped['baseline_total'] / 1000 *20,
                linewidth=1.0, color='blue', label='Baseline')
    ax.plot(grouped.index, grouped['precool_total'] / 1000 *20,
                linewidth=1.0, color='red', label='Pre-Cooling')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_ylabel('Total Load (kW)')
    ax.grid(True, alpha=0.3)
    ax.legend([], [], frameon=False)

    # Format x-axis: every 4 hours
    start_time = grouped.index[0]
    end_time = grouped.index[-1]
    ax.set_xlim(start_time, end_time)

    locator = mdates.HourLocator(interval=4)
    formatter = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Add peak hour shading (4 PM to 8 PM)
    peak_start = pd.to_datetime(start_time.date()) + pd.Timedelta(hours=16)
    peak_end = peak_start + pd.Timedelta(hours=4)
    while peak_start < end_time:
        ax.axvspan(peak_start, peak_end, color='gray', alpha=0.15)
        peak_start += timedelta(days=1)
        peak_end += timedelta(days=1)

    ax.tick_params(labelbottom=False)

    plt.tight_layout()
    plt.show()


plot_avg_hvac_shift(combined_df=combined_df)

