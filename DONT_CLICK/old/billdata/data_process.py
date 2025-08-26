import os
import math
import pandas as pd
import numpy as np
import glob
import pickle

def round_to_5(x):
    return int(math.ceil(x / 5.0) * 5)

files = glob.glob('*.csv')
building_data = {}

# --- First pass: read and collect data for both heating and cooling ---
for f in files:
    data = pd.read_csv(f)
    building = f.split('_')[0]
    mode = f.split('_')[-1].split('.csv')[0]  # "heating" or "cooling"

    if mode == "cooling":
        data["cop"] = 5 - 0.06 * (data["temp_amb"] - 25)
    else:
        data["cop"] = 3 - 0.05 * (15 - data["temp_amb"])

    data["baseline(kW)"] = data["elec_load(kW)"] + (data["thermal_load_baseline(kW)"] / data["cop"]).abs()

    # Save the processed data and stats temporarily
    if building not in building_data:
        building_data[building] = {}
    building_data[building][mode] = {
        "data": data,
        "thermal_avg": np.abs(data["thermal_load_MPC(kW)"]).mean(),
        "elec_avg": data["baseline(kW)"].mean(),
        "elec_peak": data["baseline(kW)"].max()
    }

# --- Second pass: compute sizing per building using worst-case values ---
sizing = {}
for building, modes in building_data.items():
    thermal_avg = max(m["thermal_avg"] for m in modes.values())
    elec_avg = max(m["elec_avg"] for m in modes.values())
    elec_peak = max(m["elec_peak"] for m in modes.values())

    elec_avg_5 = round_to_5(elec_avg)
    battery_capacity_5 = round_to_5(elec_avg * 4)
    tes_capacity_5 = round_to_5(thermal_avg * 4)

    pv_capacity_kw = elec_avg_5
    battery_capacity_kwh = battery_capacity_5
    tes_capacity_kwh = tes_capacity_5

    # Save sizing for this building
    sizing[building] = {
        "pv_capacity_kw": pv_capacity_kw,
        "battery_capacity_kwh": battery_capacity_kwh,
        "tes_capacity_kwh": tes_capacity_kwh,
        "elec_avg_kW": elec_avg,
        "elec_peak_kW": elec_peak,
        "thermal_avg_kW": thermal_avg
    }

    # Optionally, update and save each CSV with COP and baseline
    for mode, info in modes.items():
        df = info["data"]
        df.to_csv(f"./cook/{building}_{mode}.csv")

# --- Save output ---
# 1. pickle
with open("sizing.pkl", "wb") as fp:
    pickle.dump(sizing, fp)

# 2. CSV
rows = []
for bldg, vals in sizing.items():
    row = {"building": bldg}
    row.update(vals)
    rows.append(row)
pd.DataFrame(rows).sort_values("building").to_csv("cook/sizing_summary.csv", index=False)

print("Done. Wrote sizing.pkl and sizing_summary.csv")
