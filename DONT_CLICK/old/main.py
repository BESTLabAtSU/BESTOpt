from DONT_CLICK.old.DERs.optimization import opt
from DONT_CLICK.old.DERs.bldg import BuildingConfig
import pandas as pd
import numpy as np
import glob


# ---- load sizing dict made previously ----
def load_sizing_dict(path_csv="/home/zjiang19/Documents/GitHub/BEST_OPT/old/billdata/cook/sizing_summary.csv"):
    df = pd.read_csv(path_csv)
    return {
        r["building"]: {
            "pv_capacity_kw": r["pv_capacity_kw"],
            "battery_capacity_kwh": r["battery_capacity_kwh"],
            "tes_capacity_kwh": r["tes_capacity_kwh"],
            "elec_avg_kW": r.get("elec_avg_kW", np.nan),
            "elec_peak_kW": r.get("elec_peak_kW", np.nan),
            "thermal_avg_kW": r.get("thermal_avg_kW", np.nan),
        }
        for _, r in df.iterrows()
    }

sizing = load_sizing_dict()

files = glob.glob('./billdata/cook/*.csv')

for f in files[:3]:
    if 'sizing' in f:
        continue
    buildingname = f.split('/')[-1].split('.csv')[0]
    data = pd.read_csv(f)
    roomname = buildingname.split('_')[0]
    pv_kw = battery_kwh = tes_kwh = None
    if roomname in sizing:
        pv_kw = sizing[roomname]["pv_capacity_kw"]
        battery_kwh = sizing[roomname]["battery_capacity_kwh"]
        tes_kwh = sizing[roomname]["tes_capacity_kwh"]
    else:
        print("missing sizing info")

    # Override for 'coe'
    if 'coe' in roomname:
        data = data.fillna(0)
        pv_kw = 20
        battery_kwh = 100
        tes_kwh = 300

    sizing_info = f"_pv_kw{pv_kw}_battery_kwh{battery_kwh}_tes_kwh{tes_kwh}"

    building = BuildingConfig()
    building.get_params(
        pv_capacity_kw=pv_kw,
        battery_capacity_kwh=battery_kwh,
        tes_capacity_kwh=tes_kwh,
        ev_capacity_kwh=0,
        T_amb=data['temp_amb'].values,
        Sol=data['q_sol'].values,
        TOU=data['TOU($/kWh)'].values,
        cop=data["cop"].values,
        Load_thermal=np.abs(data['thermal_load_MPC(kW)'].values),
        Load_ele=data['elec_load(kW)'].values
    )

    building.create_building()
    controller = opt(building)
    controller.run(buildingname=buildingname + sizing_info)
    controller.display(startday=1, endday=3)
    controller.evaluation(baseline=data['baseline(kW)'].values, withmpc = data["elec_load(kW)"] + (data["thermal_load_MPC(kW)"] / data["cop"]).abs())
