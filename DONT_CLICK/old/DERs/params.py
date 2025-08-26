from typing import Dict, Tuple


def get_params(
    # --- PV ---
    pv_capacity_kw=20,

    # --- Battery ---
    battery_capacity_kwh=100.0,
    battery_c_rate=0.25,
    battery_charge_eff=0.95,
    battery_discharge_eff=0.95,
    battery_min_soc=0.1,
    battery_max_soc=0.9,
    battery_initial_soc=0.5,

    # --- EV ---
    ev_capacity_kwh=50.0,
    ev_c_rate=0.25,
    ev_charge_eff=0.95,
    ev_discharge_eff=0.95,
    ev_min_soc=0.1,
    ev_max_soc=1.0,
    ev_departure_time=8 * 4,
    ev_arrival_time=18 * 4,
    ev_arrival_soc=0.6,
    ev_required_departure_soc=0.95,

    # --- TES ---
    tes_capacity_kwh=50.0,
    tes_c_rate=0.25,
    tes_charge_eff=0.85,
    tes_discharge_eff=0.85,
    tes_min_soc=0.0,
    tes_max_soc=1.0,
    tes_initial_soc=0.5,

    # --- Global ---
    T_amb=None,
    Sol=None,
    TOU=None,
    Load_thermal=None,
    Load_ele=None,
    Res=15,
    cpus=1
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """
    Returns parameter dictionaries for global, PV, battery, EV, and TES systems.
    """
    pv_params = {
        'capacity_kw': pv_capacity_kw
    } if pv_capacity_kw else None

    battery_params = {
        'capacity_kwh': battery_capacity_kwh,
        'c_rate': battery_c_rate,
        'charge_efficiency': battery_charge_eff,
        'discharge_efficiency': battery_discharge_eff,
        'min_soc': battery_min_soc,
        'max_soc': battery_max_soc,
        'initial_soc': battery_initial_soc
    } if battery_capacity_kwh else None

    ev_params = {
        'capacity_kwh': ev_capacity_kwh,
        'c_rate': ev_c_rate,
        'charge_efficiency': ev_charge_eff,
        'discharge_efficiency': ev_discharge_eff,
        'min_soc': ev_min_soc,
        'max_soc': ev_max_soc,
        'departure_time': ev_departure_time,
        'arrival_time': ev_arrival_time,
        'arrival_soc': ev_arrival_soc,
        'required_departure_soc': ev_required_departure_soc
    } if ev_capacity_kwh else None

    tes_params = {
        'capacity_kwh': tes_capacity_kwh,
        'c_rate': tes_c_rate,
        'charge_efficiency': tes_charge_eff,
        'discharge_efficiency': tes_discharge_eff,
        'min_soc': tes_min_soc,
        'max_soc': tes_max_soc,
        'initial_soc': tes_initial_soc
    } if tes_capacity_kwh else None

    global_params = {
        'T_amb': T_amb,
        'Sol': Sol,
        'TOU': TOU,
        'Load_thermal': Load_thermal,
        'Load_ele': Load_ele,
        'Res': Res,
        'cpus': cpus
    } if T_amb is not None and Sol is not None else None

    return global_params, pv_params, battery_params, ev_params, tes_params



def help_params():
    """
    Prints explanations for all configurable parameters in `get_params()`.
    """
    print("""
    PV Parameters:
      - pv_capacity_kw: PV system capacity in kW (default: 20)

    Battery Parameters:
      - battery_capacity_kwh: Battery capacity in kWh (default: 100)
      - battery_c_rate: Charge/discharge rate as C-rate (default: 0.25)
      - battery_charge_eff: Charging efficiency (default: 0.95)
      - battery_discharge_eff: Discharging efficiency (default: 0.95)
      - battery_min_soc: Minimum state of charge (default: 0.1)
      - battery_max_soc: Maximum state of charge (default: 0.9)
      - battery_initial_soc: Initial state of charge (default: 0.5)

    EV Parameters:
      - ev_capacity_kwh: EV battery capacity in kWh (default: 50)
      - ev_c_rate: EV charge/discharge rate as C-rate (default: 0.25)
      - ev_charge_eff: EV charging efficiency (default: 0.95)
      - ev_discharge_eff: EV discharging efficiency (default: 0.95)
      - ev_min_soc: Minimum state of charge (default: 0.1)
      - ev_max_soc: Maximum state of charge (default: 1.0)
      - ev_departure_time: EV departure timestep (default: 32 for 8:00 if 15-min interval)
      - ev_arrival_time: EV arrival timestep (default: 72 for 18:00)
      - ev_arrival_soc: SOC upon arrival (default: 0.6)
      - ev_required_departure_soc: Required SOC before departure (default: 0.95)

    TES Parameters:
      - tes_capacity_kwh: Thermal storage capacity in kWh (default: 200)
      - tes_c_rate: Charging/discharging rate (default: 0.25)
      - tes_charge_eff: Charging efficiency (default: 0.85)
      - tes_discharge_eff: Discharging efficiency (default: 0.85)
      - tes_min_soc: Minimum state of charge (default: 0.0)
      - tes_max_soc: Maximum state of charge (default: 1.0)
      - tes_initial_soc: Initial state of charge (default: 0.5)

    Global Parameters:
      - T_amb: Ambient temperature time series (°C)
      - Sol: Solar radiation time series (W/m²)
      - TOU: Time-of-Use electricity price time series (USD/kWh)
      - Load_thermal: Building thermal load time series (kW)
      - Load_ele: Building electric load time series (kW)
      - Res: Data resolution in minutes (default: 15)
      - cpus: Number of CPU cores for parallel runs (default: 1)
    """)
