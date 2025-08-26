import pandas as pd
from DONT_CLICK.old.DERs import Optimal_control
from DONT_CLICK.old.DERs import Config
from DONT_CLICK.old.DERs.globals import GlobalConfig

data = pd.read_csv('df_cooling_with_weather.csv')

global_config = GlobalConfig(
    T_amb=data['temp_amb'].values,    # Temperature array
    Sol=data['solar'].values,         # Solar irradiance array
    TOU=data['TOU'].values,           # Time-of-use pricing array
    Res=15,                           # 15-minute resolution
    cpus=16                           # Use 16 CPU cores
)

config = Config(global_config)
# Define PV parameters
pv_params = {
    'capacity_kw': 20  # PV Capacity: 20 kW
}

# Define battery parameters
battery_params = {
    'capacity_kwh': 100.0,         # Battery Capacity: 100 kWh
    'c_rate': 0.25,                # 0.25C charge/discharge rate (4 hours)
    'charge_efficiency': 0.95,     # 95% charging efficiency
    'discharge_efficiency': 0.95,  # 95% discharging efficiency
    'min_soc': 0.1,                # 10% minimum SOC
    'max_soc': 0.9,                # 90% maximum SOC
    'initial_soc': 0.5             # Start at 50% SOC
}

# Define ev parameters
ev_params = {
    'capacity_kwh': 50.0,         # EV Capacity: 50 kWh
    'c_rate': 0.25,               # 0.25C charge/discharge rate (4 hours)
    'charge_efficiency': 0.95,    # 95% charging efficiency
    'discharge_efficiency': 0.95, # 95% discharging efficiency
    'min_soc': 0.1,               # 10% minimum SOC
    'max_soc': 1.0,               # 90% maximum SOC
    'departure_time': 8*4,
    'arrival_time': 18*4,
    'arrival_soc': 0.6,
    'required_departure_soc': 0.95
}

# Create building configuration
building = config.create_building(
    load=data['load'].values,       # Assign Building load (kW)
    pv_params=pv_params,            # Assign PV parameters
    battery_params=battery_params,  # Assign Battery parameters
)
