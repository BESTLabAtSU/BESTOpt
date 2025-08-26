def params():
    # Define pv parameters
    pv_params = {
        'capacity_kw': 20  # PV capacity: 20 kW
    }

    # Define battery parameters
    battery_params = {
        'capacity_kwh': 100.0,         # Battery capacity: 100 kWh
        'c_rate': 0.25,                # 0.25C charge/discharge rate (4 hours)
        'charge_efficiency': 0.95,     # 95% charging efficiency
        'discharge_efficiency': 0.95,  # 95% discharging efficiency
        'min_soc': 0.1,                # 10% minimum SOC
        'max_soc': 0.9,                # 90% maximum SOC
        'initial_soc': 0.5             # Start at 50% SOC
    }

    # Define ev parameters
    ev_params = {
        'capacity_kwh': 50.0,         # EV capacity: 50 kWh
        'c_rate': 0.25,               # 0.25C charge/discharge rate (4 hours)
        'charge_efficiency': 0.95,    # 95% charging efficiency
        'discharge_efficiency': 0.95, # 95% discharging efficiency
        'min_soc': 0.1,               # 10% minimum SOC
        'max_soc': 1.0,               # 90% maximum SOC
        'departure_time': 8*4,        # Timestep
        'arrival_time': 18*4,         # Timestep
        'arrival_soc': 0.6,
        'required_departure_soc': 0.95
    }

    return pv_params, battery_params, ev_params