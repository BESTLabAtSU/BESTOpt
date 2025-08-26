from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

@dataclass
class PCMConfig:
    """PCM Storage Configuration"""
    capacity_kwh: float  # PCM capacity in kWh
    c_rate: float  # charging/discharging speed
    charge_efficiency: float = 0.85  # Charging efficiency
    discharge_efficiency: float = 0.85  # Discharging efficiency
    min_soc: float = 0.0  # Minimum state of charge (10%)
    max_soc: float = 1.0  # Maximum state of charge (90%)
    initial_soc: float = 0.5  # Initial state of charge

@dataclass
class GlobalConfig:
    """Simulation environment configuration"""
    T_amb: np.ndarray  # Outdoor air temperature (C)
    TOU: np.ndarray # TOU price (USD/Watt)
    Res: int  # Data resolution in minutes
    cpus: int # Available cpus for this task

@dataclass
class BuildingConfig:
    """Building system configuration"""
    thermalload: np.ndarray
    tes: Optional[PCMConfig] = None

class Config:
    """Wrapper to config the overall system"""
    def __init__(self, global_config: GlobalConfig):
        self.global_config = global_config

    def calculate_chiller_cop(self, load, chiller_capacity):
        """
        Pre-calculate chiller coefficient based on ambient temp and PLR.
        """
        Tamb = self.global_config.T_amb
        load_ratio = load / chiller_capacity
        a = 5.5
        b = -0.07
        c = -1.5
        d = 0.03
        # COP equation
        cop = a + b * Tamb + c * load_ratio + d * Tamb * load_ratio
        return cop

    def create_building(self,
            thermalload: np.ndarray,
            pcm_params: Optional[dict] = None,
    ) -> BuildingConfig:
        # Battery
        tes = (
            PCMConfig(**pcm_params)
            if pcm_params is not None
            else None
        )
        return BuildingConfig(thermalload=thermalload, tes=tes)

    def create_multiple_buildings(self, config_list: list[dict]) -> list[BuildingConfig]:
        """Takes a list of parameter dicts, returns BuildingConfig list"""
        return [self.create_building(**params) for params in config_list]


class Optimal_control:
    def __init__(self, building_config: BuildingConfig, global_config: GlobalConfig):
        self.building = building_config
        self.globals = global_config
        self.thermalload = self.building.thermalload
        self.tou = self.globals.TOU
        self.tes_size = self.building.tes.capacity_kwh if self.building.tes else 0
        self.operation_result = None
        self.chiller_size = 50 # for now

    def run(self):
        return self._run_operation()

    def _run_operation(self):
        N = len(self.thermalload)
        operation_results = []
        SoC_tes_0 = self.building.tes.initial_soc if self.building.tes else 0
        one_day_step_size = int(1440 / self.globals.Res)
        tes = self.building.tes
        tes_max_chg = tes.c_rate * self.tes_size
        tes_max_dch = tes.c_rate * self.tes_size
        for i in range(0, N, one_day_step_size):  # Each optimization is based on one day horizon
            end_idx = min(i + one_day_step_size, N)
            window_steps = end_idx - i
            A = np.tril(np.ones((window_steps, window_steps)))  # State matrix

            m = gp.Model("Operation")
            m.setParam('OutputFlag', 0)
            m.setParam('Threads', self.globals.cpus)

            chiller_tes = m.addMVar((window_steps, 1), lb=0, name="Chiller2Tes")
            chiller_ahu = m.addMVar((window_steps, 1), lb=0, name="Chiller2Ahu")
            tes_ahu = m.addMVar((window_steps, 1), lb=0, name="Tes2Ahu")
            total_thermalload = self.thermalload[i:end_idx].reshape(-1, 1)
            SoC_tes = m.addMVar((window_steps, 1), lb=0, name="SoC_Tes") if self.tes_size > 0 else None
            # air load balance constraint
            m.addConstr(chiller_ahu + tes_ahu == total_thermalload, name="Air_Load_Balance")
            # tes dynamic
            m.addConstr(
                SoC_tes_0 * np.ones((window_steps, 1)) +
                A @ (chiller_tes) * (self.globals.Res/60)*tes.charge_efficiency / self.tes_size -
                A @ (tes_ahu) * (self.globals.Res/60) / (self.tes_size * tes.discharge_efficiency) == SoC_tes,
                name="TES_Balance"
            )
            m.addConstr(SoC_tes >= tes.min_soc, name="Tes_SoCmin")
            m.addConstr(SoC_tes <= tes.max_soc, name="Tes_SoCmax")
            m.addConstr(chiller_tes <= tes_max_chg, name="Tes_ChargeLimit")
            m.addConstr(tes_ahu <= tes_max_dch, name="Tes_DisChargeLimit")
            tou = self.tou[i:end_idx].reshape(-1, 1)

            Tamb = self.globals.T_amb[i:end_idx].reshape(-1, 1)
            cop = 7-0.2*Tamb+0.001*Tamb*Tamb
            cop = cop.reshape(window_steps, 1)
            ele = (chiller_tes + chiller_ahu)/cop
            m.setObjective(tou.T @ ele, GRB.MINIMIZE)
            m.setParam("NonConvex", 2)
            m.optimize()
            print(m.Status)

            # Collect results
            result_data = {
                'Chiller2Tes': chiller_tes.X.reshape(-1, ),
                'Chiller2Ahu': chiller_tes.X.reshape(-1, ),
                'Tes2Ahu': tes_ahu.X.reshape(-1, ),
                'SoC_Tes': SoC_tes.X.reshape(-1, ),
                'Opt_ele': ((chiller_tes.X + chiller_ahu.X)/cop).reshape(-1, ),
                'TOU': tou.reshape(-1, ),
                'baseLoad': (total_thermalload/cop).reshape(-1, ),
            }

            result = pd.DataFrame(result_data, index=np.arange(i, end_idx))
            operation_results.append(result)

        self.operation_result = pd.concat(operation_results)
        return self.operation_result


class Rule_control:
    def __init__(self, building_config, global_config):
        self.building = building_config
        self.globals = global_config
        self.load = self.building.load
        self.tou = self.globals.TOU
        self.pv_size = self.building.pv.capacity_kw if self.building.pv else 0
        self.battery_size = self.building.battery.capacity_kwh if self.building.battery else 0
        self.ev_size = self.building.ev.capacity_kwh if self.building.ev else 0
        self.operation_result = None

        # Identify TOU price levels
        self._identify_tou_levels()

    def _identify_tou_levels(self):
        """Identify high, mid, and low TOU price periods"""
        # Sort unique prices to identify thresholds
        unique_prices = np.unique(self.tou)
        if len(unique_prices) >= 3:
            # Use 33rd and 66th percentiles to define thresholds
            low_threshold = np.percentile(unique_prices, 33)
            high_threshold = np.percentile(unique_prices, 66)
        elif len(unique_prices) == 2:
            # If only two price levels, use median
            threshold = np.median(unique_prices)
            low_threshold = threshold
            high_threshold = threshold
        else:
            # If only one price level, treat all as mid
            low_threshold = unique_prices[0]
            high_threshold = unique_prices[0]

        # Classify each time step
        self.tou_level = np.where(self.tou <= low_threshold, 'low',
                                  np.where(self.tou >= high_threshold, 'high', 'mid'))

    def _get_ev_availability(self, t):
        """Check if EV is available at time step t"""
        if not self.building.ev:
            return False

        ev = self.building.ev
        # Convert to daily time step (assuming daily pattern)
        daily_t = t % int(1440 / self.globals.Res)

        # EV is available between arrival and departure
        if ev.arrival_time <= ev.departure_time:
            return ev.arrival_time <= daily_t < ev.departure_time
        else:  # EV stays overnight
            return daily_t >= ev.arrival_time or daily_t < ev.departure_time

    def _calculate_ev_driving_consumption(self, t):
        """Calculate EV SOC decrease during driving (when not at home)"""
        if not self.building.ev or self._get_ev_availability(t):
            return 0.0

        ev = self.building.ev
        daily_t = t % int(1440 / self.globals.Res)  # Current time in daily cycle

        # Define driving periods: 1 hour after departure and 1 hour before arrival
        departure_driving_end = (ev.departure_time + int(60 / self.globals.Res)) % int(1440 / self.globals.Res)
        arrival_driving_start = (ev.arrival_time - int(60 / self.globals.Res)) % int(1440 / self.globals.Res)

        # Check departure driving period (1 hour after departure)
        if ev.departure_time <= departure_driving_end:
            is_driving = ev.departure_time <= daily_t < departure_driving_end
        else:  # Handles midnight crossing
            is_driving = daily_t >= ev.departure_time or daily_t < departure_driving_end

        # Check arrival driving period (1 hour before arrival)
        if not is_driving:
            if arrival_driving_start <= ev.arrival_time:
                is_driving = arrival_driving_start <= daily_t < ev.arrival_time
            else:  # Handles midnight crossing
                is_driving = daily_t >= arrival_driving_start or daily_t < ev.arrival_time

        if is_driving:
            # 5% SOC decrease per hour during driving periods
            driving_consumption_rate = 0.05
            time_step_hours = self.globals.Res / 60
            return driving_consumption_rate * time_step_hours
        else:
            return 0.0

    def run(self):
        """Run the rule-based control simulation"""
        N = len(self.load)
        P_pv = self.building.pv.pv_gen if self.building.pv else np.zeros_like(self.load)

        # Initialize results arrays
        results = {
            'PV2Blgd': np.zeros(N),
            'G2Blgd': np.zeros(N),
            'Bat2Blgd': np.zeros(N),
            'EV2Blgd': np.zeros(N),
            'GridPurchase': np.zeros(N),
            'PVcurt': np.zeros(N),
            'PV2Bat': np.zeros(N),
            'PV2EV': np.zeros(N),
            'G2Bat': np.zeros(N),
            'G2EV': np.zeros(N),
            'Bat2EV': np.zeros(N),
            'EV2Bat': np.zeros(N),
            'SoC_Bat': np.zeros(N),
            'SoC_EV': np.zeros(N),
            'TOU': self.tou,
            'Load': self.load,
        }

        # Initialize SOC
        soc_bat = self.building.battery.initial_soc if self.building.battery else 0
        soc_ev = self.building.ev.arrival_soc if self.building.ev else 0

        for t in range(N):
            current_load = self.load[t]
            current_pv = P_pv[t]
            tou_level = self.tou_level[t]
            ev_available = self._get_ev_availability(t)

            # Initialize energy flows
            pv_to_bldg = 0
            pv_to_bat = 0
            pv_to_ev = 0
            pv_curtail = 0
            grid_to_bldg = 0
            grid_to_bat = 0
            grid_to_ev = 0
            bat_to_bldg = 0
            bat_to_ev = 0
            ev_to_bldg = 0
            ev_to_bat = 0

            # Step 1: Update EV SOC for driving consumption
            if not ev_available and self.building.ev:
                driving_consumption = self._calculate_ev_driving_consumption(t)
                soc_ev = max(self.building.ev.min_soc, soc_ev - driving_consumption)

            # Step 2: PV generation priority - Building first
            remaining_pv = current_pv
            if remaining_pv > 0:
                pv_to_bldg = min(remaining_pv, current_load)
                remaining_pv -= pv_to_bldg

            remaining_load = current_load - pv_to_bldg

            # Step 3: Handle remaining PV based on TOU and system states
            if remaining_pv > 0:
                # Battery charging logic
                if self.building.battery and soc_bat < self.building.battery.max_soc:
                    bat_max_charge = self.building.battery.c_rate * self.battery_size

                    if tou_level == 'low':
                        # Off-peak: charge battery aggressively
                        available_bat_capacity = (self.building.battery.max_soc - soc_bat) * self.battery_size
                        max_bat_charge_step = min(bat_max_charge * (self.globals.Res / 60), available_bat_capacity)
                        pv_to_bat = min(remaining_pv, max_bat_charge_step / self.building.battery.charge_efficiency)
                        remaining_pv -= pv_to_bat
                    elif remaining_pv > 0:
                        # Mid/high period: use excess PV if available
                        available_bat_capacity = (self.building.battery.max_soc - soc_bat) * self.battery_size
                        max_bat_charge_step = min(bat_max_charge * (self.globals.Res / 60), available_bat_capacity)
                        pv_to_bat = min(remaining_pv, max_bat_charge_step / self.building.battery.charge_efficiency)
                        remaining_pv -= pv_to_bat

                # EV charging logic
                if self.building.ev and ev_available and soc_ev < self.building.ev.max_soc:
                    ev_max_charge = self.building.ev.c_rate * self.ev_size
                    available_ev_capacity = (self.building.ev.max_soc - soc_ev) * self.ev_size
                    max_ev_charge_step = min(ev_max_charge * (self.globals.Res / 60), available_ev_capacity)

                    if remaining_pv > 0:
                        pv_to_ev = min(remaining_pv, max_ev_charge_step / self.building.ev.charge_efficiency)
                        remaining_pv -= pv_to_ev

                # Curtail remaining PV
                pv_curtail = remaining_pv

            # Step 4: Handle remaining building load and charging needs
            if remaining_load > 0 or (self.building.battery and tou_level == 'low') or (
                    self.building.ev and ev_available):

                # Battery discharge during peak hours for building load
                if remaining_load > 0 and self.building.battery and tou_level == 'high' and soc_bat > self.building.battery.min_soc:
                    bat_max_discharge = self.building.battery.c_rate * self.battery_size
                    available_bat_energy = (soc_bat - self.building.battery.min_soc) * self.battery_size
                    max_bat_discharge_step = min(bat_max_discharge * (self.globals.Res / 60), available_bat_energy)
                    bat_to_bldg = min(remaining_load,
                                      max_bat_discharge_step * self.building.battery.discharge_efficiency)
                    remaining_load -= bat_to_bldg

                # Grid charging for battery during off-peak
                if self.building.battery and tou_level == 'low' and soc_bat < self.building.battery.max_soc:
                    bat_max_charge = self.building.battery.c_rate * self.battery_size
                    available_bat_capacity = (self.building.battery.max_soc - soc_bat) * self.battery_size
                    max_bat_charge_step = min(bat_max_charge * (self.globals.Res / 60), available_bat_capacity)
                    # Only charge if we haven't already charged from PV up to the limit
                    remaining_charge_capacity = max_bat_charge_step - (
                                pv_to_bat * self.building.battery.charge_efficiency)
                    if remaining_charge_capacity > 0:
                        grid_to_bat = remaining_charge_capacity / self.building.battery.charge_efficiency

                # EV charging from grid when available
                if self.building.ev and ev_available and soc_ev < self.building.ev.required_departure_soc:
                    ev_max_charge = self.building.ev.c_rate * self.ev_size
                    available_ev_capacity = (self.building.ev.required_departure_soc - soc_ev) * self.ev_size
                    max_ev_charge_step = min(ev_max_charge * (self.globals.Res / 60), available_ev_capacity)
                    # Only charge if we haven't already charged from PV up to the limit
                    remaining_charge_capacity = max_ev_charge_step - (pv_to_ev * self.building.ev.charge_efficiency)
                    if remaining_charge_capacity > 0:
                        grid_to_ev = remaining_charge_capacity / self.building.ev.charge_efficiency

                # Remaining building load from grid
                grid_to_bldg = remaining_load

            # Calculate total grid purchase
            grid_purchase = grid_to_bldg + grid_to_bat + grid_to_ev

            # Update SOC states
            if self.building.battery:
                energy_in = (grid_to_bat + pv_to_bat + ev_to_bat) * self.building.battery.charge_efficiency * (
                            self.globals.Res / 60)
                energy_out = (bat_to_bldg + bat_to_ev) / self.building.battery.discharge_efficiency * (
                            self.globals.Res / 60)
                soc_bat += (energy_in - energy_out) / self.battery_size
                soc_bat = np.clip(soc_bat, self.building.battery.min_soc, self.building.battery.max_soc)

            if self.building.ev and ev_available:
                energy_in = (grid_to_ev + pv_to_ev + bat_to_ev) * self.building.ev.charge_efficiency * (
                            self.globals.Res / 60)
                energy_out = (ev_to_bldg + ev_to_bat) / self.building.ev.discharge_efficiency * (self.globals.Res / 60)
                soc_ev += (energy_in - energy_out) / self.ev_size
                soc_ev = np.clip(soc_ev, self.building.ev.min_soc, self.building.ev.max_soc)

            # Store results
            results['PV2Blgd'][t] = pv_to_bldg
            results['G2Blgd'][t] = grid_to_bldg
            results['Bat2Blgd'][t] = bat_to_bldg
            results['EV2Blgd'][t] = ev_to_bldg
            results['GridPurchase'][t] = grid_purchase
            results['PVcurt'][t] = pv_curtail
            results['PV2Bat'][t] = pv_to_bat
            results['PV2EV'][t] = pv_to_ev
            results['G2Bat'][t] = grid_to_bat
            results['G2EV'][t] = grid_to_ev
            results['Bat2EV'][t] = bat_to_ev
            results['EV2Bat'][t] = ev_to_bat
            results['SoC_Bat'][t] = soc_bat
            results['SoC_EV'][t] = soc_ev

        # Create DataFrame with results
        self.operation_result = pd.DataFrame(results, index=np.arange(N))
        return self.operation_result