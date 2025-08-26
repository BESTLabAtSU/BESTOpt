from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

@dataclass
class BatteryConfig:
    """Battery system configuration"""
    capacity_kwh: float  # Battery capacity in kWh
    c_rate: float  # charging/discharging speed
    charge_efficiency: float = 0.95  # Charging efficiency
    discharge_efficiency: float = 0.95  # Discharging efficiency
    min_soc: float = 0.1  # Minimum state of charge (10%)
    max_soc: float = 0.9  # Maximum state of charge (90%)
    initial_soc: float = 0.5  # Initial state of charge

@dataclass
class EVConfig:
    """Electric Vehicle configuration"""
    capacity_kwh: float  # EV battery capacity in kWh
    c_rate: float  # charging/discharging speed
    arrival_time: int  # Arrival time step
    departure_time: int  # Departure time step
    arrival_soc: float  # SOC when arriving
    required_departure_soc: float  # Required SOC when departing
    charge_efficiency: float = 0.95  # Charging efficiency
    discharge_efficiency: float = 0.95  # Discharging efficiency
    min_soc: float = 0.1  # Minimum state of charge
    max_soc: float = 1.0  # Maximum state of charge

@dataclass
class PVConfig:
    """PV system configuration"""
    capacity_kw: float  # PV capacity in kW
    pv_gen: Optional[np.ndarray] = None # PV generation, will be setup later

@dataclass
class GlobalConfig:
    """Simulation environment configuration"""
    T_amb: np.ndarray  # Outdoor air temperature (C)
    Sol: np.ndarray # Solar radiation (Watt)
    TOU: np.ndarray # TOU price (USD/Watt)
    Res: int  # Data resolution in minutes
    cpus: int # Available cpus for this task 

@dataclass
class BuildingConfig:
    """Building system configuration"""
    load: np.ndarray
    pv: Optional[PVConfig] = None
    battery: Optional[BatteryConfig] = None
    ev: Optional[EVConfig] = None

    def __post_init__(self):
        if self.load is None or not isinstance(self.load, np.ndarray):
            raise ValueError("`load` must be a non-None numpy array")

class Config:
    """Wrapper to config the overall system"""
    def __init__(self, global_config: GlobalConfig):
        self.global_config = global_config
        self.pv_coefficient = self.calculate_pv_coefficient()  # Pre-calculate coefficient

    def calculate_pv_coefficient(self):
        """
        Pre-calculate PV coefficient based on ambient temp and solar radiation.
        """
        Isol = self.global_config.Sol
        Tamb = self.global_config.T_amb
        # Parameters
        K = -3.7 / 1000
        Sol_ref = 1000
        Temp_ref = 25

        self.pv_coefficient = (Isol / Sol_ref) * (1 + K * (Tamb + (0.0256 * Isol) - Temp_ref))

    def get_pv_generation(self, pv_size: float) -> np.ndarray:
        """
        Calculate PV generation using pre-calculate PV coefficient.
        """
        if self.pv_coefficient is None:
            raise ValueError("PV coefficient not calculated correctly")
        return pv_size * self.pv_coefficient

    def create_building(
            self,
            load: np.ndarray,
            pv_params: Optional[dict] = None,
            battery_params: Optional[dict] = None,
            ev_params: Optional[dict] = None,
    ) -> BuildingConfig:

        # PV
        pv = None
        if pv_params is not None:
            if self.pv_coefficient is None:
                self.calculate_pv_coefficient()
            pv_size = pv_params['capacity_kw']
            pv_gen = self.get_pv_generation(pv_size)
            pv_params = {**pv_params, 'pv_gen': pv_gen}
            pv = PVConfig(**pv_params)

        # Battery
        battery = (
            BatteryConfig(**battery_params)
            if battery_params is not None
            else None
        )

        # EV
        ev = (
            EVConfig(**ev_params)
            if ev_params is not None
            else None
        )

        return BuildingConfig(load=load, pv=pv, battery=battery, ev=ev)

    def create_multiple_buildings(self, config_list: list[dict]) -> list[BuildingConfig]:
        """Takes a list of parameter dicts, returns BuildingConfig list"""
        return [self.create_building(**params) for params in config_list]

class Optimal_control:
    def __init__(self, building_config: BuildingConfig, global_config: GlobalConfig):
        self.building = building_config
        self.globals = global_config
        self.load = self.building.load
        self.tou = self.globals.TOU
        self.pv_size = self.building.pv.capacity_kw if self.building.pv else 0
        self.battery_size = self.building.battery.capacity_kwh if self.building.battery else 0
        self.ev_size = self.building.ev.capacity_kwh if self.building.ev else 0
        self.operation_result = None

    def run(self):
        return self._run_operation()

    def _run_operation(self):
        N = len(self.load)
        P_pv = self.building.pv.pv_gen if self.building.pv else np.zeros_like(self.load)
        operation_results = []
        SoC_bat_0 = self.building.battery.initial_soc if self.building.battery else 0
        SoC_ev_0 = self.building.ev.arrival_soc if self.building.ev else 0
        one_day_step_size = int(1440 / self.globals.Res)

        for i in range(0, N, one_day_step_size):  # Each optimization is based on one day horizon
            end_idx = min(i + one_day_step_size, N)
            window_steps = end_idx - i
            A = np.tril(np.ones((window_steps, window_steps)))  # State matrix

            m = gp.Model("Operation")
            m.setParam('OutputFlag', 0)
            m.setParam('Threads', self.globals.cpus)

            # Load to Building
            PV_blgd = m.addMVar((window_steps, 1), lb=0, name="PV2Blgd")
            Grid_blgd = m.addMVar((window_steps, 1), lb=0, name="G2Blgd")
            Bat_blgd = m.addMVar((window_steps, 1), lb=0, name="Bat2Blgd")
            EV_blgd = m.addMVar((window_steps, 1), lb=0, name="EV2Blgd")

            # Load from PV
            PV_curtail = m.addMVar((window_steps, 1), lb=0, name="PVcurt")
            PV_bat = m.addMVar((window_steps, 1), lb=0, name="PV2Bat")
            PV_ev = m.addMVar((window_steps, 1), lb=0, name="PV2EV")

            # Load from Grid
            Grid_bat = m.addMVar((window_steps, 1), lb=0, name="G2Bat")
            Grid_ev = m.addMVar((window_steps, 1), lb=0, name="G2EV")
            Grid_purchase = m.addMVar((window_steps, 1), lb=0, name="GridPurchase")

            # Load between Battery and EV
            Bat_ev = m.addMVar((window_steps, 1), lb=0, name="Bat2EV")
            Ev_Bat = m.addMVar((window_steps, 1), lb=0, name="EV2Bat")

            # SOC of Battery and EV
            SoC_bat = m.addMVar((window_steps, 1), lb=0, name="SoC_Bat") if self.battery_size > 0 else None
            SoC_ev = m.addMVar((window_steps, 1), lb=0, name="SoC_EV") if self.ev_size > 0 else None

            total_load = self.load[i:end_idx].reshape(-1, 1)

            # Building load balance constraint
            m.addConstr(PV_blgd + Grid_blgd + Bat_blgd + EV_blgd == total_load, name="Bldg_Load_Balance")

            # PV generation balance constraint
            if self.pv_size > 0:
                m.addConstr(PV_blgd + PV_curtail + PV_bat + PV_ev == P_pv[i:end_idx].reshape(-1, 1),
                            name="PV_Gen_Balance")
            else:
                # If no PV, all PV-related variables should be zero
                m.addConstr(PV_blgd == 0, name="No_PV_to_Bldg")
                m.addConstr(PV_curtail == 0, name="No_PV_Curtail")
                m.addConstr(PV_bat == 0, name="No_PV_to_Bat")
                m.addConstr(PV_ev == 0, name="No_PV_to_EV")

            # Grid purchase balance constraint
            m.addConstr(Grid_blgd + Grid_bat + Grid_ev == Grid_purchase, name="Grid_Purchase_Balance")

            # Battery constraints
            if self.battery_size > 0:
                bat = self.building.battery
                bat_max_chg = bat.c_rate * self.battery_size
                bat_max_dch = bat.c_rate * self.battery_size
                m.addConstr(
                    SoC_bat_0 * np.ones((window_steps, 1)) +
                    A @ (Grid_bat + PV_bat + Ev_Bat) * (self.globals.Res/60) *bat.charge_efficiency / self.battery_size -
                    A @ (Bat_blgd + Bat_ev) * (self.globals.Res/60) / (self.battery_size * bat.discharge_efficiency) == SoC_bat,
                    name="Battery_Balance"
                )
                m.addConstr(SoC_bat >= bat.min_soc, name="Bat_SoCmin")
                m.addConstr(SoC_bat <= bat.max_soc, name="Bat_SoCmax")
                m.addConstr(Grid_bat + PV_bat + Ev_Bat <= bat_max_chg, name="Bat_ChargeLimit")
                m.addConstr(Bat_blgd + Bat_ev <= bat_max_dch, name="Bat_DisChargeLimit")
                # m.addConstr(SoC_bat[95, 0] >= 0.5)
            else:
                # If no battery, all battery-related variables should be zero
                m.addConstr(Bat_blgd == 0, name="No_Bat_to_Bldg")
                m.addConstr(Grid_bat == 0, name="No_Grid_to_Bat")
                m.addConstr(PV_bat == 0, name="No_PV_to_Bat_constraint")
                m.addConstr(Bat_ev == 0, name="No_Bat_to_EV")
                m.addConstr(Ev_Bat == 0, name="No_EV_to_Bat")

            # EV constraints
            if self.ev_size > 0:
                ev = self.building.ev
                ev_max_chg = ev.c_rate * self.ev_size
                ev_max_dch = ev.c_rate * self.ev_size

                # Consider driving consumption
                driving_consumption = np.zeros((window_steps, 1))
                for t in range(window_steps):
                    global_t = i + t  # Global time index
                    daily_t = global_t % int(1440 / self.globals.Res)

                    # 1 hour after departure and 1 hour before arrival
                    departure_driving_end = (ev.departure_time + int(60 / self.globals.Res)) % int(
                        1440 / self.globals.Res)
                    arrival_driving_start = (ev.arrival_time - int(60 / self.globals.Res)) % int(
                        1440 / self.globals.Res)

                    # Check if in driving periods (Assume 1 hour before and after arrive and departure)
                    if ev.departure_time <= departure_driving_end:
                        is_driving = ev.departure_time <= daily_t < departure_driving_end
                    else:
                        is_driving = daily_t >= ev.departure_time or daily_t < departure_driving_end

                    if not is_driving:
                        if arrival_driving_start <= ev.arrival_time:
                            is_driving = arrival_driving_start <= daily_t < ev.arrival_time
                        else:
                            is_driving = daily_t >= arrival_driving_start or daily_t < ev.arrival_time

                    if is_driving:
                        driving_consumption[t, 0] = 0.05 * (self.globals.Res / 60)  # 5% per hour

                # Modified EV balance constraint with driving consumption
                m.addConstr(
                    SoC_ev_0 * np.ones((window_steps, 1)) +
                    A @ (Grid_ev + PV_ev + Bat_ev) * (self.globals.Res / 60) * ev.charge_efficiency / self.ev_size -
                    A @ (EV_blgd + Ev_Bat) * (self.globals.Res / 60) / (self.ev_size * ev.discharge_efficiency) -
                    A @ driving_consumption == SoC_ev,  # Add driving consumption here
                    name="EV_Balance"
                )

                m.addConstr(SoC_ev >= ev.min_soc, name="EV_SoCmin")
                m.addConstr(SoC_ev <= ev.max_soc, name="EV_SoCmax")
                m.addConstr(Grid_ev + PV_ev + Bat_ev <= ev_max_chg, name="EV_ChargeLimit")
                m.addConstr(EV_blgd + Ev_Bat <= ev_max_dch, name="EV_DisChargeLimit")

                # EV availability constraints
                for t in range(window_steps):
                    if t == ev.departure_time - 1:
                        m.addConstr(SoC_ev[t, 0] >= ev.required_departure_soc, name=f"EV_Required_SOC_at_departure")
                    if t < ev.arrival_time and t >= ev.departure_time:
                        m.addConstr(EV_blgd[t, 0] == 0, name="No_EV_to_Bldg")
                        m.addConstr(Grid_ev[t, 0] == 0, name="No_Grid_to_EV")
                        m.addConstr(PV_ev[t, 0] == 0, name="No_PV_to_EV_constraint")
                        m.addConstr(Bat_ev[t, 0] == 0, name="No_Bat_to_EV_constraint")
                        m.addConstr(Ev_Bat[t, 0] == 0, name="No_EV_to_Bat_constraint")
            else:
                # If no EV, all EV-related variables should be zero
                m.addConstr(EV_blgd == 0, name="No_EV_to_Bldg")
                m.addConstr(Grid_ev == 0, name="No_Grid_to_EV")
                m.addConstr(PV_ev == 0, name="No_PV_to_EV_constraint")
                m.addConstr(Bat_ev == 0, name="No_Bat_to_EV_constraint")
                m.addConstr(Ev_Bat == 0, name="No_EV_to_Bat_constraint")

            tou = self.tou[i:end_idx].reshape(-1, 1)
            m.setObjective(tou.T @ Grid_purchase, GRB.MINIMIZE)
            m.optimize()

            # Collect results
            result_data = {
                'PV2Blgd': PV_blgd.X.reshape(-1, ),
                'G2Blgd': Grid_blgd.X.reshape(-1, ),
                'Bat2Blgd': Bat_blgd.X.reshape(-1, ),
                'EV2Blgd': EV_blgd.X.reshape(-1, ),
                'GridPurchase': Grid_purchase.X.reshape(-1, ),
                'PVcurt': PV_curtail.X.reshape(-1, ),
                'PV2Bat': PV_bat.X.reshape(-1, ),
                'PV2EV': PV_ev.X.reshape(-1, ),
                'G2Bat': Grid_bat.X.reshape(-1, ),
                'G2EV': Grid_ev.X.reshape(-1, ),
                'Bat2EV': Bat_ev.X.reshape(-1, ),
                'EV2Bat': Ev_Bat.X.reshape(-1, ),
                'TOU': tou.reshape(-1, ),
                'Load': total_load.reshape(-1, ),
            }

            # SOC variables need special handling since they might be None
            if SoC_bat is not None:
                result_data['SoC_Bat'] = SoC_bat.X.reshape(-1, )
            else:
                result_data['SoC_Bat'] = np.zeros(window_steps)

            if SoC_ev is not None:
                result_data['SoC_EV'] = SoC_ev.X.reshape(-1, )
            else:
                result_data['SoC_EV'] = np.zeros(window_steps)

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