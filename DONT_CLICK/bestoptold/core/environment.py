from typing import Dict, Any, List
from dataclasses import dataclass
import pandas as pd


@dataclass
class SimulationStep:
    """Data structure for storing simulation step results"""
    timestamp: pd.Timestamp
    states: Dict[str, float]
    controls: Dict[str, float]
    disturbances: Dict[str, float]
    measurements: Dict[str, float]


class Environment:
    """Simulation environment"""

    def __init__(self,
                 modules: Dict[str, Any],
                 controller: Any,
                 start_time: pd.Timestamp,
                 dt: float = 900):  # 15-min timestep
        self.modules = modules
        self.controller = controller
        self.current_time = start_time
        self.dt = dt  # seconds

        # Three communication buses
        self.signal_bus = {}
        self.thermal_bus = {}
        self.electric_bus = {}

        # Data storage
        self.history: List[SimulationStep] = []

    def step(self):
        """Execute one simulation timestep"""
        # 1. Take measurements from all modules
        measurements = self._take_measurements()

        # 2. Get disturbances
        disturbances = self._get_disturbances()

        # 3. Get reference signals (from occupancy)
        references = self._get_references()

        # 4. Make control decisions
        control_actions = self.controller.compute_control(
            measurements, disturbances, references, self.current_time
        )

        # 5. Apply control actions
        self._apply_control_actions(control_actions)

        # 6. Update all modules
        self._update_modules()

        # 7. Update buses
        self._update_buses()

        # 8. Store data
        self._store_data(measurements, control_actions, disturbances)

        # 9. Advance time
        self.current_time += pd.Timedelta(seconds=self.dt)

    def _take_measurements(self) -> Dict[str, float]:
        """Collect measurements from all modules"""
        measurements = {}

        # Building measurements
        if 'building' in self.modules:
            measurements['zone_temp'] = self.modules['building'].get_state('zone_temperature')
            measurements['zone_humidity'] = self.modules['building'].get_state('zone_humidity')

        # HVAC measurements
        if 'hvac' in self.modules:
            measurements['hvac_power'] = self.modules['hvac'].get_state('power_consumption')
            measurements['cooling_rate'] = self.modules['hvac'].get_state('cooling_rate')

        # DER measurements
        if 'pv' in self.modules:
            measurements['pv_power'] = self.modules['pv'].get_state('power_output')
        if 'battery' in self.modules:
            measurements['battery_soc'] = self.modules['battery'].get_state('soc')
            measurements['battery_power'] = self.modules['battery'].get_state('power')
        if 'ev' in self.modules:
            measurements['ev_soc'] = self.modules['ev'].get_state('soc')
            measurements['ev_connected'] = self.modules['ev'].get_state('connected')

        return measurements

    def _get_disturbances(self) -> Dict[str, float]:
        """Get current disturbances values"""
        if 'disturbances' not in self.modules:
            return {}

        dist_module = self.modules['disturbances']
        return {
            'outdoor_temp': dist_module.get_disturbance('outdoor_temperature'),
            'solar_radiation': dist_module.get_disturbance('solar_radiation'),
            'electricity_price': dist_module.get_disturbance('electricity_price')
        }

    def _get_references(self) -> Dict[str, float]:
        """Get reference signals from occupancy"""
        if 'occupancy' not in self.modules:
            return {}

        occ_module = self.modules['occupancy']
        return {
            'temp_setpoint': occ_module.get_comfort_setpoint(),
            'occupancy_status': occ_module.get_occupancy_status(),
            'appliance_load': occ_module.get_appliance_load()
        }

    def _apply_control_actions(self, actions: Dict[str, float]):
        """Apply control actions to modules"""
        # HVAC controls
        if 'hvac' in self.modules and 'hvac_power' in actions:
            self.modules['hvac'].set_control('power_setpoint', actions['hvac_power'])

        # DER power flow controls
        if 'battery' in self.modules:
            if 'grid2battery' in actions:
                self.modules['battery'].set_control('grid_charge', actions['grid2battery'])
            if 'pv2battery' in actions:
                self.modules['battery'].set_control('pv_charge', actions['pv2battery'])

        if 'ev' in self.modules and self.modules['ev'].get_state('connected'):
            if 'grid2ev' in actions:
                self.modules['ev'].set_control('grid_charge', actions['grid2ev'])
            if 'pv2ev' in actions:
                self.modules['ev'].set_control('pv_charge', actions['pv2ev'])

        # Update power flow controls in electric bus
        self.electric_bus.update(actions)

    def _update_modules(self):
        """Update all modules with current bus values"""
        for name, module in self.modules.items():
            module.update(self.dt, self.signal_bus, self.thermal_bus, self.electric_bus)

    def _update_buses(self):
        """Update communication buses with latest values"""
        # Signal bus (control signals, setpoints)
        self.signal_bus['temp_setpoint'] = self.modules['occupancy'].get_comfort_setpoint()
        self.signal_bus['occupancy'] = self.modules['occupancy'].get_occupancy_status()

        # Thermal bus (temperatures, heat flows)
        if 'building' in self.modules:
            self.thermal_bus['zone_temp'] = self.modules['building'].get_state('zone_temperature')
        if 'hvac' in self.modules:
            self.thermal_bus['cooling_power'] = self.modules['hvac'].get_state('cooling_rate')

        # Electric bus (power flows, already updated in _apply_control_actions)
        if 'pv' in self.modules:
            self.electric_bus['pv_available'] = self.modules['pv'].get_state('power_output')

    def _store_data(self, measurements: Dict[str, float],
                    controls: Dict[str, float], disturbances: Dict[str, float]):
        """Store simulation data for analysis"""
        step_data = SimulationStep(
            timestamp=self.current_time,
            states={name: mod.states for name, mod in self.modules.items()},
            controls=controls,
            disturbances=disturbances,
            measurements=measurements
        )
        self.history.append(step_data)

    def run(self, steps: int):
        """Run simulation for specified number of steps"""
        for _ in range(steps):
            self.step()

    def get_results(self) -> pd.DataFrame:
        """Convert simulation history to DataFrame"""
        data = []
        for step in self.history:
            row = {'timestamp': step.timestamp}
            row.update(step.measurements)
            row.update({f'control_{k}': v for k, v in step.controls.items()})
            row.update({f'dist_{k}': v for k, v in step.disturbances.items()})
            data.append(row)
        return pd.DataFrame(data)