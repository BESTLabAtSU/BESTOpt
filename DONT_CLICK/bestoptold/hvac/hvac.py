from typing import Dict, Any
import numpy as np
from ..core.base_module import BaseModule, StateVariable, ControlVariable


class HVACModule(BaseModule):
    """HVAC system module with variable speed control"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # HVAC parameters
        self.rated_cooling_capacity = config.get('rated_cooling_capacity', 10000)  # W
        self.rated_heating_capacity = config.get('rated_heating_capacity', 12000)  # W
        self.rated_power = config.get('rated_power', 3500)  # W
        self.min_plr = config.get('min_plr', 0.2)  # Minimum part load ratio

    def _initialize_variables(self):
        """Initialize HVAC variables"""
        # States
        self.states['power_consumption'] = StateVariable(
            name='power_consumption',
            value=0.0,
            unit='W',
            min_val=0.0,
            max_val=self.rated_power * 1.2
        )

        self.states['cooling_rate'] = StateVariable(
            name='cooling_rate',
            value=0.0,
            unit='W',
            min_val=-self.rated_heating_capacity,
            max_val=self.rated_cooling_capacity
        )

        self.states['cop'] = StateVariable(
            name='cop',
            value=3.0,
            unit='dimensionless',
            min_val=1.0,
            max_val=6.0
        )

        self.states['operating_mode'] = StateVariable(
            name='operating_mode',
            value=0.0,  # 0=off, 1=cooling, 2=heating
            unit='mode',
            min_val=0.0,
            max_val=2.0
        )

        # Controls
        self.controls['power_setpoint'] = ControlVariable(
            name='power_setpoint',
            value=0.0,
            unit='normalized',
            min_val=0.0,
            max_val=1.0
        )

        self.controls['mode'] = ControlVariable(
            name='mode',
            value=1.0,  # Default to cooling
            unit='mode',
            min_val=0.0,
            max_val=2.0
        )

    def update(self, dt: float, signal_bus: Dict[str, Any],
               thermal_bus: Dict[str, Any], electric_bus: Dict[str, Any]):
        """Update HVAC state based on control signals and conditions"""

        # Get zone temperature and setpoint from buses
        zone_temp = thermal_bus.get('zone_temp', 22.0)
        setpoint = signal_bus.get('temp_setpoint', 22.0)
        outdoor_temp = signal_bus.get('outdoor_temp', 20.0)

        # Determine operating mode based on temperature difference
        temp_diff = zone_temp - setpoint

        if abs(temp_diff) < 0.5:  # Deadband
            self.states['operating_mode'].update(0.0)  # Off
        elif temp_diff > 0.5:
            self.states['operating_mode'].update(1.0)  # Cooling
        else:
            self.states['operating_mode'].update(2.0)  # Heating

        # Calculate COP based on operating conditions
        if self.states['operating_mode'].value == 1.0:  # Cooling
            # COP decreases with outdoor temperature
            cop = 4.0 - 0.05 * (outdoor_temp - 25)
        elif self.states['operating_mode'].value == 2.0:  # Heating
            # COP decreases with lower outdoor temperature
            cop = 3.5 - 0.04 * (15 - outdoor_temp)
        else:
            cop = 3.0

        cop = np.clip(cop, 2.0, 5.0)
        self.states['cop'].update(cop)

        # Calculate power consumption based on control signal
        if self.states['operating_mode'].value > 0:
            plr = max(self.min_plr, self.controls['power_setpoint'].value)
            power = self.rated_power * plr * self._plr_efficiency_factor(plr)

            # Calculate cooling/heating rate
            if self.states['operating_mode'].value == 1.0:
                cooling_rate = power * cop
            else:
                cooling_rate = -power * cop  # Negative for heating
        else:
            power = 0.0
            cooling_rate = 0.0

        self.states['power_consumption'].update(power)
        self.states['cooling_rate'].update(cooling_rate)

        # Update electric bus with power consumption
        electric_bus['hvac_power'] = power / 1000.0  # Convert to kW

    def _plr_efficiency_factor(self, plr: float) -> float:
        """Calculate efficiency degradation at part load"""
        # Typical part load efficiency curve
        if plr < 0.3:
            return 0.7 + plr
        elif plr < 0.7:
            return 1.0
        else:
            return 1.0 - 0.1 * (plr - 0.7)