"""
Fan Coil Unit (FCU) system module.
"""

from typing import Dict, Any

from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import (
    ThermalAction, Disturbance,ThermalDomainState,
    HVACLocalAction,
    CoilState, FanState, PumpState, ChillerState, CoolingTowerState
)

from bestopt.env.modules.hvac.component.fan import FanModule
from bestopt.env.modules.hvac.component.coil import CoilModule
from bestopt.env.modules.hvac.component.pump import PumpModule
from bestopt.env.modules.hvac.component.chiller import ChillerModule
from bestopt.env.modules.hvac.component.cooling_tower import CoolingTowerModule

from bestopt.env.modules.hvac.local_controller.fan_local_controller import FanLocalController
from bestopt.env.modules.hvac.local_controller.pump_local_controller import PumpLocalController


class FCUModule(BaseModule):
    """
    Fan Coil Unit (FCU) system that includes:
      - 1x Fan (airflow control)
      - 1x Coil (cooling)
      - 1x Pump (CHW)
      - 1x Chiller
      - 1x Cooling Tower

    Control sequence:
      SupervisoryController → FanLocalController → Fan → Coil
      SupervisoryController → PumpLocalController(placeholder) → Pump → Coil → Chiller → CoolingTower

    Inputs:
      - action.Q_zone_W (desired cooling, negative)
      - action.supply_air_temp_sp_C (SAT setpoint)
      - action.supervisory_supply_air_flow_rate (airflow setpoint)
      - action.chws_temp_c_sp (CHW setpoint)
      - action.condenser_temp_c_sp (CW setpoint)
      - weather.outdoor_wet_bulb_temperature

    Outputs:
      - coil_state.Q_W               → Actual cooling [W]
      - coil_state.air_outlet_temp_C → Actual SAT [°C]
      - fan_state.airflow_m3s        → Actual SA flow rate [m³/s]
      - Total system power [W]
      - Total system energy [J]
    """

    def __init__(self, config: Dict[str, Any], name: str = "fcu_system"):
        super().__init__(config, name)

        # === Submodules ===
        self.fan = FanModule(config.get("fan", {}), name="supply_fan")
        self.fan_ctrl = FanLocalController(config.get("fan_ctrl", {}), name="fan_controller")

        self.coil = CoilModule(config.get("coil", {}), name="cooling_coil")

        self.pump = PumpModule(config.get("pump", {}), name="chilled_water_pump")
        self.pump_ctrl = PumpLocalController(config.get("pump_ctrl", {}), name="pump_controller")
        
        self.chiller = ChillerModule(config.get("chiller", {}), name="chiller")
        self.tower = CoolingTowerModule(config.get("tower", {}), name="cooling_tower")

        # === States ===
        self.fan_state = FanState(component_id="fan", component_type="fan", domain="air")
        self.coil_state = CoilState(component_id="coil", component_type="coil", domain="air-water")
        self.pump_state = PumpState(component_id="pump", component_type="pump", domain="water")
        self.chiller_state = ChillerState(component_id="chiller", component_type="chiller", domain="water")
        self.tower_state = CoolingTowerState(component_id="tower", component_type="tower", domain="water")

    def initialize(self) -> None:
        self.fan.initialize()
        self.fan_ctrl.initialize()
        self.coil.initialize()
        self.pump.initialize()
        self.pump_ctrl.initialize()
        self.chiller.initialize()
        self.tower.initialize()
        self._initialized = True

    def reset(self) -> None:
        self.fan.reset()
        self.fan_ctrl.reset()
        self.coil.reset()
        self.pump.reset()
        self.pump_ctrl.reset()
        self.chiller.reset()
        self.tower.reset()
        self._initialized = False
        self.initialize()

    def step(
        self,
        state: ThermalDomainState,
        action: ThermalAction,
        disturbance: Disturbance,
        timestep: float
    ) -> Dict[str, Any]:
        """
        Step all submodules in FCU loop.
        """

        # === 1. Fan control (local) ===
        fan_local_cmd: HVACLocalAction = self.fan_ctrl.step(action, timestep)

        # === 2. Fan actuation ===
        self.fan.step(self.fan_state, fan_local_cmd, timestep)

        # === 3. Pump control (local) ===
        pump_local_cmd: HVACLocalAction = self.pump_ctrl.step(state, action, timestep)

        # === 4. Pump step ===
        self.pump.step(self.pump_state, pump_local_cmd, timestep) #disturbance, 

        # === 5. Coil step (pure thermodynamics) ===
        self.coil_state.airflow_m3s = self.fan_state.airflow_m3s
        self.coil_state.waterflow_m3s = self.pump_state.waterflow_m3s
        self.coil_state.air_inlet_temp_C = getattr(action, "return_air_temperature", 26.0)
        self.coil_state.water_inlet_temp_C = self.chiller_state.chws_temp_c

        self.coil.step(self.coil_state, action, timestep)

        # === 6. Chiller step ===
        self.chiller.step(self.chiller_state, action, self.coil_state, self.pump_state, timestep)

        # === 7. Cooling tower step ===
        self.tower.step(self.tower_state, action, self.chiller_state, disturbance.weather, timestep)

        # === Aggregate outputs ===
        power_total_W = (
            self.fan_state.power_W
            + self.pump_state.power_W
            + self.chiller_state.power_W
            + self.tower_state.fan_power_W
            + self.tower_state.pump_power_W
        )
        energy_total_J = (
            self.fan_state.energy_J_cum
            + self.pump_state.energy_J_cum
            + self.chiller_state.energy_J_cum
            + self.tower_state.energy_J_cum
        )

        self.Q_zone_actual_W = -self.coil_state.Q_W
        self.SAT_actual_C = self.coil_state.air_outlet_temp_C
        self.SA_flow_actual_m3s = self.fan_state.airflow_m3s
        self.CHW_flow_actual_m3s = self.pump_state.waterflow_m3s
        self.FCU_power_total_W = power_total_W
        self.FCU_energy_cumulative_J = energy_total_J

