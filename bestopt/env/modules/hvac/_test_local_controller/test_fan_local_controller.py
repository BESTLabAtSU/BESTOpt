# bestopt/env/controllers/test_fan_local_controller.py
from dataclasses import dataclass

from bestopt.env.modules.hvac.local_controller.fan_local_controller import FanLocalController
from bestopt.env.core.data_structure import ThermalAction, HVACLocalAction

fan_local_controller = FanLocalController({}, name="fan_local_controller")
fan_local_controller.initialize()


ip_act = ThermalAction(supervisory_supply_air_flow_rate=1.0)

op_act = fan_local_controller.step(action=ip_act, timestep=900)


print(op_act.fan_supply_air_flow_rate)   

# Next step
# For example: fan.step(action=op_act.fan_supply_air_flow_rate)