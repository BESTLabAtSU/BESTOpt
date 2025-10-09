"""
DER System Module - Orchestrates PV, Battery, and EV component modules
"""
from typing import Dict, Any, List, Optional
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import (
    ElectricalAction, Disturbance, DERSystemState,
    PVState, BatteryState, EVState, BatteryMode, EVMode
)

from bestopt.env.modules.ders.component.pv import PVModule
from bestopt.env.modules.ders.component.battery import BatteryModule
from bestopt.env.modules.ders.component.ev import EVModule



class DERModule(BaseModule):
    """
    DER System that orchestrates multiple PV, Battery, and EV component modules.

    This module acts as a system-level coordinator, managing:
    - Multiple PV modules
    - Multiple Battery modules  
    - Multiple EV modules
    - Local controller for power flow management
    """

    def __init__(self, config: Dict[str, Any], name: str = "der_system"):
        super().__init__(config, name)
        self.system_config = config.get("system_config", {})

        # Initialize component module containers
        self.pv_modules: Dict[str, PVModule] = {}
        self.battery_modules: Dict[str, BatteryModule] = {}
        self.ev_modules: Dict[str, EVModule] = {}

        # Parse configurations and create component modules
        self._create_component_modules()

        self.logger.info(f"DER system initialized with: "
                         f"{len(self.pv_modules)} PV modules, "
                         f"{len(self.battery_modules)} battery modules, "
                         f"{len(self.ev_modules)} EV modules")

    def _create_component_modules(self):
        """Create component module instances based on configuration."""

        # Create PV modules
        pv_configs = self._parse_component_config('pvs', 'pv')
        for pv_id, pv_config in pv_configs.items():
            self.pv_modules[pv_id] = PVModule(
                config=pv_config,
                name=f"{self.name}_{pv_id}"
            )

        # Create Battery modules
        battery_configs = self._parse_component_config('batteries', 'bat')
        for bat_id, bat_config in battery_configs.items():
            self.battery_modules[bat_id] = BatteryModule(
                config=bat_config,
                name=f"{self.name}_{bat_id}"
            )

        # Create EV modules
        ev_configs = self._parse_component_config('evs', 'ev')
        for ev_id, ev_config in ev_configs.items():
            self.ev_modules[ev_id] = EVModule(
                config=ev_config,
                name=f"{self.name}_{ev_id}"
            )

    def _parse_component_config(self, plural_key: str, singular_key: str) -> Dict[str, Dict[str, Any]]:
        """
        Parse component configuration supporting both single and multiple formats.

        Args:
            plural_key: Key for multiple components (e.g., 'batteries')
            singular_key: Key for single component (e.g., 'bat')

        Returns:
            Dictionary of component configurations by ID
        """
        configs = {}

        # Check for multiple components
        if plural_key in self.system_config:
            multi_config = self.system_config[plural_key]

            if isinstance(multi_config, list):
                # List format
                for idx, comp_config in enumerate(multi_config):
                    comp_id = comp_config.get('id', f"{singular_key}_{idx + 1}")
                    configs[comp_id] = comp_config
            elif isinstance(multi_config, dict):
                # Dictionary format
                for comp_id, comp_config in multi_config.items():
                    if isinstance(comp_config, dict):
                        comp_config['id'] = comp_id
                    configs[comp_id] = comp_config

        # Check for single component (backward compatibility)
        elif singular_key in self.system_config:
            single_config = self.system_config[singular_key]
            comp_id = single_config.get('id', f"{singular_key}_1")
            configs[comp_id] = single_config

        return configs

    def initialize(self) -> None:
        """Initialize all component modules and local controller."""
        # Initialize all PV modules
        for pv_id, pv_module in self.pv_modules.items():
            pv_module.initialize()
            self.logger.debug(f"Initialized PV module: {pv_id}")

        # Initialize all battery modules
        for bat_id, bat_module in self.battery_modules.items():
            bat_module.initialize()
            self.logger.debug(f"Initialized battery module: {bat_id}")

        # Initialize all EV modules
        for ev_id, ev_module in self.ev_modules.items():
            ev_module.initialize()
            self.logger.debug(f"Initialized EV module: {ev_id}")

        self._initialized = True
        self.logger.info(f"DER system fully initialized: {self.name}")

    def reset(self) -> None:
        """Reset all component modules and local controller."""
        # Reset all component modules
        for pv_module in self.pv_modules.values():
            pv_module.reset()

        for bat_module in self.battery_modules.values():
            bat_module.reset()

        for ev_module in self.ev_modules.values():
            ev_module.reset()

        self.logger.debug(f"DER system reset: {self.name}")

    def step(self, state: DERSystemState, action: Any, disturbance: Disturbance,
             resolution: int, timestep: float,
             aggregated_loads: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Enhanced step function with proper state updates from component results.
        """
        outputs = {}

        # Handle both simple and complex action formats
        if hasattr(action, 'battery_power'):
            battery_commands = action.battery_power
            ev_commands = action.ev_power
            pv_curtailment = action.pv_curtailment
        else:
            battery_commands = getattr(action, 'battery_commands', {})
            ev_commands = getattr(action, 'ev_commands', {})
            pv_curtailment = getattr(action, 'pv_commands', {})

        # Step all PV modules
        for pv_id, pv_module in self.pv_modules.items():
            if pv_id in state.pv_systems:
                pv_state = state.pv_systems[pv_id]
                curtailment = pv_curtailment.get(pv_id, 0.0)

                # Create action for PV
                pv_action = type('PVAction', (), {
                    'curtailment_factor': curtailment
                })()

                # Step PV module - it returns updated values
                pv_result = pv_module.step(
                    state=pv_state,
                    action=pv_action,
                    disturbance=disturbance,
                    timestep=timestep,
                    resolution=resolution
                )

                # Update state with results
                if 'power_generation' in pv_result:
                    pv_state.pv_generation = pv_result['power_generation']
                if 'power_available' in pv_result:
                    pv_state.pv_available = pv_result['power_available']
                if 'power_curtailed' in pv_result:
                    pv_state.pv_curtailed = pv_result['power_curtailed']

                # Store in outputs
                outputs[f'pv_{pv_id}_generation'] = pv_state.pv_generation
                outputs[f'pv_{pv_id}_available'] = pv_state.pv_available
                outputs[f'pv_{pv_id}_curtailed'] = pv_state.pv_curtailed

        # Step all battery modules
        for bat_id, bat_module in self.battery_modules.items():
            if bat_id in state.batteries:
                bat_state = state.batteries[bat_id]
                power_command = battery_commands.get(bat_id, 0.0)

                # Create action for battery
                bat_action = type('BatteryAction', (), {
                    'power_command': power_command,
                    'resolution': resolution
                })()

                # Step battery module
                bat_result = bat_module.step(
                    state=bat_state,
                    action=bat_action,
                    disturbance=disturbance,
                    timestep=timestep
                )

                # Update state with results
                if 'soc' in bat_result:
                    bat_state.battery_soc = bat_result['soc']
                if 'power_actual' in bat_result:
                    bat_state.battery_power = bat_result['power_actual']
                if 'losses' in bat_result:
                    bat_state.battery_losses = bat_result['losses']
                if 'mode' in bat_result:
                    bat_state.battery_mode = bat_result['mode']

                # Store in outputs
                outputs[f'battery_{bat_id}_soc'] = bat_state.battery_soc
                outputs[f'battery_{bat_id}_power'] = bat_state.battery_power
                outputs[f'battery_{bat_id}_mode'] = bat_state.battery_mode.value if hasattr(bat_state.battery_mode,
                                                                                            'value') else str(
                    bat_state.battery_mode)
                outputs[f'battery_{bat_id}_losses'] = bat_state.battery_losses

        # Step all EV modules
        # for ev_id, ev_module in self.ev_modules.items():
        #     if ev_id in state.evs:
        #         ev_state = state.evs[ev_id]
        #         power_command = ev_commands.get(ev_id, 0.0)
        #
        #         # Create action for EV
        #         ev_action = type('EVAction', (), {
        #             'power_command': power_command,
        #             'resolution': resolution
        #         })()
        #
        #         # Step EV module
        #         ev_result = ev_module.step(
        #             state=ev_state,
        #             action=ev_action,
        #             disturbance=disturbance,
        #             timestep=timestep
        #         )
        #
        #         # Update state with results
        #         if 'soc' in ev_result:
        #             ev_state.ev_soc = ev_result['soc']
        #         if 'power_actual' in ev_result:
        #             ev_state.ev_power = ev_result['power_actual']
        #         if 'mode' in ev_result:
        #             ev_state.ev_mode = ev_result['mode']
        #         if 'is_connected' in ev_result:
        #             ev_state.is_connected = ev_result['is_connected']
        #         if 'losses' in ev_result:
        #             ev_state.ev_losses = ev_result['losses']
        #
        #         # Store in outputs
        #         outputs[f'ev_{ev_id}_soc'] = ev_state.ev_soc
        #         outputs[f'ev_{ev_id}_power'] = ev_state.ev_power
        #         outputs[f'ev_{ev_id}_connected'] = ev_state.is_connected
        #         outputs[f'ev_{ev_id}_mode'] = ev_state.ev_mode.value if hasattr(ev_state.ev_mode, 'value') else str(
        #             ev_state.ev_mode)

        # Update system-level aggregations AFTER component updates
        state.update_aggregations()

        # Calculate system-level metrics using updated states
        outputs['total_pv_generation'] = sum(
            pv.pv_generation for pv in state.pv_systems.values()
        )
        outputs['total_pv_available'] = sum(
            pv.pv_available for pv in state.pv_systems.values()
        )
        outputs['total_battery_power'] = sum(
            bat.battery_power for bat in state.batteries.values()
        )
        # outputs['total_ev_power'] = sum(
        #     ev.ev_power for ev in state.evs.values() if ev.is_connected
        # )

        # Calculate net power flows
        total_generation = outputs['total_pv_generation']
        total_storage_discharge = sum(
            bat.battery_power for bat in state.batteries.values() if bat.battery_power < 0
        )
        total_storage_charge = sum(
            bat.battery_power for bat in state.batteries.values() if bat.battery_power > 0
        )
        # total_ev_discharge = sum(
        #     ev.ev_power for ev in state.evs.values() if ev.is_connected and ev.ev_power < 0
        # )
        # total_ev_charge = sum(
        #     ev.ev_power for ev in state.evs.values() if ev.is_connected and ev.ev_power > 0
        # )

        outputs['total_generation'] = total_generation
        outputs['total_storage_discharge'] = abs(total_storage_discharge)
        outputs['total_storage_charge'] = total_storage_charge
        # outputs['net_generation'] = total_generation + abs(total_storage_discharge) + abs(total_ev_discharge)

        # If centralized, include aggregated load info
        if aggregated_loads:
            outputs['serviced_loads'] = aggregated_loads
            outputs['total_load_served'] = sum(aggregated_loads.values())

            # Calculate grid flows based on net balance
            net_balance = outputs['net_generation'] - outputs[
                'total_load_served'] - total_storage_charge - total_ev_charge
            if net_balance > 0:
                outputs['grid_export'] = net_balance
                outputs['grid_import'] = 0
            else:
                outputs['grid_import'] = abs(net_balance)
                outputs['grid_export'] = 0

        # Store component outputs for detailed analysis
        outputs['pv_details'] = {
            pv_id: {
                'generation': state.pv_systems[pv_id].pv_generation,
                'available': state.pv_systems[pv_id].pv_available,
                'curtailed': state.pv_systems[pv_id].pv_curtailed
            } for pv_id in state.pv_systems
        }
        outputs['battery_details'] = {
            bat_id: {
                'soc': state.batteries[bat_id].battery_soc,
                'power': state.batteries[bat_id].battery_power,
                'mode': str(state.batteries[bat_id].battery_mode)
            } for bat_id in state.batteries
        }
        # outputs['ev_details'] = {
        #     ev_id: {
        #         'soc': state.evs[ev_id].ev_soc,
        #         'power': state.evs[ev_id].ev_power,
        #         'connected': state.evs[ev_id].is_connected,
        #         'mode': str(state.evs[ev_id].ev_mode)
        #     } for ev_id in state.evs
        # }

        # Log summary
        # self.logger.debug(
        #     f"DER step complete - PV: {total_generation:.2f}W, "
        #     f"Battery: {outputs['total_battery_power']:.2f}W, "
        #     f"EV: {outputs['total_ev_power']:.2f}W"
        # )

        return outputs

    def get_state(self) -> Dict[str, Any]:
        """Get current state of all component modules."""
        state = {
            'pv_modules': {},
            'battery_modules': {},
            'ev_modules': {},
        }

        # Get state from each component module
        for pv_id, pv_module in self.pv_modules.items():
            state['pv_modules'][pv_id] = pv_module.get_state()

        for bat_id, bat_module in self.battery_modules.items():
            state['battery_modules'][bat_id] = bat_module.get_state()

        for ev_id, ev_module in self.ev_modules.items():
            state['ev_modules'][ev_id] = ev_module.get_state()

        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state of all component modules."""
        # Set state for PV modules
        if 'pv_modules' in state:
            for pv_id, pv_state in state['pv_modules'].items():
                if pv_id in self.pv_modules:
                    self.pv_modules[pv_id].set_state(pv_state)

        # Set state for battery modules
        if 'battery_modules' in state:
            for bat_id, bat_state in state['battery_modules'].items():
                if bat_id in self.battery_modules:
                    self.battery_modules[bat_id].set_state(bat_state)

        # Set state for EV modules
        if 'ev_modules' in state:
            for ev_id, ev_state in state['ev_modules'].items():
                if ev_id in self.ev_modules:
                    self.ev_modules[ev_id].set_state(ev_state)

    def get_component_info(self) -> Dict[str, Any]:
        """Get information about all components in the system."""
        return {
            'pv_count': len(self.pv_modules),
            'battery_count': len(self.battery_modules),
            'ev_count': len(self.ev_modules),
            'pv_ids': list(self.pv_modules.keys()),
            'battery_ids': list(self.battery_modules.keys()),
            'ev_ids': list(self.ev_modules.keys()),
            'total_pv_capacity': sum(
                m.config.get('rated_capacity_kW', 0) for m in self.pv_modules.values()
            ),
            'total_battery_capacity': sum(
                m.config.get('rated_capacity_kWh', 0) for m in self.battery_modules.values()
            ),
            'total_ev_capacity': sum(
                m.config.get('rated_capacity_kWh', 0) for m in self.ev_modules.values()
            )
        }