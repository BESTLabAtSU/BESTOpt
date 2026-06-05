"""
DER System Module - Simplified Power Flow Model

Key changes from old version:
- battery_power: positive = charge, negative = discharge [kW]
- ev_charging: automatic charging power when connected [kW]
- No more pv2battery, grid2battery, etc. - just net power commands
"""
from typing import Dict, Any, Tuple, Optional
from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import (
    DERSystemAction, Disturbance, DERSystemState,
    PVComponentState, BatteryComponentState, EVComponentState,
    ComponentType, ComponentState
)

from bestopt.env.modules.ders.component.pv import PVModule
from bestopt.env.modules.ders.component.battery import BatteryModule
from bestopt.env.modules.ders.component.ev import EVModule


class ComponentRegistry:
    """Registry that manages component-state pairs."""

    @staticmethod
    def create_component(component_type: ComponentType,
                         component_id: str,
                         config: Dict[str, Any],
                         system_id: str) -> Tuple[BaseModule, ComponentState]:
        """Create both module and state together."""

        if component_type == ComponentType.PV:
            module = PVModule(config=config, name=component_id)
            state = PVComponentState(
                component_id=component_id,
                component_type=component_type,
                system_id=system_id
            )
            return module, state
        elif component_type == ComponentType.BATTERY:
            module = BatteryModule(config=config, name=component_id)
            state = BatteryComponentState(
                component_id=component_id,
                component_type=component_type,
                system_id=system_id
            )
            return module, state
        elif component_type == ComponentType.EV:
            module = EVModule(config=config, name=component_id)
            state = EVComponentState(
                component_id=component_id,
                component_type=component_type,
                system_id=system_id
            )
            return module, state
        else:
            raise ValueError(f"Unsupported component type: {component_type}")


class DERModule(BaseModule):
    """
    DER System using simplified power flow model.

    Power Flow Philosophy:
    - Electrons are fungible (can't distinguish PV vs grid power)
    - Only battery charge/discharge is controllable
    - EV charging is automatic when connected
    - Controller computes: grid_import = demand + battery_power - pv_generation
    """

    def __init__(self, config: Dict[str, Any], name: str = "der_system"):
        super().__init__(config, name)
        self.system_config = config.get("system_config", {})

        # Parse configurations
        self.pv_configs = self._parse_component_config('pvs', 'pv')
        self.battery_configs = self._parse_component_config('batteries', 'bat')
        self.ev_configs = self._parse_component_config('evs', 'ev')

        # Module containers
        self.pv_modules: Dict[str, Any] = {}
        self.battery_modules: Dict[str, Any] = {}
        self.ev_modules: Dict[str, Any] = {}

        # State containers
        self.pv_states: Dict[str, Any] = {}
        self.battery_states: Dict[str, Any] = {}
        self.ev_states: Dict[str, Any] = {}

        # Create registry instance
        self.registry = ComponentRegistry()

        self.logger.info(f"DER system configured with: "
                         f"{len(self.pv_configs)} PV configs, "
                         f"{len(self.battery_configs)} battery configs, "
                         f"{len(self.ev_configs)} EV configs")

    def register_component_state(self, state: DERSystemState):
        """Register all components and their initial states."""

        # Create PV components
        for pv_id, pv_config in self.pv_configs.items():
            module, component_state = self.registry.create_component(
                component_type=ComponentType.PV,
                component_id=pv_id,
                config=pv_config,
                system_id=state.system_id
            )

            self.pv_modules[pv_id] = module
            self.pv_states[pv_id] = component_state
            state.components[pv_id] = component_state

            self.logger.debug(f"Registered PV component: {pv_id}")

        # Create Battery components
        for bat_id, bat_config in self.battery_configs.items():
            module, component_state = self.registry.create_component(
                component_type=ComponentType.BATTERY,
                component_id=bat_id,
                config=bat_config,
                system_id=state.system_id,
            )
            component_state.soc = bat_config.get('initial_soc', 0.5)
            component_state.capacity_kwh = bat_config.get('rated_capacity_kWh', 10.0)
            component_state.charge_speed = bat_config.get('charge_speed', 0.5)
            component_state.discharge_speed = bat_config.get('discharge_speed', 0.5)
            component_state.charge_efficiency = bat_config.get('charge_efficiency', 0.95)

            self.battery_modules[bat_id] = module
            self.battery_states[bat_id] = component_state
            state.components[bat_id] = component_state

            self.logger.debug(f"Registered Battery component: {bat_id}")

        # Create EV components
        for ev_id, ev_config in self.ev_configs.items():
            module, component_state = self.registry.create_component(
                component_type=ComponentType.EV,
                component_id=ev_id,
                config=ev_config,
                system_id=state.system_id,
            )
            component_state.soc = ev_config.get('initial_soc', 0.5)
            component_state.capacity_kwh = ev_config.get('rated_capacity_kWh', 40.0)
            component_state.charge_speed = ev_config.get('charge_speed', 0.5)
            component_state.discharge_speed = ev_config.get('discharge_speed', 0.5)
            component_state.charge_efficiency = ev_config.get('charge_efficiency', 0.95)

            self.ev_modules[ev_id] = module
            self.ev_states[ev_id] = component_state
            state.components[ev_id] = component_state

            self.logger.debug(f"Registered EV component: {ev_id}")

        self.logger.info(f"Registered {len(state.components)} components for DER system {state.system_id}")

    def initialize(self) -> None:
        """Initialize all component modules."""
        for pv_id, pv_module in self.pv_modules.items():
            pv_module.initialize()
            self.logger.debug(f"Initialized PV module: {pv_id}")

        for bat_id, bat_module in self.battery_modules.items():
            bat_module.initialize()
            self.logger.debug(f"Initialized battery module: {bat_id}")

        for ev_id, ev_module in self.ev_modules.items():
            ev_module.initialize()
            self.logger.debug(f"Initialized EV module: {ev_id}")

        self._initialized = True
        self.logger.info(f"DER system fully initialized: {self.name}")

    def calculate_pv_generation(self, state: DERSystemState, disturbance, timestep):
        """Calculate PV generation for all PV components."""
        total_generation = 0.0

        for pv_id, pv_module in self.pv_modules.items():
            pv_state = state.components.get(pv_id)

            if pv_state and pv_state.component_type == ComponentType.PV:
                pv_state = pv_module.step(
                    state=pv_state,
                    disturbance=disturbance,
                    resolution=900,
                    timestep=timestep,
                )

                total_generation += pv_state.generation_w
                self.logger.debug(f"PV {pv_id}: {pv_state.generation_w:.1f}W")

        return total_generation

    def step(self, state: DERSystemState, action: DERSystemAction, disturbance: Disturbance,
             resolution: int, timestep: float) -> Dict[str, Any]:
        """
        Step the DER system forward using simplified power flow model.

        Args:
            state: Current DER system state
            action: DERSystemAction with:
                - battery_power: {bat_id: power_kw} (+ charge, - discharge)
                - ev_charging: {ev_id: power_kw} (automatic charging)
                - grid_import: net grid power [kW]
                - curtailment: curtailed PV [kW]
            disturbance: Current disturbances
            resolution: Time resolution in seconds
            timestep: Current timestep

        Returns:
            Results dictionary with component outcomes
        """
        results = {
            'pv_results': {},
            'battery_results': {},
            'ev_results': {},
            'total_pv_generation': 0.0,
            'total_battery_power': 0.0,
            'total_ev_charging': 0.0,
            'grid_import': getattr(action, 'grid_import', 0.0),
            'curtailment': getattr(action, 'curtailment', 0.0),
        }

        # Step 1: Process Battery modules
        # battery_power: positive = charging, negative = discharging
        battery_power_dict = getattr(action, 'battery_power', {})

        for bat_id, bat_module in self.battery_modules.items():
            bat_state = state.components.get(bat_id)

            if bat_state and bat_state.component_type == ComponentType.BATTERY:
                # Get battery power command (positive = charge, negative = discharge)
                net_power_kw = battery_power_dict.get(bat_id, 0.0)

                # Step the battery module
                bat_state = bat_module.step(
                    state=bat_state,
                    action={'net_power_kw': net_power_kw},
                    disturbance=disturbance,
                    timestep=timestep
                )

                results['battery_results'][bat_id] = {
                    'power_kw': net_power_kw,
                    'soc': bat_state.soc
                }
                results['total_battery_power'] += net_power_kw

                self.logger.debug(f"Battery {bat_id}: power={net_power_kw:.2f}kW, SOC={bat_state.soc:.1%}")

        # Step 2: Process EV modules
        # ev_charging: automatic charging power (always positive or zero)
        ev_charging_dict = getattr(action, 'ev_charging', {})

        for ev_id, ev_module in self.ev_modules.items():
            ev_state = state.components.get(ev_id)

            if ev_state and ev_state.component_type == ComponentType.EV:
                # Get EV charging power (positive = charging, no V2G in simplified model)
                charging_power_kw = ev_charging_dict.get(ev_id, 0.0)

                # Step the EV module
                ev_state = ev_module.step(
                    state=ev_state,
                    action={'net_power_kw': charging_power_kw},  # Positive = charging
                    disturbance=disturbance,
                    timestep=timestep
                )

                results['ev_results'][ev_id] = {
                    'charging_power_kw': charging_power_kw,
                    'soc': ev_state.soc,
                    'is_connected': getattr(ev_state, 'is_connected', True)
                }
                results['total_ev_charging'] += charging_power_kw

                self.logger.debug(f"EV {ev_id}: charging={charging_power_kw:.2f}kW, SOC={ev_state.soc:.1%}")

        # Log summary
        self.logger.debug(
            f"DER Step Summary - "
            f"Battery: {results['total_battery_power']:.2f}kW, "
            f"EV Charging: {results['total_ev_charging']:.2f}kW, "
            f"Grid Import: {results['grid_import']:.2f}kW, "
            f"Curtailment: {results['curtailment']:.2f}kW"
        )

        return results

    def reset(self) -> None:
        """Reset all component modules."""
        for module in self.pv_modules.values():
            module.reset()
        for module in self.battery_modules.values():
            module.reset()
        for module in self.ev_modules.values():
            module.reset()

        self.logger.debug(f"DER system reset: {self.name}")

    def _parse_component_config(self, plural_key: str, singular_key: str) -> Dict[str, Dict[str, Any]]:
        """Parse component configuration."""
        configs = {}

        if plural_key in self.system_config:
            multi_config = self.system_config[plural_key]

            if isinstance(multi_config, list):
                for idx, comp_config in enumerate(multi_config):
                    comp_id = comp_config.get('id', f"{singular_key}_{idx + 1}")
                    configs[comp_id] = comp_config
            elif isinstance(multi_config, dict):
                for comp_id, comp_config in multi_config.items():
                    if isinstance(comp_config, dict):
                        comp_config['id'] = comp_id
                    configs[comp_id] = comp_config

        elif singular_key in self.system_config:
            single_config = self.system_config[singular_key]
            comp_id = single_config.get('id', f"{singular_key}_1")
            configs[comp_id] = single_config

        return configs

    def get_state(self) -> Dict[str, Any]:
        """Get current state of all component modules."""
        state = {
            'pv_modules': {},
            'battery_modules': {},
            'ev_modules': {},
        }

        for pv_id, pv_module in self.pv_modules.items():
            state['pv_modules'][pv_id] = pv_module.get_state()

        for bat_id, bat_module in self.battery_modules.items():
            state['battery_modules'][bat_id] = bat_module.get_state()

        for ev_id, ev_module in self.ev_modules.items():
            state['ev_modules'][ev_id] = ev_module.get_state()

        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state of all component modules."""
        if 'pv_modules' in state:
            for pv_id, pv_state in state['pv_modules'].items():
                if pv_id in self.pv_modules:
                    self.pv_modules[pv_id].set_state(pv_state)

        if 'battery_modules' in state:
            for bat_id, bat_state in state['battery_modules'].items():
                if bat_id in self.battery_modules:
                    self.battery_modules[bat_id].set_state(bat_state)

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