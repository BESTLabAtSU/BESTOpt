import json
import copy
import logging
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path
from dataclasses import is_dataclass, asdict

Section = Literal["buildings", "disturbances", "controllers", "environment"]


class ConfigurationManager:
    """Manages loading and modifying configuration from JSON."""

    # Categorize into four subgroups
    SECTION_KEYS = ("buildings", "disturbances", "controllers", "environment")

    def __init__(self, json_file_path: Optional[str] = None):
        self.logger = logging.getLogger("ConfigurationManager")
        self.config_path = Path(json_file_path) if json_file_path else None

        if self.config_path and self.config_path.exists():
            self.config = self._load_json()
            self.logger.info(f"Loaded existing configuration from {json_file_path}")
        else:
            self.config = self._create_empty_config()
            if json_file_path:
                self.logger.info(f"JSON file {json_file_path} not found, starting with empty configuration")
            else:
                self.logger.info("Starting with empty configuration (no file specified)")

        # Selected configurations
        self.selected_buildings: Dict[str, Dict[str, Any]] = {}
        self.selected_controllers: Dict[str, Dict[str, Any]] = {}
        self.selected_disturbances: Dict[str, Dict[str, Any]] = {}
        self.selected_environment: Dict[str, Any] = {}
        self.selected_grid: Dict[str, Any] = {}
        # Track active controllers per building
        self.active_controllers: Dict[str, Dict[str, str]] = {}  # building_id -> {domain -> controller_name}
        # Example: {"building_1": {"electrical": "mpc_elec", "thermal": "rule_thermal"}}
        self.logger.info(f"Loaded configuration from {json_file_path}")

    # Building management
    def get_available_buildings(self) -> List[str]:
        """Get list of available building IDs."""
        return list(self.config.get('buildings', {}).keys())

    def get_building_info(self, building_id: str) -> Dict[str, Any]:
        """Get configuration info for a specific building."""
        return self.config.get('buildings', {}).get(building_id, {})

    def select_buildings(self, building_ids: List[str]) -> None:
        """Select specific buildings for the simulation."""
        available_buildings = self.config.get('buildings', {})

        for building_id in building_ids:
            if building_id in available_buildings:
                # Deep copy to avoid modifying original config
                self.selected_buildings[building_id] = self._deep_copy_dict(
                    available_buildings[building_id]
                )
                self.logger.info(f"Selected building: {building_id}")
            else:
                self.logger.warning(f"Building '{building_id}' not found in configuration")

    def select_all_buildings(self) -> None:
        """Select all available buildings."""
        self.select_buildings(self.get_available_buildings())

    def deselect_building(self, building_id: str) -> None:
        """Remove a building from selection."""
        self.selected_buildings.pop(building_id, None)
        if building_id in self.active_controllers:
            for domain in list(self.active_controllers[building_id].keys()):
                controller_key = self.active_controllers[building_id][domain]
                self.selected_controllers.pop(controller_key, None)
            self.active_controllers.pop(building_id, None)
        self.logger.info(f"Deselected building: {building_id}")

    # Component management
    def get_building_components(self, building_id: str, component_type: str) -> List[str]:
        """Get component names of specific type in a building."""
        if building_id not in self.selected_buildings:
            self.logger.warning(f"Building {building_id} not selected")
            return []

        building_config = self.selected_buildings[building_id]
        components = building_config.get(component_type, {})
        return list(components.keys()) if isinstance(components, dict) else []

    def adjust_building_component_parameter(self, building_id: str, component_type: str,
                                            component_id: str, param_name: str, value: Any) -> None:
        """Adjust parameter for a specific component in a building."""
        if building_id not in self.selected_buildings:
            self.logger.warning(f"Building {building_id} not selected")
            return

        building_config = self.selected_buildings[building_id]

        if component_type not in building_config:
            self.logger.warning(f"Component type {component_type} not found in building {building_id}")
            return

        if component_id not in building_config[component_type]:
            self.logger.warning(f"Component {component_id} not found in {building_id}.{component_type}")
            return

        # Check parameters dict exists
        component_config = building_config[component_type][component_id]
        component_config.setdefault('parameters', {})[param_name] = value

        self.logger.info(f"Updated {building_id}.{component_type}.{component_id}.{param_name} = {value}")

    def get_building_component_parameters(self, building_id: str, component_type: str,
                                          component_id: str) -> Dict[str, Any]:
        """Get parameters for a specific component."""
        if building_id not in self.selected_buildings:
            return {}

        building_config = self.selected_buildings[building_id]
        return (building_config.get(component_type, {})
                .get(component_id, {})
                .get('parameters', {}))

    def add_building_component(self, building_id: str, component_type: str, component_id: str,
                               *, parameters: Optional[Dict[str, Any]] = None,
                               class_path: Optional[str] = None,
                               dataclass_obj: Optional[object] = None,
                               overwrite: bool = True) -> None:
        """
        Add a new component to a building.
        Examples:
        1) Add a module from a pre-defined dataclass
        ConfigurationManager.add_entry(
            "modules",
            "battery",
            dataclass_obj=BatteryConfig(),          # from dataclass
            parameters={"battery_capacity": 25.0},  # overrides specific fields
            overwrite=True
        )

        2) Add a brand new module config
        ConfigurationManager.add_entry(
            "modules",
            "custom_module",
            parameters={"param1": 123, "param2": "abc"},
            overwrite=True
        )
        """

        # Ensure building exists in config
        buildings = self.config.setdefault("buildings", {})
        building_config = buildings.setdefault(building_id, {})
        components = building_config.setdefault(component_type, {})

        if not overwrite and component_id in components:
            self.logger.info(f"Component {building_id}.{component_type}.{component_id} exists, skipping")
            return

        # Merge parameters from dataclass and parameters dict
        default_params = self._dataclass_to_dict(dataclass_obj)
        merged_params = self._merge_params(default_params, parameters)

        # Create entry
        entry = {"parameters": merged_params}
        if class_path:
            entry["class_path"] = class_path

        components[component_id] = entry
        self.logger.info(f"Added {building_id}.{component_type}.{component_id}")

    # Controller management
    def get_available_controllers(self) -> List[str]:
        """Get available controller types."""
        return list(self.config.get('controllers', {}).keys())

    def get_controller_info(self, controller_name: str) -> Dict[str, Any]:
        """Get controller configuration info."""
        return self.config.get('controllers', {}).get(controller_name, {})

    def select_controller_for_building_domain(self,
                                              building_id: str,
                                              domain: str,  # "electrical", "thermal", "water"
                                              controller_name: str) -> None:
        """Assign a controller to a specific domain of a building."""
        if building_id not in self.selected_buildings:
            self.logger.warning(f"Building {building_id} not selected")
            return

        if controller_name not in self.config.get('controllers', {}):
            self.logger.warning(f"Controller {controller_name} not available")
            return

        # Initialize building's controller dict
        if building_id not in self.active_controllers:
            self.active_controllers[building_id] = {}

        # Create controller instance for this building-domain pair
        controller_config = self._deep_copy_dict(self.config['controllers'][controller_name])
        controller_key = f"{building_id}_{domain}_{controller_name}"

        self.selected_controllers[controller_key] = controller_config
        self.active_controllers[building_id][domain] = controller_key

        self.logger.info(f"Assigned {controller_name} to {building_id}.{domain}")

    def adjust_building_controller_parameter(self,
                                             building_id: str,
                                             domain: str,
                                             param_name: str,
                                             value: Any) -> None:
        """Adjust controller parameter for a specific building's domain.

        Args:
            building_id: ID of the building
            domain: One of 'electrical', 'thermal', or 'water'
            param_name: Parameter name to adjust
            value: New parameter value
        """
        if building_id not in self.active_controllers:
            self.logger.warning(f"No controllers for building {building_id}")
            return

        if domain not in self.active_controllers[building_id]:
            self.logger.warning(f"No {domain} controller for building {building_id}")
            return

        controller_key = self.active_controllers[building_id][domain]
        if controller_key not in self.selected_controllers:
            self.logger.warning(f"Controller {controller_key} not found")
            return

        self.selected_controllers[controller_key].setdefault('parameters', {})[param_name] = value
        self.logger.info(f"Updated {domain} controller for {building_id}: {param_name} = {value}")

    def add_controller(self, controller_name: str,
                       *, parameters: Optional[Dict[str, Any]] = None,
                       class_path: Optional[str] = None,
                       dataclass_obj: Optional[object] = None,
                       overwrite: bool = True) -> None:
        """Add a new controller type."""

        controllers = self.config.setdefault("controllers", {})

        if not overwrite and controller_name in controllers:
            self.logger.info(f"Controller {controller_name} exists, skipping")
            return

        default_params = self._dataclass_to_dict(dataclass_obj)
        merged_params = self._merge_params(default_params, parameters)

        entry = {"parameters": merged_params}
        if class_path:
            entry["class_path"] = class_path

        controllers[controller_name] = entry
        self.logger.info(f"Added controller {controller_name}")

    # Disturbances management
    def get_available_disturbances(self) -> List[str]:
        return list(self.config.get('disturbances', {}).keys())

    def select_disturbances(self, disturbance_names: List[str]) -> None:
        """Select system-wide disturbances."""
        available_disturbances = self.config.get('disturbances', {})

        for name in disturbance_names:
            if name in available_disturbances:
                self.selected_disturbances[name] = self._deep_copy_dict(
                    available_disturbances[name]
                )
                self.logger.info(f"Selected disturbance: {name}")
            else:
                self.logger.warning(f"Disturbance '{name}' not found")

    def adjust_disturbance_parameter(self, disturbance_name: str, param_name: str, value: Any) -> None:
        if disturbance_name not in self.selected_disturbances:
            self.logger.warning(f"Disturbance {disturbance_name} not selected")
            return

        self.selected_disturbances[disturbance_name].setdefault('parameters', {})[param_name] = value
        self.logger.info(f"Updated disturbance {disturbance_name}.{param_name} = {value}")

    def add_disturbance(self, disturbance_name: str,
                        *, parameters: Optional[Dict[str, Any]] = None,
                        class_path: Optional[str] = None,
                        dataclass_obj: Optional[object] = None,
                        overwrite: bool = True) -> None:
        """Add a new disturbance."""

        disturbances = self.config.setdefault("disturbances", {})

        if not overwrite and disturbance_name in disturbances:
            self.logger.info(f"Disturbance {disturbance_name} exists, skipping")
            return

        default_params = self._dataclass_to_dict(dataclass_obj)
        merged_params = self._merge_params(default_params, parameters)

        entry = {"parameters": merged_params}
        if class_path:
            entry["class_path"] = class_path

        disturbances[disturbance_name] = entry
        self.logger.info(f"Added disturbance {disturbance_name}")

    # Environment management
    def select_environment(self) -> None:
        env_config = self.config.get('environment', {})
        if env_config:
            self.selected_environment = self._deep_copy_dict(env_config)
            self.logger.info("Environment configuration selected")

    def adjust_environment_parameter(self, param_name: str, value: Any) -> None:
        if not self.selected_environment:
            self.logger.warning("Environment not selected")
            return
        self.selected_environment.setdefault('parameters', {})[param_name] = value

    def add_environment(self, *, parameters: Optional[Dict[str, Any]] = None,
                        class_path: Optional[str] = None,
                        dataclass_obj: Optional[object] = None,
                        overwrite: bool = True) -> None:
        """Add environment configuration."""

        if not overwrite and "environment" in self.config:
            self.logger.info("Environment exists, skipping")
            return

        default_params = self._dataclass_to_dict(dataclass_obj)
        merged_params = self._merge_params(default_params, parameters)

        entry = {"parameters": merged_params}
        if class_path:
            entry["class_path"] = class_path

        self.config["environment"] = entry
        self.logger.info("Added environment")

    # Grid management
    # @TODO add grid model later
    def select_grid(self) -> None:
        grid_config = self.config.get('grid', {})
        if grid_config:
            self.selected_grid = self._deep_copy_dict(grid_config)
            self.logger.info("Grid configuration selected")

    def adjust_grid_parameter(self, param_name: str, value: Any) -> None:
        if not self.selected_grid:
            self.logger.warning("Grid not selected")
            return
        self.selected_grid.setdefault('parameters', {})[param_name] = value

    def add_grid(self, *, parameters: Optional[Dict[str, Any]] = None,
                 class_path: Optional[str] = None,
                 dataclass_obj: Optional[object] = None,
                 overwrite: bool = True) -> None:
        """Add grid configuration."""
        if not overwrite and self.config.get("grid"):
            self.logger.info("Grid exists, skipping")
            return

        default_params = self._dataclass_to_dict(dataclass_obj)
        merged_params = self._merge_params(default_params, parameters)

        entry = {"parameters": merged_params}
        if class_path:
            entry["class_path"] = class_path

        self.config["grid"] = entry
        self.logger.info(f"Added grid with {len(merged_params)} parameters")

    # Configuration Generation and Validation
    def get_final_configuration(self) -> Dict[str, Any]:
        """Generate final configuration for the environment."""
        return {
            "buildings": self.selected_buildings,
            "controllers": self.selected_controllers,
            "active_controllers": self.active_controllers,
            "disturbances": self.selected_disturbances,
            "environment": self.selected_environment,
            "grid": self.selected_grid
        }

    def validate_configuration(self) -> List[str]:
        """Validate the current configuration and return warnings."""
        warnings = []

        # Check buildings
        if not self.selected_buildings:
            warnings.append("No buildings selected")

        # Check that each building has required components and controllers
        for building_id, building_config in self.selected_buildings.items():

            # Check for domain controllers
            if building_id not in self.active_controllers:
                warnings.append(f"Building {building_id} has no assigned controllers")
            else:
                building_controllers = self.active_controllers[building_id]

                # Check each required domain has a controller
                required_domains = []  # TODO Add required domains like ['electrical', 'thermal']
                for domain in required_domains:
                    if domain not in building_controllers:
                        warnings.append(f"Building {building_id} missing {domain} controller")

                # Warn if NO controllers at all
                if not building_controllers:
                    warnings.append(f"Building {building_id} has controller mapping but no domains assigned")

            # Check component configurations
            for component_type in ['batteries', 'pv_systems', 'hvac_systems']:
                components = building_config.get(component_type, {})
                for comp_id, comp_config in components.items():
                    if 'class_path' not in comp_config:
                        warnings.append(f"Component {building_id}.{component_type}.{comp_id} missing class_path")

        # Check global components
        if not self.selected_disturbances:
            warnings.append("No disturbances selected")

        if not self.selected_environment:
            warnings.append("Environment not selected")

        return warnings

    def print_summary(self) -> None:
        """Print configuration summary."""
        # Buildings
        print(f"\nSelected Buildings ({len(self.selected_buildings)}):")
        for building_id, building_config in self.selected_buildings.items():
            print(f"\n  Building: {building_id}")

            # Show domain controllers
            if building_id in self.active_controllers:
                building_controllers = self.active_controllers[building_id]
                if building_controllers:
                    print(f"    Controllers:")
                    for domain, controller_key in building_controllers.items():
                        # Extract controller name from key (format: building_domain_name)
                        controller_name = controller_key.replace(f"{building_id}_{domain}_", "")
                        print(f"      - {domain}: {controller_name}")
                else:
                    print(f"    Controllers: None assigned")
            else:
                print(f"    Controllers: None assigned")

            # Show components
            for component_type in ['batteries', 'pv_systems', 'hvac_systems', 'thermal_zones']:
                components = building_config.get(component_type, {})
                if components:
                    print(f"    {component_type}: {list(components.keys())}")

        # Global components
        print(f"\nDisturbances: {list(self.selected_disturbances.keys())}")
        print(f"Environment: {'Selected' if self.selected_environment else 'Not selected'}")
        print(f"Grid: {'Selected' if self.selected_grid else 'Not selected'}")

        # Validation warnings
        warnings = self.validate_configuration()
        if warnings:
            print(f"\n⚠ Warnings ({len(warnings)}):")
            for warning in warnings:
                print(f"  - {warning}")

    # Helper functions
    def _load_json(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(self.config_path, 'r') as f:
            loaded_config = json.load(f)

        # Ensure all required sections exist
        empty_config = self._create_empty_config()
        for section in empty_config:
            if section not in loaded_config:
                loaded_config[section] = empty_config[section]

        return loaded_config

    def save_final_configuration(self, filepath: str) -> None:
        """Save final configuration to JSON file."""
        final_config = self.get_final_configuration()
        with open(filepath, 'w') as f:
            json.dump(final_config, f, indent=2)
        self.logger.info(f"Saved configuration to {filepath}")

    def save_configuration(self, filepath: Optional[str] = None) -> None:
        """Save current configuration to JSON file. Use when creat JSON template"""
        if filepath:
            save_path = Path(filepath)
        elif self.config_path:
            save_path = self.config_path
        else:
            raise ValueError("No filepath specified and no original config path available")

        # Save the base configuration (not selected configuration)
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        self.logger.info(f"Saved configuration to {save_path}")

    def load_saved_configuration(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            saved_config = json.load(f)
        self.selected_buildings = saved_config.get('buildings', {})
        self.selected_disturbances = saved_config.get('disturbances', {})
        self.selected_controllers = saved_config.get('controllers', {})
        self.active_controllers = saved_config.get('controllers', {}).get('_active', None)
        self.selected_environment = saved_config.get('environment', {})
        self.logger.info(f"Loaded saved configuration from {filepath}")

    @staticmethod
    def _deep_copy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy a dictionary."""
        return copy.deepcopy(d)

    @staticmethod
    def _create_empty_config() -> Dict[str, Any]:
        """Create an empty configuration structure."""
        return {
            "buildings": {},
            "controllers": {},
            "disturbances": {},
            "environment": {},
            "grid": {}
        }

    def get_component_count_by_type(self, component_type: str) -> Dict[str, int]:
        """Get count of components by type across all buildings."""
        counts = {}
        for building_id, building_config in self.selected_buildings.items():
            components = building_config.get(component_type, {})
            counts[building_id] = len(components) if isinstance(components, dict) else 0
        return counts

    def get_total_component_count(self, component_type: str) -> int:
        """Get total count of a component type across all buildings."""
        return sum(self.get_component_count_by_type(component_type).values())

    @staticmethod
    def _merge_params(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        result = dict(base or {})
        if override:
            result.update(override)
        return result

    def _section_dict(self, section: Section) -> Dict[str, Any]:
        return self.config.setdefault(section, {})

    def _entry_dict(self, section: Section, name: Optional[str] = None) -> Dict[str, Any]:
        sd = self._section_dict(section)
        if section in ("modules", "disturbances", "controllers"):
            if name is None:
                raise ValueError(f"name must be provided for section '{section}'")
            return sd.setdefault(name, {})
        elif section == "environment":
            return sd
        else:
            raise ValueError(f"Unknown section: {section}")

    def _dataclass_to_dict(self, dataclass_obj: Optional[object]) -> Dict[str, Any]:
        if dataclass_obj is None:
            return {}
        try:
            if is_dataclass(dataclass_obj):
                return asdict(dataclass_obj)
            self.logger.warning("Provided dataclass_obj is not a dataclass; ignored.")
        except Exception as e:
            self.logger.warning(f"Failed to read dataclass: {e}")
        return {}

    # Below are old code for single building
    # Add new configs
    # def add_entry(
    #         self,
    #         section: Section,
    #         name: Optional[str] = None,
    #         *,
    #         parameters: Optional[Dict[str, Any]] = None,
    #         overwrite: bool = True,
    #         dataclass_obj: Optional[object] = None,
    #         class_path: Optional[str] = None,
    # ) -> None:
    #     """
    #     Add or update a config entry in a section.
    #     Examples:
    #     1) Add a module from a pre-defined dataclass
    #     ConfigurationManager.add_entry(
    #         "modules",
    #         "battery",
    #         dataclass_obj=BatteryConfig(),          # from dataclass
    #         parameters={"battery_capacity": 25.0},  # overrides specific fields
    #         overwrite=True
    #     )
    #
    #     2) Add a brand new module config
    #     ConfigurationManager.add_entry(
    #         "modules",
    #         "custom_module",
    #         parameters={"param1": 123, "param2": "abc"},
    #         overwrite=True
    #     )
    #
    #     3) Add a disturbance from dataclass
    #     ConfigurationManager.add_entry(
    #         "disturbances",
    #         "weather",
    #         dataclass_obj=WeatherConfig(),
    #         parameters={"xxx": "xxx"},
    #         class_path="xxx"
    #     )
    #
    #     4) Add a controller with specific parameters
    #     ConfigurationManager.add_entry(
    #         "controllers",
    #         "mpc",
    #         parameters={"horizon": 24, "dt_minutes": 15}, # Just for reference
    #         class_path="bestopt.controllers.mpc.MPCController"
    #     )
    #     # Then set it active at runtime:
    #     cm.set_active_controller("mpc")
    #
    #     5) Add environment configuration
    #     ConfigurationManager.add_entry(
    #         "environment",
    #         parameters={"dt_minutes": 15, "sim_steps": 576},
    #         class_path="bestopt.env.Environment"
    #     )
    #
    #     """
    #     if not overwrite and ((section == "environment" and "parameters" in self._section_dict(section))
    #                           or (section != "environment" and name in self._section_dict(section))):
    #         self.logger.info(f"{section} '{name or '<env>'}' overwrite=False; skipping.")
    #         return
    #
    #     default_params = self._dataclass_to_dict(dataclass_obj)
    #     merged_params = self._merge_params(default_params, parameters)
    #
    #     if section == "environment":
    #         entry = {"parameters": merged_params}
    #         if class_path: entry["class_path"] = class_path
    #         self.config["environment"] = entry
    #     else:
    #         entry = {"parameters": merged_params}
    #         if class_path: entry["class_path"] = class_path
    #         self._section_dict(section)[name] = entry
    #
    #     self.logger.info(f"Added/updated {section} '{name or 'environment'}' with {len(merged_params)} params.")
    #
    # # Modules
    # def get_available_modules(self) -> List[str]:
    #     return list(self.config.get('modules', {}).keys())
    #
    # def get_module_info(self, module_name: str) -> Dict[str, Any]:
    #     return self.config.get('modules', {}).get(module_name, {})
    #
    # def select_modules(self, module_names: List[str]) -> None:
    #     for name in module_names:
    #         if name in self.config.get('modules', {}):
    #             self.selected_modules[name] = self.config['modules'][name].copy()
    #             self.logger.info(f"Selected module: {name}")
    #         else:
    #             self.logger.warning(f"Module '{name}' not found")
    #
    # def select_all_modules(self) -> None:
    #     self.select_modules(self.get_available_modules())
    #
    # def deselect_module(self, module_name: str) -> None:
    #     self.selected_modules.pop(module_name, None)
    #     self.logger.info(f"Deselected module: {module_name}")
    #
    # def adjust_module_parameter(self, module_name: str, param_name: str, value: Any) -> None:
    #     if module_name not in self.selected_modules:
    #         self.logger.warning(f"Module '{module_name}' not selected. Select it first.")
    #         return
    #     self.selected_modules[module_name].setdefault('parameters', {})[param_name] = value
    #     self.logger.info(f"Updated {module_name}.{param_name} = {value}")
    #
    # def get_module_parameters(self, module_name: str) -> Dict[str, Any]:
    #     return self.selected_modules.get(module_name, {}).get('parameters', {})
    #
    # # Disturbances
    # def get_available_disturbances(self) -> List[str]:
    #     return list(self.config.get('disturbances', {}).keys())
    #
    # def get_disturbance_info(self, name: str) -> Dict[str, Any]:
    #     return self.config.get('disturbances', {}).get(name, {})
    #
    # def select_disturbances(self, names: List[str]) -> None:
    #     for name in names:
    #         if name in self.config.get('disturbances', {}):
    #             self.selected_disturbances[name] = self.config['disturbances'][name].copy()
    #             self.logger.info(f"Selected disturbance: {name}")
    #         else:
    #             self.logger.warning(f"Disturbance '{name}' not found")
    #
    # def deselect_disturbance(self, name: str) -> None:
    #     self.selected_disturbances.pop(name, None)
    #     self.logger.info(f"Deselected disturbance: {name}")
    #
    # def adjust_disturbance_parameter(self, name: str, param_name: str, value: Any) -> None:
    #     if name not in self.selected_disturbances:
    #         self.logger.warning(f"Disturbance '{name}' not selected. Select it first.")
    #         return
    #     self.selected_disturbances[name].setdefault('parameters', {})[param_name] = value
    #     self.logger.info(f"Updated disturbance {name}.{param_name} = {value}")
    #
    # # Controllers
    # def get_available_controllers(self) -> List[str]:
    #     # ignore the special "active" key if present
    #     return [k for k in self.config.get('controllers', {}).keys() if k != "active"]
    #
    # def get_controller_info(self, name: str) -> Dict[str, Any]:
    #     return self.config.get('controllers', {}).get(name, {})
    #
    # def set_active_controller(self, name: str) -> None:
    #     if name not in self.config.get('controllers', {}):
    #         self.logger.warning(f"Controller '{name}' not found")
    #         return
    #     self.active_controller_name = name
    #     self.selected_controller = self.config['controllers'][name].copy()
    #     self.config.setdefault('controllers', {})['active'] = name
    #     self.logger.info(f"Active controller set to: {name}")
    #
    # def adjust_controller_parameter(self, param_name: str, value: Any) -> None:
    #     if not self.selected_controller:
    #         self.logger.warning("No active controller selected.")
    #         return
    #     self.selected_controller.setdefault('parameters', {})[param_name] = value
    #     self.logger.info(f"Updated controller.{param_name} = {value}")
    #
    # # Environment
    # def get_environment_info(self) -> Dict[str, Any]:
    #     return self.config.get('environment', {})
    #
    # def select_environment(self) -> None:
    #     env = self.config.get('environment', {})
    #     if not env:
    #         self.logger.warning("No environment config found.")
    #         return
    #     self.selected_environment = env.copy()
    #     self.logger.info("Environment selected.")
    #
    # def adjust_environment_parameter(self, param_name: str, value: Any) -> None:
    #     if not self.selected_environment:
    #         self.logger.warning("Environment not selected. Call select_environment() first.")
    #         return
    #     self.selected_environment.setdefault('parameters', {})[param_name] = value
    #     self.logger.info(f"Updated environment.{param_name} = {value}")
    #
    # # Finalize / Validate / Summary
    # def get_final_configuration(self) -> Dict[str, Any]:
    #     return {
    #         "modules": self.selected_modules,
    #         "disturbances": self.selected_disturbances,
    #         "controller": {
    #             "_active": self.active_controller_name,
    #             **(self.selected_controller or {})
    #         } if self.selected_controller else {},
    #         "environment": self.selected_environment,
    #     }
    #
    # def validate_configuration(self) -> List[str]:
    #     warnings: List[str] = []
    #
    #     if not self.selected_modules:
    #         warnings.append("No modules selected")
    #     if not self.selected_disturbances:
    #         warnings.append("No disturbances selected")
    #     if not self.selected_controller:
    #         warnings.append("No active controller selected")
    #     if not self.selected_environment:
    #         warnings.append("Environment not selected")
    #
    #     # Module checks
    #     for module_name, module_config in self.selected_modules.items():
    #         if 'class_path' not in module_config:
    #             warnings.append(f"Module '{module_name}' missing class_path")
    #         params = module_config.get('parameters', {})
    #         if module_name == 'battery':
    #             if params.get('battery_capacity', 0) <= 0:
    #                 warnings.append("Battery capacity must be positive")
    #             if params.get('soc_min', 0) >= params.get('soc_max', 1):
    #                 warnings.append("Battery soc_min must be less than soc_max")
    #         if module_name == 'pv':
    #             if params.get('pv_capacity', 0) <= 0:
    #                 warnings.append("PV capacity must be positive")
    #
    #     # Disturbance checks
    #     for name, dcfg in self.selected_disturbances.items():
    #         if 'class_path' not in dcfg:
    #             warnings.append(f"Disturbance '{name}' missing class_path")
    #
    #     # Controller checks
    #     if self.selected_controller and 'class_path' not in self.selected_controller:
    #         warnings.append("Active controller missing class_path")
    #
    #     # Environment checks
    #     if self.selected_environment and 'class_path' not in self.selected_environment:
    #         warnings.append("Environment missing class_path")
    #
    #     # @TODO add later
    #
    #     return warnings
    #
    # def print_summary(self) -> None:
    #     print("CONFIGURATION SUMMARY")
    #
    #     # Modules
    #     for module_name, module_config in self.selected_modules.items():
    #         print(f"\n  [Module] {module_name}:")
    #         for k, v in module_config.get('parameters', {}).items():
    #             print(f"    - {k}: {v}")
    #
    #     # Disturbances
    #     for name, dcfg in self.selected_disturbances.items():
    #         print(f"\n  [Disturbance] {name}:")
    #         for k, v in dcfg.get('parameters', {}).items():
    #             print(f"    - {k}: {v}")
    #
    #     # Controller
    #     if self.selected_controller:
    #         print(f"\n  [Controller] active = {self.active_controller_name}")
    #         for k, v in self.selected_controller.get('parameters', {}).items():
    #             print(f"    - {k}: {v}")
    #
    #     # Environment
    #     if self.selected_environment:
    #         print("\n  [Environment]:")
    #         for k, v in self.selected_environment.get('parameters', {}).items():
    #             print(f"    - {k}: {v}")
    #
    #     # Validation
    #     warnings = self.validate_configuration()
    #     if warnings:
    #         print("\nWarnings:")
    #         for w in warnings:
    #             print(f"  ⚠ {w}")
