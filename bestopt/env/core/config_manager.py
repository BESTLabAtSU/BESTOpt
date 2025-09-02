import json
import logging
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path
from dataclasses import is_dataclass, asdict

Section = Literal["modules", "disturbances", "controllers", "environment"]


class ConfigurationManager:
    """Manages loading and modifying configuration from JSON."""

    # Categorize into four subgroups
    SECTION_KEYS = ("modules", "disturbances", "controllers", "environment")

    def __init__(self, json_file_path: str):
        self.logger = logging.getLogger("ConfigurationManager")
        self.config_path = Path(json_file_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_file_path}")

        self.config = self._load_json()
        self.selected_modules: Dict[str, Any] = {}
        self.selected_disturbances: Dict[str, Any] = {}
        self.active_controller_name: Optional[str] = None
        self.selected_controller: Dict[str, Any] = {}
        self.selected_environment: Dict[str, Any] = {}

        self.logger.info(f"Loaded configuration from {json_file_path}")

    # IO functions
    def _load_json(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def save_configuration(self, filepath: str) -> None:
        final_config = self.get_final_configuration()
        with open(filepath, 'w') as f:
            json.dump(final_config, f, indent=2)
        self.logger.info(f"Saved configuration to {filepath}")

    def load_saved_configuration(self, filepath: str) -> None:
        with open(filepath, 'r') as f:
            saved_config = json.load(f)
        self.selected_modules = saved_config.get('modules', {})
        self.selected_disturbances = saved_config.get('disturbances', {})
        self.selected_controller = saved_config.get('controller', {})
        self.active_controller_name = saved_config.get('controller', {}).get('_active', None)
        self.selected_environment = saved_config.get('environment', {})
        self.logger.info(f"Loaded saved configuration from {filepath}")

    # Helper functions
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

    # Add new configs
    def add_entry(
        self,
        section: Section,
        name: Optional[str] = None,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        overwrite: bool = True,
        dataclass_obj: Optional[object] = None,
        class_path: Optional[str] = None,
    ) -> None:
        """
        Add or update a config entry in a section.
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

        3) Add a disturbance from dataclass
        ConfigurationManager.add_entry(
            "disturbances",
            "weather",
            dataclass_obj=WeatherConfig(),
            parameters={"xxx": "xxx"},
            class_path="xxx"
        )

        4) Add a controller with specific parameters
        ConfigurationManager.add_entry(
            "controllers",
            "mpc",
            parameters={"horizon": 24, "dt_minutes": 15}, # Just for reference
            class_path="bestopt.controllers.mpc.MPCController"
        )
        # Then set it active at runtime:
        cm.set_active_controller("mpc")

        5) Add environment configuration
        ConfigurationManager.add_entry(
            "environment",
            parameters={"dt_minutes": 15, "sim_steps": 576},
            class_path="bestopt.env.Environment"
        )

        """
        if not overwrite and ((section == "environment" and "parameters" in self._section_dict(section))
                              or (section != "environment" and name in self._section_dict(section))):
            self.logger.info(f"{section} '{name or '<env>'}' overwrite=False; skipping.")
            return

        default_params = self._dataclass_to_dict(dataclass_obj)
        merged_params = self._merge_params(default_params, parameters)

        if section == "environment":
            entry = {"parameters": merged_params}
            if class_path: entry["class_path"] = class_path
            self.config["environment"] = entry
        else:
            entry = {"parameters": merged_params}
            if class_path: entry["class_path"] = class_path
            self._section_dict(section)[name] = entry

        self.logger.info(f"Added/updated {section} '{name or 'environment'}' with {len(merged_params)} params.")

    # Modules
    def get_available_modules(self) -> List[str]:
        return list(self.config.get('modules', {}).keys())

    def get_module_info(self, module_name: str) -> Dict[str, Any]:
        return self.config.get('modules', {}).get(module_name, {})

    def select_modules(self, module_names: List[str]) -> None:
        for name in module_names:
            if name in self.config.get('modules', {}):
                self.selected_modules[name] = self.config['modules'][name].copy()
                self.logger.info(f"Selected module: {name}")
            else:
                self.logger.warning(f"Module '{name}' not found")

    def select_all_modules(self) -> None:
        self.select_modules(self.get_available_modules())

    def deselect_module(self, module_name: str) -> None:
        self.selected_modules.pop(module_name, None)
        self.logger.info(f"Deselected module: {module_name}")

    def adjust_module_parameter(self, module_name: str, param_name: str, value: Any) -> None:
        if module_name not in self.selected_modules:
            self.logger.warning(f"Module '{module_name}' not selected. Select it first.")
            return
        self.selected_modules[module_name].setdefault('parameters', {})[param_name] = value
        self.logger.info(f"Updated {module_name}.{param_name} = {value}")

    def get_module_parameters(self, module_name: str) -> Dict[str, Any]:
        return self.selected_modules.get(module_name, {}).get('parameters', {})

    # Disturbances
    def get_available_disturbances(self) -> List[str]:
        return list(self.config.get('disturbances', {}).keys())

    def get_disturbance_info(self, name: str) -> Dict[str, Any]:
        return self.config.get('disturbances', {}).get(name, {})

    def select_disturbances(self, names: List[str]) -> None:
        for name in names:
            if name in self.config.get('disturbances', {}):
                self.selected_disturbances[name] = self.config['disturbances'][name].copy()
                self.logger.info(f"Selected disturbance: {name}")
            else:
                self.logger.warning(f"Disturbance '{name}' not found")

    def deselect_disturbance(self, name: str) -> None:
        self.selected_disturbances.pop(name, None)
        self.logger.info(f"Deselected disturbance: {name}")

    def adjust_disturbance_parameter(self, name: str, param_name: str, value: Any) -> None:
        if name not in self.selected_disturbances:
            self.logger.warning(f"Disturbance '{name}' not selected. Select it first.")
            return
        self.selected_disturbances[name].setdefault('parameters', {})[param_name] = value
        self.logger.info(f"Updated disturbance {name}.{param_name} = {value}")

    # Controllers
    def get_available_controllers(self) -> List[str]:
        # ignore the special "active" key if present
        return [k for k in self.config.get('controllers', {}).keys() if k != "active"]

    def get_controller_info(self, name: str) -> Dict[str, Any]:
        return self.config.get('controllers', {}).get(name, {})

    def set_active_controller(self, name: str) -> None:
        if name not in self.config.get('controllers', {}):
            self.logger.warning(f"Controller '{name}' not found")
            return
        self.active_controller_name = name
        self.selected_controller = self.config['controllers'][name].copy()
        self.config.setdefault('controllers', {})['active'] = name
        self.logger.info(f"Active controller set to: {name}")

    def adjust_controller_parameter(self, param_name: str, value: Any) -> None:
        if not self.selected_controller:
            self.logger.warning("No active controller selected.")
            return
        self.selected_controller.setdefault('parameters', {})[param_name] = value
        self.logger.info(f"Updated controller.{param_name} = {value}")

    # Environment
    def get_environment_info(self) -> Dict[str, Any]:
        return self.config.get('environment', {})

    def select_environment(self) -> None:
        env = self.config.get('environment', {})
        if not env:
            self.logger.warning("No environment config found.")
            return
        self.selected_environment = env.copy()
        self.logger.info("Environment selected.")

    def adjust_environment_parameter(self, param_name: str, value: Any) -> None:
        if not self.selected_environment:
            self.logger.warning("Environment not selected. Call select_environment() first.")
            return
        self.selected_environment.setdefault('parameters', {})[param_name] = value
        self.logger.info(f"Updated environment.{param_name} = {value}")

    # Finalize / Validate / Summary
    def get_final_configuration(self) -> Dict[str, Any]:
        return {
            "modules": self.selected_modules,
            "disturbances": self.selected_disturbances,
            "controller": {
                "_active": self.active_controller_name,
                **(self.selected_controller or {})
            } if self.selected_controller else {},
            "environment": self.selected_environment,
        }

    def validate_configuration(self) -> List[str]:
        warnings: List[str] = []

        if not self.selected_modules:
            warnings.append("No modules selected")
        if not self.selected_disturbances:
            warnings.append("No disturbances selected")
        if not self.selected_controller:
            warnings.append("No active controller selected")
        if not self.selected_environment:
            warnings.append("Environment not selected")

        # Module checks
        for module_name, module_config in self.selected_modules.items():
            if 'class_path' not in module_config:
                warnings.append(f"Module '{module_name}' missing class_path")
            params = module_config.get('parameters', {})
            if module_name == 'battery':
                if params.get('battery_capacity', 0) <= 0:
                    warnings.append("Battery capacity must be positive")
                if params.get('soc_min', 0) >= params.get('soc_max', 1):
                    warnings.append("Battery soc_min must be less than soc_max")
            if module_name == 'pv':
                if params.get('pv_capacity', 0) <= 0:
                    warnings.append("PV capacity must be positive")

        # Disturbance checks
        for name, dcfg in self.selected_disturbances.items():
            if 'class_path' not in dcfg:
                warnings.append(f"Disturbance '{name}' missing class_path")

        # Controller checks
        if self.selected_controller and 'class_path' not in self.selected_controller:
            warnings.append("Active controller missing class_path")

        # Environment checks
        if self.selected_environment and 'class_path' not in self.selected_environment:
            warnings.append("Environment missing class_path")

        # @TODO add later

        return warnings

    def print_summary(self) -> None:
        print("CONFIGURATION SUMMARY")

        # Modules
        for module_name, module_config in self.selected_modules.items():
            print(f"\n  [Module] {module_name}:")
            for k, v in module_config.get('parameters', {}).items():
                print(f"    - {k}: {v}")

        # Disturbances
        for name, dcfg in self.selected_disturbances.items():
            print(f"\n  [Disturbance] {name}:")
            for k, v in dcfg.get('parameters', {}).items():
                print(f"    - {k}: {v}")

        # Controller
        if self.selected_controller:
            print(f"\n  [Controller] active = {self.active_controller_name}")
            for k, v in self.selected_controller.get('parameters', {}).items():
                print(f"    - {k}: {v}")

        # Environment
        if self.selected_environment:
            print("\n  [Environment]:")
            for k, v in self.selected_environment.get('parameters', {}).items():
                print(f"    - {k}: {v}")

        # Validation
        warnings = self.validate_configuration()
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  ⚠ {w}")
