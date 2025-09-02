"""
Configuration manager for bestopt framework.
Handles loading JSON configuration and user customization.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import is_dataclass, asdict


class ConfigurationManager:
    """Manages loading and modifying configuration from JSON."""

    def __init__(self, json_file_path: str):
        """Load configuration from JSON file.

        Args:
            json_file_path: Path to JSON configuration file
        """
        self.logger = logging.getLogger("ConfigurationManager")
        self.config_path = Path(json_file_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_file_path}")

        self.config = self._load_json()
        self.selected_modules = {}

        self.logger.info(f"Loaded configuration from {json_file_path}")

    def _load_json(self) -> Dict[str, Any]:
        """Load JSON configuration file."""
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def get_available_modules(self) -> List[str]:
        """Get list of all available modules from config.

        Returns:
            List of module names
        """
        return list(self.config.get('modules', {}).keys())

    def get_module_info(self, module_name: str) -> Dict[str, Any]:
        """Get information about a specific module.

        Args:
            module_name: Name of the module

        Returns:
            Module configuration and parameters
        """
        return self.config.get('modules', {}).get(module_name, {})

    def select_modules(self, module_names: List[str]) -> None:
        """Select which modules to use.

        Args:
            module_names: List of module names to enable
        """
        for name in module_names:
            if name in self.config.get('modules', {}):
                # Copy the module config
                self.selected_modules[name] = self.config['modules'][name].copy()
                self.logger.info(f"Selected module: {name}")
            else:
                self.logger.warning(f"Module '{name}' not found in configuration")

    def select_all_modules(self) -> None:
        """Select all available modules."""
        all_modules = self.get_available_modules()
        self.select_modules(all_modules)
        self.logger.info("Selected all available modules")

    def deselect_module(self, module_name: str) -> None:
        """Remove a module from selection.

        Args:
            module_name: Name of the module to remove
        """
        if module_name in self.selected_modules:
            del self.selected_modules[module_name]
            self.logger.info(f"Deselected module: {module_name}")

    def adjust_module_parameter(self, module_name: str, param_name: str, value: Any) -> None:
        """Adjust a parameter for a selected module.

        Args:
            module_name: Name of the module
            param_name: Parameter name to adjust
            value: New value for the parameter
        """
        if module_name not in self.selected_modules:
            self.logger.warning(f"Module '{module_name}' not selected. Select it first.")
            return

        if 'parameters' not in self.selected_modules[module_name]:
            self.selected_modules[module_name]['parameters'] = {}

        self.selected_modules[module_name]['parameters'][param_name] = value
        self.logger.info(f"Updated {module_name}.{param_name} = {value}")

    def get_selected_modules(self) -> List[str]:
        """Get list of currently selected modules.

        Returns:
            List of selected module names
        """
        return list(self.selected_modules.keys())

    def add_module(self,
                   name: str,
                   *,
                   parameters: Optional[Dict[str, Any]] = None,
                   overwrite: bool = True,
                   dataclass_obj: Optional[object] = None,
                   class_path: str = None,
                   ) -> None:
        """
        Add (or update) a module entry under self.config['modules'].

            Args:
                name: Module name, e.g., 'battery', 'pv', 'hvac'
                parameters: Optional dict to override defaults
                overwrite: If False and the module exists, do nothing
                dataclass_obj: Optional dataclass instance to seed defaults from

            Examples:
                1) add from pre-defined data class
                ConfigurationManager.add_module(
                "battery",
                dataclass_obj=BatteryConfig(),          # seeds defaults from your dataclass
                parameters={"battery_capacity": 25.0},  # overrides specific fields
                overwrite=True
                )

                2) add a brand new config
                ConfigurationManager.add_module(
                "custom_module",
                parameters={"param1": 123, "param2": "abc"},
                overwrite=True
                )
        """
        mods = self.config.setdefault("modules", {})
        if not overwrite:
            self.logger.info(f"Module '{name}' overwrite=False; skipping.")
            return

        # start with defaults from dataclass if provided
        default_params: Dict[str, Any] = {}
        if dataclass_obj is not None:
            try:
                if is_dataclass(dataclass_obj):
                    default_params = asdict(dataclass_obj)
                else:
                    self.logger.warning(f"dataclass_obj for '{name}' is not a dataclass; ignored.")
            except Exception as e:
                self.logger.warning(f"Failed to read dataclass for '{name}': {e}")

        merged_params = dict(default_params)
        if parameters:
            merged_params.update(parameters)

        module_entry: Dict[str, Any] = {"parameters": merged_params}
        if class_path is not None:
            module_entry["class_path"] = class_path
        mods[name] = module_entry
        self.logger.info(f"Added/updated module '{name}' with {len(merged_params)} params.")

    @staticmethod
    def _merge_params(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """help function: override keys replace base."""
        result = dict(base or {})
        if override:
            result.update(override)
        return result

    def get_module_parameters(self, module_name: str) -> Dict[str, Any]:
        """Get parameters for a specific module.

        Args:
            module_name: Name of the module

        Returns:
            Module parameters dictionary
        """
        if module_name in self.selected_modules:
            return self.selected_modules[module_name].get('parameters', {})
        return {}

    def get_final_configuration(self) -> Dict[str, Any]:
        """Get the final configuration with selected modules and adjustments.

        Returns:
            Final configuration dictionary
        """
        final_config = {
            'modules': self.selected_modules,
            # @TODO
            }
        return final_config

    def save_configuration(self, filepath: str) -> None:
        """Save the current configuration to a JSON file.

        Args:
            filepath: Path to save the configuration
        """
        final_config = self.get_final_configuration()

        with open(filepath, 'w') as f:
            json.dump(final_config, f, indent=2)

        self.logger.info(f"Saved configuration to {filepath}")

    def load_saved_configuration(self, filepath: str) -> None:
        """Load a previously saved configuration.

        Args:
            filepath: Path to the saved configuration
        """
        with open(filepath, 'r') as f:
            saved_config = json.load(f)

        self.selected_modules = saved_config.get('modules', {})
        self.logger.info(f"Loaded saved configuration from {filepath}")

    def validate_configuration(self) -> List[str]:
        """Validate the current configuration.

        Returns:
            List of validation warnings/errors
        """
        warnings = []

        # Check if any modules are selected
        if not self.selected_modules:
            warnings.append("No modules selected")

        # Check module dependencies
        for module_name, module_config in self.selected_modules.items():
            # Check if class_path exists
            if 'class_path' not in module_config:
                warnings.append(f"Module '{module_name}' missing class_path")

            # Check parameters
            params = module_config.get('parameters', {})

            # Module-specific validation
            if module_name == 'battery':
                if params.get('battery_capacity', 0) <= 0:
                    warnings.append(f"Battery capacity must be positive")
                if params.get('soc_min', 0) >= params.get('soc_max', 1):
                    warnings.append(f"Battery soc_min must be less than soc_max")

            elif module_name == 'pv':
                if params.get('pv_capacity', 0) <= 0:
                    warnings.append(f"PV capacity must be positive")

            # @TODO Add more validation as needed

        return warnings

    def print_summary(self) -> None:
        print("CONFIGURATION SUMMARY")

        for module_name, module_config in self.selected_modules.items():
            print(f"\n  {module_name}:")
            params = module_config.get('parameters', {})
            if params:
                for param, value in params.items():
                    print(f"    - {param}: {value}")
            else:
                print("    - (default parameters)")

        # Validation
        warnings = self.validate_configuration()
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  ⚠ {warning}")
