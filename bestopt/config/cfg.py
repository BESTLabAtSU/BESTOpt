from dataclasses import asdict, dataclass
from typing import Dict, Any, Optional, Type
from ..env.core.data_structure import SimulationConfig


# Helper function
def _cfg(defaults: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    helper function to update config file
    :param defaults: default config info
    :param overrides: customized config info
    :return: updated config info
    """
    d = dict(defaults)
    if overrides:
        d.update(overrides)
    return d


def _validate_config(cfg: Dict[str, Any]) -> None:
    """
    helper function to validate config file
    :param cfg:
    :return:
    """
    # @TODO below is an example for what we can validate, should add more later
    assert cfg["env"]["timestep"] > 0
    assert cfg["env"]["simulation_days"] > 0
    assert 0.0 <= cfg["battery"]["soc_min"] < cfg["battery"]["soc_max"] <= 1.0
    assert 0.0 <= cfg["ev"]["soc_min"] < cfg["ev"]["soc_max"] <= 1.0


def configuration(
        *,
        env: Optional[Dict[str, Any]] = None,
        building: Optional[Dict[str, Any]] = None,
        battery: Optional[Dict[str, Any]] = None,
        hvac: Optional[Dict[str, Any]] = None,
        tes: Optional[Dict[str, Any]] = None,
        ev: Optional[Dict[str, Any]] = None,
        pv: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create configuration dict for the runtime environment.
    Each module can be overridden independently.
    """
    base = SimulationConfig()
    defaults = asdict(base)

    cfg = {
        "env": _cfg(defaults["env"], env),
        "building": _cfg(defaults["building"], building),
        "hvac": _cfg(defaults["hvac"], hvac),
        "battery": _cfg(defaults["battery"], battery),
        "tes": _cfg(defaults["tes"], tes),
        "ev": _cfg(defaults["ev"], ev),
        "pv": _cfg(defaults["pv"], pv),
    }

    _validate_config(cfg)
    return cfg


def get_component_config(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    """
    Return the config info for a component, e.g., get_component_config(cfg, "battery").
    """
    return dict(cfg[section])


@dataclass
class ModuleSpec:
    section: str  # which section of cfg to use (e.g., "battery", "pv")
    cls: Type[Any]  # module class
    expects_dict: bool = True  # True: __init__(config_dict); False: __init__(**kwargs)


MODULE_REGISTRY: Dict[str, ModuleSpec] = {}


def register_module(name: str, section: str, cls: Type[Any], *, expects_dict: bool = True) -> None:
    """
    Register a module.
    - section: which config section to pass to the module (e.g., "battery")
    - expects_dict: False if the module's __init__ takes **kwargs instead of a single dict
    """
    MODULE_REGISTRY[name] = ModuleSpec(section=section, cls=cls, expects_dict=expects_dict)


def unregister_module(name: str) -> None:
    """Remove a module from the registry."""
    MODULE_REGISTRY.pop(name, None)


def list_registered_modules() -> Dict[str, ModuleSpec]:
    """Introspect what's currently registered."""
    return dict(MODULE_REGISTRY)


def build_modules(
        cfg: Dict[str, Any],
        *,
        enabled: Optional[list[str]] = None,
        constructor_extras: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Instantiate modules from cfg.

    Args:
        cfg: config info returned by configuration()
        enabled: optional list of module names to build. Defaults to all registered.
        constructor_extras: optional extra kwargs per module name. If a module
            expects a single dict, extras are merged into that dict before passing.
            If it expects **kwargs, extras are merged into the **kwargs.

    Returns:
        dict: {name: module_instance}
    """
    names = enabled if enabled is not None else list(MODULE_REGISTRY.keys())
    constructor_extras = constructor_extras or {}

    instances: Dict[str, Any] = {}
    for name in names:
        if name not in MODULE_REGISTRY:
            raise KeyError(f"Module '{name}' is not registered.")
        spec = MODULE_REGISTRY[name]

        # pull section config
        section_cfg = dict(cfg.get(spec.section, {}))
        extra = constructor_extras.get(name, {})
        payload = {**section_cfg, **extra}

        # instantiate
        if spec.expects_dict:
            instances[name] = spec.cls(payload)
        else:
            instances[name] = spec.cls(**payload)

    return instances
