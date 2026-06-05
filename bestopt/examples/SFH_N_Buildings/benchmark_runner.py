"""
Benchmark BESTOpt platform across cluster sizes.

For each N in BUILDING_COUNTS, this script measures:
  * Config-generation time
  * Environment-initialization time (incl. ModNN encoder warmup)
  * Simulation-loop time
  * Per-step wall time (mean / max / min / std)
  * CPU usage (process + system)
  * RAM (process RSS + system)
  * GPU memory & utilization (if NVIDIA GPU available)

Results are written incrementally to
    benchmark_results/benchmark_results.json
so a crash on N=200 does NOT wipe results for N=1..100. Rerunning skips
sizes that have already completed successfully. To re-run a size, delete its
entry from the JSON or delete the whole file.

Outputs:
    benchmark_results/benchmark_results.json   # full structured results
    benchmark_results/benchmark_summary.csv    # flat table for plotting
    benchmark_results/benchmark_report.md      # human-readable summary
    benchmark_results/config_benchmark_*.json  # per-size BESTOpt configs

Right-click → Run.  Adjust BUILDING_COUNTS / MAX_STEPS / SAMPLE_INTERVAL_S
at the bottom of the file under `if __name__ == "__main__":`.
"""

import os
import sys
import json
import time
import gc
import csv
import traceback
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psutil

# ---------- optional GPU monitoring ----------
GPU_AVAILABLE = False
NUM_GPUS = 0
try:
    import pynvml  # type: ignore
    pynvml.nvmlInit()
    NUM_GPUS = pynvml.nvmlDeviceGetCount()
    GPU_AVAILABLE = NUM_GPUS > 0
except Exception:
    pynvml = None  # type: ignore

# ---------- optional torch ----------
TORCH_AVAILABLE = False
try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore


# =============================================================================
# Paths & logging
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT_PATH follows the same convention used in your setup/run scripts:
# script lives in <root>/examples/SFH_N_Buildings/, go up two levels.
PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(SCRIPT_DIR))

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "benchmark_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_JSON = os.path.join(OUTPUT_DIR, "benchmark_results.json")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "benchmark_summary.csv")
RESULTS_MD = os.path.join(OUTPUT_DIR, "benchmark_report.md")
RUN_LOG = os.path.join(OUTPUT_DIR, "benchmark.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(RUN_LOG, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("benchmark")


# =============================================================================
# Resource monitor (background sampler)
# =============================================================================

class ResourceMonitor:
    """Samples CPU / RAM / GPU on a background thread; returns summary on stop()."""

    def __init__(self, interval: float = 0.5, gpu_index: int = 0, label: str = ""):
        self.interval = interval
        self.gpu_index = gpu_index
        self.label = label
        self.samples: List[Dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process(os.getpid())
        # Prime cpu_percent counters (first call returns 0.0)
        self._process.cpu_percent(None)
        psutil.cpu_percent(None)
        self._gpu_handle = None
        if GPU_AVAILABLE:
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                self._gpu_handle = None

    def start(self) -> None:
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        return self.summary()

    def _take_sample(self) -> Dict[str, float]:
        sample: Dict[str, float] = {"ts": time.time()}
        # CPU
        try:
            sample["cpu_percent_process"] = self._process.cpu_percent(None)
        except Exception:
            sample["cpu_percent_process"] = float("nan")
        try:
            sample["cpu_percent_system"] = psutil.cpu_percent(None)
        except Exception:
            sample["cpu_percent_system"] = float("nan")
        # RAM
        try:
            mi = self._process.memory_info()
            sample["ram_mb_process_rss"] = mi.rss / (1024.0 * 1024.0)
        except Exception:
            sample["ram_mb_process_rss"] = float("nan")
        try:
            vm = psutil.virtual_memory()
            sample["ram_mb_system_used"] = vm.used / (1024.0 * 1024.0)
            sample["ram_percent_system"] = vm.percent
        except Exception:
            sample["ram_mb_system_used"] = float("nan")
            sample["ram_percent_system"] = float("nan")
        # GPU
        if self._gpu_handle is not None:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                sample["gpu_mem_used_mb"] = mem.used / (1024.0 * 1024.0)
                sample["gpu_mem_total_mb"] = mem.total / (1024.0 * 1024.0)
                sample["gpu_util_percent"] = float(util.gpu)
                sample["gpu_mem_util_percent"] = float(util.memory)
            except Exception:
                pass
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                sample["gpu_mem_used_mb"] = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
                sample["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
            except Exception:
                pass
        return sample

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self._take_sample())
            except Exception:
                pass
            self._stop.wait(self.interval)

    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {"num_samples": 0}
        keys = set()
        for s in self.samples:
            keys.update(s.keys())
        keys.discard("ts")
        summary: Dict[str, Any] = {"num_samples": len(self.samples)}
        for k in sorted(keys):
            vals = [s[k] for s in self.samples if k in s and not _isnan(s[k])]
            if not vals:
                continue
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_max"] = float(np.max(vals))
            summary[f"{k}_min"] = float(np.min(vals))
        try:
            summary["duration_s"] = float(self.samples[-1]["ts"] - self.samples[0]["ts"])
        except Exception:
            pass
        return summary


def _isnan(x: Any) -> bool:
    try:
        return np.isnan(x)
    except Exception:
        return False


# =============================================================================
# Config builder — auto-discovers create_building_config in a sibling .py
# =============================================================================

# Make sibling modules importable when run via right-click
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Optional explicit override. Set this to the absolute path of your setup script
# if auto-discovery picks the wrong file. Leave as None for auto-discovery.
SETUP_SCRIPT_PATH: Optional[str] = None

# Module-level cache so we only locate / load the setup script once.
_CREATE_BUILDING_CONFIG = None
_SETUP_SOURCE: Optional[str] = None


def _load_module_from_path(name: str, path: str):
    """Load a Python file as a module without touching sys.modules cache for it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _discover_create_building_config():
    """Find `create_building_config` in a sibling .py file. Cached after first call."""
    global _CREATE_BUILDING_CONFIG, _SETUP_SOURCE
    if _CREATE_BUILDING_CONFIG is not None:
        return _CREATE_BUILDING_CONFIG

    candidates: List[str] = []

    # 1) Explicit override
    if SETUP_SCRIPT_PATH and os.path.isfile(SETUP_SCRIPT_PATH):
        candidates.append(SETUP_SCRIPT_PATH)

    # 2) Common filenames in the script directory
    common_names = [
        "setup_multi_buildings.py",
        "setup_buildings.py",
        "setup.py",
        "config_setup.py",
        "multi_buildings_setup.py",
        "build_config.py",
    ]
    for name in common_names:
        path = os.path.join(SCRIPT_DIR, name)
        if os.path.isfile(path):
            candidates.append(path)

    # 3) Any other .py in the script directory (skip ourselves and the runner)
    self_name = os.path.basename(os.path.abspath(__file__))
    skip = {self_name, "run_multi_buildings.py", "__init__.py"}
    try:
        for fname in sorted(os.listdir(SCRIPT_DIR)):
            if not fname.endswith(".py") or fname in skip:
                continue
            full = os.path.join(SCRIPT_DIR, fname)
            if full not in candidates:
                candidates.append(full)
    except OSError:
        pass

    errors: List[str] = []
    for path in candidates:
        mod_name = os.path.splitext(os.path.basename(path))[0]
        # Use a unique synthetic name so repeated failures aren't cached against
        # the real module name.
        synthetic = f"_bestopt_setup__{mod_name}"
        try:
            mod = _load_module_from_path(synthetic, path)
        except Exception as e:
            errors.append(f"  - {path}: {type(e).__name__}: {e}")
            continue
        fn = getattr(mod, "create_building_config", None)
        if callable(fn):
            _CREATE_BUILDING_CONFIG = fn
            _SETUP_SOURCE = path
            logger.info(f"Loaded create_building_config from: {path}")
            return fn
        else:
            errors.append(f"  - {path}: no `create_building_config` function")

    msg_lines = [
        "Could not locate a sibling Python file defining `create_building_config`.",
        f"Searched in: {SCRIPT_DIR}",
        "Tried:",
        *errors,
        "",
        "Fix options:",
        "  (a) Put your building-setup script in the same directory as this "
        "benchmark script (it must define `create_building_config`).",
        "  (b) Set SETUP_SCRIPT_PATH near the top of this benchmark script "
        "to the absolute path of your setup script.",
    ]
    raise RuntimeError("\n".join(msg_lines))


def build_config_for_n_buildings(n_buildings: int, output_path: str) -> None:
    """Generate a BESTOpt configuration JSON for N buildings."""
    create_building_config = _discover_create_building_config()

    from bestopt.env.core.config_manager import ConfigurationManager

    cm = ConfigurationManager()
    cluster_id = "residential_cluster_multi"
    cm.add_cluster(cluster_id, parameters={"location": "Syracuse, NY"})

    all_buildings: List[str] = []
    all_systems: List[str] = []
    building_configs: Dict[str, Dict[str, Any]] = {}

    for i in range(1, n_buildings + 1):
        building_id = f"SFH_{i}"
        cfg = create_building_config(cm, building_id, cluster_id, building_seed=i * 100)
        all_buildings.append(building_id)
        all_systems.extend([cfg["hvac_system"], cfg["der_system"]])
        building_configs[building_id] = cfg

    # Shared disturbances
    cm.add_disturbance(
        "weather",
        parameters={
            "file_path": os.path.join(
                PROJECT_ROOT_PATH, "data", "SFH", "DIST", "weather", "weather.csv"
            ),
            "simulation_start_time": "2023-08-01 00:00:00",
        },
        class_path="bestopt.env.disturbances.weather.WeatherModule",
    )
    cm.add_disturbance(
        "occupancy",
        parameters={
            "file_path": os.path.join(
                PROJECT_ROOT_PATH, "data", "SFH", "DIST", "occupancy", "occupancy.csv"
            ),
            "simulation_start_time": "2023-08-01 00:00:00",
        },
        class_path="bestopt.env.disturbances.occupancy.OccupancyModule",
    )
    cm.add_disturbance(
        "price",
        parameters={},
        class_path="bestopt.env.disturbances.price.PriceModule",
    )

    # Environment
    cm.add_environment(
        parameters={
            "resolution": 900,
            "duration": 86400 * 2,
            "enable_history": True,
            "logging_level": "INFO",
            "simulation_start_time": "2023-08-01 00:00:00",
        },
        class_path="bestopt.environment.BESTOptEnvironment",
    )

    # Selections
    cm.select_cluster(cluster_id)
    cm.select_buildings(all_buildings)
    cm.select_systems(all_systems)
    for _, cfg in building_configs.items():
        cm.select_controller_for_system(cfg["hvac_system"], cfg["hvac_controller"])
        cm.select_controller_for_system(cfg["der_system"], cfg["der_controller"])
    cm.select_disturbances(["weather", "occupancy", "price"])
    cm.select_environment()

    cm.save_final_configuration(output_path)


# =============================================================================
# Per-N benchmark runner
# =============================================================================

def run_benchmark_for_n(
    n_buildings: int,
    max_steps: int,
    sample_interval_s: float = 0.5,
) -> Dict[str, Any]:
    """Run one full pipeline (config → init → simulate) for N buildings.

    Each stage is wrapped in its own try/except so a failure in (say) the
    simulation still yields useful config/init numbers in the result.
    """
    result: Dict[str, Any] = {
        "n_buildings": n_buildings,
        "max_steps": max_steps,
        "timestamp_start": datetime.now().isoformat(),
        "status": "pending",
    }

    config_path = os.path.join(OUTPUT_DIR, f"config_benchmark_{n_buildings}buildings.json")

    # -------- Stage 1: config generation --------
    mon = ResourceMonitor(interval=sample_interval_s, label="config")
    mon.start()
    t0 = time.perf_counter()
    try:
        build_config_for_n_buildings(n_buildings, config_path)
        wall = time.perf_counter() - t0
        result["config_stage"] = {
            "wall_time_s": wall,
            "status": "success",
            "resources": mon.stop(),
        }
        logger.info(f"[N={n_buildings}] Config built in {wall:.2f}s")
    except Exception as e:
        wall = time.perf_counter() - t0
        result["config_stage"] = {
            "wall_time_s": wall,
            "status": "failed",
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "resources": mon.stop(),
        }
        result["status"] = "failed_config"
        result["timestamp_end"] = datetime.now().isoformat()
        logger.error(f"[N={n_buildings}] Config stage FAILED: {e}")
        return result

    _cleanup()

    # -------- Stage 2: environment initialization (incl. ModNN warmup) --------
    mon = ResourceMonitor(interval=sample_interval_s, label="env_init")
    mon.start()
    t0 = time.perf_counter()
    env = None
    try:
        from bestopt.env.core.config_manager import ConfigurationManager
        from bestopt.env.core.environment import BESTOptEnvironment

        cm = ConfigurationManager(config_path)
        env = BESTOptEnvironment(cm.config)
        wall = time.perf_counter() - t0
        result["env_init_stage"] = {
            "wall_time_s": wall,
            "status": "success",
            "resources": mon.stop(),
        }
        logger.info(f"[N={n_buildings}] Env initialized in {wall:.2f}s")
    except Exception as e:
        wall = time.perf_counter() - t0
        result["env_init_stage"] = {
            "wall_time_s": wall,
            "status": "failed",
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "resources": mon.stop(),
        }
        result["status"] = "failed_env_init"
        result["timestamp_end"] = datetime.now().isoformat()
        logger.error(f"[N={n_buildings}] Env init FAILED: {e}")
        return result

    _cleanup()

    # -------- Stage 3: simulation loop --------
    mon = ResourceMonitor(interval=sample_interval_s, label="simulation")
    mon.start()
    t0 = time.perf_counter()
    step_times: List[float] = []
    try:
        total_steps = max_steps if max_steps else env.total_step
        for step in range(total_steps):
            t_step = time.perf_counter()
            _obs, done, _info = env.step()
            step_times.append(time.perf_counter() - t_step)
            if done:
                break
        wall = time.perf_counter() - t0
        step_arr = np.array(step_times) if step_times else np.array([0.0])
        result["simulation_stage"] = {
            "wall_time_s": wall,
            "n_steps_completed": len(step_times),
            "n_steps_requested": total_steps,
            "avg_step_time_s": float(step_arr.mean()),
            "max_step_time_s": float(step_arr.max()),
            "min_step_time_s": float(step_arr.min()),
            "std_step_time_s": float(step_arr.std()),
            "steps_per_second": (len(step_times) / wall) if wall > 0 else 0.0,
            "status": "success",
            "resources": mon.stop(),
        }
        logger.info(
            f"[N={n_buildings}] Sim done: {len(step_times)} steps in {wall:.2f}s "
            f"({len(step_times)/max(wall,1e-9):.2f} steps/s)"
        )
    except Exception as e:
        wall = time.perf_counter() - t0
        step_arr = np.array(step_times) if step_times else np.array([0.0])
        result["simulation_stage"] = {
            "wall_time_s": wall,
            "n_steps_completed": len(step_times),
            "avg_step_time_s": float(step_arr.mean()) if step_times else 0.0,
            "max_step_time_s": float(step_arr.max()) if step_times else 0.0,
            "status": "failed",
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "resources": mon.stop(),
        }
        result["status"] = "failed_simulation"
        result["timestamp_end"] = datetime.now().isoformat()
        logger.error(f"[N={n_buildings}] Simulation FAILED at step {len(step_times)}: {e}")
        try:
            del env
        except Exception:
            pass
        _cleanup()
        return result

    result["status"] = "success"
    result["total_wall_time_s"] = (
        result["config_stage"]["wall_time_s"]
        + result["env_init_stage"]["wall_time_s"]
        + result["simulation_stage"]["wall_time_s"]
    )
    result["timestamp_end"] = datetime.now().isoformat()

    try:
        del env
    except Exception:
        pass
    _cleanup()
    return result


def _cleanup() -> None:
    """Force GC + CUDA cache release between stages/runs."""
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass


# =============================================================================
# System info
# =============================================================================

def collect_system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    }
    try:
        freq = psutil.cpu_freq()
        if freq:
            info["cpu_freq_mhz_current"] = freq.current
            info["cpu_freq_mhz_max"] = freq.max
    except Exception:
        pass

    if TORCH_AVAILABLE:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            try:
                props = torch.cuda.get_device_properties(0)
                info["cuda_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
            except Exception:
                pass

    if GPU_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info["nvml_driver"] = pynvml.nvmlSystemGetDriverVersion()
            info["nvml_device_name"] = pynvml.nvmlDeviceGetName(handle)
            info["nvml_num_gpus"] = NUM_GPUS
        except Exception:
            pass

    return info


# =============================================================================
# Persistence (crash-safe save)
# =============================================================================

def load_existing_results() -> Dict[str, Any]:
    if not os.path.exists(RESULTS_JSON):
        return {"system_info": None, "results": []}
    try:
        with open(RESULTS_JSON, "r") as f:
            data = json.load(f)
        if "results" not in data:
            data = {"system_info": data.get("system_info"), "results": []}
        return data
    except Exception as e:
        logger.warning(f"Could not parse existing results file ({e}); starting fresh.")
        return {"system_info": None, "results": []}


def save_results_atomic(data: Dict[str, Any]) -> None:
    tmp = RESULTS_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, RESULTS_JSON)


# =============================================================================
# Reports (CSV + Markdown)
# =============================================================================

CSV_COLUMNS = [
    "n_buildings", "status",
    "config_wall_s", "env_init_wall_s", "sim_wall_s", "total_wall_s",
    "sim_n_steps", "sim_avg_step_ms", "sim_max_step_ms", "sim_steps_per_s",
    "sim_ram_mb_proc_mean", "sim_ram_mb_proc_max",
    "sim_ram_mb_sys_max", "sim_ram_pct_sys_max",
    "sim_cpu_pct_proc_mean", "sim_cpu_pct_proc_max",
    "sim_cpu_pct_sys_mean", "sim_cpu_pct_sys_max",
    "sim_gpu_mem_mb_mean", "sim_gpu_mem_mb_max",
    "sim_gpu_util_mean", "sim_gpu_util_max",
    "init_ram_mb_proc_max", "init_gpu_mem_mb_max",
]


def _g(d: Dict[str, Any], *path: str, default: Any = "") -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _row_for_result(r: Dict[str, Any]) -> Dict[str, Any]:
    sim_res = _g(r, "simulation_stage", "resources", default={}) or {}
    init_res = _g(r, "env_init_stage", "resources", default={}) or {}
    return {
        "n_buildings": r.get("n_buildings", ""),
        "status": r.get("status", ""),
        "config_wall_s": round(_g(r, "config_stage", "wall_time_s", default=0.0) or 0.0, 3),
        "env_init_wall_s": round(_g(r, "env_init_stage", "wall_time_s", default=0.0) or 0.0, 3),
        "sim_wall_s": round(_g(r, "simulation_stage", "wall_time_s", default=0.0) or 0.0, 3),
        "total_wall_s": round(r.get("total_wall_time_s", 0.0) or 0.0, 3),
        "sim_n_steps": _g(r, "simulation_stage", "n_steps_completed", default=""),
        "sim_avg_step_ms": round((_g(r, "simulation_stage", "avg_step_time_s", default=0.0) or 0.0) * 1000.0, 3),
        "sim_max_step_ms": round((_g(r, "simulation_stage", "max_step_time_s", default=0.0) or 0.0) * 1000.0, 3),
        "sim_steps_per_s": round(_g(r, "simulation_stage", "steps_per_second", default=0.0) or 0.0, 3),
        "sim_ram_mb_proc_mean": round(sim_res.get("ram_mb_process_rss_mean", 0.0), 1),
        "sim_ram_mb_proc_max": round(sim_res.get("ram_mb_process_rss_max", 0.0), 1),
        "sim_ram_mb_sys_max": round(sim_res.get("ram_mb_system_used_max", 0.0), 1),
        "sim_ram_pct_sys_max": round(sim_res.get("ram_percent_system_max", 0.0), 2),
        "sim_cpu_pct_proc_mean": round(sim_res.get("cpu_percent_process_mean", 0.0), 2),
        "sim_cpu_pct_proc_max": round(sim_res.get("cpu_percent_process_max", 0.0), 2),
        "sim_cpu_pct_sys_mean": round(sim_res.get("cpu_percent_system_mean", 0.0), 2),
        "sim_cpu_pct_sys_max": round(sim_res.get("cpu_percent_system_max", 0.0), 2),
        "sim_gpu_mem_mb_mean": round(sim_res.get("gpu_mem_used_mb_mean", 0.0), 1),
        "sim_gpu_mem_mb_max": round(sim_res.get("gpu_mem_used_mb_max", 0.0), 1),
        "sim_gpu_util_mean": round(sim_res.get("gpu_util_percent_mean", 0.0), 2),
        "sim_gpu_util_max": round(sim_res.get("gpu_util_percent_max", 0.0), 2),
        "init_ram_mb_proc_max": round(init_res.get("ram_mb_process_rss_max", 0.0), 1),
        "init_gpu_mem_mb_max": round(init_res.get("gpu_mem_used_mb_max", 0.0), 1),
    }


def write_csv(results: List[Dict[str, Any]]) -> None:
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x.get("n_buildings", 0)):
            writer.writerow(_row_for_result(r))


def write_markdown(system_info: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# BESTOpt Scaling Benchmark\n")
    lines.append(f"_Generated: {datetime.now().isoformat()}_\n")

    lines.append("## System\n")
    if system_info:
        for k, v in system_info.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("## Wall time per stage (seconds)\n")
    lines.append("| N | status | config | env_init | sim | total | steps | avg step (ms) | steps/s |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: x.get("n_buildings", 0)):
        row = _row_for_result(r)
        lines.append(
            f"| {row['n_buildings']} | {row['status']} | {row['config_wall_s']} | "
            f"{row['env_init_wall_s']} | {row['sim_wall_s']} | {row['total_wall_s']} | "
            f"{row['sim_n_steps']} | {row['sim_avg_step_ms']} | {row['sim_steps_per_s']} |"
        )
    lines.append("")

    lines.append("## Memory (simulation stage)\n")
    lines.append("| N | RAM proc max (MB) | RAM sys max (MB) | RAM sys max (%) | GPU mem max (MB) | GPU util max (%) |")
    lines.append("|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: x.get("n_buildings", 0)):
        row = _row_for_result(r)
        lines.append(
            f"| {row['n_buildings']} | {row['sim_ram_mb_proc_max']} | {row['sim_ram_mb_sys_max']} | "
            f"{row['sim_ram_pct_sys_max']} | {row['sim_gpu_mem_mb_max']} | {row['sim_gpu_util_max']} |"
        )
    lines.append("")

    lines.append("## CPU (simulation stage)\n")
    lines.append("| N | CPU proc mean (%) | CPU proc max (%) | CPU sys mean (%) | CPU sys max (%) |")
    lines.append("|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: x.get("n_buildings", 0)):
        row = _row_for_result(r)
        lines.append(
            f"| {row['n_buildings']} | {row['sim_cpu_pct_proc_mean']} | {row['sim_cpu_pct_proc_max']} | "
            f"{row['sim_cpu_pct_sys_mean']} | {row['sim_cpu_pct_sys_max']} |"
        )
    lines.append("")

    # Failures
    failures = [r for r in results if r.get("status") != "success"]
    if failures:
        lines.append("## Failures\n")
        for r in failures:
            n = r.get("n_buildings", "?")
            stage = r.get("status", "?")
            err = (
                _g(r, "simulation_stage", "error")
                or _g(r, "env_init_stage", "error")
                or _g(r, "config_stage", "error")
                or "(no error message captured)"
            )
            lines.append(f"- **N={n}** ({stage}): {err}")
        lines.append("")

    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(lines))


# =============================================================================
# Main loop
# =============================================================================

def main(
    building_counts: List[int],
    max_steps: int,
    sample_interval_s: float = 0.5,
    rerun_failed: bool = True,
) -> None:
    logger.info("=" * 70)
    logger.info(f"BESTOpt benchmark starting: sizes={building_counts}, max_steps={max_steps}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 70)

    data = load_existing_results()
    if data.get("system_info") is None:
        data["system_info"] = collect_system_info()
    logger.info(f"System: {json.dumps(data['system_info'], default=str)}")

    by_n: Dict[int, Dict[str, Any]] = {r["n_buildings"]: r for r in data["results"]}

    # Purge stale failures left over from the pre-fix import-discovery bug,
    # so a rerun cleanly re-attempts them.
    _stale_markers = (
        "Could not import `setup_multi_buildings`",
        "Could not locate a sibling Python file",
    )
    purged = []
    for n, r in list(by_n.items()):
        if r.get("status") == "success":
            continue
        err = (
            _g(r, "config_stage", "error", default="")
            or _g(r, "env_init_stage", "error", default="")
            or ""
        )
        if any(m in str(err) for m in _stale_markers):
            purged.append(n)
            by_n.pop(n)
    if purged:
        logger.info(f"Purged stale import-error entries for N={purged}; will re-run.")

    for n in building_counts:
        prev = by_n.get(n)
        if prev is not None and prev.get("status") == "success":
            logger.info(f"[N={n}] Already completed successfully — skipping.")
            continue
        if prev is not None and not rerun_failed:
            logger.info(f"[N={n}] Previous run failed ({prev.get('status')}) — skipping (rerun_failed=False).")
            continue
        if prev is not None:
            logger.info(f"[N={n}] Re-running previously failed entry ({prev.get('status')}).")

        logger.info(f"\n{'='*70}\nRunning N={n}\n{'='*70}")
        try:
            result = run_benchmark_for_n(
                n_buildings=n,
                max_steps=max_steps,
                sample_interval_s=sample_interval_s,
            )
        except KeyboardInterrupt:
            logger.warning("Interrupted by user. Saving progress and exiting.")
            data["results"] = list(by_n.values())
            save_results_atomic(data)
            write_csv(data["results"])
            write_markdown(data["system_info"], data["results"])
            raise
        except Exception as e:
            # Defensive: should not happen since run_benchmark_for_n catches its own
            result = {
                "n_buildings": n,
                "status": "crashed_outer",
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "timestamp_end": datetime.now().isoformat(),
            }
            logger.exception(f"[N={n}] OUTER CRASH: {e}")

        by_n[n] = result
        data["results"] = list(by_n.values())

        # Incremental save after every N
        save_results_atomic(data)
        write_csv(data["results"])
        write_markdown(data["system_info"], data["results"])

        _print_one_result(result)
        _cleanup()

    logger.info("\n" + "=" * 70)
    logger.info("Benchmark complete.")
    logger.info(f"  JSON:     {RESULTS_JSON}")
    logger.info(f"  CSV:      {RESULTS_CSV}")
    logger.info(f"  Markdown: {RESULTS_MD}")
    logger.info(f"  Log:      {RUN_LOG}")
    logger.info("=" * 70)


def _print_one_result(r: Dict[str, Any]) -> None:
    n = r.get("n_buildings")
    status = r.get("status")
    if status == "success":
        cs = _g(r, "config_stage", "wall_time_s", default=0.0)
        es = _g(r, "env_init_stage", "wall_time_s", default=0.0)
        ss = _g(r, "simulation_stage", "wall_time_s", default=0.0)
        sps = _g(r, "simulation_stage", "steps_per_second", default=0.0)
        ram_max = _g(r, "simulation_stage", "resources", "ram_mb_process_rss_max", default=0.0)
        gpu_max = _g(r, "simulation_stage", "resources", "gpu_mem_used_mb_max", default=0.0)
        logger.info(
            f"  N={n} ✓  config={cs:.2f}s  init={es:.2f}s  sim={ss:.2f}s  "
            f"({sps:.2f} steps/s)  RAM_max={ram_max:.0f}MB  GPU_max={gpu_max:.0f}MB"
        )
    else:
        logger.info(f"  N={n} ✗  status={status}")


# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    # ------- knobs -------
    BUILDING_COUNTS = [1, 5, 10, 30, 50, 100, 200]
    MAX_STEPS = 96 * 3          # 3 simulated days at 15-min resolution
    SAMPLE_INTERVAL_S = 0.5     # resource sampler period
    RERUN_FAILED = True         # set False to skip past failures on re-run
    # ---------------------

    main(
        building_counts=BUILDING_COUNTS,
        max_steps=MAX_STEPS,
        sample_interval_s=SAMPLE_INTERVAL_S,
        rerun_failed=RERUN_FAILED,
    )