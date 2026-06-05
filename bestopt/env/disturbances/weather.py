"""
Weather disturbance module — fixed for dynamic start time.

Change from original: _get_weather_from_data and _get_weather_forecast_from_data
use self.sim_start_time instead of hardcoded "2023-08-01 00:00:00".
Added update_start_time() for episode randomization.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import os
import pandas as pd
from ..core.base import BaseModule
from ..core.data_structure import WeatherDisturbance

REQUIRED_COLS_CSV = {"outdoor_temperature", "solar_radiation"}


class WeatherModule(BaseModule):
    """
    Weather module that provides outdoor temperature and solar radiation.
    """

    def __init__(self, config: Dict[str, Any], name: str = "Weather"):
        super().__init__(config, name)
        self.weather_data: Optional[pd.DataFrame] = None
        self.weather = WeatherDisturbance()
        self.current_timestep = 0
        self.location = config.get("location", "Syracuse, NY")

    def initialize(self) -> None:
        """Initialize weather data source."""
        file_path = self.config.get("file_path")
        self.sim_start_time = pd.Timestamp(self.config.get("simulation_start_time", "2023-08-01 00:00:00"))

        if file_path:
            try:
                if not os.path.isfile(file_path):
                    raise FileNotFoundError(f"No such file: {file_path}")
                self._load_weather_file(file_path)
            except Exception as e:
                self.logger.warning(f"Failed to load weather file {file_path}: {e}")
                self.logger.info("Falling back to no weather data.")
                self.weather_data = None
        else:
            self.logger.info("No weather data provided.")
            self.weather_data = None

        # Pre-compute sim_data slice for the default start time
        self._update_sim_data()

        self.weather = WeatherDisturbance(outdoor_dry_bulb_temp=0.0, outdoor_wet_bulb_temp=0.0, solar_radiation_w_m2=0.0)
        self.logger.info(f"Weather module initialized: {self.name}, start={self.sim_start_time}")

    def _update_sim_data(self):
        """Pre-slice weather data from sim_start_time onwards."""
        if self.weather_data is not None and not self.weather_data.empty:
            self.weather_data['Time'] = pd.to_datetime(self.weather_data['Time'])
            self._sim_data = self.weather_data[self.weather_data['Time'] >= self.sim_start_time]
            self.logger.debug(f"Weather sim_data: {len(self._sim_data)} rows from {self.sim_start_time}")
        else:
            self._sim_data = pd.DataFrame()

    def update_start_time(self, new_start_time: str):
        """
        Update simulation start time (called by gym wrapper for episode randomization).

        Args:
            new_start_time: e.g. "2023-08-15 00:00:00"
        """
        self.sim_start_time = pd.Timestamp(new_start_time)
        self._update_sim_data()
        self.logger.debug(f"Weather start time updated to {self.sim_start_time}, "
                         f"{len(self._sim_data)} rows available")

    def get_max_steps(self) -> int:
        """Return max available steps from current start time."""
        return len(self._sim_data) if self._sim_data is not None else 0

    def step(self, current_step: int) -> Optional[WeatherDisturbance]:
        self.current_timestep = current_step

        if self._sim_data is None or self._sim_data.empty:
            self.logger.error("Weather data not loaded; step() returning None.")
            return None

        if not (0 <= current_step < len(self._sim_data)):
            self.logger.error(
                f"Requested step {current_step} out of range [0, {len(self._sim_data) - 1}]."
            )
            return None

        self._get_weather_from_data(current_step)
        self._get_weather_forecast_from_data(current_step)

        return self.weather

    def _load_weather_file(self, file_path: str) -> None:
        """Load weather data from CSV or EPW file."""
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path, index_col=0)
            missing = REQUIRED_COLS_CSV - set(df.columns)
            if missing:
                raise ValueError(f"CSV is missing required columns: {missing}")
            self.weather_data = df
            self.logger.info(f"Loaded CSV weather data: {len(self.weather_data)} records.")

        elif file_path.lower().endswith(".epw"):
            names = [
                "year", "month", "day", "hour", "minute", "data_source",
                "dry_bulb_temp", "dew_point_temp", "relative_humidity", "atmospheric_pressure",
                "extraterrestrial_horizontal_radiation", "extraterrestrial_direct_radiation",
                "horizontal_infrared_radiation", "global_horizontal_radiation",
                "direct_normal_radiation", "diffuse_horizontal_radiation"
            ] + [f"field_{i}" for i in range(16, 35)]
            try:
                epw = pd.read_csv(file_path, skiprows=8, header=None, names=names)
                epw["hour"] = epw["hour"].clip(1, 24) - 1
                epw["minute"] = epw.get("minute", 0).fillna(0).astype(int)
                epw["datetime"] = pd.to_datetime(
                    epw[["year", "month", "day", "hour", "minute"]], errors="coerce"
                )
                if epw["datetime"].isna().any():
                    n_bad = int(epw["datetime"].isna().sum())
                    self.logger.warning(f"{n_bad} EPW rows had invalid datetimes.")
                    epw = epw.dropna(subset=["datetime"])

                epw = epw.set_index("datetime").sort_index()
                epw["outdoor_temperature"] = epw["dry_bulb_temp"].astype(float)
                epw["solar_radiation"] = epw["global_horizontal_radiation"].astype(float)
                self.weather_data = epw[["outdoor_temperature", "solar_radiation"]]
                self.logger.info(f"Loaded EPW weather data: {len(self.weather_data)} records.")
            except Exception as e:
                raise ValueError(f"Failed to parse EPW file: {e}") from e
        else:
            raise ValueError(f"Unsupported weather file format: {file_path}")

    def _get_weather_from_data(self, current_step: int):
        """Read weather for current step — uses pre-sliced _sim_data."""
        row = self._sim_data.iloc[current_step]
        ot = (float(row["outdoor_temperature"]) - 32) * 5 / 9  # @TODO standardize units
        sr = float(row["solar_radiation"])
        self.weather.outdoor_dry_bulb_temp = ot
        self.weather.outdoor_wet_bulb_temp = ot
        self.weather.solar_radiation_w_m2 = sr

    def _get_weather_forecast_from_data(self, current_step: int):
        """Read weather forecast — uses pre-sliced _sim_data."""
        forecast = self._sim_data.iloc[current_step:current_step + 96]
        temp_forecast = ((forecast["outdoor_temperature"] - 32) * 5 / 9).to_numpy()
        solar_forecast = forecast["solar_radiation"].to_numpy()
        self.weather.forecast_outdoor_dry_bulb_temp = temp_forecast
        self.weather.forecast_outdoor_wet_bulb_temp = temp_forecast
        self.weather.forecast_solar_radiation_w_m2 = solar_forecast

    def reset(self) -> None:
        self.current_timestep = 0
        self.weather = WeatherDisturbance(outdoor_dry_bulb_temp=0, outdoor_wet_bulb_temp=0, solar_radiation_w_m2=0)
        self.logger.debug(f"Reset weather module: {self.name}")

    def get_state(self) -> Dict[str, Any]:
        return {
            "current_timestep": self.current_timestep,
            "outdoor_dry_bulbtemperature": self.weather.outdoor_dry_bulb_temp,
            "outdoor_wet_bulb_temperature": self.weather.outdoor_wet_bulb_temp,
            "solar_radiation_w_m2": self.weather.solar_radiation_w_m2,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.current_timestep = int(state.get("current_timestep", 0))
        self.weather.outdoor_dry_bulb_temp = float(state.get("outdoor_dry_bulbtemperature", 0.0))
        self.weather.outdoor_wet_bulb_temp = float(state.get("outdoor_wet_bulb_temperature", 0.0))
        self.weather.solar_radiation_w_m2 = float(state.get("solar_radiation_w_m2", 0.0))