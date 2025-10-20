"""
PV module.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import logging

from bestopt.env.core.base import BaseModule
from bestopt.env.core.data_structure import PVComponentState, DERSystemAction, Disturbance


class PVModule(BaseModule):
    """

    """
    def __init__(self, config: Dict[str, Any], name: str = "PV"):
        """

        """
        super().__init__(config, name)
        # https://pvwatts.nrel.gov/downloads/pvwattsv5.pdf
        # PV system specifications
        self.rated_power_kw = config.get("rated_power_kw", 5.0)   # Nominal power in kW
        self.panel_area_m2 = config.get("panel_area_m2", 25.0)    # Total panel area in m²
        self.efficiency_stc = config.get("efficiency_stc", 0.20)  # Efficiency at STC (20%)

        # Temperature coefficients
        self.temp_coeff_power = config.get("temp_coeff_power", -0.004)  # %/°C
        self.noct = config.get("noct", 45.0)  # Nominal Operating Cell Temperature (°C)
        self.stc_temp = config.get("stc_temp", 25.0)  # Standard Test Conditions temp (°C)
        self.stc_irradiance = config.get("stc_irradiance", 1000.0)  # W/m²

        # Degradation and losses
        self.annual_degradation = config.get("annual_degradation", 0.005)  # 0.5% per year
        self.soiling_factor = config.get("soiling_factor", 0.98)  # 2% soiling loss
        self.shading_factor = config.get("shading_factor", 1.0)   # No shading by default
        self.inverter_efficiency = config.get("inverter_efficiency", 0.97)  # 97% inverter efficiency
        self.dc_losses = config.get("dc_losses", 0.98)  # 2% DC wiring losses
        self.min_irradiance = config.get("min_irradiance", 10.0)  # Minimum irradiance for operation (W/m²)
        self.max_power_output = self.rated_power_kw * 1000  # Convert to Watts

        # Installation details
        self.tilt_angle = config.get("tilt_angle", 30.0)  # degrees
        self.azimuth = config.get("azimuth", 180.0)       # degrees (180 = south-facing)

        # Tracking
        self.lifetime_energy_kwh = 0.0
        self.operating_hours = 0.0
        self.age_years = config.get("initial_age_years", 0.0)

        self.logger = logging.getLogger(f"{__name__}.{name}")

    def initialize(self) -> None:
        """Initialize the PV module."""
        self.logger.info(f"Initialized PV module: {self.name}")
        self.logger.info(f"Rated power: {self.rated_power_kw} kW")
        self.logger.info(f"Panel area: {self.panel_area_m2} m²")
        self._initialized = True

    def step(self,
             state: PVComponentState,
             disturbance: Disturbance,
             resolution: int,
             timestep: int) -> Dict[str, Any]:
        """

        """
        try:
            # Get weather conditions
            irradiance = disturbance.weather.solar_radiation_w_m2  # W/m²
            ambient_temp = disturbance.weather.outdoor_dry_bulb_temp  # °C

            # Calculate cell temperature using NOCT model
            cell_temp = self._calculate_cell_temperature(irradiance, ambient_temp)

            # Calculate degradation factor based on age
            degradation_factor = self._calculate_degradation_factor()

            # Calculate theoretical power output
            power_output = self._calculate_power_output(
                irradiance,
                cell_temp,
                degradation_factor
            )
            # print(irradiance, cell_temp, degradation_factor, power_output)
            # Apply curtailment if requested
            curtailment_factor = 1
                #self._get_curtailment_factor(action))
            power_output *= curtailment_factor

            # Update state
            state.generation_w = power_output
            state.irradiance = irradiance
            state.cell_temperature = cell_temp
            state.efficiency = self._calculate_current_efficiency(irradiance, cell_temp)
            state.curtailment = 1.0

            # Update tracking metrics
            energy_kwh = (power_output / 1000) * (resolution / 3600)
            self.lifetime_energy_kwh += energy_kwh
            if irradiance > self.min_irradiance:
                self.operating_hours += resolution / 3600

            # Log generation details
            if power_output > 10:  # Only log when generating
                self.logger.debug(
                    f"PV Generation: {power_output:.1f}W, "
                    f"Irradiance: {irradiance:.1f}W/m², "
                    f"Cell Temp: {cell_temp:.1f}°C, "
                    f"Efficiency: {state.efficiency:.1%}"
                )

            return {
                "power_generation": power_output,
                "energy_generated_kwh": energy_kwh,
                "cell_temperature": cell_temp,
                "efficiency": state.efficiency,
                "curtailment": state.curtailment
            }

        except Exception as e:
            self.logger.error(f"Error in PV step calculation: {e}")
            state.power_generation = 0.0
            return {"power_generation": 0.0, "error": str(e)}

    def _calculate_cell_temperature(self, irradiance: float, ambient_temp: float) -> float:
        """
        Calculate PV cell temperature using NOCT model.

        Args:
            irradiance: Solar irradiance (W/m²)
            ambient_temp: Ambient temperature (°C)

        Returns:
            Cell temperature in °C
        """
        # NOCT model: Tc = Ta + (NOCT - 20) * (G / 800)
        if irradiance <= 0:
            return ambient_temp

        cell_temp = ambient_temp + (self.noct - 20) * (irradiance / 800)
        return cell_temp

    def _calculate_degradation_factor(self) -> float:
        """
        Calculate degradation factor based on PV system age.

        Returns:
            Degradation factor (0-1)
        """
        # Linear degradation model
        degradation = 1.0 - (self.age_years * self.annual_degradation)
        return max(0.7, degradation)  # Minimum 70% of original capacity

    def _calculate_power_output(self,
                                irradiance: float,
                                cell_temp: float,
                                degradation_factor: float) -> float:
        """
        Calculate PV power output.

        Args:
            irradiance: Solar irradiance (W/m²)
            cell_temp: Cell temperature (°C)
            degradation_factor: Aging degradation factor

        Returns:
            Power output in Watts
        """
        # Check minimum irradiance threshold
        if irradiance < self.min_irradiance:
            return 0.0

        # Calculate temperature derating
        temp_derating = 1.0 + self.temp_coeff_power * (cell_temp - self.stc_temp)

        # Calculate power using efficiency-based model
        # P = η * A * G * losses
        power = (
            self.efficiency_stc *
            self.panel_area_m2 *
            irradiance *
            temp_derating *
            self.soiling_factor *
            self.shading_factor *
            self.inverter_efficiency *
            self.dc_losses *
            degradation_factor
        )

        # Alternative: Use rated power scaling
        # power = self.rated_power_kw * 1000 * (irradiance / self.stc_irradiance) * temp_derating * ...

        return power

    def _calculate_current_efficiency(self, irradiance: float, cell_temp: float) -> float:
        """
        Calculate current conversion efficiency.

        Args:
            irradiance: Solar irradiance (W/m²)
            cell_temp: Cell temperature (°C)

        Returns:
            Current efficiency (0-1)
        """
        if irradiance < self.min_irradiance:
            return 0.0

        # Account for temperature effects
        temp_derating = 1.0 + self.temp_coeff_power * (cell_temp - self.stc_temp)

        # Low-light performance adjustment (simplified)
        if irradiance < 200:
            low_light_factor = 0.95  # Slightly reduced efficiency at low light
        else:
            low_light_factor = 1.0

        efficiency = (
            self.efficiency_stc *
            temp_derating *
            low_light_factor *
            self.soiling_factor
        )

        return max(0, min(efficiency, 0.25))  # Cap at 25% max efficiency

    def _get_curtailment_factor(self, action: DERSystemAction) -> float:
        """
        Get curtailment factor from action.

        Args:
            action: Electrical action containing curtailment command

        Returns:
            Curtailment factor (0-1, where 1 = no curtailment)
        """
        # Check if action has PV curtailment command
        if hasattr(action, 'pv_curtailment'):
            # Curtailment is typically given as fraction to curtail
            # So available power factor = 1 - curtailment
            return 1.0 - min(1.0, max(0.0, action.pv_curtailment))
        return 1.0  # No curtailment by default

    def get_forecast(self, hours_ahead: int = 24) -> np.ndarray:
        """
        Get PV generation forecast (placeholder for forecasting logic).

        Args:
            hours_ahead: Number of hours to forecast

        Returns:
            Array of forecasted generation values
        """
        # This would typically use weather forecast data
        # For now, return a simple pattern
        forecast = np.zeros(hours_ahead)
        for h in range(hours_ahead):
            # Simple sine curve for daylight hours (6 AM to 6 PM)
            hour_of_day = h % 24
            if 6 <= hour_of_day <= 18:
                angle = (hour_of_day - 6) * np.pi / 12
                forecast[h] = self.max_power_output * 0.7 * np.sin(angle)

        return forecast

    def reset(self) -> None:
        """Reset the PV module to initial state."""
        self.lifetime_energy_kwh = 0.0
        self.operating_hours = 0.0
        self.logger.debug(f"Reset PV module: {self.name}")

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of PV module state.

        Returns:
            Dictionary with key metrics
        """
        return {
            "rated_power_kw": self.rated_power_kw,
            "lifetime_energy_kwh": self.lifetime_energy_kwh,
            "operating_hours": self.operating_hours,
            "age_years": self.age_years,
            "degradation": 1.0 - self._calculate_degradation_factor(),
            "max_power_w": self.max_power_output
        }