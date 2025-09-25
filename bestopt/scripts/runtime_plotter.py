import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


class HVACDashboard:
    def __init__(self, max_points=200, window_title="HVAC System Monitor"):
        self.max_points = max_points

        # Data storage
        self.timesteps = deque(maxlen=max_points)
        self.temperatures = deque(maxlen=max_points)
        self.hvac_thermal_loads = deque(maxlen=max_points)
        self.hvac_powers = deque(maxlen=max_points)
        self.cooling_setpoints = deque(maxlen=max_points)
        self.heating_setpoints = deque(maxlen=max_points)
        self.supply_air_temps_real = deque(maxlen=max_points)
        self.supply_air_temps_setpt = deque(maxlen=max_points)
        self.supply_air_flows_real = deque(maxlen=max_points)
        self.supply_air_flows_setpt = deque(maxlen=max_points)

        # Create figure with 5 subplots
        plt.ion()
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.canvas.manager.set_window_title(window_title)

        # Create subplots in a grid layout
        gs = self.fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        self.ax1 = self.fig.add_subplot(gs[0, :])  # Zone Temperature (full width)
        self.ax2 = self.fig.add_subplot(gs[1, 0])  # Thermal Load
        self.ax3 = self.fig.add_subplot(gs[1, 1])  # Power Consumption
        self.ax4 = self.fig.add_subplot(gs[2, 0])  # Supply Air Temperature
        self.ax5 = self.fig.add_subplot(gs[2, 1])  # Supply Air Flow Rate

        # Initialize lines for Zone Temperature
        self.temp_line, = self.ax1.plot([], [], 'b-', linewidth=2, label='Zone Temperature')
        self.cool_setpt_line, = self.ax1.plot([], [], '--', linewidth=1.5,
                                              label='Cooling Setpoint', color='red',
                                              drawstyle='steps-post')
        self.heat_setpt_line, = self.ax1.plot([], [], '--', linewidth=1.5,
                                              label='Heating Setpoint', color='blue',
                                              drawstyle='steps-post')

        # Initialize line for Thermal Load
        self.thermal_load_line, = self.ax2.plot([], [], 'orange', linewidth=2,
                                                label='Thermal Load')

        # Initialize line for Power Consumption
        self.power_line, = self.ax3.plot([], [], 'purple', linewidth=2,
                                         label='HVAC Power')

        # Initialize lines for Supply Air Temperature
        self.sat_real_line, = self.ax4.plot([], [], 'g-', linewidth=2,
                                            label='Actual SAT')
        self.sat_setpt_line, = self.ax4.plot([], [], 'g--', linewidth=1.5,
                                             label='Setpoint SAT', alpha=0.7)

        # Initialize lines for Supply Air Flow Rate
        self.saf_real_line, = self.ax5.plot([], [], 'c-', linewidth=2,
                                            label='Actual Flow')
        self.saf_setpt_line, = self.ax5.plot([], [], 'c--', linewidth=1.5,
                                             label='Setpoint Flow', alpha=0.7)

        # Setup axes
        self._setup_axes()

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        print("✓ Enhanced HVAC Dashboard initialized")
        print("  Monitoring: Zone Temp, Thermal Load, Power, SAT, SAF")

    def _setup_axes(self):
        """Configure all subplot axes"""
        # Zone Temperature
        self.ax1.set_title('Zone Temperature & Setpoints', fontsize=10, fontweight='bold')
        self.ax1.set_ylabel('Temperature (°C)', fontsize=9)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='upper right', fontsize=8)

        # Thermal Load
        self.ax2.set_title('HVAC Thermal Load', fontsize=10, fontweight='bold')
        self.ax2.set_xlabel('Timestep', fontsize=9)
        self.ax2.set_ylabel('Thermal Load (W)', fontsize=9)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc='upper right', fontsize=8)

        # Power Consumption
        self.ax3.set_title('HVAC Power Consumption', fontsize=10, fontweight='bold')
        self.ax3.set_xlabel('Timestep', fontsize=9)
        self.ax3.set_ylabel('Power (W)', fontsize=9)
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend(loc='upper right', fontsize=8)

        # Supply Air Temperature
        self.ax4.set_title('Supply Air Temperature', fontsize=10, fontweight='bold')
        self.ax4.set_xlabel('Timestep', fontsize=9)
        self.ax4.set_ylabel('Temperature (°C)', fontsize=9)
        self.ax4.grid(True, alpha=0.3)
        self.ax4.legend(loc='upper right', fontsize=8)

        # Supply Air Flow Rate
        self.ax5.set_title('Supply Air Flow Rate', fontsize=10, fontweight='bold')
        self.ax5.set_xlabel('Timestep', fontsize=9)
        self.ax5.set_ylabel('Flow Rate (m³/s)', fontsize=9)
        self.ax5.grid(True, alpha=0.3)
        self.ax5.legend(loc='upper right', fontsize=8)

    def add_data_point(self, timestep, zone_temperature, hvac_thermal_load, hvac_power,
                       supervisory_cooling_setpoint=None, supervisory_heating_setpoint=None,
                       supply_air_temp_real=None, supply_air_temp_setpt=None,
                       supply_air_flow_real=None, supply_air_flow_setpt=None):
        """Add a data point and update all plots"""

        # Store data
        self.timesteps.append(timestep)
        self.temperatures.append(zone_temperature)
        self.hvac_thermal_loads.append(hvac_thermal_load)
        self.hvac_powers.append(hvac_power)
        self.cooling_setpoints.append(
            float(supervisory_cooling_setpoint) if supervisory_cooling_setpoint is not None else np.nan
        )
        self.heating_setpoints.append(
            float(supervisory_heating_setpoint) if supervisory_heating_setpoint is not None else np.nan
        )
        self.supply_air_temps_real.append(
            float(supply_air_temp_real) if supply_air_temp_real is not None else np.nan
        )
        self.supply_air_temps_setpt.append(
            float(supply_air_temp_setpt) if supply_air_temp_setpt is not None else np.nan
        )
        self.supply_air_flows_real.append(
            float(supply_air_flow_real) if supply_air_flow_real is not None else np.nan
        )
        self.supply_air_flows_setpt.append(
            float(supply_air_flow_setpt) if supply_air_flow_setpt is not None else np.nan
        )

        # Convert to arrays for plotting
        x_data = list(self.timesteps)
        temp_data = list(self.temperatures)
        thermal_load_data = list(self.hvac_thermal_loads)
        power_data = list(self.hvac_powers)
        cool_setpt_data = np.array(self.cooling_setpoints, dtype=float)
        heat_setpt_data = np.array(self.heating_setpoints, dtype=float)
        sat_real_data = np.array(self.supply_air_temps_real, dtype=float)
        sat_setpt_data = np.array(self.supply_air_temps_setpt, dtype=float)
        saf_real_data = np.array(self.supply_air_flows_real, dtype=float)
        saf_setpt_data = np.array(self.supply_air_flows_setpt, dtype=float)

        # Update all lines
        self.temp_line.set_data(x_data, temp_data)
        self.cool_setpt_line.set_data(x_data, cool_setpt_data)
        self.heat_setpt_line.set_data(x_data, heat_setpt_data)
        self.thermal_load_line.set_data(x_data, thermal_load_data)
        self.power_line.set_data(x_data, power_data)
        self.sat_real_line.set_data(x_data, sat_real_data)
        self.sat_setpt_line.set_data(x_data, sat_setpt_data)
        self.saf_real_line.set_data(x_data, saf_real_data)
        self.saf_setpt_line.set_data(x_data, saf_setpt_data)

        # Auto-scale axes
        if len(x_data) > 1:
            self._autoscale_axes(x_data, temp_data, cool_setpt_data, heat_setpt_data,
                                 thermal_load_data, power_data, sat_real_data, sat_setpt_data,
                                 saf_real_data, saf_setpt_data)

        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

        # Debug output every 20 points
        if len(self.timesteps) % 20 == 0:
            print(f"  Step {timestep}: T={zone_temperature:.2f}°C, "
                  f"Load={hvac_thermal_load:.1f}W, Power={hvac_power:.1f}W "
                  f"[{len(self.timesteps)} points]")

    def _autoscale_axes(self, x_data, temp_data, cool_setpt_data, heat_setpt_data,
                        thermal_load_data, power_data, sat_real_data, sat_setpt_data,
                        saf_real_data, saf_setpt_data):
        """Autoscale all axes based on data"""
        xmin, xmax = min(x_data), max(x_data)

        # Set x-limits for all axes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
            ax.set_xlim(xmin, xmax)

        # Zone temperature (include setpoints)
        all_temps = np.concatenate([temp_data, cool_setpt_data, heat_setpt_data])
        tmin, tmax = np.nanmin(all_temps), np.nanmax(all_temps)
        margin = max(0.5, (tmax - tmin) * 0.1)
        self.ax1.set_ylim(tmin - margin, tmax + margin)

        # Thermal load
        if len(thermal_load_data) > 0:
            lmin, lmax = min(thermal_load_data), max(thermal_load_data)
            if lmax == lmin:
                self.ax2.set_ylim(lmin - 100, lmin + 100)
            else:
                lmargin = (lmax - lmin) * 0.1
                self.ax2.set_ylim(lmin - lmargin, lmax + lmargin)

        # Power
        if len(power_data) > 0:
            pmin, pmax = min(power_data), max(power_data)
            if pmax == pmin:
                self.ax3.set_ylim(pmin - 10, pmin + 10)
            else:
                pmargin = (pmax - pmin) * 0.1
                self.ax3.set_ylim(pmin - pmargin, pmax + pmargin)

        # Supply air temperature
        all_sat = np.concatenate([sat_real_data, sat_setpt_data])
        if not np.all(np.isnan(all_sat)):
            sat_min, sat_max = np.nanmin(all_sat), np.nanmax(all_sat)
            sat_margin = max(0.5, (sat_max - sat_min) * 0.1)
            self.ax4.set_ylim(sat_min - sat_margin, sat_max + sat_margin)

        # Supply air flow
        all_saf = np.concatenate([saf_real_data, saf_setpt_data])
        if not np.all(np.isnan(all_saf)):
            saf_min, saf_max = np.nanmin(all_saf), np.nanmax(all_saf)
            if saf_max == saf_min:
                self.ax5.set_ylim(saf_min - 0.01, saf_min + 0.01)
            else:
                saf_margin = (saf_max - saf_min) * 0.1
                self.ax5.set_ylim(saf_min - saf_margin, saf_max + saf_margin)

    def stop(self):
        """Stop the plotter and show final statistics"""
        print(f"\n✓ Stopping dashboard. Total points: {len(self.timesteps)}")

        if len(self.timesteps) > 0:
            # Print statistics
            print(f"  Zone Temp: {min(self.temperatures):.1f} to {max(self.temperatures):.1f}°C "
                  f"(avg: {np.mean(self.temperatures):.1f}°C)")
            print(f"  Thermal Load: {min(self.hvac_thermal_loads):.1f} to "
                  f"{max(self.hvac_thermal_loads):.1f}W "
                  f"(avg: {np.mean(self.hvac_thermal_loads):.1f}W)")
            print(f"  Power: {min(self.hvac_powers):.1f} to {max(self.hvac_powers):.1f}W "
                  f"(avg: {np.mean(self.hvac_powers):.1f}W)")

        plt.ioff()
        plt.show()


def create_hvac_dashboard(max_points=200):
    return HVACDashboard(max_points=max_points)