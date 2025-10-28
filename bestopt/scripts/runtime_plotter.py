import matplotlib
# matplotlib.use("QtAgg")  # Uncomment for interactive display
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from PIL import Image
import io


class HVACDashboard:
    """Dashboard for monitoring a single building's HVAC system"""

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
        self.fig = plt.figure(figsize=(6, 6), dpi=300)
        self.fig.canvas.manager.set_window_title(window_title)

        # Create subplots in a grid layout
        gs = self.fig.add_gridspec(3, 2, hspace=0.6, wspace=0.4)
        self.ax1 = self.fig.add_subplot(gs[0, :])
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.fig.add_subplot(gs[1, 1])
        self.ax4 = self.fig.add_subplot(gs[2, 0])
        self.ax5 = self.fig.add_subplot(gs[2, 1])

        # Initialize lines
        self.temp_line, = self.ax1.plot([], [], 'b-', linewidth=2, label='Zone Temperature')
        self.cool_setpt_line, = self.ax1.plot([], [], '--', linewidth=1.5,
                                              label='Cooling Setpoint', color='gray',
                                              drawstyle='steps-post')
        self.heat_setpt_line, = self.ax1.plot([], [], '--', linewidth=1.5,
                                              label='Heating Setpoint', color='gray',
                                              drawstyle='steps-post')
        self.thermal_load_line, = self.ax2.plot([], [], 'orange', linewidth=2,
                                                label='Thermal Load')
        self.power_line, = self.ax3.plot([], [], 'purple', linewidth=2,
                                         label='HVAC Power')

        # FIX: Add lines for both actual and setpoint values
        self.sat_real_line, = self.ax4.plot([], [], 'g-', linewidth=2,
                                            label='Actual SAT')
        self.sat_setpt_line, = self.ax4.plot([], [], 'g--', linewidth=1.5,
                                             label='Setpoint SAT', alpha=0.7,
                                             drawstyle='steps-post')

        self.saf_real_line, = self.ax5.plot([], [], 'c-', linewidth=2,
                                            label='Actual Flow')
        self.saf_setpt_line, = self.ax5.plot([], [], 'c--', linewidth=1.5,
                                             label='Setpoint Flow', alpha=0.7,
                                             drawstyle='steps-post')

        self._setup_axes()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.0001)

        print(f"✓ Individual HVAC Dashboard initialized: {window_title}")

    def _setup_axes(self):
        """Configure all subplot axes"""
        self.ax1.set_title('Zone Temperature & Setpoints', fontsize=7, fontweight='bold')
        self.ax1.set_ylabel('Temperature (°C)', fontsize=7)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='upper right', fontsize=6)

        self.ax2.set_title('HVAC Thermal Load', fontsize=7, fontweight='bold')
        self.ax2.set_xlabel('Timestep', fontsize=7)
        self.ax2.set_ylabel('Thermal Load (W)', fontsize=7)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc='upper right', fontsize=6)

        self.ax3.set_title('HVAC Power Consumption', fontsize=7, fontweight='bold')
        self.ax3.set_xlabel('Timestep', fontsize=7)
        self.ax3.set_ylabel('Power (W)', fontsize=7)
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend(loc='upper right', fontsize=6)

        self.ax4.set_title('Supply Air Temperature', fontsize=7, fontweight='bold')
        self.ax4.set_xlabel('Timestep', fontsize=7)
        self.ax4.set_ylabel('Temperature (°C)', fontsize=7)
        self.ax4.grid(True, alpha=0.3)
        self.ax4.legend(loc='upper right', fontsize=6)

        self.ax5.set_title('Supply Air Flow Rate', fontsize=7, fontweight='bold')
        self.ax5.set_xlabel('Timestep', fontsize=7)
        self.ax5.set_ylabel('Flow Rate (m³/s)', fontsize=7)
        self.ax5.grid(True, alpha=0.3)
        self.ax5.legend(loc='upper right', fontsize=6)

    def add_data_point(self, timestep, zone_temperature, hvac_thermal_load, hvac_power,
                       supervisory_cooling_setpoint=None, supervisory_heating_setpoint=None,
                       supply_air_temp_real=None, supply_air_temp_setpt=None,
                       supply_air_flow_real=None, supply_air_flow_setpt=None):
        """Add a data point and update all plots"""
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

        # Update plots
        x_data = list(self.timesteps)
        self.temp_line.set_data(x_data, list(self.temperatures))
        self.cool_setpt_line.set_data(x_data, np.array(self.cooling_setpoints, dtype=float))
        self.heat_setpt_line.set_data(x_data, np.array(self.heating_setpoints, dtype=float))
        self.thermal_load_line.set_data(x_data, list(self.hvac_thermal_loads))
        self.power_line.set_data(x_data, list(self.hvac_powers))

        # FIX: Update both actual and setpoint lines
        self.sat_real_line.set_data(x_data, np.array(self.supply_air_temps_real, dtype=float))
        self.sat_setpt_line.set_data(x_data, np.array(self.supply_air_temps_setpt, dtype=float))
        self.saf_real_line.set_data(x_data, np.array(self.supply_air_flows_real, dtype=float))
        self.saf_setpt_line.set_data(x_data, np.array(self.supply_air_flows_setpt, dtype=float))

        if len(x_data) > 1:
            self._autoscale_axes()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.0001)

    def _autoscale_axes(self):
        """Autoscale all axes based on data"""
        x_data = list(self.timesteps)
        xmin, xmax = min(x_data), max(x_data)

        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
            ax.set_xlim(xmin, xmax)

        # Zone temperature
        all_temps = np.concatenate([
            list(self.temperatures),
            np.array(self.cooling_setpoints, dtype=float),
            np.array(self.heating_setpoints, dtype=float)
        ])
        tmin, tmax = np.nanmin(all_temps), np.nanmax(all_temps)
        margin = max(0.5, (tmax - tmin) * 0.1)
        self.ax1.set_ylim(tmin - margin, tmax + margin)

        # Thermal load
        if len(self.hvac_thermal_loads) > 0:
            lmin, lmax = min(self.hvac_thermal_loads), max(self.hvac_thermal_loads)
            lmargin = (lmax - lmin) * 0.1 if lmax != lmin else 100
            self.ax2.set_ylim(lmin - lmargin, lmax + lmargin)

        # Power
        if len(self.hvac_powers) > 0:
            pmin, pmax = min(self.hvac_powers), max(self.hvac_powers)
            pmargin = (pmax - pmin) * 0.1 if pmax != pmin else 10
            self.ax3.set_ylim(pmin - pmargin, pmax + pmargin)

        # FIX: Supply Air Temperature autoscaling
        all_sat = np.concatenate([
            np.array(self.supply_air_temps_real, dtype=float),
            np.array(self.supply_air_temps_setpt, dtype=float)
        ])
        if len(all_sat) > 0 and not np.all(np.isnan(all_sat)):
            sat_min, sat_max = np.nanmin(all_sat), np.nanmax(all_sat)
            sat_margin = max(0.5, (sat_max - sat_min) * 0.1)
            self.ax4.set_ylim(sat_min - sat_margin, sat_max + sat_margin)

        # FIX: Supply Air Flow autoscaling
        all_saf = np.concatenate([
            np.array(self.supply_air_flows_real, dtype=float),
            np.array(self.supply_air_flows_setpt, dtype=float)
        ])
        if len(all_saf) > 0 and not np.all(np.isnan(all_saf)):
            saf_min, saf_max = np.nanmin(all_saf), np.nanmax(all_saf)
            saf_margin = max(0.001, (saf_max - saf_min) * 0.1)
            self.ax5.set_ylim(saf_min - saf_margin, saf_max + saf_margin)

    def save_as_gif(self, filename='hvac_dashboard.gif', fps=10):
        """Save the current figure as a GIF (requires capturing frames during simulation)"""
        print(f"⚠ Note: GIF saving requires frame capture during simulation")
        print(f"  Consider using screen recording or implementing frame capture in add_data_point()")

    def stop(self):
        """Stop the plotter"""
        print(f"✓ Stopping individual dashboard. Total points: {len(self.timesteps)}")
        plt.ioff()
        plt.show()


# Keep the rest of MultiHVACDashboard class unchanged...
class MultiHVACDashboard:
    """Dashboard for monitoring multiple buildings with 4 subplots and GIF export"""

    def __init__(self, building_names, max_points=200, save_gif=True, gif_filename="hvac_animation.gif"):
        self.building_names = building_names
        self.max_points = max_points
        self.num_buildings = len(building_names)
        self.save_gif = save_gif
        self.gif_filename = gif_filename

        # Data storage
        self.timesteps = deque(maxlen=max_points)
        self.building_data = {
            name: {
                'temperature': deque(maxlen=max_points),
                'thermal_load': deque(maxlen=max_points),
                'power': deque(maxlen=max_points),
                'cumulative_energy': deque(maxlen=max_points)
            }
            for name in building_names
        }

        # GIF frames storage
        self.gif_frames = []

        # Color palette - consistent colors for each building
        self.colors = plt.cm.tab10(np.linspace(0, 1, self.num_buildings))
        self.color_map = {name: self.colors[i] for i, name in enumerate(building_names)}

        # Create figure with 4 subplots
        plt.ion()
        self.fig = plt.figure(figsize=(14, 10), dpi=300)
        gs = self.fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        self.ax4 = self.fig.add_subplot(gs[1, 1])

        # Initialize line plots
        self.temp_lines = {}
        for name in building_names:
            line, = self.ax1.plot([], [], linewidth=2.5, label=name,
                                  color=self.color_map[name])
            self.temp_lines[name] = line

        self.power_line, = self.ax2.plot([], [], 'r-', linewidth=3, label='Total Power')

        self._setup_axes()
        plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1, hspace=0.35, wspace=0.3)
        plt.show(block=False)
        plt.pause(0.001)

        print(f"✓ Multi-Building HVAC Dashboard initialized for {self.num_buildings} buildings")
        if save_gif:
            print(f"✓ GIF recording enabled: {gif_filename}")

    def _setup_axes(self):
        """Configure axes"""
        self.ax1.set_title('Individual Space Air Temperature', fontsize=12, fontweight='bold')
        self.ax1.set_xlabel('Timestep', fontsize=10)
        self.ax1.set_ylabel('Temperature (°C)', fontsize=10)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='best', fontsize=9)

        self.ax2.set_title('Aggregated Electric Load', fontsize=12, fontweight='bold')
        self.ax2.set_xlabel('Timestep', fontsize=10)
        self.ax2.set_ylabel('Power (W)', fontsize=10)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc='best', fontsize=9)

        self.ax3.set_title('Real-Time Thermal Load Comparison', fontsize=12, fontweight='bold')
        self.ax3.set_ylabel('Thermal Load (W)', fontsize=10)
        self.ax3.grid(True, alpha=0.3, axis='y')

        self.ax4.set_title('Cumulative Energy Consumption Comparison', fontsize=12, fontweight='bold')
        self.ax4.set_ylabel('Energy (kWh)', fontsize=10)
        self.ax4.grid(True, alpha=0.3, axis='y')

    def add_data_point(self, timestep, building_data_dict, timestep_duration_s=900):
        """Add data point for all buildings"""
        self.timesteps.append(timestep)

        for building_name in self.building_names:
            if building_name in building_data_dict:
                data = building_data_dict[building_name]
                self.building_data[building_name]['temperature'].append(data['temperature'])
                self.building_data[building_name]['thermal_load'].append(data['thermal_load'])
                self.building_data[building_name]['power'].append(data['power'])

                if len(self.building_data[building_name]['cumulative_energy']) == 0:
                    cumulative = data['power'] * timestep_duration_s / 3600 / 1000
                else:
                    previous = list(self.building_data[building_name]['cumulative_energy'])[-1]
                    cumulative = previous + data['power'] * timestep_duration_s / 3600 / 1000
                self.building_data[building_name]['cumulative_energy'].append(cumulative)

        self._update_plots()

        if self.save_gif:
            self._save_frame()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

        if len(self.timesteps) % 20 == 0:
            total_power = sum(building_data_dict[name]['power']
                              for name in self.building_names
                              if name in building_data_dict)
            print(f"  Step {timestep}: Total Power={total_power:.1f}W")

    def _update_plots(self):
        """Update all plots"""
        x_data = list(self.timesteps)

        if len(x_data) < 1:
            return

        for building_name in self.building_names:
            temp_data = list(self.building_data[building_name]['temperature'])
            self.temp_lines[building_name].set_data(x_data, temp_data)

        total_power = np.zeros(len(x_data))
        for building_name in self.building_names:
            power_array = np.array(list(self.building_data[building_name]['power']))
            if len(power_array) < len(x_data):
                power_array = np.pad(power_array, (0, len(x_data) - len(power_array)),
                                     constant_values=0)
            total_power += power_array

        self.power_line.set_data(x_data, total_power)

        current_loads = [
            list(self.building_data[name]['thermal_load'])[-1]
            if len(self.building_data[name]['thermal_load']) > 0 else 0
            for name in self.building_names
        ]

        self.ax3.clear()
        bars3 = self.ax3.bar(range(len(self.building_names)), current_loads,
                             color=[self.color_map[name] for name in self.building_names],
                             alpha=0.7, edgecolor='black', linewidth=1.5)
        self.ax3.set_xticks(range(len(self.building_names)))
        self.ax3.set_xticklabels(self.building_names, rotation=45, ha='right', fontsize=9)
        self.ax3.set_ylabel('Thermal Load (W)', fontsize=10)
        self.ax3.set_title('Real-Time Thermal Load Comparison', fontsize=12, fontweight='bold')
        self.ax3.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars3, current_loads):
            height = bar.get_height()
            self.ax3.text(bar.get_x() + bar.get_width() / 2., height,
                          f'{val:.0f}', ha='center', va='bottom', fontsize=8)

        cumulative_energies = [
            list(self.building_data[name]['cumulative_energy'])[-1]
            if len(self.building_data[name]['cumulative_energy']) > 0 else 0
            for name in self.building_names
        ]

        self.ax4.clear()
        bars4 = self.ax4.bar(range(len(self.building_names)), cumulative_energies,
                             color=[self.color_map[name] for name in self.building_names],
                             alpha=0.7, edgecolor='black', linewidth=1.5)
        self.ax4.set_xticks(range(len(self.building_names)))
        self.ax4.set_xticklabels(self.building_names, rotation=45, ha='right', fontsize=9)
        self.ax4.set_ylabel('Energy (kWh)', fontsize=10)
        self.ax4.set_title('Cumulative Energy Consumption Comparison', fontsize=12, fontweight='bold')
        self.ax4.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars4, cumulative_energies):
            height = bar.get_height()
            self.ax4.text(bar.get_x() + bar.get_width() / 2., height,
                          f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        if len(x_data) > 1:
            xmin, xmax = min(x_data), max(x_data)
            self.ax1.set_xlim(xmin, xmax)
            self.ax2.set_xlim(xmin, xmax)
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()

    def _save_frame(self):
        """Save current plot as a frame for GIF"""
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        self.gif_frames.append(img.copy())
        buf.close()

    def export_gif(self, duration=100):
        """Export saved frames to GIF file"""
        if not self.gif_frames:
            print("⚠ No frames to export")
            return

        print(f"✓ Exporting {len(self.gif_frames)} frames to {self.gif_filename}...")
        self.gif_frames[0].save(
            self.gif_filename,
            save_all=True,
            append_images=self.gif_frames[1:],
            duration=duration,
            loop=0
        )
        print(f"✓ GIF saved successfully: {self.gif_filename}")

    def stop(self):
        """Stop the dashboard and export GIF"""
        print(f"\n✓ Stopping multi-building dashboard. Total points: {len(self.timesteps)}")

        if len(self.timesteps) > 0:
            total_energy_all = 0
            for building_name in self.building_names:
                temps = list(self.building_data[building_name]['temperature'])
                powers = list(self.building_data[building_name]['power'])
                loads = list(self.building_data[building_name]['thermal_load'])
                energy = list(self.building_data[building_name]['cumulative_energy'])[-1] if \
                self.building_data[building_name]['cumulative_energy'] else 0
                total_energy_all += energy

                print(f"  {building_name}: Avg Temp={np.mean(temps):.1f}°C, "
                      f"Avg Power={np.mean(powers):.1f}W, Avg Load={np.mean(loads):.1f}W, "
                      f"Total Energy={energy:.2f} kWh")

            print(f"\n  Total Energy Consumed (All Buildings): {total_energy_all:.2f} kWh")

        if self.save_gif:
            self.export_gif()

        plt.close(self.fig)


def create_hvac_dashboard(max_points=200, window_title="HVAC System Monitor"):
    """Create a single building HVAC dashboard"""
    return HVACDashboard(max_points=max_points, window_title=window_title)


def create_multi_dashboard(building_names, max_points=200, save_gif=True,
                           gif_filename="hvac_animation.gif"):
    """Create a multi-building dashboard with GIF export"""
    return MultiHVACDashboard(building_names=building_names, max_points=max_points,
                              save_gif=save_gif, gif_filename=gif_filename)