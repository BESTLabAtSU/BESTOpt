"""
Real-time Electrical Dashboard - Simplified Power Flow Model

3 subplots:
  1. Load Disaggregation: bold black total load line, stacked shading
     (HVAC, Lighting, Other, EV Charging), PV as negative area,
     curtailment as hatched region showing wasted generation
  2. Storage SOC: Battery + EV SOC traces with peak shading
  3. Net Grid Import: just the net grid power line
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from PIL import Image
import io


class ElectricalDashboard:
    """Dashboard for monitoring building electrical system with simplified power flow."""

    def __init__(self, max_points=200, window_title="Electrical System Monitor"):
        self.max_points = max_points

        # Data storage
        self.timesteps = deque(maxlen=max_points)

        # SOCs
        self.pv_generation = deque(maxlen=max_points)
        self.battery_soc = deque(maxlen=max_points)
        self.ev_tesla_soc = deque(maxlen=max_points)
        self.ev_nissan_soc = deque(maxlen=max_points)
        self.peak_signal = deque(maxlen=max_points)

        # Simplified power flow
        self.battery_power = deque(maxlen=max_points)
        self.ev_charging = deque(maxlen=max_points)
        self.grid_import = deque(maxlen=max_points)
        self.curtailment = deque(maxlen=max_points)
        self.electricity_price = deque(maxlen=max_points)

        # Load components
        self.total_load = deque(maxlen=max_points)
        self.hvac_load = deque(maxlen=max_points)
        self.cooking_load = deque(maxlen=max_points)
        self.pc_load = deque(maxlen=max_points)
        self.tv_load = deque(maxlen=max_points)
        self.lighting_load = deque(maxlen=max_points)

        # Component colors
        self.colors = {
            'hvac': '#4E79A7',
            'lighting': '#59A14F',
            'other': '#B07AA1',
            'ev': '#E15759',
            'pv': '#F28E2B',
            'curtailment': '#FF4444',
        }

        # GIF frames
        self.gif_frames = []

        # Create figure with 3 subplots
        plt.ion()
        self.fig = plt.figure(figsize=(14, 11), dpi=100)
        self.fig.canvas.manager.set_window_title(window_title)

        gs = self.fig.add_gridspec(3, 1, hspace=0.35,
                                   height_ratios=[2, 1, 1])
        self.ax1 = self.fig.add_subplot(gs[0])  # Load disaggregation
        self.ax2 = self.fig.add_subplot(gs[1])  # SOCs
        self.ax3 = self.fig.add_subplot(gs[2])  # Net grid import

        self._setup_axes()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

        print(f"Electrical Dashboard initialized: {window_title}")

    def _setup_axes(self):
        """Configure all subplot axes."""
        self.ax1.set_title('Load Disaggregation & PV Generation', fontsize=12,
                           fontweight='bold')
        self.ax1.set_ylabel('Power (kW)', fontsize=10)
        self.ax1.grid(True, alpha=0.3)

        self.ax2.set_title('Storage State of Charge', fontsize=12, fontweight='bold')
        self.ax2.set_ylabel('SOC (%)', fontsize=10)
        self.ax2.set_ylim(0, 105)
        self.ax2.grid(True, alpha=0.3)

        self.ax3.set_title('Net Grid Import', fontsize=12, fontweight='bold')
        self.ax3.set_xlabel('Timestep', fontsize=10)
        self.ax3.set_ylabel('Power (kW)', fontsize=10)
        self.ax3.grid(True, alpha=0.3)

    def add_data_point(self, timestep, pv_generation_kw, battery_soc, ev_tesla_soc,
                       ev_nissan_soc, total_load_kw, hvac_power_kw, cooking_kw,
                       pc_kw, tv_kw, lighting_kw, is_peak=False,
                       battery_power_kw=0.0, ev_charging_kw=0.0,
                       grid_import_kw=0.0, curtailment_kw=0.0,
                       electricity_price=0.0):
        """Add a data point and update all plots."""
        self.timesteps.append(timestep)

        self.pv_generation.append(pv_generation_kw)
        self.battery_soc.append(battery_soc * 100)
        self.ev_tesla_soc.append(ev_tesla_soc * 100)
        self.ev_nissan_soc.append(ev_nissan_soc * 100)
        self.peak_signal.append(is_peak)

        self.battery_power.append(battery_power_kw)
        self.ev_charging.append(ev_charging_kw)
        self.grid_import.append(grid_import_kw)
        self.curtailment.append(curtailment_kw)
        self.electricity_price.append(electricity_price)

        self.total_load.append(total_load_kw)
        self.hvac_load.append(hvac_power_kw)
        self.cooking_load.append(cooking_kw)
        self.pc_load.append(pc_kw)
        self.tv_load.append(tv_kw)
        self.lighting_load.append(lighting_kw)

        self._update_plots()
        self._save_frame()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def _update_plots(self):
        """Update all plots with current data."""
        x = list(self.timesteps)
        if len(x) < 1:
            return

        # ===== TOP: Load Disaggregation + PV negative + Curtailment =====
        self.ax1.clear()

        hvac = np.array(list(self.hvac_load))
        lighting = np.array(list(self.lighting_load))
        cooking = np.array(list(self.cooking_load))
        pc = np.array(list(self.pc_load))
        tv = np.array(list(self.tv_load))
        ev_chg = np.array(list(self.ev_charging))
        pv = np.array(list(self.pv_generation))
        curt = np.array(list(self.curtailment))
        total = np.array(list(self.total_load))

        # Other = cooking + pc + tv
        other = cooking + pc + tv

        # Stacked demand (bottom to top): HVAC, Lighting, Other, EV
        layer0 = np.zeros_like(hvac)

        self.ax1.fill_between(x, layer0, hvac,
                              color=self.colors['hvac'], alpha=0.80, label='HVAC')
        layer1 = hvac

        self.ax1.fill_between(x, layer1, layer1 + lighting,
                              color=self.colors['lighting'], alpha=0.80, label='Lighting')
        layer2 = layer1 + lighting

        self.ax1.fill_between(x, layer2, layer2 + other,
                              color=self.colors['other'], alpha=0.80, label='Other')
        layer3 = layer2 + other

        self.ax1.fill_between(x, layer3, layer3 + ev_chg,
                              color=self.colors['ev'], alpha=0.80, label='EV Charging')

        # Bold black total load line
        self.ax1.plot(x, total + ev_chg, color='black', linewidth=2,
                      label='Total Load', zorder=5)

        # Negative side: PV generation
        pv_used = pv - curt
        self.ax1.fill_between(x, 0, -pv_used,
                              color=self.colors['pv'], alpha=0.70, label='PV')

        # Curtailment: hatched region between usable PV and total PV
        if np.any(curt > 0.01):
            self.ax1.fill_between(x, -pv_used, -pv,
                                  color=self.colors['curtailment'], alpha=0.35,
                                  hatch='///',
                                  edgecolor=self.colors['curtailment'],
                                  linewidth=0.0, label='Curtailed')

        # PV outline
        self.ax1.plot(x, -pv, color='#D45B00', linewidth=0.8, zorder=3)

        self.ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')

        self.ax1.set_title('Load Disaggregation & PV Generation', fontsize=12,
                           fontweight='bold')
        self.ax1.set_ylabel('Power (kW)', fontsize=10)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='upper left', fontsize=8, ncol=4)

        # Y-limits
        pos_max = np.max(total + ev_chg) * 1.15 if np.max(total + ev_chg) > 0 else 1
        neg_min = -np.max(pv) * 1.15 if np.max(pv) > 0 else -0.5
        self.ax1.set_ylim([neg_min, pos_max])

        # ===== MIDDLE: SOCs =====
        self.ax2.clear()

        self.ax2.plot(x, list(self.battery_soc), 'b-', linewidth=2,
                      label='Battery SOC')
        self.ax2.plot(x, list(self.ev_tesla_soc), 'g-', linewidth=2,
                      label='Tesla EV SOC')
        self.ax2.plot(x, list(self.ev_nissan_soc), 'r-', linewidth=2,
                      label='Nissan EV SOC')

        self._draw_peak_shading(self.ax2)

        self.ax2.set_title('Storage State of Charge', fontsize=12, fontweight='bold')
        self.ax2.set_ylabel('SOC (%)', fontsize=10)
        self.ax2.set_ylim(0, 105)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc='upper left', fontsize=9)

        # ===== BOTTOM: Net Grid Import =====
        self.ax3.clear()

        grid = np.array(list(self.grid_import))
        self.ax3.plot(x, grid, color='gray', linewidth=2, label='Net Grid Import')
        self.ax3.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)

        self._draw_peak_shading(self.ax3)

        self.ax3.set_title('Net Grid Import', fontsize=12, fontweight='bold')
        self.ax3.set_xlabel('Timestep', fontsize=10)
        self.ax3.set_ylabel('Power (kW)', fontsize=10)
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend(loc='upper left', fontsize=9)

        # Autoscale x
        if len(x) > 1:
            xmin, xmax = min(x), max(x)
            self.ax1.set_xlim(xmin, xmax)
            self.ax2.set_xlim(xmin, xmax)
            self.ax3.set_xlim(xmin, xmax)

    def _draw_peak_shading(self, ax):
        """Draw peak period shading on a given axes."""
        x_data = list(self.timesteps)
        peak_data = list(self.peak_signal)
        if len(x_data) < 2:
            return

        in_peak = False
        peak_start = None
        for i, is_peak in enumerate(peak_data):
            if is_peak and not in_peak:
                peak_start = x_data[i]
                in_peak = True
            elif not is_peak and in_peak:
                ax.axvspan(peak_start, x_data[i], alpha=0.15, color='red')
                in_peak = False
        if in_peak and peak_start is not None:
            ax.axvspan(peak_start, x_data[-1], alpha=0.15, color='red')

    def _save_frame(self):
        """Save current plot as a frame for GIF."""
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        self.gif_frames.append(img.copy())
        buf.close()

    def save_as_gif(self, filename='electrical_dashboard.gif', fps=10):
        """Export saved frames to GIF file."""
        if not self.gif_frames:
            print("No frames to export")
            return
        print(f"Exporting {len(self.gif_frames)} frames to {filename}...")
        self.gif_frames[0].save(
            filename, save_all=True, append_images=self.gif_frames[1:],
            duration=1000 // fps, loop=0
        )
        print(f"GIF saved: {filename}")

    def stop(self):
        """Stop the plotter and show final statistics."""
        print(f"Stopping electrical dashboard. Total points: {len(self.timesteps)}")

        if len(self.timesteps) > 0:
            avg_pv = np.mean(list(self.pv_generation))
            max_pv = np.max(list(self.pv_generation))
            avg_total_load = np.mean(list(self.total_load))
            max_total_load = np.max(list(self.total_load))
            avg_grid = np.mean(list(self.grid_import))
            avg_bat = np.mean(list(self.battery_power))
            avg_ev_chg = np.mean(list(self.ev_charging))
            total_curtailment = np.sum(list(self.curtailment)) * 0.25

            final_bat_soc = list(self.battery_soc)[-1]
            final_tesla_soc = list(self.ev_tesla_soc)[-1]
            final_nissan_soc = list(self.ev_nissan_soc)[-1]

            print(f"\nElectrical System Statistics:")
            print(f"  PV Generation:   Avg={avg_pv:.2f}kW, Max={max_pv:.2f}kW")
            print(f"  Total Load:      Avg={avg_total_load:.2f}kW, Max={max_total_load:.2f}kW")
            print(f"  Grid Import:     Avg={avg_grid:.2f}kW")
            print(f"  Battery Power:   Avg={avg_bat:+.2f}kW (+=charge, -=discharge)")
            print(f"  EV Charging:     Avg={avg_ev_chg:.2f}kW")
            print(f"  Total Curtailed: {total_curtailment:.2f}kWh")
            print(f"  Final SOCs: Battery={final_bat_soc:.1f}%, "
                  f"Tesla={final_tesla_soc:.1f}%, Nissan={final_nissan_soc:.1f}%")

            avg_hvac = np.mean(list(self.hvac_load))
            avg_cooking = np.mean(list(self.cooking_load))
            avg_pc = np.mean(list(self.pc_load))
            avg_tv = np.mean(list(self.tv_load))
            avg_lighting = np.mean(list(self.lighting_load))

            print(f"\n  Load Breakdown (Average):")
            if avg_total_load > 0:
                print(f"    HVAC:     {avg_hvac:.2f}kW ({avg_hvac / avg_total_load * 100:.1f}%)")
                print(f"    Cooking:  {avg_cooking:.2f}kW ({avg_cooking / avg_total_load * 100:.1f}%)")
                print(f"    PC:       {avg_pc:.2f}kW ({avg_pc / avg_total_load * 100:.1f}%)")
                print(f"    TV:       {avg_tv:.2f}kW ({avg_tv / avg_total_load * 100:.1f}%)")
                print(f"    Lighting: {avg_lighting:.2f}kW ({avg_lighting / avg_total_load * 100:.1f}%)")

        plt.ioff()
        plt.show()


def create_electrical_dashboard(max_points=200, window_title="Electrical System Monitor"):
    """Create an electrical system dashboard."""
    return ElectricalDashboard(max_points=max_points, window_title=window_title)