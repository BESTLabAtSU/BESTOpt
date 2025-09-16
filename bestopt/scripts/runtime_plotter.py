import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque

class WorkingRealTimePlotter:
    def __init__(self, max_points=100, window_title="HVAC Monitor"):
        self.max_points = max_points

        # Data storage
        self.timesteps = deque(maxlen=max_points)
        self.temperatures = deque(maxlen=max_points)
        self.hvac_powers = deque(maxlen=max_points)

        # Create the plot
        plt.ion()  # Interactive mode on
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.canvas.manager.set_window_title(window_title)

        # Initialize empty plots
        self.temp_line, = self.ax1.plot([], [], 'b-', linewidth=2, label='Temperature')
        self.power_line, = self.ax2.plot([], [], 'r-', linewidth=2, label='HVAC Power')

        # Setup axes
        self.ax1.set_title('Zone Temperature', fontsize=12)
        self.ax1.set_ylabel('Temperature (°C)', fontsize=11)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()

        self.ax2.set_title('HVAC Power Consumption', fontsize=12)
        self.ax2.set_xlabel('Timestep', fontsize=11)
        self.ax2.set_ylabel('Power (W)', fontsize=11)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()

        plt.tight_layout()

        # Show the window immediately
        plt.show(block=False)
        plt.pause(0.1)

        print("✓ Real-time plotter initialized and window displayed")
        print("  Window should be visible on your screen now")

    def add_data_point(self, timestep, temperature, hvac_power):
        """Add a data point and update the plot"""
        # Add data
        self.timesteps.append(timestep)
        self.temperatures.append(temperature)
        self.hvac_powers.append(hvac_power)

        # Convert to lists for plotting
        x_data = list(self.timesteps)
        temp_data = list(self.temperatures)
        power_data = list(self.hvac_powers)

        # Update the lines
        self.temp_line.set_data(x_data, temp_data)
        self.power_line.set_data(x_data, power_data)

        # Auto-scale the axes
        if len(x_data) > 1:
            # X-axis
            self.ax1.set_xlim(min(x_data), max(x_data))
            self.ax2.set_xlim(min(x_data), max(x_data))

            # Temperature Y-axis
            temp_min, temp_max = min(temp_data), max(temp_data)
            temp_margin = max(0.1, (temp_max - temp_min) * 0.1)
            self.ax1.set_ylim(temp_min - temp_margin, temp_max + temp_margin)

            # Power Y-axis
            power_min, power_max = min(power_data), max(power_data)
            if power_max == power_min:
                # Handle case where all values are the same (like all zeros)
                center = power_min
                self.ax2.set_ylim(center - 1, center + 1)
            else:
                power_margin = (power_max - power_min) * 0.1
                self.ax2.set_ylim(power_min - power_margin, power_max + power_margin)

        # Force the plot to update
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Small pause to allow GUI to update
        plt.pause(0.001)

        # Debug output every 10 points
        if len(self.timesteps) % 10 == 0:
            print(f"  Step {timestep}: T={temperature:.2f}°C, P={hvac_power:.2f}W [{len(self.timesteps)} points]")

    def stop(self):
        """Stop the real-time plotting and show final result"""
        print(f"\n✓ Stopping plotter. Total points collected: {len(self.timesteps)}")

        if len(self.timesteps) > 0:
            # Final statistics
            temp_stats = f"Temperature: {min(self.temperatures):.1f} to {max(self.temperatures):.1f}°C (avg: {np.mean(self.temperatures):.1f}°C)"
            power_stats = f"HVAC Power: {min(self.hvac_powers):.1f} to {max(self.hvac_powers):.1f}W (avg: {np.mean(self.hvac_powers):.1f}W)"

            print(f"  {temp_stats}")
            print(f"  {power_stats}")

            # Check for potential issues
            if all(p == 0 for p in self.hvac_powers):
                print("  ⚠️  NOTE: HVAC power was always 0 - system may not be active")

        # Turn off interactive mode and show final plot
        plt.ioff()
        plt.show()


def quick_realtime_plot(max_points=200):
    """Simple function to create a working real-time plotter"""
    return WorkingRealTimePlotter(max_points=max_points)


# Test function
def test_realtime_plotter():
    """Test the real-time plotter with synthetic data"""
    print("=== TESTING REAL-TIME PLOTTER ===")

    plotter = quick_realtime_plot(max_points=50)

    print("Generating test data (30 points)...")
    print("Watch the plot window - it should update in real-time!")

    try:
        for i in range(30):
            # Generate realistic HVAC data
            temp = 22 + 2 * np.sin(i * 0.1) + np.random.normal(0, 0.2)

            # Simulate HVAC turning on/off
            if i < 10:
                power = 0  # Initially off
            elif i < 20:
                power = 800 + np.random.normal(0, 50)  # On
            else:
                power = max(0, 400 * np.sin(i * 0.3) + np.random.normal(0, 30))  # Variable

            plotter.add_data_point(i, temp, power)
            time.sleep(0.2)  # Update every 200ms

        print("\nTest complete! Press Enter to close...")
        input()

    except KeyboardInterrupt:
        print("\nTest interrupted by user")

    plotter.stop()


# Alternative: Non-animated version for difficult environments
class SimpleUpdatingPlotter:
    """Even simpler version - just updates the plot without animation"""

    def __init__(self, max_points=100):
        self.max_points = max_points
        self.data = {'time': [], 'temp': [], 'power': []}

        # Create static plot that we'll update
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('HVAC System Monitor', fontsize=14)

        plt.ion()
        plt.show(block=False)
        print("✓ Simple updating plotter ready")

    def add_data_point(self, timestep, temperature, hvac_power):
        """Add data and refresh plot"""
        self.data['time'].append(timestep)
        self.data['temp'].append(temperature)
        self.data['power'].append(hvac_power)

        # Keep only recent data
        if len(self.data['time']) > self.max_points:
            for key in self.data:
                self.data[key] = self.data[key][-self.max_points:]

        # Clear and replot
        self.ax1.clear()
        self.ax2.clear()

        # Plot new data
        self.ax1.plot(self.data['time'], self.data['temp'], 'b-', linewidth=2)
        self.ax1.set_title('Temperature')
        self.ax1.set_ylabel('°C')
        self.ax1.grid(True, alpha=0.3)

        self.ax2.plot(self.data['time'], self.data['power'], 'r-', linewidth=2)
        self.ax2.set_title('HVAC Power')
        self.ax2.set_xlabel('Timestep')
        self.ax2.set_ylabel('W')
        self.ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Update display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

        if len(self.data['time']) % 20 == 0:
            print(f"  Updated plot: {len(self.data['time'])} points")

    def stop(self):
        """Show final plot"""
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    # Run the test
    test_realtime_plotter()