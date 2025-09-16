import pandas as pd
import pickle
import numpy as np
import torch
from modnn.Config import _args
from modnn.utils import Mod
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.dates as mdates
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class HVACBuildingSimulator:
    """
    Comprehensive HVAC and Building Load Simulator
    Handles thermal-to-electric conversion, occupancy detection, and appliance loads
    """

    def __init__(self, model_args: Dict, scaler_path: str, data_path: str):
        """
        Initialize the HVAC Building Simulator

        Args:
            model_args: Dictionary containing model parameters
            scaler_path: Path to the scaler pickle file
            data_path: Path to the dataset CSV file
        """
        self.model_args = model_args
        self.scaler_path = scaler_path
        self.data_path = data_path

        # Initialize model and load scalers
        self._setup_model()
        self._load_scalers()
        self._load_data()

        # Initialize PI controller (will be updated with dynamic setpoints)
        self.controller = None

        # Results storage
        self.results = {}

        # Building parameters
        self.building_area = 100  # m2, adjust as needed
        self.lighting_power_density = 7.5  # W/m2

    def _setup_model(self):
        """Setup and load the neural network model"""
        args = _args(**self.model_args)
        self.mdl = Mod(args=args)
        self.mdl.data_ready()
        # self.mdl.train()
        self.mdl.load()
        self.mdl.test()
        self.mdl.check()
        self.mdl.dynamiccheck()
        self.mdl.check_show()
        self.dynamic = self.mdl.step_mdl()

    def _load_scalers(self):
        """Load the preprocessing scalers"""
        with open(self.scaler_path, "rb") as f:
            self.scalers = pickle.load(f)

    def _load_data(self):
        """Load and preprocess the dataset"""
        self.df = pd.read_csv(self.data_path, index_col=0)
        self.df.index = pd.to_datetime(self.df.index, format="%Y-%m-%d %H:%M:%S")

        # Add time features
        time_hours = self.df.index.hour + self.df.index.minute / 60
        self.df["day_sin"] = np.sin(2 * np.pi * time_hours / 24) / 2 + 0.5
        self.df["day_cos"] = np.cos(2 * np.pi * time_hours / 24) / 2 + 0.5
        self.df["hours"] = self.df.index.hour

    def run_simulation(self, sim_start: str, sim_end: str, sce: str,
                       controller_params: Dict = None) -> pd.DataFrame:
        """
        Run the complete building simulation

        Args:
            sim_start: Start date string
            sim_end: End date string
            controller_params: PI controller parameters

        Returns:
            results_df: DataFrame with all simulation results
        """
        # Default controller parameters
        if controller_params is None:
            controller_params = {'stage1_power': 2000.0,
                                 'stage2_power': 4000.0,
                                 'deadband': 0.5}
            # controller_params = {
            #     'kp': 2000.0,
            #     'ki': 500.0,
            #     'dt': 0.25,
            #     'hvac_min': -4000.0,
            #     'hvac_max': 4000.0,
            #     'deadband': 0.5
            # }

        # Get simulation data
        df_sim = self.df.loc[sim_start:sim_end]
        enco_len = 48

        # Get initial buffer
        buffer = self.df.loc[:sim_start][-enco_len:]

        # Scale initial buffer data
        scaled_temp_room = self.scalers["temp"].transform(buffer[["temp_room"]].values)
        scaled_temp_amb = self.scalers["temp"].transform(buffer[["temp_amb"]].values)
        scaled_solar = self.scalers["solar"].transform(buffer[["solar"]].values)
        scaled_phvac = self.scalers["flux"].transform(buffer[["phvac"]].values)
        scaled_occ = self.scalers["occ"].transform(buffer[["occ"]].values)

        # Initialize encoder sequence
        encoder_sequence = np.hstack([
            scaled_temp_room,
            scaled_temp_amb,
            scaled_solar,
            buffer[["day_sin", "day_cos"]].values,
            scaled_occ,
            scaled_phvac
        ])


        # Main simulation loop
        for i in range(len(df_sim)):
            current_row = df_sim.iloc[i]
            current_time = df_sim.index[i]

            if sce == "groundtruth":
                setpt_cool = current_row['setpt_cool']
                setpt_heat = current_row['setpt_heat']
                active_setpoint = setpt_cool

            elif sce == "constant":
                setpt_cool = 24
                setpt_heat = 21
                active_setpoint = setpt_cool

            elif sce == "precool":
                setpt_cool = 24
                setpt_heat = 21
                if current_row['hours'] < 16 and current_row['hours'] >= 14:
                    setpt_cool = np.round(df_sim["setpt_cool"].min(), 1) - 0
                active_setpoint = setpt_cool

            occupied = self.identify_occupancy(df_sim, setpt_cool, setpt_heat)

            # Initialize/Update PI controller with current setpoint
            if self.controller is None or self.controller.setpoint != active_setpoint:
                self.controller = OnOffController(

                    setpoint=active_setpoint,
                    **controller_params
                )

            # Model prediction
            model_input = encoder_sequence.reshape(1, enco_len, -1)
            model_input = torch.from_numpy(model_input).to('cuda:1').float()

            predicted_temp_scaled, _, _ = self.dynamic(model_input)
            predicted_temp = self.scalers["temp"].inverse_transform(
                predicted_temp_scaled.cpu().detach().numpy().reshape(-1, 1)
            )[0, 0]

            # PI Controller update
            thermal_load, error, hvac_on = self.controller.update(predicted_temp)

            # Convert thermal load to electric load
            hvac_electric = self.thermal_to_electric_cop(
                thermal_load, predicted_temp, current_row['temp_amb']
            )

            # Calculate lighting load
            lighting_load = self.calculate_lighting_load(current_time, occupied)

            # Generate appliance loads
            appliance_loads = self.generate_appliance_loads(current_time, occupied)

            # Calculate total building load
            total_building_load = hvac_electric + lighting_load + appliance_loads['total']

            # Store results
            results['timestamp'].append(current_time)
            results['predicted_temp'].append(predicted_temp)
            results['actual_temp'].append(current_row['temp_room'])
            results['thermal_load'].append(thermal_load)
            results['hvac_electric'].append(hvac_electric)
            results['baseline_hvac'].append(current_row['phvac'])
            results['hvac_on'].append(hvac_on)
            results['control_error'].append(error)
            results['setpoint_cool'].append(setpt_cool)
            results['setpoint_heat'].append(setpt_heat)
            results['active_setpoint'].append(active_setpoint)
            results['occupied'].append(occupied)
            results['lighting_load'].append(lighting_load)
            results['appliance_total'].append(appliance_loads['total'])
            results['appliance_tv'].append(appliance_loads['tv'])
            results['appliance_computer'].append(appliance_loads['computer'])
            results['appliance_cooking'].append(appliance_loads['cooking'])
            results['appliance_washing'].append(appliance_loads['washing'])
            results['appliance_misc'].append(appliance_loads['misc'])
            results['total_building_load'].append(total_building_load)
            results['temp_error'].append(predicted_temp - current_row['temp_room'])

            # Update encoder sequence for next iteration
            next_temp_amb_scaled = self.scalers["temp"].transform([[current_row['temp_amb']]])[0, 0]
            next_solar_scaled = self.scalers["solar"].transform([[current_row['solar']]])[0, 0]
            next_phvac_scaled = self.scalers["flux"].transform([[thermal_load]])[0, 0]
            next_occ_scaled = self.scalers["occ"].transform([[current_row['occ']]])[0, 0]

            next_features = np.array([
                predicted_temp_scaled.cpu().detach().numpy().reshape(-1, 1)[0, 0],
                next_temp_amb_scaled,
                next_solar_scaled,
                current_row['day_sin'],
                current_row['day_cos'],
                next_occ_scaled,
                next_phvac_scaled
            ])

            encoder_sequence = np.vstack([
                encoder_sequence[1:],
                next_features.reshape(1, -1)
            ])

            # Progress update
            if (i + 1) % 96 == 0:
                print(f"Completed {i + 1}/{len(df_sim)} timesteps")

        # Convert to DataFrame
        self.results_df = pd.DataFrame(results)
        return self.results_df

    def plot_results(self, figsize=(4, 3.5), dpi=300):
        """
        Create comprehensive plots of simulation results.

        Args:
            figsize: Figure size tuple
            dpi: Figure DPI
        """
        if not hasattr(self, 'results_df'):
            raise ValueError("No simulation results found. Run simulation first.")

        plt.rcParams.update({'font.size': 7})
        fig, axs = plt.subplots(3, 1, figsize=figsize, dpi=dpi, sharex=True)

        # 1. Temperature Control
        axs[0].plot(self.results_df['timestamp'], self.results_df['actual_temp'],
                    label='Temp (Baseline)', linewidth=1.0, color='blue')
        axs[0].plot(self.results_df['timestamp'], self.results_df['predicted_temp'],
                    label='Temp (BESTOpt)', linewidth=1.0, color='red')
        axs[0].step(self.results_df['timestamp'], self.results_df['active_setpoint'],
                    label='Setpoint', linestyle='--', linewidth=1.0, color='gray')
        # axs[0].fill_between(self.results_df['timestamp'],
        #                     axs[0].get_ylim()[0], axs[0].get_ylim()[1],
        #                     where=self.results_df['occupied'], alpha=0.1, color='orange',
        #                     label='Occupied')
        axs[0].set_ylabel('Temp (°C)')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)

        # 2. HVAC Loads
        axs[1].plot(self.results_df['timestamp'], self.results_df['thermal_load']/1000,
                    linewidth=1.0, color='red', label='HVAC (BESTOpt)')
        axs[1].plot(self.results_df['timestamp'], self.results_df['baseline_hvac']/1000,
                    linewidth=1.0, color='blue', label='HVAC (Baseline)')
        axs[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        axs[1].set_ylabel('HVAC Power (kW)')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

        # 3. All Load
        axs[2].plot(self.results_df['timestamp'], self.results_df['total_building_load']/1000,
                    linewidth=1.5, color='black', label='Total Load')
        axs[2].fill_between(self.results_df['timestamp'], 0/1000,
                            self.results_df['lighting_load']/1000,
                            alpha=0.7, color='yellow', label='Lighting')
        axs[2].fill_between(self.results_df['timestamp'],
                            self.results_df['lighting_load']/1000,
                            self.results_df['lighting_load']/1000 + self.results_df['hvac_electric']/1000,
                            alpha=0.7, color='red', label='HVAC')
        axs[2].fill_between(self.results_df['timestamp'],
                            self.results_df['lighting_load']/1000 + self.results_df['hvac_electric']/1000,
                            self.results_df['total_building_load']/1000,
                            alpha=0.7, color='purple', label='Appliances')
        axs[2].set_ylabel('Elec Load (kW)')
        axs[2].set_xlabel('Time')
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False)

        # Hide x tick labels for all but the last subplot
        for ax in axs[:-1]:
            ax.tick_params(labelbottom=False)

        plt.tight_layout()
        plt.show()

    def plot_compare_results(self, baseline_results, precool_results, figsize=(4, 2.5), dpi=300):
        """
        Create comprehensive plots of simulation results with 4-hour xticks and shaded peak hours.

        Args:
            baseline_results: DataFrame with baseline simulation
            precool_results: DataFrame with pre-cooling simulation
            figsize: Figure size tuple
            dpi: Figure DPI
        """
        if not hasattr(self, 'results_df'):
            raise ValueError("No simulation results found. Run simulation first.")

        plt.rcParams.update({'font.size': 7})
        fig, axs = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)

        # 1. Temperature Control
        axs[0].plot(baseline_results['timestamp'], baseline_results['predicted_temp'],
                    label='Baseline', linewidth=1.0, color='blue')
        axs[0].plot(precool_results['timestamp'], precool_results['predicted_temp'],
                    label='Pre-Cooling', linewidth=1.0, color='red')
        axs[0].step(baseline_results['timestamp'], baseline_results['active_setpoint'],
                    linestyle='--', linewidth=0.8, where='post', color='blue')
        axs[0].step(precool_results['timestamp'], precool_results['active_setpoint'],
                    linestyle='--', linewidth=0.8, where='post', color='red')
        axs[0].set_ylabel('Temp (°C)')
        axs[0].set_ylim(20.5, 25.5)
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False)

        # 2. HVAC Loads
        axs[1].plot(baseline_results['timestamp'], baseline_results['thermal_load'] / 1000,
                    linewidth=1.0, color='blue', label='Baseline')
        axs[1].plot(precool_results['timestamp'], precool_results['thermal_load'] / 1000,
                    linewidth=1.0, color='red', label='Pre-Cooling')
        axs[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        axs[1].set_ylabel('HVAC Power (kW)')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend([],[], frameon=False)

        # 3. All Load
        # axs[2].plot(self.results_df['timestamp'], self.results_df['total_building_load'] / 1000,
        #             linewidth=1.5, color='black', label='Total Load')
        # axs[2].fill_between(self.results_df['timestamp'], 0,
        #                     self.results_df['lighting_load'] / 1000,
        #                     alpha=0.7, color='yellow', label='Lighting')
        # axs[2].fill_between(self.results_df['timestamp'],
        #                     self.results_df['lighting_load'] / 1000,
        #                     self.results_df['lighting_load'] / 1000 + self.results_df['hvac_electric'] / 1000,
        #                     alpha=0.7, color='red', label='HVAC')
        # axs[2].fill_between(self.results_df['timestamp'],
        #                     self.results_df['lighting_load'] / 1000 + self.results_df['hvac_electric'] / 1000,
        #                     self.results_df['total_building_load'] / 1000,
        #                     alpha=0.7, color='purple', label='Appliances')
        # axs[2].set_ylabel('Elec Load (kW)')
        # axs[2].set_xlabel('Time')
        # axs[2].grid(True, alpha=0.3)
        # axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False)

        # Format x-axis: every 4 hours
        start_time = self.results_df['timestamp'].iloc[0]
        end_time = self.results_df['timestamp'].iloc[-1]
        axs[1].set_xlim(start_time, end_time)

        locator = mdates.HourLocator(interval=4)
        formatter = mdates.DateFormatter('%H:%M')
        axs[1].xaxis.set_major_locator(locator)
        axs[1].xaxis.set_major_formatter(formatter)

        # Add peak hour shading (4 PM to 8 PM)
        peak_start = pd.to_datetime(start_time.date()) + pd.Timedelta(hours=16)
        peak_end = peak_start + pd.Timedelta(hours=4)
        while peak_start < end_time:
            for ax in axs:
                ax.axvspan(peak_start, peak_end, color='gray', alpha=0.15)
            peak_start += timedelta(days=1)
            peak_end += timedelta(days=1)

        # Hide x tick labels for all but the last subplot
        for ax in axs[:-1]:
            ax.tick_params(labelbottom=False)

        plt.tight_layout()
        plt.show()
    def calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics

        Returns:
            metrics: Dictionary of performance metrics
        """
        if not hasattr(self, 'results_df'):
            raise ValueError("No simulation results found. Run simulation first.")

        df = self.results_df

        # Temperature control metrics
        mae_temp = np.mean(np.abs(df['temp_error']))
        rmse_temp = np.sqrt(np.mean(df['temp_error'] ** 2))
        mae_control = np.mean(np.abs(df['control_error']))
        rmse_control = np.sqrt(np.mean(df['control_error'] ** 2))

        # Energy metrics (convert W to kWh with 0.25h timestep)
        dt = 0.25
        total_hvac_energy = np.sum(df['hvac_electric']) * dt / 1000
        total_lighting_energy = np.sum(df['lighting_load']) * dt / 1000
        total_appliance_energy = np.sum(df['appliance_total']) * dt / 1000
        total_building_energy = np.sum(df['total_building_load']) * dt / 1000

        # Baseline comparison
        baseline_thermal_energy = np.sum(np.abs(df['baseline_hvac'])) * dt / 1000

        # Occupancy metrics
        occupied_hours = np.sum(df['occupied']) * dt
        total_hours = len(df) * dt
        occupancy_rate = occupied_hours / total_hours

        # HVAC operation metrics
        hvac_on_time = np.sum(df['hvac_on']) * dt
        hvac_duty_cycle = np.mean(df['hvac_on'])

        metrics = {
            'temperature_control': {
                'mae_temp_prediction': mae_temp,
                'rmse_temp_prediction': rmse_temp,
                'mae_control_error': mae_control,
                'rmse_control_error': rmse_control,
                'temp_range': [np.min(df['predicted_temp']), np.max(df['predicted_temp'])]
            },
            'energy_consumption': {
                'total_building_energy_kwh': total_building_energy,
                'hvac_energy_kwh': total_hvac_energy,
                'lighting_energy_kwh': total_lighting_energy,
                'appliance_energy_kwh': total_appliance_energy,
                'baseline_thermal_energy_kwh': baseline_thermal_energy,
                'energy_breakdown_percent': {
                    'hvac': total_hvac_energy / total_building_energy * 100,
                    'lighting': total_lighting_energy / total_building_energy * 100,
                    'appliances': total_appliance_energy / total_building_energy * 100
                }
            },
            'operation_metrics': {
                'occupancy_rate': occupancy_rate,
                'occupied_hours': occupied_hours,
                'total_hours': total_hours,
                'hvac_on_hours': hvac_on_time,
                'hvac_duty_cycle_percent': hvac_duty_cycle * 100,
                'avg_hvac_power_w': np.mean(df['hvac_electric']),
                'peak_building_load_w': np.max(df['total_building_load'])
            }
        }

        return metrics

    def print_summary(self):
        """Print a comprehensive summary of simulation results"""
        metrics = self.calculate_metrics()

        print("\n" + "=" * 60)
        print("           HVAC BUILDING SIMULATION SUMMARY")
        print("=" * 60)

        print(f"\n→ TEMPERATURE CONTROL PERFORMANCE:")
        temp_metrics = metrics['temperature_control']
        print(f"   Temperature Prediction MAE: {temp_metrics['mae_temp_prediction']:.3f} °C")
        print(f"   Temperature Prediction RMSE: {temp_metrics['rmse_temp_prediction']:.3f} °C")
        print(f"   Control Error MAE: {temp_metrics['mae_control_error']:.3f} °C")
        print(f"   Control Error RMSE: {temp_metrics['rmse_control_error']:.3f} °C")
        print(f"   Temperature Range: {temp_metrics['temp_range'][0]:.1f} °C - {temp_metrics['temp_range'][1]:.1f} °C")

        print(f"\n→ ENERGY CONSUMPTION BREAKDOWN:")
        energy_metrics = metrics['energy_consumption']
        print(f"   Total Building Energy: {energy_metrics['total_building_energy_kwh']:.2f} kWh")
        print(f"   HVAC Energy: {energy_metrics['hvac_energy_kwh']:.2f} kWh "
              f"({energy_metrics['energy_breakdown_percent']['hvac']:.1f}%)")
        print(f"   Lighting Energy: {energy_metrics['lighting_energy_kwh']:.2f} kWh "
              f"({energy_metrics['energy_breakdown_percent']['lighting']:.1f}%)")
        print(f"   Appliance Energy: {energy_metrics['appliance_energy_kwh']:.2f} kWh "
              f"({energy_metrics['energy_breakdown_percent']['appliances']:.1f}%)")
        print(f"   Baseline Thermal Energy: {energy_metrics['baseline_thermal_energy_kwh']:.2f} kWh")

        print(f"\n→ BUILDING OPERATION METRICS:")
        op_metrics = metrics['operation_metrics']
        print(f"   Occupancy Rate: {op_metrics['occupancy_rate']:.1f}%")
        print(f"   Occupied Hours: {op_metrics['occupied_hours']:.1f} / {op_metrics['total_hours']:.1f} hours")
        print(f"   HVAC On Time: {op_metrics['hvac_on_hours']:.1f} hours")
        print(f"   HVAC Duty Cycle: {op_metrics['hvac_duty_cycle_percent']:.1f}%")
        print(f"   Average HVAC Power: {op_metrics['avg_hvac_power_w']:.1f} W")
        print(f"   Peak Building Load: {op_metrics['peak_building_load_w']:.1f} W")

        print("=" * 60)


class PIController:
    """PI Controller for HVAC system with dynamic setpoints"""

    def __init__(self, kp=1.0, ki=0.1, setpoint=21.0, dt=0.25,
                 hvac_min=-5000.0, hvac_max=5000.0, deadband=0.5):
        self.kp = kp
        self.ki = ki
        self.setpoint = setpoint
        self.dt = dt
        self.hvac_min = hvac_min
        self.hvac_max = hvac_max
        self.deadband = deadband

        self.integral = 0.0
        self.previous_error = 0.0
        self.hvac_on = False

    def update(self, current_temp):
        """Update controller and return HVAC thermal power"""
        error = current_temp - self.setpoint
        error = np.clip(error, 0, None)

        self.integral += error * self.dt
        pi_output = self.kp * error + self.ki * self.integral

        # On-off control with deadband
        if not self.hvac_on:
            if abs(error) > self.deadband:
                self.hvac_on = True
        else:
            if abs(error) <= self.deadband:
                self.hvac_on = False

        # Determine HVAC power
        if self.hvac_on:
            if pi_output > 0:  # Cooling needed
                hvac_power = -np.clip(abs(pi_output) * 100, 0, abs(self.hvac_min))
            elif pi_output < 0:  # Heating needed
                hvac_power = np.clip(abs(pi_output) * 100, 0, self.hvac_max)
            else:
                hvac_power = 0.0
        else:
            hvac_power = 0.0

        return hvac_power, error, self.hvac_on

class OnOffController:
    """
    Two-Stage On-Off Controller for HVAC system with deadband (cooling season only).
    Stage 1 triggers when temperature is 0.5°C above setpoint.
    Stage 2 triggers when temperature is >0.8°C above setpoint.
    """
    def __init__(self, setpoint=21.0, deadband=0.5,
                 stage1_power=2000.0, stage2_power=4000.0):
        self.setpoint = setpoint
        self.deadband = deadband  # Typically 0.5°C
        self.stage1_power = stage1_power  # 2kW
        self.stage2_power = stage2_power  # 4kW
        self.current_stage = 0  # 0: off, 1: stage1, 2: stage2
        self.previous_stage = 0

    def update(self, current_temp):
        """
        Update controller and return HVAC thermal power (cooling season only)
        """
        error = current_temp - self.setpoint
        self.previous_stage = self.current_stage

        # Cooling logic only (no heating)
        if error <= 0:
            # At or below setpoint, turn off
            self.current_stage = 0
        elif error <= self.deadband:
            # Within deadband (0 ~ 0.5°C)
            if self.previous_stage == 0:
                self.current_stage = 0
            else:
                self.current_stage = self.previous_stage  # Hold stage
        elif error <= 0.8:
            # Between 0.5 ~ 0.8°C
            self.current_stage = 1
        else:
            # error > 0.8°C
            self.current_stage = 2

        # Determine HVAC power (negative for cooling)
        if self.current_stage == 0:
            hvac_power = 0.0
        elif self.current_stage == 1:
            hvac_power = -self.stage1_power  # -2kW
        else:
            hvac_power = -self.stage2_power  # -4kW

        hvac_on = self.current_stage > 0
        return hvac_power, error, hvac_on

# Example usage:
if __name__ == "__main__":
    # Model parameters
    model_params = {
        "para": {"Int_h": 14, "Ext_h": 12, "epochs": 30},
        "modeltype": "PI-modnn",
        "startday": 30,
        "trainday": 150,
        "testday": 1,
        "datapath": "../dataset/dataset_1.csv"
    }

    # Initialize simulator
    simulator = HVACBuildingSimulator(
        model_args=model_params,
        scaler_path="../Scaler/Eplus/ModNN_scaler.pkl",
        data_path="../dataset/dataset_1.csv"
    )

    # Run simulation
    baseline_results = simulator.run_simulation('2023-07-01', '2023-07-01', "constant")
    precool_results = simulator.run_simulation('2023-07-01', '2023-07-01', "precool")

    # Plot results
    simulator.plot_compare_results(baseline_results=baseline_results,
                                   precool_results=precool_results)

    # Print summary
    # simulator.print_summary()