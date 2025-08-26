import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class OccConfig:
    def __init__(self):
        """
        Initialize the occupancy model with three states:
        0: Home, 1: Work, 2: Other
        """
        self.states = {'home': 0, 'work': 1, 'other': 2}
        self.state_names = ['Home', 'Work', 'Other']
        self.colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange

        # Initialize transition matrices for different time periods
        self.transition_matrices = self._create_transition_matrices()

    def _create_transition_matrices(self):
        """Create time-dependent transition matrices (96 matrices for 24 hours)"""
        matrices = []

        for timestep in range(96):  # 96 steps = 24 hours with 15-min resolution
            hour = (timestep * 15) // 60  # Convert to hour of day

            if 0 <= hour < 6:  # Night (12 AM - 6 AM): stay home
                matrix = np.array([
                    [0.98, 0.01, 0.01],  # From Home
                    [0.80, 0.18, 0.02],  # From Work (night shift workers)
                    [0.90, 0.05, 0.05]   # From Other
                ])
            elif 6 <= hour < 8:  # Early morning (6 AM - 8 AM): Gradual transition to work
                matrix = np.array([
                    [0.85, 0.12, 0.03],  # From Home - gradual departure
                    [0.05, 0.92, 0.03],  # From Work - stay at work
                    [0.60, 0.25, 0.15]   # From Other - go to work or home
                ])
            elif 8 <= hour < 12:  # Morning work (8 AM - 12 PM): working period
                matrix = np.array([
                    [0.20, 0.75, 0.05],  # From Home - go to work
                    [0.02, 0.95, 0.03],  # From Work
                    [0.10, 0.80, 0.10]   # From Other - go to work
                ])
            elif 12 <= hour < 14:  # Lunch time (12 PM - 2 PM): Some movement
                matrix = np.array([
                    [0.70, 0.20, 0.10],  # From Home
                    [0.05, 0.80, 0.15],  # From Work - some leave for lunch
                    [0.20, 0.30, 0.50]   # From Other
                ])
            elif 14 <= hour < 17:  # Afternoon work (2 PM - 5 PM): Back to work
                matrix = np.array([
                    [0.30, 0.65, 0.05],  # From Home - go to work
                    [0.02, 0.95, 0.03],  # From Work
                    [0.15, 0.75, 0.10]   # From Other - back to work
                ])
            elif 17 <= hour < 19:  # Evening (5 PM - 7 PM): Gradual transition home
                matrix = np.array([
                    [0.90, 0.05, 0.05],  # From Home - stay home
                    [0.40, 0.50, 0.10],  # From Work - gradual departure
                    [0.60, 0.05, 0.35]   # From Other - go home
                ])
            else:  # Night (7 PM - 12 AM): evening at home
                matrix = np.array([
                    [0.92, 0.03, 0.05],  # From Home
                    [0.70, 0.15, 0.15],  # From Work - go home
                    [0.75, 0.05, 0.20]   # From Other - go home
                ])

            matrices.append(matrix)

        return matrices

    def get_current_state(self, timestep, trajectory=None):
        """
        Get the current state at a given timestep

        Args:
            timestep (int): Timestep (0-95 for 24 hours)
            trajectory (list): Generated trajectory

        Returns:
            int: Current state (0=Home, 1=Work, 2=Other)
        """
        if 0 <= timestep < len(trajectory):
            return trajectory[timestep]
        else:
            raise ValueError(f"Timestep {timestep} is out of range for trajectory of length {len(trajectory)}")


    def generate_trajectory(self, initial_state=None, steps=96, min_duration=4):
        """
        Generate a trajectory for the next 24 hours (96 steps)

        Args:
            initial_state (int): Starting state (if None, will be determined by current time)
            steps (int): Number of steps to generate
            min_duration (int): Minimum duration to stay in a state (in timesteps)

        Returns:
            list: Trajectory of states
        """
        if initial_state is None:
            initial_state = self.get_current_state(0)

        trajectory = [initial_state]
        current_state = initial_state
        time_in_current_state = 1

        for step in range(1, steps):
            # Get transition probabilities for current time
            transition_matrix = self.transition_matrices[step % 96]
            probabilities = transition_matrix[current_state]

            # Apply stability constraint
            if time_in_current_state < min_duration:
                # Force staying in current state if haven't been there long enough
                next_state = current_state
                time_in_current_state += 1
            else:
                next_state = np.random.choice(3, p=probabilities)
                if next_state == current_state:
                    time_in_current_state += 1
                else:
                    time_in_current_state = 1

            trajectory.append(next_state)
            current_state = next_state

        return trajectory

    def to_binary(self, trajectory, target_state):
        """
        Convert trajectory to binary representation

        Args:
            trajectory (list): State trajectory
            target_state (str or int): Target state ('home', 'work', 'other' or 0, 1, 2)

        Returns:
            list: Binary representation (1 if at target state, 0 otherwise)
        """
        if isinstance(target_state, str):
            target_state = self.states[target_state.lower()]

        return [1 if state == target_state else 0 for state in trajectory]

    def generate_multiple_days(self, num_days=30, min_duration=4):
        """
        Generate trajectories for multiple days

        Args:
            num_days (int): Number of days to simulate
            min_duration (int): Minimum duration to stay in a state

        Returns:
            list: List of daily trajectories
        """
        trajectories = []

        for day in range(num_days):
            # Start each day at home (midnight)
            trajectory = self.generate_trajectory(initial_state=0, steps=96, min_duration=min_duration)
            trajectories.append(trajectory)

        return trajectories

    def generate_appliance_and_lighting_usage(self, trajectory):
        np.random.seed(42)  # consistent results
        steps = len(trajectory)
        usage = {
            'lighting': np.zeros(steps, dtype=int),
            'cooking': np.zeros(steps, dtype=int),
            'dishwashing': np.zeros(steps, dtype=int),
            'laundry': np.zeros(steps, dtype=int),
            'TV': np.zeros(steps, dtype=int)
        }

        # Flags for once-a-day activities
        dishwashing_done = False
        laundry_done = False
        tv_session_started = False

        for t in range(steps):
            state = trajectory[t]
            hour = (t * 15) // 60

            if state == self.states['home']:
                # Lighting
                if hour >= 18:
                    usage['lighting'][t] = np.random.rand() < 0.9

                # Cooking (breakfast, lunch, dinner)
                if hour in [7,8, 12,13,  18, 19]:
                    usage['cooking'][t] = np.random.rand() < 0.7

                # Dishwashing: one meal cleanup per day
                if not dishwashing_done and hour in [8, 13, 20]:
                    if np.random.rand() < 0.5:
                        usage['dishwashing'][t:t + 2] = 1  # assume 30 mins
                        dishwashing_done = True

                # Laundry: once per day, 1-hour block
                if not laundry_done and hour in [10, 11, 18, 19]:
                    if np.random.rand() < 0.3:
                        usage['laundry'][t:t + 4] = 1  # assume 1 hour = 4 steps
                        laundry_done = True

                # TV/computer: one evening session, continuous block
                if not tv_session_started and 19 <= hour <= 22:
                    if np.random.rand() < 0.6:
                        session_len = np.random.randint(8, 16)  # 2–4 hours
                        usage['TV'][t:t + session_len] = 1
                        tv_session_started = True

            elif state in [self.states['work'], self.states['other']]:
                # Basic lighting if dark
                if hour < 6 or hour >= 18:
                    usage['lighting'][t] = np.random.rand() < 0.2

        return usage

    def plot_single_day(self, trajectory=None, title="Daily Mobility Pattern", show_transitions=True):
        """
        Plot a single day's mobility pattern with transition analysis

        Args:
            trajectory (list): Trajectory to plot (if None, generates new one)
            title (str): Plot title
            show_transitions (bool): Whether to show transition statistics
        """
        if trajectory is None:
            trajectory = self.generate_trajectory(min_duration=4)

        # Create time labels
        time_labels = [f"{(i * 15) // 60:02d}:{(i * 15) % 60:02d}" for i in range(96)]

        # Calculate transition statistics
        transitions = 0
        state_durations = []
        current_state = trajectory[0]
        duration = 1

        for i in range(1, len(trajectory)):
            if trajectory[i] != current_state:
                transitions += 1
                state_durations.append(duration)
                current_state = trajectory[i]
                duration = 1
            else:
                duration += 1
        state_durations.append(duration)

        avg_duration = np.mean(state_durations) * 15  # Convert to minutes

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Plot 1: State over time
        ax1.plot(range(96), trajectory, 'o-', linewidth=2, markersize=3)
        ax1.set_ylabel('State')
        ax1.set_title(f'{title} - State Transitions')
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(self.state_names)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 95)

        # Add transition statistics
        if show_transitions:
            ax1.text(0.02, 0.98, f'Transitions: {transitions}\nAvg Duration: {avg_duration:.1f} min',
                     transform=ax1.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Set x-axis labels for every 4 hours
        tick_positions = list(range(0, 96, 16))  # Every 4 hours
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels([time_labels[i] for i in tick_positions])

        # Plot 2: Colored timeline
        colors = [self.colors[state] for state in trajectory]
        ax2.scatter(range(96), [1] * 96, c=colors, s=50, alpha=0.8)
        ax2.set_ylabel('Timeline')
        ax2.set_xlabel('Time of Day')
        ax2.set_title('Timeline View')
        ax2.set_ylim(0.5, 1.5)
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 95)

        # Set x-axis labels
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([time_labels[i] for i in tick_positions])

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=self.colors[i], label=self.state_names[i])
                           for i in range(3)]
        ax2.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.show()

    def plot_distribution(self, num_days=30, title="Occupancy Distribution"):
        """
        Plot distribution of states across multiple days

        Args:
            num_days (int): Number of days to simulate
            title (str): Plot title
        """
        trajectories = self.generate_multiple_days(num_days)

        # Flatten all trajectories
        all_states = [state for trajectory in trajectories for state in trajectory]

        # Calculate hourly distributions
        hourly_dist = np.zeros((24, 3))
        for day_traj in trajectories:
            for i, state in enumerate(day_traj):
                hour = (i * 15) // 60
                hourly_dist[hour, state] += 1

        # Normalize to probabilities
        hourly_dist = hourly_dist / num_days

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Overall distribution
        state_counts = Counter(all_states)
        states = [self.state_names[i] for i in range(3)]
        counts = [state_counts[i] for i in range(3)]

        ax1.bar(states, counts, color=self.colors)
        ax1.set_title(f'Overall State Distribution ({num_days} days)')
        ax1.set_ylabel('Total Count')

        # Plot 2: Hourly distribution (stacked)
        hours = list(range(24))
        ax2.bar(hours, hourly_dist[:, 0], color=self.colors[0], label='Home', alpha=0.8)
        ax2.bar(hours, hourly_dist[:, 1], bottom=hourly_dist[:, 0],
                color=self.colors[1], label='Work', alpha=0.8)
        ax2.bar(hours, hourly_dist[:, 2],
                bottom=hourly_dist[:, 0] + hourly_dist[:, 1],
                color=self.colors[2], label='Other', alpha=0.8)

        ax2.set_title('Hourly State Distribution')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Average Occupancy')
        ax2.legend()
        ax2.set_xticks(range(0, 24, 4))

        # Plot 3: Heatmap of hourly distributions
        im = ax3.imshow(hourly_dist.T, cmap='YlOrRd', aspect='auto')
        ax3.set_title('Hourly State Probability Heatmap')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('State')
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(self.state_names)
        ax3.set_xticks(range(0, 24, 4))
        plt.colorbar(im, ax=ax3, label='Probability')

        # Plot 4: Binary occupancy for home
        home_binary = []
        for trajectory in trajectories:
            home_binary.extend(self.to_binary(trajectory, 'home'))

        hourly_home = np.zeros(24)
        for i, is_home in enumerate(home_binary):
            hour = ((i % 96) * 15) // 60
            hourly_home[hour] += is_home

        hourly_home = hourly_home / num_days

        ax4.plot(hours, hourly_home, 'o-', color=self.colors[0], linewidth=2)
        ax4.set_title('Home Occupancy Probability by Hour')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Probability of Being Home')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(range(0, 24, 4))
        ax4.set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

    def plot_hourly_probability_distribution(self, num_days=30):
        """
        Plot hourly distribution as probability percentages (0-100%)

        Args:
            num_days (int): Number of days to simulate
            title (str): Plot title
        """
        trajectories = self.generate_multiple_days(num_days)

        # Calculate hourly distributions
        hourly_counts = np.zeros((24, 3))

        for day_traj in trajectories:
            for timestep, state in enumerate(day_traj):
                hour = (timestep * 15) // 60
                if hour < 24:
                    hourly_counts[hour, state] += 1

        # Convert to probabilities (percentage of time spent in each state per hour)
        timesteps_per_hour = 4
        total_timesteps_per_hour = num_days * timesteps_per_hour
        hourly_probs = (hourly_counts / total_timesteps_per_hour) * 100  # Convert to percentages

        # Create the plot
        fig, ax = plt.subplots(figsize=(3,2), dpi=300)

        hours = np.arange(24)
        width = 0.8

        # Create stacked bar chart
        bottom_work = hourly_probs[:, 0]
        bottom_other = hourly_probs[:, 0] + hourly_probs[:, 1]

        # Plot bars
        bars1 = ax.bar(hours, hourly_probs[:, 0], width,
                       label='Home', color=self.colors[0], alpha=0.8)
        bars2 = ax.bar(hours, hourly_probs[:, 1], width, bottom=bottom_work,
                       label='Work', color=self.colors[1], alpha=0.8)
        bars3 = ax.bar(hours, hourly_probs[:, 2], width, bottom=bottom_other,
                       label='Other', color=self.colors[2], alpha=0.8)

        # Customize the plot
        ax.set_xlabel('Hour of Day', fontsize=7)
        ax.set_ylabel('Probability (%)', fontsize=7)
        ax.set_xticks(np.arange(0, 24, 4))
        ax.set_xticklabels([f'{h}' for h in range(0, 24, 4)])
        ax.tick_params(labelsize=7)
        ax.legend(ncol = 3, loc='upper center',
                  bbox_to_anchor=(0.5, 1.15), fontsize=7, frameon=False)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.show()

    def plot_setpoint(self, data):
        fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
        hours = np.arange(96) / 4  # match 15-min resolution

        occ_status = np.array(data)
        coolsetpt = 24 * occ_status + 28 * (1 - occ_status)
        heatsetpt = 21 * occ_status + 17 * (1 - occ_status)

        ax.plot(hours, coolsetpt, label='coolsetpt', color="gray", linewidth=2)
        ax.plot(hours, heatsetpt, label='heatsetpt', color="gray", linewidth=2)

        ax.set_xlabel('Hour of Day', fontsize=7)
        ax.set_ylabel('Setpoint (°C)', fontsize=7)
        ax.set_xticks(np.arange(0, 25, 4))
        ax.set_xticklabels([str(h) for h in range(0, 25, 4)])
        ax.tick_params(labelsize=7)
        ax.legend(ncol=2, loc='upper center',
                  bbox_to_anchor=(0.5, 1.15), fontsize=7, frameon=False)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

    def plot_appliance_lighting_usage(self, usage_dict):
        """
        Plot all appliance and lighting usage timelines.

        Args:
            usage_dict (dict): Output from generate_appliance_and_lighting_usage()
        """
        fig, ax = plt.subplots(figsize=(2.7, 2), dpi=300)
        hours = np.arange(len(next(iter(usage_dict.values())))) / 4  # 15-min resolution

        for i, (appliance, usage) in enumerate(usage_dict.items()):
            ax.plot(hours, usage + i * 1.5, label=appliance, drawstyle='steps-post')

        ax.tick_params(labelsize=7)
        ax.set_yticks([])
        ax.set_xlabel('Hour of Day', fontsize=7)
        ax.set_xticks(np.arange(0, 25, 4))
        ax.set_xticklabels([str(h) for h in range(0, 25, 4)])
        ax.legend(ncol=3, loc='upper center',
                  bbox_to_anchor=(0.5, 1.3), fontsize=6.5, frameon=False)
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create model
    model = OccConfig()

    # Example 1: Generate trajectory first
    trajectory = model.generate_trajectory(initial_state=0, steps=96, min_duration=4)

    # Example 2: Get current state at 7 AM (timestep 28) from generated trajectory
    current_state = model.get_current_state(28, trajectory)
    print(f"Actual state at 7 AM (timestep 28): {model.state_names[current_state]}")

    # Example 4: Sample trajectory (first 20 steps)
    print(f"Sample trajectory (first 20 steps): {trajectory[:20]}")

    # Example 5: Convert to binary for home occupancy
    home_binary = model.to_binary(trajectory, 'home')
    # model.plot_setpoint(home_binary)
    # print(f"Home occupancy (first 20 steps): {home_binary[:20]}")
    #
    # # Example 6: Plot single day with transition analysis
    # model.plot_single_day(trajectory, show_transitions=True)
    #
    # # Example 7: Plot distribution across multiple days
    # model.plot_distribution(num_days=30)
    # model.plot_hourly_probability_distribution(num_days=30)

    usage = model.generate_appliance_and_lighting_usage(trajectory)
    model.plot_appliance_lighting_usage(usage)