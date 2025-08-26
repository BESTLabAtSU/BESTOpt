from typing import Dict, Any
import numpy as np
import pandas as pd
from ..core.base_module import BaseModule, StateVariable, DisturbanceVariable


class OccupancyModule(BaseModule):
    """Occupancy module with mobility, comfort, and activity attributes"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_source = config.get('data_source', None)
        self.current_index = 0

        # Load occupancy schedule if provided
        if self.data_source:
            self.schedule = self._load_schedule(self.data_source)
        else:
            self.schedule = None

    def _initialize_variables(self):
        """Initialize occupancy-related variables"""
        # States
        self.states['occupancy_status'] = StateVariable(
            name='occupancy_status',
            value=0.0,  # 0 = vacant, 1 = occupied
            unit='binary',
            min_val=0.0,
            max_val=1.0
        )

        self.states['comfort_preference'] = StateVariable(
            name='comfort_preference',
            value=22.0,  # Default comfort temperature
            unit='°C',
            min_val=18.0,
            max_val=26.0
        )

        # Activity states affecting appliance usage
        self.states['activity_cooking'] = StateVariable(
            name='activity_cooking',
            value=0.0,
            unit='binary',
            min_val=0.0,
            max_val=1.0
        )

        self.states['activity_entertainment'] = StateVariable(
            name='activity_entertainment',
            value=0.0,
            unit='binary',
            min_val=0.0,
            max_val=1.0
        )

        self.states['activity_work'] = StateVariable(
            name='activity_work',
            value=0.0,
            unit='binary',
            min_val=0.0,
            max_val=1.0
        )

        # Appliance power consumption
        self.appliance_power = {
            'lighting': 200,  # W
            'tv': 150,
            'computer': 300,
            'cooking': 2000,
            'other': 100
        }

    def _load_schedule(self, source: str) -> pd.DataFrame:
        """Load occupancy schedule from file or generate synthetic"""
        if isinstance(source, str) and source.endswith('.csv'):
            return pd.read_csv(source, parse_dates=['timestamp'])
        else:
            # Generate synthetic schedule
            return self._generate_synthetic_schedule()

    def _generate_synthetic_schedule(self) -> pd.DataFrame:
        """Generate synthetic occupancy patterns"""
        # Simple weekday/weekend pattern
        timestamps = pd.date_range(start='2024-01-01', periods=365 * 24 * 4, freq='15min')
        data = []

        for ts in timestamps:
            hour = ts.hour
            is_weekend = ts.dayofweek >= 5

            # Occupancy probability
            if is_weekend:
                occ_prob = 0.8 if 8 <= hour <= 22 else 0.9
            else:
                occ_prob = 0.3 if 9 <= hour <= 17 else 0.8

            occupancy = 1 if np.random.random() < occ_prob else 0

            # Activities based on time and occupancy
            cooking = 1 if occupancy and hour in [7, 12, 18, 19] else 0
            entertainment = 1 if occupancy and 19 <= hour <= 22 else 0
            work = 1 if occupancy and not is_weekend and 9 <= hour <= 17 else 0

            # Comfort preferences (slight variations)
            comfort = 22 + np.random.normal(0, 0.5)

            data.append({
                'timestamp': ts,
                'occupancy': occupancy,
                'cooking': cooking,
                'entertainment': entertainment,
                'work': work,
                'comfort_temp': comfort
            })

        return pd.DataFrame(data)

    def update(self, dt: float, signal_bus: Dict[str, Any],
               thermal_bus: Dict[str, Any], electric_bus: Dict[str, Any]):
        """Update occupancy state"""
        if self.schedule is not None:
            # Use schedule data
            current_data = self.schedule.iloc[self.current_index]

            self.states['occupancy_status'].update(current_data['occupancy'])
            self.states['activity_cooking'].update(current_data['cooking'])
            self.states['activity_entertainment'].update(current_data['entertainment'])
            self.states['activity_work'].update(current_data['work'])
            self.states['comfort_preference'].update(current_data['comfort_temp'])

            self.current_index = (self.current_index + 1) % len(self.schedule)
        else:
            # Use stochastic model or external input
            pass

    def get_occupancy_status(self) -> float:
        """Get current occupancy status"""
        return self.states['occupancy_status'].value

    def get_comfort_setpoint(self) -> float:
        """Get temperature setpoint based on occupancy and comfort preference"""
        if self.states['occupancy_status'].value > 0:
            return self.states['comfort_preference'].value
        else:
            # Setback temperature when vacant
            return self.config.get('vacant_setpoint', 25.0)

    def get_appliance_load(self) -> float:
        """Calculate total appliance load based on activities"""
        load = 0.0

        if self.states['occupancy_status'].value > 0:
            # Base lighting when occupied
            load += self.appliance_power['lighting']

            # Activity-based loads
            if self.states['activity_cooking'].value > 0:
                load += self.appliance_power['cooking']
            if self.states['activity_entertainment'].value > 0:
                load += self.appliance_power['tv']
            if self.states['activity_work'].value > 0:
                load += self.appliance_power['computer']

            # Other miscellaneous loads
            load += self.appliance_power['other']

        return load / 1000.0