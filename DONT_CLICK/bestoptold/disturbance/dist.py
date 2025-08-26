from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from ..core.base_module import BaseModule, DisturbanceVariable


class DisturbanceModule(BaseModule):
    """External disturbance module for weather and utility pricing"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_source = config.get('data_source', None)
        self.current_index = 0

        # Load disturbance data if provided
        if self.data_source:
            self.data = self._load_data(self.data_source)
        else:
            self.data = None

    def _initialize_variables(self):
        """Initialize disturbance variables"""
        self.disturbances['outdoor_temperature'] = DisturbanceVariable(
            name='outdoor_temperature',
            value=20.0,
            unit='°C'
        )

        self.disturbances['solar_radiation'] = DisturbanceVariable(
            name='solar_radiation',
            value=0.0,
            unit='W/m²'
        )

        self.disturbances['electricity_price'] = DisturbanceVariable(
            name='electricity_price',
            value=0.10,
            unit='$/kWh'
        )

        self.disturbances['wind_speed'] = DisturbanceVariable(
            name='wind_speed',
            value=2.0,
            unit='m/s'
        )

    def _load_data(self, source: str) -> pd.DataFrame:
        """Load disturbance data from file"""
        if isinstance(source, str) and source.endswith('.csv'):
            return pd.read_csv(source, parse_dates=['timestamp'])
        else:
            # Generate synthetic data
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic weather and price data"""
        timestamps = pd.date_range(start='2024-01-01', periods=365 * 24 * 4, freq='15min')
        data = []

        for i, ts in enumerate(timestamps):
            hour = ts.hour
            day_of_year = ts.dayofyear

            # Temperature (sinusoidal daily and yearly variation)
            daily_temp = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
            yearly_temp = 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            temp = 20 + daily_temp + yearly_temp + np.random.normal(0, 1)

            # Solar radiation
            if 6 <= hour <= 18:
                solar_angle = np.sin(np.pi * (hour - 6) / 12)
                solar = 800 * solar_angle * (1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365))
                solar = max(0, solar + np.random.normal(0, 50))
            else:
                solar = 0

            # Electricity price (TOU with peak hours)
            if 14 <= hour <= 20:  # Peak hours
                price = 0.15 + 0.05 * np.random.random()
            elif 7 <= hour <= 14 or 20 <= hour <= 22:  # Mid-peak
                price = 0.10 + 0.02 * np.random.random()
            else:  # Off-peak
                price = 0.06 + 0.02 * np.random.random()

            # Wind speed
            wind = 3 + 2 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 0.5)
            wind = max(0, wind)

            data.append({
                'timestamp': ts,
                'outdoor_temperature': temp,
                'solar_radiation': solar,
                'electricity_price': price,
                'wind_speed': wind
            })

        return pd.DataFrame(data)

    def update(self,
               dt: float,
               signal_bus: Dict[str, Any],
               thermal_bus: Dict[str, Any],
               electric_bus: Dict[str, Any]):
        """Update disturbance values"""
        if self.data is not None:
            # Use data from file/synthetic
            current_data = self.data.iloc[self.current_index]

            self.disturbances['outdoor_temperature'].value = current_data['outdoor_temperature']
            self.disturbances['solar_radiation'].value = current_data['solar_radiation']
            self.disturbances['electricity_price'].value = current_data['electricity_price']
            self.disturbances['wind_speed'].value = current_data['wind_speed']

            self.current_index = (self.current_index + 1) % len(self.data)

    def get_disturbance(self, name: str) -> float:
        """Get current disturbance value"""
        return self.disturbances[name].value if name in self.disturbances else None

    def predict(self, name: str, horizon: int) -> np.ndarray:
        """Predict future disturbance values"""
        if self.data is None:
            return np.array([self.disturbances[name].value] * horizon)

        # Return future values from data
        future_indices = [(self.current_index + i) % len(self.data) for i in range(horizon)]
        return self.data.iloc[future_indices][name].values