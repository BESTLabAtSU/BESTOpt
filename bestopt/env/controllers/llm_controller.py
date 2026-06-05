"""
LLM-based Building Control System
Uses OpenAI GPT-4 to optimize HVAC control for energy efficiency while maintaining comfort
"""
import json
from typing import Tuple, List, Dict, Any, Optional
from collections import deque
import openai
import logging
import numpy as np
from datetime import datetime
import time
from dataclasses import dataclass, asdict


@dataclass
class ControllerMetrics:
    """Metrics for comparing different prompt levels."""
    level: int
    timestamp: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time_seconds: float
    flow_rate: float
    supply_temp: float
    reasoning: Dict
    prompt_length: int
    success: bool
    error_message: Optional[str] = None


class LLMThermalController:
    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.7):
        """
        Initialize the multi-level LLM controller.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o)
            temperature: LLM temperature parameter for response variability
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

        # Store metrics for each level
        self.metrics_history = {0: [], 1: [], 2: [], 3: [], 4: []}

        # Physical constraints for HVAC system
        self.constraints = {
            'supply_air_flow_rate': {
                'min': 0.0,  # m³/s
                'max': 2.0,  # Maximum air flow rate
            },
            'supply_air_temperature': {
                'min': 12.0,  # °C - Minimum supply air temp (cooling)
                'max': 35.0,  # °C - Maximum supply air temp (heating)
            }
        }

        # Conversion factors
        self.cp_air = 1.006  # kJ/kg·K - Specific heat of air
        self.rho_air = 1.2  # kg/m³ - Density of air

        # Define prompt components
        self._define_prompt_components()

    def _define_prompt_components(self):
        """Define reusable prompt components."""

        self.SYSTEM_PROMPT = """You are an expert HVAC controller to make optimal HVAC control decisions."""

        self.CURRENT_STATE = """
CURRENT STATE:
- Zone Temperature: {current_temp:.2f}°C
- Comfort Range: {heating_setpoint}°C to {cooling_setpoint}°C
- Current Timestep: {timestep}
"""

        self.OBJECTIVE = """
CONTROL OBJECTIVES:
- Maintain zone temperature within the comfort range
- Minimize energy consumption
"""

        self.EXPERT_KNOWLEDGE = """
HVAC CONTROL PRINCIPLES:
- Proportional Response: Control intensity should match the magnitude and urgency of the temperature error
- Confidence Threshold: Only activate HVAC when confident that inaction will violate comfort bounds
- Natural Drift Utilization: When temperature is moving toward or within comfort zone naturally, avoid unnecessary intervention

GENERAL CONSTRAINT:
- Supply Air Flow Rate: {flow_min} to {flow_max} m³/s
- Supply Air Temperature: {temp_min}°C to {temp_max}°C
"""

        self.BUILDING_INFO = """
BUILDING SYSTEM SPECIFICATIONS:
- Constant fan - either OFF (0 m³/s) or ON (0.4 m³/s)
"""

        self.DISTURBANCE_FORECAST = """
PREDICTED DISTURBANCES (Next Hour):
{forecast_data}

FORECAST GUIDANCE:
The temperature setpoints have setback
- Pre heating or cooling can avoid comfort violation before occupied
- Turn off system can save energy before unoccupied
"""

        self.HISTORICAL_PATTERNS_TEMPLATE = """
LEARNED PATTERNS (Recent 2 Hours):
{history_patterns}

PATTERN INSIGHTS:
- Analyze temperature rise/fall rates under different conditions
- Use the updated dynamic information to optimize control action, only activate HVAC when confident that inaction will violate comfort bounds
"""

        self.OUTPUT_GUIDE = """
RESPONSE FORMAT:
Provide your control decision in the following JSON format:
{{
    "supply_air_flow_rate": <float between {flow_min} and {flow_max}>,
    "supply_air_temperature": <float between {temp_min} and {temp_max}>,
    "reasoning": {{
        "current_analysis": "<assessment of current state and urgency>",
        "control_strategy": "<chosen approach and why>",
        "energy_consideration": "<how this balances comfort and efficiency>",
        "confidence": "<high/medium/low confidence in maintaining comfort>"
    }}
}}
"""

    def _extract_recent_history(self, history_info: deque, num_steps: int = 8) -> List[Dict]:
        """Extract the most recent history steps for analysis."""
        recent_history = list(history_info)[-num_steps:] if len(history_info) >= num_steps else list(history_info)
        return recent_history

    def _create_pattern_pairs(self, history: List[Dict]) -> str:
        """Create state-action-result pairs from history for pattern recognition."""
        patterns = []
        for i in range(len(history) - 1):
            current = history[i]
            next_state = history[i + 1]

            # Handle the history format from your temperature_buffer
            if isinstance(current, dict):
                hvac_power = current.get('phvac')
                zone_temp = current.get('temp_room')
                supply_temp = 13  # Fixed assumption
                temp_diff = zone_temp - supply_temp

                if abs(temp_diff) > 0.1 and hvac_power != 0:  # Avoid division by zero
                    # Power in kW, convert to flow rate in m³/s
                    flow_rate = abs(hvac_power/1000) / (self.rho_air * self.cp_air * abs(temp_diff))
                else:
                    flow_rate = 0

                pattern = {
                    'timestep': i,
                    'state': {
                        'zone_temp': current.get('temp_room', current.get('temperature', 0)),
                        'ambient_temp': current.get('temp_amb', current.get('ambient_temp', 0)),
                        'solar_gain': current.get('solar', current.get('solar_radiation', 0)),
                        'occupancy': current.get('occ', current.get('occupancy', 0))
                    },
                    'action': {
                        'hvac_power': current.get('phvac', current.get('hvac_power', 0)),
                        'supply_temp': supply_temp,
                        'flow_rate': flow_rate,
                    },
                    'result': {
                        'next_temp': next_state.get('temp_room', next_state.get('temperature', 0)),
                        'temp_change': next_state.get('temp_room', next_state.get('temperature', 0)) -
                                       current.get('temp_room', current.get('temperature', 0))
                    }
                }
            else:
                # Handle if history is just temperature values
                pattern = {
                    'timestep': i,
                    'state': {
                        'zone_temp': float(current) if not isinstance(current, dict) else current
                    },
                    'result': {
                        'next_temp': float(next_state) if not isinstance(next_state, dict) else next_state
                    }
                }
            patterns.append(pattern)

        # Create a summary of key patterns
        summary = {
            'recent_patterns': patterns[-4:],  # Last 3 patterns
        }

        return json.dumps(summary, indent=2)

    def _format_disturbance_forecast(self, disturbance: Any, num_steps: int = 4) -> str:
        """Format disturbance forecast for higher level prompts."""
        try:
            forecast_data = {}

            # Weather forecast
            # if hasattr(disturbance, 'weather'):
            #     weather = disturbance.weather
            #     if hasattr(weather, 'forecast_outdoor_dry_bulb_temp'):
            #         forecast_data['outdoor_temp_forecast'] = weather.forecast_outdoor_dry_bulb_temp[:num_steps].tolist()
            #     if hasattr(weather, 'forecast_solar_radiation_w_m2'):
            #         forecast_data['solar_radiation_forecast'] = weather.forecast_solar_radiation_w_m2[
            #                                                     :num_steps].tolist()
            #     forecast_data['current_outdoor_temp'] = weather.outdoor_dry_bulb_temp
            #     forecast_data['current_solar_radiation'] = weather.solar_radiation_w_m2

            # Occupancy forecast
            if hasattr(disturbance, 'occupancy'):
                occupancy = disturbance.occupancy
                if hasattr(occupancy, 'occupancy_forecast'):
                    forecast_data['occupancy_forecast'] = occupancy.occupancy_forecast[:num_steps].tolist()
                forecast_data['current_occupancy'] = occupancy.occupancy_fraction

            return json.dumps(forecast_data, indent=2)
        except Exception as e:
            # Return minimal forecast if parsing fails
            return json.dumps({"error": f"Could not parse forecast: {str(e)}"}, indent=2)

    def _build_prompt(self, level: int, current_temp: float, cooling_setpoint: float,
                      heating_setpoint: float, timestep: int,
                      history_patterns: Optional[str] = None,
                      disturbance_forecast: Optional[str] = None) -> str:
        """Build prompt based on complexity level."""

        # Format constraint values
        format_dict = {
            'current_temp': current_temp,
            'cooling_setpoint': cooling_setpoint,
            'heating_setpoint': heating_setpoint,
            'timestep': timestep,
            'flow_min': self.constraints['supply_air_flow_rate']['min'],
            'flow_max': self.constraints['supply_air_flow_rate']['max'],
            'temp_min': self.constraints['supply_air_temperature']['min'],
            'temp_max': self.constraints['supply_air_temperature']['max']
        }

        # Start with system prompt
        prompt_parts = [self.SYSTEM_PROMPT]

        # Add current state (all levels)
        prompt_parts.append(self.CURRENT_STATE.format(**format_dict))

        # Add objectives (all levels)
        prompt_parts.append(self.OBJECTIVE)

        # Level 1+: Add expert knowledge
        if level >= 1:
            prompt_parts.append(self.EXPERT_KNOWLEDGE.format(**format_dict))

        # Level 2+: Add building information
        if level >= 2:
            prompt_parts.append(self.BUILDING_INFO)

        # Level 3+: Add disturbance forecast
        if level >= 3 and disturbance_forecast:
            forecast_section = self.DISTURBANCE_FORECAST.format(forecast_data=disturbance_forecast)
            prompt_parts.append(forecast_section)

        # Level 4: Add historical patterns
        if level >= 4 and history_patterns:
            history_section = self.HISTORICAL_PATTERNS_TEMPLATE.format(history_patterns=history_patterns)
            prompt_parts.append(history_section)

        # Add output guide (all levels)
        prompt_parts.append(self.OUTPUT_GUIDE.format(**format_dict))

        # Add level-specific closing guidance
        if level == 0:
            prompt_parts.append("\nMake your best decision based on the current state information.")
        elif level == 1:
            prompt_parts.append("\nApply the expert knowledge to make an informed control decision.")
        elif level == 2:
            prompt_parts.append("\nConsider both the expert principles and specific system constraints.")
        elif level == 3:
            prompt_parts.append(
                "\nIntegrate current state, expert knowledge, and predictive forecasts for optimal control.")
        elif level == 4:
            prompt_parts.append(
                "\nSynthesize all available information - current state, predictions, and historical learning - to make the most informed decision.")

        return "\n".join(prompt_parts)

    def _parse_llm_response(self, response_text: str) -> Tuple[float, float, Dict]:
        """Parse the LLM response and extract control values."""
        try:
            # Clean the response text
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            elif cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()

            # Parse JSON response
            response_data = json.loads(cleaned_text)

            # Extract and validate values
            flow_rate = float(response_data['supply_air_flow_rate'])
            supply_temp = float(response_data['supply_air_temperature'])
            reasoning = response_data.get('reasoning', {})

            # Apply constraints
            flow_rate = np.clip(flow_rate,
                                self.constraints['supply_air_flow_rate']['min'],
                                self.constraints['supply_air_flow_rate']['max'])
            supply_temp = np.clip(supply_temp,
                                  self.constraints['supply_air_temperature']['min'],
                                  self.constraints['supply_air_temperature']['max'])

            return flow_rate, supply_temp, reasoning

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing LLM response: {e}")
            return self._get_fallback_control(str(e))

    def _get_fallback_control(self, error_msg: str) -> Tuple[float, float, Dict]:
        """Provide fallback control values if LLM response parsing fails."""
        return (
            0.4,  # Default to moderate cooling for safety
            20.0,  # Mild cooling temperature
            {
                "error": "Fallback mode activated",
                "error_details": error_msg[:200]
            }
        )

    def get_control_action(self, current_temp: float,
                           history_info: Optional[deque] = None,
                           cooling_setpoint: float = 26.0,
                           heating_setpoint: float = 22.0,
                           timestep: int = 0,
                           disturbance: Optional[Any] = None,
                           level: int = 2) -> Tuple[float, float, Dict]:
        """
        Get HVAC control actions using specified prompt level.

        Args:
            current_temp: Current zone temperature
            history_info: Historical building state information
            cooling_setpoint: Upper temperature limit
            heating_setpoint: Lower temperature limit
            timestep: Current simulation timestep
            disturbance: Disturbance object with forecasts
            level: Prompt complexity level (0-4)

        Returns:
            Tuple of (flow_rate, supply_temp, reasoning)
        """
        start_time = time.time()

        try:
            # Prepare optional components based on level
            history_patterns = None
            disturbance_forecast = None

            # Prepare historical patterns for level 4
            if level >= 4 and history_info and len(history_info) > 0:
                recent_history = self._extract_recent_history(history_info)
                history_patterns = self._create_pattern_pairs(recent_history)

            # Prepare disturbance forecast for level 3+
            if level >= 3 and disturbance is not None:
                disturbance_forecast = self._format_disturbance_forecast(disturbance)

            # Build the prompt
            prompt = self._build_prompt(
                level=level,
                current_temp=current_temp,
                cooling_setpoint=cooling_setpoint,
                heating_setpoint=heating_setpoint,
                timestep=timestep,
                history_patterns=history_patterns,
                disturbance_forecast=disturbance_forecast
            )

            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=1024
            )

            # Parse response
            response_text = response.choices[0].message.content
            flow_rate, supply_temp, reasoning = self._parse_llm_response(response_text)

            # Calculate metrics
            response_time = time.time() - start_time

            # Add metadata to reasoning
            reasoning['level'] = level
            reasoning['timestamp'] = timestep
            reasoning['current_temp'] = current_temp
            reasoning['response_time_seconds'] = response_time
            reasoning['prompt_tokens'] = response.usage.prompt_tokens
            reasoning['completion_tokens'] = response.usage.completion_tokens
            reasoning['total_tokens'] = response.usage.total_tokens

            # Create and store metrics
            metrics = ControllerMetrics(
                level=level,
                timestamp=timestep,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                response_time_seconds=response_time,
                flow_rate=flow_rate,
                supply_temp=supply_temp,
                reasoning=reasoning,
                prompt_length=len(prompt),
                success=True
            )
            self.metrics_history[level].append(metrics)

            return flow_rate, supply_temp, reasoning

        except Exception as e:
            print(f"Error in LLM control (Level {level}): {e}")
            # Return safe defaults with error in reasoning
            reasoning = {
                "error": str(e),
                "level": level,
                "timestamp": timestep,
                "fallback_mode": True
            }

            # Store error metrics
            metrics = ControllerMetrics(
                level=level,
                timestamp=timestep,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                response_time_seconds=time.time() - start_time,
                flow_rate=0.4,
                supply_temp=20.0,
                reasoning=reasoning,
                prompt_length=0,
                success=False,
                error_message=str(e)
            )
            self.metrics_history[level].append(metrics)

            return 0.4, 20.0, reasoning

    def save_metrics_to_file(self, filename: str = "llm_controller_metrics.json"):
        """Save all collected metrics to a JSON file for analysis."""
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "temperature": self.temperature,
            "metrics_by_level": {}
        }

        for level, metrics_list in self.metrics_history.items():
            output_data["metrics_by_level"][f"level_{level}"] = [
                asdict(m) for m in metrics_list
            ]

        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Metrics saved to {filename}")

    def get_metrics_summary(self) -> Dict:
        """Generate summary statistics for each level."""
        summary = {}

        for level, metrics_list in self.metrics_history.items():
            if not metrics_list:
                continue

            successful_runs = [m for m in metrics_list if m.success]

            if successful_runs:
                summary[f"level_{level}"] = {
                    "total_runs": len(metrics_list),
                    "successful_runs": len(successful_runs),
                    "avg_response_time": np.mean([m.response_time_seconds for m in successful_runs]),
                    "std_response_time": np.std([m.response_time_seconds for m in successful_runs]),
                    "avg_total_tokens": np.mean([m.total_tokens for m in successful_runs]),
                    "avg_prompt_tokens": np.mean([m.prompt_tokens for m in successful_runs]),
                    "avg_completion_tokens": np.mean([m.completion_tokens for m in successful_runs]),
                    "avg_prompt_length": np.mean([m.prompt_length for m in successful_runs]),
                    "total_tokens_used": sum([m.total_tokens for m in successful_runs]),
                    "estimated_cost_usd": sum([m.total_tokens for m in successful_runs]) * 0.00001,
                    "error_rate": (len(metrics_list) - len(successful_runs)) / len(metrics_list) if len(
                        metrics_list) > 0 else 0
                }

        return summary

    def print_comparison_report(self):
        """Print a formatted comparison report of all levels."""
        summary = self.get_metrics_summary()

        if not summary:
            print("No data to report. Run some experiments first.")
            return

        print("\n" + "=" * 80)
        print(" LLM CONTROLLER PERFORMANCE COMPARISON REPORT ")
        print("=" * 80)

        # Headers
        metrics_to_show = [
            ("Successful Runs", "successful_runs", "{:.0f}"),
            ("Avg Response Time (s)", "avg_response_time", "{:.3f}"),
            ("Std Response Time (s)", "std_response_time", "{:.3f}"),
            ("Avg Total Tokens", "avg_total_tokens", "{:.1f}"),
            ("Avg Prompt Tokens", "avg_prompt_tokens", "{:.1f}"),
            ("Avg Completion Tokens", "avg_completion_tokens", "{:.1f}"),
            ("Total Tokens Used", "total_tokens_used", "{:.0f}"),
            ("Estimated Cost ($)", "estimated_cost_usd", "${:.4f}"),
            ("Error Rate (%)", "error_rate", "{:.1%}")
        ]

        # Print header
        print(f"\n{'Metric':<25}", end="")
        for i in range(5):  # Now supports 0-4
            print(f"{'Level ' + str(i):>15}", end="")
        print("\n" + "-" * 95)

        # Print metrics
        for metric_name, metric_key, format_str in metrics_to_show:
            print(f"{metric_name:<25}", end="")
            for level in range(5):
                level_data = summary.get(f"level_{level}", {})
                value = level_data.get(metric_key, 0)
                if value is not None:
                    formatted = format_str.format(value)
                    print(f"{formatted:>15}", end="")
                else:
                    print(f"{'N/A':>15}", end="")
            print()

        print("=" * 95)

        # Print level descriptions
        print("\nLevel Descriptions:")
        print("  Level 0: Basic - System prompt + Current state + Objectives")
        print("  Level 1: Expert - Adds HVAC control principles and constraints")
        print("  Level 2: Informed - Adds specific building system specifications")
        print("  Level 3: Predictive - Adds disturbance forecasts for proactive control")
        print("  Level 4: Adaptive - Adds historical patterns for learning-based control")


class LLMDERController:
    """
    LLM-based DER controller for intelligent demand response.
    Optimizes storage usage based on forecasts to minimize costs.
    """

    def __init__(self,
                 api_key: str,
                 model: str = "gpt-4o",
                 temperature: float = 0.3,
                 max_grid_import: float = 20.0,  # kW
                 max_grid_export: float = 5.0,  # kW
                 bat_soc_min: float = 0.1,
                 bat_soc_max: float = 0.9,
                 bat_soc_reserve: float = 0.2,
                 ev_soc_min: float = 0.2,
                 ev_soc_max: float = 0.9,
                 ev_v2g_enabled: bool = True,
                 timestep_hours: float = 0.25):  # 15-minute timesteps
        """
        Initialize the LLM DER controller.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o)
            temperature: LLM temperature for response consistency
            max_grid_import: Maximum grid import power (kW)
            max_grid_export: Maximum grid export power (kW)
            bat_soc_min: Minimum battery SOC
            bat_soc_max: Maximum battery SOC
            bat_soc_reserve: Battery SOC reserve for peak (used when peak)
            ev_soc_min: Minimum EV SOC
            ev_soc_max: Maximum EV SOC
            ev_v2g_enabled: Whether V2G is enabled for EVs
            timestep_hours: Duration of each timestep in hours
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

        # Grid constraints
        self.max_grid_import = max_grid_import
        self.max_grid_export = max_grid_export

        # SOC constraints
        self.bat_soc_min = bat_soc_min
        self.bat_soc_max = bat_soc_max
        self.bat_soc_reserve = bat_soc_reserve
        self.ev_soc_min = ev_soc_min
        self.ev_soc_max = ev_soc_max
        self.ev_v2g_enabled = ev_v2g_enabled

        # Timestep duration for energy calculations
        self.timestep_hours = timestep_hours

        self.logger = logging.getLogger(__name__)

    def _get_capacity(self, component: Any, comp_type: str, is_peak: bool = False) -> Tuple[float, float]:
        """
        Calculate max charge/discharge capacity for a component.
        Matches the logic from rule-based controller.

        Args:
            component: Component state object
            comp_type: 'battery' or 'ev'
            is_peak: Whether currently in peak period (affects battery reserve)

        Returns:
            Tuple of (max_charge_kw, max_discharge_kw)
        """
        # Extract component properties
        soc = float(component.soc)

        if comp_type == 'battery':
            capacity_kwh = float(component.capacity_kwh)
            charge_c = float(component.charge_speed)
            discharge_c = float(component.discharge_speed)

            # SOC limits (use reserve during peak for batteries)
            if is_peak:
                soc_min = self.bat_soc_reserve  # Can discharge to reserve during peak
            else:
                soc_min = self.bat_soc_min
            soc_max = self.bat_soc_max

        else:  # 'ev'
            # Handle capacity in Wh or kWh
            if hasattr(component, 'capacity_kwh'):
                capacity_kwh = float(component.capacity_kwh)
            elif hasattr(component, 'capacity_wh'):
                capacity_kwh = float(component.capacity_wh) / 1000 if component.capacity_wh > 0 else 40.0
            else:
                capacity_kwh = 40.0  # Default

            charge_c = float(component.charge_speed)
            discharge_c = float(component.discharge_speed)

            # EV SOC limits
            soc_min = self.ev_soc_min
            soc_max = self.ev_soc_max

        # Clamp SOC limits
        soc_min = max(0.0, min(1.0, soc_min))
        soc_max = max(0.0, min(1.0, soc_max))

        # Power limits from C-rate (kW)
        rate_charge_kw = charge_c * capacity_kwh
        rate_discharge_kw = discharge_c * capacity_kwh

        # Energy windows to bounds (kWh)
        energy_to_max = max(0.0, (soc_max - soc) * capacity_kwh)
        energy_above_min = max(0.0, (soc - soc_min) * capacity_kwh)

        # Convert energy to power considering timestep duration
        # For 15-min timestep (0.25 hours), multiply by 4 to get kW
        max_charge_kw = min(rate_charge_kw, energy_to_max / self.timestep_hours)
        max_discharge_kw = min(rate_discharge_kw, energy_above_min / self.timestep_hours)

        return max_charge_kw, max_discharge_kw

    def _extract_der_capabilities(self, der_state: Dict, is_peak: bool) -> Dict:
        """
        Extract capabilities and constraints from DER state.
        Dynamically handles any component IDs.

        Args:
            der_state: Current DER component states
            is_peak: Current peak status

        Returns:
            Dictionary of capabilities for each component
        """
        capabilities = {
            'batteries': {},
            'evs': {},
            'pvs': {},
            'summary': {
                'total_battery_charge_kw': 0,
                'total_battery_discharge_kw': 0,
                'total_ev_charge_kw': 0,
                'total_ev_discharge_kw': 0,
                'battery_ids': [],
                'ev_ids': [],
                'pv_ids': []
            }
        }

        for component_id, state in der_state.items():
            if 'bat' in component_id.lower():
                # Battery capabilities
                max_charge_kw, max_discharge_kw = self._get_capacity(state, 'battery', is_peak)

                capabilities['batteries'][component_id] = {
                    'soc': float(state.soc),
                    'capacity_kwh': float(state.capacity_kwh),
                    'max_charge_kw': max_charge_kw,
                    'max_discharge_kw': max_discharge_kw,
                    'energy_available_kwh': max_discharge_kw * self.timestep_hours,
                    'energy_capacity_kwh': max_charge_kw * self.timestep_hours
                }

                capabilities['summary']['battery_ids'].append(component_id)
                capabilities['summary']['total_battery_charge_kw'] += max_charge_kw
                capabilities['summary']['total_battery_discharge_kw'] += max_discharge_kw

            elif 'ev' in component_id.lower():
                # Check if EV is connected and active
                is_connected = getattr(state, 'initially_connected', True)
                is_active = getattr(state, 'is_active', True)

                if is_connected and is_active:
                    max_charge_kw, max_discharge_kw = self._get_capacity(state, 'ev', is_peak)

                    # Disable discharge if V2G not enabled
                    if not self.ev_v2g_enabled:
                        max_discharge_kw = 0
                else:
                    max_charge_kw = 0
                    max_discharge_kw = 0

                capabilities['evs'][component_id] = {
                    'soc': float(state.soc),
                    'max_charge_kw': max_charge_kw,
                    'max_discharge_kw': max_discharge_kw,
                    'is_connected': is_connected,
                    'is_active': is_active,
                    'v2g_enabled': self.ev_v2g_enabled and is_connected
                }

                if is_connected and is_active:
                    capabilities['summary']['ev_ids'].append(component_id)
                    capabilities['summary']['total_ev_charge_kw'] += max_charge_kw
                    capabilities['summary']['total_ev_discharge_kw'] += max_discharge_kw

            elif 'pv' in component_id.lower():
                capabilities['pvs'][component_id] = {
                    'current_generation_kw': float(state.generation_w) / 1000
                }
                capabilities['summary']['pv_ids'].append(component_id)

        return capabilities

    def _analyze_forecast_windows(self,
                                  forecast_peaksignal: np.ndarray,
                                  forecast_solar: np.ndarray,
                                  load_forecast: np.ndarray) -> Dict:
        """
        Analyze forecasts to identify key time windows for optimization.

        Args:
            forecast_peaksignal: 96-step peak signal forecast
            forecast_solar: 96-step solar radiation forecast (W/m²)
            load_forecast: 96-step load forecast (kW)

        Returns:
            Analysis of forecast windows
        """
        # Find peak periods
        peak_indices = np.where(forecast_peaksignal)[0]
        peak_start = int(peak_indices[0]) if len(peak_indices) > 0 else -1
        peak_duration = len(peak_indices)

        # Estimate PV generation from solar radiation
        # Rough estimation: 5kW system with 15% efficiency
        pv_capacity_kw = 5.0
        panel_area = 25  # m² for 5kW system
        efficiency = 0.15
        estimated_pv = forecast_solar * panel_area * efficiency / 1000  # kW

        # Calculate energy needs during peak
        if len(peak_indices) > 0:
            peak_load_kwh = np.sum(load_forecast[peak_indices]) * self.timestep_hours
            peak_solar_kwh = np.sum(estimated_pv[peak_indices]) * self.timestep_hours
            net_peak_energy_needed = peak_load_kwh - peak_solar_kwh
        else:
            peak_load_kwh = 0
            peak_solar_kwh = 0
            net_peak_energy_needed = 0

        # Find optimal pre-charging windows (high solar, low load, before peak)
        if peak_start > 0:
            pre_peak_window = slice(max(0, peak_start - 20), peak_start)
            pre_peak_solar_kwh = np.sum(estimated_pv[pre_peak_window]) * self.timestep_hours
            pre_peak_load_kwh = np.sum(load_forecast[pre_peak_window]) * self.timestep_hours
            pre_peak_net_solar = pre_peak_solar_kwh - pre_peak_load_kwh
        else:
            pre_peak_solar_kwh = 0
            pre_peak_load_kwh = 0
            pre_peak_net_solar = 0

        # Calculate next 4 hours (16 timesteps)
        next_4h_solar_kwh = np.sum(estimated_pv[:16]) * self.timestep_hours
        next_4h_load_kwh = np.sum(load_forecast[:16]) * self.timestep_hours

        return {
            'peak_starts_in': peak_start,
            'peak_duration_steps': peak_duration,
            'peak_load_kwh': round(peak_load_kwh, 2),
            'peak_solar_kwh': round(peak_solar_kwh, 2),
            'net_peak_energy_needed': round(net_peak_energy_needed, 2),
            'pre_peak_net_solar_kwh': round(pre_peak_net_solar, 2),
            'next_4h_solar_kwh': round(next_4h_solar_kwh, 2),
            'next_4h_load_kwh': round(next_4h_load_kwh, 2),
            'current_is_peak': bool(forecast_peaksignal[0]),
            'next_4h_has_peak': bool(np.any(forecast_peaksignal[:16])),
            'high_solar_coming_2h': bool(np.max(forecast_solar[:8]) > 500),
            'high_solar_coming_4h': bool(np.max(forecast_solar[:16]) > 500)
        }

    def _build_dynamic_json_template(self, capabilities: Dict) -> str:
        """
        Build a dynamic JSON template based on available components.

        Args:
            capabilities: Component capabilities dictionary

        Returns:
            JSON template string
        """
        battery_dict = {bid: "<float>" for bid in capabilities['summary']['battery_ids']}
        ev_dict = {eid: "<float>" for eid in capabilities['summary']['ev_ids']}

        template = {
            "pv2building": "<float>",
            "pv2battery": battery_dict if battery_dict else {},
            "pv2ev": ev_dict if ev_dict else {},
            "pv2grid": "<float>",
            "battery2building": battery_dict if battery_dict else {},
            "grid2battery": battery_dict if battery_dict else {},
            "ev2building": ev_dict if ev_dict else {},
            "grid2ev": ev_dict if ev_dict else {},
            "grid2building": "<float>",
            "reasoning": {
                "strategy": "<current strategy>",
                "storage_decision": "<why charge/discharge now>",
                "forecast_consideration": "<how forecast affects decision>",
                "cost_optimization": "<how this minimizes cost>"
            }
        }

        return json.dumps(template, indent=2)

    def _build_optimization_prompt(self,
                                   building_load: float,
                                   pv_generation: float,
                                   capabilities: Dict,
                                   forecast_analysis: Dict,
                                   is_peak: bool) -> str:
        """
        Build the optimization prompt for the LLM.

        Args:
            building_load: Current building load (kW)
            pv_generation: Current PV generation (kW)
            capabilities: DER component capabilities
            forecast_analysis: Analysis of forecasts
            is_peak: Current peak status

        Returns:
            Formatted prompt string
        """
        # Create dynamic JSON template
        json_template = self._build_dynamic_json_template(capabilities)

        prompt = f"""You are an expert DER controller optimizing for minimum electricity cost through intelligent storage management.

CURRENT STATE:
- Building Load: {building_load:.2f} kW
- PV Generation: {pv_generation:.2f} kW
- Peak Period: {'YES (expensive)' if is_peak else 'NO (cheap)'}
- Net Load: {(building_load - pv_generation):.2f} kW

STORAGE CAPABILITIES:
Batteries: {json.dumps(capabilities['batteries'], indent=2)}
EVs: {json.dumps(capabilities['evs'], indent=2)}

TOTAL AVAILABLE:
- Battery Charge Capacity: {capabilities['summary']['total_battery_charge_kw']:.2f} kW
- Battery Discharge Capacity: {capabilities['summary']['total_battery_discharge_kw']:.2f} kW
- EV Charge Capacity: {capabilities['summary']['total_ev_charge_kw']:.2f} kW
- EV Discharge Capacity: {capabilities['summary']['total_ev_discharge_kw']:.2f} kW

FORECAST ANALYSIS:
- Peak starts in: {forecast_analysis['peak_starts_in']} steps (15 min each)
- Peak duration: {forecast_analysis['peak_duration_steps']} steps
- Net energy needed during peak: {forecast_analysis['net_peak_energy_needed']} kWh
- Available solar before peak: {forecast_analysis['pre_peak_net_solar_kwh']} kWh
- Next 4h solar generation: {forecast_analysis['next_4h_solar_kwh']} kWh
- Next 4h load: {forecast_analysis['next_4h_load_kwh']} kWh
- High solar in 2h: {forecast_analysis['high_solar_coming_2h']}
- High solar in 4h: {forecast_analysis['high_solar_coming_4h']}

GRID CONSTRAINTS:
- Max Import: {self.max_grid_import} kW
- Max Export: {self.max_grid_export} kW

OPTIMIZATION RULES:
1. The max_charge_kw and max_discharge_kw values ALREADY include SOC constraints - don't reduce them further
2. During OFF-PEAK: 
   - If peak coming soon AND net_peak_energy_needed > current storage: charge aggressively
   - If high solar coming soon: charge conservatively to leave room for PV
   - Use excess PV first, then grid if needed

3. During PEAK:
   - Discharge storage to minimize grid import (most expensive)
   - Prioritize batteries over EVs for reliability
   - Export excess PV if profitable

4. SMART DECISIONS:
   - Calculate if pre-charging is needed: compare net_peak_energy_needed vs available storage
   - Don't overcharge if solar will provide energy during peak
   - Consider round-trip efficiency (~90%) when pre-charging

CRITICAL: 
- All values must be non-negative
- Don't exceed the max_charge_kw or max_discharge_kw limits (they already account for SOC)
- Ensure power balance: sources ≈ sinks
- Only use component IDs that exist in the capabilities

Respond with this exact JSON structure (use actual component IDs):
{json_template}"""

        return prompt

    def _parse_llm_response(self, response_text: str, capabilities: Dict) -> Tuple[Dict, Dict]:
        """
        Parse LLM response into action dictionary.

        Args:
            response_text: Raw LLM response
            capabilities: Component capabilities for validation

        Returns:
            Tuple of (action_dict, reasoning)
        """
        try:
            # Clean response text
            cleaned = response_text.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            elif cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Parse JSON
            data = json.loads(cleaned)

            # Extract action components, filtering by available components
            action_dict = {}

            # Single value fields
            for field in ['pv2building', 'pv2grid', 'grid2building']:
                action_dict[field] = max(0, float(data.get(field, 0)))

            # Dictionary fields - only include components that exist
            battery_ids = capabilities['summary']['battery_ids']
            ev_ids = capabilities['summary']['ev_ids']

            # Battery-related flows
            for field in ['pv2battery', 'battery2building', 'grid2battery']:
                action_dict[field] = {}
                if field in data and isinstance(data[field], dict):
                    for bat_id in battery_ids:
                        if bat_id in data[field]:
                            action_dict[field][bat_id] = max(0, float(data[field][bat_id]))

            # EV-related flows
            for field in ['pv2ev', 'ev2building', 'grid2ev']:
                action_dict[field] = {}
                if field in data and isinstance(data[field], dict):
                    for ev_id in ev_ids:
                        if ev_id in data[field]:
                            action_dict[field][ev_id] = max(0, float(data[field][ev_id]))

            reasoning = data.get('reasoning', {})

            return action_dict, reasoning

        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return self._get_fallback_action_dict(), {"error": str(e)}

    def _get_fallback_action_dict(self) -> Dict:
        """
        Provide fallback action dictionary for error cases.

        Returns:
            Safe default action dictionary
        """
        return {
            'pv2building': 0.0,
            'pv2battery': {},
            'pv2ev': {},
            'pv2grid': 0.0,
            'battery2building': {},
            'grid2battery': {},
            'ev2building': {},
            'grid2ev': {},
            'grid2building': 1.0  # Default minimal grid import
        }

    def _validate_and_apply_action(self,
                                   action: Any,
                                   action_dict: Dict,
                                   building_load: float,
                                   pv_generation: float,
                                   capabilities: Dict) -> Any:
        """
        Validate action dictionary and apply to DERSystemAction object.

        Args:
            action: Pre-defined DERSystemAction object
            action_dict: Action dictionary from LLM
            building_load: Current building load
            pv_generation: Current PV generation
            capabilities: Component capabilities

        Returns:
            Modified action object with validated values
        """
        # First, apply all values from action_dict to action object
        action.pv2building = action_dict.get('pv2building', 0)
        action.pv2grid = action_dict.get('pv2grid', 0)
        action.grid2building = action_dict.get('grid2building', 0)

        # Apply battery flows with validation
        for bat_id, charge_kw in action_dict.get('pv2battery', {}).items():
            if bat_id in capabilities['batteries']:
                max_charge = capabilities['batteries'][bat_id]['max_charge_kw']
                action.pv2battery[bat_id] = min(charge_kw, max_charge)

        for bat_id, charge_kw in action_dict.get('grid2battery', {}).items():
            if bat_id in capabilities['batteries']:
                max_charge = capabilities['batteries'][bat_id]['max_charge_kw']
                action.grid2battery[bat_id] = min(charge_kw, max_charge)

        for bat_id, discharge_kw in action_dict.get('battery2building', {}).items():
            if bat_id in capabilities['batteries']:
                max_discharge = capabilities['batteries'][bat_id]['max_discharge_kw']
                action.battery2building[bat_id] = min(discharge_kw, max_discharge)

        # Apply EV flows with validation
        for ev_id, charge_kw in action_dict.get('pv2ev', {}).items():
            if ev_id in capabilities['evs']:
                max_charge = capabilities['evs'][ev_id]['max_charge_kw']
                action.pv2ev[ev_id] = min(charge_kw, max_charge)

        for ev_id, charge_kw in action_dict.get('grid2ev', {}).items():
            if ev_id in capabilities['evs']:
                max_charge = capabilities['evs'][ev_id]['max_charge_kw']
                action.grid2ev[ev_id] = min(charge_kw, max_charge)

        for ev_id, discharge_kw in action_dict.get('ev2building', {}).items():
            if ev_id in capabilities['evs']:
                max_discharge = capabilities['evs'][ev_id]['max_discharge_kw']
                action.ev2building[ev_id] = min(discharge_kw, max_discharge)

        # Validate PV allocation doesn't exceed generation
        total_pv_out = (action.pv2building + action.pv2grid +
                        sum(action.pv2battery.values()) + sum(action.pv2ev.values()))

        if total_pv_out > pv_generation * 1.01:  # 1% tolerance
            # Scale down PV allocations
            if total_pv_out > 0:
                scale = pv_generation / total_pv_out
                action.pv2building *= scale
                action.pv2grid *= scale
                for k in action.pv2battery:
                    action.pv2battery[k] *= scale
                for k in action.pv2ev:
                    action.pv2ev[k] *= scale

        # Validate grid constraints
        total_grid_import = action.grid2building + sum(action.grid2battery.values()) + sum(action.grid2ev.values())
        if total_grid_import > self.max_grid_import:
            # Scale down grid imports
            scale = self.max_grid_import / total_grid_import if total_grid_import > 0 else 0
            action.grid2building *= scale
            for k in action.grid2battery:
                action.grid2battery[k] *= scale
            for k in action.grid2ev:
                action.grid2ev[k] *= scale

        action.pv2grid = min(action.pv2grid, self.max_grid_export)

        # Ensure building load is met
        total_to_building = (action.pv2building + action.grid2building +
                             sum(action.battery2building.values()) +
                             sum(action.ev2building.values()))

        if total_to_building < building_load * 0.99:  # 1% tolerance
            # Increase grid import to meet load
            deficit = building_load - total_to_building
            action.grid2building += deficit

        return action

    def get_control_action(self,
                           action: Any,  # Pre-defined DERSystemAction
                           building_load: float,
                           pv_generation: float,
                           der_state: Dict,
                           is_peak: bool,
                           forecast_peaksignal: np.ndarray,
                           forecast_outdoor_temp: np.ndarray,
                           forecast_solar: np.ndarray,
                           load_forecast: np.ndarray) -> Tuple[Any, Dict]:
        """
        Main method to get DER control action from LLM.

        Args:
            action: Pre-defined DERSystemAction object to populate
            building_load: Current building load (kW)
            pv_generation: Current PV generation (kW)
            der_state: Current state of all DER components
            is_peak: Current peak period status
            forecast_peaksignal: 96-step peak signal forecast
            forecast_outdoor_temp: 96-step temperature forecast
            forecast_solar: 96-step solar radiation forecast (W/m²)
            load_forecast: 96-step load forecast (kW)

        Returns:
            Tuple of (modified action object, reasoning)
        """
        try:
            # Extract capabilities with current peak status
            capabilities = self._extract_der_capabilities(der_state, is_peak)

            # Analyze forecasts
            forecast_analysis = self._analyze_forecast_windows(
                forecast_peaksignal, forecast_solar, load_forecast
            )

            # Build prompt
            prompt = self._build_optimization_prompt(
                building_load, pv_generation, capabilities,
                forecast_analysis, is_peak
            )

            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are an expert DER optimization controller focused on minimizing electricity costs while respecting all constraints."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=800
            )

            # Parse response
            response_text = response.choices[0].message.content
            action_dict, reasoning = self._parse_llm_response(response_text, capabilities)

            # Validate and apply to action object
            action = self._validate_and_apply_action(
                action, action_dict, building_load, pv_generation, capabilities
            )

            # Add metadata to reasoning
            reasoning['capabilities'] = capabilities
            reasoning['forecast_analysis'] = forecast_analysis

            return action, reasoning

        except Exception as e:
            self.logger.error(f"Error in LLM DER control: {e}")
            # Return action with minimal grid import as fallback
            action.grid2building = building_load
            return action, {"error": str(e), "fallback": True}


# Integration helper for your existing code
def create_llm_der_controller(api_key: str, **kwargs) -> LLMDERController:
    """
    Factory function to create LLM DER controller.

    Args:
        api_key: OpenAI API key
        **kwargs: Additional configuration parameters

    Returns:
        Configured LLMDERController instance
    """
    return LLMDERController(api_key=api_key, **kwargs)
