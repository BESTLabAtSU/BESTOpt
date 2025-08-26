import numpy as np
import matplotlib.pyplot as plt

def cooling_cop(temp_diff):
    cop = 4.5 - 0.1 * temp_diff - 0.002 * temp_diff ** 2
    return cop

def heating_cop(temp_diff):
    cop = 3.5 + 0.08 * temp_diff - 0.001 * temp_diff ** 2
    return cop

# Define temperature differences
temp_diff_cooling = np.linspace(-15, 15, 100)  # outdoor - indoor
temp_diff_heating = np.linspace(-15, 15, 100)  # indoor - outdoor

# Calculate COPs
cop_cool = cooling_cop(temp_diff_cooling)
cop_heat = heating_cop(temp_diff_heating)

# Plotting
plt.rcParams.update({'font.size': 7})
plt.figure(figsize=(3, 2), dpi=300)
plt.plot(temp_diff_cooling, cop_cool, label='Cooling COP', color='blue')
plt.plot(temp_diff_heating, cop_heat, label='Heating COP', color='red')
plt.axhline(1, linestyle='--', color='gray', linewidth=1)
plt.xlabel('Tamb-Tzone (°C)')
plt.ylabel('COP')
plt.title('COP Curve for Heating and Cooling')
plt.legend(frameon=False)  # Remove legend frame
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()