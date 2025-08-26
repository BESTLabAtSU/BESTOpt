import numpy as np
import matplotlib.pyplot as plt

Tamb = np.linspace(10, 50, 100)  # Ambient temperature from -15 to 40
cop = 7-0.2*Tamb+0.001*Tamb*Tamb

fig = plt.figure(figsize=(4, 2), dpi=300)
ax = fig.add_subplot(111)
ax.plot(Tamb, cop)
ax.set_xlabel('Ambient Temperature (°C)')
ax.set_ylabel('COP')
plt.tight_layout()
plt.show()
