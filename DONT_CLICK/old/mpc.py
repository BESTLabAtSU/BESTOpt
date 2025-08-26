import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('df_cooling_with_weather.csv')
startday = 0
endday = 6
fig, ax = plt.subplots(1, 1, figsize=(5, 2), dpi=300)
ax.plot(np.arange(len(data))[96*startday:96*endday], data["Load_ele"][96*startday:96*endday]/100,
        '-', linewidth=1, color='red', label="predict")
ax.plot(np.arange(len(data))[96*startday:96*endday], data["Load_ele"][96*startday+1:96*endday+1]/100,
        '-', linewidth=1, color='black', label="measure")
ax.grid(False)
ax.set_ylabel('Load(kW)', fontsize=7)
ax.tick_params(axis='both', which='both', labelsize=7)
ax.set_xlabel('Time Step', fontsize=7)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=3, fontsize=7, frameon=False)
plt.tight_layout()
plt.show()