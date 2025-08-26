import os
import pandas as pd
import numpy as np

ENV_PATH = "../dataset"
LOAD_PATH = "../results"

for bldg in range(1,21):
    env_file = os.path.join("..", ENV_PATH, f"dataset_{bldg}.csv")
    load_file = os.path.join(LOAD_PATH, f"baseline_results_{bldg}.csv")

    env = pd.read_csv(env_file, index_col=0)
    load = pd.read_csv(load_file, index_col=0)
    load.index = pd.to_datetime(load.index)
    env.index = pd.to_datetime(env.index)
    env_day = env.loc[load.index]
    combined = pd.concat([load, env_day], axis=1)
    combined['hour_of_day'] = combined.index.hour
    total_steps = len(combined)
    TOU = np.zeros(total_steps)
    for i, hour in enumerate(combined['hour_of_day']):
        if 6 <= hour < 14:
            TOU[i] = 0.12  # Low rate
        elif 14 <= hour < 20:
            TOU[i] = 0.25  # High rate
        else:
            TOU[i] = 0.08  # Super low rate

    # Add TOU column to the DataFrame
    combined['TOU'] = TOU

    combined.to_csv(f"bldg{bldg-1}.csv")
