import pandas as pd
import matplotlib.pyplot as plt

for dataset_id in range (1,21):
    data_path = f"dataset_{dataset_id}.csv"
    df=pd.read_csv(data_path)
    df.temp_room[96*180:96*183].plot()
plt.show()