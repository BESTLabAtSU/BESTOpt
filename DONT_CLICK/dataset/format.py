import data_process as dp
import glob
import pandas as pd

files = glob.glob('./raw_data/'+'*.csv')
n=0
for file in files:
    n+=1
    df = pd.read_csv(file)
    df_format = dp.data_process(Tamb=df['temp_outdoor'],
                                Tspace=df['temp_zone_0'],
                                pHVAC=df['phvac_0'],
                                Occ=df['occ_0'],
                                Solar=df['solar'],
                                Index=df['Time'],
                                Tunit="F",
                                Punit="watt")
    df_format["setpt_cool"] = df["cooling_setpt_0"].values
    df_format["setpt_heat"] = df["heating_setpt_0"].values
    df_format.to_csv(f'dataset_{n}.csv')



    #
    #
    # dp.plot_data(df_format, plot_type="distribution")
    # dp.plot_data(df_format, plot_type="daily", day=50)
