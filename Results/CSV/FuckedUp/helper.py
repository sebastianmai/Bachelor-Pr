import pandas as pd

source_file = "/Results/CSV/Final/HEAT/Measurements/P2/P2_2024-03-09 12:00:00:002.csv"
destination_file = "/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Results/CSV/FuckedUp/Heat/P2_2024_03_19_n.csv"

start_timestamp = "2024-03-09 14:30:00:003"

df = pd.read_csv(source_file)

pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S:%f')

start_index = df[df['timestamp'] == start_timestamp].index[0]
df_subset = df.iloc[start_index:]
print(start_index)

df_subset.to_csv(destination_file, index=False)

df.drop(df.index[start_index:], inplace=True)
df.to_csv(source_file, index=False)
