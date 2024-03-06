import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


def load_cybres(folder_path, save):
    """
    Load CYBRES files fo light (blue and red stimuli) as well as the heat and wind stimuli and transform the single files into
    combined file. It should also cut off non-important data such as all measurements in the night so after 22pm and
    before 8am where the plant rests.
    """

    dfs = []

    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)

        # drop irrelevant info. Since we only classify based on the EP of CH1 and CH2, only those two columns as well as
        # the timestamp column are kept
        df = df.drop(["sender_hostname", "MU_MM", "MU_ID", "mag_X", "mag_Y", "mag_Z", "temp-PCB", "RF_power_emission",
                      "air_pressure", "soil_moisture", "soil_temperature", "light-external", "transpiration",
                      "temp-external", "humidity-external"], axis=1)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[(df['timestamp'].dt.hour >= 8) & (df['timestamp'].dt.hour < 22)]

        # normalize the data
        df['differential_potential_CH1'] = normalize((df['differential_potential_CH1'] -512000) / 1000)
        df['differential_potential_CH2'] = normalize((df['differential_potential_CH2'] -512000) / 1000)

        dfs.append(df)
    cybres_data = pd.concat(dfs)

    # optionally write data to file.
    if save:
        cybres_data.to_csv('cybres_data.csv', index=False)

    return cybres_data


def load_pyhto(folder_path, save, num):
    """
    Load Phytonodes files for light (blue and red stimuli) as well as the heat and wind stimuli and transform the single
    files into combined file. It should also cut off non-important data such as all measurements in the night so after
    22pm and before 8am where the plant rests as well as the filtered column (sensor not sending data there).
    """

    dfs = []

    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        print(file_path)
        df = pd.read_csv(file_path)

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S:%f')
        df = df[(df['timestamp'].dt.hour >= 8) & (df['timestamp'].dt.hour < 22)]

        # drop irrelevant info. We only classify based on the differential potential column as well as
        # the timestamp column
        df = df.drop(["filtered"], axis=1)

        # normalize the data
        df['differential_potential'] = normalize(df['differential_potential'])

        # sampling
        df = df.iloc[::1000, :]

        dfs.append(df)
    phyto_data = pd.concat(dfs)

    # optionally write data to file.
    if save:
        phyto_data.to_csv('phyto_data' + str(num) + '.csv', index=False)

    return phyto_data


def normalize(data):
    # normalize the data using standard min max normalization
    return (data - data.min()) / (data.max() - data.min())

def fft_transformation(data):
    data = np.fft.fft(data['differential_potential'])
    return data

def fft_per_experiment(data):
    specific_times = [
        (8, 0),
        (10, 10),
        (12, 20),
        (14, 30),
        (16, 40),
        (18, 50)
    ]

    fft, start_dates, test = [], [], []


    for timestamp in data['timestamp']:
        for hour, minute in specific_times:
            if timestamp.hour == hour and timestamp.minute == minute:
                start_dates.append(timestamp)

    for date in start_dates:
        for hour, minute in specific_times:
            start = pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=hour, minute=minute)
            end = start + pd.Timedelta(hours=2, minutes=10)

            experiment = data[(data['timestamp'] >= start) & (data['timestamp'] < end)]

            if len(experiment) not in [224, 225]:
                continue

            test.append(experiment)

            transformed = fft_transformation(experiment)
            fft.append(transformed)

    return fft, test


if __name__ == '__main__':
    home_dir = '/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Results/CSV/'
    #load_cybres(home_dir + '2-Day-test/CYBRES', True)
    #load_pyhto(home_dir + '2-Day-test/PN/P5', True, 5)

    #load_pyhto(home_dir + '2-Day-test/PN/P9', False, 9)

    data = pd.read_csv('/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Classification_Code/phyto_data9.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    transformed_data = fft_transformation(data)
    plt.plot(np.abs(transformed_data))
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('FFT Transformed Data')
    plt.show()

    '''
    ffts, tests = fft_per_experiment(data)
    plt.figure()
    plt.plot(np.abs(ffts[0]))
    plt.title(f"Experiment 0")
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(tests[2]['timestamp'], tests[2]['differential_potential'])
    plt.show()
    
    '''
