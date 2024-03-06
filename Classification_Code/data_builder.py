import matplotlib
import pandas as pd
import os
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

#matplotlib.use('Qt5Agg')

def load_cybres(folder_path, save, cybres):
    """
    Load data files for light (blue and red stimuli) as well as the heat and wind stimuli and transform the single files into
    combined file. It should also cut off non-important data such as all measurements in the night so after 22pm and
    before 8am where the plant rests.
    """

    dfs = []

    for file_name in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)

        if cybres:
            # drop irrelevant info. Since we only classify based on the EP of CH1 and CH2, only those two columns as well as
            # the timestamp column are kept
            df = df.drop(["sender_hostname", "MU_MM", "MU_ID", "mag_X", "mag_Y", "mag_Z", "temp-PCB", "RF_power_emission",
                      "air_pressure", "soil_moisture", "soil_temperature", "light-external", "transpiration",
                      "temp-external", "humidity-external"], axis=1)

            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # normalize the data
            df['differential_potential_CH1'] = (df['differential_potential_CH1'] - 512000) / 1000
            df['differential_potential_CH2'] = (df['differential_potential_CH2'] - 512000) / 1000
        else:
            # drop irrelevant info and convert timestamp
            df = df.drop(['filtered'], axis=1)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S:%f')

        # we only look at the data between 8 and 22 o'clock
        df = df[(df['timestamp'].dt.hour >= 8) & (df['timestamp'].dt.hour < 22)]

        dfs.append(df)
    data = pd.concat(dfs)

    # optionally write data to file.
    if save:
        data.to_csv('cybres_data.csv', index=False)

    return get_experiments(data, cybres)


def get_experiments(data, cybres):

    """
    This function is used to split the data set consisting of continuous data into experiments that in total last for
    2 hours and 10 min and then the next starts
    """

    specific_times = [
        (8, 0),
        (10, 10),
        (12, 20),
        (14, 30),
        (16, 40),
        (18, 50)
    ]

    start_dates, experiment = [], []
    encountered = set()

    # get all start dates for the experiments
    for timestamp in data['timestamp']:
        minute = timestamp.minute
        if (minute, timestamp.day) not in encountered:
            for hour, specific_minute in specific_times:
                if minute == specific_minute and timestamp.hour == hour:
                    start_dates.append(timestamp)
                    encountered.add((minute, timestamp.day))

    # for each start data the data in between the start and end date is taken and appended to the experiment list
    for date in start_dates:
        start = pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=date.hour, minute=date.minute)
        end = start + pd.Timedelta(hours=2, minutes=10)

        helper = data[(data['timestamp'] >= start) & (data['timestamp'] < end)]

        if len(helper) >= 700 and cybres:
            experiment.append(helper)
        elif len(helper) >= 2000000 and not cybres:
            experiment.append(helper)

    return experiment


def get_interval(data, mod):

    """
    helper function for the background subtraction. It is used to split the data set consisting of continuous data for
    each experiment. It will just keep 20min of prestimulus, 10 min stimulus and, 20 min of poststimulus if mod is true.
    Else it will just get the 60 min of prestimulus
    """

    interval = []
    for df in data:
        date = df['timestamp'].iloc[0].replace(second=0)

        if mod:
            start = pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=date.hour, minute=date.minute) + pd.Timedelta(minutes=40)
            end = start + pd.Timedelta(minutes=50)
        else:
            start = pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=date.hour, minute=date.minute)
            end = start + pd.Timedelta(minutes=60)

        interval.append(df[(df['timestamp'] >= start) & (df['timestamp'] < end)].copy())

    return interval


def mean_cybres(data, cybres):

    """
    function to calculate the mean for the prestimulus based on whether it is data from CYBRES or the Phytonodes
    """

    mean = []
    if cybres:
        for i in range(0, len(data)):
            ch1 = np.mean(data[i]['differential_potential_CH1'])
            ch2 = np.mean(data[i]['differential_potential_CH2'])
            mean.append((ch1, ch2))
    else:
        for i in range(0, len(data)):
            mean.append(np.mean(data[i]['differential_potential']))

    return mean


def background_subtract(data, cybres):

    """
    function to subtract the mean from the entire set since this brings the potential closer if not to zero since we
    are more interested in the change rather than the actual values
    """

    interval_data = get_interval(data, True)
    mean_data = mean_cybres(get_interval(data, False), cybres)

    if cybres:
        for data, tuple in zip(interval_data, mean_data):
            data['differential_potential_CH1'] = data['differential_potential_CH1'] - tuple[0]
            data['differential_potential_CH2'] = data['differential_potential_CH2'] - tuple[1]
    else:
        for data, data_mean in zip(interval_data, mean_data):
            data['differential_potential'] = data['differential_potential'] - data_mean

    return interval_data

def fast_fourier_transform(data, cybres):

    """
    function to calculate the complete FFT: It includes the transformation from time to frequency domain, removing
    unnecessary frequencies, converting it back to the time domain, cut the set to the desired 30 min and writing it
    back into the data.
    """

    fft1, fft2 , freq = [], [], []

    if cybres:
        for i in range(len(data)):
            length = len(data[i]['differential_potential_CH1'])
            sampling_rate = 0.1
            freq.append(np.fft.rfftfreq(length, 1/sampling_rate))
            fft1.append(np.fft.rfft(data[i]['differential_potential_CH1']))
            fft2.append(np.fft.rfft(data[i]['differential_potential_CH2']))
        fft1 = remove_hz(fft1, freq, True)
        fft2 = remove_hz(fft2, freq, True)

        '''
        plt.figure(figsize=(10, 5))
        plt.plot(freq[0], np.abs(fft1[0]))
        plt.show()
        '''

        transform_CH1, transform_CH2 = [], []

        for i in range(len(fft1)):
            transform_CH1.append(np.fft.irfft(fft1[i], n=len(data[i]['differential_potential_CH1'])))
            transform_CH2.append(np.fft.irfft(fft2[i], n=len(data[i]['differential_potential_CH2'])))

        '''
        plt.figure(figsize=(10, 5))
        plt.plot(transform_CH1[0])
        plt.plot(transform_CH2[0])
        plt.plot(np.arange(len(data[0])), data[0]['differential_potential_CH1'])
        plt.plot(np.arange(len(data[0])), data[0]['differential_potential_CH2'])
        plt.show()
        '''

        plt.figure(figsize=(10, 5))
        plt.title('Raw and Transformed Data for CH1 and CH2 of the CYBRES Sensor')
        plt.plot(np.arange(178), data[1]['differential_potential_CH1'].iloc[60:238], label='CH1')
        plt.plot(np.arange(178), data[1]['differential_potential_CH2'].iloc[60:238], label='CH2')

        data_trans = to_df(data, transform_CH1, transform_CH2, True)

        plt.plot(np.arange(len(data_trans[1])), data_trans[1]['differential_potential_CH1'], label='Transformed CH1')
        plt.plot(np.arange(len(data_trans[1])), data_trans[1]['differential_potential_CH2'], label='Transformed CH2')
        plt.xlabel('Number of samples')
        plt.ylabel('Electrical potential')
        plt.legend()
        plt.show()

    else:
        for i in range(len(data)):
            length = len(data[i]['differential_potential'])
            sampling_rate = 280
            freq.append(np.fft.rfftfreq(length, 1/sampling_rate))
            fft1.append(np.fft.rfft(data[i]['differential_potential']))

        fft1 = remove_hz(fft1, freq, False)

        transform = []

        for i in range(len(fft1)):
            transform.append(np.fft.irfft(fft1[i], n=len(data[i]['differential_potential'])))

        data_trans = to_df(data, transform, None, False)

    return data_trans


def remove_hz(fft, freq, cybres):

    """
    helper function to remove the undesired frequencies from the dataset
    """

    if cybres:
        cut = 0.006
    else:
        cut = 25

    for i in range(len(fft)):
        fft[i] = np.where(freq[i] <= cut, fft[i], 0)
    return fft


def to_df(data_original, data_transformed_1, data_transformed_2, cybres):

    """
    helper function to convert the data back and write it to the original data
    """

    if cybres:
        for i in range(len(data_transformed_1)):
            index = data_original[i].index
            data_original[i]['differential_potential_CH1'] = pd.DataFrame(data=data_transformed_1[i], index=index, columns=['differential_potential_CH1'])
            data_original[i]['differential_potential_CH2'] = pd.DataFrame(data=data_transformed_2[i], index=index, columns=['differential_potential_CH2'])
    else:
        for i in range(len(data_transformed_1)):
            index = data_original[i].index
            data_original[i]['differential_potential'] = pd.DataFrame(data=data_transformed_1[i], index=index, columns=['differential_potential'])
    return cut_length(data_original)


def cut_length(data):

    """
    helper function which cuts the data to 30 min interval
    """

    interval = []
    for df in data:
        date = df['timestamp'].iloc[0].replace(second=0)
        start = pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=date.hour,
                             minute=date.minute) + pd.Timedelta(minutes=10)
        end = start + pd.Timedelta(minutes=30)

        interval.append(df[(df['timestamp'] >= start) & (df['timestamp'] < end)].copy())
    return interval


if __name__ == '__main__':
    home_dir = '/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Results/CSV/'
    data = load_cybres(home_dir + '2-Day-test/CYBRES', False, True)
    data_sub = background_subtract(data, True)
    print(fast_fourier_transform(data_sub, True))

    #fast_fourier_transform(data, cybres=False)

    data2 = load_cybres(home_dir + '2-Day-test/PN/P5', False, False)
    #data_sub = background_subtract(data2, False)
    #print(fast_fourier_transform(data_sub, False))



