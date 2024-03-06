import numpy as np
import pandas as pd
import scipy.stats as stats
from data_builder import *
import tsfel

def features(data, cybres):
    means = []
    for i in range(len(data)):
        if cybres:
            data_ch1 = data[i][['timestamp', 'differential_potential_CH1']]
            data_ch2 = data[i][['timestamp', 'differential_potential_CH1']]

            data_ch1.drop(['timestamp'])
            data_ch2.drop(['timestamp'])

            intervals = []
            for start in [0, 10, 20]:
                interval = get_first_min_interval(data_ch1, start, start + 10)
                intervals.append(interval)

            ch1_mean_1 = tsfel.calc_mean(intervals[0])
            ch1_mean_2 = tsfel.calc_mean(intervals[1])

            ch2_mean = np.mean(data_ch2)
            ch1_variance = np.var(data_ch1)
            ch2_variance = np.var(data_ch2)
            ch1_skweness = stats.skew(data_ch1)
            ch2_skewness = stats.skew(data_ch2)
            ch1_kurtosis = stats.kurtosis(data_ch1)
            ch2_kurtosis = stats.kurtosis(data_ch2)
            means.append((ch1_mean, ch2_mean, ch1_variance, ch2_variance, ch1_skweness, ch2_skewness, ch1_kurtosis,
                          ch2_kurtosis))

    return means

def get_first_min_interval(data, start, ending):
    date = data['timestamp'].iloc[0].replace(second=0, microsecond=0)
    start = date + pd.Timedelta(minutes=start)
    end = start + pd.Timedelta(minutes=ending)
    interval = data[(data['timestamp'] >= start) & (data['timestamp'] < end)].copy()

    return interval

if __name__ == '__main__':
    home_dir = '/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Results/CSV/'
    data = load_cybres(home_dir + '2-Day-test/CYBRES', False, True)
    processed_data = fast_fourier_transform(background_subtract(data, True), True)
    #print(processed_data)

    #data2 = load_cybres(home_dir + '2-Day-test/PN/P5', False, False)
    #processed_data2 = background_subtract(data2, False)

    features(processed_data, True)
    #print(mean(processed_data2, False))
