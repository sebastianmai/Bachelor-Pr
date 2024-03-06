import numpy as np
import pandas as pd
import scipy.stats as stats
from data_builder import *
import tsfel

def features(data, cybres):
    means = []
    for i in range(0,1):
        if cybres:
            data_ch1 = data[i][['timestamp', 'differential_potential_CH1']]
            data_ch2 = data[i][['timestamp', 'differential_potential_CH1']]

            intervals = []
            for start in [0, 10, 20]:
                interval = get_first_min_interval(data_ch1, start, 10)
                intervals.append(interval)
            print(type(intervals[0]))
            ch1_mean_1 = tsfel.calc_mean(intervals[0])
            ch1_mean_2 = tsfel.calc_mean(intervals[1])
            ch2_mean_3 = tsfel.calc_mean(intervals[2])
    return means

def get_first_min_interval(data, start, ending):
    date = data['timestamp'].iloc[0].replace(second=0, microsecond=0)
    start = date + pd.Timedelta(minutes=start)
    end = start + pd.Timedelta(minutes=ending)
    interval = data[(data['timestamp'] >= start) & (data['timestamp'] < end)].copy().drop(columns=['timestamp'])

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
