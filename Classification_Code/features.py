import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from data_builder import *
import tsfel
import pywt


matplotlib.use('Qt5Agg')

def features(data, cybres):

    features_cybres = {"CH1_mean": [], "CH2_mean": [],
                       "CH1_var": [], "CH2_var": [],
                       "CH1_skew": [], "CH2_skew": [],
                       "CH1_kurtosis": [], "CH2_kurtosis": [],
                       "CH1_iqr": [], "CH2_iqr": [],
                       "CH1_wpe": []}
    features_phyto = {"mean": [],
                      "variance": [],
                      "skew": [],
                      "kurtosis": [],
                      "iqr": [],
                      "wpe": []}

    if cybres:
        for i in range(len(data)):
            data_ch1 = data[i][['timestamp', 'differential_potential_CH1']]
            data_ch2 = data[i][['timestamp', 'differential_potential_CH2']]

            intervals_CH1, intervals_CH2 = [], []
            for start in [0, 10, 20]:
                interval = get_first_min_interval(data_ch1, start, 10)
                intervals_CH1.append(interval)
                interval = get_first_min_interval(data_ch2, start, 10)
                intervals_CH2.append(interval)

            features_cybres["CH1_mean"].append([tsfel.calc_mean(interval) for interval in intervals_CH1])
            features_cybres["CH2_mean"].append([tsfel.calc_mean(interval) for interval in intervals_CH2])
            features_cybres["CH1_var"].append([np.var(interval['differential_potential_CH1'], axis=0) for interval in intervals_CH1])  # needed np.var since TSFEL doesnt allow axis specification resulting in depreciation warnings
            features_cybres["CH2_var"].append([np.var(interval['differential_potential_CH2'], axis=0) for interval in intervals_CH2])  # needed np.var since TSFEL doesnt allow axis specification resulting in depreciation warnings
            features_cybres["CH1_skew"].append([tsfel.skewness(interval)[0] for interval in intervals_CH1])
            features_cybres["CH2_skew"].append([tsfel.skewness(interval)[0]for interval in intervals_CH2])
            features_cybres["CH1_kurtosis"].append([tsfel.kurtosis(interval)[0] for interval in intervals_CH1])
            features_cybres["CH2_kurtosis"].append([tsfel.kurtosis(interval)[0] for interval in intervals_CH2])
            features_cybres["CH1_iqr"].append([tsfel.interq_range(interval) for interval in intervals_CH1])
            features_cybres["CH2_iqr"].append([tsfel.interq_range(interval) for interval in intervals_CH2])
            features_cybres["CH1_wpe"].append([wavelet_entropy(interval) for interval in intervals_CH1])
            #features_cybres["CH2_iqr"].append([tsfel.wavelet_entropy(interval) for interval in intervals_CH2])


        return features_cybres


    else:
        for i in range(0, 2):
            data_phyto = data[i][['timestamp', 'differential_potential']]

            intervals = []
            for start in [0, 10, 20]:
                interval = get_first_min_interval(data_phyto, start, 10)
                intervals.append(interval)

            features_phyto["mean"].append([tsfel.calc_mean(interval) for interval in intervals])
            features_phyto["variance"].append([tsfel.calc_var(interval) for interval in intervals])
            features_cybres["skew"].append([tsfel.skewness(interval) for interval in intervals])
            features_cybres["kurtosis"].append([tsfel.kurtosis(interval) for interval in intervals])

        return features_phyto


def get_first_min_interval(data, start, ending):
    date = data['timestamp'].iloc[0].replace(second=0, microsecond=0)
    start = date + pd.Timedelta(minutes=start)
    end = start + pd.Timedelta(minutes=ending)
    interval = data[(data['timestamp'] >= start) & (data['timestamp'] < end)].copy().drop(columns=['timestamp'])

    return interval


def wavelet_entropy(signal):
    coefficients, frequencies = pywt.dwt(signal['differential_potential_CH1'], 'db1')
    entropy = -np.sum((coefficients ** 2)*(np.log(coefficients**2)))
    return entropy


def normalize(feature):
    return (feature - feature.min()) / (feature.max() - feature.min())


if __name__ == '__main__':
    home_dir = '/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Results/CSV/'
    #home_dir = '/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Results/CSV/Final/BLUE/measurements'

    data = load_cybres(home_dir + '2-Day-test/CYBRES', False, True)
    #data = load_cybres(home_dir, False, True)
    processed_data = fast_fourier_transform(background_subtract(data, True), True)
    f = features(processed_data, True)
    #print(f)

    res = []
    for column in f:
        columns = []
        for i in [0, 1, 2]:
            columns.append(f"{column}_{i}")
        res.append(pd.DataFrame(f[column], columns=[columns]))

    result = pd.concat(res, axis=1)

    for index in result:
        result[index] = normalize(result[index])

    print('smth')
    result.to_csv('test.csv')

    #result = pd.DataFrame(f)
    #result.to_csv("test.csv")

    #print(f['CH1_mean'])
    mean = normalize(pd.DataFrame(f['CH1_mean']))
    mean2 = normalize(pd.DataFrame(f['CH2_mean']))

    #print(f['CH1_var'])
    var = normalize(pd.DataFrame(f['CH1_var']))
    var2 = normalize(pd.DataFrame(f['CH2_var']))

    kurtosis = normalize(pd.DataFrame(f['CH1_kurtosis']))
    kurtosis2 = normalize(pd.DataFrame(f['CH2_kurtosis']))

    #print(normalize(mean))

    #print(processed_data)

    #dataP5 = load_cybres(home_dir + '2-Day-test/PN/P5', False, False)
    #processed_data2 = background_subtract(data2, False)


    #print(mean(processed_data2, False))


    #print(f['CH1_mean'])
    #print(f['CH1_wpe'])

    '''
    plt.figure(figsize=(10, 5))
    plt.scatter(mean[0], var[0], color='blue', label='Pre CH1')
    plt.scatter(mean2[0], var2[0], color='red', label='Pre CH2')
    plt.scatter(mean[1], var[1], color='blue', label='Stim CH1', marker='x')
    plt.scatter(mean2[1], var2[1], color='red', label='Stim CH2', marker='x')
    plt.scatter(mean[2], var[2], color='blue', label='Post CH1', marker='^')
    plt.scatter(mean2[2], var2[2], color='red', label='Post CH2', marker='^')
    plt.xlabel("mean")
    plt.ylabel("variance")
    plt.legend()
    plt.grid(True)
    #plt.show()


    plt.figure(figsize=(10, 5))
    plt.scatter(kurtosis[0], var[0], color='blue', label='Pre CH1')
    plt.scatter(kurtosis2[0], var2[0], color='red', label='Pre CH2')
    plt.scatter(kurtosis[1], var[1], color='blue', label='Stim CH1', marker='x')
    plt.scatter(kurtosis2[1], var2[1], color='red', label='Stim CH2', marker='x')
    plt.scatter(kurtosis[2], var[2], color='blue', label='Post CH1', marker='^')
    plt.scatter(kurtosis2[2], var2[2], color='red', label='Post CH2', marker='^')
    plt.xlabel("kurtosis")
    plt.ylabel("variance")
    plt.legend()
    plt.grid(True)
    #plt.show()

    plt.figure(figsize=(10, 5))
    plt.scatter(kurtosis[0], mean[0], color='blue', label='Pre CH1')
    plt.scatter(kurtosis2[0], mean2[0], color='blue', label='Pre CH2')
    plt.scatter(kurtosis[1], mean[1], color='red', label='Stim CH1', marker='x')
    plt.scatter(kurtosis2[1], mean2[1], color='red', label='Stim CH2', marker='x')
    plt.scatter(kurtosis[2], mean[2], color='green', label='Post CH1', marker='^')
    plt.scatter(kurtosis2[2], mean2[2], color='green', label='Post CH2', marker='^')
    plt.xlabel("kurtosis")
    plt.ylabel("mean")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    '''