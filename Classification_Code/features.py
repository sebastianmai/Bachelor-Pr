import pandas as pd
from scipy.signal import welch
from data_builder import *
import tsfel
import pywt
import antropy as ant

matplotlib.use('Qt5Agg')


def features(data, cybres, channel):
    features_phyto = {"mean": [],
                      "variance": [],
                      "skew": [],
                      "kurtosis": [],
                      "iqr": [],
                      "wpe": [],
                      "mobility": [],
                      "complexity": [],
                      "asp": []}

    for i in range(0,len(data)):
        if cybres and channel:
            ch = "differential_potential_CH1"
        elif cybres and not channel:
            ch = "differential_potential_CH2"
        else:
            ch = "differential_potential"
        data_phyto = data[i][['timestamp', ch]]

        intervals = []
        for start in [0, 10, 20]:
            interval = get_first_min_interval(data_phyto, start, 10)
            intervals.append(interval)

        features_phyto["mean"].append([tsfel.calc_mean(interval) for interval in intervals])
        features_phyto["variance"].append(
            [np.var(interval[ch], axis=0) for interval in intervals])
        features_phyto["skew"].append([tsfel.skewness(interval)[0] for interval in intervals])
        features_phyto["kurtosis"].append([tsfel.kurtosis(interval)[0] for interval in intervals])
        features_phyto["iqr"].append([tsfel.interq_range(interval) for interval in intervals])
        features_phyto["wpe"].append([wavelet_entropy(interval, ch) for interval in intervals])
        features_phyto["mobility"].append([ant.hjorth_params(interval, axis=0)[0][0] for interval in intervals])
        features_phyto["complexity"].append([ant.hjorth_params(interval, axis=0)[1][0] for interval in intervals])
        features_phyto["asp"].append([calculate_ASP(interval, ch) for interval in intervals])

    return features_phyto


def get_first_min_interval(data, start, ending):
    date = data['timestamp'].iloc[0].replace(second=0, microsecond=0)
    start = date + pd.Timedelta(minutes=start)
    end = start + pd.Timedelta(minutes=ending)
    interval = data[(data['timestamp'] >= start) & (data['timestamp'] < end)].copy().drop(columns=['timestamp'])

    return interval


def wavelet_entropy(signal, name):
    coefficients, frequencies = pywt.dwt(signal[name], 'db1')
    entropy = -np.sum((coefficients ** 2) * (np.log(coefficients ** 2)))
    return entropy


def calculate_ASP(signal, name):
    s, PSD = welch(signal[name], fs=280, nperseg=50)
    ASP = np.sum(PSD)

    return ASP


def normalize(feature):
    return (feature - feature.min()) / (feature.max() - feature.min())


if __name__ == '__main__':
    home_dir = '/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Results/CSV/Final/HEAT'
    data = load_cybres(home_dir + '/Measurements/P5', False, False)
    f = f_CH1 = features(fast_fourier_transform(background_subtract(data, False), False), False, False)
    #data = load_cybres(home_dir + '/measurements', False, True)
    #f_CH1 = features(fast_fourier_transform(background_subtract(data, True), True), True, False)


    res = []
    for column in f_CH1:
        columns = []
        for i in [0, 1, 2]:
            columns.append(f"{column}_{i}")
        res.append(pd.DataFrame(f_CH1[column], columns=[columns]))
    result = pd.concat(res, axis=1)

    # normalize
    for index in result:
        result[index] = normalize(result[index])

    #bring it into the correct format for classification
    keys = result.columns.get_level_values(0)
    combined = pd.DataFrame()

    for i in range(0, len(keys), 3):
        val = []
        for j in range(len(result)):
            val.extend([result.iloc[j, i], result.iloc[j, i + 1], result.iloc[j, i + 2]])
        column_name = keys[i].replace("_0", "")
        combined[column_name] = val

    # add class column
    class_col = [i % 3 for i in range(len(combined))]
    combined['class'] = class_col[:len(combined)]

    combined.to_csv('/home/basti/DATEN/Universität/Bachelor/Projekt/Bachelor-Pr/Features/HEAT/Phyto_BLUE_P5.csv')

