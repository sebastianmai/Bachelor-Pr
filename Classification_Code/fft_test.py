import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

#matplotlib.use('Qt5Agg')

def fast_fourier_transform(data, cybres):
    fft1, fft2 = [], []
    freq = []

    if cybres:
        length = len(data['differential_potential_CH1'])
        print(len(data['differential_potential_CH1']))
        sampling_rate = 0.1
        freq.append(np.fft.rfftfreq(length, 1/sampling_rate))
        fft1.append(np.fft.rfft(data['differential_potential_CH1']))
        fft2.append(np.fft.rfft(data['differential_potential_CH2']))

        fft1 = remove_hz(fft1, freq)
        fft2 = remove_hz(fft2, freq)

        print(len(fft1), len(fft2))

        plt.figure(figsize=(10, 5))
        plt.plot(freq[0], np.abs(fft1))
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(freq[0], np.abs(fft2))
        plt.show()

        transform_CH1, transform_CH2 = [], []

        transform_CH1.append(np.fft.irfft(fft1))
        transform_CH2.append(np.fft.irfft(fft2))

        print(len(transform_CH1))

def remove_hz(fft, freq):
    fft = np.where(freq[0] <= 0.006, fft[0], 0)
    return fft

if __name__ == '__main__':
    data = pd.read_csv('test.csv')
    data = data.iloc[:, 1:]
    #print(data[0:10])

    fast_fourier_transform(data, True)