# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import re
import numpy as np
from scipy.fftpack import fft, fftfreq, ifft
from numpy import cos, sin, pi
# import heapq
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
# import mglearn
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import GridSearchCV

# def load_files(data_path,p_num):
#     data_files = os.listdir(data_path+participant[p_num])
#     regex = re.compile('xethru_respiration_(\d{8})_(\d{6}).csv')
#     # regex = re.compile('DataT(\d+).csv')
#
#     matches = [m for m in map(regex.match, data_files) if m is not None]
#
#     for match in matches:
#         # data_raw = pd.read_csv(data_path+'\\'+match.group(0), header=14, delimiter=';')
#         data_raw = pd.read_csv(data_path+participant[p_num]+'\\'+match.group(0), header=14, delimiter=';')
#
#     return data_raw


def fft_respiration(data, sampling_rate):
    k = np.arange(len(data))
    T = len(data)/sampling_rate #timestamp[-1]
    freq = k/T
    freq = freq[range(int(len(data)/2))]
    twosidedY = fft(np.array(data))/len(data)
    onesidedY = 2*twosidedY[range(int(len(data)/2))]

    return freq, onesidedY, twosidedY


def ifft_respiration(N, twosidedY):
    origin = np.ones(N) * twosidedY.real[0]
    Xm = np.zeros(N)

    for m in range(N):
        for n in range(N):
            Xm[m] += twosidedY.real[n]*cos(2*pi*m*n/N) - twosidedY.imag[n]*sin(2*pi*m*n/N)

    iffty = origin + Xm

    return iffty


def feature_extraction(data, window_size, sampling_rate):
    pre_window = data[-2*window_size:-window_size]
    curr_window = data[-window_size:]

        #### normalization 추가
    norm_pre = np.array((pre_window - pre_window.mean()) / (pre_window.std() + 0.0000001))
    norm_curr = np.array((curr_window - curr_window.mean()) / (curr_window.std() + 0.0000001))

    freq, onesidedY, twosidedY = fft_respiration(curr_window, sampling_rate)  # FFT
    # freq, onesidedY, twosidedY = fft_respiration(norm_curr, sampling_rate)  # FFT
    amplitude = abs(onesidedY)

    idx = np.argpartition(amplitude, -2)[-2:]

    F1 = freq[idx[-1]]  # first harmonic
    F2 = freq[idx[-2]]  # second harmonic

    A0 = amplitude[0]  # dc
    A1 = amplitude[1]  # amplitude at respiration freq (8Hz: 0.25, 17Hz:0.266)

    # Time domain features #
    # 1. auto correlation
    corr = np.corrcoef(pre_window.astype(float), curr_window.astype(float))[0][1]
    # corr = np.corrcoef(norm_pre.astype(float), norm_curr.astype(float))[0][1]
    if np.isnan(corr):
        corr = 2

    return [curr_window[-1], A0, A1, F1, F2, corr]
    # return [F1, F2, A0, A1, A2, corr]



