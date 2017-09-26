__author__ = 'rdandoo'

import sys
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.statespace.kalman_smoother import KalmanFilter
import numpy as np
from pykalman import KalmanFilter


def main(filename):
    cpu_data = pd.read_table(
        filename,
        sep=',',
        header=0,
        names=['cpu_percent', 'sys_load_1', 'temperature', 'timestamp'],
        parse_dates=['timestamp']
    )

    kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1']]
    initial_state = kalman_data.iloc[0]
    observation_covariance = np.diag([1, 1, 1]) ** 2
    transition_covariance = np.diag([0.25, 0.25, 0.25]) ** 2
    transition = [
        [1, -0.27, 0],
        [0, -1.14, 0.85],
        [0, 0.37, 0.06]
    ]
    kf = KalmanFilter(
        initial_state_mean=initial_state,
        transition_matrices=transition,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )
    kalman_smoothed, _ = kf.smooth(kalman_data)
    col = 'temperature'
    loess_smoothed = lowess(
        endog=cpu_data[col],
        exog=cpu_data['timestamp'],
        frac=0.025
    )
    plt.figure(figsize=(12, 4))
    plt.plot(cpu_data['timestamp'], cpu_data[col], 'b.', alpha=0.5, label='Temperature')
    plt.plot(cpu_data['timestamp'], loess_smoothed[:, 1], 'r-', label='Loess Smoothed')
    plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-', label='Kalman Smoothed')
    plt.legend(loc='upper left', shadow=True, fontsize='large')
    plt.show()  # easier for testing
    plt.savefig('cpu.svg')  # for final submission


if __name__ == '__main__':
    main(sys.argv[1])
