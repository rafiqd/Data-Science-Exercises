import sys

import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv('data.csv', header=0, names=['variable', 'value'])
    posthoc = pairwise_tukeyhsd(
        data['value'],
        data['variable'],
        alpha=0.5
    )
    print(posthoc)
    fig = posthoc.plot_simultaneous()
    plt.show()

if __name__ == '__main__':
    main()
