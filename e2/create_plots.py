import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def wiki_plots(filename1, filename2):

    data1 = pd.read_table(filename1, sep=' ', header=None, index_col=1, names=['lang', 'page', 'views', 'bytes'])
    data2 = pd.read_table(filename2, sep=' ', header=None, index_col=1, names=['lang', 'page', 'views', 'bytes'])
    data1 = data1.sort_values(by='views', ascending=False)

    merged_data = data1.merge(data2, left_index=True, right_index=True)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, len(data1['views'])), data1['views'],)
    plt.xlabel('Rank')
    plt.ylabel('Views')
    plt.margins(0.05, 0.05)
    plt.title('Popularity Distribution')

    plt.subplot(1, 2, 2)

    plt.plot(merged_data['views_x'], merged_data['views_y'], 'b.')
    plt.xlabel('Day 1 views')
    plt.ylabel('Day 2 views')
    plt.yscale('log')
    plt.xscale('log')
    plt.margins(0.05, 0.05)
    plt.title('Daily Correlation')

    plt.savefig('wikipedia.png')


def main(fn1, fn2):
    wiki_plots(fn1, fn2)

if __name__ == '__main__':
    filename_1 = sys.argv[1]
    filename_2 = sys.argv[2]
    main(filename_1, filename_2)
