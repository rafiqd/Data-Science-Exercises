import sys
import pandas as pd
import numpy as np
import difflib
from pprint import pprint

def main(movie_list_filename, movie_ratings_filename, output_filename):
    movie_list = open(movie_list_filename).readlines()

    def convert_title(dataframe):
        title = dataframe['title']
        x = difflib.get_close_matches(title, movie_list, n=1)
        if len(x) == 0:
            dataframe['title'] = np.nan
        else:
            dataframe['title'] = x[0].replace('\n', '')
        try:
            rating = float(dataframe['rating'])
        except:
            rating = np.nan
        dataframe['rating'] = rating
        return dataframe

    movie_ratings = open(movie_ratings_filename).readlines()
    movie_ratings_series = pd.Series(movie_ratings)
    movie_ratings_series = movie_ratings_series.str.replace('\n', '')
    movie_ratings_series = movie_ratings_series[1:]
    data = movie_ratings_series.str.rsplit(',', expand=True, n=1)
    data = data.rename(columns={0: 'title', 1: 'rating'})
    newdata = data.apply(convert_title, axis=1)
    clean_data = newdata.dropna(axis=0)
    grouped = clean_data.groupby(['title'])
    averages = grouped.mean().reset_index().round(2)
    sorted_averages = averages.sort_values(['title'])
    sorted_averages.to_csv(output_filename, index=False)
    return



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
