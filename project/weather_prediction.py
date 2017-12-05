import pandas as pd
import numpy as np
import sys
import os
import re
from skimage.color import rgb2gray
from skimage import io
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt


def transform_weather(weather_val):
    weather_val = re.sub('(Mainly )|(Mostly )|( Showers)|(Moderate )|(Freezing )|( Pellets)|(Heavy )', '', weather_val)
    weather_val = re.sub('(Drizzle|Thunderstorms|Cloudy)', 'Rain', weather_val)
    weather_val = re.sub('(Rain,Rain)', 'Rain', weather_val)
    return weather_val


def main(weather_data_dir, katkam_dir):

    weather_files = os.listdir(weather_data_dir)
    df = pd.concat((pd.read_csv(os.path.join(weather_data_dir, f), header=14, parse_dates=['Date/Time']) for f in weather_files))
    df = df[df['Weather'].notnull()]
    weather_df = df[['Date/Time', 'Time', 'Weather']].copy()
    images = katkam_dir + '/*.*'
    x_images = io.imread_collection(images)
    images = pd.DataFrame({'filename': x_images.files, 'img': np.arange(0, len(x_images.files)) })
    images['Date/Time'] = pd.to_datetime(images['filename'].str.extract('-([0-9]+)\.', expand=False), format='%Y%m%d%H%M%S')
    images = images.merge(weather_df, on='Date/Time')
    images['Weather'] = images['Weather'].apply(transform_weather)

    data = []
    target = []
    filenames = []

    # need to do this for loop to get the images out of x_images or else we'd need to load the whole
    # x_images array instead of just the images we have data for
    for i, x in images.iterrows():
        matrix = rgb2gray(x_images[x['img']])
        matrix = np.reshape(matrix, (192*256))
        data.append(matrix)
        target.append(x['Weather'].split(','))
        filenames.append(x['filename'])

    mlb = MultiLabelBinarizer()
    y_enc = mlb.fit_transform(target)

    X_train, X_test, y_train, y_test, idx1, idx2 = train_test_split(np.array(data), y_enc, np.array(filenames))

    model = make_pipeline(
        PCA(250),
        KNeighborsClassifier(n_neighbors=15)
    )

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print("KNN Model Score: %f" % model.score(X_test, y_test))
    result = np.empty(predicted.shape[0], dtype=np.bool)

    for i, (x,y) in enumerate(zip(predicted, y_test)):
        result[i] = np.array_equal(x,y)

    wrong = mlb.inverse_transform(predicted[~result])
    real = mlb.inverse_transform(y_test[~result])
    results_df = pd.DataFrame({'filename': idx2[~result], 'predicted': wrong, 'actual': real})
    aggregated = results_df.groupby(['predicted', 'actual']).count().rename(columns={'filename': 'Predicted Incorrectly'})
    aggregated.plot.bar()
    plt.tight_layout()
    plt.legend()
    plt.savefig('errors.png')

    correct = mlb.inverse_transform(predicted[result])
    real = mlb.inverse_transform(y_test[result])
    results_df = pd.DataFrame({'filename': idx2[result], 'predicted': correct, 'actual': real})
    aggregated = results_df.groupby(['predicted', 'actual']).count().rename(columns={'filename': 'Predicted Correctly'})
    aggregated.plot.bar()
    plt.tight_layout()
    plt.legend()
    plt.savefig('correct.png')


if __name__ == '__main__':
    weather_dir = sys.argv[1]
    katkam_dir = sys.argv[2]
    main(weather_dir, katkam_dir)

#'%Y%m%d%H%M%S'