import sys
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main(stations_filename, city_data_filename, output_filename):

    station_fh = gzip.open(stations_filename, 'rt', encoding='utf-8')
    stations = pd.read_json(station_fh, lines=True)

    city_data = pd.read_csv(city_data_filename).dropna(subset=['population', 'area'])
    city_data = city_data.apply(convert_area, axis=1)
    city_data = city_data[city_data['area'] <= 10000]
    city_data = city_data.apply(best_tmax, stations=stations, axis=1)
    city_data['avg_tmax'] = np.divide(city_data['avg_tmax'], 10)
    city_data['density'] = np.divide(city_data['population'], city_data['area'])

    plt.plot(city_data['avg_tmax'], city_data['density'], 'b.')
    plt.xlabel('Avg Max Temperature (\u00b0C)')
    plt.ylabel('Population Density (people/km\u00b2)')
    plt.title('Temperature vs Population Density')
    plt.savefig(output_filename)


def convert_area(dataframe):
    try:
        area_km = float(dataframe['area']) / 1000000
    except:
        area_km = np.nan
    dataframe['area'] = area_km
    return dataframe


def best_tmax(city, stations):
    distance_array = distance(city, stations)
    min_val = distance_array.argmin()
    city['avg_tmax'] = stations.iloc[min_val]['avg_tmax']
    return city


def distance(city, stations):
    r2 = 12742000  # 2x the radius of earth in meters

    city_lat = np.radians(city['latitude'])
    city_lon = np.radians(city['longitude'])
    station_lat = np.radians(stations['latitude'])
    station_lon = np.radians(stations['longitude'])

    lat_delta = np.subtract(station_lat, city_lat)
    lon_delta = np.subtract(station_lon, city_lon)

    x1 = np.square(np.sin(np.divide(lat_delta, 2)))
    x2 = np.square(np.sin(np.divide(lon_delta, 2)))
    x3 = np.multiply(np.cos(city_lat), np.cos(station_lat))
    x4 = np.multiply(x3, x2)
    x5 = np.arcsin(np.sqrt(np.add(x1, x4)))

    return np.multiply(r2, x5)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
