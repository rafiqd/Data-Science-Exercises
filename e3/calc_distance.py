import sys
import pandas as pd
import xml.etree.ElementTree as et
import re
import math
import numpy as np
from pykalman import KalmanFilter

def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    from xml.dom.minidom import getDOMImplementation
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')


def get_data(filename):

    tree = et.parse(filename)
    root = tree.getroot()

    match = re.match('\{.*\}', root.tag)
    ns = match.group(0) if match else ''
    element_list = root.findall('.//%strkpt' % ns)
    record_list = []
    for element in element_list:
        record = {
            'lat': float(element.attrib['lat']),
            'lon': float(element.attrib['lon'])
        }
        record_list.append(record)
    return pd.DataFrame(record_list)


def distance(points):
    new_df = pd.merge(points, points.shift(1), right_index=True, left_index=True).fillna(0)
    new_df['distance'] = new_df.apply(haversine, axis=1)
    distance_sum = np.sum(new_df['distance'][1:])
    return distance_sum


def smooth(points):

    initial_state = points.iloc[0]
    observation_covariance = np.diag([0.00020, 0.00020]) ** 2
    transition_covariance = np.diag([0.0001, 0.0001]) ** 2
    transition = [
        [1, 0],
        [0, 1]
    ]
    kf = KalmanFilter(
        initial_state_mean=initial_state,
        transition_matrices=transition,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )
    kalman_smoothed, _ = kf.smooth(points)

    return pd.DataFrame(kalman_smoothed, columns=['lat', 'lon'])

def haversine(df):
    lat1 = math.radians(df['lat_y'])
    lat2 = math.radians(df['lat_x'])
    lon1 = math.radians(df['lon_y'])
    lon2 = math.radians(df['lon_x'])
    r = 6371000  # earth's radius in meters -- maybe not totally right since earth isn't a sphere but close enough
    x = math.sqrt((math.sin((lat2 - lat1)/2)**2) + math.cos(lat1)*math.cos(lat2)*(math.sin((lon2 - lon1)/2) ** 2))
    dist = 2 * r * math.asin(x)
    return dist

def main():
    points = get_data(sys.argv[1])
    print('Unfiltered distance: %0.2f' % (distance(points),))
    smoothed_points = smooth(points)
    print('Filtered distance: %0.2f' % (distance(smoothed_points),))
    output_gpx(smoothed_points, 'out.gpx')


if __name__ == '__main__':
    main()
