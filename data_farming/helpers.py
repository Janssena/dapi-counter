import re
import os
import csv
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def csv_to_df(path):
    df = pd.read_csv(path, header=0)
    df.crop = df.crop.apply(lambda x: eval(str(x)))
    df.original_size = df.original_size.apply(lambda x: eval(str(x)))
    df.locations = df.locations.apply(lambda x: eval(str(x)))
    return df


def is_processed(filename, processed_files):
    props = get_properties_from_filename(filename)
    name = '{}-{}-[{}].png'.format(props["id"],
                                   props["crop"], str(props["slice"]))
    return True if any(line.endswith(name) for line in processed_files) else False


def get_properties_from_filename(filename):
    name = filename.split('/')[-1].split(" ")[-1][:-4]
    id = re.sub(r".GFP.*\w{3}\d{3}", "", name)
    id = re.sub("_width.*", "", id).replace('_', '-')
    (width, d, interval, slice) = name.split('_')[-4:]

    return {
        "name": '',
        "id": id,
        "crop": [0, 0, 256, 256],
        "original_size": (),
        "slice": int(_get_digits(slice)),
        "pixel_width": float(_get_digits(width)),
        "height_interval": float(_get_digits(interval)),
        "count": 0,
        "locations": []
    }


def _get_digits(s):
    return re.sub(r"[^(\d|.)]", "", s)


def write_to_csv(data, file):
    file_exist = os.path.isfile(file)
    with open(file, 'a') as csv_file:
        header = [*data.keys()]
        writer = csv.DictWriter(csv_file, fieldnames=header)
        if not file_exist:
            writer.writeheader()
        writer.writerow(data)


def find_centroids_in_image(image, threshold, n_clusters, n_init=50):
    if type(image) is not np.ndarray:
        raise Exception('image should be a numpy array.')
    points = np.where(image > threshold)
    x = points[1]
    y = points[0]
    points_len = len(x)
    points = [[x[i], y[i]] for i in range(points_len)]
    kmeans = KMeans(init='k-means++',
                    n_clusters=n_clusters, n_init=n_init).fit(points)
    return kmeans.cluster_centers_
