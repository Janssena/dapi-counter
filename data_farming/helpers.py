import re
import os
import csv
import pandas as pd


def csv_to_df(path):
    df = pd.read_csv(path, header=0)
    df.crop = df.crop.apply(lambda x: eval(str(x)))
    df.original_size = df.original_size.apply(lambda x: eval(str(x)))
    df.locations = df.locations.apply(lambda x: eval(str(x)))
    return df


def is_processed(filename, processed_files):
    props = get_properties_from_filename(filename)
    name = '{}-{}-[{}].png'.format(props["id"], props["crop"], str(props["slice"]))
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
        print(data)
        writer.writerow(data)
        print('csv file updated')
