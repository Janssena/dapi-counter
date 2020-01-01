import os
import cv2
import glob
import numpy as np
from definitions import PATH
from data_farming.helpers import csv_to_df
from sklearn.utils import shuffle


def simcep(exclude=None):
    """
    Grabs and shuffles the SIMCEP data.
    Returns:
    - data (single channel image)
    - labels (cell count as Integer)
    """
    data = []
    labels = []
    csv_path = PATH['SIMCEP_CSV']
    data_root = PATH['SIMCEP']
    df = csv_to_df(csv_path)
    df_shuffle = df.sample(frac=1).reset_index(drop=True)
    for i, file in enumerate(df_shuffle['path']):
        if exclude is not None:
            if file.startswith(exclude):
                continue
        img_path = os.path.join(data_root, file)
        image = cv2.imread(img_path)[:, :, :1]
        label = int(df_shuffle['count'][i])
        data.append(image)
        labels.append(label)
    return np.array(data), np.array(labels)


def simcep_masks(exclude=None):
    """
    Grabs and shuffles the simcep mask data from the x-var subfolders, where
    x is one of [4, 6, 8, 12].
    Optional argument exclude removes on of these entries.
    Returns:
    - data (single channel image)
    - labels (single channel mask image)
    """
    data = []
    labels = []
    data_root = PATH['SIMCEP']
    options = ['4-var', '6-var', '8-var', '12-var']
    if exclude is not None:
        options.remove(exclude)
    for subfolder in options:
        files = os.path.join(data_root, subfolder, '*_RGB_*.png')
        files_shuffle = shuffle(files, random_state=42)
        for rgb_file in files_shuffle:
            bw_file = rgb_file.replace('_RGB_', '_BW_')
            image = cv2.imread(rgb_file)[:, :, :1]
            label = cv2.imread(bw_file)[:, :, :1]
            data.append(image)
            labels.append(label)
    return np.array(data), np.array(labels)


def latest_weights(model):
    """
    Get the latest weights for the specified model.
    Input:
    - model (string representation of model)
    Outputs:
    - path to weights file (to load using model.load_weights)
    """
    weights_path = os.path.join(PATH['MODEL_OUT'], '{}*.h5'.format(model))
    files = glob.glob(weights_path)
    files.sort(key=lambda x: os.path.getmtime(x))
    latest = files[-1]
    date = latest.split('_')[-1][:-3]
    print('[INFO] taking {} model from {} to continue training'.format(
        model, date))
    return latest
