import os
import cv2
import numpy as np
from definitions import PATH
from data_farming.helpers import csv_to_df


def simcep():
    """
    Grabs and shuffles the SIMCEP data.
    Returns a array containing the image paths and their corresponding labels
    """
    data = []
    labels = []
    csv_path = PATH['SIMCEP_CSV']
    data_root = PATH['SIMCEP']
    df = csv_to_df(csv_path)
    df_shuffle = df.sample(frac=1).reset_index(drop=True)
    for i, file in enumerate(df_shuffle['path']):
        img_path = os.path.join(data_root, file)
        image = cv2.imread(img_path)[:, :, :1]
        label = int(df_shuffle['count'][i])
        data.append(image)
        labels.append(label)
    return np.array(data), np.array(labels)
