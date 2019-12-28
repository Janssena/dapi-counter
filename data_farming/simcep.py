import os
import sys
import cv2
import glob
import numpy as np
from definitions import PATH
from data_farming.helpers import write_to_csv
from data_farming.helpers import find_centroids_in_image

# This script processes the simcep data and returns all the data in a single
# csv. It should only run if the csv is not already created otherwise we get
# duplicate entries in the csv

if os.path.exists(PATH['SIMCEP_CSV']):
    print('File already exists, aborting...')
    sys.exit()

for subfolder in ('4-var', '6-var', '8-var', '12-var'):
    images_path = os.path.join(
        PATH['SIMCEP'], str(subfolder), '*_BW_*.png')
    images = glob.glob(images_path)
    csv_path = PATH['SIMCEP_CSV']
    for image in images:
        filename = image.split('/')[-1]
        image_id, _, count = filename[:-4].split('_')
        image_id = int(image_id)
        count = int(count) if int(count) >= 0 else 0
        # only take the blue channel
        img = cv2.imread(image)
        img = np.array(img)[:, :, 0]
        locations = find_centroids_in_image(
            img, 0, count) if count > 0 else []
        assert len(locations) == count
        line = {'image_id': image_id, 'path': os.path.join(
            subfolder, filename), 'count': count, 'locations': locations}
        write_to_csv(line, csv_path)
    print('Processed all images in /{}'.format(subfolder))
print('Succesfully processed the simcep images')
