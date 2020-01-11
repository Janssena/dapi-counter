from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
from model.count import count_model
import matplotlib.pyplot as plt
from definitions import PATH
import numpy as np
import get_data
import sys
import cv2
import os

image_path = sys.argv[1]
label = image_path.split('_')[2][:-4]
image = cv2.imread(image_path)[:, :, :1]
test = np.expand_dims(image, axis=0)

BATCH_SIZE = 16
INPUT_SHAPE = (256, 256, 1)

model_path = os.path.join(PATH["MODEL_OUT"], 'count_05-01-2020.h5')
# data, labels = get_data.simcep()

# data = data[:5]
# labels = labels[:5]

model = count_model(INPUT_SHAPE)
model.load_weights(model_path)

# predictions = model.predict(data, batch_size=BATCH_SIZE)
prediction = model.predict(test)

# labels = labels * 100
# predictions = predictions * 100
# mae = mean_absolute_error(labels, predictions)
# print('mae:', mae)

# for i in range(5):
# plt.imshow(data[i][:, :, 0])
# print('true:', labels[i], ', predict:', predictions[i],
# ', diff:', (labels[i] - predictions[i]))
# plt.show()

plt.imshow(cv2.imread(image_path))
print('true:', label, 'predict:', prediction)
plt.show()
