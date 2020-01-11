from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from model.count import count_model
import matplotlib.pyplot as plt
from definitions import PATH
import get_data
import os

BATCH_SIZE = 16
INPUT_SHAPE = (256, 256, 1)

model_path = os.path.join(PATH["MODEL_OUT"], 'count_05-01-2020_0.39.h5')
data, labels = get_data.simcep()
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.25, random_state=42)

model = count_model(INPUT_SHAPE)
model.load_weights(model_path)

predictions = model.predict(testX, batch_size=BATCH_SIZE)

labels = labels * 100
predictions = predictions * 100

for i in range(5):
    plt.imshow(data[i][:, :, 0])
    print('true:', labels[i], ', predict:', predictions[i],
          ', diff:', (labels[i] - predictions[i]))
    plt.show()
