from model.model import count_net
import definitions
from data_farming.helpers import csv_to_df

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os


def check(image, locations, count):
    """ function that checks if the data is loaded correctly """
    img_tmp = image[:, :, 0]
    plt.imshow(img_tmp)
    x = [loc[0] for loc in locations]
    y = [loc[1] for loc in locations]
    plt.plot(x, y, 'yX')
    plt.axis('off')
    print('len(locations) == count? =>', len(locations) == count)
    assert len(locations) / \
        93 == count, 'The cell count and number of locations do not match'
    print('image shape:', image.shape)
    plt.show()


data = []
labels = []
locs = []

df = csv_to_df(definitions.PATH['CSV'])
df_shuffle = df.sample(frac=1).reset_index(drop=True)

for i, image in enumerate(df_shuffle['name']):
    im_path = os.path.join(definitions.PATH['IMAGE_OUT'] + image)
    img = cv2.imread(im_path)[:, :, :1]
    # shape is (256, 256, 1)
    data.append(img)
    label = df_shuffle['count'][i]
    loc = df_shuffle['locations'][i]
    labels.append(label)
    locs.append(loc)

data = np.array(data, dtype="float")
labels = np.array(labels) / max(labels)
locs = np.array(locs)

for i in range(5):
    check(data[i], locs[i], labels[i])

(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.25, random_state=42)

opt = tf.keras.optimizers.Adam(lr=1e-5, decay=1e-5 / 200)

aug = tf.keras.preprocessing.image.ImageDataGenerator(
    samplewise_center=True,
    horizontal_flip=True,
    vertical_flip=True,
)

BS = 16
EPOCHS = 20

model = count_net((256, 256, 1))
model.compile(optimizer=opt, loss='mean_absolute_percentage_error')

print('Model compiled!\nStart to train model:\n')

H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS
)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(mean_absolute_error(testY,
                          predictions) * 93)

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.show()
