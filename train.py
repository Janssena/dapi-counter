# set the matplotlib backend so figures can be saved in the background

from definitions import PATH
from dapi_counter.model.model import CountNet
from dapi_counter.data_farming.helpers import csv_to_df
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import numpy as np
import cv2
import os

# get data and shuffle
df = csv_to_df(PATH['CSV'])
df_shuffle = df.sample(frac=1).reset_index(drop=True)

# append image and label to data and label array
data = []
labels = []
locations = []

print("[INFO] preparing data...")
for i, image in enumerate(df_shuffle['name']):
    im_path = os.path.join(PATH['IMAGE_OUT'] + image)
    img = cv2.imread(im_path, )[:, :, :1]  # so size is (256, 256, 1)
    data.append(img)
    label = df_shuffle['count'][i]
    loc = df_shuffle['locations'][i]
    labels.append(label)
    locations.append(loc)

y_factor = max(labels)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels) / y_factor
locations = np.array(locations)

# split in train and test
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
aug = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
model = CountNet.build(256, 256, 1)

# setup training
BS = 16
EPOCHS = 10
opt = Adam(lr=1e-4, decay=10)
model.compile(optimizer=opt, loss='mean_absolute_percentage_error')

# train
print("[INFO] starting training...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS
)

# evaluate
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(mean_absolute_error(testY, predictions * y_factor))

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

# save model
