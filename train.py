from definitions import PATH
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import mean_absolute_error
from model.count import count_model
from model.model import count_net
from model.mask import u_net, dice_coef, dice_coef_loss
import matplotlib.pyplot as plt
from datetime import date
import numpy as np
import argparse
import get_data
import os

# logic for parsing arguments
models = {'count': count_model, 'count_net': count_net, 'mask': u_net}
sources = {'simcep': get_data.simcep, 'binary': get_data.simcep_masks}

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', '-m',
    help='determines which model to load, should be one of {}'.format(models.keys()))
parser.add_argument(
    '--data', '-d',
    help='determines what data to use, should be one of {}'.format(sources.keys()))
parser.add_argument('--lr', help='set learning rate, standard at 1e-4')
parser.add_argument('--bs', help='set batch size, standard at 16')
parser.add_argument(
    '--epochs', help='set number of epochs to train for, standard 10')
args = parser.parse_args()

if args.model in models.keys():
    model = models[args.model]((256, 256, 1))
else:
    raise Exception(
        'Unrecognized model, should be one of {}'.format(models.keys()))

if args.data in sources.keys():
    data, labels = sources[args.data]()
else:
    raise Exception(
        'Unrecognized data source, should be one of {}'.format(sources.keys()))

# constants
LEARN_RATE = float(args.lr) if args.lr else 1e-5
BATCH_SIZE = int(args.bs) if args.bs else 20
EPOCHS = int(args.epochs) if args.epochs else 10
OPT = Adam(lr=LEARN_RATE, decay=10)

# split in train and test
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.25, random_state=42)

# setup training
aug = ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True, samplewise_center=True)
if args.model == 'mask':
    model.compile(optimizer=OPT, loss=dice_coef_loss, metrics=[dice_coef])
else:
    model.compile(optimizer=OPT, loss='mean_absolute_percentage_error')

# train
try:
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // BATCH_SIZE,
        epochs=EPOCHS
    )
except KeyboardInterrupt:
    if not input('Save model before stopping? (Y/n) ').lower().startswith('y'):
        raise

# evaluate
print("Evaluating network...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)
print('MAE:', mean_absolute_error(testY, predictions))

# create a figure for the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# save model and figure
print("Saving model...")
model_name = '{model}_{date}.h5'.format(
    model=args.model, date=date.today().strftime("%d-%m-%Y"))
model.save_weights(os.path.join(PATH['MODEL_OUT'], model_name))
plt.savefig(os.path.join(PATH['MODEL_OUT'], model_name.replace('.h5', '.png')))
