from definitions import PATH
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
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

# argument constants
models = {'count': count_model, 'count_net': count_net,
          'mask': u_net, 'resnet': ResNet50}
sources = {'simcep': get_data.simcep, 'binary': get_data.simcep_masks}

# logic for parsing arguments
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
parser.add_argument(
    '--resume',
    help='boolean flag. determines if training should resume with the latest weights',
    action='store_true')
args = parser.parse_args()

# constants
INPUT_SHAPE = (256, 256, 1)
LEARN_RATE = float(args.lr) if args.lr else 1e-5
BATCH_SIZE = int(args.bs) if args.bs else 20
EPOCHS = int(args.epochs) if args.epochs else 10
OPT = Adam(lr=LEARN_RATE, decay=10)

if args.model in models.keys():
    if args.model == 'resnet':
        base = ResNet50(include_top=False,
                        input_shape=INPUT_SHAPE, weights=None)
        x = Flatten()(base.output)
        x = Dense(1, activation='relu')(x)
        model = Model(inputs=base.inputs, outputs=x)
    else:
        model = models[args.model](INPUT_SHAPE)
else:
    raise Exception(
        'Unrecognized model, should be one of {}'.format(models.keys()))

if args.data in sources.keys():
    if args.data == 'simcep':
        data, labels = sources[args.data]()
    else:
        data, labels = sources[args.data]()
else:
    raise Exception(
        'Unrecognized data source, should be one of {}'.format(sources.keys()))

# split in train and test
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.25, random_state=42)

# setup training
aug = ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True, samplewise_center=True)
if args.model == 'mask':
    model.compile(optimizer=OPT, loss=dice_coef_loss, metrics=[dice_coef])
else:
    model.compile(optimizer=OPT, loss='mean_squared_error')


if args.resume:
    weights = get_data.latest_weights(args.model)
    model.load_weights(weights)

# train
try:
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // BATCH_SIZE,
        epochs=EPOCHS
    )
except KeyboardInterrupt:
    if input('\nSave model before stopping? (Y/n) ').lower().startswith('n'):
        raise

# evaluate
print("Evaluating network...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)
mae = mean_absolute_error(testY, predictions)
print('MAE:', mae)
print(predictions * 100)

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
model_name = '{model}_{date}_{mae}.h5'.format(
    model=args.model, date=date.today().strftime("%d-%m-%Y"), mae=str(mae)[:4])
model.save_weights(os.path.join(PATH['MODEL_OUT'], model_name))
plt.savefig(os.path.join(PATH['MODEL_OUT'], model_name.replace('.h5', '.png')))
