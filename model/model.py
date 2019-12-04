from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


class VGGNet:
    @staticmethod
    def build(width, height, depth):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        if depth > 1 or depth < 1:
            raise ValueError('The model expects grayscale images of DAPI stained cells')

        model = Sequential()
        input_shape = (height, width, depth)

        # Stem block (Inception ResNet v2)
        # model.add(Conv2D(32, (3, 3), strides=2, padding='valid', input_shape=input_shape))
        # model.add(Conv2D(32, (3, 3), padding='valid'))
        # model.add(Conv2D(64, (3, 3)))
        # model.add(MaxPooling2D(3, strides=2))
        # model.add(Conv2D(80, (1, 1), padding='valid'))
        # model.add(Conv2D(192, (3, 3), padding='valid'))
        # model.add(MaxPooling2D(3, strides=2))
        #
        # model.add(Flatten())
        # model.add(Dense(1, activation='linear'))

        # CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))   # size is incorrect
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 3 => POOL layer set
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        # model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(1))     # prob only one
        model.add(Activation("linear"))     # linear for regression

        # return the constructed network architecture
        return model

class CountNet:

    def ConvBlock():
        # conv (1,1) + BN + relu  -> conv(ks,ks) + BN + relu -> conv(1,1) + BN all stride (2,2) 
        # shortcut conv (1,1), stride(2,2) + BN
        # add layers
        # relu



    @staticmethod
    def build(width, height, depth):
        model = Sequential()
        input_shape = (height, width, depth)

        # maybe use faster R-CNN to detect object boundaries
        # use boudning box detection algorithm to detect boxes and just count?

        # architecture:
        #   Conv2D 16
        #   encoder ResNet50 block -> ConvBlock 32 - Identity 64 - Identity 128
        #   decoder ResNet50 block -> Identity 64 - Identity 32 - ConvBlock 16
        #   GlobalAvgPool
        #   Dense

        # # 3 x Conv2D -> MaxPooling2D
        # model.add(Conv2D(16, (3, 3), padding="same",
        #                 input_shape=input_shape))
        # model.add(Activation("relu"))
        # model.add(Conv2D(16, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(Conv2D(16, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # # 3 x Conv2D -> MaxPooling2D
        # model.add(Conv2D(32, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(Conv2D(32, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(Conv2D(32, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # # 3 x Conv2D
        # model.add(Conv2D(64, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(Conv2D(64, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(Conv2D(64, (3, 3), padding="same"))
        # model.add(Activation("relu"))
