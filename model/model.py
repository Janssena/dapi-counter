from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D
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

    def resnet_conv_block(input_tensor, kernel_size, filters, strides=(2,2)):
        filter1, filter2, filter3 = filters

        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        # CONV 1
        x = Conv2D(filter1, (1,1), strides=strides, kernel_initializer='he_normal')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis)(x)
        x = layers.Activation('relu')(x)
        # CONV 2
        x = Conv2D(filter2, kernel_size, padding='same', kernel_initializer='he_normal')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis)(x)
        x = layers.Activation('relu')(x)
        # CONV 3
        x = Conv2D(filter3, (1,1), strides=strides, kernel_initializer='he_normal')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis)(x)
        # SHORTCUT
        shortcut = layers.Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal')(input_tensor)
        shortcut = layers.BatchNormalization(axis=bn_axis)(shortcut)

        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def resnet_identity_block(input_tensor, kernel_size, filters):
        filter1, filter2, filter3 = filters

        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        x = layers.Conv2D(filters1, (1, 1), kernel_initializer='he_normal')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis)(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization(axis=bn_axis)(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization(axis=bn_axis)(x)

        x = layers.add([x, input_tensor])
        x = layers.Activation('relu')(x)
        return x


    def count_net(img_input, input_shape):

        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        # zero pad the image so border pixels are saved
        x = ZeroPadding2D(padding=(3, 3))(img_input)

        # MODEL BASE
        x = Conv2D(16, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=bn_axis)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # ENCODER
        x = resnet_conv_block(x, (3, 3), 32)(x)
        x = resnet_identity_block(x, (3, 3), 64)(x)
        x = resnet_identity_block(x, (3, 3), 128)(x)

        # DECODER



    @staticmethod
    def build(width, height, depth):
        model = Sequential()
        input_shape = (height, width, depth)

        # maybe use faster R-CNN to detect object boundaries
        # use bounding box detection algorithm to detect boxes and just count?

        # architecture:
        #   Conv2D 16
        #   encoder ResNet50 block -> ConvBlock 32 - Identity 64 - Identity 128
        #   decoder ResNet50 block -> Identity 64 - Identity 32 - ConvBlock 16
        #   GlobalAvgPool
        #   Dense
