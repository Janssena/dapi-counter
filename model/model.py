from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import add
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential, models


class VGGNet:
    @staticmethod
    def build(width, height, depth):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        if depth > 1 or depth < 1:
            raise ValueError(
                'The model expects grayscale images of DAPI stained cells')

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


def resnet_conv_block(input_tensor, kernel_size, filters, strides=(2, 2)):
    """ Resnet conv block """
    filter1, filter2, filter3 = filters

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # CONV 1
    x = Conv2D(filter1, (1, 1), strides=strides,
               kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    # CONV 2
    x = Conv2D(filter2, kernel_size, padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    # CONV 3
    x = Conv2D(filter3, (1, 1), kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    # SHORTCUT
    shortcut = Conv2D(
        filter3, (1, 1), strides=strides, kernel_initializer='he_normal')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis)(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet_identity_block(input_tensor, kernel_size, filters):
    """ Resnet identity block """
    filter1, filter2, filter3 = filters

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(filter1, (1, 1),
               kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=bn_axis)(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def count_net(input_shape, input_tensor=None):

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    # zero pad the image so border pixels are saved
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)

    # MODEL BASE
    x = Conv2D(16, (7, 7), strides=(2, 2), padding='valid',
               kernel_initializer='he_normal', input_shape=input_shape, name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)

    # ENCODER
    x = resnet_conv_block(x, (3, 3), (32, 32, 4*32))
    x = resnet_identity_block(x, (3, 3), (32, 32, 4*32))
    x = resnet_conv_block(x, (3, 3), (64, 64, 4*64))
    x = resnet_identity_block(x, (3, 3), (64, 64, 4*64))
    x = resnet_identity_block(x, (3, 3), (64, 64, 4*64))
    x = resnet_conv_block(x, (3, 3), (128, 128, 4*128))
    x = resnet_identity_block(x, (3, 3), (128, 128, 4*128))
    x = resnet_identity_block(x, (3, 3), (128, 128, 4*128))

    # DECODER
    x = resnet_identity_block(x, (3, 3), (128, 128, 4*128))
    x = resnet_identity_block(x, (3, 3), (128, 128, 4*128))
    x = resnet_conv_block(x, (3, 3), (128, 128, 4*128), strides=(1, 1))
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2))(x)
    x = resnet_identity_block(x, (3, 3), (64, 64, 2*64))
    x = resnet_conv_block(x, (3, 3), (64, 64, 4*64), strides=(1, 1))
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2))(x)
    x = resnet_identity_block(x, (3, 3), (32, 32, 2*32))
    x = resnet_conv_block(x, (3, 3), (32, 32, 4*32), strides=(1, 1))
    x = Conv2DTranspose(32, (3, 3), strides=(2, 2))(x)

    # TAIL
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)

    inputs = img_input
    model = models.Model(inputs, x, name='countnet')
    return model
