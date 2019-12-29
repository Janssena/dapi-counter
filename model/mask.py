from tensorflow.keras import backend as K
from tensorflow.keras.layers import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose


def dice_coef(y_true, y_pred, smooth=1e-7):
    """This function calculates the dice coefficient loss"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    """To get the true loss we should use the negative of the loss"""
    return -dice_coef(y_true, y_pred)


def u_net(input_shape):
    """
    This model processes images to return a binary image representing the image
    mask.
    """
    inputs = Input(input_shape)
    # downsampling 1/4
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    down_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(down_1)
    # downsampling 2/4
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    down_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(down_2)
    # downsampling 3/4
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    down_3 = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(down_3)
    # downsampling 4/4
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    down_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(down_4)
    # bottom layer
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    # upsample 1/4
    up_1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = concatenate([up_1, down_4], axis=3)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    # upsample 2/4
    up_2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = concatenate([up_2, down_3], axis=3)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # upsample 3/4
    up_3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = concatenate([up_3, down_2], axis=3)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    # upsample 4/4
    up_4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
    x = concatenate([up_4, down_1], axis=3)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # output layer
    x = Conv2D(1, (1, 1), activation='sigmoid')(x)

    return Model(inputs=[inputs], outputs=[x])
