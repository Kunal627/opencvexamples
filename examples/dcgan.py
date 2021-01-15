from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten


class DCGAN:

    @staticmethod
    def build_generator(dim=7,depth=64, channels=1, inputDim=100, outputDim=512):

        inputShape = (dim, dim, depth)
        chanDim = -1

        model = Sequential()
        model.add(Dense(input_dim=inputDim, units=outputDim))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        model.add(Dense(dim * dim * depth))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        model.add(Reshape(inputShape))
        model.add(Conv2DTranspose(32, (5,5), strides=(2,2), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2DTranspose(channels, (5,5), strides=(2,2), padding="same"))
        model.add(Activation("tanh"))

        return model

    @staticmethod
    def build_discriminator(width, height, depth, alpha=0.2):

        model = Sequential()
        inputShape = (height, width, depth)

        model.add(Conv2D(32, (5,5), padding="same", strides=(2,2), input_shape=inputShape))
        model.add(LeakyReLU(alpha=alpha))

        model.add(Conv2D(32, (5,5), padding="same", strides=(2,2), input_shape=inputShape))
        model.add(LeakyReLU(alpha=alpha))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=alpha))

        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        return model

