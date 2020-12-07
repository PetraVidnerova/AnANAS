import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, InputLayer
from utils import roulette
from individual import Layer
import config

class ConvLayer:
    """ Specification of one convolutional layer.
        Includes number of filters, kernel size, activation.
    """

    def __init__(self):
        pass

    def randomInit(self):
        self.filters = random.randint(
            config.global_config["network"]["min_filters"],
            config.global_config["network"]["max_filters"]
        )
        # filters are squares kernel_size x kernel_size
        self.kernel_size = random.randint(
            config.global_config["network"]["min_kernel_size"],
            config.global_config["network"]["max_kernel_size"]
        )
        self.activation = random.choice(config.global_config["network"]["activations"])
        return self

    def __str__(self):
        return "conv #{} kernelsize={} activation={}".format(self.filters, self.kernel_size, self.activation)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False


class MaxPoolLayer:
    """ Specification of one max pooling layer.
    """

    def __init__(self):
        pass

    def randomInit(self):
        # pooling size is (pool_size, pool_size)
        self.pool_size = random.randint(
            config.global_config["network"]["min_pool_size"],
            config.global_config["network"]["max_pool_size"]
        )
        return self

    def __str__(self):
        return "pool poolsize={} ".format(self.pool_size)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False


def createRandomLayer():

    create = roulette([lambda: ConvLayer(),
                       lambda: MaxPoolLayer()],
                      [
                          config.global_config["network"]["conv_layer"],
                          config.global_config["network"]["max_pool_layer"]
                      ]
    )
    if create:
        return create()
    else:
        return ConvLayer()


class ConvIndividual:
    """ Individual coding convolutional network architecture.
        Individual consists of two parts, first of convolutioanal
        and max pooling layers, the second part of dense layers.
    """

    def __init__(self):
        self.input_shape = config.global_config["input_shape"]
        self.noutputs = config.global_config["noutputs"]
        self.nparams = None

    def randomInit(self):
        self.conv_layers = []
        num_conv_layers = random.randint(1, config.global_config["network"]["max_conv_layers"])
        for l in range(num_conv_layers):
            layer = createRandomLayer().randomInit()
            self.conv_layers.append(layer)

        self.dense_layers = []
        num_dense_layers = random.randint(1, config.global_config["network"]["max_dense_layers"])
        for l in range(num_dense_layers):
            layer = Layer().randomInit()
            self.dense_layers.append(layer)

    def createNetwork(self, input_layer=None):
        model = Sequential()

        model.add(input_layer or InputLayer(config.global_config["input_shape"]))

        # convolutional part
        for l in self.conv_layers:
            if type(l) is ConvLayer:
                model.add(
                    Conv2D(l.filters, (l.kernel_size, l.kernel_size), padding='same'))
                model.add(Activation(l.activation))
            elif type(l) is MaxPoolLayer:
                # check if pooling is possible
                if model.output_shape[1] >= l.pool_size and model.output_shape[2] >= l.pool_size:
                    model.add(MaxPooling2D(
                        pool_size=(l.pool_size, l.pool_size)))
            else:
                raise TypeError("unknown type of layer")

        # dense part
        model.add(Flatten())
        for l in self.dense_layers:
            model.add(Dense(l.size))
            model.add(Activation(l.activation))
            if l.dropout > 0:
                model.add(Dropout(l.dropout))

        # final part
        model.add(Dense(self.noutputs))
        if config.global_config["main_alg"]["task_type"] == "classification":
            model.add(Activation('softmax'))
        elif config.global_config["main_alg"]["task_type"] == "binary_classification":
            model.add(Activation('sigmoid'))

            
        self.nparams = model.count_params()

        return model

    def __str__(self):

        ret = "------------------------\n"
        for l in self.conv_layers+self.dense_layers:
            ret += str(l)
            ret += "\n"
            ret += "------------------------\n"
        return ret

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False
