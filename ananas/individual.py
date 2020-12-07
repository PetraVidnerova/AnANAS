import random
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import InputLayer
# from keras.optimizers import RMSprop

import config

class Layer:
    """ Specification of one layer.
        Includes size, regularization, activaton. 
    """

    def __init__(self):
        pass

    def randomInit(self):
        network_config = config.global_config["network"] 
        self.size = random.randint(
            network_config["min_layer_size"],
            network_config["max_layer_size"],
        )
        self.dropout = random.choice(network_config["dropout"])
        self.activation = random.choice(network_config["activations"])
        return self

    def __str__(self):
        return " #{} dropout={} activation={}".format(self.size,
                                                      self.dropout,
                                                      self.activation)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False


class Individual:
    """  Individual coding network architecture. 
    """

    def __init__(self):
        self.input_shape = config.global_config["input_shape"]
        self.noutputs = config.global_config["noutputs"]
        # print(self.input_shape, self.noutputs)

    def randomInit(self):
        self.layers = []
        num_layers = random.randint(1, config.global_config["network"]["max_layers"])
        for l in range(num_layers):
            layer = Layer().randomInit()
            self.layers.append(layer)

    def createNetwork(self, input_layer=None):

        model = Sequential()

        model.add(input_layer or InputLayer(config.global_config["input_shape"]))

        for l in self.layers:
            model.add(Dense(l.size))
            model.add(Activation(l.activation))
            if l.dropout > 0:
                model.add(Dropout(l.dropout))

        # final part
        model.add(Dense(self.noutputs))
        
        if config.global_config["main_alg"]["task_type"] == "classification":
            model.add(Activation('softmax'))

        if config.global_config["main_alg"]["task_type"] == "binary_classification":
            model.add(Activation('sigmoid'))

            
        # model.compile(loss=Config.loss,
        #               optimizer=RMSprop())

        return model

    def __str__(self):

        ret = "------------------------\n"
        for l in self.layers:
            ret += str(l)
            ret += "\n"
            ret += "------------------------\n"
        return ret

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False


def initIndividual(indclass):
    ind = indclass()
    ind.randomInit()
    return ind
