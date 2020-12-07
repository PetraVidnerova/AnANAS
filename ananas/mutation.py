import random
from utils import roulette
from individual import Individual, Layer
import config

PROB_MUTATE_LAYER = 0.5
PROB_ADD_LAYER = 0.25
PROB_DEL_LAYER = 0.25

PROB_CHANGE_LAYER_SIZE = 0.4
PROB_CHANGE_DROPOUT = 0.25
PROB_CHANGE_ACTIVATION = 0.25
PROB_CHANGE_ALL = 0.1 

PROB_RANDOM_LAYER_SIZE = 0.4
PROB_ADD_NEURON = 0.3
PROB_DEL_NEURON = 0.3 

class Mutation:
    """ Class encapsulating various mutation functions.
   
    Usage:
          mut = Mutation()
          ...
          mut.mutate(individual)

    Mutation is selected ad random:
        ------ mutate layer (change layer params) 
         |            |
         |             ----- choose random layer and apply one of 
         |                               |
         |                                ---- mutate size 
         |                               |     |
         |                               |      ---- add neuron 
         |                               |     |
         |                               |      ---- del neuron 
         |                               |     |
         |                               |      ---- new size
         |                               |
         |                                ---- mutate dropout 
         |                               |
         |                                ---- mutate activation
         |                               |
         |                                ---- random init   
          ---- add layer 
         |
          ---- delete layer 
    """


    def changeSize(self, layer):
        """ Changes size of given layer. """ 
        layer.size = random.randint(
            config.global_config["network"]["min_layer_size"],
            config.global_config["network"]["max_layer_size"]
        )

    def addNeuron(self, layer):
        """ Adds one neuron to the given layer. """
        layer.size += 1

    def delNeuron(self, layer):
        """ Deletion of one neuron from the given layer. """ 
        if layer.size > 0:
            layer.size -= 1 
    
    def mutateSize(self, layer):
        """ Mutates size of the layer, chooses randomly from relevant mutations. """

        mutfunc = roulette([self.changeSize, self.addNeuron, self.delNeuron],
                           [PROB_RANDOM_LAYER_SIZE, PROB_ADD_NEURON, PROB_DEL_NEURON])
        if mutfunc:
            mutfunc(layer)


    def mutateDropout(self, layer):
        """ Changes the value of dropout on the given layer. """ 
        layer.dropout = random.choice(config.global_config["network"]["dropout"])
        
    def mutateActivation(self, layer):
        """ Changes activation of given layer. """ 
        layer.activation = random.choice(config.global_config["network"]["activations"])

    def randomInit(self, layer):
        """ Random initialization of given layer. """ 
        layer.randomInit()

    def mutateLayer(self, individual):
        """ Selects random layer and applies randomly selected
        mutation on it. """

        # select layer random 
        l = random.randint(0, len(individual.layers)-1)
        layer = individual.layers[l]

        mutfunc = roulette([self.mutateSize, self.mutateDropout, self.mutateActivation,
                            self.randomInit],
                           [PROB_CHANGE_LAYER_SIZE, PROB_CHANGE_DROPOUT, PROB_CHANGE_ACTIVATION,
                            PROB_CHANGE_ALL])

        if mutfunc:
            mutfunc(layer)
        
        return individual,

    def addLayer(self, individual):
        """ Inserts one random layer into individual.""" 
        l = random.randint(0, len(individual.layers)-1)
        individual.layers.insert(l, Layer().randomInit())
        return individual, 

    def delLayer(self, individual):
        """ Removes one randomly selected layers. """ 
        if len(individual.layers)>1:
            l = random.randint(0, len(individual.layers)-1)
            del individual.layers[l]
        return individual, 
    
    def mutate(self, individual): 
        """ Selects mutation type at random and applies it. """ 
        
        mutfunc = roulette([self.mutateLayer, self.addLayer, self.delLayer],
                           [PROB_MUTATE_LAYER, PROB_ADD_LAYER, PROB_DEL_LAYER])
        if mutfunc:
            return mutfunc(individual)
        
        return individual, 




    
from convindividual import ConvIndividual, Layer, ConvLayer, MaxPoolLayer

PROB_CHANGE_NUM_FILTERS = 0.3
PROB_MUTATE_KERNEL_SIZE = 0.3
PROB_CHANGE_CONV_ACTIVATION = 0.3
PROB_CHANGE_CONV_ALL = 0.1



class MutationConv(Mutation):

    """ Class encapsulating various mutation functions for convolutional
    networks.
   
    Usage:
          mut = Mutation()
          ...
          mut.mutate(individual)

    Mutation is selected ad random:
       --  mutate layer, add layer, del layer 
    Mutate layer: if layer dense, works as in Mutation 
                  if layer convolutional, selects from 
                       - mutate filters
                       - mutate kernel size / pool size 
                       - mutate activation
                       - random init 
    """


    def mutateFilters(self, layer):
        """ Changes the number of filters. """
        layer.filters = random.randint(
            config.global_config["network"]["min_filters"],
            config.global_config["network"]["max_filters"]
        )

    def mutateKernelSize(self, layer):
        """ Chagnes size of kernels. """ 
        layer.kernel_size = random.randint(
            config.global_config["network"]["min_kernel_size"],
            config.global_config["network"]["max_kernel_size"]
        )

    def mutatePoolSize(self, layer):
        """ Changes size of pool for max pool layer. """ 
        layer.pool_size =  random.randint(
            config.global_config["network"]["min_pool_size"],
            config.global_config["network"]["max_pool_size"]
        )
        
        
    def mutateLayer(self, individual):
        """ Mutate the given layer by randomly chosen mutation. """ 

        layers = individual.conv_layers + individual.dense_layers
        # select layer random 
        l = random.randint(0, len(layers)-1)
        layer = layers[l]

        if type(layer) is Layer:
            mutfunc = roulette([self.mutateSize, self.mutateDropout, self.mutateActivation,
                                self.randomInit],
                               [PROB_CHANGE_LAYER_SIZE, PROB_CHANGE_DROPOUT, PROB_CHANGE_ACTIVATION,
                               PROB_CHANGE_ALL])
            if mutfunc:
                mutfunc(layer)
        elif type(layer) is ConvLayer:
            mutfunc = roulette([self.mutateFilters, self.mutateKernelSize, self.mutateActivation,
                                self.randomInit],
                               [PROB_CHANGE_NUM_FILTERS, PROB_MUTATE_KERNEL_SIZE, PROB_CHANGE_CONV_ACTIVATION,
                                PROB_CHANGE_CONV_ALL])
            if mutfunc:
                mutfunc(layer)
        elif type(layer) is MaxPoolLayer:
            self.mutatePoolSize(layer)
        else:
            raise(TypeError("unknown type of layer"))
            
                
        return individual,

    def addLayer(self, individual):
        """ Add one randomly generated layer. """
        conv_part = random.randint(0,1)
        if (conv_part):
            l = random.randint(0, len(individual.conv_layers)-1)
            if random.randint(0,1):
                individual.conv_layers.insert(l, ConvLayer().randomInit())
            else:
                individual.conv_layers.insert(l, MaxPoolLayer().randomInit())
        else:
            l = random.randint(0, len(individual.dense_layers)-1)
            individual.dense_layers.insert(l, Layer().randomInit())
            
        return individual, 

    def delLayer(self, individual):
        """ Removes random layer. """ 
        conv_part = random.randint(0,1)
        if (conv_part):
            if len(individual.conv_layers)>1:
                l = random.randint(0, len(individual.conv_layers)-1)
                del individual.conv_layers[l]
        else:
            if len(individual.dense_layers)>1:
                l = random.randint(0, len(individual.dense_layers)-1)
                del individual.dense_layers[l]
                
        return individual,
    
    
    def mutate(self, individual):
        """ Mutate the network with one randomly selected mutation. """ 
        
        mutfunc = roulette([self.mutateLayer, self.addLayer, self.delLayer],
                           [PROB_MUTATE_LAYER, PROB_ADD_LAYER, PROB_DEL_LAYER])
        if mutfunc:
            return mutfunc(individual)
        
        return individual, 
