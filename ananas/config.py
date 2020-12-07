import yaml 
from pprint import pprint 

# default configuration
# values can be rewritten by user defined config file 
global_config = {
    'network_type': 'dense',
    'nsga': 2,
    'main_alg': {
        'batch_size': 128,
        'eval_batch_size': 30,
        'epochs': 10,
        'loss': 'categorical_crossentropy',
        'task_type': 'classification',
        'final_epochs': 20
    },
    'ga': {
        'pop_size': 20,
        'n_gen': 20
    },
    'network': {
        'max_layers': 5,
        'max_layer_size': 500,
        'min_layer_size': 10,
        'dropout': [0.0, 0.2, 0.3, 0.4],
        'activations': ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
        # convolutional part
        'max_dense_layers': 5,
        'max_conv_layers': 4,
        'conv_layer': 0.7,
        'max_pool_layer': 0.3,
        'min_pool_size': 2,
        'max_pool_size': 4,
        'min_filters': 10,
        'max_filters': 100,
        'min_kernel_size': 2,
        'max_kernel_size': 5
    },
    'device': {
        'device_type': 'CPU',
        'n_cpus': 10
    }
}



def load_config(name):
    """ Loads config from file given by name. 
    Saves it in global Config object in module config. """
    
    with open(name, "r") as f:
        loaded_config = yaml.full_load(f)
    global_config.update(loaded_config)
        
def print_config():
    """ Just pprint the actual global configuration """
    pprint(global_config)
