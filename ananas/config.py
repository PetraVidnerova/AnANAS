import yaml 
from pprint import pprint 


global_config = {}


def load_config(name):
    """ Loads config from file given by name. 
    Saves it in global Config object in module config. """

    global global_config
    
    with open(name, "r") as f:
        global_config = yaml.full_load(f)

        
def print_config():
    """ Just pprint the actual global configuration """
    pprint(global_config)
