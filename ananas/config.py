import configparser
import re


class Cfg:
    pass


class CfgMnist:

    GPU = True

    final_evals  = 5
    
    eval_batch_size = 30

    batch_size = 128
    epochs = 20
    loss = 'categorical_crossentropy'
    # loss = 'mean_squared_error'

    task_type = "classification"

    pop_size = 30
    ngen = 150

    MAX_LAYERS = 5
    MAX_LAYER_SIZE = 300
    MIN_LAYER_SIZE = 10
    DROPOUT = [0.0, 0.2, 0.3, 0.4]
    ACTIVATIONS = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

    # for convolutional networks
    MIN_FILTERS = 10
    MAX_FILTERS = 50
    MIN_KERNEL_SIZE = 2
    MAX_KERNEL_SIZE = 5
    MIN_POOL_SIZE = 2
    MAX_POOL_SIZE = 3
    MAX_CONV_LAYERS = 3
    MAX_DENSE_LAYERS = 3

    # DENSE_LAYER = 0.5
    CONV_LAYER = 0.7
    MAX_POOL_LAYER = 0.3


Config = CfgMnist()


def is_int(s):
    if re.fullmatch(r'[0-9]+', s):
        return True
    else:
        return False


def is_float(s):
    if re.fullmatch(r'[0-9]+\.[0-9]+', s):
        return True
    else:
        return False


def is_list(s):
    if re.fullmatch(r'\[.+\]', s):
        return True
    else:
        return False


def convert(s):
    if is_int(s):
        val = int(s)
    elif is_float(s):
        val = float(s)
    elif is_list(s):
        s = s.strip()
        s = s.strip("[")
        s = s.strip("]")
        val = s.split(',')
        newval = []
        for v in val:
            v = v.strip()
            v = v.strip("'")
            v = v.strip('"')
            newval.append(convert(v))
        val = newval
    else:
        val = s
    return val


def load_config(name):

    config = configparser.ConfigParser()
    config.read(name)

    global Config
    Config = Cfg()

    for sec in config.sections():
        for key, val in config[sec].items():
            val = convert(val)
            setattr(Config, key.lower(), val)
            setattr(Config, key.upper(), val)
