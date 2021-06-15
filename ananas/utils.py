import numpy as np
import config
import random

#from sklearn.metrics import accuracy_score 

def roulette(functions, probs):
    """ Implements roulette wheel selection. """
    
    r = random.random()
    for func, prob in zip(functions, probs):
        if r < prob:
            return func
        else:
            r -= prob
    return None


def mean_sq_error(y1, y2):
    """ Mean square error. """ 
    diff = y1 - y2
    E = 100 * sum(diff*diff) / len(y1)
    return E


def my_accuracy_score(y1, y2):
    """ Accuracy score including argmax. """ 
    assert y1.shape == y2.shape

    y1_argmax = np.argmax(y1, axis=1)
    y2_argmax = np.argmax(y2, axis=1)
    score = sum(y1_argmax == y2_argmax)
    return score / len(y1)

def bin_accuracy_score(y1, y2):
    """ Accuracy score including argmax. """ 
    assert y1.shape == y2.shape

    y2_rounded  = np.round(y2)
    score = sum(y1 == y2_rounded)
    return score / len(y1)


def error(y1, y2):
    """ Return either accuracy score for classification task 
         or mean square error fo regression. """ 
    if config.global_config["main_alg"]["task_type"] == "classification":
        return my_accuracy_score(y1, y2)
    elif config.global_config["main_alg"]["task_type"] == "binary_classification":
        return bin_accuracy_score(y1, y2)
    else:
        return mean_sq_error(y1, y2)


def print_stat(E, name):
    """ Outputs statistics """
    print(f"E_{name:6} avg={np.mean(E):.4f} std={np.std(E):.4f}"
          f"min={np.min(E):.4f} max={np.max(E):.4f}")
