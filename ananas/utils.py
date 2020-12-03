import numpy as np
from config import Config
import random


def roulette(functions, probs):
    r = random.random()
    for func, prob in zip(functions, probs):
        if r < prob:
            return func
        else:
            r -= prob
    return None


def mean_sq_error(y1, y2):
    diff = y1 - y2
    E = 100 * sum(diff*diff) / len(y1)
    return E


def accuracy_score(y1, y2):
    assert y1.shape == y2.shape

    y1_argmax = np.argmax(y1, axis=1)
    y2_argmax = np.argmax(y2, axis=1)
    score = sum(y1_argmax == y2_argmax)
    return score / len(y1)


def error(y1, y2):
    if Config.task_type == "classification":
        return accuracy_score(y1, y2)
    else:
        return mean_sq_error(y1, y2)


def print_stat(E, name):
    print("E_{:6} avg={:.4f} std={:.4f}".format(name, np.mean(E), np.std(E)),
          "min={:.4f} max={:.4f}".format(np.min(E), np.max(E)))
