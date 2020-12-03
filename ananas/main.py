import random
import pickle
import numpy as np
import multiprocessing
import json

from deap import base
from deap import creator
from deap import tools

from individual import Individual, initIndividual
from convindividual import ConvIndividual
from fitness import Fitness
from mutation import Mutation, MutationConv
from crossover import Crossover, CrossoverConv
import alg
from dataset import load_data
from config import Config, load_config
# from utils import error
from nsga import selectNSGA

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', help='either "conv" or "dense"')
parser.add_argument('--trainset', help='filename of training set')
parser.add_argument('--testset', help='filename of test set')
parser.add_argument('--nsga', help='0,1,2,3')
parser.add_argument('--id', help='computation id')
parser.add_argument(
    '--checkpoint', help='checkpoint file to load the initial state from')
parser.add_argument('--config', help='json config filename')

args = parser.parse_args()
trainset_name = args.trainset
testset_name = args.testset
use_conv_layers = args.type == "conv"
if use_conv_layers:
    print("**** Using convolutional layers.")
exp_id = args.id
if exp_id is None:
    exp_id = ""
nsga_number = int(args.nsga) if args.nsga else 2
checkpoint_file = args.checkpoint
config_name = args.config
if config_name is not None:
    load_config(config_name)

# for classification fitness is accuracy, for approximation fitness is error
# second fitness element is network size, should be minimised
if Config.task_type == "classification":
    creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))
else:
    creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0))

creator.create("Individual",
               ConvIndividual if use_conv_layers else Individual,
               fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Structure initializers
toolbox.register("individual", initIndividual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# use multiple processors
if not Config.GPU:
    pool = multiprocessing.Pool(10)
    toolbox.register("map", pool.map)

# register operators
fit = Fitness("data/"+trainset_name)
mut = MutationConv() if use_conv_layers else Mutation()
cross = CrossoverConv() if use_conv_layers else Crossover()

toolbox.register("eval_batch", fit.evaluate_batch)
toolbox.register("evaluate", fit.evaluate)
toolbox.register("mate", cross.cxOnePoint)
toolbox.register("mutate", mut.mutate)
if nsga_number == 3:
    ref_points = tools.uniform_reference_points(2, 12)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
elif nsga_number == 2:
    # nsgaII - deap implementation
    toolbox.register("select", tools.selNSGA2)
elif nsga_number == 1:
    # stepan's version of nsga
    toolbox.register("select", selectNSGA)
elif nsga_number == 0:
    # use vanilla GA
    toolbox.register("select", tools.selTournament, tournsize=3)
else:
    raise NotImplementedError()


def main(exp_id, checkpoint_name=None):
    # random.seed(64)
    global Config

    if checkpoint_name:
        # A file name has been given, then load the data from the file
        cp = pickle.load(open(checkpoint_name, "rb"))
        pop = cp["population"]
        start_gen = cp["generation"] + 1
        hof = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
        if "config" in cp:
            Config = cp["config"]

    else:
        pop = toolbox.population(n=Config.pop_size)
        start_gen = 0
        hof = tools.ParetoFront()
        logbook = tools.Logbook()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # nsga_number 0 stands for vanilla GA
    algorithm = alg.nsga if nsga_number != 0 else alg.vanilla_ga

    pop, log = algorithm(pop,
                         start_gen,
                         toolbox,
                         cxpb=0.6,
                         mutpb=0.2,
                         ngen=Config.ngen,
                         stats=stats,
                         halloffame=hof,
                         logbook=logbook,
                         verbose=True,
                         exp_id=exp_id)

    return pop, log, hof


if __name__ == "__main__":

    # load the whole data
    X_train, y_train = load_data("data/"+trainset_name)
    X_test, y_test = load_data("data/"+testset_name)

    # set cfg
    Config.input_shape = X_train[0].shape
    Config.noutputs = y_train.shape[1]
    #    print(Config.input_shape, Config.noutputs)

    pop, log, hof = main(exp_id, checkpoint_file)

    # print and save the pareto front
    json_list = []
    for ind in hof:
        print(ind.fitness.values)
        json_list.append(ind.createNetwork().to_json())

    with open("best_model_{}.json".format(exp_id), "w") as f:
        f.write(json.dumps(json_list))

    # learn on the whole set
    #
    # E_train, E_test = [], []
    # for _ in range(10):
    #     network = hof[0].createNetwork()
    #     network.fit(X_train, y_train,
    #                 batch_size=Config.batch_size, nb_epoch=Config.epochs, verbose=0)

    #     yy_train = network.predict(X_train)
    #     E_train.append(error(yy_train, y_train))

    #     yy_test = network.predict(X_test)
    #     E_test.append(error(yy_test, y_test))

    # def print_stat(E, name):
    #     print("E_{:6} avg={:.4f} std={:.4f}  min={:.4f} max={:.4f}".format(name,
    #                                                                        np.mean(
    #                                                                            E),
    #                                                                        np.std(
    #                                                                            E),
    #                                                                        np.min(
    #                                                                            E),
    #                                                                        np.max(E)))

    # print_stat(E_train, "train")
    # print_stat(E_test, "test")
