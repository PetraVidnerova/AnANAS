import random
import pickle
import numpy as np
import multiprocessing
import json
import logging

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
from utils import error
from nsga import selectNSGA

import argparse

import config
from config import load_config, print_config



parser = argparse.ArgumentParser()
parser.add_argument('task', metavar='TASK', help="name of a yaml file with a task definition")
parser.add_argument('--id', help='computation id')
parser.add_argument('--log', help='set logging level, default INFO')
args = parser.parse_args()

if args.log is None:
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=args.log)
    
load_config(args.task)
print_config()

if "dataset" not in config.global_config:
    config.global_config["dataset"] = {"source_type": "keras", "name": "mnist"}
    logging.warning("Using default dataset MNIST") 


use_conv_layers = config.global_config.get("network_type","dense") == "conv"

if use_conv_layers:
    logging.info("Using convolutional layers")
else:
    logging.info("Using dense layers only.")

    
exp_id = args.id
if exp_id is None:
    exp_id = ""
    logging.info("No experiment id used.")
    
nsga_number = config.global_config.get("nsga", 2)
logging.info(f"Using NSGA{nsga_number}")

checkpoint_file = config.global_config.get("checkpoint_file", None)


# for classification fitness is accuracy, for approximation fitness is error
# second fitness element is network size, should be minimised
# approximation is default 
if config.global_config["main_alg"].get("task_type", "") in  ("classification", "binary_classification"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))
    logging.info("Classification task setup.")
else:
    creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0))
    logging.info("Regression task setup.")

creator.create("Individual",
               ConvIndividual if use_conv_layers else Individual,
               fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Structure initializers
toolbox.register("individual", initIndividual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# use multiple processors or GPU
if config.global_config["device"]["device_type"] == "CPU":
    n_cpus = config.global_config["device"]["n_cpus"]
    pool = multiprocessing.Pool(n_cpus)
    toolbox.register("map", pool.map)
    logging.info(f"Running on {n_cpus} CPUs")
else:
    logging.info(f"Running on GPU.")

    
# register operators
fit = Fitness(**config.global_config["dataset"])
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

    if "random_seed" in config.global_config: 
        random.seed(config.global_config["random_seed"])

    if checkpoint_name:
        # A file name has been given, then load the data from the file
        cp = pickle.load(open(checkpoint_name, "rb"))
        pop = cp["population"]
        start_gen = cp["generation"] + 1
        hof = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
        if "config" in cp:
            config.global_config.update(cp["config"])

    else:
        pop_size = config.global_config["ga"]["pop_size"]
        pop = toolbox.population(n=pop_size)
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
                         ngen=config.global_config["ga"]["n_gen"],
                         stats=stats,
                         halloffame=hof,
                         logbook=logbook,
                         verbose=True,
                         exp_id=exp_id)

    return pop, log, hof


if __name__ == "__main__":

    # set network cfg
    input_shape = fit.get_data_size()
    noutputs = fit.get_n_outputs()

    config.global_config["input_shape"] = input_shape
    config.global_config["noutputs"] = noutputs

    
    print_config()
    
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
