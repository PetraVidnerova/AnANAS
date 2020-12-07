import click
import pickle
import matplotlib.pyplot as plt

from keras import Model
from keras.layers import InputLayer
from keras.optimizers import RMSprop
from deap import creator, base


from individual import Individual
from convindividual import ConvIndividual
from dataset import load_data
from utils import error, print_stat

import config

# TODO config
USE_CONV = False
FINAL_EVALS = 10 

def init():
    creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual",
                   ConvIndividual if USE_CONV else Individual,
                   fitness=creator.FitnessMax)


def load_checkpoint(name):
    init()
    cp = pickle.load(open(name, "rb"))
    # print(cp.keys())
    pop = cp["population"]
    front = cp["halloffame"]
    log = cp["logbook"]
    cfg = cp["config"]
        
    return pop, front, log, cfg


@click.group()
@click.option("--conv", default=False, type=bool)
def main(conv):
    global USE_CONV

    if conv:
        USE_CONV = True



@main.command()
@click.argument("cp_name")
def list_front(cp_name):
    """ List fitness values of individuals from the pareto front. """ 

    front = load_checkpoint(cp_name)[1]
    print("size", len(front))
    for i, ind in enumerate(front):
        print("{}: {} {}".format(i, ind.fitness.values[0], ind.fitness.values[1]))
    

def eval_mean(ind, X_train, y_train, X_test, y_test, evals=10):
    #        E_train, E_test = [], []  # list of accuracies

    input_features = InputLayer(X_train[0].shape)

    individual_models = [
        ind.createNetwork(input_features)
        for _ in range(evals)
    ]

    individual_models[0].summary()
    
    multi_model = Model(
        inputs = [input_features.input],
        outputs = [
            individual_model.output
            for individual_model in individual_models
        ]
    )

    multi_model.compile(
        loss = config.global_config["main_alg"]["loss"],
        optimizer = RMSprop()
    )

    multi_model.fit(
        X_train,
        [y_train for _ in range(evals)],
        batch_size=config.global_config["main_alg"]["batch_size"],
        epochs=config.global_config["main_alg"]["epochs"], verbose=0
    )

    pred_test = multi_model.predict(X_test)
    E_test = [
        error(y_test, yy_test)
        for yy_test in pred_test
    ]

    pred_train = multi_model.predict(X_train)
    E_train = [
        error(y_train, yy_train)
        for yy_train in pred_train
    ]
    
    # for _ in range(5):
    #     network = ind.createNetwork()
    #     network.fit(X_train, y_train,
    #                 batch_size=Config.batch_size,
    #                 nb_epoch=20,
    #                 verbose=0)
    
    #     yy_train = network.predict(X_train)
    #     E_train.append(error(yy_train, y_train))
    
    #     yy_test = network.predict(X_test)
    #     E_test.append(error(yy_test, y_test))
    #     del network 
    return E_train, E_test 

@main.command()
@click.argument("cp_name")
@click.option("--data_source", default=None)
@click.option("--trainset", default=None)
@click.option("--testset", default=None)
def eval_front(cp_name, data_source, trainset, testset):
    """ Evaluate all individuals from the pareto front. """ 
    
    _, front, _, cfg = load_checkpoint(cp_name) 

    config.global_config.update(cfg)

    if data_source is None:
        assert trainset is None and testset is None
        data_source  = config.global_config["dataset"]["source_type"]
        trainset = config.global_config["dataset"]["name"]
        test_name = config.global_config["dataset"]["test_name"]

        
    # load the whole data
    X_train, y_train = load_data(data_source, trainset)
    X_test, y_test = load_data(data_source, trainset, test=True, test_name=test_name)

    for i, ind in enumerate(front):
        
        E_train, E_test = eval_mean(ind, X_train, y_train, X_test, y_test)
        print(i, ": ", end="")
        print_stat(E_train, "train")
        print(i, ": ", end="")
        print_stat(E_test, "test")
        print(i, ": ", end="")
        print(ind.fitness.values)
        print(flush=True)


@main.command()
@click.argument("i", type=int)
@click.argument("cp_name")
@click.option("--data_source", default=None)
@click.option("--trainset", default=None)
@click.option("--testset", default=None)
def evaluate(i,  cp_name, data_source, trainset, testset):
    """ Evaluate the I-th individual from the pareto front. """ 
    _, front, _, cfg = load_checkpoint(cp_name) 
        
    config.global_config.update(cfg)

    if data_source is None:
        assert trainset is None and testset is None
        data_source  = config.global_config["dataset"]["source_type"]
        trainset = config.global_config["dataset"]["name"]
        test_name = config.global_config["dataset"]["test_name"]
    else:
        test_name = testset
        
    # load the whole data
    X_train, y_train = load_data(data_source, trainset)
    X_test, y_test = load_data(data_source, trainset, test=True, test_name=test_name)

    ind = front[i] 
    E_train, E_test = eval_mean(ind, X_train, y_train, X_test, y_test)
    print(i, ": ", end="")
    print_stat(E_train, "train")
    print(i, ": ", end="")
    print_stat(E_test, "test")
    print(i, ": ", end="")
    print(ind.fitness.values)
    print(flush=True)


@main.command()
@click.argument("cp_name")
def plot(cp_name):
    log = load_checkpoint(cp_name)[2]
    acc_avg = [line["avg"][0] for line in log]
    acc_size = [line["avg"][1] for line in log]

    acc_max = [line["max"][0] for line in log]
    size_min = [line["min"][1] for line in log]

    ax1, ax2 = plt.subplot(221), plt.subplot(222)

    ax1.plot(acc_avg, color="blue")
    ax2.plot(acc_size, color="green")
    ax1.set_title("avg acc")
    ax2.set_title("avg size")

    ax3, ax4 = plt.subplot(223), plt.subplot(224)
    ax3.plot(acc_max, color="blue")
    ax4.plot(size_min, color="green")
    ax3.set_title("max acc")
    ax4.set_title("min size")

    plt.show()


@main.command()
@click.argument("cp_name")
def query_iter(cp_name):
    """ Show the numer of last iteration. """
    
    log = load_checkpoint(cp_name)[2]

    print("Last generation:", [line["gen"] for line in log][-1])


if __name__ == "__main__":
    main()
