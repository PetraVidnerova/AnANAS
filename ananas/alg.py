import keras
import random
import pickle
import datetime

from deap import algorithms

from config import Config


def eval_invalid_inds_single_gpu(pop, toolbox):
    """ Evaluate the individuals with an invalid fitness 
    Returns the number of reevaluated individuals.
    """
    invalid_pop = [ind for ind in pop if not ind.fitness.valid]
    eval_size = len(invalid_pop)

    for batch_begin in range(0, eval_size, Config.eval_batch_size):
        batch_end = min(batch_begin + Config.eval_batch_size, eval_size)
        batch_individuals = invalid_pop[batch_begin:batch_end]
        batch_fitness = toolbox.eval_batch(batch_individuals)

        for individual, fitness in zip(batch_individuals, batch_fitness):
            individual.fitness.values = fitness

    return eval_size

# TODO: stupid way how to do it, fix 
def eval_invalid_inds_cpu_parallel(pop, toolbox): 
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    ind_batches = [
        [individual]
        for individual in invalid_ind
    ] 
    # fitnesses = toolbox.map(
    #     lambda x: toolbox.eval_batch(x)[0], # returns one lement list 
    #     ind_batches 
    # )
    
    fitnesses_in_lists = toolbox.map(
        toolbox.eval_batch,
        ind_batches
    )
    
    fitnesses = [l[0] for l in fitnesses_in_lists]

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    return len(invalid_ind)

def nsga(population, start_gen, toolbox, cxpb, mutpb, ngen,
         stats, halloffame, logbook, verbose, exp_id=None):

    eval_invalid_inds = ( 
        eval_invalid_inds_single_gpu 
        if Config.GPU 
        else eval_invalid_inds_cpu_parallel
    )

    popsize = len(population)
    total_time = datetime.timedelta(seconds=0)

    eval_invalid_inds(population, toolbox)

    for gen in range(start_gen, ngen):
        start_time = datetime.datetime.now()

        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb,
                                      mutpb=mutpb)

        evals = eval_invalid_inds(offspring, toolbox)

        population = toolbox.select(population+offspring, k=popsize)

        halloffame.update(offspring)  # updates pareto front
        # update statics
        record = stats.compile(population)
        logbook.record(gen=gen, evals=evals, **record)
        if verbose:
            print(logbook.stream, flush=True)

        # save checkpoint
        if gen % 1 == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(
                population=population,
                generation=gen,
                halloffame=halloffame,
                logbook=logbook,
                rndstate=random.getstate(),
                config=Config,
            )
            if exp_id is None:
                cp_name = "checkpoint_nsga.pkl"
            else:
                cp_name = "checkpoint_nsga_{}.pkl".format(exp_id)
            pickle.dump(cp, open(cp_name, "wb"))

        # check hard time limit
        gen_time = datetime.datetime.now() - start_time
        total_time = total_time + gen_time
        print("Time ", total_time)

        # hard time limit was necessary at metacentrum
        #
        # if total_time > datetime.timedelta(hours=4*24):
        #     print("Time limit exceeded.")
        #     break

    return population, logbook


def vanilla_ga(population, start_gen, toolbox, cxpb, mutpb, ngen,
               stats, halloffame, logbook, verbose, exp_id=None):
    """ Basic GA algorithm. Just to have a baseline. """

    eval_invalid_inds = ( 
        eval_invalid_inds_single_gpu 
        if Config.GPU 
        else eval_invalid_inds_cpu_parallel
    )

    total_time = datetime.timedelta(seconds=0)
    eval_invalid_inds(population, toolbox)

    for gen in range(start_gen, ngen):
        start_time = datetime.datetime.now()
        population = algorithms.varAnd(population, toolbox,
                                       cxpb=cxpb, mutpb=mutpb)

        # Evaluate the individuals with an invalid fitness
        evals = eval_invalid_inds(population, toolbox)

        halloffame.update(population)
        record = stats.compile(population)
        logbook.record(gen=gen, evals=evals, **record)
        if verbose:
            print(logbook.stream)

        population = toolbox.select(population, k=len(population))

        if gen % 1 == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen,
                      halloffame=halloffame,
                      logbook=logbook, rndstate=random.getstate())
            if exp_id is None:
                cp_name = "checkpoint_ea.pkl"
            else:
                cp_name = "checkpoint_ea_{}.pkl".format(exp_id)
            pickle.dump(cp, open(cp_name, "wb"))

        gen_time = datetime.datetime.now() - start_time
        total_time = total_time + gen_time
        print("Time ", total_time)
        if total_time > datetime.timedelta(hours=4*24):
            print("Time limit exceeded.")
            break

    return population, logbook
