# author: Štěpán Procházka
import numpy as np


def nsga(fitnesses, population_size):
    """
    NSGA implementation with crowding-distance for secondary sorting

    Parameters
    ----------
    fitnesses : np.ndarray
        array of shape (n_individuals, m_fitnesses),
        with columns sorted by significance (descending)
    population_size : int
        number of individuals to be chosen

    Returns
    -------
    Sequence[int]
        indices of chosen individuals (based on maximization of given fitnesses)
    """
    pool_size = len(fitnesses)

    fitnesses_count = fitnesses.shape[1]
    fitnesses_rank_to_index = np.argsort(fitnesses, axis=0)
    fitnesses_index_to_rank = np.argsort(fitnesses_rank_to_index, axis=0)

    # find dominating and dominated
    # indices of individuals one is dominating
    subs_indices = [None] * (pool_size)
    # number of individuals one is dominated by
    doms_count = np.empty(shape=pool_size, dtype=np.int)

    for i_i in range(pool_size):
        # iteratively build dominating and dominated
        i_rank = fitnesses_index_to_rank[i_i, 0]
        subs = fitnesses_rank_to_index[:i_rank, 0]
        doms = fitnesses_rank_to_index[i_rank + 1:, 0]
        for f_i in range(1, fitnesses_count):
            i_rank = fitnesses_index_to_rank[i_i, f_i]
            subs = np.intersect1d(
                subs, fitnesses_rank_to_index[:i_rank, f_i], assume_unique=True)
            doms = np.intersect1d(
                doms, fitnesses_rank_to_index[i_rank + 1:, f_i], assume_unique=True)

        # assign dominating and dominated
        subs_indices[i_i] = subs
        doms_count[i_i] = len(doms)

    # build non-dominated fronts
    fronts = []  # list of fronts (indices of individuals per front)
    choice_size = 0  # number of individuals chosen so far
    while True:
        # fetch non-dominated solutions
        front_indices, *_ = np.where(doms_count == 0)

        # keep front indices and update count of individuals chosen so far
        fronts.append(front_indices)
        choice_size += len(front_indices)

        if choice_size >= population_size:
            break

        # invalidate individuals for future loop executions
        doms_count[front_indices] = -1

        # remove current front
        for i_i in range(len(front_indices)):
            doms_count[subs_indices[front_indices[i_i]]] -= 1

    # secondary sorting
    if choice_size > population_size:
        # crowding-distance
        last_front = fronts[-1]

        normalised_distances = np.empty((len(last_front), fitnesses_count))
        for f_i in range(fitnesses_count):
            front_rank_to_index = np.argsort(
                fitnesses_index_to_rank[last_front, f_i])
            sorted_front = last_front[front_rank_to_index]

            dim_distances = fitnesses[sorted_front[2:],
                                      f_i] - fitnesses[sorted_front[:-2], f_i]
            dim_range = fitnesses[sorted_front[-1],
                                  f_i] - fitnesses[sorted_front[0], f_i]
            normalised_distances[front_rank_to_index[1:-1],
                                 f_i] = dim_distances / dim_range
            normalised_distances[front_rank_to_index[[0, -1]], f_i] = np.inf

        averaged_distances = np.average(normalised_distances, axis=1)

        choice = np.concatenate(
            fronts[:-1] + [last_front[np.argsort(averaged_distances)[(choice_size - population_size):]]])
    else:
        choice = np.concatenate(fronts)

    return choice


def selectNSGA(pool, n):
    """ select n individuals from the pool
    returns list of references to original objects """ 

    fitnesses = [ ind.fitness.values for ind in pool ] 
    selected = nsga(fitnesses, n) 
    
    return [ pool[i] for i in selected ]
