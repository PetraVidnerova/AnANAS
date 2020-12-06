import keras
import random
import numpy as np
import pickle
from sklearn.model_selection import KFold
from dataset import load_data
from utils import error
from keras import backend as K
import config

class Database:

    def __init__(self):
        self.data = []

    def insert(self, individual, fitness):
        self.data.append((individual, fitness))

    def save(self, name):
        with open(name, "wb") as f:
            pickle.dump(self.data, f)


class Fitness:

    def __init__(self, source_type="keras", name="mnist"):

        if source_type != "keras":
            raise NotImplementedError("temporarily only keras datasets available")
        # load train data

        flatten = config.global_config["network_type"] == "dense"
        self.X, self.y = load_data(source_type, name, flatten=flatten)

        self.input_shape = self.X[0].shape
        self.noutputs = self.y.shape[1]

    def evaluate_batch(self, individuals):

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        xval_datasets = np.asarray([
            (self.X[train], self.y[train], self.X[test], self.y[test])
            for train, test in kf.split(self.X)
        ], dtype=object)

    
        xval_features = [
            keras.layers.InputLayer(self.input_shape)
            for _ in xval_datasets
        ]

        xval_models = []
        for input_features in xval_features:
            individual_models = [
                individual.createNetwork(input_features)
                for individual in individuals
            ]
            # TODO(proste) is it intended to effectively bin model sizes?
            sizes = [(m.count_params() // 1000) for m in individual_models]

            xval_models.extend(individual_models)

        multi_model = keras.Model(
            inputs=[input_features.input for input_features in xval_features],
            outputs=[
                individual_model.output
                for individual_model in xval_models
            ]
        )
        multi_model.compile(
            loss=config.global_config["main_alg"]["loss"],
            optimizer=keras.optimizers.RMSprop()
        )

        multi_model.fit(
            list(xval_datasets[:, 0]),
            [y_train for y_train in xval_datasets[:, 1] for _ in individuals],
            batch_size=config.global_config["main_alg"]["batch_size"],
            epochs=config.global_config["main_alg"]["epochs"],
            verbose=0
        )

        pred_test = multi_model.predict(list(xval_datasets[:, 2]))
        scores = np.array([
            error(xval_datasets[test_i // len(individuals), 3], yy_test)
            for test_i, yy_test in enumerate(pred_test)
        ]).reshape(-1, len(individuals))

        K.clear_session()  # free resources allocated by models

        fitness = np.mean(scores, axis=0)

        return list(zip(fitness, sizes))

    def evaluate(self, individual):

        
        random.seed(42)
        # perform KFold crossvalidation
        kf = KFold(n_splits=5)
        scores = []
        for train, test in kf.split(self.X):   # train, test are indicies
            X_train, X_test = self.X[train], self.X[test]
            y_train, y_test = self.y[train], self.y[test]

            model = individual.createNetwork()
            size = model.count_params() // 1000
            model.fit(X_train, y_train,
                      batch_size=config.global_config["main_alg"]["batch_size"],
                      epochs=config.global_config["main_alg"]["epochs"],
                      verbose=0)

            yy_test = model.predict(X_test)
            scores.append(error(y_test, yy_test))

        fitness = np.mean(scores)

        # I try this to prevent memory leaks in nsga2-keras
        K.clear_session()

        return fitness, size
