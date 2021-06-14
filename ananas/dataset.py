import pandas as pd
import numpy as np 
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.utils import np_utils

import config


data_modules_dict = {
    "mnist": mnist,
    "cifar10": cifar10,
    "fashion_mnist": fashion_mnist
}
    
def load_data(source_type, name, test=False, flatten=True, **kwargs):
    """ Load dataset and returns X and Y either for trainset or test set. 

    parameters:
        source_type: either "keras" or "csv". "keras" uses the keras.datasets.
                     "csv" data will be readed from text file. 
        name: for keras datasets "mnist", "cifar10", "fashion_mnist" are supported
              for "csv" it stands for the name of file with trainset (including suffix) 
        test: default False, set True if you want to return test set instead of trainset 
        flatten: default True, use False if using convolutional networks 
        test_name: optional, if using "csv", specifies name of file with test set 
    """ 
    
    if source_type == "keras":
        try:
            (X_train, y_train), (X_test, y_test) = data_modules_dict[name].load_data()
        except KeyError:
            raise ValueError("unsuported dataset") 

        
        if not test:
            X = X_train
            y = y_train
        else:
            X = X_test
            y = y_test 

        if flatten:
            X = X.reshape(X.shape[0], -1)
        else:
            X  = X_train[..., np.newaxis]
            
        X = X.astype('float32')
        X /= 255

        Y = np_utils.to_categorical(y)
            
        return X, Y
            
    if source_type == "csv":
        if not test:
            df = pd.read_csv(name, header=None)
        else:
            df = pd.read_csv(kwargs["test_name"], header=None)

        # last column as output
        y = df.pop(df.columns[-1])
        if flatten:
            X = df.to_numpy() 
        else:
            raise NotImplementedError("unsuported dataset type") 
        
        X = X.astype('float32')
        if config.global_config["main_alg"]["task_type"] == "binary_classification":
            assert len(y.unique()) == 2
            return X, y.to_numpy().reshape(-1,1)
        
        Y = np_utils.to_categorical(y) 
        return X, Y

    if source_type == "numpy":
        if not test:
            X_name = name + "X_train.npy"
            y_name = name + "y_train.npy"
        else:
            X_name = name + "X_test.npy"
            y_name = name + "y_test.npy"
            
        return np.load(X_name),  np.load(y_name) 
        
    raise NotImplementedError("unsuported dataset type") 




