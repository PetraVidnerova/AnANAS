import pandas as pd
import numpy as np 
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.utils import np_utils

data_modules_dict = {
    "mnist": mnist,
    "cifar10": cifar10,
    "fashion_mnist": fashion_mnist
}
    
def load_data(source_type, name, test=False, flatten=True):

    if source_type == "keras":
        try:
            (X_train, y_train), (X_test, y_test) = data_modules_dict[name].load_data()
        except KeyError:
            raise ValueError("unsuported dataset") 

        
        if not test:
            if flatten:
                X_train = X_train.reshape(X_train.shape[0], -1)
            else:
                X_train  = X_train[..., np.newaxis] 
            X_train = X_train.astype('float32')
            X_train /= 255

            Y_train = np_utils.to_categorical(y_train)
            
            return X_train, Y_train

        else:
            if flatten:
                X_test = X_test.reshape(X_test.shape[0], -1)
            else:
                X_test  = X_test[..., np.newaxis] 
            X_test = X_test.astype('float32')
            X_test /= 255

            Y_test = np_utils.to_categorical(y_test)
            
            return X_test, Y_test
            
        
        
    raise NotImplementedError("unsuported dataset type") 




