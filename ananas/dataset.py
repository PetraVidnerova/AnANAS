import pandas as pd
import numpy as np 
import tensorflow as tf
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.utils import np_utils

import config


data_modules_dict = {
    "mnist": mnist,
    "cifar10": cifar10,
    "fashion_mnist": fashion_mnist
}

def xy_zip(list_of_datasets):

    n_epochs = config.global_config["main_alg"]["epochs"]
    
    input_parts = []
    output_parts = [] 
    for dataset in list_of_datasets:
        input_parts.append(
            dataset.map(lambda image, label: image).repeat(n_epochs)
        )
        output_parts.append(
            dataset.map(lambda image, label: label).repeat(n_epochs)
        )
    return zip(zip(*input_parts), zip(*output_parts))

class KFoldC():

    def __init__(self, dataset, k=10): 

        data_len = config.global_config["dataset"]["len_train"]

        fold_size = data_len // k


        dataset = dataset.enumerate()
        folds = [] 
        for i in range(k-1):
            fold = (
                dataset
                .filter(lambda f, data: f >= i*fold_size and f < (i+1)*fold_size)
                .map(lambda f, data: data)
            )
            folds.append(fold)
        last_fold = (
            dataset
            .filter(lambda f, data: f >= (fold_size)*(k-1))
            .map(lambda f, data: data)
        )
        folds.append(last_fold)

        self.k = k 
        self.folds = folds

        self.train_sets = [
            self.create_i_train(i)
            for i in range(self.k)
        ]
        self.test_sets = [
            self.create_i_test(i)
            for i in range(self.k)
        ]
        
    def create_i_train(self, i):
        if i >= self.k:
            raise ValueError("i out of range")

        trainset_folds = [
            fold
            for j, fold in enumerate(self.folds)
            if j != i
        ]

        trainset = trainset_folds[0]
        for j in range(1, self.k-1):
            trainset = trainset.concatenate(trainset_folds[j])
        
        return trainset

    def create_i_test(self, i):
        return self.folds[i]

    def get_i_train(self, i):
        return self.train_sets[i]

    def get_i_test(self, i):
        return self.test_sets[i]
    
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

    if source_type == "tfrecords":
        ximg = kwargs["ximg"]
        yimg = kwargs["yimg"]

        def _extract_fn(tfrecord):
            # Extract features
            features = {
                'fpath': tf.io.FixedLenFeature([1], tf.string),
                'image': tf.io.FixedLenFeature([ximg * yimg], tf.int64),
                'label': tf.io.FixedLenFeature([6], tf.float32)
            }

            # Extract the data record
            sample = tf.io.parse_single_example(tfrecord, features)
            fpath = sample['fpath']
            image = sample['image']
            label = sample['label']

            fpath = tf.cast(fpath, tf.string)

            image = tf.reshape(image, [ximg, yimg, 1])
            image = tf.cast(image, 'float32')

            coords = tf.cast(label, 'float32')
            
            # return fpath, image, coords
            return image, coords

        dstype = 'train'
        tfrecord_file = f"{name}train.tfrecord"
        dataset = tf.data.TFRecordDataset([tfrecord_file])
        dataset = dataset.map(_extract_fn)
        dataset1 = dataset
        train_dataset = dataset1.shuffle(buffer_size=3000, reshuffle_each_iteration=False)

        dstype = 'test'
        tfrecord_file = f"{name}test.tfrecord"
        dataset = tf.data.TFRecordDataset([tfrecord_file])
        test_dataset = dataset.map(_extract_fn)

        if test:
            return train_dataset, test_dataset #, faketrain_dataset
        else:
            return train_dataset
        
    elif source_type == "keras":
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
        
        
        
    raise NotImplementedError("unsuported dataset type") 




