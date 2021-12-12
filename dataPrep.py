import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def integration_task(seq_len,num_samples):
    """
    ...
        Args:
            seq_len <int>:
            num_samples <int>:
        Returns:
            (noise,target) <tuple<list,list>>:
    """
    for i in range(num_samples):

        # creating inputs and targets
        noise = np.random.normal(size=seq_len)

        target = int(np.sum(noise,axis=-1)>1)

        #adjusting shapes
        noise= np.expand_dims(noise,-1)
        target = np.expand_dims(target,-1)

        yield (noise,target)

def my_integration_task():
    """
    Wrapper for function integration_task
        Returns:
            (x,y) <tuple<list,list>>:
    """

    # declaring data parameters
    num_samples = 96000
    seq_len= 25

    return integration_task(seq_len,num_samples)


def load_data():
    """
    Loading and preprocessing the data.
        Returns:
          - train_ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: our training dataset
          - valid_ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: our validation dataset
          - test_ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: our test dataset
    """
    # creating dataset with self-defined generator
    ds = tf.data.Dataset.from_generator(my_integration_task,(tf.float32,tf.int16))

    # splitting ds in training, validation and test data
    train_ds = ds.take(64000)
    remaining = ds.skip(64000)
    valid_ds = remaining.take(16000)
    test_ds = remaining.skip(16000)

    # preprocessing
    train_ds = preprocess(train_ds)
    valid_ds = preprocess(valid_ds)
    test_ds = preprocess(test_ds)

    return train_ds, valid_ds, test_ds

def preprocess(ds):
    """
    Preparing our data for our model.
      Args:
        - ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: the dataset we want to preprocess

      Returns:
        - ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: preprocessed dataset
    """

    # cache
    ds = ds.cache()
    # shuffle, batch, prefetch our dataset
    ds = ds.shuffle(5000)
    ds = ds.batch(32)
    ds = ds.prefetch(20)
    return ds
