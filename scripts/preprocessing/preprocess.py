import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse


parser = argparse.ArgumentParser(
    description='Specify the path from where the datasets should be loaded and where the preprocessed datasets should be stored')
parser.add_argument('-input', type=str, help="Path to dataset folders")
parser.add_argument('-output', type=str,
                    help="Path to where the preprocessed datasets should be stored")

args = parser.parse_args()


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


# loading our created raw data
train_ds = tf.data.experimental.load(args.input+"/train",element_spec=(tf.TensorSpec(shape=(64000,),dtype=tf.float32, name=None),tf.TensorSpec(shape=(64000,),dtype=tf.int16, name=None)))
valid_ds = tf.data.experimental.load(args.input+"/valid",element_spec=(tf.TensorSpec(shape=(64000,),dtype=tf.float32, name=None),tf.TensorSpec(shape=(64000,),dtype=tf.int16, name=None)))
test_ds = tf.data.experimental.load(args.input+"/test",element_spec=(tf.TensorSpec(shape=(64000,),dtype=tf.float32, name=None),tf.TensorSpec(shape=(64000,),dtype=tf.int16, name=None)))

# performing preprocessing steps
train_ds = preprocess(train_ds)
valid_ds = preprocess(valid_ds)
test_ds = preprocess(test_ds)

# saving our preprocessed data
tf.data.experimental.save(train_ds, args.output+"/train")
tf.data.experimental.save(valid_ds, args.output+"/valid")
tf.data.experimental.save(test_ds, args.output+"/test")
