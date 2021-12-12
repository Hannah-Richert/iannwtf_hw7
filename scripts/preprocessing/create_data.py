import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='Specify the path where the created datasets should be stored')
parser.add_argument('-output', type=str,
                    help="Path to where the created datasets should be stored")

args = parser.parse_args()


def integration_task(seq_len, num_samples):
    """
    Generates our data.
    noise is our input: random, normal, distributed numbers.
    target categorizes the noise based on the sum of the noise and is either 0 or 1.
        Args:
            seq_len <int>: number of our random numbers for our noise-sample
            num_samples <int>: numbeer of samples the generator could yield
        Returns:
            (noise,target) <tuple<list,list>>: tuple of our noise sample and a target, which splits the noise samples into wto categories
    """
    for i in range(num_samples):

        # creating inputs and targets
        noise = np.random.normal(size=seq_len)

        target = int(np.sum(noise, axis=-1) > 1)

        # adjusting shapes
        noise = np.expand_dims(noise, -1)
        target = np.expand_dims(target, -1)

        yield (noise, target)


def my_integration_task():
    """
    Wrapper for function integration_task
        Returns:
            (x,y) <tuple<list,list>>:
    """

    # declaring data parameters
    num_samples = 96000
    seq_len = 25

    return integration_task(seq_len, num_samples)


# creating dataset with self-defined generator
ds = tf.data.Dataset.from_generator(
    my_integration_task, (tf.float32, tf.int16))

# splitting ds in training, validation and test data
train_ds = ds.take(64000)
remaining = ds.skip(64000)
valid_ds = remaining.take(16000)
test_ds = remaining.skip(16000)

# locally saving the created datasets
tf.data.experimental.save(train_ds, args.output+"/train")
tf.data.experimental.save(valid_ds, args.output+"/valid")
tf.data.experimental.save(test_ds, args.output+"/test")
