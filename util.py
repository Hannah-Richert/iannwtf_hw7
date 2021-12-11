import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np

def integration_task(seq_len,num_samples):
    for i in range(num_samples):
        noise = np.random.normal(size=seq_len)
        target = int(np.sum(noise,axis=-1)>1)
        noise= np.expand_dims(noise,-1)
        target = np.expand_dims(target,-1)
        yield (noise,target)
def my_integration_task():
    num_samples = 80000 #80000
    seq_len= 25 #25
    for (x,y) in integration_task(seq_len,num_samples):
        yield (x,y)


def load_data():
    """
    Loading and preprocessing the data.
        Returns:
          - train_ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: our training dataset
          - valid_ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: our validation dataset
          - test_ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: our test dataset
    """

    ds = tf.data.Dataset.from_generator(my_integration_task,(tf.float32,tf.int16))
    train_ds = ds.take(64000)
    remaining = ds.skip(64000)
    valid_ds = remaining.take(8000)
    test_ds = remaining.skip(8000)
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

def train_step(model, input, target, loss_function, optimizer, is_training):
    """
    Performs a forward and backward pass for  one dataponit of our training set
      Args:
        - model <tensorflow.keras.Model>: our created MLP model
        - input <tensorflow.tensor>: our input
        - target <tensorflow.tensor>: our target
        - loss_funcion <keras function>: function we used for calculating our loss
        - optimizer <keras function>: our optimizer used for backpropagation

      Returns:
        - loss <float>: our calculated loss for the datapoint
      """

    with tf.GradientTape() as tape:

        # forward step
        prediction = model(input)
        # calculating loss
        loss = loss_function(target, prediction)

        # calculaing the gradients
        gradients = tape.gradient(loss, model.trainable_variables)

    # updating weights and biases
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def test(model, test_data, loss_function, is_training):
    """
    Test our MLP, by going through our testing dataset,
    performing a forward pass and calculating loss and accuracy
      Args:
        - model <tensorflow.keras.Model>: our created MLP model
        - test_data <tensorflow.python.data.ops.dataset_ops.PrefetchDataset> our preprocessed test dataset
        - loss_funcion <keras function>: function we used for calculating our loss

      Returns:
          - loss <float>: our mean loss for this epoch
          - accuracy <float>: our mean accuracy for this epoch
    """

    # initializing lists for accuracys and loss
    accuracy_aggregator = []
    loss_aggregator = []
    optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    for (input, target) in test_data:

        # forward step
        prediction = model(input)
        #print("Pred:",prediction)
        ##prediction = prediction2[-1]
        #print("Target:",target)
        # calculating loss
        loss = loss_function(target, prediction)

        # add loss and accuracy to the lists
        loss_aggregator.append(loss.numpy())

        for t, p in zip(target, prediction):
            accuracy_aggregator.append(tf.cast(np.round(t.numpy(),0) == np.round(p.numpy(),0), tf.float32))
            #accuracy_aggregator.append(tf.reduce_mean(tf.cast(tf.math.argmax(t) == tf.math.argmax(p), tf.float32)))

    # calculate the mean of the loss and accuracy (for this epoch)
    loss = tf.reduce_mean(loss_aggregator)
    accuracy = tf.reduce_mean(accuracy_aggregator)


    return loss, accuracy


def visualize(train_losses, valid_losses, valid_accuracies):
    """
    Displays the losses and accuracies from the different models in a plot-grid.
      Args:
        - train_losses- <list>: mean training losses per epoch
        - valid_losses <list>: mean testing losses per epoch
        - valid_accuracies <list>: mean accuracies (testing dataset) per epoch
    """

    fig, axs = plt.subplots(2,1)

    axs[0].plot(train_losses)
    axs[0].plot(valid_losses)
    axs[1].plot(valid_accuracies)
    axs[1].sharex(axs[0])

    fig.legend([" Train_ds loss", " Valid_ds loss", " Valid_ds accuracy"])
    plt.xlabel("Training epoch")
    fig.tight_layout()
    plt.show()

def visualize_xxx(train_losses,valid_losses,valid_accuracies):
    """
    Displays the losses and accuracies from the different models in a plot-grid.
    Args:
      train_losses <list<list<float>>>: mean training losses per epoch
      valid_losses <list<list<float>>>: mean testing losses per epoch
      valid_accuracies <list<list<float>>>: mean accuracies (testing dataset) per epoch
    """

    titles = ["ResNet","DenseNet","SimpleModel"]
    fig, axs = plt.subplots(1, 3)
    #fig.set_size_inches(13, 6)
    parameters = ["16,266","13,490","33,686"]
    # making a grid with subplots
    for j in range(3):
        axs[j].plot(train_losses[j])
        axs[j].plot(valid_losses[j])
        axs[j].plot(valid_accuracies[j])
        last_accuracy = valid_accuracies[j][-1].numpy()
        axs[j].sharex(axs[0])
        axs[j].set_title(titles[j]+" \n Last Accuracy: "+str(round(last_accuracy,4))+" \n Trainable Parameters: "+parameters[j])


    fig.legend([" Train_ds loss"," Valid_ds loss"," Valid_ds accuracy"],loc='center right')
    plt.xlabel("Training epoch")
    fig.tight_layout()
    plt.show()
