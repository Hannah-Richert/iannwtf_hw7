import tensorflow as tf
from util import load_data, test, visualize
import numpy as np
from lstm import LSTMmodel
from classify import classify

tf.keras.backend.clear_session()

train_ds, valid_ds, test_ds = load_data()
optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)


models = [LSTMmodel()]


with tf.device('/device:gpu:0'):
    # training the model
    for model in models:
        results, trained_model = classify(model, optimizer, 5, train_ds, valid_ds)
        trained_model.summary()
        # saving results for visualization
        train_losses = results[0]
        valid_losses = results[1]
        valid_accuracies= results[2]

        # testing the trained model
        # (this code snippet should only be inserted when one decided on all hyperparameters)
        #_, test_accuracy = test(trained_model, test_ds,tf.keras.losses.CategoricalCrossentropy(),False)
        #print("Accuracy (test set):", test_accuracy)


        # visualizing losses and accuracy
    visualize(train_losses,valid_losses,valid_accuracies)
