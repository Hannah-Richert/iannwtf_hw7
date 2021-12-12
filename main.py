import tensorflow as tf
from util import  test, visualize
from dataPrep import load_data
import numpy as np
from lstm import LSTMmodel
from classify import classify

tf.keras.backend.clear_session()

train_ds, valid_ds, test_ds = load_data()

learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

models = [LSTMmodel(num_layer=1),LSTMmodel(num_layer=2)]

train_losses = []
valid_losses = []
valid_accuracies= []

with tf.device('/device:gpu:0'):
    # training the model
    for model in models:
        results, trained_model = classify(model, optimizer, 3, train_ds, valid_ds)
        trained_model.summary()

        # saving results for visualization
        train_losses.append(results[0])
        valid_losses.append(results[1])
        valid_accuracies.append(results[2])

        # testing the trained model
        # (this code snippet should only be inserted when one decided on all hyperparameters)
        _, test_accuracy = test(trained_model, test_ds,tf.keras.losses.CategoricalCrossentropy(),False)
        print("Accuracy (test set):", test_accuracy)


    # visualizing losses and accuracy
    visualize(train_losses,valid_losses,valid_accuracies)
