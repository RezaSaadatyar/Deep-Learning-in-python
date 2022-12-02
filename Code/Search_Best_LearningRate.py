import numpy as np
from tensorflow import keras  # print(keras.__version__, tf.__version__)


class learnig_rate(keras.callbacks.Callback):
    def __init__(self, factor):              # Factor leads to reduce learning rate after each bach
        self.previous_loss = None
        self.factor = factor
        self.lr = []
        self.loss = []

    def on_epoch_begin(self, epoch, logs):   #
        self.previous_loss = 0

    def on_batch_end(self, batch, logs):     # Update learning rate in end of each bach
        current_loss = logs["loss"] * (batch + 1) - self.previous_loss * batch       # logs["loss"] = (loss(t+1)*loss(t)*batch)/(batch+1)
        self.previous_loss = logs["loss"]
        self.lr.append(keras.backend.get_value(self.model.optimizer.learning_rate))  # get learning_rate time t
        self.loss.append(current_loss)
        keras.backend.set_value(self.model.optimizer.learning_rate, self.model.optimizer.learning_rate * self.factor)  # update learning_rate time t+1


def search_best_learningrate(model, X, y, epochs=1, batch_size=32, min_lr=1e-6, max_lr=10):
    model_weights = model.get_weights()
    iterations = int(X.shape[0] / batch_size) * epochs   # X.shape[0]: The number of row data
    factor = np.exp(np.log(max_lr / min_lr) / iterations)
    initial_lr = keras.backend.get_value(model.optimizer.learning_rate)
    keras.backend.set_value(model.optimizer.learning_rate, min_lr)  # Start model with min_lr
    epx_lr = learnig_rate(factor)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[epx_lr])
    keras.backend.set_value(model.optimizer.learning_rate, initial_lr)
    model.set_weights(model_weights)
    return epx_lr.lr, epx_lr.loss
