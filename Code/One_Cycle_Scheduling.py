from tensorflow import keras


class one_cycle_scheduling(keras.callbacks.Callback):
    def __init__(self, iterations, max_lr, init_lr=None, final_steps=None, final_lr=None):
        self.iterations = iterations
        self.max_lr = max_lr
        self.init_lr = init_lr or max_lr / 10
        self.final_steps = final_steps or int(iterations / 10) + 1
        self.half_iterations = int((iterations - self.final_steps) / 2)
        self.final_lr = final_lr or self.init_lr / 1000
        self.iteration = 0

    def interpolation(self, y2, y1, x2, x1):
        return (y2 - y1) / (x2 - x1) * (self.iteration - x1) + y1

    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iterations:
            lr = self.interpolation(self.max_lr, self.init_lr, self.half_iterations, 0)
        elif self.iteration < 2 * self.half_iterations:
            lr = self.interpolation(self.init_lr, self.max_lr, 2 * self.half_iterations, self.half_iterations)
        else:
            lr = self.interpolation(self.final_lr, self.init_lr, self.iterations, 2 * self.half_iterations)
        self.iteration += 1
        keras.backend.set_value(self.model.optimizer.learning_rate, lr)