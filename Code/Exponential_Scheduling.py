def exponential_scheduling(eta0, s):  # Exponential scheduling
    def exp_lr(epoch):
        return eta0 * 0.1 ** (epoch / s)

    return exp_lr
