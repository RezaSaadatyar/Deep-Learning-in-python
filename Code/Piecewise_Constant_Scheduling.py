def piecewise_constant_scheduling(epoch):       # Piecewise constant scheduling
    if epoch < 10:
        return 0.01
    elif epoch < 20:
        return 0.005
    elif epoch < 40:
        return 0.001
    else:
        return 0.0001
