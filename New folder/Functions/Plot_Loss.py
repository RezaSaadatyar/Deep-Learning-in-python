import matplotlib.pyplot as plt

# ====================================== Plot loss for train & test ==================================================   
def plot_loss(train_model, axs, title=None):
    # _, axs = plt.subplots(nrows=1, sharey='row', figsize=(4, 3))
    axs.plot(train_model.history["loss"], label='training loss')
    axs.plot(train_model.history["val_loss"], label='validation loss',)
    axs.legend(fontsize=10, ncol=1, loc='best', labelcolor='linecolor', handlelength=0)
    axs.autoscale(enable=True, axis="x",tight=True)
    axs.set_xlabel('Epochs', fontsize=10)
    axs.set_title(title, fontsize=10)
    plt.gca().yaxis.set_major_formatter('{:.2f}'.format)
    axs.grid(axis='y', linestyle='--', alpha=0.25)
    plt.subplots_adjust(wspace=0.035, hspace=0.2)