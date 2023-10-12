import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# ============================================= Plot prediction ======================================================
def plot_prediction(y_train, y_test, pred_train, pred_test, mod, axs, name=None): 

    if name == "RNN":                                   # Plot the data only once
        axs[0].plot(range(0, len(y_train)), mod.inverse_transform(y_train[:,0].reshape(-1,1)), label='Data')
        axs[1].plot(range(len(y_train)+1, len(y_train)+len(y_test)+1), mod.inverse_transform(y_test[:,0].reshape(-1,1)), label='Data')  
        
    r2_tr = metrics.r2_score(y_train, pred_train)
    r2_te = metrics.r2_score(y_test, pred_test)
    axs[0].plot(range(0, len(y_train)), mod.inverse_transform(pred_train[:,0].reshape(-1,1)), 
                label=f' $R^2_{{{name}}}:${np.round(r2_tr, 2)}')
    axs[0].legend(fontsize=10, ncol=2, loc='best', labelcolor='linecolor', handlelength=0, handletextpad=0, columnspacing=0, title='Train')

    axs[1].plot(range(len(y_train)+1, len(y_train)+len(y_test)+1), mod.inverse_transform(pred_test[:,0].reshape(-1,1)),
                label=f' $R^2_{{{name}}}:${np.round(r2_te, 2)}')
    axs[1].legend(fontsize=10, ncol=2, loc='best', labelcolor='linecolor', handlelength=0, handletextpad=0, columnspacing=0, title='Test')
    axs[0].autoscale(enable=True, axis="x",tight=True)
    axs[1].autoscale(enable=True, axis="x",tight=True)
    plt.subplots_adjust(wspace=0.035, hspace=0.2)