# ============================================= Import Libraries ===========================================
from tensorflow import keras  # print(keras.__version__, tf.__version__)
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection, preprocessing

# =========================================== Regression ===================================================
# --------------------------------------- Step 1. Load data ------------------------------------------------
Data = datasets.fetch_california_housing()
X_train0, X_test, y_train0, y_test = model_selection.train_test_split(Data["data"], Data["target"], test_size=0.25)
X_train1, X_validation, y_train1, y_validation = model_selection.train_test_split(X_train0, y_train0, test_size=0.25)
# --------------------------------------- Step 2. Normalize data -------------------------------------------
sc = preprocessing.StandardScaler()
X_train_s = sc.fit_transform(X_train1)
X_validation_s = sc.transform(X_validation)
X_test_s = sc.transform(X_test)


# normalize_regression(X_train1, type_normalize='MinMaxScaler', display_figure='on')
# ------------------------ Step 3. Creating Model, Hyperparameter optimization & Compile -------------------
def ann_model(number_of_hidden_layers=1, number_of_neurons=50, lr=0.01):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[8]))
    for hidden_layer in range(number_of_hidden_layers):
        model.add(keras.layers.Dense(number_of_neurons, activation="selu"))
    model.add(keras.layers.Dense(1))
    sgd = keras.optimizers.SGD(lr=lr)
    model.compile(loss="mse", optimizer=sgd)
    return model


keras_sk_reg = keras.wrappers.scikit_learn.KerasRegressor(build_fn=ann_model)
param_grid = {"number_of_hidden_layers": [1, 3, 5], "number_of_neurons": [50, 100, 150], "lr": [0.001, 0.001, 0.1]}
keras_sk_reg_gs = model_selection.GridSearchCV(keras_sk_reg, param_grid)
# ------------------------------- Step 5: Fit, evaluate & predictModel -------------------------------------
keras_sk_reg_gs.fit(X_train_s, y_train1, epochs=2, validation_data=(X_validation_s, y_validation),
                    callbacks=[keras.callbacks.EarlyStopping(patience=5)])
keras_sk_reg_gs.param_grid
keras_sk_reg_gs.best_params_
final_model = keras_sk_reg_gs.best_estimator_.model

