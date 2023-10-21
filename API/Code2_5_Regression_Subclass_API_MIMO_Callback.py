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
# ------------------------------------ Step 3. Creating a Functional API -----------------------------------
X_train_s_1, X_train_s_2 = X_train_s[:, :6], X_train_s[:, -4:]
X_validation_s_1, X_validation_s_2 = X_validation_s[:, :6], X_validation_s[:, -4:]
X_test_s_1, X_test_s_2 = X_test_s[:, :6], X_test_s[:, -4:]


class WideAndDeepANN(keras.Model):
    def __init__(self, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden_layer_1 = keras.layers.Dense(50, activation=activation)
        self.hidden_layer_2 = keras.layers.Dense(10, activation=activation)
        self.final_output = keras.layers.Dense(1)
        self.helper_output = keras.layers.Dense(1)

    def call(self, inputs):
        input_1, input_2 = inputs
        h1_out = self.hidden_layer_1(input_1)
        h2_out = self.hidden_layer_2(h1_out)
        concat_out = keras.layers.concatenate([input_2, h2_out])
        final_out = self.final_output(concat_out)
        helper_out = self.helper_output(h2_out)
        return final_out, helper_out


model = WideAndDeepANN()
# --------------------------------------- Step 4. Compile Model --------------------------------------------
model.compile(loss=["mse", "mse"], loss_weights=[0.8, 0.2], optimizer="sgd",  # mse 1 for output_1: 0.8 more important from mse 2 for output_2
              metrics=["mean_absolute_error"])
# --------------------------------------Step 5: Callback ---------------------------------------------------
model_checkpoint_callback = keras.callbacks.ModelCheckpoint("model_cb_reg_housing.tf", save_best_only=True)
earlystopping_callback = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
tb_callback = keras.callbacks.TensorBoard("tb_log")
# ------------------------------- Step 5: Fit, evaluate & predictModel -------------------------------------
history = model.fit((X_train_s_1, X_train_s_2), (y_train1, y_train1), epochs=5, validation_data=((X_validation_s_1, X_validation_s_2),
                    (y_validation, y_validation)), callbacks=[model_checkpoint_callback, earlystopping_callback, tb_callback])
history.params
history.history
model.summary()
weights, bias = model.layers[1].get_weights()
# -------------------------------------------- Plot---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(history.history["loss"], label="train loss")
ax.plot(history.history["val_loss"], label="validation loss")
ax.legend()
plt.show()
# -------------------------------------------- Save model --------------------------------------------------
model.save("housing_reg_model_f.tf")
model_reg = keras.models.load_model("housing_reg_model_f.tf")
model_reg.summary()

# %load_ext tensorboard
# %tensorboard --logdir=./tb_log
"""
class MyCallback(keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        print("")
    def on_epoch_end(self, epoch, logs):  # After that epoch finished
        print(logs["val_loss"])


mycb = MyCallback()
"""