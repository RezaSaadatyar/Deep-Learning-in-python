# ============================================= Import Libraries ===========================================
from tensorflow import keras      # print(keras.__version__, tf.__version__)
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

input_1 = keras.layers.Input(shape=[6])        # Multi input single output (MISO); Shape for first layer: 6 column
input_2 = keras.layers.Input(shape=[4])        # Shape for second layer: 4 column
hidden_layer1 = keras.layers.Dense(50, activation="relu")(input_1)
hidden_layer2 = keras.layers.Dense(10, activation="relu")(hidden_layer1)
concatenate_layer = keras.layers.Concatenate()([input_2, hidden_layer2])
output = keras.layers.Dense(1)(concatenate_layer)
model = keras.Model(inputs=[input_1, input_2], outputs=[output])
# --------------------------------------- Step 4. Compile Model --------------------------------------------
model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["mean_absolute_error"])
# ------------------------------- Step 5: Fit, evaluate & predictModel -------------------------------------
history = model.fit((X_train_s_1, X_train_s_2), y_train1, epochs=30, validation_data=((X_validation_s_1, X_validation_s_2), y_validation))
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
