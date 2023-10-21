# ============================================= Import Libraries ===========================================
import numpy as np
from tensorflow import keras  # print(keras.__version__, tf.__version__)
import matplotlib.pyplot as plt

# ======================================== Classification ==================================================
# --------------------------------------- Step 1. Load data ------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train.shape  # (60000, 28, 28) = (Number of images, 28 , 28)
x_test.shape  # (10000, 28, 28) = (Number of images, 28 , 28)
y_train.shape  # (60000,) = (Number of label)
x_train[0,]  # (28, 28) = Ankle boot
y_train[:10]  # array([9, 0, 0, 3, 0, 2, 7, 2, 5, 5], dtype=uint8)
# --------------------------------------- Step 2. Normalize data -------------------------------------------
x_train, x_test = x_train / 255.0, x_test / 255.0  # pixel values between 0 and 255; Scale these values to a range of 0 to 1 before feeding them to the neural network model.
# ---------------------- Step 3. Creating a Sequential model & Overfitting ---------------------------------
init_1 = keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')  # Initializer

# Method 2:
model = keras.models.Sequential([keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dropout(0.2),  # AlphaDropout with selu
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal",
                       kernel_regularizer=keras.regularizers.l2(0.01),      # L2 regularization; l2 greater reduce over_fitting
                       kernel_constraint=keras.constraints.max_norm(1.0)),  # max_norm smaller reduce over_fitting
    keras.layers.Dropout(0.2),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(50, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation="softmax")
])
# weights, bias = model.layers[1].get_weights()
model.summary()
# -------------------- Step 4. Compile Model, Fit, evaluate & predictModel ------------------------------
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history = model.fit(x_train, y_train, validation_split=0.15, epochs=10)

# Monte carlo
"""
# Training=True i.e., using a model trained by dropouts for test data, 50 network with dropout random for increase accuracy
# model_mcd.shape: (50, 10000, 10) --> 50:number network with prediction 10000*10, 10000: number image, 10: probability each class
model_mcd = np.stack([model(x_test, training=True) for _ in range(50)])
np.round(model.predict(x_test[:1]), 3)  # prediction for first image
np.round(model_mcd.mean(axis=0)[:1], 3)


class MonteCarloDropout(keras.layers.Dropout):      # If model has bach_normalization
    def call(self, inputs):
        return super().call(inputs, training=True)  # Using model that trained by bach_normalization 
"""
history.history.keys()
history.params
history.history
model.evaluate(x_test, y_test, verbose=0)
x3 = x_test[1:4, :, :]  # x3.shape: (3, 28, 28)
Label_predict = np.argmax(model.predict(x3).round(3), axis=1)
Label_test = y_test[1:4]
# -------------------------------------------- Plot------------------------------------------------------
plt.imshow(x_train[0,], cmap="gray")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(history.history["loss"], label="train loss")
ax.plot(history.history["accuracy"], label="train accuracy")
ax.plot(history.history["val_loss"], label="validation loss")
ax.plot(history.history["val_accuracy"], label="validation accuracy")
ax.legend()
plt.show()
