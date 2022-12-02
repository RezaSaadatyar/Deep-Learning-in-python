# ============================================= Import Libraries ===========================================
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------- Step 1. Load data ------------------------------------------------
"""
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
images = x_train[1:3, :, :, :]/255.0
x_train.shape  # (50000, 32, 32, 3) number image:50000, rows:32, column:32, number channels:3
plt.imshow(x_train[2, :, :, :])
plt.show()
"""
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train.shape            # (60000, 28, 28)
# --------------------------------------- Step 2. Normalize data -------------------------------------------
x_train, x_test = x_train / 255.0, x_test / 255.0  # pixel values between 0 and 255; Scale these values to a range of 0 to 1 before feeding them to the neural network model.
# ----------------------------------- Step 3. Creating a Conv2D model --------------------------------------
model = keras.models.Sequential([
    keras.layers.Conv2D(50, 5, activation="relu", padding="same", input_shape=[28, 28, 1]),  # filters:50; kernel_size:5
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(100, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(200, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10, activation="softmax")
])
# ------------------------------------------ Step 4. Compile model -----------------------------------------
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# -------------------------------------------- Step 5.Fit model --------------------------------------------
history = model.fit(x_train, y_train, epochs=2, validation_split=0.15)
