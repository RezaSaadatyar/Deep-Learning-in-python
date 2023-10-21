# ============================================= Import Libraries ===========================================
import numpy as np
from tensorflow import keras       # print(keras.__version__, tf.__version__)
import matplotlib.pyplot as plt
# ======================================== Classification ==================================================
# --------------------------------------- Step 1. Load data ------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train.shape  # (60000, 28, 28) = (Number of images, 28 , 28)
x_test.shape   # (10000, 28, 28) = (Number of images, 28 , 28)
y_train.shape  # (60000,) = (Number of label)
x_train[0, ]   # (28, 28) = Ankle boot
y_train[:10]   # array([9, 0, 0, 3, 0, 2, 7, 2, 5, 5], dtype=uint8)
# --------------------------------------- Step 2. Normalize data -------------------------------------------
x_train, x_test = x_train/255.0, x_test/255.0  # pixel values between 0 and 255; Scale these values to a range of 0 to 1 before feeding them to the neural network model.
# ------------------------------------ Step 3. Creating a Sequential model ---------------------------------
"""
Method 1:
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28])) 
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(75, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
"""
# Method 2:
model = keras.models.Sequential([                 # Define Sequential model with 3 layers
    keras.layers.Flatten(input_shape=[28, 28]),   # Layer input : Input must be vectored
    keras.layers.Dense(100, activation="relu"),   # Layer 1: The first hidden layer
    keras.layers.Dense(75, activation="relu"),    # Layer 2: The second hidden layer
    keras.layers.Dense(10, activation="softmax")  # Layer output: 10 neuron = 10 class; Multi classification then activation is softmax
])
model.summary()
weights, bias = model.layers[1].get_weights()
# --------------------------------------- Step 4. Compile Model ------------------------------------------
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])  # Multi Classification: sparse_categorical_crossentropy
# ------------------------------- Step 5: Fit, evaluate & predictModel -----------------------------------
history = model.fit(x_train, y_train, epochs=10, validation_split=0.15)
history.params
history.history
model.evaluate(x_test, y_test, verbose=0)
x3 = x_test[1:4, :, :]                    # x3.shape: (3, 28, 28)
Label_predict = np.argmax(model.predict(x3).round(3), axis=1)
Label_test = y_test[1:4]
# -------------------------------------------- Plot------------------------------------------------------
plt.imshow(x_train[0, ], cmap="gray")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(history.history["loss"], label="train loss")
ax.plot(history.history["accuracy"], label="train accuracy")
ax.plot(history.history["val_loss"], label="validation loss")
ax.plot(history.history["val_accuracy"], label="validation accuracy")
ax.legend()
plt.show()


