# ============================================= Import Libraries ===========================================
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# --------------------------------------- Step 1. Load data ------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train.shape                                     # (50000, 32, 32, 3)
# --------------------------------------- Step 2. Normalize data -------------------------------------------
x_train = x_train[:1000, ]/255.0                  # (1000, 32, 32, 3)
y_train = y_train[:1000, ]
# ----------------------------------- Step 3. Creating a Conv2D model --------------------------------------
xception1 = keras.applications.Xception(include_top=False)           # Change last layer ----> include_top=False
avg_layer = keras.layers.GlobalAveragePooling2D()(xception1.output)
output = keras.layers.Dense(10, activation="softmax")(avg_layer)    # 10 class -----> activation="softmax"
model = keras.Model(inputs=xception1.input, outputs=output)
for layer in xception1.layers:
  layer.trainable = False
model.summary()
# -------------------------------------------- Step 4. Compile ---------------------------------------------
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# -------------------------------------------- Step 5. Fit -------------------------------------------------
model.fit(x_train, y_train, epochs=20, validation_split=0.15)
