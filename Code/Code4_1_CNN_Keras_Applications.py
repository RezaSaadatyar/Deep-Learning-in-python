# ============================================= Import Libraries ===========================================
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# --------------------------------------- Step 1. Load data ------------------------------------------------
ship = cv2.imread("Cruiseships.PNG")
ship.shape                                        # (500, 800, 3)
plt.imshow(ship)
plt.show()
# --------------------------------------- Step 2. Model CNN ------------------------------------------------
xception = keras.applications.Xception()          # (None, 299, 299, 3)
ship = ship.reshape(1, 500, 800, 3)/255.0
ship = tf.image.resize(ship, [299, 299])
y_prob = xception.predict(ship)
print(keras.applications.xception.decode_predictions(y_prob, top=5))


