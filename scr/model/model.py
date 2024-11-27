import tensorflow as tf
import keras
model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, activation='relu', input_shape=(3,)),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(8, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
