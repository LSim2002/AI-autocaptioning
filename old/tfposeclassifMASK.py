# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:35:22 2023

@author: LSIMON
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Your data preparation here
# X, y = ...

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))  # Adjust input_dim to match your data
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer: binary classification


##LE MASQUE EST CREE AUTOMATIQUEMENT, SUFFIT QUE LANGLE VAUT -1

# Custom loss function with masking
def masked_binary_crossentropy(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, -1))
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

model.compile(loss=masked_binary_crossentropy, optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')
