import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
import keras
import numpy as np
import matplotlib.pyplot as plt

cifar = tf.keras.datasets.cifar10
print(cifar)

mnist = tf.keras.datasets.mnist
data = mnist.load_data()

(x_train, y_train), (x_test, y_test) = data

x_train = x_train / 255.0
x_test = x_test / 255.0

models = Sequential()
models.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(28,28,1),activation='relu'))
models.add(MaxPooling2D(pool_size=(2,2)))
models.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
models.add(MaxPooling2D(pool_size=(2,2)))
models.add(Flatten())
models.add(Dense(100,activation='relu'))
models.add(Dense(10, activation='softmax'))

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

opt = keras.optimizers.Adam(learning_rate=0.01)
models.compile(loss='categorical_crossentropy', metrics=["accuracy"], optimizer= opt)
models.summary()

result = models.fit(x_train, y_train, validation_split= 0.1, epochs=2, batch_size=30)
pre = models.predict(x_train)

print(pre)

loss = result.history['loss']
val_loss = result.history['val_loss']

acc = result.history['accuracy']
val_acc = result.history['val_accuracy']

plt.plot(acc, label = 'Accuracy')
plt.plot(val_acc, label = 'Validation Accuracy')

plt.plot(loss, label = 'loss')
plt.plot(val_loss, label = 'Validation loss')
plt.legend()
plt.show()

