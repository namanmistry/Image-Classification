import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
(X_train, y_train),(x_test, y_test) = datasets.cifar10.load_data()

y_train = y_train.reshape(-1,)


classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def plot_img(y_train,index):
    plt.figure(figsize=(15,2))
    plt.imshow(X_train[index])
    plt.xlabel(classes[y_train[index]])


X_train = X_train / 255
x_test = x_test / 255

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3,3),activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters=64, kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10,activation='softmax')
])

cnn.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=["accuracy"])

cnn.fit(X_train, y_train, epochs=10)

cnn.evaluate(x_test, y_test)

cnn.save("model.h5")