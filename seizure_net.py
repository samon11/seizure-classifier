#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 17:13:17 2018

@author: michael
"""

from keras import models 
from keras import layers
import csv 
import numpy as np 


# http://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition 

csv_file = open('seizure_data.csv')
reader = csv.reader(csv_file)
data = np.array(list(reader))[1:,1:]
csv_file.close()

x = data[:,0:-2].astype(float) 
y = data[:,-1].astype(float)

one_hot_index = np.arange(-1885, 2047)
def one_hot_sequences(sequences, dimension = 3932):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        ix = np.isin(one_hot_index, sequence)
        results[i, ix] = 1
        
    return results
           
x = one_hot_sequences(x)


def seizured(labels):
    results = np.zeros((len(y))) 
    results[np.where(labels == 1)] = 1
    return results

y = np.apply_along_axis(seizured, 0, y) 


x_train = x[500:]
x_test = x[:500]
y_train = y[500:]
y_test = y[:500] 
 

model = models.Sequential()
model.add(layers.Dense(40, activation="relu", input_shape=(3932,)))
model.add(layers.Dense(40, activation="relu"))
model.add(layers.Dense(1,activation="sigmoid")) 

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, 
                    y_train, 
                    epochs=4, 
                    batch_size=500,
                    validation_data=(x_test, y_test)
                    )




history_dict = history.history

import matplotlib.pyplot as plt

loss_values = history_dict["loss"]
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

# Loss vs Epochs --------
plt.plot(epochs, loss_values, "bo", label="Training Loss")
plt.plot(epochs, val_loss_values, 'b', label="Validation Loss")
plt.title('Training and Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Accuracy vs Epochs
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, "bo", label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.title('Training and Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# results[0] = loss; results[1] = accuracy 
results = model.evaluate(x_test, y_test)


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))



















