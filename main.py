from collect import *
from activations import *
from network import *
import sys
import numpy as np

#Los tamanos de cada uno de los layer. Hay dos capas ocultas, una de entrada y una de salida
layers = [784,20,20,10]

#Learning rate
learning_rate = 0.5

#Se usa minibatch, porque se esta usando stohastic gradient descent
mini_batch_size = 16

#Cantida de epoch
epochs = 20

# Se cargan los datos de mnist.pkl.gz en el formato en el que vienen training, validation and testing data
training_data, validation_data, test_data = load_mnist()

#Inicia NN
nn = NeuralNetwork(layers, learning_rate, mini_batch_size, epochs)

#Entrenamiento NN
nn.fit(training_data, validation_data)

#Test NN
accuracy = nn.validate(test_data) / 100.0
print("Test Accuracy: " + str(accuracy) + "%")

#Guardar los valores para luego solo cargarlos con load()
nn.save()

