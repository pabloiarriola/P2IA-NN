import numpy as np
from PIL import Image
from collect import *
from activations import *
from network import *
import sys
import numpy as np

layers = [784,30,30,30,30,10]
learning_rate = 0.01
mini_batch_size = 16
epochs = 30
training_data, validation_data, test_data = load_mnist()


nn = NeuralNetwork(layers, learning_rate, mini_batch_size, epochs)
nn.load('model.npz')


img = Image.open('E:\p\Prueba.png')
img.load()
img.convert("L")
data = np.asarray(img,dtype='int32')
final = np.reshape(data, (784, 1))


print(nn.predict(final))

