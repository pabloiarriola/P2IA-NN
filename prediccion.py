import numpy as np
from PIL import Image
from collect import *
from activations import *
from network import *
import sys
import numpy as np

img = Image.open('E:\p\Prueba.png').convert("L")
imgarr = np.array(img)

print (imgarr.shape)
print (imgarr.size)

network.load('model.npz')

#print (imgarr)
