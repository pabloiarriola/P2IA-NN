import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from os import listdir
from os.path import splitext

def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.2989,0.5870,0.1140])

target_directory= '.'
target = '.jpg'

for file in listdir(target_directory):
    filename, extension = splitext(file)
    try:
        if extension not in ['.py',target]:
            img=mpimg.imread(filename+extension)
            gray = rgb2gray(img)
            final = gray/255
            if final.shape[0] > 28 or final.shape[1] >28:
                print("Problema tamano")
                print(final.shape)
            if final.ndim == 3:
                print("Problema forma")
                print (final.shape)
    except OSError:
        print('Problema %s' %file)

