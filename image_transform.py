import cv2
import numpy as np
import pickle

from helpers import Helpers
from PIL import Image
import pytesseract

helpers = Helpers()
img = helpers.loadImage('ecc4.JPG')
print(img.shape)
helpers.show(img)
shape = img.shape
new_img = np.empty((shape[1], shape[0], shape[2]))

print(img[0][0][0])

reflect = np.array([[0, 1], [1, 0]])
#xy = np.array([4, 2])

#print(np.matmul(reflect, xy))

print(shape[0], "x", shape[1])
for x in range(shape[0]):
    for y in range(shape[1]):
        temp = np.array([x, y])
        #print(temp)
        reflected = np.dot(reflect, temp)
        #print(x, y, reflected)
        #print()
        new_img[reflected[0]][reflected[1]] = img[x, y]
print(new_img.shape)
helpers.show(new_img)




#print(pytesseract.image_to_string(img))