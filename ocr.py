import cv2
import numpy as np
import pickle

from helpers import Helpers
from PIL import Image
import pytesseract

helpers = Helpers()
img = helpers.loadImage('processed_org.png')
reflect = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
print(img.shape)
#helpers.show(img * reflect)




print(pytesseract.image_to_string(img))