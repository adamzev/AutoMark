# http://hanzratech.in/2015/02/24/handwritten-digit-recognition-using-opencv-sklearn-and-python.html
# Import the modules
import cv2
import helpers as Helpers
import numpy as np
import pytesseract
from skimage.feature import hog
from sklearn.externals import joblib
import models

class OCR(object):
    def __init__(self, image):
        # Load the classifier
        
        '''The original black and white (bilevel) images from 
        NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. 
        The resulting images contain grey levels as a result of the anti-aliasing technique used by the 
        normalization algorithm. the images were centered in a 28x28 image by computing the center 
        of mass of the pixels, and translating the image so as to position this point at the center of 
        the 28x28 field.'''
        thresh = cv2.threshold(image, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        #Helpers.show(thresh)
        im = Helpers.resize_and_fill(255-thresh, 28)
        
        images = im.reshape(1, 28, 28, 1)
        images = images.astype('float32')
        
        images/=255

        model = models.create_model()
        model.load_weights('cnn_mnist.h5')
        result = model.predict(images, batch_size=1, verbose=0)
        result = list(result[0])
        for i, r in enumerate(result):
            if r > 0.01:
                print(i, round(r, 2))
        self.value = result.index(max(result))
        Helpers.show(im, str(self.value))



#print(pytesseract.image_to_string(img))
