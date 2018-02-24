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
        
        
        thresh = cv2.threshold(image, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        #Helpers.show(thresh)
        im = Helpers.resize_and_fill(255-thresh, 28)
        #cv2.imshow("Resulting Image with Rectangular ROIs", im)
        print(im.shape)
        images = im.reshape(im.shape[0], 28, 28, 1)
        images = images.astype('float32')
        
        images/=255

        model = models.create_model()
        model.load_weights('cnn_mnist.h5')
        print(model.predict([im], batch_size=1, verbose=0))
        cv2.waitKey()



#print(pytesseract.image_to_string(img))
