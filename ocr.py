import cv2
import numpy as np

import helpers as Helpers
import pytesseract

# http://hanzratech.in/2015/02/24/handwritten-digit-recognition-using-opencv-sklearn-and-python.html
# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

class OCR(object):
    def __init__(self, number):
        # Load the classifier
        clf = joblib.load("digits_cls.pkl")


        '''
        height, width =Helpers.get_size(number)
        dst = cv.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)
            # Draw the rectangles
            #leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
            # Resize the image
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            # Calculate the HOG features
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
            cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        '''
        im = Helpers.resize_and_fill(number, 28)
        cv2.imshow("Resulting Image with Rectangular ROIs", im)
        cv2.waitKey()



#print(pytesseract.image_to_string(img))