import cv2
import numpy as np
import pickle

import helpers as Helpers
#from cells import Cells
from pipeline import Pipeline
import pytesseract
    
# code adpated from https://github.com/prajwalkr/SnapSudoku/blob/master/scripts/sudokuExtractor.py
class Extractor(object):
    '''
        Stores and manipulates the input image to extract the Sudoku puzzle
        all the way to the cells
    '''

    def __init__(self, path, show_steps=False):
        self.image = Helpers.loadImage(path)
        
        # build the prepocessing pipleine
        pipeline = Pipeline([
            lambda image: Helpers.convert_to_grayscale(image),
            lambda image: Helpers.blur(image, 5),
            lambda image: Helpers.thresholdify(image),
            lambda image: Helpers.ellipse_morph(image)
        ])

        processed_image = pipeline.process_pipeline(self.image)

        # get the contour, crop it out, find the corners and straighten
        contour = Helpers.largestContour(processed_image)
        processed_image_cropped = Helpers.cut_out_rect(processed_image, contour)
        corners = Helpers.get_corners(processed_image_cropped)
        
        # apply the same cropping and warping to the original image
        image_cropped = Helpers.cut_out_rect(self.image, contour)
        straigtened_image = Helpers.warp_perspective(corners, image_cropped)

        if show_steps:
            Helpers.show(processed_image, 'Preprocessing')
            Helpers.show(processed_image_cropped, 'Processed image cropped')
            Helpers.show(image_cropped, 'Original image cropped')
            Helpers.show(straigtened_image, 'Final image')

        
        self.final = straigtened_image

        #processed_straight_image = self.process_pipeline(straigtened_image, preprocess_pipeline)
        
        #self.get_rectangles(processed_straight_image, straigtened_image, count=14)
        '''self.gray2, self.thresh2, self.morph2 = self.preprocess(self.warp)
        Helpers.show(self.gray2, 'post processed gray')
        Helpers.show(self.thresh2, 'post processed tresh')
        Helpers.show(self.morph2, 'post processed morph')
        #cv2.imwrite('processed_org2.png', self.gray2)
        '''




        return None
if __name__ == '__main__':
    ext = Extractor('images/BCBA8F9752.jpg', True)
