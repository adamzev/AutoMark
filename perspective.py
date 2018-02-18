import cv2
''' maps a perspective transformed image to a square '''

import numpy as np
import pickle

from helpers import Helpers
from cells import Cells
    
# code adpated from https://github.com/prajwalkr/SnapSudoku/blob/master/scripts/sudokuExtractor.py
class Transform(object):
    '''
        Stores and manipulates the input image to extract the Sudoku puzzle
        all the way to the cells
    '''

    def __init__(self, path, corners):
        self.helpers = Helpers()  # Image helpers
        self.image = self.loadImage(path)
        for corner in corners:
            cv2.circle(self.image, (corner[0], corner[1]), 5, (0, 255, 0), -1)
        #self.helpers.show(self.image, 'Original')
        image = self.straighten(self.image, corners)
        #self.helpers.show(image, 'Final image')

    def loadImage(self, path):
        color_img = cv2.imread(path)
        if color_img is None:
            raise IOError('Image not loaded')
        print('Image loaded.')
        return color_img

    def straighten(self, image, corners):
        print('Straightening image...')
        
        #Embed the quadrilateral vertices as (q_ij − q_00, 0) which translates one of the vertices to the origin
        temp_corners = []
        for corner in corners:
            print(corner, corners[0])
            temp_corners.append(corner - corners[0])

        corners = temp_corners

        print(corners)
        # eyepoint E (e_0, e_1, e_2) where e_2 != 0

        # map a point on the image r to a point p (x_0, x_1, 0) in the square

        # for all points r on the image?
        # p = E + t(r - E) for some t


        # t = e_2/(e_2 -r_2)

        # x_0 = (e_2 * r_0 - e_0 * r_2)/(e_2 - r_2)

        # x_1 = (e_2 * r_1 - e_1 * r_2)/(e_2 - r_2)
        print('done.')
        return image

if __name__ == '__main__':
    corners = np.array([[0,  55], [209, 0], [354, 230,], [54, 354]])
    ext = Transform('start.jpg', corners)
