import cv2
import numpy as np
import pickle

from helpers import Helpers
from cells import Cells
    
# code adpated from https://github.com/prajwalkr/SnapSudoku/blob/master/scripts/sudokuExtractor.py
class Extractor(object):
    '''
        Stores and manipulates the input image to extract the Sudoku puzzle
        all the way to the cells
    '''

    def __init__(self, path):
        self.helpers = Helpers()  # Image helpers
        self.image = self.helpers.loadImage(path)
        self.helpers.show(self.image, 'Original')
        self.preprocess()
        self.helpers.show(self.image, 'After Preprocessing')
        sudoku = self.cropSudoku()
        self.helpers.show(sudoku, 'After Cropping out grid')
        sudoku = self.straighten(sudoku)
        self.helpers.show(255- sudoku, 'Final Sudoku grid')
        cv2.imwrite('processed.png', 255 - sudoku)

    def preprocess(self):
        print('Preprocessing...')
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.helpers.show(self.image, 'After Preprocessing')
        self.image = self.helpers.thresholdify(self.image)
        self.helpers.show(self.image, 'After Preprocessing')

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
        print('done.')

    def cropSudoku(self):
        print('Cropping out Sudoku...')
        contour = self.helpers.largestContour(self.image.copy())
        sudoku = self.helpers.cut_out_sudoku_puzzle(self.image.copy(), contour)
        print('done.')
        return sudoku

    def straighten(self, sudoku):
        print('Straightening image...')
        largest = self.helpers.largest4SideContour(sudoku.copy())
        app = self.helpers.approx(largest)
        corners = self.helpers.get_rectangle_corners(app)
        print(corners)
        sudoku = self.helpers.warp_perspective(corners, sudoku)
        print('done.')
        return sudoku

if __name__ == '__main__':
    ext = Extractor('BCBA8F9752.jpg')
