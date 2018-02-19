import cv2
import numpy as np
import pickle

from helpers import Helpers
from cells import Cells
import pytesseract
    
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
        self.gray, self.thresh, self.morph = self.preprocess(self.image)
        self.helpers.show(self.thresh, 'After thresholdify')
        self.helpers.show(self.morph, 'With Morph')
        
        self.cropped_morph, self.cropped_tresh, self.cropped_gray, self.cropped = \
            self.crop_largest_contour(self.morph, [self.thresh, self.gray, self.image])
        self.helpers.show(self.cropped, 'After Cropping out image')
        self.warp_morph, self.warp_thresh, self.warp_gray, self.warp = \
            self.straighten(self.cropped_morph, [self.cropped_tresh, self.cropped_gray, self.cropped])
        self.helpers.show(self.warp, 'Final image')
        self.helpers.show(self.warp_morph, 'Warp Morph')
        self.helpers.show(self.warp_thresh, 'Warp Thresh')
        self.helpers.show(self.warp_gray, 'Warp Gray')


        self.get_rectangles(self.warp_morph, self.warp, count=14)
        '''self.gray2, self.thresh2, self.morph2 = self.preprocess(self.warp)
        self.helpers.show(self.gray2, 'post processed gray')
        self.helpers.show(self.thresh2, 'post processed tresh')
        self.helpers.show(self.morph2, 'post processed morph')
        #cv2.imwrite('processed_org2.png', self.gray2)
        '''
    def preprocess(self, image):
        print('Preprocessing...')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = self.helpers.thresholdify(blur)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return gray, thresh, morph
        print('done.')

    def crop_largest_contour(self, processed_image, images=None):
        # prepared image is a black and white image
        # images are optional additional images to undergo the same cropping
        print('Cropping out the image...')
        contour = self.helpers.largestContour(processed_image)
        images.insert(0, processed_image)
        result = []
        for image in images:
            result.append(self.helpers.cut_out_rect(image, contour))
        print('done.')
        return result

    def straighten(self, processed_image, images=None):
        print('Straightening image...')
        largest = self.helpers.largest4SideContour(processed_image)
        app = self.helpers.approx(largest)
        corners = self.helpers.get_rectangle_corners(app)
        # print(corners)
        result = []
        images.insert(0, processed_image)
        for image in images:
            result.append(self.helpers.warp_perspective(corners, image))
        print('done.')
        return result

    def get_rectangles(self, processed, display_image, count=14, min_size=4000, max_size=8000):
        im2, contours, h = cv2.findContours(
            processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if not display_image.any():
            display_image = processed

        n = 0
        rects = []
        for cnt in contours:
            if cv2.contourArea(cnt) > max_size or len(self.helpers.approx(cnt)) not in [4, 5]:
                continue
            n += 1
            print(n, cv2.contourArea(cnt))
            if cv2.contourArea(cnt) < min_size or n > count:
                break
            im = display_image.copy()
            cv2.drawContours(im, [cnt], -1, (0,255,120), 5)
            rect = self.helpers.cut_out_rect(display_image, cnt)
            rects.append(rect)
            ocr_num = pytesseract.image_to_string(rect)
            self.helpers.show(rect, '{}) {} | contour sides:{}, area:{}'.format(n, ocr_num, len(self.helpers.approx(cnt)), cv2.contourArea(cnt)))
        return None
if __name__ == '__main__':
    ext = Extractor('BCBA8F9752.jpg')
