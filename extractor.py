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

    def __init__(self, path, show_steps):
        self.helpers = Helpers()  # Image helpers
        self.image = self.helpers.loadImage(path)
        
        self.gray, self.thresh, self.morph = self.preprocess(self.image)        
        self.cropped_morph, self.cropped_tresh, self.cropped_gray, self.cropped = \
            self.crop_largest_contour(self.morph, [self.thresh, self.gray, self.image])

        self.warp_morph, self.warp_thresh, self.warp_gray, self.warp = \
            self.straighten(self.cropped_morph, [self.cropped_tresh, self.cropped_gray, self.cropped])

        if show_steps:
            self.helpers.show(self.image, 'Original')
            self.helpers.show(self.thresh, 'After thresholdify')
            self.helpers.show(self.morph, 'With Morph')
            self.helpers.show(self.cropped, 'After Cropping out image')
            self.helpers.show(self.warp, 'Final image')


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
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if cv2.contourArea(cnt) < min_size or n > count:
                break
            im = display_image.copy()
            cv2.drawContours(im, [cnt], -1, (0,255,120), 5)
            rect = self.helpers.cut_out_rect(display_image, cnt)
            rects.append((rect, cX, cY))

        rects.sort(key=lambda rect: rect[2])
        sorted_rects = []
        if count % 2 == 1:
            raise ValueError("Only even currently supported")
        for i in range(0, count, 2):
            if rects[i][1] < rects[i+1][1]:
                sorted_rects.extend([rects[i][0], rects[i+1][0]])
            else:
                sorted_rects.extend([rects[i+1][0], rects[i][0]])

        for i, rect in enumerate(sorted_rects):
            self.helpers.show(rect, str(i+1))
            #cv2.imwrite("rect{}.png".format(str(i+1)), rect)
            gray, thresh, morph = self.preprocess(rect)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            rect_d = cv2.dilate(rect, kernel, iterations=3)
            im2, contours, h = cv2.findContours(
                morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            im1, im2 = self.straighten(morph, [rect])
            self.helpers.show(im1, str(i+1))
            self.helpers.show(im2, str(i+1))
            #self.helpers.mask(rect, contours[0])
            #for cnt in contours:
                #im = rect_d.copy()
                #cv2.drawContours(im, [cnt], -1, (0,255,120), 5)
                #self.helpers.show(im, 'rect contour')


        return None
if __name__ == '__main__':
    ext = Extractor('BCBA8F9752.jpg', False)
