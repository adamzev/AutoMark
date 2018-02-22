import numpy as np
import cv2
import pickle

import helpers as Helpers
from digit import Digit
from pipeline import Pipeline

class Cells(object):
    '''
    Extracts each handwritten answer from the worksheet obtained
    from the Extractor
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

        self.cells = self.extractCells(processed_image, self.image)

        for cell in self.cells:
            print(cell.shape)
            gray = Helpers.convert_to_grayscale(cell)
            #blur = Helpers.blur(gray, 3)
            
            thresh = Helpers.thresholdify(gray)
            thresh = Helpers.ellipse_morph(thresh, 1)
            #cell = Helpers.ellipse_morph(cell)

            #corners = Helpers.get_corners(thresh)
            contour = Helpers.largest4SideContour(thresh, show=False, display_image=None)
            centerX, centerY = Helpers.get_centers_of_contour(contour)
            cv2.circle(cell, (centerX, centerY), 2, (0, 128, 255), -1)
            app = Helpers.approx(contour)
            corners = Helpers.get_rectangle_corners(app)
            avg_color_per_row = np.average(gray, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            
            for i in range(len(corners)):
                cX, cY = corners[i]
                cX, cY = int(cX), int(cY)
                # while we are on black, step towards the center
                n = 0
                print(gray[cY][cX])    
                while(gray[cY][cX] < avg_color):
                    n += 1
                    diff = centerX-cX
                    step = diff // abs(diff)

                    cX += step

                    diff = centerY-cY
                    step = diff // abs(diff)

                    cY += step
                corners[i] = [cX, cY]
                
                cv2.circle(cell, (cX, cY), 2, (0, 128, 255), -1)

            #cell = Helpers.dilate(cell, 3)
            #cell = Helpers.thresholdify(cell)
            
            warp = Helpers.warp_perspective(corners, gray, size="same", verbose=False)
            Helpers.show(warp, "clean cell")
            warp = Helpers.dilate(warp, 1)
            blur = Helpers.blur(warp, 3)
            
            #thresh = Helpers.thresholdify(warp)
            ret, thresh = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            Helpers.show(thresh, "clean cell")
            '''
            # Setup SimpleBlobDetector parameters.
            params = cv2.SimpleBlobDetector_Params()

            # Change thresholds
            #params.minThreshold = 10
            #params.maxThreshold = 200
            
            # Filter by Area.
            params.filterByArea = True
            params.minArea = 100
            
            # Filter by Circularity
            params.filterByCircularity = False
            params.minCircularity = 0.1
            
            # Filter by Convexity
            params.filterByConvexity = True
            params.minConvexity = 0.87
            
            # Filter by Inertia
            params.filterByInertia = True
            params.minInertiaRatio = 0.01

            # Create a detector with the parameters
            ver = (cv2.__version__).split('.')
            if int(ver[0]) < 3 :
                detector = cv2.SimpleBlobDetector(params)
            else : 
                detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(thresh)
            print(len(keypoints)) 
            im_with_keypoints = cv2.drawKeypoints(thresh, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            '''
            # Show blobs
            cv2.imshow("Keypoints", im_with_keypoints)

        if show_steps:
            Helpers.show(self.image, "start")
            Helpers.show(processed_image, "processed")


    def extractCells(self, processed, display_image, count=14, min_size=4000, max_size=8000):

        contours = Helpers.get_sorted_contours(processed, sort_by="area")
        contours = Helpers.filter_contours(contours, sides=[4, 5], min_area=4000, max_area=8000)

        # get just the count biggest contours
        contours = contours[:count]

        rects = []
        for contour in contours:
            rects.append(Helpers.cut_out_rect(display_image, contour))

        sorted_rects = self.sort_by_problem_num(rects, contours)
        #for i, rect in enumerate(sorted_rects):
            #Helpers.show(rect, str(i+1))
        return rects

    
    def sort_by_problem_num(self, rects, contours):
        n = 0
        centers = []
        for contour in contours:
            centers.append(Helpers.get_centers_of_contour(contour))

        # sort by the y value of the centers
        rects_centers = [(r, c) for r, c in sorted(zip(rects, centers), key=lambda pair: pair[1][1])]

        # sort pairs by the x value
        sorted_rects = []
        for i in range(0, len(rects), 2):
            # if there is only one rect left, append it
            if i+1 > len(rects_centers):
                sorted_rects.append(rects_centers[i])
            elif rects_centers[i][1] < rects_centers[i+1][1]:
                sorted_rects.extend([rects_centers[i][0], rects_centers[i+1][0]])
            else:
                sorted_rects.extend([rects_centers[i+1][0], rects_centers[i][0]])
        return sorted_rects
    
    def save_files(rects):
        for rect in enumerate(rects):
            Helpers.save_image("rect{}.png".format(str(i+1), rect))





    def clean(self, cell):
        contour = Helpers.largestContour(cell.copy())
        x, y, w, h = cv2.boundingRect(contour)
        cell = Helpers.make_it_square(cell[y:y + h, x:x + w], 28)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kernel)
        cell = 255 * (cell / 130)
        return cell

    def centerDigit(self, digit):
        digit = self.centerX(digit)
        digit = self.centerY(digit)
        return digit

    def centerX(self, digit):
        topLine = Helpers.getTopLine(digit)
        bottomLine = Helpers.getBottomLine(digit)
        if topLine is None or bottomLine is None:
            return digit
        centerLine = (topLine + bottomLine) >> 1
        imageCenter = digit.shape[0] >> 1
        digit = Helpers.rowShift(
            digit, start=topLine, end=bottomLine, length=imageCenter - centerLine)
        return digit

    def centerY(self, digit):
        leftLine = Helpers.getLeftLine(digit)
        rightLine = Helpers.getRightLine(digit)
        if leftLine is None or rightLine is None:
            return digit
        centerLine = (leftLine + rightLine) >> 1
        imageCenter = digit.shape[1] >> 1
        digit = Helpers.colShift(
            digit, start=leftLine, end=rightLine, length=imageCenter - centerLine)
        return digit

if __name__ == '__main__':
    ext = Cells('images/processed_org2.png', True)
