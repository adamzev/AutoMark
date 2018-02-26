import pickle

import cv2
import helpers as Helpers
import numpy as np
from digit import Digit
from pipeline import Pipeline


class Cells(object):
    '''
    Extracts each handwritten answer from the worksheet obtained
    from the Extractor
    '''

    def __init__(self, image, show_steps=False):
        self.image = image
        self.show_steps = show_steps

        # build the prepocessing pipleine
        pipeline = Pipeline([
            Helpers.convert_to_grayscale,
            lambda image: Helpers.blur(image, 5),
            Helpers.thresholdify,
            Helpers.ellipse_morph
        ])

        processed_image = pipeline.process_pipeline(self.image)

        self.cells = self.extract_cells(processed_image, self.image)

        cell_pipeline = Pipeline([
            Helpers.convert_to_grayscale,
            Helpers.thresholdify,
            lambda image: Helpers.ellipse_morph(image, 1)
        ])

        answer_text = {}

        for num, cell in enumerate(self.cells):
            # clean the cell
            gray = Helpers.convert_to_grayscale(cell)
            thresh = cell_pipeline.process_pipeline(cell)

            corners = self.find_corners_inside_largest_contour(thresh, gray)

            warp = Helpers.warp_perspective(corners, gray, size="same", verbose=False)
            #Helpers.show(warp, "clean cell")

            # find the contour of the text
            warp = Helpers.dilate(warp, 1)
            blur = Helpers.blur(warp, 3)

            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

            contours = Helpers.find_contours(255-thresh, mode_type="external", min_area=20)

            contours = Helpers.sort_contours(contours, method="left-to_right")

            rects = []
            for contour in contours:
                rect = Helpers.cut_out_rect(warp, contour)
                rects.append(rect)
            answer_text[num] = rects

        if show_steps:
            Helpers.show(self.image, "start")
            Helpers.show(processed_image, "processed")

        self.answer_text = answer_text


    def extract_cells(self, processed, display_image, count=14):
        page_area = Helpers.area(processed)
        min_percent = 0.0012
        max_percent = 0.0020
        contours = Helpers.get_sorted_contours(processed, sort_by="area")

        contours = Helpers.filter_contours(contours, sides=[4], min_area=page_area*min_percent, max_area=page_area*max_percent)

        # get just the count biggest contours
        contours = contours[:count]

        rects = []
        for contour in contours:
            rects.append(Helpers.cut_out_rect(display_image, contour))

        sorted_rects, sorted_contours = self.sort_by_problem_num(rects, contours)

        self.sorted_contours = sorted_contours

        if self.show_steps:
            for i, rect in enumerate(sorted_rects):
                Helpers.show(rect, str(i+1))

        return sorted_rects

    
    def sort_by_problem_num(self, rects, contours):
        n = 0
        centers = []
        for contour in contours:
            centers.append(Helpers.get_centers_of_contour(contour))

        # sort by the y value of the centers
        rects_centers = [(r, c, contour) for r, c, contour in sorted(zip(rects, centers, contours), key=lambda pair: pair[1][1])]

        # sort pairs by the x value
        sorted_rects = []
        sorted_contours = []


        # index 0 is the rect, index 1 is the center, index 2 is the contour
        for i in range(0, len(rects), 2):
            # if there is only one rect left, append it
            if i+1 > len(rects_centers):
                sorted_rects.append(rects_centers[i][0])
                sorted_contours.append(rects_centers[i][2])

            elif rects_centers[i][1] < rects_centers[i+1][1]:
                sorted_rects.extend([rects_centers[i][0], rects_centers[i+1][0]])
                sorted_contours.extend([rects_centers[i][2], rects_centers[i+1][2]])
            else:
                sorted_rects.extend([rects_centers[i+1][0], rects_centers[i][0]])
                sorted_contours.extend([rects_centers[i+1][2], rects_centers[i][2]])
        return sorted_rects, sorted_contours

    def save_files(self, rects):
        for i, rect in enumerate(rects):
            Helpers.save_image("rect{}.png".format(str(i+1)), rect)



    def find_corners_inside_largest_contour(self, thresh, gray):

        # find the largest contour
        contour = Helpers.largest4SideContour(thresh)
        # find its center
        centerX, centerY = Helpers.get_centers_of_contour(contour)
        app = Helpers.approx(contour)

        # find its corners
        corners = Helpers.get_rectangle_corners(app)

        # find the average color of the image
        avg_color_per_row = np.average(gray, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        
        # for each corner, move the corner's coordinates towards the center until you find a 
        # white spot (a spot where the color is less than the average color)
        for i in range(len(corners)):
            cX, cY = corners[i]
            cX, cY = int(cX), int(cY)
            # while we are on black, step towards the center
            n = 0

            while gray[cY][cX] < avg_color:
                n += 1
                diff = centerX-cX
                step = diff // abs(diff)
                cX += step

                diff = centerY-cY
                step = diff // abs(diff)
                cY += step
            corners[i] = [cX, cY]

        # return the new found corners
        return corners



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
    image = Helpers.loadImage('images/processed_org2.png')
    ext = Cells(image, True)
