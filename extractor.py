import cv2

import helpers as Helpers
from pipeline import Pipeline


class Extractor(object):
    '''
        Stores and manipulates the input image to extract the Sudoku puzzle
        all the way to the cells
    '''

    def __init__(self, path, show_steps=False, save=False):
        self.image = Helpers.load_image(path)

        # build the prepocessing pipleine
        pipeline = Pipeline([
            Helpers.convert_to_grayscale,
            lambda image: Helpers.blur(image, 5),
            Helpers.thresholdify,
            Helpers.ellipse_morph
        ])

        processed_image = pipeline.process_pipeline(self.image)

        # get the contour, crop it out, find the corners and straighten
        contour = Helpers.largest_contour(processed_image)
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

        if save:
            Helpers.save_image('images/processed_org2.png', self.gray2)

        return None
if __name__ == '__main__':
    ext = Extractor('images/BCBA8F9752.jpg', True)
