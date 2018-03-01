import cv2

import helpers as Helpers
from pipeline import Pipeline


class Extractor(object):
    '''
        Stores and manipulates the input image to extract the worksheet
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
            Helpers.save_image(f'{Helpers.IMAGE_DIRECTORY}/extractor_finish.png', self.final)

        return None
if __name__ == '__main__':
    # Try out this class seperately by using this example:
    # This code will run when the file is exectued directly but
    # it will be ignored when it is imported.
    ext = Extractor(f'{Helpers.EXAMPLE_DIRECTORY}/extractor_start.jpg', show_steps=True)
    # Helpers.save_image(f'{Helpers.EXAMPLE_DIRECTORY}/extractor_finish.png', ext.final)
