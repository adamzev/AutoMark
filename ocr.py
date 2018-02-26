import cv2

import helpers as Helpers

def OCR(image, model):
    '''The original black and white (bilevel) images from
    NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio.
    The resulting images contain grey levels as a result of the anti-aliasing technique used by the
    normalization algorithm. the images were centered in a 28x28 image by computing the center
    of mass of the pixels, and translating the image so as to position this point at the center of
    the 28x28 field.'''
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    img = Helpers.resize_and_fill(255-thresh, 28)

    images = img.reshape(1, 28, 28, 1)
    images = images.astype('float32')

    images /= 255

    result = model.predict(images, batch_size=1, verbose=0)
    result = list(result[0])

    sorted_result = sorted(result, reverse=True)
    confidence = sorted_result[0]
    value = result.index(sorted_result[0])
    return value, confidence
