import sys

from cells import Cells
from extractor import Extractor
from qr import reader
from grader import Grader
import helpers as Helpers

def main(image_path):
    # get the final worksheet from the image
    ext = Extractor(image_path, False)
    final = ext.final

    # get the form code by checking the image's QR code
    decoded_qr_code = reader(final)

    # extract the cells and student's responses
    cells = Cells(final)

    # grade the worksheet by using a CNN to OCR the student's responses
    grader = Grader(decoded_qr_code)
    grader.grade(cells.student_responses)
    worksheet = grader.display(final, cells.sorted_contours)
    Helpers.save_image(f'{Helpers.IMAGE_DIRECTORY}/graded.png', worksheet)

if __name__ == '__main__':
    try:
        main(image_path=sys.argv[1])
    except IndexError:
        message = 'usage: {} image_path'
        print(message.format(__file__.split('/')[-1]))
