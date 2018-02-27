import sys

from cells import Cells
from extractor import Extractor
from qr import reader
from grader import Grader

def main(image_path):
    ext = Extractor(image_path, False)
    # get the final worksheet extracted from the image

    final = ext.final
    decoded_qr_code = reader(final)

    cells = Cells(final)

    grader = Grader(decoded_qr_code)

    grader.grade(cells.student_responses)

    grader.display(final, cells.sorted_contours)

if __name__ == '__main__':
    try:
        main(image_path=sys.argv[1])
    except IndexError:
        message = 'usage: {} image_path'
        print(message.format(__file__.split('/')[-1]))
