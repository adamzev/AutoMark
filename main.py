from cells import Cells
from extractor import Extractor
from qr import reader
from grader import Grader

def main():
    ext = Extractor('images/IMG_20180225_152706090.jpg', False)
    # get the final worksheet extracted from the image

    final = ext.final
    decoded_qr_code = reader(final)

    cells = Cells(final)

    grader = Grader(decoded_qr_code)

    grader.grade(cells.student_responses)

    grader.display(final, cells.sorted_contours)

main()
