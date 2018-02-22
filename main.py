from extractor import Extractor
from cells import Cells
import helpers as Helpers

ext = Extractor('images/BCBA8F9752.jpg', False)

final = ext.final
#Helpers.show(final)

cells = Cells(final)
answers = cells.answer_text

for answer, numbers in answers.items():
    for char, number in enumerate(numbers):
        ocr = OCR(char)
        value = ocr.value
        Helpers.show(number, "prob {}, char {}, value {}".format(answer + 1, char + 1, value))

