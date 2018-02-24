import helpers as Helpers
from cells import Cells
from extractor import Extractor
from ocr import OCR


ext = Extractor('images/BCBA8F9752.jpg', False)

final = ext.final
#Helpers.show(final)

cells = Cells(final)
answers = cells.answer_text

for problem_number in range(len(answers)):
    images = answers[problem_number]
    for image in images:
        ocr = OCR(image)
        #value = ocr.value
        #Helpers.show(number, "prob {}, char {}, value {}".format(answer + 1, char + 1, value))
