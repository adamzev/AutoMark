import cv2
from keras.models import load_model
from PIL import Image
from pyzbar.pyzbar import decode

import helpers as Helpers
from cells import Cells
from extractor import Extractor
from ocr import OCR

ext = Extractor('images/IMG_20180225_161951595.jpg', False)

final = ext.final

cv2_im = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
pil_im = Image.fromarray(cv2_im)


img = pil_im.convert('L')

result = decode(img)

decoded_qr_code = result[0].data
decoded_qr_code = decoded_qr_code.decode("utf-8") # convert from byte to string

cells = Cells(final)

answers = cells.answer_text

 
ANSWER_KEYS = { 
    "BCBA8F9752" : [165, 123, 52, 130, 99, 87, 138, 104, 152, 90, 122, 150, 103, \
            69, 130, 63, 29, 98, 133, 133, 62, 187, 138, 91, 124, 109, 68, 113, \
            74, 108, 124, 55, 119, 102, 72, 17, 91, 25, 46, 120]
}

answer_key = ANSWER_KEYS[decoded_qr_code]

model = load_model('cnn_mnist.h5')

marks = []

for problem_number in range(len(answers)):
    images = answers[problem_number]
    students_answer = ""
    confident = True
    for image in images:
        ocr = OCR(image, model)
        value = ocr.value
        students_answer += str(value)
        if ocr.confidence < 0.9:
            confident = False
    correct_ans = answer_key[problem_number]
    if int(students_answer) == correct_ans:
        marks.append("correct")
    elif not confident:
        marks.append("unknown")
    else:
        marks.append("wrong")

contours = cells.sorted_contours
for i, contour in enumerate(contours):
    app = Helpers.approx(contour, 0.05)
    rect = Helpers.get_rectangle_corners(app)

    fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    fontScale = 1.5
    thickness = 3

    text_size = cv2.getTextSize(str(answer_key[i]), fontFace, fontScale, thickness)
    text_x = int(rect[0][0] - text_size[0][0]) - 20
    text_y = int(rect[3][1]) - 10
    
    if marks[i] == "correct":
        contour_color = (0, 255, 120)
        text_color = (0, 60, 30)
    elif marks[i] == "unknown":
        contour_color = (66, 244, 220)
        text_color = (0, 0, 0)
    else: 
        contour_color = (0, 38, 232)
        text_color = (0, 7, 80)


    cv2.putText(final, str(answer_key[i]),(text_x, text_y), fontFace, fontScale, text_color, thickness, cv2.LINE_AA)
    cv2.drawContours(final, [contour], -1, contour_color, 5)
    
Helpers.show(final, 'graded')
Helpers.save_image('images/graded2.png', final)
