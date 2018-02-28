from ocr import OCR
from keras.models import load_model

import helpers as Helpers

class Grader(object):
    answer_keys = {
        "BCBA8F9752" : [165, 123, 52, 130, 99, 87, 138, 104, 152, 90, 122, 150, 103, \
                69, 130, 63, 29, 98, 133, 133, 62, 187, 138, 91, 124, 109, 68, 113, \
                74, 108, 124, 55, 119, 102, 72, 17, 91, 25, 46, 120]
    }

    def __init__(self, form_code):
        self.answer_key = self.answer_keys[form_code]
        self.model = model = load_model('../data/cnn_mnist.h5')
        self.marks = []

    def grade(self, student_responses):
        for problem_number, segmented_digits in student_responses.items():
            students_answer = ""
            confident = True
            for digit_image in segmented_digits:
                value, confidence = OCR(digit_image, self.model)
                students_answer += str(value)
                if confidence < 0.97:
                    confident = False
            correct_ans = self.answer_key[problem_number]
            if int(students_answer) == correct_ans:
                self.marks.append("correct")
            elif not confident:
                self.marks.append("unknown")
            else:
                self.marks.append("wrong")

    def display(self, worksheet, sorted_contours):
        contours = sorted_contours
        for i, contour in enumerate(contours):
            app = Helpers.approx(contour, 0.05)
            rect = Helpers.get_rectangle_corners(app)

            font_face = Helpers.font("Hershey")
            font_scale = 1.5
            thickness = 3

            text_size = Helpers.text_size(str(self.answer_key[i]), font_face, font_scale, thickness)
            text_x = int(rect[0][0] - text_size[0][0]) - 20
            text_y = int(rect[3][1]) - 10

            if self.marks[i] == "correct":
                contour_color = Helpers.BGR_COLORS["GREEN"]
                text_color = Helpers.BGR_COLORS["DARK GREEN"]
            elif self.marks[i] == "unknown":
                contour_color = Helpers.BGR_COLORS["YELLOW"]
                text_color = Helpers.BGR_COLORS["BLACK"]
            else:
                contour_color = Helpers.BGR_COLORS["RED"]
                text_color = Helpers.BGR_COLORS["DARK RED"]

            Helpers.draw_text(worksheet, str(self.answer_key[i]), (text_x, text_y), font_face, font_scale, \
                        text_color, thickness)
            Helpers.draw_contour(worksheet, contour, contour_color)

        Helpers.show(worksheet, 'graded')
        return worksheet