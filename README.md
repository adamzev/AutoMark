AutoMark
===================

Take a picture of a "Grade-it" math worksheet and AutoMark will automatically grade it!

This project is intended as a proof-of-concept (that automatically grading math worksheets is feasable) prior to creating a website that generates worksheets and an app that grades them.

![Final result](https://www.tiny-robot.com/static/img/graded_small2.png "Final Result")

AutoMark reads a "Grade-it" QR code/form code and uses that to get the associated answer key. It then reads students' handwritten responses. It displays a color coded worksheet showing which problems the student got right and wrong.

*Adapted from [SnapSuduku](https://github.com/prajwalkr/SnapSudoku/) by [Prajwal](https://github.com/prajwalkr)*


----------
 TODO:
---------
 - Tests AutoMark with a larger variety of images from different cameras.
 - Improve digit string segmentation (watershed?). 
 - Implement a form code database.

Related Projects:
---
[Grade-it](https://github.com/tutordelphia/grade-it): A math worksheet generator

Prerequisites:
---

- Python 3
    - Download from [here](https://www.python.org/downloads/)

- OpenCV 3.4.0
    - `sudo apt-get install python-opencv` (preferred)
    - Install OpenCV from [here](http://opencv.org/downloads.html) 

- Numpy
    - `pip install numpy` (preferred)
    - You can build it from source [here](https://github.com/numpy/numpy)

- Other modules:
    - Details to be added regard PyZBar, Keras, H5PY, etc

Running AutoMark: 
---
    git clone https://github.com/tutordelphia/AutoMark.git
    cd AutoMark
    python main.py <path-to-input-image>

An Example Input:
---
> Here's a image of a "Grade-it" worksheet from a smartphone:

![Input Worksheet Image](https://www.tiny-robot.com/static/img/worksheet_start.jpg "Input image")
</br>

> The current code gives this output: 
![Final result](https://www.tiny-robot.com/static/img/graded_small2.png "Final Result")

The student filled in 9 problems correctly and 5 problems incorrectly.

The program correctly marked 12 problems (8 correct in green and 4 incorrect in red).

The program marked 2 problems as unknown (with a yellow box and black text). It was unable to read the student's handwriting on one problem and unable to segment the connected digits on problem #2. 

Algorithm
---

> 1. Basic image preprocessing - **Thresholding**.
> 2. Crop out the worksheet.
> 3. **Warp the perspective** of the worksheet (to make it rectangular).
> 4. Read the form code from the QR Code (using PyZBar)
> 5. Get the boxes where the student entered their handwritten answers (filtering contours). 
> 6. Segment the handwritten digit string (this needs to be improved, currently contours are used which don't work when digits touch)
> 7. Transform the segmented digits to MNIST's standards
> 8. Look up the answer key from the form code.
> 9. Predict the value of the handwritten digits using a pretrained model from [Keras's](https://github.com/keras-team/) example [Convolutional Neural Network for MNIST](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py). 
> 10. Display the graded result on the worksheet.

A Detailed Step-by-Step Example:
---

Two images are shown at some steps. The various steps perform best with different preprocessing so rather than keeping one preprocessed image, the original image is also cropped and kept.

> After Preprocessing:

![Preprocessing](https://www.tiny-robot.com/static/img/AutoMark/ext1.png "Preprocessing")

> Crop out the largest contour:

![Cropped worksheet thresh](https://www.tiny-robot.com/static/img/AutoMark/ext2.png "Cropped worksheet thresh")
![Cropped worksheet](https://www.tiny-robot.com/static/img/AutoMark/ext3.png "Cropped worksheet")

> Warp perspective to a letter sized rectangle:

![Warped](https://www.tiny-robot.com/static/img/AutoMark/ext4.png "Warped")
> Crop out the cells:

![Cell 1](https://www.tiny-robot.com/static/img/AutoMark/cell1.png "Cell 1")
![Cell 3](https://www.tiny-robot.com/static/img/AutoMark/cell3.png "Cell 3")

> Remove the borders:

![Without borders 1](https://www.tiny-robot.com/static/img/AutoMark/student1.png "Without borders 1")
![Without borders 2](https://www.tiny-robot.com/static/img/AutoMark/student2.png "Without borders 2")

> Apply thresholding and use contours to segment digit strings

![Thresh digits](https://www.tiny-robot.com/static/img/AutoMark/student_thresh3.png "Thresh digits")
![Thresh digits 2](https://www.tiny-robot.com/static/img/AutoMark/student_thresh2.png "Thresh digits 2")
<br>

> Note this process will not work when digits touch (like the 2 and 3 below):

![Touching digits](https://www.tiny-robot.com/static/img/AutoMark/student_thresh1.png "Final Result")


> Segmented the digits:


![Segmented 1](https://www.tiny-robot.com/static/img/AutoMark/ocr1.png "Segmented 1")
<br>
![Segmented 2](https://www.tiny-robot.com/static/img/AutoMark/ocr8.png "Segmented 2")


>  Format the digits to match MNIST specs:

![MNIST 1](https://www.tiny-robot.com/static/img/AutoMark/ocr_1_2.png "MNIST 1")
![MNIST 2](https://www.tiny-robot.com/static/img/AutoMark/ocr_1_8.png "MNIST 2")

> Final Result:
![Final result](https://www.tiny-robot.com/static/img/graded_small2.png "Final Result")

----------
Acknowledgements
---
Thank you to the creators and contributors of [SnapSuduku](https://github.com/prajwalkr/SnapSudoku/)

Thank you to [octagonaltree](https://www.reddit.com/user/octagonaltree) for the mentorship and guidance.
