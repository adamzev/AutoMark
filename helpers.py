import math

import cv2
import numpy as np


'''
Image manipulation helper functions
code adpated from https://github.com/prajwalkr/SnapSudoku/
'''

def show(img, windowName='Image'):
    screen_res = 1280.0, 720.0
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, window_width, window_height)

    cv2.imshow(windowName, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def loadImage(path):
    print(path)
    color_img = cv2.imread(path)
    if color_img is None:
        raise IOError('Image not loaded')
    print('Image loaded.')
    return color_img

def isCv2():
    return cv2.__version__.startswith('2.')
        
def thresholdify(img):
    img = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    return 255 - img




def get_sorted_contours(processed, sort_by='area'):
    im2, contours, h = cv2.findContours(
        processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if sort_by == "area":
        return sorted(contours, key=cv2.contourArea, reverse=True)
    #elif sort_by == "left-to-right":
        #return contours.sort_contours(contours, method="left-to-right")[0]
    else:
        raise ValueError("Sort by {} is not implemented".format(sort_by))

def sort_contours(contours, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    index = 0 # 0 is x, 1 is y

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True


    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    centers = []
    for contour in contours:
        centers.append(get_centers_of_contour(contour))

    # sort by the y value of the centers
    sorted_contours = [contour for contour, center in sorted(zip(contours, centers), key=lambda pair: pair[1][index], reverse=reverse)]
 
    # return the list of sorted contours and bounding boxes
    return sorted_contours


def filter_contours(contours, sides=None, min_area=None, max_area=None):
    result = []
    for contour in contours:
        valid = True
        if sides:
            n_sides = len(approx(contour, 0.05))
            if n_sides not in sides:
                valid = False
        if min_area and cv2.contourArea(contour) < min_area:
            valid= False
        if max_area and cv2.contourArea(contour) > max_area:
            valid= False
        if valid:
            result.append(contour)
    return result

def save_image(full_path, image):
    cv2.imwrite(full_path, image)


def find_contours(image, mode_type="default", method_type="default", min_area=20):
    '''
        RETR_EXTERNAL: retrieves only the extreme outer contours. 

        RETR_LIST: retrieves all of the contours without establishing any hierarchical relationships.

        RETR_TREE: retrieves all of the contours and reconstructs a full hierarchy of nested contours.

        RETR_FLOODFILL 

        CHAIN_APPROX_SIMPLE  compresses horizontal, vertical, and diagonal segments and leaves only their end points. 

        CHAIN_APPROX_TC89_L1 applies one of the flavors of the Teh-Chin chain approximation algorithm [168]

        CHAIN_APPROX_TC89_KCOS 	applies one of the flavors of the Teh-Chin chain approximation algorithm [168]

    '''
    
    mode = cv2.RETR_TREE
    method = cv2.CHAIN_APPROX_SIMPLE

    if mode_type == "external":
        mode = cv2.RETR_EXTERNAL
    
    if method_type == "Teh-Chin":
        method = cv2.CHAIN_APPROX_TC89_L1
    
    if method_type == "Teh-Chin2":
        method = cv2.CHAIN_APPROX_TC89_KCOS
    
    contours = cv2.findContours(image, mode=mode, method=method)[1]

    return filter_contours(contours, min_area=min_area)
    


def largestContour(image):
    if isCv2():
        contours, h = cv2.findContours(
            image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, h = cv2.findContours(
            image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)

def largest4SideContour(processed, show_contours=False, display_image=None):
    im2, contours, h = cv2.findContours(
        processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:min(5,len(contours))]:
        if show_contours:
            im = processed.copy()
            im = cv2.cvtColor(im,  cv2.COLOR_GRAY2BGR)
            cv2.drawContours(im, [cnt], -1, (0,255,120), 5)
            show(im,'contour {}'.format(len(approx(cnt))))
        if len(approx(cnt)) == 4:
            return cnt
    return None

def show_all_contours(processed, display_image=False, min_size=100):
    im2, contours, h = cv2.findContours(
        processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if not display_image.any():
        display_image = processed
    for cnt in contours:
        if cv2.contourArea(cnt) < min_size:
            break
        im = display_image.copy()
        cv2.drawContours(im, [cnt], -1, (0,255,120), 5)
        show(im,'contour sides:{}, area:{}'.format(len(approx(cnt)), cv2.contourArea(cnt)))
    return None

def round_to_multiple(x, base=5):
    return int(base * round(float(x)/base))

def resize_and_fill(image, size, border_size=4):
    new_image = np.zeros((size, size))

    height, width = get_size(image)
    
    image_size = size - border_size * 2
    if height > width:
        target_height = image_size
        target_width = round_to_multiple(int(math.ceil(width / height * image_size)), 2)
        top = bottom = 0
        left = right = (image_size - target_width) // 2
    else:
        target_width = image_size
        target_height = round_to_multiple(int(math.ceil(height / width * image_size)), 2)
        top = bottom = (image_size - target_height) // 2
        left = right = 0

    # resize the image
    resized = cv2.resize(image, (target_width, target_height), interpolation = cv2.INTER_AREA)
    with_border = cv2.copyMakeBorder(resized, top, bottom, right, left, cv2.BORDER_CONSTANT,value=0)
    
    contour = largestContour(with_border)
    cX, cY = get_centers_of_contour(contour)
    x_shift = cX - image_size//2
    y_shift = cY - image_size//2
    if abs(x_shift) > border_size or abs(y_shift) > border_size:
        print(x_shift, y_shift)
        print("large shift")
        x_shift = min(x_shift, 4)
        x_shift = max(x_shift, -4)
        y_shift = min(y_shift, 4)
        y_shift = max(y_shift, -4)
        print(x_shift, y_shift)
    start_x = border_size - x_shift
    start_y = border_size - y_shift

    new_image[start_y:start_y + image_size, start_x:start_x + image_size] = with_border[0:image_size, 0:image_size]
    
    cX, cY = get_centers_of_contour(new_image)

    return new_image

def make_it_letter(image, side_length=306):
    return cv2.resize(image, (side_length, int(side_length * (11/8.5))))

def area(image):
    return float(image.shape[0] * image.shape[1])


def get_centers_of_contour(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def cut_out_rect(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    image = image[y:y + h, x:x + w]
    return image

def binarized(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j] = 255 * int(image[i][j] != 255)
    return image

def approx(cnt, perc_of_arc=0.01):
    ''' cnt: opencv contour
        perc_of_arc: see https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
    '''
    peri = cv2.arcLength(cnt, True)
    app = cv2.approxPolyDP(cnt, perc_of_arc * peri, True)
    return app

def get_corners(processed_image):
    ''' gets corners from the largest quadrlateral in an image '''
    largest = largestContour(processed_image)
    app = approx(largest)
    corners = get_rectangle_corners(app)
    return corners

def get_rectangle_corners(cnt):
    ''' gets corners from a contour '''
    
    pts = cnt.reshape(cnt.shape[0], 2)
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def blur(image, pixels):
    return cv2.GaussianBlur(image, (pixels, pixels), 0)

def get_size(image):
    return image.shape[0:2]

def warp_perspective(rect, grid, size="letter", verbose=False):
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    if size == "letter":
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = int(maxWidth * (11/8.5))
    
    if size == "same": 
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)

    if verbose:
        print(M)
    warp = cv2.warpPerspective(grid, M, (maxWidth, maxHeight))
    #return make_it_square(warp)
    return warp

def get_structuring_element(kernel_type="ellispe"):
    if kernel_type == "ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    else:
        raise ValueError("Unimplemented kernel type")

def dilate(image, iterations, kernel_type="ellipse"):
    kernel = get_structuring_element(kernel_type)
    return cv2.dilate(image, kernel, iterations=iterations)

def erode(image, iterations, kernel_type="ellipse"):
    kernel = get_structuring_element(kernel_type)
    return cv2.erode(image, kernel, iterations=iterations)

def ellipse_morph(image, size=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
