import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

import sys, getopt
import logging

logger = logging.getLogger(__name__)
MIN_PLATE_NUM_LEN = 6

def show_img(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def find_contours(img):
        # Find contour step - rectangular shape of license plate
    c_points = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Return as tree, simplify line points
    contours = imutils.grab_contours(c_points)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] # sort, return first 10 contours
    return contours

def pre_process_image(img):
    # Greyscale step
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection step
    b_filter = cv2.bilateralFilter(grey_img, 11, 17, 17) # nosie reduction
    edge = cv2.Canny(b_filter, 30, 200) # edge detection
    return edge, grey_img, find_contours(edge)

def crop_image(img, mask):
    (x, y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    return img[x1:x2+1, y1:y2+1]

def maybe_find_license_plate_text(raw_img, grey_img, contours):
    # find rectangles
    locations = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True) # approx polygon in contour
        if len(approx) == 4:
            locations.append(approx) # found a rectangle location

    # print(locations)
    for location in locations:
        mask = np.zeros(grey_img.shape, np.uint8)

        new_img = cv2.drawContours(mask, [location], 0, 255, -1)
        new_img = cv2.bitwise_and(raw_img, raw_img, mask=mask)

        cropped_img = crop_image(grey_img, mask)

        # instantiate easyocr reader with english language
        reader = easyocr.Reader(['en'])
        # pass cropped image to reader
        result = reader.readtext(cropped_img)

        try:
            text = result[-1][-2]
            if len(text) >= MIN_PLATE_NUM_LEN:
                # valid license plate text found
                return text, location
            else:
                raise IndexError()
        except IndexError:
            pass
    raise Exception("No license plate found :(")

def draw_res(raw_img, text, location):
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(raw_img, text=text, org=(location[0][0][0], location[1][0][1]+100), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(raw_img, tuple(location[0][0]), tuple(location[2][0]), (0, 255,0), 3)
    return res

def main(argv):
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["input="])
    except getopt.GetoptError:
        print('main.py -i <inputfile>')
        sys.exit(2)
    if len(opts) == 0:
        print('main.py -i <inputfile>')
        sys.exit()

    for opt, arg in opts:
        if opt == '-h':
            print('main.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--input"):
            inputfile = arg
        else:
            print('main.py -i <inputfile>')
            sys.exit()

    # Import image
    raw_img = cv2.imread(inputfile)

    # Get preprocessed images, contours
    grey_img, edge_img, contours = pre_process_image(raw_img)

    # Try to get license plate and text
    try:
        text, location = maybe_find_license_plate_text(raw_img, grey_img, contours)
    except Exception as error:
        print(error)
        quit()

    show_img(draw_res(raw_img, text, location))

if __name__ == "__main__":
   main(sys.argv[1:])