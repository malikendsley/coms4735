import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import os 
import sys

# tuning handles

# if a contour's largest 4 defects are on average longer than this, it is a splayed hand
DEFECT_LENGTH_THRESHOLD = None

# if a non-splay hand's overall eccentricity is less than this, it is a fist, otherwise it is a palm
OVAL_THRESHOLD = None

# the geometric center of the outline of a hand is not the center of the hand, this corrects for that
CENTERLINE_OFFSET = None

# constants

LIBRARY_PATH_IN = os.path.join(os.getcwd(), 'library\in')
LIBRARY_PATH_OUT = os.path.join(os.getcwd(), 'library\out')

# user flags
DEBUG = False

def main():
    # if the program is run with --debug flag, set the DEBUG flag to true, it doesn't matter if the flag is in any position
    
    if '--debug' in sys.argv:
        print("Running in debug mode")
        global DEBUG
        DEBUG = True
    
    # walk through the in folder and run the getHand function on each image
    for root, _, files in os.walk(LIBRARY_PATH_IN):
        for file in files:
            if file.endswith('.jpg'):
                if DEBUG:
                    print("Processing file: " + file)
                getHand(os.path.join(root, file))
    
    
# this function outputs a dictionary containing all the information about the hand
def getHand(path) -> dict: 
    # get the image from the supplied path
    img = cv.imread(path)
    # scale the image proportionally until it is less than 1000 pixels on one side
    resize = 1000 / max(img.shape[0], img.shape[1])
    resized = cv.resize(img, (0,0), fx=resize, fy=resize)
    
    # get the contour and hull from the color image
    hand_contour, hull = hull_from_color(resized)
    
    # get the convexity defects from the contour and hull
    
    # if debug, show the original image with the contour and defects drawn on it

    # if the largest 4 defects are on average longer than DEFECT_LENGTH_THRESHOLD, it is a splayed hand
    # otherwise, it is a non-splay hand
    # if the non-splay hand is more oval than OVAL_THRESHOLD, it is a palm, otherwise it is a fist
    
    # get the centerline of the hand from the contour, then offset it by CENTERLINE_OFFSET
    
    
    return dict()


# given an image, this function returns a binary image of skin and non-skin pixels
def hull_from_color(image: cv.Mat) -> np.ndarray:

    # pull out the red channel
    red_channel = image[:,:,2]

    #gaussian blur the image to improve thresholding
    blurred = cv.GaussianBlur(red_channel, (11,11), 0)

    #otsu thresholding to get a binary image
    _, thresholded_image = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    
    #remove some noise by eroding then dilating
    kernel = np.ones((15,15), np.uint8)
    thresholded_image = cv.erode(thresholded_image, kernel, iterations=1)
    thresholded_image = cv.dilate(thresholded_image, kernel, iterations=1)
    
    #fill in gaps by dilating then eroding
    thresholded_image = cv.dilate(thresholded_image, kernel, iterations=1)
    thresholded_image = cv.erode(thresholded_image, kernel, iterations=1)
    
    #get the largest contour and remove the rest, final denoising step
    contours, hierarchy = cv.findContours(thresholded_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    hand_contour = max(contours, key=lambda x: cv.contourArea(x))

    # get contour indices of the convex hull
    hull = cv.convexHull(hand_contour, returnPoints=False)

    # if debug, show the original, binary, and hull images with the contour drawn on the original
    if DEBUG:
        _, axs = plt.subplots(1, 3)
        
        # draw the contour on the original image
        axs[0].imshow(cv.drawContours(cv.cvtColor(image, cv.COLOR_BGR2RGB), contours, -1, (0,255,0), 5))
        # show the binary image
        axs[1].imshow(thresholded_image, cmap='gray')
        
        # connect the dots on the hull and draw it on a copy of the image
        # use modulus to wrap around to the first point
        hull_image = image.copy()
        
        for i in range(len(hull)):
            cv.line(hull_image, tuple(hand_contour[hull[i][0]][0]), tuple(hand_contour[hull[(i+1) % len(hull)][0]][0]), (0,255,0), 5)
        axs[2].imshow(cv.cvtColor(hull_image, cv.COLOR_BGR2RGB))
        plt.show()
    
    # return a dict of "hull" and "contour"
    return dict(hull=hull, contour=hand_contour)

if __name__ == '__main__':
    main()