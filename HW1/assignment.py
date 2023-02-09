import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import os 
import sys

# constants

LIBRARY_PATH_IN = os.path.join(os.getcwd(), 'library\in')
LIBRARY_PATH_OUT = os.path.join(os.getcwd(), 'library\out')

# flags

DEBUG = False

def main():
    # if the program is run with --debug flag, set the DEBUG flag to true, it doesn't matter if the flag is in any position
    
    if '--debug' in sys.argv:
        print("Running in debug mode")
        global DEBUG
        DEBUG = True
    
    # walk through the in folder and run the getHand function on each image
    for root, dirs, files in os.walk(LIBRARY_PATH_IN):
        for file in files:
            if file.endswith('.jpg'):
                print("Processing file: " + file)
                getHand(os.path.join(root, file))
    
    
# this function outputs a dictionary containing all the information about the hand
def getHand(path) -> dict: 
    # get the image from the supplied path
    img = cv.imread(path)
    # convert to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # get the value channel
    v = hsv[:,:,2]
    # denoise v
    # threshold the image based on the value channel
    ret, thresholded_image = cv.threshold(v, 0, 255, cv.THRESH_BINARY |cv.THRESH_OTSU)
    # simple denoise the image
    thresholded_image = cv.medianBlur(thresholded_image, 55)
    # show the intermediate image if the debug flag is set
    if DEBUG:
        img2 = cv.resize(thresholded_image, (1000, 1000))
        cv.imshow('image', img2)
        cv.waitKey(0)
    # find the contours
    contours, hierarchy = cv.findContours(thresholded_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # get the largest contour and remove the rest
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    contours = contours[:1]
        
    # draw the largest contour over the image in blue and show it
    cv.drawContours(img, contours, -1, (255,0,0), 3)

    # show these images if the debug flag is set    
    if DEBUG:
        img2 = cv.resize(img, (1000, 1000))
        cv.imshow('image', img2)
        cv.waitKey(0)
        
    cv.destroyAllWindows()
    return dict()

if __name__ == '__main__':
    main()