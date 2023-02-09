import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import os 
import sys

# constants

LIBRARY_PATH_IN = 'HW1\library\in'
LIBRARY_PATH_OUT = 'HW1\library\out'

# flags

DEBUG = False

def main():
    # if the program is run with --debug flag, set the DEBUG flag to true, it doesn't matter if the flag is in any position
    
    if '--debug' in sys.argv:
        print("Running in debug mode")
        DEBUG = True
    
    # walk through the in folder and run the getHand function on each image
    for root, dirs, files in os.walk(LIBRARY_PATH_IN):
        for file in files:
            if file.endswith('.jpg'):
                path = os.path.join(root, file)
                getHand(path)
    
    
# this function outputs a dictionary containing all the information about the hand
def getHand(path) -> dict: 
    # get the image from the supplied path
    img = cv.imread(path)
    # convert to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # get the red channel
    red = hsv[:,:,2]
    # threshold the red channel 
    ret, thresh = cv.threshold(red, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # find the contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # get the largest contour and remove the rest
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    contours = contours[:1]
    # draw the largest contour over the image and show it
    cv.drawContours(img, contours, -1, (0,255,0), 3)
    # resize the image to 1000x1000
    
    if DEBUG:
        img2 = cv.resize(img, (1000, 1000))
        cv.imshow('image', img2)
        cv.waitKey(0)
        
    cv.destroyAllWindows()
    return dict()

if __name__ == '__main__':
    main()