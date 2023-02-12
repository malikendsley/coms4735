import re
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import os 
import sys
from hand import *

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
STATS = False

def main():
    # if the program is run with --debug flag, set the DEBUG flag to true, it doesn't matter if the flag is in any position
    
    if '--debug' in sys.argv:
        print("Running in debug mode")
        global DEBUG
        DEBUG = True
    
    if '--stats':
        print("Gathering calibration statistics")
        global STATS
        STATS = True
    # walk through the in folder and run the getHand function on each image
    for root, _, files in os.walk(LIBRARY_PATH_IN):
        for file in files:
            if file.endswith('.jpg'):
                # if DEBUG:
                    # print("Processing file: " + file)
                analyzeHand(os.path.join(root, file))
    
    
# this function outputs a dictionary containing all the information about the hand
def analyzeHand(path) -> Hand:
    # initialize the hand object
    hand = Hand(path)
    
    if DEBUG:
        print(hand)
    
    # get the contour and convex hull of the hand
    hull, hand_contour = hull_from_color(hand.img)
    
    # get the convexity defects from the contour and hull
    defects = cv.convexityDefects(hand_contour, hull)
    data = {
        "defectTop4Avg": None,
        "eccentricity": None,
        "contourCenter": None,
        "hullCenter": None,
    }
    # if debug, show the original image with the contour and defects drawn on it
    if DEBUG:
        image_with_defects = hand.img.copy()
        # strip to the 4 largest defects
        defects = defects[np.argsort(defects[:,0,3])][::-1][:4]
        # draw each defect as a triangle between the start, end, and far points
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(hand_contour[s][0])
            end = tuple(hand_contour[e][0])
            far = tuple(hand_contour[f][0])
            cv.line(image_with_defects,start,end,[0, 0, 255],2)
            cv.line(image_with_defects,start,far,[0, 0, 255],2)
            cv.line(image_with_defects,end,far,[0, 0, 255],2)
        # fit a line to the contour and draw it longer than the contour to show the centerline
        [vx,vy,x,y] = cv.fitLine(hand_contour, cv.DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((image_with_defects.shape[1]-x)*vy/vx)+y)
        cv.line(image_with_defects,(image_with_defects.shape[1]-1,righty),(0,lefty),(0,255,0),2)
        
        # draw the geometric center of the contour in a thick blue circle
        M = cv.moments(hand_contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv.circle(image_with_defects, (cx, cy), 30, (255, 0, 0), -1)
        
        # select the hullth points of the contour and draw them in a thick green circle to make other functions work
        contour_of_hull = hand_contour[hull[:,0]]

        # draw the geometric center of the convex hull in a thick yellow circle
        M = cv.moments(contour_of_hull)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv.circle(image_with_defects, (cx, cy), 30, (0, 255, 255), -1)
        
        plt.imshow(cv.cvtColor(image_with_defects, cv.COLOR_BGR2RGB))
        
        # label the image with the name of the file
        plt.title("Analysis")
        # annotate each color with its meaning and the color of the circle at the bottom of the image
        plt.annotate("Red: Concave Defects", (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
        plt.annotate("Green: Centerline", (0, 0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top')
        plt.annotate("Blue: Contour Center", (0, 0), (0, -60), xycoords='axes fraction', textcoords='offset points', va='top')
        plt.annotate("Yellow: Hull Center", (0, 0), (0, -80), xycoords='axes fraction', textcoords='offset points', va='top')
        plt.show()

    # collect the reduced data from the hand. To set our tuning handles, we will need to collect the data from a large number of images
    # the STAT flag will tell the program to show histograms of the data instead of making a decision

    # splayed hands have 4 large convexity defects, non-splayed hands do not, this can be used to distinguish between the two
    data['defectTop4Avg'] = defects[:,0,3].mean()
    
    # palms and fists have no large convexity defects but they do have different shapes, this can be used to distinguish between the two
    # eccentricity measures ovalness, so a high eccentricity means the hand is more oval, a low eccentricity means the hand is more circular
    # roughly, circle = first and oval = palm
    ellipse = cv.fitEllipse(contour_of_hull)
    data['eccentricity'] = ellipse[1][0] / ellipse[1][1]
    
    # get the geometric center of the contour and the convex hull relative to the image in pixels
    M = cv.moments(hand_contour)
    data['contourCenter'] = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
    M = cv.moments(contour_of_hull)
    data['hullCenter'] = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
    
    # return the hand with the data filled in
    hand.data = data
    return hand

# given an image, this function returns a binary image of skin and non-skin pixels
def hull_from_color(image: cv.Mat) -> tuple:

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
         
        _, axs = plt.subplots(1, 4)
        
        # draw the contour on the original image, titled "original"
        axs[0].set_title("Original")
        axs[0].imshow(cv.drawContours(cv.cvtColor(image, cv.COLOR_BGR2RGB), contours, -1, (0,255,0), 5))
        # show the binary image
        axs[1].set_title("Binary")
        axs[1].imshow(thresholded_image, cmap='gray')
        
        # connect the dots on the hull and draw it on a copy of the image
        # use modulus to wrap around to the first point
        hull_image = image.copy()
        
        for i in range(len(hull)):
            cv.line(hull_image, tuple(hand_contour[hull[i][0]][0]), tuple(hand_contour[hull[(i+1) % len(hull)][0]][0]), (0,255,0), 5)
        axs[2].set_title("Hull")
        axs[2].imshow(cv.cvtColor(hull_image, cv.COLOR_BGR2RGB))
    # return a dict of "hull" and "contour"
    return hull, hand_contour

if __name__ == '__main__':
    main()