import os
import re
import cv2 as cv
# enums for hand type and position
# the poses are splay, fist, and palm
# the positions are both extremes and center 
# there is a field for vertical and horizontal position
# each Hand object has a field for each of these enums

#anomalies are unrecognized hands, initialized as anomaly, unknown, unknown
from enum import Enum

class HandType(Enum):
    SPLAY = 0
    FIST = 1
    PALM = 2
    ANOMALY = 3
    
class HandVPos(Enum):
    TOP = 0
    CENTER = 1
    BOTTOM = 2
    UNKNOWN = 3
    
class HandHPos(Enum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2
    UNKNOWN = 3

# hand has ground truth information about the hand and the processing information in a dictionary
class Hand:
    def __init__(self, path):
        image = cv.imread(path)

        # warn if the image is not square or close to square (10% difference)
        if abs(image.shape[0] - image.shape[1]) / image.shape[0] > 0.1:
            print(f'Warning: {path} is not very square, the results may be inaccurate.')
        
        #if not 1000x1000 scale the image proportionally until it is 1000 pixels on one side
        if image.shape[0] != 1000 and image.shape[1] != 1000:
            print("Resizing image...")
            resize_factor = 1000 / max(image.shape[0], image.shape[1])
            image = cv.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)
        
        # defaults for the hand object
        self.img = image
        self.hand_type = HandType.ANOMALY
        self.vpos = HandVPos.UNKNOWN
        self.hpos = HandHPos.UNKNOWN
        self.data = {
        "fullContour": None,
        "handContour": None,
        "hull": None,
        "binary": None,
        "defectTop4Avg": None,
        "eccentricity": None,
        "center": (0, 0),
        "predictedHandType": HandType.ANOMALY.name,
        "predictedVPos": HandVPos.UNKNOWN.name,
        "predictedHPos": HandHPos.UNKNOWN.name,
    }
        
        # parse the filename for the hand type, vertical position, and horizontal position
        filename = os.path.splitext(os.path.basename(path))[0]

        filename = re.sub(r'\d+', '', filename)
        tags = [tag.upper() for tag in re.findall(r'[A-Z][^A-Z]*', filename)]
        
        #anomalies are parsed as a special case, but if the file is not an anomaly, parse the tags
        if "ANOMALY" not in tags:
            self.hand_type = HandType[tags[0]]
            self.vpos = HandVPos[tags[1]]
            self.hpos = HandHPos[tags[2]]
    
    #assemble the hand object and return it, anomalies are parsed as a special case
        
    def __str__(self):
        return (f'{"Ground Truth:":<20} {self.hand_type.name:<10} {self.vpos.name:<10} {self.hpos.name:<10}\n {"Prediction:":<20} {self.data["predictedHandType"]:<10} {self.data["predictedVPos"]:<10} {self.data["predictedHPos"]:<10}')

    def stat(self):
        return (f'{"Eccentricity:":<10} {self.data["eccentricity"]:<10.2} {"Defect Top 4 Avg:":<10} {self.data["defectTop4Avg"]:<10.6} {"Contour Center:":<10} {str(self.data["center"]):<10}')