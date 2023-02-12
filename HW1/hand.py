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
    
class VPosition(Enum):
    TOP = 0
    CENTER = 1
    BOTTOM = 2
    UNKNOWN = 3
    
class HPosition(Enum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2
    UNKNOWN = 3

# hand has ground truth information about the hand and the processing information in a dictionary
class Hand:
    def __init__(self, path):
        image = cv.imread(path)
    
        # scale the image proportionally until it is less than 1000 pixels on one side
        resize_factor = 1000 / max(image.shape[0], image.shape[1])
        resized = cv.resize(image, (0,0), fx=resize_factor, fy=resize_factor)
        
        # defaults for the hand object
        self.img = resized
        self.hand_type = HandType.ANOMALY
        self.vpos = VPosition.UNKNOWN
        self.hpos = HPosition.UNKNOWN
        self.data = dict()
        
        # parse the filename for the hand type, vertical position, and horizontal position
        filename = os.path.splitext(os.path.basename(path))[0]

        filename = re.sub(r'\d+', '', filename)
        tags = [tag.upper() for tag in re.findall(r'[A-Z][^A-Z]*', filename)]
        
        #anomalies are parsed as a special case, but if the file is not an anomaly, parse the tags
        if "ANOMALY" not in tags:
            self.hand_type = HandType[tags[0]]
            self.vpos = VPosition[tags[1]]
            self.hpos = HPosition[tags[2]]
    
    #assemble the hand object and return it, anomalies are parsed as a special case
        
    def __str__(self):
        if self.hand_type == HandType.ANOMALY:
            return (f"Anomaly at unkown position")
        else:
            return (f"{self.hand_type.name} at {self.vpos.name} {self.hpos.name} of image (Ground Truth)")
