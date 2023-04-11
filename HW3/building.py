
from typing import Tuple

import cv2 as cv


table = {}
r_table = {}


#open Table.txt file
#each line in the file is read and split into two parts
#the first part is a number and the second part is a string
#the number is converted to an integer and the string is stored in the lookup table
#store it both ways so that we can convert either way
with open('Table.txt') as f:
    for line in f:
        (key, val) = line.split()
        table[int(key)] = val
        r_table[val] = int(key)

#return the string associated with the integer
def id_2_name(x: int) -> str:
    if x == -1:
        return "None"
    return table[x]

#return the integer associated with the string
def name_2_id(x: str) -> int:
    if x == "None":
        return -1
    return r_table[x]

# buildings are read from the map based on an integer grayscale value
class Building:
    def __init__(self, id: int, img: cv.Mat):
        self.id = id
        self.name = id_2_name(id)
        self.MBR = (0, 0, 0, 0)
        self.COM = (0.0, 0.0)
        self.area = 0
        self.diag = 0.0
        self.MBR, self.COM, self.area, self.diag = self.calc_stats(img)
        MBR_height = abs(self.MBR[1] - self.MBR[3]) + 1
        MBR_width = abs(self.MBR[2] - self.MBR[0]) + 1
        #iterate through image, if pixel doesn't match id, set to 0
        self.img = img.copy()
        #skip this step if id is -1, its a special case
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                if self.img[y, x] != self.id:
                    self.img[y, x] = 0
                
                
        # divide the smaller dimension by the larger dimension to get the aspect ratio
        if MBR_height > MBR_width:
            self.aspect = MBR_width / MBR_height
        else:
            self.aspect = MBR_height / MBR_width
        # divide the area by the area of the MBR to get the occupied ratio
        self.occupied_ratio = self.area / (MBR_height * MBR_width)

    def calc_stats(self, img: cv.Mat)-> Tuple[Tuple[float, float, float, float], Tuple[float, float], int]:
        area = 0
        MBR1x, MBR1y, MBR2x, MBR2y = 0, 0, 0, 0
        COMx, COMy = 0.0, 0.0
        
        # iterate through all pixels in the image
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if img[y, x] == self.id:
                    area +=1
                    # update MBR1 and MBR2
                    #MBR 0 = x1, 1 = y1, 2 = x2, 3 = y2
                    if MBR1x == 0 and MBR1y == 0.:
                        MBR1x, MBR1y = x, y
                    if MBR2x == 0 and MBR2y == 0:
                        MBR2x, MBR2y = x, y
                    if x < MBR1x:
                        MBR1x = x
                    if y < MBR1y:
                        MBR1y = y
                    if x > MBR2x:
                        MBR2x = x
                    if y > MBR2y:
                        MBR2y = y

                    # update COM
                    COMx += x
                    COMy += y
        COMx /= area
        COMy /= area
        diag = ((MBR2x - MBR1x)**2 + (MBR2y - MBR1y)**2)**0.5
        return (MBR1x, MBR1y, MBR2x, MBR2y), (COMx, COMy), area, diag